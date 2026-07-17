"""五任务训练器共用工程件基类（``BaseTrainer``）。

设计（模板方法的"显式装配"变体）：各任务训练循环（``fit`` / ``_train_epoch`` /
``_validate``）差异大且各有历史语义，基类**不**吞并循环本身；而是把五份训练器
中逐字重复的工程 blocks 收敛为 protected helpers，子类在自己的 ``__init__``
里按原有顺序显式调用，行为与拆分前一致：

* ``_setup_channels_last``  —— 可选 channels_last 内存格式（数值等价）；
* ``_setup_optim_sched``    —— optimizer + warmup/调度器（按优化步推进）；
* ``_setup_amp``            —— amp_dtype 解析 / GradScaler；
* ``_setup_ema``            —— ModelEMA（decay warmup / CPU offload）；
* ``_maybe_compile``        —— torch.compile（最后：optimizer/EMA 已绑裸参数）；
* ``_setup_output_dir``     —— 输出目录创建；
* ``_setup_best_tracking``  —— 最优指标 / 早停计数初始化。

运行期共用件：``_ema_swapped``（异常安全的 EMA 换入换出）、``_effective_accum``
（梯度累积尾批分母）、模型健康监测 helpers（全局梯度/权重范数、update ratio、
``_collect_health_metrics``）。

子类约定：调用 helpers 前需已设置 ``self.cfg`` / ``self.model`` /
``self.device`` / ``self.train_loader``。
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional

import torch

from ..utils import AverageMeter, ModelEMA, ModelSWA, seed_everything
from .amp import _AMP_DTYPES, GradScaler, resolve_auto_amp_dtype
from .bn_stats import collect_bn_modules, estimate_bn_stats
from .checkpoint import (
    atomic_torch_save, extract_model_state_dict, restore_rng_state,
    strip_common_prefixes, unwrap_compile,
)
from .dist_utils import all_reduce_bn_running_stats_
from .optim import WarmupScheduler, build_optimizer, build_scheduler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RNG helper
# ---------------------------------------------------------------------------
def _reseed_rank_rng(seed: int, rank: int, epoch: int, deterministic: bool) -> None:
    """按 rank/epoch 重新分流 RNG。rank0 保持原流不动。"""
    if int(rank) <= 0:
        return
    # resume 后 rank>0 重新分流，避免所有 DDP rank 退化成 rank0 的随机流。
    mix = int(seed) + int(epoch) * 100003 + int(rank)
    seed_everything(mix, deterministic)


# ---------------------------------------------------------------------------
# BaseTrainer
# ---------------------------------------------------------------------------
class BaseTrainer:
    """共用训练工程件；子类实现自己的 ``fit`` / ``_train_epoch`` / ``_validate``。"""

    # ------------------------------------------------------------------
    # Construction helpers（子类 __init__ 按原有顺序显式调用）
    # ------------------------------------------------------------------
    def _setup_channels_last(self) -> None:
        """可选 channels_last 内存格式（数值等价；输入由 conv 自动适配）。"""
        tc = self.cfg.train
        self._memory_format = None
        if tc.channels_last:
            self._memory_format = (
                torch.channels_last_3d
                if int(self.cfg.model.spatial_dims) == 3
                else torch.channels_last)
            self.model = self.model.to(memory_format=self._memory_format)

    def _setup_optim_sched(self) -> None:
        """Optimizer + warmup 调度器。

        调度器以优化步（``optimizer.step`` 次数）为单位推进：梯度累积下每
        epoch 的优化步数为 ceil(micro-batch 数 / accum)（epoch 尾部不满
        accum 的尾组也触发一步，见各子类 ``_train_epoch`` 的 step 边界判定）。
        one_cycle 用 warmup_epochs 映射 pct_start，外层不再做线性 warmup，
        避免 warmup 双重叠加。
        """
        tc = self.cfg.train
        self.optimizer = build_optimizer(self.model, self.cfg)
        self.grad_accum_steps = max(tc.grad_accum_steps, 1)
        steps_per_epoch = math.ceil(len(self.train_loader) / self.grad_accum_steps)
        warmup_steps    = tc.warmup_epochs * steps_per_epoch
        total_steps     = tc.epochs * steps_per_epoch
        post_warmup     = total_steps - warmup_steps

        base_scheduler = build_scheduler(
            self.optimizer, self.cfg, steps_per_epoch,
            post_warmup_steps=post_warmup)
        warmup_steps = 0 if tc.scheduler == "one_cycle" else warmup_steps
        self.scheduler = WarmupScheduler(
            self.optimizer, base_scheduler, warmup_steps=warmup_steps,
            warmup_lr=tc.warmup_lr, base_lr=tc.lr)

    def _setup_amp(self) -> None:
        """解析 ``amp_dtype``（含 'auto'），构建版本无关 GradScaler。"""
        tc = self.cfg.train
        amp_dtype_cfg = tc.amp_dtype
        if amp_dtype_cfg == "auto":
            amp_dtype_cfg = resolve_auto_amp_dtype(self.device)
            logger.info("amp_dtype='auto' resolved to %r (device=%s).",
                        amp_dtype_cfg, self.device)
        if amp_dtype_cfg not in _AMP_DTYPES:
            raise ValueError(
                f"Unknown amp_dtype: {tc.amp_dtype!r}. "
                f"Expected one of {sorted(_AMP_DTYPES) + ['auto']}.")
        self.amp_dtype = _AMP_DTYPES[amp_dtype_cfg]
        self._amp_dtype_name = amp_dtype_cfg
        self.use_amp = tc.use_amp and self.device.type == "cuda"
        self._scaler_active = self.use_amp and self.amp_dtype == torch.float16
        self.scaler = GradScaler("cuda", enabled=self._scaler_active)

    def _setup_ema(self) -> None:
        """EMA。``ema_device="cpu"`` 时 shadow/backup 常驻 CPU（省 1× 参数量
        GPU 显存，数学等价）；默认 "" 跟随模型设备。"""
        tc = self.cfg.train
        self.ema = (ModelEMA(self.model, tc.ema_decay, warmup=tc.ema_warmup,
                             offload_device=(tc.ema_device or None))
                    if tc.use_ema else None)

    def _maybe_compile(self) -> None:
        """torch.compile（须最后调用：optimizer / EMA / checkpoint 绑裸参数）。"""
        tc = self.cfg.train
        self._compile_enabled = False
        if tc.compile_mode != "none" and hasattr(torch, "compile"):
            triton_ok = True
            if self.device.type == "cuda":
                import importlib.util
                if importlib.util.find_spec("triton") is None:
                    triton_ok = False
                    logger.warning(
                        "torch.compile (mode='%s') requested but Triton not installed; "
                        "falling back to eager. Install Triton or set compile_mode='none'.",
                        tc.compile_mode)
            if triton_ok:
                logger.info("Compiling model with mode='%s'", tc.compile_mode)
                self.model = torch.compile(self.model, mode=tc.compile_mode)
                self._compile_enabled = True

    def _setup_output_dir(self) -> None:
        self.output_dir = Path(self.cfg.train.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _setup_best_tracking(self, mode: str = "max") -> None:
        """最优指标 / 早停初始化。``mode``："max"（指标越大越好）| "min"。"""
        self._best_mode       = mode
        self.best_metric      = (-math.inf if mode == "max" else math.inf)
        self.has_best         = False
        self.best_epoch       = 0
        self.start_epoch      = 0
        self.patience_counter = 0

    # ------------------------------------------------------------------
    # EMA swap helper (exception-safe)
    # ------------------------------------------------------------------
    @contextmanager
    def _ema_swapped(self) -> Iterator[None]:
        """临时将 EMA 权重换入 model；try/finally 保证异常时也能还原在线权重。"""
        if self.ema is None:
            yield
            return
        self.ema.apply_shadow(unwrap_compile(self.model))
        try:
            yield
        finally:
            self.ema.restore(unwrap_compile(self.model))

    # ------------------------------------------------------------------
    # Effective grad-accum denominator (尾批不满 accum 时取真实尾长)
    # ------------------------------------------------------------------
    @staticmethod
    def _effective_accum(step: int, total_steps: int, accum: int) -> int:
        """尾批 micro-batch 数不满 ``accum`` 时，用真实尾长作分母，
        以免最后一组 micro-batch 因被除以 ``accum`` 而权重偏小。"""
        if accum <= 1:
            return 1
        remainder = total_steps % accum
        partial_start = total_steps - remainder
        return remainder if (remainder > 0 and step >= partial_start) else accum

    # ------------------------------------------------------------------
    # Model-health helpers (轻量；仅 rank0 监测启用时调用)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _global_grad_norm(self) -> "float | None":
        """当前已 unscale 的全局梯度 L2 范数（遍历一次有梯度的参数）。

        仅在未开启 grad_clip（无现成范数可复用）且监测需要时手动调用；调用方
        负责在 AMP fp16 下先 ``scaler.unscale_``，以免量纲被 loss scale 污染。
        """
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        if not grads:
            return None
        # foreach 批量算逐张量范数后聚合，全程仅末尾一次 .item() 同步（逐参数
        # .item() 会打断 CUDA 流水）。任一梯度含 inf/NaN 时结果同样非有限。
        norms = torch._foreach_norm(grads, 2)
        return float(torch.linalg.vector_norm(
            torch.stack([n.float() for n in norms])).item())

    @torch.no_grad()
    def _global_weight_norm(self) -> "float | None":
        """全部参数的全局 L2 范数（每 epoch 仅算一次，开销可忽略）。"""
        params = [p.detach() for p in self.model.parameters()]
        if not params:
            return None
        norms = torch._foreach_norm(params, 2)
        return float(torch.linalg.vector_norm(
            torch.stack([n.float() for n in norms])).item())

    @torch.no_grad()
    def _param_snapshot(self) -> "list":
        """对全部参数做一次瞬时 clone（用于 update/weight 比值，每 epoch 仅一次）。"""
        return [p.detach().clone() for p in self.model.parameters()]

    @torch.no_grad()
    def _update_ratio_from_snapshot(self, snapshot: "list") -> "float | None":
        """由 ``optimizer.step`` 前的快照算全局 ‖Δw‖/‖w‖（用完释放快照）。

        参数遍历顺序与 ``_param_snapshot`` 一致，逐张量累加更新量与原权重的平方和。
        若原权重范数为 0（理论上不会发生）则返回 ``None`` 以免除零。
        """
        params = [p.detach() for p in self.model.parameters()]
        if not params:
            return None
        diffs = torch._foreach_sub(params, snapshot)
        upd = torch.linalg.vector_norm(torch.stack(
            [n.float() for n in torch._foreach_norm(diffs, 2)]))
        w = torch.linalg.vector_norm(torch.stack(
            [n.float() for n in torch._foreach_norm(snapshot, 2)]))
        w_norm = float(w.item())
        if w_norm <= 0.0:
            return None
        return float(upd.item()) / w_norm

    # ------------------------------------------------------------------
    # Resume / Pretrain 公共加载策略
    # ------------------------------------------------------------------
    def _restore_train_state(self, ckpt: Dict) -> int:
        """resume 公共段：model/EMA/optimizer/scheduler/scaler/best/RNG。

        返回续训起始 epoch（= ckpt epoch + 1）。任务专有字段（history、
        早停计数等）由子类自行恢复。旧版 ckpt 无 rng_state 时静默跳过
        （训练仍正常但非位精确）。"""
        bare = unwrap_compile(self.model)
        bare.load_state_dict(ckpt["model_state_dict"])
        if self.ema is not None and "ema_state_dict" in ckpt:
            self.ema.load_state_dict(ckpt["ema_state_dict"])
        for key, obj in (("optimizer_state_dict", self.optimizer),
                         ("scheduler_state_dict", self.scheduler),
                         ("scaler_state_dict", self.scaler)):
            if key in ckpt:
                obj.load_state_dict(ckpt[key])
        self.best_metric = float(ckpt.get("best_metric", -math.inf))
        self.best_epoch = int(ckpt.get("best_epoch", 0))
        rng = ckpt.get("rng_state")
        if rng:
            try:
                restore_rng_state(rng)
                logger.info("Restored RNG state from checkpoint.")
            except Exception as e:  # pragma: no cover
                logger.warning("Failed to restore RNG state: %s", e)
        return int(ckpt.get("epoch", -1)) + 1

    def _load_pretrain_weights(self, path: str, *, strict: bool,
                               load_ema: bool) -> None:
        """pretrain 公共策略：仅加载模型权重作迁移初始化。

        不动 optimizer/scheduler/scaler/RNG、不推进 epoch；支持任意任务的
        checkpoint 容器（``extract_model_state_dict`` 选 EMA/在线权重，
        ``strip_common_prefixes`` 去 compile/DDP 前缀）；加载后重对齐 EMA
        shadow 以免带着随机初始泄露。"""
        logger.info(
            "Loading pretrain weights: %s (strict=%s, load_ema=%s)",
            path, strict, load_ema)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        sd, source = extract_model_state_dict(ckpt, prefer_ema=load_ema)
        sd = strip_common_prefixes(sd)
        bare = unwrap_compile(self.model)
        result = bare.load_state_dict(sd, strict=strict)
        missing = list(getattr(result, "missing_keys", []) or [])
        unexpected = list(getattr(result, "unexpected_keys", []) or [])

        def _preview(keys, n=8):
            head = ", ".join(keys[:n])
            return head + (f", ... (+{len(keys) - n} more)"
                           if len(keys) > n else "")

        if missing:
            logger.warning(
                "Pretrain: %d missing key(s) [%s]. These params keep their "
                "random init.", len(missing), _preview(missing))
        if unexpected:
            logger.warning(
                "Pretrain: %d unexpected key(s) [%s]. These ckpt params are "
                "discarded.", len(unexpected), _preview(unexpected))
        if not missing and not unexpected:
            logger.info("Pretrain: all keys matched cleanly.")

        if self.ema is not None:
            with torch.no_grad():
                live_sd = bare.state_dict()
                for k, v in live_sd.items():
                    if k in self.ema.shadow:
                        self.ema.shadow[k].copy_(v)
            logger.info("Pretrain: EMA shadow re-aligned to loaded weights.")

        logger.info(
            "Pretrain loaded from `%s` slot. Training will start from "
            "epoch 0 with fresh optimizer / scheduler / scaler / best / RNG.",
            source)

    def _try_pretrain(self) -> None:
        """按公共策略处理 ``train.pretrain``；同设 resume 时 resume 优先。"""
        tc = self.cfg.train
        pretrain = str(tc.pretrain or "").strip()
        if not pretrain:
            return
        if str(tc.resume or "").strip():
            logger.warning(
                "Both `train.resume` and `train.pretrain` are set; "
                "using resume (%s). Pretrain weights from %s are ignored.",
                tc.resume, tc.pretrain)
            return
        if not Path(pretrain).is_file():
            raise FileNotFoundError(
                f"train.pretrain checkpoint not found: {pretrain!r}")
        self._load_pretrain_weights(
            pretrain, strict=tc.pretrain_strict,
            load_ema=tc.pretrain_load_ema)

    # ------------------------------------------------------------------
    # SWA 尾段权重平均（opt-in，见 TrainConfig.swa_enabled）
    # ------------------------------------------------------------------
    def _setup_swa(self) -> None:
        """创建 ``self.swa``（未启用时为 None）与起始 epoch。"""
        tc = self.cfg.train
        self.swa = ModelSWA(self.model) if tc.swa_enabled else None
        self._swa_start_epoch = (
            int(math.floor(tc.swa_start_ratio * tc.epochs))
            if tc.swa_enabled else 0)

    def _swa_update(self, epoch: int) -> None:
        """epoch 末调用：到达起始 epoch 后每 epoch 纳入一次在线权重快照。"""
        if self.swa is not None and epoch >= self._swa_start_epoch:
            self.swa.update(unwrap_compile(self.model))

    def _swa_recalibrate_bn(self, run_forward: Callable[[], None]) -> None:
        """在若干 train batch 上重估 BN running stats（AdaBN 同款累积平均）。

        平均权重下各层激活分布改变，BN 的 running stats 不再匹配；
        ``run_forward`` 由子类提供（未增强的训练数据前向，与推理分布同构）。
        模型无 BatchNorm（instance/group norm）时为 no-op。DDP 下聚合各
        rank 的 running stats。"""
        if int(self.cfg.train.swa_bn_update_steps) <= 0:
            return
        bn_modules = collect_bn_modules(unwrap_compile(self.model))
        if not bn_modules:
            return
        logger.info(
            "SWA: re-estimating %d BatchNorm module(s) running stats.",
            len(bn_modules))
        estimate_bn_stats(bn_modules, run_forward)
        all_reduce_bn_running_stats_(bn_modules)

    def _finalize_swa(
        self,
        *,
        validate_fn: "Optional[Callable[[], Dict[str, float]]]" = None,
        bn_forward_fn: "Optional[Callable[[], None]]" = None,
        is_main: bool = True,
    ) -> None:
        """SWA 收尾：换入平均权重 → 重估 BN → 验证 → 另存 swa_model.pth。

        不改变 best_model.pth 选模逻辑；结束后恢复在线权重。早停等原因未
        收集到任何快照时跳过。``validate_fn`` 应在**当前（SWA）权重**上
        评测，不得再换入 EMA。"""
        if getattr(self, "swa", None) is None:
            return
        tc = self.cfg.train
        if self.swa.n_averaged == 0:
            logger.warning(
                "SWA enabled but no snapshots collected (training ended "
                "before start epoch %d = swa_start_ratio %.2f x %d epochs); "
                "skipping SWA finalization.",
                self._swa_start_epoch + 1, tc.swa_start_ratio, tc.epochs)
            return
        bare = unwrap_compile(self.model)
        self.swa.apply_shadow(bare)
        try:
            self.model.eval()
            if bn_forward_fn is not None:
                self._swa_recalibrate_bn(bn_forward_fn)
            metrics: Dict[str, float] = {}
            if validate_fn is not None:
                try:
                    metrics = validate_fn()
                except Exception:
                    logger.warning(
                        "SWA validation failed; saving SWA weights anyway.",
                        exc_info=True)
            metric_str = ", ".join(
                f"{k}={v:.4f}" for k, v in metrics.items()
                if isinstance(v, (int, float))) or "n/a"
            logger.info(
                "SWA (avg of %d epoch snapshots, from epoch %d): %s",
                self.swa.n_averaged, self._swa_start_epoch + 1, metric_str)
            if is_main:
                path = self.output_dir / "swa_model.pth"
                atomic_torch_save({
                    "model_state_dict": bare.state_dict(),
                    "swa_n_averaged": self.swa.n_averaged,
                    "swa_val_metrics": metrics,
                    "config": self.cfg,
                }, path)
                logger.info("SWA model saved: %s", path)
        finally:
            self.swa.restore(bare)

    def _collect_health_metrics(
        self,
        out: Dict[str, float],
        *,
        grad_norm_meter: AverageMeter,
        grad_norm_max: float,
        nonfinite_steps: int,
        clipped_steps: int,
        opt_steps: int,
        grad_clip_norm: float,
        update_ratio: "float | None" = None,
    ) -> None:
        """把本 epoch 聚合的健康指标并入 ``out``（仅写入有意义的键）。

        - ``grad_norm`` / ``grad_norm_max``：仅在采集到范数时写入。
        - ``nonfinite_steps``：非有限 loss 的 micro-batch 计数。
        - ``grad_clip_frac``：开启 grad_clip 时，范数超阈值的优化步占比。
        - ``weight_norm``：每 epoch 末一次全参数范数。
        - ``amp_scale``：AMP fp16 scaler 标度（仅 scaler 实际启用时）。
        - ``update_ratio``：全局 ‖Δw‖/‖w‖（仅开启且本 epoch 成功测得时）。
        """
        if grad_norm_meter.count > 0:
            out["grad_norm"] = grad_norm_meter.avg
            out["grad_norm_max"] = grad_norm_max
        out["nonfinite_steps"] = float(nonfinite_steps)
        if grad_clip_norm > 0 and opt_steps > 0:
            out["grad_clip_frac"] = clipped_steps / opt_steps
        wn = self._global_weight_norm()
        if wn is not None:
            out["weight_norm"] = wn
        if self._scaler_active:
            out["amp_scale"] = float(self.scaler.get_scale())
        if update_ratio is not None and math.isfinite(update_ratio):
            out["update_ratio"] = update_ratio


__all__ = ["BaseTrainer", "_reseed_rank_rng"]
