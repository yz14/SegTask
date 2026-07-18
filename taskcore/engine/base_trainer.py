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
    atomic_torch_save, extract_model_state_dict, relocate_optimizer_state,
    restore_rng_state, strip_common_prefixes, unwrap_compile,
)
from .dist_utils import (
    all_reduce_bn_running_stats_, get_rank, get_world_size,
    is_dist_avail_and_initialized, is_main_process,
)
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
    def _setup_channels_last(
        self, module: "Optional[torch.nn.Module]" = None,
    ) -> "Optional[torch.nn.Module]":
        """可选 channels_last 内存格式（数值等价；输入由 conv 自动适配）。

        默认作用于 ``self.model``；传入 ``module`` 时转换并返回该模块
        （供训练对象不是 ``self.model`` 的任务，如 SSL 的 method.module）。"""
        tc = self.cfg.train
        self._memory_format = None
        if tc.channels_last:
            self._memory_format = (
                torch.channels_last_3d
                if int(self.cfg.model.spatial_dims) == 3
                else torch.channels_last)
            if module is not None:
                return module.to(memory_format=self._memory_format)
            self.model = self.model.to(memory_format=self._memory_format)
        return module

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

    def _maybe_compile(
        self, module: "Optional[torch.nn.Module]" = None,
    ) -> "Optional[torch.nn.Module]":
        """torch.compile（须最后调用：optimizer / EMA / checkpoint 绑裸参数）。

        默认作用于 ``self.model``；传入 ``module`` 时编译并返回该模块
        （供训练对象不是 ``self.model`` 的任务，如 SSL 的 method.module）。"""
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
                self._compile_enabled = True
                if module is not None:
                    return torch.compile(module, mode=tc.compile_mode)
                self.model = torch.compile(self.model, mode=tc.compile_mode)
        return module

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
        """resume 公共段：model/EMA/optimizer/scheduler/scaler/SWA/best/RNG。

        返回续训起始 epoch（= ckpt epoch + 1）。任务专有字段（history 等）
        由子类自行恢复。旧版 ckpt 无 rng_state 时静默跳过
        （训练仍正常但非位精确）。"""
        bare = unwrap_compile(self.model)
        # EMA-as-online 布局的 ckpt（如 seg）把在线权重存
        # ``model_online_state_dict``；其余任务只有 ``model_state_dict``。
        model_sd = ckpt.get("model_online_state_dict",
                            ckpt["model_state_dict"])
        bare.load_state_dict(model_sd)
        if self.ema is not None and "ema_state_dict" in ckpt:
            self.ema.load_state_dict(ckpt["ema_state_dict"])
        for key, obj in (("optimizer_state_dict", self.optimizer),
                         ("scheduler_state_dict", self.scheduler),
                         ("scaler_state_dict", self.scaler)):
            if key in ckpt:
                obj.load_state_dict(ckpt[key])
        if "optimizer_state_dict" in ckpt:
            # CPU 写盘的优化器状态搬回参数所在设备（无 CPU offload 时 no-op）。
            relocate_optimizer_state(self.optimizer)
        if getattr(self, "swa", None) is not None and "swa_state_dict" in ckpt:
            self.swa.load_state_dict(ckpt["swa_state_dict"])
        mode = getattr(self, "_best_mode", "max")
        default_best = -math.inf if mode == "max" else math.inf
        self.best_metric = float(ckpt.get("best_metric", default_best))
        self.best_epoch = int(ckpt.get("best_epoch", 0))
        self.has_best = bool(ckpt.get(
            "has_best", math.isfinite(self.best_metric)))
        self.patience_counter = int(ckpt.get("patience_counter", 0))
        rng = ckpt.get("rng_state")
        if rng:
            try:
                restore_rng_state(rng)
                logger.info("Restored RNG state from checkpoint.")
            except Exception as e:  # pragma: no cover
                logger.warning("Failed to restore RNG state: %s", e)
        return int(ckpt.get("epoch", -1)) + 1

    def _pretrain_transform_state_dict(
        self, sd: Dict, bare: "torch.nn.Module") -> Dict:
        """pretrain 钩子：子类可在加载前变换 state_dict（如 seg 的 UpKern
        升核重采样）。默认原样返回。"""
        return sd

    def _load_pretrain_weights(self, path: str, *, strict: bool,
                               load_ema: bool) -> None:
        """pretrain 公共策略：仅加载模型权重作迁移初始化。

        不动 optimizer/scheduler/scaler/RNG、不推进 epoch；支持任意任务的
        checkpoint 容器（``extract_model_state_dict`` 选 EMA/在线权重，
        ``strip_common_prefixes`` 去 compile/DDP 前缀）；任务专有的 state_dict
        变换经 ``_pretrain_transform_state_dict`` 钩子接入；加载后重对齐 EMA
        shadow 以免带着随机初始泄露。"""
        logger.info(
            "Loading pretrain weights: %s (strict=%s, load_ema=%s)",
            path, strict, load_ema)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        sd, source = extract_model_state_dict(ckpt, prefer_ema=load_ema)
        sd = strip_common_prefixes(sd)
        bare = unwrap_compile(self.model)
        sd = self._pretrain_transform_state_dict(sd, bare)
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

    # ------------------------------------------------------------------
    # DDP 装配与训练采样器识别（公共工程件）
    # ------------------------------------------------------------------
    def _setup_ddp(self) -> None:
        """DDP（多卡）装配。

        ``self.model`` 始终保持“裸 / 已 compile”模块：optimizer / EMA /
        checkpoint / predictor 全部继续作用其上，参数张量与单卡完全一致
        （DDP 不复制参数，仅挂反传 all-reduce 钩子）。前向走
        ``self.fwd_model``（DDP 包装）；``world_size<=1`` 时 fwd_model 即裸
        模块，单卡路径零变化。需在 ``_maybe_compile`` 之后调用。"""
        tc = self.cfg.train
        self._rank       = get_rank()
        self._world_size = get_world_size()
        self._is_main    = is_main_process()
        self._is_dist    = (is_dist_avail_and_initialized()
                            and self._world_size > 1)
        if self._is_dist:
            ddp_kwargs: Dict[str, object] = {}
            if self.device.type == "cuda" and self.device.index is not None:
                ddp_kwargs = {"device_ids": [self.device.index],
                              "output_device": self.device.index}
            self.fwd_model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                find_unused_parameters=bool(tc.ddp_find_unused_parameters),
                gradient_as_bucket_view=bool(tc.ddp_gradient_as_bucket_view),
                static_graph=bool(tc.ddp_static_graph),
                **ddp_kwargs)
            if tc.ddp_static_graph and tc.ddp_find_unused_parameters:
                logger.warning(
                    "ddp_static_graph=True 与 ddp_find_unused_parameters=True "
                    "同时开启：static_graph 首步后会接管 unused-parameter 处理，"
                    "建议将 ddp_find_unused_parameters 设为 False 以免首步额外开销。")
            logger.info(
                "DDP enabled: rank=%d/%d, device=%s, "
                "find_unused_parameters=%s, gradient_as_bucket_view=%s, "
                "static_graph=%s. "
                "Training grads all-reduce per backward. Note: math-equivalence "
                "to single-GPU under grad-accum holds for per-sample separable "
                "losses (BCE/Focal/per-sample Dice); batch-pooled ratio losses "
                "(batch_dice/Tversky/GDL) pool over the per-rank micro-batch, "
                "so their effective statistics window shrinks with accum/ranks "
                "(approximate, not strictly equivalent).",
                self._rank, self._world_size, self.device,
                tc.ddp_find_unused_parameters,
                tc.ddp_gradient_as_bucket_view,
                tc.ddp_static_graph)
        else:
            self.fwd_model = self.model

    def _setup_train_sampler(self) -> None:
        """训练采样器识别（每 epoch ``set_epoch`` 重洗）：单源 DDP 为
        DistributedSampler（loader.sampler），双源混合为 MixedBatchSampler
        （loader.batch_sampler）；均按 set_epoch 协议鸭子识别。"""
        self._train_sampler = None
        for _s in (getattr(self.train_loader, "sampler", None),
                   getattr(self.train_loader, "batch_sampler", None)):
            if _s is not None and callable(getattr(_s, "set_epoch", None)):
                self._train_sampler = _s
                break

    # ------------------------------------------------------------------
    # Training monitor（可选、异常隔离；见 taskcore.monitor）
    # ------------------------------------------------------------------
    def _setup_monitor(
        self,
        resume_active: bool,
        *,
        run_name_default: str = "run",
        save_best_metric: "Optional[str]" = None,
        save_best_mode: "Optional[str]" = None,
        save_best_criterion: "Optional[str]" = None,
        num_classes: int = 0,
        config_meta: "Optional[Dict]" = None,
    ) -> None:
        """实例化 ``MetricsLogger``、确定 HTML 落点并解析健康监测开关。

        由 ``cfg.monitor.enabled`` 守卫；落盘 / 渲染等副作用仅在 rank0 进行
        （无分布式属性的任务视为 rank0）。任何失败仅告警，绝不影响训练。
        ``save_best_metric``/``save_best_mode``/``save_best_criterion`` 缺省取
        ``train.save_best_metric``/``train.save_best_mode``/``train.save_best_criterion``。"""
        tc = self.cfg.train
        self._monitor = None
        self._monitor_html = None
        self._monitor_cfg = getattr(self.cfg, "monitor", None)
        mc = self._monitor_cfg
        is_main = bool(getattr(self, "_is_main", True))
        if mc is not None and mc.enabled and is_main:
            try:
                from ..monitor import MetricsLogger

                root = Path(mc.output_dir) if mc.output_dir else self.output_dir
                mon_dir = root / "monitor"
                self._monitor_html = root / (mc.filename
                                             or "training_monitor.html")
                run_name = (mc.run_name or self.output_dir.name
                            or run_name_default)
                self._monitor = MetricsLogger(
                    mon_dir,
                    run_name=run_name,
                    save_best_metric=(save_best_metric
                                      if save_best_metric is not None
                                      else tc.save_best_metric),
                    save_best_mode=(save_best_mode
                                    if save_best_mode is not None
                                    else tc.save_best_mode),
                    save_best_criterion=(save_best_criterion
                                         if save_best_criterion is not None
                                         else tc.save_best_criterion),
                    num_classes=num_classes,
                    total_epochs=tc.epochs,
                    config_meta=dict(config_meta or {}),
                    resume=resume_active,
                )
                logger.info(
                    "Training monitor enabled → metrics: %s | dashboard: %s",
                    mon_dir, self._monitor_html)
            except Exception as e:  # 隔离：监测初始化失败绝不阻断训练
                self._monitor = None
                logger.warning(
                    "Training monitor disabled (init failed): %s", e)
        # 模型健康监测：仅当监测启用、配置开启且在 rank0 时采集（成本极低，
        # 失败被隔离）。非有限步计数 / 梯度范数 / 裁剪比例 / 权重范数 / AMP 标度。
        self._health_monitor = bool(
            self._monitor is not None
            and getattr(mc, "health_monitor", False))
        self._health_grad_norm_when_no_clip = bool(
            getattr(mc, "health_grad_norm_when_no_clip", True))
        self._health_update_ratio = bool(
            self._health_monitor
            and getattr(mc, "health_update_ratio", False))

    def _monitor_log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: "Optional[Dict[str, float]]" = None,
        *,
        lr: float,
        gpu_peak_mib,
        wall_time_s: float,
        is_best: bool,
        last_epoch: bool,
    ) -> None:
        """落盘一个 epoch 并按 ``update_every`` 节奏重渲染 HTML（全程异常隔离）。"""
        if self._monitor is None:
            return
        mc = self._monitor_cfg
        try:
            self._monitor.log_epoch(
                epoch, train=train_metrics, val=val_metrics, lr=lr,
                gpu_peak_mib=gpu_peak_mib, wall_time_s=wall_time_s,
                is_best=is_best)
        except Exception as e:
            logger.warning(
                "Training monitor: log_epoch failed at epoch %d: %s",
                epoch + 1, e)
            return
        every = max(int(mc.update_every), 1)
        if is_best or last_epoch or ((epoch + 1) % every == 0):
            self._monitor_render(
                auto_reload_seconds=int(mc.auto_reload_seconds))

    def _monitor_render(self, *, auto_reload_seconds: int) -> None:
        """从已落盘历史重渲染单 run 仪表盘 HTML（异常隔离）。"""
        if self._monitor is None or self._monitor_html is None:
            return
        try:
            from ..monitor import MetricsHistory, write_dashboard

            hist = MetricsHistory.from_dir(self._monitor.dir)
            write_dashboard(hist, self._monitor_html,
                            auto_reload_seconds=auto_reload_seconds)
        except Exception as e:
            logger.warning("Training monitor: dashboard render failed: %s", e)

    def _monitor_finalize(self, status: str) -> None:
        """训练收尾：更新 run 状态并做一次静态（无自动刷新）终渲染。"""
        if self._monitor is None:
            return
        try:
            self._monitor.finalize(status)
        except Exception as e:
            logger.warning("Training monitor: finalize failed: %s", e)
        self._monitor_render(auto_reload_seconds=0)

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
