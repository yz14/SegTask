"""自监督预训练通用训练循环（方法无关）。

只负责 optimizer / scheduler / AMP / EMA / ckpt / 日志；"破坏→损失"全部交给
:class:`ssltask.methods.base.SSLMethod`。优化器/调度器/AMP/EMA/输出目录/epochs/lr
复用 segtask ``train.*`` 配置与工具（不另造轮子）。

产出 ckpt 的 ``model_state_dict`` 由 ``method.export_backbone_state_dict()`` 给出，键与
``segtask_v1.models.factory.build_model`` 同名 → 下游 ``train.pretrain`` 非严格加载衔接。
"""

from __future__ import annotations

import logging
import math
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DistributedSampler

from segtask_v1.data.augment import GPUAugmentor
from segtask_v1.trainer.amp import (
    _AMP_DTYPES, GradScaler, autocast, resolve_auto_amp_dtype)
from segtask_v1.trainer.checkpoint import (
    relocate_optimizer_state, restore_rng_state, unwrap_compile)
from segtask_v1.trainer.dist_utils import (
    all_reduce_flag_any, get_rank, get_world_size,
    is_dist_avail_and_initialized, is_main_process)
from segtask_v1.trainer.trainer import _reseed_rank_rng
from segtask_v1.trainer.optim import (
    WarmupScheduler, build_optimizer, build_scheduler)
from segtask_v1.utils import AverageMeter, ModelEMA, Timer

from ..methods.base import SSLMethod

logger = logging.getLogger(__name__)


class SSLTrainer:
    """方法无关的 SSL 预训练 pipeline。"""

    def __init__(self, method: SSLMethod, cfg, ssl, train_loader, device):
        self.method = method
        self.cfg = cfg
        self.ssl = ssl
        self.device = device
        self.train_loader = train_loader
        tc = cfg.train
        model = method.module

        # --- DDP（多卡）---------------------------------------------------
        # SSL 方法插件在 ``compute_loss`` 内直调子模块（dino 系 ``module.student``
        # 等），不经过单一 forward 入口，无法套 ``DistributedDataParallel``。改用
        # “初始广播 + accum 边界手动梯度 all-reduce 均值”：对任意调用模式都
        # 正确，代价是无反传-通信重叠（SSL 步内计算占比高，可接受）。
        self._rank = get_rank()
        self._world_size = get_world_size()
        self._is_main = is_main_process()
        self._is_dist = is_dist_avail_and_initialized() and self._world_size > 1
        if self._is_dist:
            with torch.no_grad():
                for t in model.state_dict().values():
                    if torch.is_tensor(t):
                        dist.broadcast(t, src=0)
            logger.info(
                "SSL DDP enabled: rank=%d/%d (manual grad all-reduce at "
                "accumulation boundaries; params/buffers broadcast from "
                "rank0).", self._rank, self._world_size)
        _sampler = getattr(train_loader, "sampler", None)
        self._train_sampler = (
            _sampler if isinstance(_sampler, DistributedSampler) else None)

        # --- Optimizer + scheduler (复用 segtask train.*) ---
        self.optimizer = build_optimizer(model, cfg)
        steps_per_epoch = len(train_loader)
        warmup_steps = tc.warmup_epochs * steps_per_epoch
        total_steps = tc.epochs * steps_per_epoch
        post_warmup = total_steps - warmup_steps
        if tc.scheduler == "one_cycle" and warmup_steps > 0:
            raise ValueError(
                "OneCycleLR has built-in warmup; set train.warmup_epochs=0.")
        base_scheduler = build_scheduler(
            self.optimizer, cfg, steps_per_epoch, post_warmup_steps=post_warmup)
        self.scheduler = WarmupScheduler(
            self.optimizer, base_scheduler, warmup_steps=warmup_steps,
            warmup_lr=tc.warmup_lr, base_lr=tc.lr)

        # --- AMP ---
        amp_dtype_cfg = tc.amp_dtype
        if amp_dtype_cfg == "auto":
            amp_dtype_cfg = resolve_auto_amp_dtype(device)
        if amp_dtype_cfg not in _AMP_DTYPES:
            raise ValueError(f"Unknown amp_dtype: {tc.amp_dtype!r}.")
        self.amp_dtype = _AMP_DTYPES[amp_dtype_cfg]
        self.use_amp = tc.use_amp and device.type == "cuda"
        self._scaler_active = self.use_amp and self.amp_dtype == torch.float16
        self.scaler = GradScaler("cuda", enabled=self._scaler_active)

        # --- EMA (over the method's module; orthogonal to any method-internal teacher) ---
        # 绑裸模型（compile 包装前），shadow key 无 ``_orig_mod.`` 前缀。
        self.ema = ModelEMA(model, tc.ema_decay) if tc.use_ema else None

        # --- 通用增强（仅重建类方法，见 SSLMethod.trainer_augment）-----------
        # 复用 segtask GPUAugmentor（cfg.augment 控制）：在 corruption/mask 之前
        # 对 batch 图像做空间/强度增强，增强后图即新的自洽重建样本。SSL 为
        # 单 FOV（multi_res_scales==[1.0]），max_scale=1。
        self.augmentor = None
        if method.trainer_augment and bool(cfg.augment.enabled):
            if int(cfg.model.spatial_dims) == 3:
                self.augmentor = GPUAugmentor(cfg.augment, max_scale=1.0)
                logger.info(
                    "Trainer-level augmentation ENABLED for SSL method %r "
                    "(GPUAugmentor, cfg.augment).", ssl.method)
            else:
                # 2.5D 在 dataset 层已把 D 折进通道（4D batch），而 GPUAugmentor
                # 的空间变换按 (B,C,D,H,W) 3D 体实现，不适用折叠布局。
                logger.warning(
                    "cfg.augment.enabled=True but SSL trainer-level "
                    "augmentation only supports spatial_dims=3 (2.5D input "
                    "is depth-folded 4D); augmentation disabled.")

        self.grad_accum_steps = max(tc.grad_accum_steps, 1)
        self.output_dir = Path(tc.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = math.inf
        self.best_probe = -math.inf
        self._best_saved = False
        self._global_step = 0
        self.start_epoch = 0

        # 告知方法总优化步数（= boundary 数：每 grad_accum 个 micro-step 一次）。
        # 供自蒸馏等方法预计算 EMA 动量 / teacher 温度的 cosine 调度（默认 no-op）。
        opt_steps_per_epoch = math.ceil(steps_per_epoch / self.grad_accum_steps)
        self.method.configure_schedule(tc.epochs * max(opt_steps_per_epoch, 1))

        # --- Online seg probe (§0.5): drives best selection by representation
        #     quality instead of the SSL proxy loss. Optional, isolated.
        #     DDP 下仅 rank0 跑探针（选模/落盘也仅 rank0）。 ---
        self.probe = None
        if bool(getattr(ssl, "probe_enabled", False)) and self._is_main:
            from ..eval.probe import SegProbe
            self.probe = SegProbe(cfg, ssl, device)
            logger.info(
                "Online seg probe ENABLED: every %d epoch(s), %d iters, "
                "select_best_by=%s.", ssl.probe_every, ssl.probe_iters,
                "probe_dice" if ssl.probe_select_best else "train_loss")

        # --- Resume (全状态：method/optimizer/scheduler/scaler/EMA/RNG) ---
        if tc.resume:
            if os.path.isfile(tc.resume):
                self._load_resume(tc.resume)
            else:
                logger.warning(
                    "`train.resume` is set but file not found: %s. "
                    "Starting SSL pretrain from scratch.", tc.resume)

        # --- torch.compile（最后：optimizer/EMA/resume 已绑裸模型参数）-----
        # 替换 ``method.module``：直接前向的重建类方法（genesis/simmim/spark 等
        # ``self.module(x)``）被编译；子模块直调的方法（dino 系 ``module.student``）
        # 经 OptimizedModule 属性代理回到裸子模块，行为不变。各方法的
        # ``export_backbone_state_dict`` 已统一走 ``unwrap_compile``。
        if tc.compile_mode != "none" and hasattr(torch, "compile"):
            triton_ok = True
            if device.type == "cuda":
                import importlib.util
                if importlib.util.find_spec("triton") is None:
                    triton_ok = False
                    logger.warning(
                        "torch.compile (mode='%s') requested but Triton not "
                        "installed; falling back to eager.", tc.compile_mode)
            if triton_ok:
                logger.info("Compiling SSL module with mode='%s'",
                            tc.compile_mode)
                self.method.module = torch.compile(
                    self.method.module, mode=tc.compile_mode)

    # ------------------------------------------------------------------
    def _prepare(self, batch: Dict) -> Dict:
        """把 batch 内张量搬到 device 并转 fp32（非张量原样透传）。"""
        out: Dict = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device, non_blocking=True).float()
            else:
                out[k] = v
        return out

    def _export_state_dict(self) -> Dict:
        """EMA 优先的可迁移骨干权重快照（键与 build_model 同名）。"""
        if self.ema is not None:
            bare = unwrap_compile(self.method.module)
            self.ema.apply_shadow(bare)
            try:
                return self.method.export_backbone_state_dict()
            finally:
                self.ema.restore(bare)
        return self.method.export_backbone_state_dict()

    def _save(self, epoch: int, tag: str) -> Path:
        path = self.output_dir / f"ssl_{tag}.pt"
        if not self._is_main:          # DDP：仅 rank0 落盘
            if tag == "best":
                self._best_saved = True
            return path
        torch.save({
            "epoch": epoch,
            "model_state_dict": self._export_state_dict(),
            "ssl_method": self.ssl.method,
            "best_loss": self.best_loss,
            "best_probe": self.best_probe,
        }, path)
        if tag == "best":
            self._best_saved = True
        return path

    def _save_resume(self, epoch: int) -> Path:
        """全状态 resume checkpoint（与 ssl_best/last 的导出快照正交）：
        method 完整 state_dict（含方法内 teacher/queue/center 等 buffer）+
        optimizer/scheduler/scaler/EMA + 进度/最优指标 + RNG。"""
        rng_state = {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        state: Dict = {
            "epoch": epoch,
            "ssl_method": self.ssl.method,
            "method_state_dict": unwrap_compile(self.method.module).state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "global_step": self._global_step,
            "best_loss": self.best_loss,
            "best_probe": self.best_probe,
            "best_saved": self._best_saved,
            "rng_state": rng_state,
        }
        if self.ema is not None:
            state["ema_state_dict"] = self.ema.state_dict()
        path = self.output_dir / "ssl_resume.pt"
        if self._is_main:              # DDP：仅 rank0 落盘
            torch.save(state, path)
        return path

    def _load_resume(self, path: str) -> None:
        logger.info("Resuming SSL pretrain from: %s", path)
        # ckpt 含 numpy/python RNG 对象，为本 trainer 自写；显式关 weights_only。
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        method = ckpt.get("ssl_method")
        if method is not None and method != self.ssl.method:
            raise ValueError(
                f"Resume ckpt was written by ssl.method={method!r}, but current "
                f"config uses {self.ssl.method!r}.")
        unwrap_compile(self.method.module).load_state_dict(
            ckpt["method_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        relocate_optimizer_state(self.optimizer)
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if self.ema is not None and "ema_state_dict" in ckpt:
            self.ema.load_state_dict(ckpt["ema_state_dict"])
        self._global_step = int(ckpt.get("global_step", 0))
        self.method.on_resume(self._global_step)
        self.start_epoch = int(ckpt.get("epoch", -1)) + 1
        self.best_loss = float(ckpt.get("best_loss", math.inf))
        self.best_probe = float(ckpt.get("best_probe", -math.inf))
        self._best_saved = bool(ckpt.get("best_saved", False))
        rng = ckpt.get("rng_state")
        if rng:
            try:
                restore_rng_state(rng)
                logger.info("Restored RNG state from resume checkpoint.")
            except Exception as e:  # pragma: no cover
                logger.warning("Failed to restore RNG state: %s", e)
        # ckpt 的 RNG 快照来自 rank0；rank>0 重新分流，避免各 rank 退化同流。
        if self._is_dist and self._rank > 0:
            _reseed_rank_rng(
                self.cfg.train.seed, self._rank, self.start_epoch,
                self.cfg.train.deterministic)
        logger.info(
            "Resume: start_epoch=%d, global_step=%d, best_loss=%.5f, "
            "best_probe=%.4f", self.start_epoch, self._global_step,
            self.best_loss, self.best_probe)

    # ------------------------------------------------------------------
    def fit(self) -> Dict[str, float]:
        tc = self.cfg.train
        timer = Timer()
        total_params = sum(p.numel() for p in self.method.parameters()) / 1e6
        logger.info("=" * 60)
        logger.info("SSL pretrain (%s): %d epochs, device=%s",
                    self.ssl.method, tc.epochs, self.device)
        logger.info("Model params: %.2fM | recon_loss=%s | AMP=%s EMA=%s",
                    total_params, self.ssl.recon_loss,
                    self.use_amp, self.ema is not None)
        logger.info("Train batches: %d, output_dir=%s",
                    len(self.train_loader), self.output_dir)
        logger.info("=" * 60)

        use_probe_select = self.probe is not None and bool(self.ssl.probe_select_best)
        for epoch in range(self.start_epoch, tc.epochs):
            train_loss = self._train_epoch(epoch)
            elapsed = timer.elapsed()
            logger.info("[SSL epoch %d/%d] loss=%.5f lr=%.2e (%.1fs)",
                        epoch + 1, tc.epochs, train_loss,
                        self.scheduler.get_lr(), elapsed)

            improved_loss = train_loss < self.best_loss
            if improved_loss:
                self.best_loss = train_loss

            # --- online probe (§0.5): periodic representation-quality eval ---
            is_last = epoch == tc.epochs - 1
            probe_dice = None
            if self.probe is not None and (
                    (epoch + 1) % self.ssl.probe_every == 0 or is_last):
                try:
                    probe_dice = self.probe.evaluate(
                        self._export_state_dict())["probe_dice"]
                    logger.info("[SSL epoch %d] online probe Dice=%.4f "
                                "(best=%.4f)", epoch + 1, probe_dice,
                                max(self.best_probe, probe_dice))
                except Exception:  # 探针失败绝不打断预训练
                    logger.warning("Online probe failed at epoch %d; skipping.",
                                   epoch + 1, exc_info=True)
                    probe_dice = None
            improved_probe = probe_dice is not None and probe_dice > self.best_probe
            if improved_probe:
                self.best_probe = probe_dice

            # --- best ckpt selection: probe Dice (if enabled) else train loss ---
            if use_probe_select:
                if improved_probe:
                    p = self._save(epoch, "best")
                    logger.info("New best probe Dice=%.4f -> %s", probe_dice, p)
            elif improved_loss:
                p = self._save(epoch, "best")
                logger.info("New best SSL loss=%.5f -> %s", train_loss, p)

            if (epoch + 1) % self.ssl.save_every == 0 or is_last:
                self._save(epoch, "last")
                self._save_resume(epoch)

        # 保底：若选模策略从未保存过 best（如探针全程失败），最后兜底存一次。
        if not self._best_saved:
            self._save(tc.epochs - 1, "best")
            logger.info("No best ckpt selected during training; saved final "
                        "state as ssl_best.pt (fallback).")

        logger.info("SSL pretrain done. best_loss=%.5f, best_probe=%.4f. Use "
                    "ssl_best.pt via train.pretrain for downstream.",
                    self.best_loss, self.best_probe)
        out = {"best_loss": self.best_loss}
        if self.probe is not None:
            out["best_probe"] = self.best_probe
        return out

    @staticmethod
    def _effective_accum(step: int, total_steps: int, accum: int) -> int:
        """尾批 micro-batch 数不满 ``accum`` 时，用真实尾长作分母，
        以免最后一组 micro-batch 因被除以 ``accum`` 而权重偏小。"""
        if accum <= 1:
            return 1
        remainder = total_steps % accum
        partial_start = total_steps - remainder
        return remainder if (remainder > 0 and step >= partial_start) else accum

    def _sync_grads(self) -> None:
        """DDP：accum 边界对全部梯度做均值 all-reduce。

        方法插件直调子模块、无单一 forward 入口，故不用 DDP wrapper 而手动
        同步；与 wrapper 的 bucket all-reduce 数学等价（无重叠优化）。fp16 下
        梯度带 scale，均值后 scale 不变，GradScaler 语义不受影响。"""
        if not self._is_dist:
            return
        grads = [p.grad for p in self.method.parameters()
                 if p.grad is not None]
        if not grads:
            return
        torch._foreach_div_(grads, float(self._world_size))
        for g in grads:
            dist.all_reduce(g, op=dist.ReduceOp.SUM)

    def _train_epoch(self, epoch: int) -> float:
        self.method.train()
        loss_meter = AverageMeter()
        tc = self.cfg.train
        accum = self.grad_accum_steps
        total_steps = len(self.train_loader)

        # DDP：每 epoch 重置 DistributedSampler 的洗牌种子。
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(epoch)

        self.optimizer.zero_grad(set_to_none=True)
        group_has_nonfinite = False
        for step, batch in enumerate(self.train_loader):
            batch = self._prepare(batch)
            bs = batch["image"].shape[0] if "image" in batch else tc.batch_size
            if self.augmentor is not None and "image" in batch:
                img = batch["image"]
                # image-only：传入 dummy 单通道 label 满足管道接口，结果丢弃。
                img, _, _ = self.augmentor(
                    img, torch.zeros_like(img[:, :1]))
                batch["image"] = img

            with autocast(device_type="cuda", enabled=self.use_amp,
                          dtype=self.amp_dtype):
                loss, logs = self.method.compute_loss(batch)
            # 尾批 micro-batch 数不满 accum 时用真实尾长作分母，避免尾组梯度
            # 因除以 accum 而权重偏小。
            effective_accum = self._effective_accum(step, total_steps, accum)
            if effective_accum > 1:
                loss = loss / effective_accum
            self.scaler.scale(loss).backward()

            step_loss = loss.item() * effective_accum
            if not math.isfinite(step_loss):
                group_has_nonfinite = True

            is_boundary = ((step + 1) % accum == 0 or (step + 1) == total_steps)
            if is_boundary:
                # DDP：先均值 all-reduce 各 rank 梯度（与 DDP wrapper 数学等价）。
                self._sync_grads()
                # fp16 由 GradScaler 内部跳过含 inf/NaN 梯度的优化步；bf16/fp32
                # 无此保护，loss 非有限时丢弃本 accum 组梯度，避免 NaN 永久污染
                # 权重与 EMA（scheduler 照常推进，EMA 不推进）。loss 非有限是
                # rank 本地信息，DDP 下跳步决策需 all-reduce(any) 统一，维持
                # 各副本施加相同更新的不变量。
                skip_optim_step = group_has_nonfinite and not self._scaler_active
                if self._is_dist and not self._scaler_active:
                    skip_optim_step = all_reduce_flag_any(
                        group_has_nonfinite, self.device)
                group_has_nonfinite = False
                if skip_optim_step:
                    logger.warning(
                        "Skipping optimizer step at epoch %d step %d/%d: "
                        "non-finite loss in this accumulation group "
                        "(amp_dtype without GradScaler protection).",
                        epoch + 1, step + 1, total_steps)
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                else:
                    if tc.grad_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            self.method.parameters(), tc.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    if self.ema is not None:
                        self.ema.update(unwrap_compile(self.method.module))
                self._global_step += 1
                self.method.on_after_step(self._global_step)

            if math.isfinite(step_loss):
                loss_meter.update(step_loss, bs)
            else:
                logger.warning(
                    "Non-finite SSL loss at epoch %d step %d/%d; excluded "
                    "from loss meter.", epoch + 1, step + 1, total_steps)

            if (step + 1) % tc.log_every == 0 or step == 0:
                logger.debug("  [%d/%d] loss=%.5f lr=%.2e",
                             step + 1, total_steps, step_loss,
                             self.scheduler.get_lr())
        return loss_meter.avg


__all__ = ["SSLTrainer"]
