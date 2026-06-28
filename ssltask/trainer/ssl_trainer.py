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
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from segtask_v1.trainer.amp import (
    _AMP_DTYPES, GradScaler, autocast, resolve_auto_amp_dtype)
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
        self.ema = ModelEMA(model, tc.ema_decay) if tc.use_ema else None

        self.grad_accum_steps = max(tc.grad_accum_steps, 1)
        self.output_dir = Path(tc.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = math.inf
        self.best_probe = -math.inf
        self._best_saved = False
        self._global_step = 0

        # 告知方法总优化步数（= boundary 数：每 grad_accum 个 micro-step 一次）。
        # 供自蒸馏等方法预计算 EMA 动量 / teacher 温度的 cosine 调度（默认 no-op）。
        opt_steps_per_epoch = math.ceil(steps_per_epoch / self.grad_accum_steps)
        self.method.configure_schedule(tc.epochs * max(opt_steps_per_epoch, 1))

        # --- Online seg probe (§0.5): drives best selection by representation
        #     quality instead of the SSL proxy loss. Optional, isolated. ---
        self.probe = None
        if bool(getattr(ssl, "probe_enabled", False)):
            from ..eval.probe import SegProbe
            self.probe = SegProbe(cfg, ssl, device)
            logger.info(
                "Online seg probe ENABLED: every %d epoch(s), %d iters, "
                "select_best_by=%s.", ssl.probe_every, ssl.probe_iters,
                "probe_dice" if ssl.probe_select_best else "train_loss")

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
            self.ema.apply_shadow(self.method.module)
            try:
                return self.method.export_backbone_state_dict()
            finally:
                self.ema.restore(self.method.module)
        return self.method.export_backbone_state_dict()

    def _save(self, epoch: int, tag: str) -> Path:
        path = self.output_dir / f"ssl_{tag}.pt"
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
        for epoch in range(tc.epochs):
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

    def _train_epoch(self, epoch: int) -> float:
        self.method.train()
        loss_meter = AverageMeter()
        tc = self.cfg.train
        accum = self.grad_accum_steps
        total_steps = len(self.train_loader)

        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(self.train_loader):
            batch = self._prepare(batch)
            bs = batch["image"].shape[0] if "image" in batch else tc.batch_size

            with autocast(device_type="cuda", enabled=self.use_amp,
                          dtype=self.amp_dtype):
                loss, logs = self.method.compute_loss(batch)
            if accum > 1:
                loss = loss / accum
            self.scaler.scale(loss).backward()

            is_boundary = ((step + 1) % accum == 0 or (step + 1) == total_steps)
            if is_boundary:
                if tc.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.method.parameters(), tc.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                if self.ema is not None:
                    self.ema.update(self.method.module)
                self._global_step += 1
                self.method.on_after_step(self._global_step)

            step_loss = loss.item() * (accum if accum > 1 else 1)
            if math.isfinite(step_loss):
                loss_meter.update(step_loss, bs)
            else:
                logger.warning(
                    "Non-finite SSL loss at epoch %d step %d/%d; skipping.",
                    epoch + 1, step + 1, total_steps)

            if (step + 1) % tc.log_every == 0 or step == 0:
                logger.debug("  [%d/%d] loss=%.5f lr=%.2e",
                             step + 1, total_steps, step_loss,
                             self.scheduler.get_lr())
        return loss_meter.avg


__all__ = ["SSLTrainer"]
