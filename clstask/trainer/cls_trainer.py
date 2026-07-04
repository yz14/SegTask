"""分类训练器（复用 segtask 训练基建：optim / warmup / AMP / EMA）。

与 ``gentask.GenerationTrainer`` 同构的独立精简训练器：

* 训练：patch → forward → BCE/CE/Focal（可选 mixup/cutmix）→ 反传；
  损失在 autocast 外以 fp32 计算（logit clamp 防溢出，沿用 segtask 惯例）。
* 验证：收集全量 logits/targets → AUC / F1 / acc；按 ``cls.save_best_metric``
  选模保存 ``best_model.pth``（含 config / EMA）。
* encoder 差分学习率：``cls.encoder_lr_mult`` 对 encoder 参数组缩放 lr
  （微调预训练权重的惯用手段）；头部参数保持 ``train.lr``。
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from segtask_v1.config import Config as SegConfig
from segtask_v1.trainer.amp import (
    _LOGIT_CLAMP,
    GradScaler,
    resolve_auto_amp_dtype,
)
from segtask_v1.trainer.checkpoint import unwrap_compile
from segtask_v1.trainer.optim import (
    WarmupScheduler,
    build_optimizer,
    build_scheduler,
)
from segtask_v1.utils import AverageMeter, ModelEMA

from ..config import ClsConfig, resolve_num_classes
from ..losses.cls_loss import build_cls_loss
from ..metrics import multilabel_metrics, singlelabel_metrics
from .mixup import apply_mixup_cutmix

logger = logging.getLogger(__name__)

_AMP_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16,
               "float32": torch.float32}


def _build_optimizer_with_lr_mult(model: nn.Module, cfg: SegConfig,
                                  encoder_lr_mult: float):
    """AdamW/Adam/SGD + weight-decay 分组 + encoder 学习率倍率。

    分组口径与 ``segtask_v1.trainer.optim._param_groups`` 一致（ndim<=1 免
    decay），再按参数属于 encoder 与否二分（2×2 组）。注意 warmup 段
    ``WarmupScheduler`` 对所有组施加统一 lr，倍率在 warmup 结束后由 base
    scheduler 按各组 base_lr 恢复。
    """
    tc = cfg.train
    enc_ids = {id(p) for p in model.encoder.parameters()}
    groups = []
    for is_enc in (True, False):
        lr = tc.lr * (encoder_lr_mult if is_enc else 1.0)
        for wd, pred in ((tc.weight_decay, lambda p: p.ndim >= 2),
                         (0.0, lambda p: p.ndim <= 1)):
            params = [p for p in model.parameters()
                      if p.requires_grad and (id(p) in enc_ids) == is_enc
                      and pred(p)]
            if params:
                groups.append(
                    {"params": params, "lr": lr, "weight_decay": wd})
    if tc.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=tc.lr)
    if tc.optimizer == "adam":
        return torch.optim.Adam(groups, lr=tc.lr)
    if tc.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=tc.lr, momentum=tc.momentum,
                               nesterov=tc.nesterov)
    raise ValueError(f"Unknown optimizer: {tc.optimizer}")


class ClsTrainer:
    """分类训练器；``fit()`` 返回 best 指标摘要。"""

    def __init__(self, model: nn.Module, cfg: SegConfig, cls: ClsConfig,
                 train_loader, val_loader, device: torch.device):
        self.cfg = cfg
        self.cls = cls
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(device)
        tc = cfg.train

        self.num_classes = resolve_num_classes(cls, cfg)
        self.loss_fn = build_cls_loss(cfg, cls).to(device)
        self.single_label = not cls.multi_label

        if abs(cls.encoder_lr_mult - 1.0) > 1e-9:
            self.optimizer = _build_optimizer_with_lr_mult(
                self.model, cfg, cls.encoder_lr_mult)
        else:
            self.optimizer = build_optimizer(self.model, cfg)
        steps_per_epoch = max(len(train_loader), 1)
        warmup_steps = tc.warmup_epochs * steps_per_epoch
        total_steps = tc.epochs * steps_per_epoch
        base_scheduler = build_scheduler(
            self.optimizer, cfg, steps_per_epoch,
            post_warmup_steps=total_steps - warmup_steps)
        self.scheduler = WarmupScheduler(
            self.optimizer, base_scheduler, warmup_steps=warmup_steps,
            warmup_lr=tc.warmup_lr, base_lr=tc.lr)

        amp_name = tc.amp_dtype
        if amp_name == "auto":
            amp_name = resolve_auto_amp_dtype(device)
        self.amp_dtype = _AMP_DTYPES.get(amp_name, torch.float32)
        self.use_amp = tc.use_amp and device.type == "cuda"
        self.scaler = GradScaler(
            "cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)

        self.ema = ModelEMA(self.model, tc.ema_decay) if tc.use_ema else None
        self.grad_accum_steps = max(tc.grad_accum_steps, 1)

        self.output_dir = Path(tc.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_key = cls.save_best_metric        # auc | f1 | acc | loss
        self.best_sign = -1.0 if self.best_key == "loss" else 1.0
        self.best_metric = -math.inf
        self.best_epoch = 0

    # ------------------------------------------------------------------
    def _loss_fp32(self, logits: torch.Tensor,
                   target: torch.Tensor) -> torch.Tensor:
        """autocast 外 fp32 损失（logit clamp 防 fp16 溢出）。"""
        logits = logits.float().clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)
        with torch.autocast(device_type=self.device.type, enabled=False):
            return self.loss_fn(logits, target)

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        loss_meter = AverageMeter()
        accum = self.grad_accum_steps
        total = len(self.train_loader)
        self.optimizer.zero_grad(set_to_none=True)
        use_mix = ((self.cls.mixup_alpha > 0 or self.cls.cutmix_alpha > 0)
                   and self.cls.label_granularity == "volume")
        for step, batch in enumerate(self.train_loader):
            img = batch["image"].to(self.device, non_blocking=True).float()
            target = batch["target"].to(self.device, non_blocking=True)
            if use_mix:
                img, target = apply_mixup_cutmix(
                    img, target, self.num_classes,
                    self.cls.mixup_alpha, self.cls.cutmix_alpha,
                    self.cls.mixup_prob)
            with torch.autocast(device_type=self.device.type,
                                enabled=self.use_amp, dtype=self.amp_dtype):
                logits = self.model(img)
            loss = self._loss_fp32(logits, target)
            loss_scaled = loss / accum if accum > 1 else loss
            self.scaler.scale(loss_scaled).backward()

            if (step + 1) % accum == 0 or (step + 1) == total:
                if self.cfg.train.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.train.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                if self.ema is not None:
                    self.ema.update(self.model)

            if math.isfinite(loss.item()):
                loss_meter.update(loss.item(), img.shape[0])
            if (step + 1) % max(self.cfg.train.log_every, 1) == 0 or step == 0:
                logger.debug("  [%d/%d] loss=%.4f lr=%.2e", step + 1, total,
                             loss.item(), self.scheduler.get_lr())
        return {"loss": loss_meter.avg}

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        if self.ema is not None:
            self.ema.apply_shadow(self.model)
        all_logits, all_targets = [], []
        loss_meter = AverageMeter()
        try:
            for batch in self.val_loader:
                img = batch["image"].to(self.device).float()
                target = batch["target"].to(self.device)
                logits = self.model(img)
                loss_meter.update(
                    float(self._loss_fp32(logits, target).item()),
                    img.shape[0])
                all_logits.append(logits.float().cpu())
                all_targets.append(target.cpu())
        finally:
            if self.ema is not None:
                self.ema.restore(self.model)
        logits = torch.cat(all_logits)
        targets = torch.cat(all_targets)
        if self.single_label:
            probs = torch.softmax(logits, dim=1)
            m = singlelabel_metrics(probs, targets)
        else:
            probs = torch.sigmoid(logits)
            m = multilabel_metrics(probs, targets)
        m["loss"] = loss_meter.avg
        if m.get("auc_defined_classes", 1.0) == 0.0:
            logger.warning(
                "val AUC undefined for all classes (single-class val split); "
                "reported auc=0.5. Consider more val volumes or stratified "
                "split.")
        return m

    def _save_best(self, epoch: int, metrics: Dict[str, float]) -> None:
        bare = unwrap_compile(self.model)
        state = {
            "epoch": epoch,
            "model_state_dict": bare.state_dict(),
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "metrics": metrics,
            "config": self.cfg,
            "cls_config": self.cls,
        }
        if self.ema is not None:
            state["ema_state_dict"] = self.ema.state_dict()
        torch.save(state, self.output_dir / "best_model.pth")
        logger.info("Best cls model saved (%s=%.4f) @ epoch %d",
                    self.best_key, metrics[self.best_key], epoch + 1)

    def fit(self) -> Dict[str, float]:
        last: Dict[str, float] = {}
        self.history: list = []   # 逐 epoch {train_loss, val_*}
        for epoch in range(self.cfg.train.epochs):
            tr = self._train_epoch(epoch)
            val = self._validate(epoch)
            self.scheduler.step_epoch(val.get(self.best_key))
            logger.info(
                "Epoch %d/%d: train_loss=%.4f val_loss=%.4f val_auc=%.4f "
                "val_f1=%.4f val_acc=%.4f",
                epoch + 1, self.cfg.train.epochs, tr["loss"], val["loss"],
                val["auc"], val["f1"], val["acc"])
            score = self.best_sign * val[self.best_key]
            if score > self.best_metric:
                self.best_metric = score
                self.best_epoch = epoch
                self._save_best(epoch, val)
            last = {**tr, **{f"val_{k}": v for k, v in val.items()}}
            self.history.append(last)
        return {f"best_{self.best_key}": self.best_sign * self.best_metric,
                "best_epoch": float(self.best_epoch), **last}


__all__ = ["ClsTrainer"]
