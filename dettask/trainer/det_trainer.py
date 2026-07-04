"""检测训练器（复用 segtask 训练基建：optim / warmup / AMP / EMA）。

* 训练：patch + 变长框 → ``DetectorModel`` 损失 dict → 求和反传；
  head 内部损失已在 fp32 计算（focal / GIoU 数值敏感）。
* 验证：patch 级 predict → mAP@``det.eval_iou_thresh``（体级 FROC 由
  predictor 在拼接后的 3D 框上给出）；按 ``det.save_best_metric`` 选模。
* encoder 差分学习率复用 clstask 的分组实现。
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from segtask_v1.config import Config as SegConfig
from segtask_v1.trainer.amp import GradScaler, resolve_auto_amp_dtype
from segtask_v1.trainer.checkpoint import unwrap_compile
from segtask_v1.trainer.optim import (
    WarmupScheduler,
    build_optimizer,
    build_scheduler,
)
from segtask_v1.utils import AverageMeter, ModelEMA

from clstask.trainer.cls_trainer import _build_optimizer_with_lr_mult

from ..config import DetConfig, resolve_num_classes
from ..metrics import detection_map

logger = logging.getLogger(__name__)

_AMP_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16,
               "float32": torch.float32}


class DetTrainer:
    """检测训练器；``fit()`` 返回 best 指标摘要。"""

    def __init__(self, model: nn.Module, cfg: SegConfig, det: DetConfig,
                 train_loader, val_loader, device: torch.device):
        self.cfg = cfg
        self.det = det
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(device)
        tc = cfg.train

        self.num_classes = resolve_num_classes(det, cfg)
        if abs(det.encoder_lr_mult - 1.0) > 1e-9:
            self.optimizer = _build_optimizer_with_lr_mult(
                self.model, cfg, det.encoder_lr_mult)
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
        self.best_key = det.save_best_metric        # map | loss
        self.best_sign = -1.0 if self.best_key == "loss" else 1.0
        self.best_metric = -math.inf
        self.best_epoch = 0

    # ------------------------------------------------------------------
    def _to_device(self, batch):
        img = batch["image"].to(self.device, non_blocking=True).float()
        boxes = [b.to(self.device) for b in batch["boxes"]]
        labels = [l.to(self.device) for l in batch["labels"]]
        return img, boxes, labels

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        loss_meter = AverageMeter()
        part_meters: Dict[str, AverageMeter] = {}
        accum = self.grad_accum_steps
        total = len(self.train_loader)
        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(self.train_loader):
            img, boxes, labels = self._to_device(batch)
            with torch.autocast(device_type=self.device.type,
                                enabled=self.use_amp, dtype=self.amp_dtype):
                losses = self.model(img, boxes, labels)
            loss = sum(losses.values())
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
                for k, v in losses.items():
                    part_meters.setdefault(k, AverageMeter()).update(
                        float(v.item()), img.shape[0])
            if (step + 1) % max(self.cfg.train.log_every, 1) == 0 or step == 0:
                logger.debug("  [%d/%d] loss=%.4f (%s) lr=%.2e",
                             step + 1, total, loss.item(),
                             ", ".join(f"{k}={v.item():.3f}"
                                       for k, v in losses.items()),
                             self.scheduler.get_lr())
        out = {"loss": loss_meter.avg}
        out.update({k: m.avg for k, m in part_meters.items()})
        return out

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        if self.ema is not None:
            self.ema.apply_shadow(self.model)
        preds, gts = [], []
        loss_meter = AverageMeter()
        try:
            for batch in self.val_loader:
                img, boxes, labels = self._to_device(batch)
                dets = self.model(img)
                losses = self.model.det_head.compute_loss(
                    self.model.extract_pyramid(img), boxes, labels,
                    list(img.shape[2:]))
                loss_meter.update(float(sum(losses.values()).item()),
                                  img.shape[0])
                preds.extend(dets)
                gts.extend([(b.cpu(), l.cpu())
                            for b, l in zip(boxes, labels)])
        finally:
            if self.ema is not None:
                self.ema.restore(self.model)
        m = detection_map(preds, gts, self.num_classes,
                          self.det.eval_iou_thresh)
        m["loss"] = loss_meter.avg
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
            "det_config": self.det,
        }
        if self.ema is not None:
            state["ema_state_dict"] = self.ema.state_dict()
        torch.save(state, self.output_dir / "best_model.pth")
        logger.info("Best det model saved (%s=%.4f) @ epoch %d",
                    self.best_key, metrics[self.best_key], epoch + 1)

    def fit(self) -> Dict[str, float]:
        last: Dict[str, float] = {}
        self.history: list = []
        for epoch in range(self.cfg.train.epochs):
            tr = self._train_epoch(epoch)
            val = self._validate(epoch)
            self.scheduler.step_epoch(val.get(self.best_key))
            logger.info(
                "Epoch %d/%d: train_loss=%.4f val_loss=%.4f val_map=%.4f",
                epoch + 1, self.cfg.train.epochs, tr["loss"], val["loss"],
                val["map"])
            score = self.best_sign * val[self.best_key]
            if score > self.best_metric:
                self.best_metric = score
                self.best_epoch = epoch
                self._save_best(epoch, val)
            last = {**tr, **{f"val_{k}": v for k, v in val.items()}}
            self.history.append(last)
        return {f"best_{self.best_key}": self.best_sign * self.best_metric,
                "best_epoch": float(self.best_epoch), **last}


__all__ = ["DetTrainer"]
