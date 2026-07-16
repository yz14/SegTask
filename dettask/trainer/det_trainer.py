"""检测训练器（复用 segtask 训练基建：optim / warmup / AMP / EMA）。

* 训练：patch + 变长框 → ``DetectorModel`` 损失 dict → 求和反传；
  head 内部损失已在 fp32 计算（focal / GIoU 数值敏感）。空间增强（flip）
  在 dataset 内与框联动施加；强度增强复用 seg ``GPUAugmentor``（仅强度
  分支，不改变几何，故不动框）。bf16/fp32（无 GradScaler）路径下 loss/
  梯度非有限时丢弃本 accum 组，不推 scheduler/EMA（口径同 segtask）。
* 验证：patch 级 predict → mAP@``det.eval_iou_thresh``（体级 FROC 由
  predictor 在拼接后的 3D 框上给出）；按 ``det.save_best_metric`` 选模。
* encoder 差分学习率复用 clstask 的分组实现；warmup 段保留各组倍率
  （:class:`clstask.trainer.cls_trainer._GroupWarmupScheduler`）。
* 工程能力（口径同 seg / cls Trainer）：每 epoch 落盘 ``latest_model.pth``
  （含 model/EMA/optimizer/scheduler/scaler/epoch/best 状态）与
  ``history.json``；``train.resume`` 完整恢复续训；``train.early_stopping``
  连续 N 次验证无提升时提前停止。
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from segtask_v1.config import Config as SegConfig
from segtask_v1.data.augment import GPUAugmentor
from segtask_v1.trainer.amp import GradScaler, resolve_auto_amp_dtype
from segtask_v1.trainer.checkpoint import unwrap_compile
from segtask_v1.trainer.optim import build_optimizer, build_scheduler
from segtask_v1.trainer.prefetch import CudaPrefetcher
from segtask_v1.utils import AverageMeter, ModelEMA

from clstask.trainer.cls_trainer import (
    _build_optimizer_with_lr_mult,
    _GroupWarmupScheduler,
)

from ..config import DetConfig, resolve_num_classes
from ..metrics import detection_map

logger = logging.getLogger(__name__)

_AMP_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16,
               "float32": torch.float32}


def _intensity_only_augcfg(aug):
    """AugConfig → 仅保留强度分支（空间/遮挡概率清零）的副本。

    检测的空间增强必须与框联动（dataset 内 flip 已联动）；affine/elastic/
    grid_dropout 无框变换实现，这里显式关闭，避免图像几何与框错位。
    """
    return dataclasses.replace(
        aug,
        random_flip_prob=0.0,
        random_affine_prob=0.0,
        elastic_deform_prob=0.0,
        grid_dropout_prob=0.0,
    )


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
        # 调度时钟按优化步（非 micro-step）计，与 scheduler.step() 的调用频率
        # 一致（accum 尾组也触发一步，口径同 segtask Trainer）。
        self.grad_accum_steps = max(tc.grad_accum_steps, 1)
        steps_per_epoch = max(
            math.ceil(len(train_loader) / self.grad_accum_steps), 1)
        warmup_steps = tc.warmup_epochs * steps_per_epoch
        total_steps = tc.epochs * steps_per_epoch
        base_scheduler = build_scheduler(
            self.optimizer, cfg, steps_per_epoch,
            post_warmup_steps=total_steps - warmup_steps)
        # one_cycle 自带 warmup（pct_start），外层不再叠加线性 warmup。
        warmup_steps = 0 if tc.scheduler == "one_cycle" else warmup_steps
        # 差分学习率下 warmup 保留各组倍率（全组同 lr 时退化为父类行为）。
        self.scheduler = _GroupWarmupScheduler(
            self.optimizer, base_scheduler, warmup_steps=warmup_steps,
            warmup_lr=tc.warmup_lr, base_lr=tc.lr,
            group_base_lrs=[float(pg["lr"])
                            for pg in self.optimizer.param_groups])

        amp_name = tc.amp_dtype
        if amp_name == "auto":
            amp_name = resolve_auto_amp_dtype(device)
        self.amp_dtype = _AMP_DTYPES.get(amp_name, torch.float32)
        self.use_amp = tc.use_amp and device.type == "cuda"
        self.scaler = GradScaler(
            "cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)
        self._scaler_active = self.use_amp and self.amp_dtype == torch.float16

        self.ema = ModelEMA(self.model, tc.ema_decay) if tc.use_ema else None
        self.grad_clip_norm = float(tc.grad_clip_norm)

        # 强度增强（复用 seg GPUAugmentor，仅强度分支；框不受影响）。
        self.augmentor = (
            GPUAugmentor(_intensity_only_augcfg(cfg.augment), max_scale=1.0)
            if cfg.augment.enabled else None)
        self.fold_2_5d = int(cfg.model.spatial_dims) == 2

        self.output_dir = Path(tc.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_key = det.save_best_metric        # map | loss
        self.best_sign = -1.0 if self.best_key == "loss" else 1.0
        self.best_metric = -math.inf
        self.best_epoch = 0
        self.start_epoch = 0
        self.history: "list[Dict[str, float]]" = []

    # ------------------------------------------------------------------
    def _to_device(self, batch, augment: bool = False):
        img = batch["image"].to(self.device, non_blocking=True).float()
        boxes = [b.to(self.device) for b in batch["boxes"]]
        labels = [l.to(self.device) for l in batch["labels"]]
        if augment and self.augmentor is not None:
            x = img if not self.fold_2_5d else img.unsqueeze(1)
            # 强度分支不使用 label（几何分支已清零）；zeros 仅占位。
            x, _, _ = self.augmentor(x, torch.zeros_like(x))
            img = x if not self.fold_2_5d else x.squeeze(1)
        return img, boxes, labels

    @torch.no_grad()
    def _global_grad_norm(self) -> float:
        """当前全局梯度 L2 范数（clip 关时供非有限守护；inf 上限不会裁剪）。"""
        return float(nn.utils.clip_grad_norm_(
            self.model.parameters(), float("inf")))

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        loss_meter = AverageMeter()
        part_meters: Dict[str, AverageMeter] = {}
        accum = self.grad_accum_steps
        total = len(self.train_loader)
        # GPU 预取（口径同 seg/cls Trainer）：独立 copy stream 提前一个
        # batch 上卡；boxes/labels 是 list 字段由预取器原样透传。
        batch_iter = self.train_loader
        if self.cfg.train.prefetch_to_gpu and self.device.type == "cuda":
            batch_iter = CudaPrefetcher(self.train_loader, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        group_has_nonfinite = False
        for step, batch in enumerate(batch_iter):
            img, boxes, labels = self._to_device(batch, augment=True)
            with torch.autocast(device_type=self.device.type,
                                enabled=self.use_amp, dtype=self.amp_dtype):
                losses = self.model(img, boxes, labels)
            loss = sum(losses.values())
            # 尾组 micro-batch 不满 accum 时用真实尾长作分母。
            group_start = (step // accum) * accum
            eff_accum = min(accum, total - group_start)
            loss_scaled = loss / eff_accum if eff_accum > 1 else loss
            self.scaler.scale(loss_scaled).backward()

            loss_val = float(loss.item())
            if math.isfinite(loss_val):
                loss_meter.update(loss_val, img.shape[0])
                for k, v in losses.items():
                    part_meters.setdefault(k, AverageMeter()).update(
                        float(v.item()), img.shape[0])
            else:
                group_has_nonfinite = True
                logger.warning(
                    "Non-finite train loss (%s) at epoch %d step %d/%d.",
                    loss_val, epoch + 1, step + 1, total)

            if (step + 1) % accum == 0 or (step + 1) == total:
                # 梯度范数：clip 开时复用其范数；bf16/fp32 无 GradScaler
                # 保护，clip 关时也算一次全局范数作非有限守护（同 seg）。
                grad_norm_val = None
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    gn = nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm)
                    if not self._scaler_active:
                        grad_norm_val = float(gn)
                elif not self._scaler_active:
                    grad_norm_val = self._global_grad_norm()
                grad_nonfinite = (grad_norm_val is not None
                                  and not math.isfinite(grad_norm_val))

                # bf16/fp32：loss/梯度非有限则丢弃本 accum 组，不推
                # scheduler/EMA，避免 NaN 永久污染权重；fp16 由 GradScaler
                # 内部跳过含 inf/NaN 梯度的优化步。
                if not self._scaler_active and (group_has_nonfinite
                                                or grad_nonfinite):
                    logger.warning(
                        "Skipping optimizer step at epoch %d step %d/%d: "
                        "non-finite %s in this accumulation group.",
                        epoch + 1, step + 1, total,
                        "loss" if group_has_nonfinite else "gradient")
                    self.optimizer.zero_grad(set_to_none=True)
                    group_has_nonfinite = False
                    continue
                group_has_nonfinite = False

                scale_before = (self.scaler.get_scale()
                                if self._scaler_active else None)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scaler_skipped = (scale_before is not None
                                  and self.scaler.get_scale() < scale_before)
                self.optimizer.zero_grad(set_to_none=True)
                # scheduler/EMA 仅在 optimizer 真正更新后推进（同 seg）。
                if not scaler_skipped:
                    self.scheduler.step()
                    if self.ema is not None:
                        self.ema.update(self.model)

            if (step + 1) % max(self.cfg.train.log_every, 1) == 0 or step == 0:
                logger.debug("  [%d/%d] loss=%.4f (%s) lr=%.2e",
                             step + 1, total, loss_val,
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
                # 金字塔只前向一次，predict 与 loss 共用；autocast 口径同
                # 训练/推理（head 内损失仍 fp32）。
                img_size = list(img.shape[2:])
                with torch.autocast(device_type=self.device.type,
                                    enabled=self.use_amp,
                                    dtype=self.amp_dtype):
                    feats = self.model.extract_pyramid(img)
                    dets = self.model.det_head.predict(feats, img_size)
                    losses = self.model.det_head.compute_loss(
                        feats, boxes, labels, img_size)
                loss_val = float(sum(losses.values()).item())
                if math.isfinite(loss_val):
                    loss_meter.update(loss_val, img.shape[0])
                else:
                    logger.warning("Non-finite val loss (%s) at epoch %d; "
                                   "excluded from val_loss.", loss_val,
                                   epoch + 1)
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
        """best checkpoint（口径同 seg：EMA 为主）：启用 EMA 时
        ``model_state_dict`` 存 EMA 权重（与选模/部署一致），在线权重另存
        ``model_online_state_dict``。"""
        bare = unwrap_compile(self.model)
        if self.ema is not None:
            online_sd = {k: v.detach().cpu().clone()
                         for k, v in bare.state_dict().items()}
            self.ema.apply_shadow(self.model)
            try:
                primary_sd = {k: v.detach().cpu().clone()
                              for k, v in bare.state_dict().items()}
            finally:
                self.ema.restore(self.model)
        else:
            online_sd = None
            primary_sd = bare.state_dict()
        state = {
            "epoch": epoch,
            "model_state_dict": primary_sd,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "metrics": metrics,
            "config": self.cfg,
            "det_config": self.det,
        }
        if online_sd is not None:
            state["model_online_state_dict"] = online_sd
        if self.ema is not None:
            state["ema_state_dict"] = self.ema.state_dict()
        torch.save(state, self.output_dir / "best_model.pth")
        logger.info("Best det model saved (%s=%.4f) @ epoch %d",
                    self.best_key, metrics[self.best_key], epoch + 1)

    def _save_latest(self, epoch: int, metrics: Dict[str, float]) -> None:
        """latest checkpoint（续训用，口径同 seg）：在线权重 + EMA +
        optimizer/scheduler/scaler/epoch/best 状态/history 全量落盘；先写
        临时文件再原子替换，防中断留半个 checkpoint。"""
        bare = unwrap_compile(self.model)
        state = {
            "epoch": epoch,
            "model_state_dict": bare.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "metrics": metrics,
            "history": self.history,
            "config": self.cfg,
            "det_config": self.det,
        }
        if self.ema is not None:
            state["ema_state_dict"] = self.ema.state_dict()
        tmp = self.output_dir / "latest_model.pth.tmp"
        torch.save(state, tmp)
        tmp.replace(self.output_dir / "latest_model.pth")

    def _try_resume(self) -> None:
        """``train.resume`` 指向 checkpoint 时完整恢复（model/EMA/optimizer/
        scheduler/scaler/epoch/best 状态/history）；路径不存在则报错，
        避免静默从头训。"""
        path = str(self.cfg.train.resume or "").strip()
        if not path:
            return
        if not Path(path).is_file():
            raise FileNotFoundError(f"train.resume checkpoint not found: "
                                    f"{path!r}")
        ckpt = torch.load(path, map_location=self.device,
                          weights_only=False)
        bare = unwrap_compile(self.model)
        bare.load_state_dict(ckpt["model_state_dict"])
        if self.ema is not None and "ema_state_dict" in ckpt:
            self.ema.load_state_dict(ckpt["ema_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.start_epoch = int(ckpt.get("epoch", -1)) + 1
        self.best_metric = float(ckpt.get("best_metric", -math.inf))
        self.best_epoch = int(ckpt.get("best_epoch", 0))
        self.history = list(ckpt.get("history", []))
        logger.info(
            "Resumed from %s: next epoch %d, best %s=%.4f @ epoch %d",
            path, self.start_epoch + 1, self.best_key,
            self.best_sign * self.best_metric, self.best_epoch + 1)

    def _write_history(self) -> None:
        (self.output_dir / "history.json").write_text(
            json.dumps(self.history, indent=2), encoding="utf-8")

    def fit(self) -> Dict[str, float]:
        self._try_resume()
        last: Dict[str, float] = dict(self.history[-1]) if self.history else {}
        patience = max(int(self.cfg.train.early_stopping), 0)
        epochs_no_improve = max(self.start_epoch - 1 - self.best_epoch, 0)
        for epoch in range(self.start_epoch, self.cfg.train.epochs):
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
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            last = {"epoch": float(epoch), **tr,
                    **{f"val_{k}": v for k, v in val.items()}}
            self.history.append(last)
            self._save_latest(epoch, val)
            self._write_history()
            if patience > 0 and epochs_no_improve >= patience:
                logger.info(
                    "Early stopping at epoch %d: no %s improvement for %d "
                    "validation(s) (best @ epoch %d).", epoch + 1,
                    self.best_key, patience, self.best_epoch + 1)
                break
        return {f"best_{self.best_key}": self.best_sign * self.best_metric,
                "best_epoch": float(self.best_epoch), **last}


__all__ = ["DetTrainer"]
