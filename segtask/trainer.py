"""Training pipeline for segmentation.

Handles the full training loop including:
- Mixed precision (AMP)
- EMA
- Learning rate scheduling with warmup
- Gradient clipping
- Validation and metric tracking
- Checkpointing (best + periodic)
- Early stopping
- Logging and visualization
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .config import Config
from .data.transforms import GPUAugmentor, MixupCutmix
from .losses.losses import build_loss
from .models.unet import UNet
from .visualization import visualize_batch
from .utils import (
    AverageMeter,
    ModelEMA,
    Timer,
    compute_dice_per_class,
    compute_metrics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------
def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    tc = cfg.train
    params = [p for p in model.parameters() if p.requires_grad]

    if tc.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=tc.lr, weight_decay=tc.weight_decay)
    elif tc.optimizer == "adam":
        return torch.optim.Adam(params, lr=tc.lr, weight_decay=tc.weight_decay)
    elif tc.optimizer == "sgd":
        return torch.optim.SGD(
            params, lr=tc.lr, weight_decay=tc.weight_decay,
            momentum=tc.momentum, nesterov=tc.nesterov,
        )
    else:
        raise ValueError(f"Unknown optimizer: {tc.optimizer}")


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------
def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: Config, steps_per_epoch: int
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    tc = cfg.train
    total_steps = tc.epochs * steps_per_epoch

    if tc.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=tc.cosine_min_lr,
        )
    elif tc.scheduler == "poly":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: (1 - step / max(total_steps, 1)) ** tc.poly_power,
        )
    elif tc.scheduler == "step":
        step_milestones = list(range(
            tc.step_size * steps_per_epoch,
            total_steps,
            tc.step_size * steps_per_epoch,
        ))
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=step_milestones, gamma=tc.step_gamma,
        )
    elif tc.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=tc.plateau_patience,
            factor=tc.plateau_factor,
        )
    elif tc.scheduler == "one_cycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=tc.lr, total_steps=total_steps,
            pct_start=tc.warmup_epochs / max(tc.epochs, 1),
        )
    else:
        raise ValueError(f"Unknown scheduler: {tc.scheduler}")


# ---------------------------------------------------------------------------
# Warmup wrapper
# ---------------------------------------------------------------------------
class WarmupScheduler:
    """Linear warmup wrapper around any LR scheduler.

    During warmup: LR linearly ramps from warmup_lr to base_lr.
    After warmup: delegates to the base scheduler.
    ReduceLROnPlateau is only stepped via step_epoch() with a metric.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        warmup_steps: int,
        warmup_lr: float,
        base_lr: float,
    ):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr
        self.base_lr = base_lr
        self.current_step = 0
        self._is_plateau = isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        )

        # Apply warmup_lr immediately so the first training step uses it
        if warmup_steps > 0:
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

    def step(self) -> None:
        """Per-iteration step. Do NOT pass metric here; use step_epoch() for plateau."""
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            alpha = self.current_step / max(self.warmup_steps, 1)
            lr = self.warmup_lr + alpha * (self.base_lr - self.warmup_lr)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        else:
            # ReduceLROnPlateau is stepped per-epoch via step_epoch()
            if self.scheduler is not None and not self._is_plateau:
                self.scheduler.step()

    def step_epoch(self, metric: Optional[float] = None) -> None:
        """Per-epoch step. Only needed for ReduceLROnPlateau."""
        if (
            self._is_plateau
            and self.scheduler is not None
            and self.current_step > self.warmup_steps
            and metric is not None
        ):
            self.scheduler.step(metric)

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    """Full training pipeline for segmentation."""

    def __init__(
        self, model: UNet, cfg: Config,
        train_loader: DataLoader, val_loader: DataLoader,
        device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        tc = cfg.train

        # Loss (wrap with deep supervision only when model uses it)
        from .losses.losses import DeepSupervisionLoss
        self.criterion = build_loss(cfg.loss)
        if cfg.model.deep_supervision and cfg.loss.deep_supervision_weights:
            self.criterion = DeepSupervisionLoss(
                self.criterion, cfg.loss.deep_supervision_weights)

        # Optimizer
        self.optimizer = build_optimizer(model, cfg)

        # Scheduler
        steps_per_epoch = len(train_loader)
        base_scheduler = build_scheduler(self.optimizer, cfg, steps_per_epoch)
        warmup_steps = tc.warmup_epochs * steps_per_epoch
        self.scheduler = WarmupScheduler(
            self.optimizer, base_scheduler,
            warmup_steps=warmup_steps, warmup_lr=tc.warmup_lr, base_lr=tc.lr)

        # AMP
        self.use_amp = tc.use_amp and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.amp_dtype = torch.float16 if tc.amp_dtype == "float16" else torch.bfloat16

        # EMA
        self.ema = ModelEMA(model, tc.ema_decay) if tc.use_ema else None

        # Output mode and class metadata
        self.total_slices = getattr(model, 'total_slices', 1)
        num_classes = cfg.data.num_classes
        self.semantic_classes = num_classes
        self.is_25d = (cfg.data.mode == "2.5d")
        self.output_mode = cfg.loss.output_mode
        self.num_fg_classes = getattr(model, 'num_fg_classes', num_classes - 1)

        # Augmentation (GPU)
        # For 2.5D: dataset returns 3D sub-volumes (1, D, H, W), so augment in 3D
        # to preserve spatial consistency across slices. Reshape to 2.5D AFTER aug.
        aug_spatial_dims = 3 if self.is_25d else cfg.model.spatial_dims
        self.augmentor = GPUAugmentor(cfg.augment, spatial_dims=aug_spatial_dims)
        self.mixup = MixupCutmix(alpha=cfg.augment.mixup_alpha, prob=cfg.augment.mixup_prob)

        # Tracking
        self.best_metric = -float("inf") if tc.save_best_mode == "max" else float("inf")
        self.best_epoch = 0
        self.start_epoch = 0
        self.patience_counter = 0

        # Output
        self.output_dir = Path(tc.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Resume
        if tc.resume and os.path.isfile(tc.resume):
            self._load_checkpoint(tc.resume)

    def fit(self) -> Dict[str, float]:
        """Run the full training loop.

        Returns:
            Dict of best validation metrics.
        """
        tc = self.cfg.train
        timer = Timer()

        logger.info("=" * 60)
        logger.info("Training started: %d epochs, device=%s", tc.epochs, self.device)
        logger.info("Model params: %.2fM", sum(p.numel() for p in self.model.parameters()) / 1e6)
        logger.info("Train batches: %d, Val batches: %d", len(self.train_loader), len(self.val_loader))
        logger.info("AMP=%s, EMA=%s (decay=%.4f)", self.use_amp, tc.use_ema, tc.ema_decay)
        logger.info("=" * 60)

        best_metrics = {}

        for epoch in range(self.start_epoch, tc.epochs):
            # Train one epoch
            train_metrics = self._train_epoch(epoch)

            # Validate
            val_metrics = {}
            if (epoch + 1) % tc.val_every == 0 or epoch == tc.epochs - 1:
                val_metrics = self._validate(epoch)

            # Per-epoch scheduler step (for ReduceLROnPlateau)
            plateau_metric = val_metrics.get(tc.save_best_metric.replace("val_", ""), None)
            self.scheduler.step_epoch(metric=plateau_metric)

            # Log
            lr = self.scheduler.get_lr()
            logger.info(
                "Epoch %d/%d | LR=%.2e | train_loss=%.4f | val_dice=%.4f | best=%.4f (ep%d) | %s",
                epoch + 1, tc.epochs, lr,
                train_metrics.get("loss", 0),
                val_metrics.get("mean_dice", 0),
                self.best_metric if self.best_metric > -float("inf") else 0,
                self.best_epoch + 1,
                timer.elapsed_str())

            # Checkpointing
            tracked = val_metrics.get(tc.save_best_metric.replace("val_", ""), 0)
            is_best = False
            if tracked > 0:
                if tc.save_best_mode == "max" and tracked > self.best_metric:
                    is_best = True
                elif tc.save_best_mode == "min" and tracked < self.best_metric:
                    is_best = True

            if is_best:
                self.best_metric = tracked
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
                best_metrics = val_metrics
                logger.info("★ New best: %s=%.4f at epoch %d", tc.save_best_metric, tracked, epoch + 1)
            else:
                self.patience_counter += 1

            if (epoch + 1) % tc.save_every == 0:
                self._save_checkpoint(epoch, is_best=False)

            # Visualization
            if tc.vis_every > 0 and (epoch + 1) % tc.vis_every == 0:
                try:
                    vis_batch = next(iter(self.val_loader))
                    if self.ema is not None:
                        self.ema.apply_shadow(self.model)
                    visualize_batch(
                        model=self.model,
                        batch=vis_batch,
                        epoch=epoch,
                        output_dir=str(self.output_dir),
                        device=self.device,
                        semantic_classes=self.semantic_classes,
                        total_slices=self.total_slices,
                        output_mode=self.output_mode,
                    )
                    if self.ema is not None:
                        self.ema.restore(self.model)
                except Exception as e:
                    logger.warning("Visualization failed: %s", e)

            # Early stopping
            if tc.early_stopping > 0 and self.patience_counter >= tc.early_stopping:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, tc.early_stopping)
                break

        logger.info("=" * 60)
        logger.info(
            "Training complete. Best %s=%.4f at epoch %d. Total time: %s",
            tc.save_best_metric, self.best_metric, self.best_epoch + 1, timer.elapsed_str(),
        )
        logger.info("=" * 60)

        return best_metrics

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        loss_meter = AverageMeter()
        dice_meter = AverageMeter()
        tc = self.cfg.train

        for step, batch in enumerate(self.train_loader):
            image = batch["image"].to(self.device, non_blocking=True)
            label = batch["label"].to(self.device, non_blocking=True)

            # GPU augmentation (3D for 2.5D/3D, 2D for 2D)
            image, label = self.augmentor(image, label)

            # 2.5D: after 3D augmentation, squeeze channel dim
            # image: (B, 1, D, H, W) → (B, D, H, W) where D=total_slices as input channels
            if self.is_25d:
                image = image.squeeze(1)

            image, label = self.mixup(image, label)

            # Forward
            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                pred = self.model(image)
                pred_loss, label_loss = self._reshape_for_loss(pred, label)
                loss = self.criterion(pred_loss, label_loss)

            # Backward
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()

            # Gradient clipping (unscale once before any clipping)
            if tc.grad_clip_norm > 0 or tc.grad_clip_value > 0:
                self.scaler.unscale_(self.optimizer)
                if tc.grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip_norm)
                if tc.grad_clip_value > 0:
                    nn.utils.clip_grad_value_(self.model.parameters(), tc.grad_clip_value)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Scheduler step (per-iteration for cosine/poly/one_cycle)
            self.scheduler.step()

            # EMA update
            if self.ema is not None:
                self.ema.update(self.model)

            # Metrics
            loss_meter.update(loss.item(), image.shape[0])

            # Compute dice occasionally for logging
            if (step + 1) % tc.log_every == 0 or step == 0:
                with torch.no_grad():
                    p = pred[0] if isinstance(pred, list) else pred
                    p_m, l_m = self._reshape_for_loss(p.detach(), label)
                    dice = compute_dice_per_class(p_m, l_m, output_mode=self.output_mode)
                    mean_dice = dice.mean().item()
                    dice_meter.update(mean_dice, image.shape[0])

                logger.debug(
                    "  [%d/%d] loss=%.4f dice=%.4f lr=%.2e",
                    step + 1, len(self.train_loader), loss.item(),
                    mean_dice, self.scheduler.get_lr(),
                )

        return {"loss": loss_meter.avg, "dice": dice_meter.avg}

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """Validate on the validation set."""
        # Use EMA weights for validation
        if self.ema is not None:
            self.ema.apply_shadow(self.model)

        self.model.eval()
        loss_meter = AverageMeter()
        all_dice = []

        for batch in self.val_loader:
            image = batch["image"].to(self.device, non_blocking=True)
            label = batch["label"].to(self.device, non_blocking=True)

            # 2.5D: squeeze channel dim before model input
            if self.is_25d:
                image = image.squeeze(1)

            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                pred = self.model(image)
                if isinstance(pred, list):
                    pred = pred[0]
                pred_loss, label_loss = self._reshape_for_loss(pred, label)
                loss = self.criterion(pred_loss, label_loss)

            loss_meter.update(loss.item(), image.shape[0])
            dice = compute_dice_per_class(pred_loss, label_loss, output_mode=self.output_mode)
            all_dice.append(dice.cpu())

        # Restore original weights
        if self.ema is not None:
            self.ema.restore(self.model)

        # Aggregate dice
        mean_dice_per_class = torch.stack(all_dice).mean(dim=0)
        metrics = {"val_loss": loss_meter.avg}
        nc = len(mean_dice_per_class)
        for c in range(nc):
            metrics[f"dice_class_{c}"] = mean_dice_per_class[c].item()

        if self.output_mode == "per_class":
            # All channels are foreground — mean of all
            metrics["mean_dice"] = mean_dice_per_class.mean().item()
        elif nc > 1:
            # Softmax mode — exclude background (class 0)
            metrics["mean_dice"] = mean_dice_per_class[1:].mean().item()
        else:
            metrics["mean_dice"] = mean_dice_per_class[0].item()

        logger.info(
            "  Val: loss=%.4f, mean_dice=%.4f, per_class=%s",
            metrics["val_loss"],
            metrics["mean_dice"],
            [f"{d:.4f}" for d in mean_dice_per_class.tolist()],
        )

        return metrics

    def _reshape_for_loss(
        self, pred: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare pred and label for per-class loss in 2.5D mode.

        For 2.5D:
          pred:   (B, N*C, H, W) → reshape → (B, N, C, H, W)
                  softmax over C (dim=2): each class's C slice predictions → probability
                  then transpose → (B*C, N, H, W) for per-class Dice
          label:  (B, N, D, H, W) → center slice → (B, N, 1, H, W)
                  expand → (B*C, N, H, W) to match pred

        The Dice Loss then computes per-class Dice independently (softmax over dim=1).
        """
        if not self.is_25d:
            return pred, label

        C = self.total_slices
        N = self.semantic_classes

        # pred: (B, N*C, H, W) → (B, N, C, H, W)
        pred = pred.view(-1, N, C, pred.shape[2], pred.shape[3])
        # softmax over slice dim per class
        pred = torch.softmax(pred, dim=2)  # (B, N, C, H, W)
        # transpose: (B, N, C, H, W) → (B, C, N, H, W) → (B*C, N, H, W)
        pred = pred.permute(0, 2, 1, 3, 4).contiguous()
        pred = pred.view(-1, N, pred.shape[3], pred.shape[4])

        # label: (B, N, D, H, W) → center slice → (B, N, 1, H, W)
        D = label.shape[2]
        center_d = D // 2
        label_center = label[:, :, center_d:center_d + 1].contiguous()  # (B, N, 1, H, W)
        # Expand to match pred's C dim: (B, N, 1, H, W) → (B, N, C, H, W)
        label_expanded = label_center.expand(-1, -1, C, -1, -1)
        # Same transpose/reshape: (B, N, C, H, W) → (B*C, N, H, W)
        label_expanded = label_expanded.permute(0, 2, 1, 3, 4).contiguous()
        label_expanded = label_expanded.view(-1, N, label_expanded.shape[3], label_expanded.shape[4])

        # Support deep supervision
        if isinstance(pred, list):
            return pred, [label_expanded] * len(pred)

        return pred, label_expanded

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "config": self.cfg,
        }
        if self.ema is not None:
            state["ema_state_dict"] = self.ema.state_dict()

        if is_best:
            # Save EMA weights as the best model
            if self.ema is not None:
                self.ema.apply_shadow(self.model)
                state["model_state_dict"] = self.model.state_dict()
                self.ema.restore(self.model)

            path = self.output_dir / "best_model.pth"
            torch.save(state, path)
            logger.info("Best model saved: %s", path)
        else:
            path = self.output_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save(state, path)
            logger.debug("Checkpoint saved: %s", path)

    def _load_checkpoint(self, path: str) -> None:
        """Load checkpoint and resume training."""
        logger.info("Loading checkpoint: %s", path)
        ckpt = torch.load(path, map_location=self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_metric = ckpt.get("best_metric", self.best_metric)
        self.best_epoch = ckpt.get("best_epoch", 0)

        if self.ema is not None and "ema_state_dict" in ckpt:
            self.ema.load_state_dict(ckpt["ema_state_dict"])

        logger.info(
            "Resumed from epoch %d, best=%s=%.4f",
            self.start_epoch, self.cfg.train.save_best_metric, self.best_metric,
        )
