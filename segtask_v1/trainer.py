"""Training pipeline for 3D segmentation.

Handles:
- Mixed precision (AMP)
- EMA (exponential moving average)
- Learning rate scheduling with warmup
- Gradient clipping + gradient accumulation
- torch.compile acceleration
- Validation and per-class Dice tracking
- Checkpointing (best + periodic)
- Early stopping
- GPU data augmentation
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .config import Config
from .data.augment import GPUAugmentor
from .losses.losses import build_loss, DeepSupervisionLoss
from .models.unet import UNet3D
from .utils import AverageMeter, ModelEMA, Timer, compute_dice_per_class

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
    raise ValueError(f"Unknown optimizer: {tc.optimizer}")


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------
def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    tc = cfg.train
    total_steps = tc.epochs * steps_per_epoch

    if tc.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=tc.cosine_min_lr)
    elif tc.scheduler == "poly":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: (1 - step / max(total_steps, 1)) ** tc.poly_power)
    elif tc.scheduler == "step":
        milestones = list(range(
            tc.step_size * steps_per_epoch, total_steps,
            tc.step_size * steps_per_epoch))
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=tc.step_gamma)
    elif tc.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=tc.plateau_patience,
            factor=tc.plateau_factor)
    elif tc.scheduler == "cosine_warm_restarts":
        T_0 = tc.cosine_restart_period * steps_per_epoch
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(T_0, 1), T_mult=tc.cosine_restart_mult,
            eta_min=tc.cosine_min_lr)
    elif tc.scheduler == "one_cycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=tc.lr, total_steps=total_steps,
            pct_start=tc.warmup_epochs / max(tc.epochs, 1))
    raise ValueError(f"Unknown scheduler: {tc.scheduler}")


# ---------------------------------------------------------------------------
# Warmup wrapper
# ---------------------------------------------------------------------------
class WarmupScheduler:
    """Linear warmup then delegates to base scheduler.

    During warmup: LR ramps linearly from warmup_lr to base_lr.
    After warmup: base scheduler controls LR.
    ReduceLROnPlateau is stepped per-epoch via step_epoch().
    """

    def __init__(self, optimizer, scheduler, warmup_steps: int,
                 warmup_lr: float, base_lr: float):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr
        self.base_lr = base_lr
        self.current_step = 0
        self._is_plateau = isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        if warmup_steps > 0:
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

    def step(self) -> None:
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            alpha = self.current_step / max(self.warmup_steps, 1)
            lr = self.warmup_lr + alpha * (self.base_lr - self.warmup_lr)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        elif self.scheduler is not None and not self._is_plateau:
            self.scheduler.step()

    def step_epoch(self, metric: Optional[float] = None) -> None:
        if (self._is_plateau and self.scheduler is not None
                and self.current_step > self.warmup_steps and metric is not None):
            self.scheduler.step(metric)

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    """Full training pipeline for 3D segmentation."""

    def __init__(
        self,
        model: UNet3D,
        cfg: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        tc = cfg.train

        # torch.compile (PyTorch 2.0+)
        if tc.compile_mode != "none" and hasattr(torch, "compile"):
            logger.info("Compiling model with mode='%s'", tc.compile_mode)
            self.model = torch.compile(self.model, mode=tc.compile_mode)

        # Loss  # TODO 这里似乎是支持对不同类别的损失赋予不同的损失权重，但是无法对特定区域赋予不同的权重，例如B,C,D,H,W的标签体积中有1，2，3，4标签值，我想对值为1和2区域赋予更大的权重。也就是在像素点层面赋权重。
        self.criterion = build_loss(cfg.loss)
        if cfg.model.deep_supervision and cfg.loss.deep_supervision_weights:
            self.criterion = DeepSupervisionLoss(
                self.criterion, cfg.loss.deep_supervision_weights)

        # Optimizer + scheduler
        self.optimizer  = build_optimizer(model, cfg)
        steps_per_epoch = len(train_loader)
        base_scheduler  = build_scheduler(self.optimizer, cfg, steps_per_epoch)
        warmup_steps = tc.warmup_epochs * steps_per_epoch
        self.scheduler = WarmupScheduler(
            self.optimizer, base_scheduler,
            warmup_steps=warmup_steps, warmup_lr=tc.warmup_lr, base_lr=tc.lr)

        # AMP
        self.use_amp   = tc.use_amp and device.type == "cuda"
        self.scaler    = GradScaler(enabled=self.use_amp)
        self.amp_dtype = torch.float16 if tc.amp_dtype == "float16" else torch.bfloat16

        # EMA
        self.ema = ModelEMA(model, tc.ema_decay) if tc.use_ema else None

        # GPU augmentation  TODO 丰富了数据增强方法，需要验证
        self.augmentor = GPUAugmentor(cfg.augment)

        # Gradient accumulation
        self.grad_accum_steps = max(tc.grad_accum_steps, 1)

        # Tracking
        self.num_fg = cfg.num_fg_classes
        self.best_metric = -float("inf") if tc.save_best_mode == "max" else float("inf")
        self.best_epoch = 0
        self.start_epoch = 0
        self.patience_counter = 0

        # Output directory
        self.output_dir = Path(tc.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Resume
        if tc.resume and os.path.isfile(tc.resume):
            self._load_checkpoint(tc.resume)

    def fit(self) -> Dict[str, float]:
        """Run the full training loop. Returns best validation metrics."""
        tc = self.cfg.train
        timer = Timer()

        logger.info("=" * 60)
        logger.info("Training: %d epochs, device=%s", tc.epochs, self.device)
        logger.info("Model params: %.2fM",
                     sum(p.numel() for p in self.model.parameters()) / 1e6)
        logger.info("Train batches: %d, Val batches: %d",
                     len(self.train_loader), len(self.val_loader))
        logger.info("AMP=%s, EMA=%s (decay=%.4f)", self.use_amp, tc.use_ema, tc.ema_decay)
        logger.info("Grad accum=%d, Effective batch=%d",
                     self.grad_accum_steps, self.cfg.data.batch_size * self.grad_accum_steps)
        logger.info("Foreground classes: %d, Loss: %s", self.num_fg, self.cfg.loss.name)
        if tc.compile_mode != "none":
            logger.info("torch.compile mode: %s", tc.compile_mode)
        logger.info("=" * 60)

        best_metrics = {}

        for epoch in range(self.start_epoch, tc.epochs):
            train_metrics = self._train_epoch(epoch)

            val_metrics = {}
            if (epoch + 1) % tc.val_every == 0 or epoch == tc.epochs - 1:
                val_metrics = self._validate(epoch)

            # Plateau scheduler step
            plateau_metric = val_metrics.get(tc.save_best_metric, None)
            self.scheduler.step_epoch(metric=plateau_metric)

            # Logging
            lr = self.scheduler.get_lr()
            logger.info(
                "Epoch %d/%d | LR=%.2e | loss=%.4f | val_dice=%.4f | "
                "best=%.4f (ep%d) | %s",
                epoch + 1, tc.epochs, lr,
                train_metrics.get("loss", 0),
                val_metrics.get("mean_dice", 0),
                self.best_metric if self.best_metric > -float("inf") else 0,
                self.best_epoch + 1,
                timer.elapsed_str())

            # Checkpointing
            tracked = val_metrics.get(tc.save_best_metric, 0)
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
                logger.info("★ New best: %s=%.4f at epoch %d",
                            tc.save_best_metric, tracked, epoch + 1)
            else:
                self.patience_counter += 1

            if (epoch + 1) % tc.save_every == 0:
                self._save_checkpoint(epoch, is_best=False)

            # Early stopping
            if tc.early_stopping > 0 and self.patience_counter >= tc.early_stopping:
                logger.info("Early stopping at epoch %d (patience=%d)",
                            epoch + 1, tc.early_stopping)
                break

        logger.info("=" * 60)
        logger.info("Training complete. Best %s=%.4f at epoch %d. Time: %s",
                     tc.save_best_metric, self.best_metric,
                     self.best_epoch + 1, timer.elapsed_str())
        logger.info("=" * 60)
        return best_metrics

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with optional gradient accumulation."""
        self.model.train()
        loss_meter = AverageMeter()
        dice_meter = AverageMeter()
        tc = self.cfg.train
        accum = self.grad_accum_steps

        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(self.train_loader):
            image = batch["image"].to(self.device, non_blocking=True)
            label = batch["label"].to(self.device, non_blocking=True)
            wmap = batch.get("weight_map")
            if wmap is not None:
                wmap = wmap.to(self.device, non_blocking=True)

            # GPU augmentation — cat weight_map as extra label channel so
            # spatial transforms are applied consistently, then split back.
            if wmap is not None:
                label_aug = torch.cat([label, wmap], dim=1)  # (B, C+1, D, H, W)
                image, label_aug = self.augmentor(image, label_aug)
                label = label_aug[:, :-1]
                wmap = label_aug[:, -1:]
            else:
                image, label = self.augmentor(image, label)

            # Forward
            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                pred = self.model(image)
                loss = self.criterion(pred, label, weight_map=wmap)
                # Scale loss by accumulation steps for correct gradient magnitude
                if accum > 1:
                    loss = loss / accum

            # Backward (accumulate gradients)
            self.scaler.scale(loss).backward()

            # Step optimizer every accum steps or at end of epoch
            if (step + 1) % accum == 0 or (step + 1) == len(self.train_loader):
                # Gradient clipping
                if tc.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                # Scheduler step (per optimizer step)
                self.scheduler.step()

                # EMA update (per optimizer step)
                if self.ema is not None:
                    self.ema.update(self.model)

            # Metrics (use unscaled loss for logging)
            loss_val = loss.item() * accum if accum > 1 else loss.item()
            loss_meter.update(loss_val, image.shape[0])

            if (step + 1) % tc.log_every == 0 or step == 0:
                with torch.no_grad():
                    p = pred[0] if isinstance(pred, list) else pred
                    dice = compute_dice_per_class(p.detach(), label)
                    mean_dice = dice.mean().item()
                    dice_meter.update(mean_dice, image.shape[0])
                logger.debug("  [%d/%d] loss=%.4f dice=%.4f lr=%.2e",
                             step + 1, len(self.train_loader),
                             loss.item(), mean_dice, self.scheduler.get_lr())

        return {"loss": loss_meter.avg, "dice": dice_meter.avg}

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """Validate on the validation set."""
        if self.ema is not None:
            self.ema.apply_shadow(self.model)

        self.model.eval()
        loss_meter = AverageMeter()
        all_dice = []

        for batch in self.val_loader:
            image = batch["image"].to(self.device, non_blocking=True)
            label = batch["label"].to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                pred = self.model(image)
                if isinstance(pred, list):
                    pred = pred[0]
                loss = self.criterion(pred, label)

            loss_meter.update(loss.item(), image.shape[0])
            dice = compute_dice_per_class(pred, label)
            all_dice.append(dice.cpu())

        if self.ema is not None:
            self.ema.restore(self.model)

        # Aggregate
        mean_dice_per_class = torch.stack(all_dice).mean(dim=0)
        metrics = {"val_loss": loss_meter.avg}
        for c in range(len(mean_dice_per_class)):
            metrics[f"dice_class_{c}"] = mean_dice_per_class[c].item()
        metrics["mean_dice"] = mean_dice_per_class.mean().item()

        logger.info("  Val: loss=%.4f, mean_dice=%.4f, per_class=%s",
                     metrics["val_loss"], metrics["mean_dice"],
                     [f"{d:.4f}" for d in mean_dice_per_class.tolist()])
        return metrics

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
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
        logger.info("Loading checkpoint: %s", path)
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_metric = ckpt.get("best_metric", self.best_metric)
        self.best_epoch = ckpt.get("best_epoch", 0)
        if self.ema is not None and "ema_state_dict" in ckpt:
            self.ema.load_state_dict(ckpt["ema_state_dict"])
        logger.info("Resumed from epoch %d, best=%s=%.4f",
                     self.start_epoch, self.cfg.train.save_best_metric, self.best_metric)
