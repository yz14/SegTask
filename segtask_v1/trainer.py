"""Training pipeline for 3D segmentation.

Handles:
- Mixed precision (AMP, fp16 + bf16) with scaler disabled in bf16
- EMA with context-manager-based swap (exception-safe)
- Learning rate scheduling with warmup (step-aligned with base scheduler)
- Gradient clipping + gradient accumulation (partial-tail corrected)
- torch.compile acceleration (state_dict unwrapping on save / load)
- Validation and per-class Dice tracking (DS-safe loss path)
- Full-state checkpointing (model/ema/optimizer/scheduler/scaler/early-stop)
- Early stopping
- GPU data augmentation
"""

from __future__ import annotations

import logging
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn
# GradScaler moved from torch.cuda.amp to torch.amp in PyTorch ≥ 2.3; fall
# back to the CUDA-namespace import on older builds (e.g. the 2.2 shipped
# with the py310 test env).
try:
    from torch.amp import GradScaler, autocast  # type: ignore
except ImportError:  # pragma: no cover - version-dependent
    from torch.cuda.amp import GradScaler  # type: ignore
    from torch.amp import autocast  # type: ignore
from torch.utils.data import DataLoader

from .config import Config
from .data.augment import GPUAugmentor
from .losses.losses import (
    build_loss, DeepSupervisionLoss, MultiResolutionLoss, SliceChannelLoss)
from .models.unet import UNet3D
from .utils import (
    AverageMeter, ModelEMA, Timer,
    compute_dice_per_class, dice_batch_stats,
)

logger = logging.getLogger(__name__)


_AMP_DTYPES = {
    "float16": torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _unwrap_compile(m: nn.Module) -> nn.Module:
    """Strip the `_orig_mod` wrapper added by `torch.compile` so state_dict
    keys don't get a `_orig_mod.` prefix that breaks reloading into an
    uncompiled model."""
    return getattr(m, "_orig_mod", m)


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
def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    steps_per_epoch: int,
    post_warmup_steps: int,
):
    """Build the base LR scheduler that runs AFTER warmup.

    `post_warmup_steps` is the number of optimizer steps the base scheduler
    will actually see, so `T_max` / poly's horizon / step milestones are
    aligned with the warmup-excluded segment of training.
    """
    tc = cfg.train
    horizon = max(post_warmup_steps, 1)

    if tc.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=horizon, eta_min=tc.cosine_min_lr)
    elif tc.scheduler == "poly":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: (1 - step / horizon) ** tc.poly_power)
    elif tc.scheduler == "step":
        milestones = list(range(
            tc.step_size * steps_per_epoch, horizon,
            tc.step_size * steps_per_epoch))
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=tc.step_gamma)
    elif tc.scheduler == "plateau":
        # Plateau direction must match the best-metric direction so LR
        # reduction fires on stagnation of the ACTUAL optimization target
        # (previously hardcoded "max", which silently minimized loss-style
        # metrics).
        plateau_mode = tc.save_best_mode if tc.save_best_mode in ("max", "min") else "max"
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=plateau_mode, patience=tc.plateau_patience,
            factor=tc.plateau_factor)
    elif tc.scheduler == "cosine_warm_restarts":
        T_0 = tc.cosine_restart_period * steps_per_epoch
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(T_0, 1), T_mult=tc.cosine_restart_mult,
            eta_min=tc.cosine_min_lr)
    elif tc.scheduler == "one_cycle":
        # OneCycleLR manages its own rising segment via `pct_start`; stacking
        # WarmupScheduler on top is rejected in Trainer.__init__.
        total_steps = tc.epochs * steps_per_epoch
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=tc.lr, total_steps=total_steps,
            pct_start=max(tc.warmup_epochs, 1) / max(tc.epochs, 1))
    raise ValueError(f"Unknown scheduler: {tc.scheduler}")


# ---------------------------------------------------------------------------
# Warmup wrapper
# ---------------------------------------------------------------------------
class WarmupScheduler:
    """Linear warmup, then delegate to a base scheduler.

    During warmup: LR ramps linearly from `warmup_lr` to `base_lr` over
    `warmup_steps` optimizer steps. The base scheduler is NOT stepped here.
    After warmup: the base scheduler drives LR. `ReduceLROnPlateau` is the
    only base scheduler stepped per epoch (via `step_epoch`); all others are
    stepped per optimizer step.

    Because warmup consumes `warmup_steps`, the base scheduler's horizon
    must be built with `post_warmup_steps = total_steps - warmup_steps`
    (see `build_scheduler`), otherwise cosine / poly / step never reach
    their full schedules.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler,
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
        if (self._is_plateau
                and self.scheduler is not None
                and self.current_step > self.warmup_steps
                and metric is not None):
            self.scheduler.step(metric)

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self) -> Dict:
        # Persist the defining warmup parameters too, so `load_state_dict`
        # can detect accidental config changes (e.g. `warmup_epochs` edited
        # before resume) that would otherwise silently mis-align the LR
        # schedule. The base scheduler's own state is kept unchanged.
        return {
            "current_step": self.current_step,
            "warmup_steps": self.warmup_steps,
            "warmup_lr": self.warmup_lr,
            "base_lr": self.base_lr,
            "base_scheduler": (self.scheduler.state_dict()
                               if self.scheduler is not None else None),
        }

    def load_state_dict(self, state: Dict) -> None:
        ckpt_warmup_steps = state.get("warmup_steps")
        ckpt_warmup_lr = state.get("warmup_lr")
        ckpt_base_lr = state.get("base_lr")
        # Warn loudly on config drift across resume. Mismatching warmup
        # config would slot `current_step` into a different schedule shape.
        mismatches = []
        if (ckpt_warmup_steps is not None
                and int(ckpt_warmup_steps) != int(self.warmup_steps)):
            mismatches.append(
                f"warmup_steps: ckpt={ckpt_warmup_steps} vs cfg={self.warmup_steps}")
        if ckpt_warmup_lr is not None and float(ckpt_warmup_lr) != float(self.warmup_lr):
            mismatches.append(
                f"warmup_lr: ckpt={ckpt_warmup_lr} vs cfg={self.warmup_lr}")
        if ckpt_base_lr is not None and float(ckpt_base_lr) != float(self.base_lr):
            mismatches.append(
                f"base_lr: ckpt={ckpt_base_lr} vs cfg={self.base_lr}")
        if mismatches:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "Warmup config drift on resume (%s). `current_step` will be "
                "restored but the schedule shape differs; LR trajectory "
                "may not match the original run.", "; ".join(mismatches))

        self.current_step = int(state.get("current_step", 0))
        base_state = state.get("base_scheduler", None)
        if base_state is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(base_state)


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
        device: torch.device,
    ):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        tc = cfg.train

        # --- Device placement FIRST. Optimizer/EMA must bind to the already
        #     placed parameters; `torch.compile` is applied LAST so the only
        #     part that needs to know about the wrapper is state_dict I/O.
        self.model = model.to(device)

        # --- Loss ------------------------------------------------------
        # `base_loss` is kept separately for validation. The training-time
        # criterion wraps it in DeepSupervisionLoss / MultiResolutionLoss,
        # which assume list-of-tensors pred and multi-resolution label
        # stacks. Validation collapses both down to 1x and calls `base_loss`
        # directly to avoid a shape-contract mismatch.
        self.base_loss = build_loss(cfg.loss)
        self.is_2_5d = cfg.data.patch_mode == "2_5d"

        # Composition order matters:
        #   INNER — wraps `base_loss` for the patch-mode contract:
        #     - 3D modes: ``MultiResolutionLoss`` splits pred channels by
        #       resolution scale (C_res). Pred:  (B, num_fg*C_res, ...).
        #     - 2.5D mode: ``SliceChannelLoss`` splits pred channels by
        #       foreground class (D slices per class). Pred is rank-4:
        #       (B, num_fg*D, H, W); label is (B, D, H, W) raw.
        #   OUTER = DeepSupervisionLoss(INNER) — iterates over the list of
        #     per-decoder-level tensors, downsamples label+weight_map to
        #     each, and delegates to INNER. DS uses nearest interpolation
        #     in spatial dims of pred, so it works for both 3D and 2D paths.
        if self.is_2_5d:
            num_slices = int(cfg.data.patch_size[0])
            inner = SliceChannelLoss(
                base_loss=self.base_loss,
                num_fg_classes=cfg.num_fg_classes,
                num_slices=num_slices,
                label_values=cfg.data.label_values,
            )
            num_res = 1   # for logging only; SliceChannelLoss has C_res==1
            logger.info(
                "Loss: %s [2.5D], num_slices=%d, fg_classes=%d",
                cfg.loss.name, num_slices, cfg.num_fg_classes)
        else:
            num_res = len(cfg.data.multi_res_scales)
            inner = MultiResolutionLoss(
                base_loss=self.base_loss,
                num_fg_classes=cfg.num_fg_classes,
                num_res=num_res,
                label_values=cfg.data.label_values,
            )
            logger.info(
                "Loss: %s, scales=%d, fg_classes=%d",
                cfg.loss.name, num_res, cfg.num_fg_classes)

        if cfg.model.deep_supervision and cfg.loss.deep_supervision_weights:
            self.criterion = DeepSupervisionLoss(
                inner, cfg.loss.deep_supervision_weights)
        else:
            self.criterion = inner
        # Keep a handle to the INNER wrapper for unified metric reshaping.
        # Both wrappers expose ``split_for_metrics(pred, label_raw) ->
        # (pred_per_class, target_binary)`` so trainer code is mode-agnostic.
        self._inner_loss = inner

        # --- Optimizer + scheduler ------------------------------------
        self.optimizer = build_optimizer(self.model, cfg)
        steps_per_epoch = len(train_loader)
        warmup_steps = tc.warmup_epochs * steps_per_epoch
        total_steps = tc.epochs * steps_per_epoch
        post_warmup = total_steps - warmup_steps

        # OneCycleLR carries its own rising segment via pct_start; stacking
        # WarmupScheduler on top produces a double warmup and mis-aligned
        # total_steps. Refuse this combination explicitly.
        if tc.scheduler == "one_cycle" and warmup_steps > 0:
            raise ValueError(
                "OneCycleLR has built-in warmup (pct_start). "
                "Set train.warmup_epochs=0 when using scheduler='one_cycle'.")

        base_scheduler = build_scheduler(
            self.optimizer, cfg, steps_per_epoch,
            post_warmup_steps=post_warmup)
        self.scheduler = WarmupScheduler(
            self.optimizer, base_scheduler,
            warmup_steps=warmup_steps,
            warmup_lr=tc.warmup_lr, base_lr=tc.lr)

        # --- AMP -------------------------------------------------------
        if tc.amp_dtype not in _AMP_DTYPES:
            raise ValueError(
                f"Unknown amp_dtype: {tc.amp_dtype!r}. "
                f"Expected one of {sorted(_AMP_DTYPES)}.")
        self.amp_dtype = _AMP_DTYPES[tc.amp_dtype]
        self.use_amp = tc.use_amp and device.type == "cuda"
        # GradScaler is only meaningful for fp16; bf16 has fp32-range
        # mantissa-clipped values and does not require loss scaling. Leaving
        # the scaler disabled skips a redundant unscale pass.
        self._scaler_active = self.use_amp and self.amp_dtype == torch.float16
        self.scaler = GradScaler("cuda", enabled=self._scaler_active)

        # --- EMA (bind to placed, not-yet-compiled model) -------------
        self.ema = ModelEMA(self.model, tc.ema_decay) if tc.use_ema else None

        # --- torch.compile (last) -------------------------------------
        if tc.compile_mode != "none" and hasattr(torch, "compile"):
            logger.info("Compiling model with mode='%s'", tc.compile_mode)
            self.model = torch.compile(self.model, mode=tc.compile_mode)

        # --- Augmentation ---------------------------------------------
        # The augmentor applies spatial transforms jointly to image and
        # label tensors so alignment holds. When a weight_map is present
        # it is concatenated onto the label along dim=1 and thus inherits
        # the label-path interpolation (nearest-neighbour). This is the
        # correct behaviour for segmentation masks; if weight maps ever
        # need continuous-value resampling, the augmentor must be extended
        # to accept per-channel interpolation modes.
        # Pass the largest multi-res scale so the augmentor can keep elastic
        # deformation physically conservative (BUG-E). For single-resolution
        # inputs (multi_res_scales == [1.0] or empty), max_scale==1.0 and the
        # augmentor is bit-identical to the previous behaviour.
        _scales = cfg.data.multi_res_scales or [1.0]
        self.augmentor = GPUAugmentor(cfg.augment, max_scale=max(_scales))

        # --- Cropping (oversampled patches) ---------------------------
        # Both `z_axis` and `cubic` patch modes now honour
        # `aug_oversample_ratio` (BUG-B): the dataset emits an oversized
        # patch, the augmentor applies spatial transforms (whose
        # `padding_mode="zeros"` at rotated corners would otherwise leak
        # into the effective field-of-view), and we center-crop back to
        # `patch_size` here after augmentation.
        self.target_patch_size = tuple(cfg.data.patch_size)  # (D, H, W)
        self.needs_crop = cfg.data.aug_oversample_ratio > 1.0

        # --- Gradient accumulation ------------------------------------
        self.grad_accum_steps = max(tc.grad_accum_steps, 1)

        # --- Tracking --------------------------------------------------
        self.num_fg = cfg.num_fg_classes
        self._best_mode = tc.save_best_mode  # "max" or "min"
        self.best_metric: float = (
            -math.inf if self._best_mode == "max" else math.inf)
        self.has_best = False
        self.best_epoch = 0
        self.start_epoch = 0
        self.patience_counter = 0

        # --- Output directory -----------------------------------------
        self.output_dir = Path(tc.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- Resume ----------------------------------------------------
        if tc.resume and os.path.isfile(tc.resume):
            self._load_checkpoint(tc.resume)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self) -> Dict[str, float]:
        """Run the full training loop. Returns best validation metrics."""
        tc = self.cfg.train
        timer = Timer()

        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info("=" * 60)
        logger.info("Training: %d epochs, device=%s", tc.epochs, self.device)
        logger.info("Model params: %.2fM", total_params)
        logger.info("Train batches: %d, Val batches: %d",
                    len(self.train_loader), len(self.val_loader))
        logger.info("AMP=%s (dtype=%s, scaler=%s), EMA=%s (decay=%.4f)",
                    self.use_amp, tc.amp_dtype, self._scaler_active,
                    tc.use_ema, tc.ema_decay)
        logger.info("Grad accum=%d, Effective batch=%d",
                    self.grad_accum_steps,
                    self.cfg.data.batch_size * self.grad_accum_steps)
        logger.info("Foreground classes: %d, Loss: %s",
                    self.num_fg, self.cfg.loss.name)
        if tc.compile_mode != "none":
            logger.info("torch.compile mode: %s", tc.compile_mode)
        logger.info("=" * 60)

        best_metrics: Dict[str, float] = {}

        for epoch in range(self.start_epoch, tc.epochs):
            train_metrics = self._train_epoch(epoch)

            val_metrics: Dict[str, float] = {}
            if (epoch + 1) % tc.val_every == 0 or epoch == tc.epochs - 1:
                val_metrics = self._validate(epoch)

            # Plateau is the only base scheduler driven per-epoch.
            plateau_metric = val_metrics.get(tc.save_best_metric, None)
            self.scheduler.step_epoch(metric=plateau_metric)

            # --- Best-checkpoint decision (no magic >0 guard) ----------
            is_best = False
            if tc.save_best_metric in val_metrics:
                tracked = val_metrics[tc.save_best_metric]
                if not self.has_best:
                    is_best = True
                elif self._best_mode == "max":
                    is_best = tracked > self.best_metric
                else:
                    is_best = tracked < self.best_metric

                if is_best:
                    self.best_metric = tracked
                    self.best_epoch = epoch
                    self.has_best = True
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, is_best=True)
                    best_metrics = val_metrics
                    logger.info("★ New best: %s=%.4f at epoch %d",
                                tc.save_best_metric, tracked, epoch + 1)
                else:
                    self.patience_counter += 1

            # --- Epoch summary ----------------------------------------
            best_str = (f"{self.best_metric:.4f} (ep{self.best_epoch + 1})"
                        if self.has_best else "n/a")
            logger.info(
                "Epoch %d/%d | LR=%.2e | loss=%.4f | val_dice=%.4f | "
                "best=%s | %s",
                epoch + 1, tc.epochs, self.scheduler.get_lr(),
                train_metrics.get("loss", 0.0),
                val_metrics.get("mean_dice", 0.0),
                best_str,
                timer.elapsed_str(),
            )

            # --- Periodic checkpoint ----------------------------------
            if (epoch + 1) % tc.save_every == 0:
                self._save_checkpoint(epoch, is_best=False)

            # --- Early stopping ---------------------------------------
            if tc.early_stopping > 0 and self.patience_counter >= tc.early_stopping:
                logger.info("Early stopping at epoch %d (patience=%d)",
                            epoch + 1, tc.early_stopping)
                break

        logger.info("=" * 60)
        if self.has_best:
            logger.info(
                "Training complete. Best %s=%.4f at epoch %d. Time: %s",
                tc.save_best_metric, self.best_metric,
                self.best_epoch + 1, timer.elapsed_str())
        else:
            logger.info("Training complete. No validation best recorded. "
                        "Time: %s", timer.elapsed_str())
        logger.info("=" * 60)
        return best_metrics

    # ------------------------------------------------------------------
    # EMA swap helper (exception-safe)
    # ------------------------------------------------------------------
    @contextmanager
    def _ema_swapped(self) -> Iterator[None]:
        """Temporarily swap EMA weights into the model. `try/finally`
        guarantees the online weights are restored even if the enclosed
        block raises — without this, an OOM during validation would leave
        the trainer running on EMA weights for the rest of training."""
        if self.ema is None:
            yield
            return
        self.ema.apply_shadow(self.model)
        try:
            yield
        finally:
            self.ema.restore(self.model)

    # ------------------------------------------------------------------
    # Training / validation loops
    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with optional gradient accumulation."""
        self.model.train()
        loss_meter = AverageMeter()
        dice_meter = AverageMeter()
        tc = self.cfg.train
        accum = self.grad_accum_steps

        total_steps = len(self.train_loader)
        # Any steps beyond `partial_start` belong to a partial accumulation
        # tail (len(loader) not divisible by accum). Divide those by the
        # real tail length so the effective LR doesn't shrink on them.
        remainder = total_steps % accum if accum > 1 else 0
        partial_start = total_steps - remainder

        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(self.train_loader):
            image = batch["image"].to(self.device, non_blocking=True)
            label = batch["label"].to(self.device, non_blocking=True)
            wmap = batch.get("weight_map")
            if wmap is not None:
                wmap = wmap.to(self.device, non_blocking=True)
                if wmap.numel() == 0 or wmap.shape[1] == 0:
                    wmap = None  # treat empty collation sentinels as absent

            # --- GPU augmentation: image + (label [+ weight_map]) share
            #     one sampled transform so spatial alignment holds.
            if wmap is not None:
                n_lbl = label.shape[1]
                label_aug = torch.cat([label, wmap], dim=1)
                image, label_aug = self.augmentor(image, label_aug)
                label, wmap = label_aug[:, :n_lbl], label_aug[:, n_lbl:]
            else:
                image, label = self.augmentor(image, label)

            # --- Center-crop when dataset returned oversampled patches
            if self.needs_crop:
                image, label, wmap = self._center_crop(image, label, wmap)

            # --- 2.5D adaptation: collapse the C_res=1 channel so the D
            #     axis becomes the model's input-channel dimension.
            if self.is_2_5d:
                image, label, wmap = self._squeeze_2_5d(image, label, wmap)

            # --- Effective accumulation denominator for this step
            if remainder > 0 and step >= partial_start:
                effective_accum = remainder
            else:
                effective_accum = accum

            # --- Forward + loss
            with autocast(device_type="cuda", enabled=self.use_amp,
                          dtype=self.amp_dtype):
                pred = self.model(image)
                loss = self.criterion(pred, label, weight_map=wmap)
                if effective_accum > 1:
                    loss = loss / effective_accum

            # --- Backward (accumulates into .grad)
            self.scaler.scale(loss).backward()

            # --- Step boundary: every `accum` micro-steps, or at end of
            #     epoch to flush the partial tail.
            is_step_boundary = (
                (step + 1) % accum == 0 or (step + 1) == total_steps)
            if is_step_boundary:
                if tc.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), tc.grad_clip_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                self.scheduler.step()
                if self.ema is not None:
                    self.ema.update(self.model)

            # --- Metrics (log unscaled loss)
            loss_val = (loss.item() * effective_accum
                        if effective_accum > 1 else loss.item())
            loss_meter.update(loss_val, image.shape[0])

            if (step + 1) % tc.log_every == 0 or step == 0:
                with torch.no_grad():
                    p = pred[0] if isinstance(pred, list) else pred
                    # Mode-agnostic via the inner wrapper's contract:
                    #   3D : returns (B, num_fg, *spatial), (B, num_fg, *spatial)
                    #   2.5D: returns (B*D, num_fg, H, W),  (B*D, num_fg, H, W)
                    p_1x, lbl_1x = self._inner_loss.split_for_metrics(
                        p.detach(), label)
                    dice = compute_dice_per_class(p_1x, lbl_1x)
                    mean_dice = dice.mean().item()
                    dice_meter.update(mean_dice, image.shape[0])
                logger.debug("  [%d/%d] loss=%.4f dice=%.4f lr=%.2e",
                             step + 1, total_steps,
                             loss_val, mean_dice, self.scheduler.get_lr())

        return {"loss": loss_meter.avg, "dice": dice_meter.avg}

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """Validate on the validation set under EMA weights (if enabled).

        Uses POOLED per-class dice:
            dice[c] = 2 * Σ_batches intersection[c] / Σ_batches denom[c]
        This matches the nnU-Net convention and avoids the negative bias of
        averaging per-batch dice when some classes are empty in some batches.
        """
        self.model.eval()
        loss_meter = AverageMeter()
        inter_sum: Optional[torch.Tensor] = None  # (C,)
        denom_sum: Optional[torch.Tensor] = None  # (C,)
        cov_sum:   Optional[torch.Tensor] = None  # (C,) number of samples with non-empty GT per class

        n_samples = 0

        with self._ema_swapped():
            for batch in self.val_loader:
                image = batch["image"].to(self.device, non_blocking=True)
                label = batch["label"].to(self.device, non_blocking=True)

                # 2.5D: squeeze C_res=1 for both image and label before
                # forward. (No GPU augmentation in val — directly squeeze.)
                if self.is_2_5d:
                    image, label, _ = self._squeeze_2_5d(image, label, None)

                with autocast(device_type="cuda", enabled=self.use_amp,
                              dtype=self.amp_dtype):
                    pred = self.model(image)
                    if isinstance(pred, list):
                        pred = pred[0]
                    pred_1x, target_1x = self._inner_loss.split_for_metrics(
                        pred, label)
                    loss = self.base_loss(pred_1x, target_1x)

                loss_meter.update(loss.item(), image.shape[0])
                stats = dice_batch_stats(pred_1x.float(), target_1x)
                if inter_sum is None:
                    inter_sum = stats["inter"].clone()
                    denom_sum = stats["denom"].clone()
                    cov_sum   = stats["n_with_gt"].clone()
                else:
                    inter_sum += stats["inter"]
                    denom_sum += stats["denom"]
                    cov_sum   += stats["n_with_gt"]
                n_samples += image.shape[0]

        if inter_sum is None:
            logger.warning("Validation loader yielded no batches.")
            return {"val_loss": float("nan"), "mean_dice": 0.0}

        # Pooled dice with 1e-5 smoothing to match training loss behaviour.
        smooth = 1e-5
        dice_per_class = (2.0 * inter_sum + smooth) / (denom_sum + smooth)
        dice_per_class = dice_per_class.cpu()

        metrics: Dict[str, float] = {"val_loss": loss_meter.avg}
        for c in range(len(dice_per_class)):
            metrics[f"dice_class_{c}"] = dice_per_class[c].item()
        metrics["mean_dice"] = dice_per_class.mean().item()

        # Per-class coverage helps diagnose "val dice is low because this
        # class barely appears in the val set" vs. genuine model failure.
        cov = cov_sum.cpu().tolist()
        logger.info(
            "  Val: loss=%.4f, pooled_mean_dice=%.4f, per_class=%s, "
            "coverage=%s/%d samples",
            metrics["val_loss"], metrics["mean_dice"],
            [f"{d:.4f}" for d in dice_per_class.tolist()],
            [int(c) for c in cov], n_samples)
        return metrics

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _squeeze_2_5d(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        wmap: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Collapse the C_res=1 channel for 2.5D mode.

        Input shapes (post-augment, post-crop):
          image: (B, 1, D, H, W) → (B, D, H, W)
          label: (B, 1, D, H, W) raw int labels → (B, D, H, W)
          wmap : (B, 1, D, H, W) per-voxel weights or None → (B, D, H, W)

        The squeezed shape is the input contract for the 2D model and for
        ``SliceChannelLoss``: D becomes the input-channel axis of the
        model and the slice axis of the loss.
        """
        assert image.shape[1] == 1 and label.shape[1] == 1, (
            "2.5D mode expects single-resolution dataset (C_res=1); got "
            f"image={tuple(image.shape)}, label={tuple(label.shape)}")
        image = image.squeeze(1)
        label = label.squeeze(1)
        if wmap is not None:
            wmap = wmap.squeeze(1)
        return image, label, wmap

    def _center_crop(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        wmap: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Center-crop oversized tensors to target patch_size after
        augmentation (used when `aug_oversample_ratio > 1.0` in cubic mode).
        """
        tD, tH, tW = self.target_patch_size
        _, _, D, H, W = image.shape
        d0, h0, w0 = (D - tD) // 2, (H - tH) // 2, (W - tW) // 2
        image = image[:, :, d0:d0 + tD, h0:h0 + tH, w0:w0 + tW]
        label = label[:, :, d0:d0 + tD, h0:h0 + tH, w0:w0 + tW]
        if wmap is not None:
            wmap = wmap[:, :, d0:d0 + tD, h0:h0 + tH, w0:w0 + tW]
        return image, label, wmap

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _build_state_dict(self, ema_as_primary: bool) -> Dict:
        """Assemble a complete training state.

        When `ema_as_primary` is True the saved `model_state_dict` holds EMA
        weights (deployment-friendly) and online weights are preserved in
        `model_online_state_dict` for correct resuming. Otherwise
        `model_state_dict` is online and EMA lives in `ema_state_dict`.
        """
        bare = _unwrap_compile(self.model)
        online_sd = bare.state_dict()

        # Snapshot RNG state for bit-exact resume. Covers torch CPU / CUDA,
        # numpy, and Python's random — the three sources seed_everything sets.
        rng_state = {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None),
            "numpy": __import__("numpy").random.get_state(),
            "python": __import__("random").getstate(),
        }

        state: Dict = {
            "epoch": 0,  # filled by caller
            "model_state_dict": online_sd,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "has_best": self.has_best,
            "patience_counter": self.patience_counter,
            "rng_state": rng_state,
            "config": self.cfg,
        }

        if self.ema is not None:
            state["ema_state_dict"] = self.ema.state_dict()
            if ema_as_primary:
                # Capture EMA weights as the primary state_dict. try/finally
                # ensures the model is never left with EMA weights bound.
                self.ema.apply_shadow(self.model)
                try:
                    state["model_state_dict"] = _unwrap_compile(
                        self.model).state_dict()
                finally:
                    self.ema.restore(self.model)
                state["model_online_state_dict"] = online_sd

        return state

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        state = self._build_state_dict(ema_as_primary=is_best)
        state["epoch"] = epoch

        if is_best:
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

        # Prefer the online copy if present (best-model checkpoints store
        # EMA as the primary state_dict, online as a sibling).
        model_sd = ckpt.get("model_online_state_dict",
                            ckpt["model_state_dict"])
        _unwrap_compile(self.model).load_state_dict(model_sd)

        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if self.ema is not None and "ema_state_dict" in ckpt:
            self.ema.load_state_dict(ckpt["ema_state_dict"])

        self.start_epoch = ckpt.get("epoch", -1) + 1
        default_best = -math.inf if self._best_mode == "max" else math.inf
        self.best_metric = ckpt.get("best_metric", default_best)
        self.best_epoch = ckpt.get("best_epoch", 0)
        self.has_best = ckpt.get(
            "has_best", math.isfinite(self.best_metric))
        self.patience_counter = ckpt.get("patience_counter", 0)

        # Restore RNG state when present. Missing keys (older checkpoints)
        # are silently skipped — training still works, just not bit-exact.
        rng = ckpt.get("rng_state")
        if rng:
            try:
                if rng.get("torch_cpu") is not None:
                    torch.set_rng_state(rng["torch_cpu"])
                if rng.get("torch_cuda") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(rng["torch_cuda"])
                if rng.get("numpy") is not None:
                    import numpy as _np
                    _np.random.set_state(rng["numpy"])
                if rng.get("python") is not None:
                    import random as _rnd
                    _rnd.setstate(rng["python"])
                logger.info("Restored RNG state from checkpoint.")
            except Exception as e:  # pragma: no cover
                logger.warning("Failed to restore RNG state: %s", e)

        logger.info(
            "Resumed from epoch %d, best=%s=%s (patience=%d)",
            self.start_epoch, self.cfg.train.save_best_metric,
            f"{self.best_metric:.4f}" if self.has_best else "n/a",
            self.patience_counter)