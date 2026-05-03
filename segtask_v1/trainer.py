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
# GradScaler moved from torch.cuda.amp to torch.amp in PyTorch ≥ 2.3. The
# new constructor takes a device string as the first positional argument;
# the legacy CUDA-namespace constructor does NOT and would silently bind
# "cuda" to `init_scale`, producing a confusing TypeError later inside
# `torch.full(..., self._init_scale, ...)`. Probe the signature to call
# whichever form the installed torch supports.
import inspect as _inspect
try:
    from torch.amp import GradScaler as _GradScaler  # type: ignore
    from torch.amp import autocast  # type: ignore
except ImportError:  # pragma: no cover - version-dependent
    from torch.cuda.amp import GradScaler as _GradScaler  # type: ignore
    from torch.amp import autocast  # type: ignore


def GradScaler(device: str = "cuda", **kwargs):  # noqa: N802 - mimic class
    """Version-agnostic GradScaler constructor.

    Passes ``device`` only when the underlying class accepts it (PyTorch
    ≥ 2.3). On older builds (e.g. 2.2) the legacy ``torch.cuda.amp.GradScaler``
    is CUDA-only and rejects the argument.
    """
    try:
        params = _inspect.signature(_GradScaler).parameters
    except (TypeError, ValueError):
        params = {}
    if "device" in params:
        return _GradScaler(device, **kwargs)
    return _GradScaler(**kwargs)
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
                reduction=cfg.loss.slice_loss_reduction,
            )
            num_res = 1   # for logging only; SliceChannelLoss has C_res==1
            logger.info(
                "Loss: %s [2.5D, reduction=%s], num_slices=%d, fg_classes=%d",
                cfg.loss.name, cfg.loss.slice_loss_reduction,
                num_slices, cfg.num_fg_classes)
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
        # The augmentor applies spatial transforms jointly to image,
        # label, and (optionally) weight_map so alignment holds. label
        # uses nearest-neighbour interpolation (preserves discrete
        # values); weight_map uses bilinear (preserves continuous,
        # hand-annotated per-voxel weight gradients).
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

        # --- Resume / Pretrain ----------------------------------------
        # Semantics:
        #   * `resume`   → full state restore (training continues exactly).
        #   * `pretrain` → model-weights-only initialisation; epoch / optim /
        #                  scheduler / scaler / best / RNG all stay fresh.
        # `resume` wins when both are configured *and* the file exists.
        resume_active = bool(tc.resume) and os.path.isfile(tc.resume)
        pretrain_active = bool(tc.pretrain) and os.path.isfile(tc.pretrain)

        if resume_active:
            if tc.pretrain:
                logger.warning(
                    "Both `train.resume` and `train.pretrain` are set; "
                    "using resume (%s). Pretrain weights from %s are ignored.",
                    tc.resume, tc.pretrain)
            self._load_checkpoint(tc.resume)
        elif pretrain_active:
            self._load_pretrain(
                tc.pretrain,
                strict=tc.pretrain_strict,
                load_ema=tc.pretrain_load_ema)
        else:
            # Surface mis-configurations early instead of silently ignoring.
            if tc.resume and not os.path.isfile(tc.resume):
                logger.warning(
                    "`train.resume` is set but file not found: %s. "
                    "Training will start from scratch.", tc.resume)
            if tc.pretrain and not os.path.isfile(tc.pretrain):
                logger.warning(
                    "`train.pretrain` is set but file not found: %s. "
                    "Training will start from scratch.", tc.pretrain)

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

            # --- GPU augmentation: image, label and (optional) weight_map
            #     share one sampled spatial transform so alignment holds.
            #     The augmentor uses nearest interpolation for label and
            #     bilinear for weight_map (continuous values).
            image, label, wmap = self.augmentor(image, label, wmap)

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

            # --- Forward (AMP) + loss (fp32)
            #
            # Model forward stays under autocast for the fp16 speedup, but
            # the loss is computed in fp32. Dice / BCE compounds reduce
            # over millions of voxels per batch (e.g. B*D*H*W ≈ 3.1M in
            # 2.5D); the running sums in the dice numerator/denominator
            # quickly exceed fp16's ±65504 range once the model starts
            # confidently predicting foreground, producing inf → NaN
            # losses and poisoning training (root cause of the epoch-20
            # NaN explosion observed historically). Casting pred to fp32
            # outside autocast is the standard nnU-Net-style fix.
            with autocast(device_type="cuda", enabled=self.use_amp,
                          dtype=self.amp_dtype):
                pred = self.model(image)
            loss = self._compute_loss_fp32(
                self.criterion, pred, label, weight_map=wmap)
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
            # Guard the running average against single-batch non-finite
            # losses. A single NaN/Inf would otherwise poison ``sum`` and
            # render every later ``loss_meter.avg`` NaN for the rest of the
            # epoch (and all subsequent epochs' logs) even though training
            # itself continues healthily via GradScaler's inf-skip path.
            if math.isfinite(loss_val):
                loss_meter.update(loss_val, image.shape[0])
            else:
                logger.warning(
                    "Non-finite train loss (%s) at epoch %d step %d/%d; "
                    "skipping meter update. GradScaler will skip this "
                    "optimizer step.",
                    loss_val, epoch + 1, step + 1, total_steps)

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
                # Loss in fp32 — see `_train_epoch` for rationale (fp16 dice
                # reductions overflow on large patches and produce NaN).
                loss = self._compute_loss_fp32(
                    self.base_loss, pred_1x, target_1x)

                loss_val = loss.item()
                if math.isfinite(loss_val):
                    loss_meter.update(loss_val, image.shape[0])
                else:
                    logger.warning(
                        "Non-finite val loss (%s) at epoch %d; skipping "
                        "meter update.", loss_val, epoch + 1)
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
    # Maximum absolute logit value fed into the loss. AMP fp16 can produce
    # ``±inf`` in the forward path (instance-norm variance underflow,
    # saturated activations, large late-training weights). ``pred.float()``
    # does NOT recover those: ``inf`` in fp16 stays ``inf`` in fp32. The
    # numerically-stable BCE-with-logits kernel then evaluates
    #     max(x,0) - x*z + log1p(exp(-|x|))
    # which for ``x=+inf, target=1`` is ``inf - inf + 0 = NaN`` — the
    # dominant cause of single-batch NaN losses late in training. Clamping
    # to ±_LOGIT_CLAMP wipes that failure mode at zero cost: ``sigmoid(50)``
    # is already indistinguishable from 1.0 in fp32 (Dice is unaffected),
    # and BCE's gradient ``sigmoid(x) - target`` at ``|x|=50`` is 0 or 1 —
    # the same as the unclamped limit, so healthy training (|x| << 20) sees
    # bit-identical behaviour.
    _LOGIT_CLAMP: float = 50.0

    @staticmethod
    def _compute_loss_fp32(
        loss_fn: nn.Module,
        pred,
        target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run ``loss_fn`` outside autocast with fp32 inputs.

        Casts every prediction tensor (single tensor or list-of-tensors as
        produced by deep supervision) to ``float32`` and disables autocast
        for the loss call. The label tensor is cast to fp32 only if it is
        a float dtype — integer label tensors must remain integer-typed
        because some compound losses round / index them. ``weight_map``
        is cast to fp32 when provided.

        Prediction logits are additionally clamped to
        ``[-_LOGIT_CLAMP, +_LOGIT_CLAMP]`` to defuse occasional ±inf from
        the fp16 autocast forward path (see ``_LOGIT_CLAMP`` docstring).
        """
        c = Trainer._LOGIT_CLAMP
        if isinstance(pred, list):
            pred_fp32 = [p.float().clamp(-c, c) for p in pred]
        else:
            pred_fp32 = pred.float().clamp(-c, c)
        target_fp32 = target.float() if target.is_floating_point() else target
        wmap_fp32 = weight_map.float() if weight_map is not None else None
        with autocast(device_type="cuda", enabled=False):
            if wmap_fp32 is None:
                return loss_fn(pred_fp32, target_fp32)
            return loss_fn(pred_fp32, target_fp32, weight_map=wmap_fp32)

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
        # PyTorch 2.6+ flipped ``weights_only`` default to True, which
        # rejects our checkpoints because they contain numpy RNG state
        # (``numpy._core.multiarray._reconstruct``) and the ``Config``
        # dataclass — neither is on the safe-globals allowlist. These
        # checkpoints are written by this trainer itself (trusted source),
        # so we opt back into full unpickling explicitly.
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

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

    # ------------------------------------------------------------------
    # Pretrain (weights-only initialisation)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_model_state_dict(ckpt, prefer_ema: bool):
        """Locate the model state_dict inside a checkpoint, format-tolerant.

        Supports three layouts:
          1. Trainer's own checkpoints (dict with ``model_state_dict`` /
             optionally ``model_online_state_dict`` / ``ema_state_dict``).
          2. Common third-party convention: ``{"state_dict": OrderedDict}``.
          3. Raw state_dict pickled directly (an OrderedDict of tensors).

        Returns:
            (state_dict, source_label) where ``source_label`` is a short string
            describing which slot was picked — useful for logs.
        """
        # Case 3: raw state_dict
        if not isinstance(ckpt, dict) or all(
                isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt, "raw_state_dict"

        # Case 1a: prefer EMA shadow when requested and present
        if prefer_ema and "ema_state_dict" in ckpt:
            ema_state = ckpt["ema_state_dict"]
            if isinstance(ema_state, dict) and "shadow" in ema_state:
                return ema_state["shadow"], "ema_shadow"

        # Case 1b: trainer-format online weights (best-model ckpts keep online
        # in a sibling slot when EMA is the primary).
        if "model_online_state_dict" in ckpt:
            return ckpt["model_online_state_dict"], "model_online_state_dict"
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"], "model_state_dict"

        # Case 2: third-party "state_dict" wrapper
        if "state_dict" in ckpt:
            return ckpt["state_dict"], "state_dict"

        raise KeyError(
            "Pretrain checkpoint does not contain a recognisable model "
            "state_dict. Expected one of: 'model_state_dict', "
            "'model_online_state_dict', 'state_dict', or a raw OrderedDict.")

    @staticmethod
    def _strip_common_prefixes(sd):
        """Drop ``module.`` (DDP) and ``_orig_mod.`` (torch.compile) prefixes.

        Pretrain sources are often produced by DDP / compiled training; our
        target model here is the bare unwrapped module, so unifying the
        namespace before loading avoids spurious "missing key" floods.
        """
        if not isinstance(sd, dict):
            return sd
        prefixes = ("module.", "_orig_mod.")
        out = {}
        changed = False
        for k, v in sd.items():
            new_k = k
            # Strip iteratively in case prefixes are nested (compile(DDP(...))).
            while new_k.startswith(prefixes):
                for p in prefixes:
                    if new_k.startswith(p):
                        new_k = new_k[len(p):]
                        changed = True
                        break
            out[new_k] = v
        return out if changed else sd

    def _load_pretrain(self, path: str, strict: bool, load_ema: bool) -> None:
        """Load model weights only — used for transfer-learning init.

        Crucially distinct from ``_load_checkpoint``:
            * Does NOT touch optimizer / scheduler / scaler / RNG.
            * Does NOT advance ``start_epoch`` or restore best-metric tracking.
            * Re-aligns EMA shadow to the freshly loaded weights so EMA does
              not silently drift back toward the random init.
        """
        logger.info(
            "Loading pretrain weights: %s (strict=%s, load_ema=%s)",
            path, strict, load_ema)
        # weights_only=False mirrors `_load_checkpoint`: trainer-format ckpts
        # contain numpy RNG state and the Config dataclass which the safe
        # unpickler rejects. Pretrain sources are explicitly user-provided.
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        sd, source = self._extract_model_state_dict(ckpt, prefer_ema=load_ema)
        sd = self._strip_common_prefixes(sd)

        bare = _unwrap_compile(self.model)
        result = bare.load_state_dict(sd, strict=strict)
        missing = list(getattr(result, "missing_keys", []) or [])
        unexpected = list(getattr(result, "unexpected_keys", []) or [])

        # Compact, actionable diagnostics. Showing the first few keys is far
        # more useful than dumping hundreds when the head doesn't match.
        def _preview(keys, n=8):
            head = ", ".join(keys[:n])
            return head + (f", ... (+{len(keys) - n} more)" if len(keys) > n else "")

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

        # Re-align EMA shadow with the loaded weights. Without this, the
        # shadow would still hold the model's *random* init from
        # ``ModelEMA.__init__`` and EMA-based validation/checkpoints would
        # regress toward noise for the first ~1/(1-decay) steps.
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