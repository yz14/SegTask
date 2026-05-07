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
from typing import Dict, Iterator, List, Optional, Tuple

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

        # ``aux_keep_native_d`` flag is decided early so the aux loss
        # block below can build per-view ``SliceChannelLoss`` instances
        # (one per native D_k) instead of a single shared one.
        # ``aux_view_depths`` and the inflated ``target_patch_size`` D
        # axis are still finalised later in __init__ once we know
        # ``cfg.model.in_channels`` and crop targets.
        self.aux_keep_native_d = bool(
            getattr(cfg.data, "aux_keep_native_d", False)
            and self.is_2_5d
            and len(cfg.data.multi_res_scales) > 1)
        # Provisional — overwritten in the cropping section once we have
        # the full cfg context. Kept here so the aux loss block can read it.
        self.aux_view_depths: List[int] = (
            list(cfg.aux_view_depths) if self.aux_keep_native_d else [])

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

        # ---- Multi-FOV aux segmentation supervision (2.5D mode) -----
        # Active only when (1) we're in 2.5D mode, (2) the user opted in via
        # ``model.aux_seg_supervision``, AND (3) there is at least one aux
        # FOV view (n_views>1). Otherwise the aux path is fully bypassed
        # (no extra modules built, no behaviour change).
        n_views_data = len(cfg.data.multi_res_scales)
        self.aux_seg_supervision = bool(
            getattr(cfg.model, "aux_seg_supervision", False)
            and self.is_2_5d
            and n_views_data > 1)
        if self.aux_seg_supervision:
            n_aux = n_views_data - 1
            user_w = list(getattr(cfg.loss, "aux_supervision_weights", []))
            if not user_w:
                # Geometric decay default — wider FOVs (less precise pixel
                # alignment with view 0's true geometry) get smaller weight.
                # Same convention as deep_supervision_weights.
                user_w = [0.5 ** (k + 1) for k in range(n_aux)]
            elif len(user_w) != n_aux:
                # Defensive — Config.validate() already guards this, but
                # repeat the check so a hand-crafted Config dict still
                # fails fast if it bypasses YAML loading.
                raise ValueError(
                    f"loss.aux_supervision_weights length ({len(user_w)}) "
                    f"must equal n_views-1 ({n_aux}); got {user_w}.")
            self.aux_weights = [float(w) for w in user_w]
            # Aux paths compute single-resolution per-view losses (no DS
            # for aux — DS structure is reserved for the main path's
            # multi-scale supervision). The inner SliceChannelLoss is the
            # same shape contract as the main one: (B, num_fg*D_k, H, W)
            # pred + (B, D_k, H, W) raw label.
            #
            # Two layouts:
            #   - Legacy (aux_keep_native_d=False): every aux view has
            #     ``num_slices = D`` (input is z-resampled back to D).
            #     A single shared ``SliceChannelLoss`` handles every view.
            #   - Native depth (aux_keep_native_d=True): aux view k has
            #     ``num_slices = D_k = round(D * s_k)``. We build ONE
            #     ``SliceChannelLoss`` per aux view so the slice-channel
            #     contract is exact (the wrapper uses ``num_slices`` to
            #     reshape (B, num_fg*D_k, H, W) → (B*num_fg, D_k, H, W)).
            if getattr(self, "aux_keep_native_d", False):
                # ``self.aux_view_depths`` was built earlier in __init__;
                # validated to match ``cfg.aux_view_depths``.
                aux_depths = self.aux_view_depths[1:]  # skip view 0
                assert len(aux_depths) == n_aux, (
                    f"aux_view_depths excluding view 0 has length "
                    f"{len(aux_depths)}; expected {n_aux}.")
                self.aux_inner_loss = None
                self.aux_inner_losses = [
                    SliceChannelLoss(
                        base_loss=self.base_loss,
                        num_fg_classes=cfg.num_fg_classes,
                        num_slices=int(d_k),
                        label_values=cfg.data.label_values,
                        reduction=cfg.loss.slice_loss_reduction,
                    )
                    for d_k in aux_depths
                ]
                logger.info(
                    "Aux seg supervision: ENABLED (native depth), "
                    "n_aux_views=%d, per-view depths=%s, weights=%s, "
                    "fusion=%s",
                    n_aux, aux_depths, self.aux_weights,
                    cfg.model.context_fusion)
            else:
                self.aux_inner_loss = SliceChannelLoss(
                    base_loss=self.base_loss,
                    num_fg_classes=cfg.num_fg_classes,
                    num_slices=int(cfg.data.patch_size[0]),
                    label_values=cfg.data.label_values,
                    reduction=cfg.loss.slice_loss_reduction,
                )
                self.aux_inner_losses = None
                logger.info(
                    "Aux seg supervision: ENABLED, n_aux_views=%d, weights=%s, "
                    "fusion=%s",
                    n_aux, self.aux_weights, cfg.model.context_fusion)
        else:
            self.aux_weights = []
            self.aux_inner_loss = None
            self.aux_inner_losses = None

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
        #
        # ``aux_keep_native_d`` (2.5D ONLY): the dataset emits a single
        # max-FOV cube of depth ``round(D * max_scale)`` (× oversample
        # margin). The post-augment center crop must therefore preserve
        # the ENTIRE max-FOV (not just D) — view 0 takes the centered D
        # slices in the per-view split below; aux views need wider z spans.
        # We override the D-axis target accordingly and keep H, W at
        # their patch_size values.
        if self.aux_keep_native_d:
            max_scale = max(cfg.data.multi_res_scales)
            target_d_native = int(round(int(cfg.data.patch_size[0]) * max_scale))
            self.target_patch_size = (target_d_native,
                                      int(cfg.data.patch_size[1]),
                                      int(cfg.data.patch_size[2]))
            # Sanity-check the provisional ``aux_view_depths`` set early
            # in __init__ now that we have the full Config view.
            assert self.aux_view_depths[0] == int(cfg.data.patch_size[0]), (
                "aux_view_depths[0] must equal patch_size[0]; got "
                f"{self.aux_view_depths[0]} vs {cfg.data.patch_size[0]}.")
            assert sum(self.aux_view_depths) == int(cfg.model.in_channels), (
                f"sum(aux_view_depths)={sum(self.aux_view_depths)} must "
                f"equal model.in_channels={cfg.model.in_channels}.")
            logger.info(
                "Trainer aux_keep_native_d=True: max-FOV crop D=%d, "
                "per-view depths=%s, channel layout sum=%d.",
                target_d_native, self.aux_view_depths,
                int(cfg.model.in_channels))
        else:
            self.target_patch_size = tuple(cfg.data.patch_size)
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
            # Aggregate the aux component averages into a compact suffix
            # using the same renderer as the per-step debug line. Empty
            # for non-aux runs → epoch line is bit-identical to legacy.
            aux_summary_dict = {
                k: v for k, v in train_metrics.items()
                if k.startswith("L_main") or k.startswith("L_aux_")
                or k.startswith("w_aux_")
            }
            aux_msg = self._format_breakdown(aux_summary_dict)
            logger.info(
                "Epoch %d/%d | LR=%.2e | loss=%.4f | val_dice=%.4f | "
                "best=%s | %s%s",
                epoch + 1, tc.epochs, self.scheduler.get_lr(),
                train_metrics.get("loss", 0.0),
                val_metrics.get("mean_dice", 0.0),
                best_str,
                timer.elapsed_str(),
                aux_msg,
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
        # Per-component meters for the multi-FOV aux supervision diagnostic
        # log. Populated lazily on the first batch (we don't know the aux
        # view names until ``_compute_loss_aux_fp32`` runs once); from then
        # on each component is averaged across the epoch independently.
        # Keys: "L_main", "L_aux_1", "L_aux_2", ...
        component_meters: Dict[str, AverageMeter] = {}
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
            #     With aux seg supervision active the label/wmap tensors
            #     are kept at rank-5 ``(B, C_res, D, H, W)`` so per-view
            #     losses can index ``label[:, k]`` for aux head ``k``.
            label_all_views: Optional[torch.Tensor] = None
            wmap_all_views: Optional[torch.Tensor] = None
            # ``aux_view_labels[k]`` / ``aux_view_wmaps[k]`` carry the per-
            # view native-depth supervision targets when aux_keep_native_d
            # is on. They are list-form because views have varying D_k —
            # the legacy rank-5 ``label_all_views`` cannot represent that.
            aux_view_labels: Optional[List[torch.Tensor]] = None
            aux_view_wmaps: Optional[List[Optional[torch.Tensor]]] = None
            if self.is_2_5d:
                if self.aux_seg_supervision and self.aux_keep_native_d:
                    # Native-depth path: dataset emits a single max-FOV
                    # cube; we center-crop per view BEFORE forward, with
                    # view 0 = the standard ``D``-deep main supervision
                    # and views 1..K = native ``D_k``-deep aux targets.
                    (image, label, wmap,
                     aux_view_labels, aux_view_wmaps) = (
                        self._split_views_native_d(image, label, wmap))
                elif self.aux_seg_supervision:
                    image, label_all_views, wmap_all_views = (
                        self._squeeze_2_5d_keep_views(image, label, wmap))
                    # ``label`` / ``wmap`` shadow view 0 so the dice metric
                    # block below — which is shared with the no-aux path —
                    # operates on the same supervision target as the main
                    # head (bit-equivalent metric definition).
                    label = label_all_views[:, 0]
                    wmap = (wmap_all_views[:, 0]
                            if wmap_all_views is not None else None)
                else:
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
            breakdown: Dict[str, float] = {}
            if self.aux_seg_supervision and self.aux_keep_native_d:
                # Native-depth aux path: ``aux_view_labels[k]`` /
                # ``aux_view_wmaps[k]`` carry view-k targets at native
                # depth ``D_k``. ``label`` / ``wmap`` are view 0 (D-deep).
                loss = self._compute_loss_aux_native_d_fp32(
                    pred, label, wmap,
                    aux_view_labels, aux_view_wmaps,
                    breakdown=breakdown)
            elif self.aux_seg_supervision:
                # Aux-aware path: pred is a dict and we route per-view
                # supervision through ``_compute_loss_aux_fp32``. The main
                # path inside that helper still goes through the
                # DS-wrapped criterion, so deep_supervision composes with
                # aux supervision orthogonally. The ``breakdown`` dict is
                # filled in-place with detached scalars per component
                # (L_main, L_aux_k, w_aux_k, L_total) for diagnostic logs.
                loss = self._compute_loss_aux_fp32(
                    pred, label_all_views, wmap_all_views,
                    breakdown=breakdown)
            else:
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
                # Aux breakdown averaging — only update meters with finite
                # scalars to mirror the main loss meter's NaN guard.
                for name, val in breakdown.items():
                    if not math.isfinite(val):
                        continue
                    if name not in component_meters:
                        component_meters[name] = AverageMeter()
                    component_meters[name].update(val, image.shape[0])
            else:
                logger.warning(
                    "Non-finite train loss (%s) at epoch %d step %d/%d; "
                    "skipping meter update. GradScaler will skip this "
                    "optimizer step.",
                    loss_val, epoch + 1, step + 1, total_steps)

            if (step + 1) % tc.log_every == 0 or step == 0:
                with torch.no_grad():
                    # Unify dict / list / tensor outputs to the main full-
                    # res tensor before metric splitting. ``label`` is
                    # already view 0 in both aux and no-aux paths above.
                    p = self._extract_main_pred(pred)
                    # Mode-agnostic via the inner wrapper's contract:
                    #   3D : returns (B, num_fg, *spatial), (B, num_fg, *spatial)
                    #   2.5D: returns (B*D, num_fg, H, W),  (B*D, num_fg, H, W)
                    p_1x, lbl_1x = self._inner_loss.split_for_metrics(
                        p.detach(), label)
                    dice = compute_dice_per_class(p_1x, lbl_1x)
                    mean_dice = dice.mean().item()
                    dice_meter.update(mean_dice, image.shape[0])
                # Compact diagnostic line: when aux supervision is on, append
                # "L_main / L_aux_k=val(w=...)" so the user can immediately
                # see whether each aux head is contributing meaningful
                # gradient (similar magnitude to L_main scaled by w_k).
                aux_msg = self._format_breakdown(breakdown)
                logger.debug(
                    "  [%d/%d] loss=%.4f dice=%.4f lr=%.2e%s",
                    step + 1, total_steps,
                    loss_val, mean_dice, self.scheduler.get_lr(),
                    aux_msg)

        # Surface the epoch-mean of each aux component alongside the
        # standard loss/dice metrics. ``fit()`` formats them into the
        # epoch summary line so the user sees aux contributions at a
        # cadence that matches the main metric log.
        out: Dict[str, float] = {"loss": loss_meter.avg, "dice": dice_meter.avg}
        for name, meter in component_meters.items():
            out[name] = meter.avg
        return out

    @staticmethod
    def _format_breakdown(breakdown: Dict[str, float]) -> str:
        """Render a compact " | L_main=... L_aux_1=...(w=...) ..." string.

        Returns "" when the breakdown is empty (single-FOV or aux disabled),
        keeping legacy log lines bit-identical for non-aux runs.
        """
        if not breakdown:
            return ""
        parts: List[str] = []
        # Stable ordering: L_main first, then L_aux_k in ascending k.
        if "L_main" in breakdown:
            parts.append(f"L_main={breakdown['L_main']:.4f}")
        aux_keys = sorted(
            (k for k in breakdown if k.startswith("L_aux_")),
            key=lambda k: int(k.split("_")[-1]))
        for k in aux_keys:
            view_k = k.split("_")[-1]
            w_key = f"w_aux_{view_k}"
            if w_key in breakdown:
                parts.append(
                    f"{k}={breakdown[k]:.4f}(w={breakdown[w_key]:.3g})")
            else:
                parts.append(f"{k}={breakdown[k]:.4f}")
        return " | " + " ".join(parts)

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
                # In aux_keep_native_d mode the dataset emits a single
                # max-FOV cube; we run the same per-view split as the
                # training loop but discard aux targets (val metric only
                # exercises view 0 = main supervision).
                if self.is_2_5d:
                    if self.aux_keep_native_d:
                        image, label, _, _, _ = (
                            self._split_views_native_d(image, label, None))
                    else:
                        image, label, _ = self._squeeze_2_5d(image, label, None)

                with autocast(device_type="cuda", enabled=self.use_amp,
                              dtype=self.amp_dtype):
                    pred = self.model(image)
                    # ``_extract_main_pred`` is dict/list/tensor-safe.
                    # In eval mode UNet3D returns a tensor (aux gated by
                    # ``self.training``), but the helper keeps the val
                    # contract robust if a future change relaxes that.
                    pred = self._extract_main_pred(pred)
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

    # ------------------------------------------------------------------
    # Aux-aware helpers (multi-FOV deep supervision in 2.5D mode)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_main_pred(pred):
        """Unwrap the model output to its main-path tensor.

        Accepts:
          - dict {"main": tensor|list, "aux": [...]} → returns tensor
            (DS list is collapsed to its head, the main full-res output).
          - list [main_out, ds_2nd, ...] → returns list[0].
          - tensor → returns as-is.

        Mirrors how ``_train_epoch`` historically picked the main scale
        before the aux-supervision dict contract was added; centralising
        the logic here keeps the metric / log path branch-free.
        """
        if isinstance(pred, dict):
            pred = pred["main"]
        if isinstance(pred, list):
            pred = pred[0]
        return pred

    def _split_views_native_d(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        wmap: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        List[torch.Tensor],
        List[Optional[torch.Tensor]],
    ]:
        """Per-view native-depth center crop for the 2.5D simplified path.

        Input contract (post-augment, post-center-crop):
          image : ``(B, 1, eD_max, H, W)`` — single max-FOV cube emitted
                  by ``SegDataset3D._getitem_native_d``. Here
                  ``eD_max == round(D * max_scale) == self.target_patch_size[0]``.
          label : ``(B, 1, eD_max, H, W)`` raw integer labels.
          wmap  : optional ``(B, 1, eD_max, H, W)`` continuous weights.

        Output:
          image_2d : ``(B, sum_k D_k, H, W)`` — per-view native-depth
                      slices concatenated along the channel axis. View 0
                      occupies channels ``[0, D_0)``; view k occupies
                      ``[Σ_{<k} D_j, Σ_{<=k} D_j)``. The 2D model consumes
                      this directly. ``D_k = round(D * s_k)`` and view 0
                      sits at the centered ``D_0 == D`` slices.
          label_main : ``(B, D, H, W)`` — view-0 supervision (raw int).
          wmap_main  : ``(B, D, H, W)`` or ``None`` — view-0 weight map.
          aux_labels : list of ``(B, D_k, H, W)`` for k=1..K-1.
          aux_wmaps  : list (same length) of weight maps or None per view.

        Geometric note
        --------------
        All views share the same z-center by construction (``_sample_z``
        runs once before extraction). Center-cropping ``D_k`` slices out
        of ``eD_max`` reproduces, voxel-for-voxel, the per-view
        independent extraction used by the legacy False-path — but with
        a single shared augmentation grid_sample applied to the whole
        cube, eliminating cross-view warp drift.
        """
        if not self.aux_keep_native_d:
            raise RuntimeError(
                "_split_views_native_d called but aux_keep_native_d=False")
        if image.ndim != 5 or image.shape[1] != 1:
            raise ValueError(
                "native-d split expects (B, 1, eD_max, H, W); got "
                f"image.shape={tuple(image.shape)}")
        if label.shape[:2] != image.shape[:2] or label.shape[2:] != image.shape[2:]:
            raise ValueError(
                "image / label shape mismatch: "
                f"image={tuple(image.shape)}, label={tuple(label.shape)}")
        B, _, eD_max, H, W = image.shape
        depths = self.aux_view_depths
        D = depths[0]
        if eD_max != int(self.target_patch_size[0]):
            raise ValueError(
                f"native-d split expects depth axis == target_patch_size[0]"
                f"={self.target_patch_size[0]}; got {eD_max}. The post-"
                "augment center crop should already have removed the "
                "augment oversample margin.")
        if max(depths) > eD_max:
            raise ValueError(
                f"max(aux_view_depths)={max(depths)} exceeds eD_max={eD_max}; "
                "this indicates a multi_res_scales / patch_size mismatch.")

        def _center_slab(t: torch.Tensor, d_k: int) -> torch.Tensor:
            """Return the center ``d_k`` slices along axis=2 of ``t`` (B, 1, D, H, W)."""
            d0 = (eD_max - d_k) // 2
            return t[:, 0, d0:d0 + d_k].contiguous()  # (B, d_k, H, W)

        # ---- View 0: main supervision target ---------------------------
        # ``image_view_0 == _center_slab(image, D)`` is exactly the (B, D, H, W)
        # tensor the legacy single-FOV / view-0 path would consume.
        view0_img = _center_slab(image, D)
        label_main = _center_slab(label, D)
        wmap_main = _center_slab(wmap, D) if wmap is not None else None

        # ---- Aux views ---------------------------------------------------
        aux_imgs: List[torch.Tensor] = []
        aux_labels: List[torch.Tensor] = []
        aux_wmaps: List[Optional[torch.Tensor]] = []
        for d_k in depths[1:]:
            aux_imgs.append(_center_slab(image, d_k))
            aux_labels.append(_center_slab(label, d_k))
            aux_wmaps.append(_center_slab(wmap, d_k) if wmap is not None else None)

        # ---- Concatenate views along the channel axis for the model -----
        if aux_imgs:
            image_2d = torch.cat([view0_img] + aux_imgs, dim=1).contiguous()
        else:
            # Defensive: aux_keep_native_d implies n_views > 1, but degrade
            # gracefully to single-view layout if someone bypasses validate.
            image_2d = view0_img.contiguous()
        expected_in = sum(depths)
        if image_2d.shape[1] != expected_in:
            raise RuntimeError(
                f"native-d split produced {image_2d.shape[1]} input "
                f"channels; expected sum(depths)={expected_in}.")
        return image_2d, label_main, wmap_main, aux_labels, aux_wmaps

    def _compute_loss_aux_native_d_fp32(
        self,
        pred,
        label_main: torch.Tensor,
        wmap_main: Optional[torch.Tensor],
        aux_labels: Optional[List[torch.Tensor]],
        aux_wmaps: Optional[List[Optional[torch.Tensor]]],
        breakdown: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """Native-depth aux loss aggregator (mirror of ``_compute_loss_aux_fp32``).

        Differences from the False-path counterpart:
          * Targets arrive as PER-VIEW LIST tensors (one (B, D_k, H, W)
            per view) instead of rank-5 ``(B, C_res, D, H, W)``, because
            views have varying depths.
          * Each view dispatches to its OWN ``SliceChannelLoss`` from
            ``self.aux_inner_losses[k-1]`` (constructed with
            ``num_slices = D_k`` so the slice/channel reshape is exact).

        Loss formula and ``breakdown`` schema are bit-identical to the
        False-path so downstream logging code (``_format_breakdown`` and
        the per-component meters in ``_train_epoch``) need no changes.
        """
        if isinstance(pred, dict):
            main_pred = pred["main"]
            aux_preds = pred.get("aux", []) or []
        else:
            main_pred, aux_preds = pred, []

        main_l = self._compute_loss_fp32(
            self.criterion, main_pred, label_main, weight_map=wmap_main)
        total = main_l
        if breakdown is not None:
            breakdown["L_main"] = float(main_l.detach().item())

        if not aux_preds or not self.aux_inner_losses:
            if breakdown is not None:
                breakdown["L_total"] = float(total.detach().item())
            return total
        if aux_labels is None:
            raise RuntimeError(
                "aux_keep_native_d aux loss path requires aux_labels list "
                "but received None — likely a missing _split_views_native_d "
                "call upstream.")
        if not (len(aux_preds) == len(self.aux_weights)
                == len(self.aux_inner_losses) == len(aux_labels)):
            raise RuntimeError(
                "aux_keep_native_d arity mismatch: "
                f"preds={len(aux_preds)}, weights={len(self.aux_weights)}, "
                f"losses={len(self.aux_inner_losses)}, "
                f"labels={len(aux_labels)}.")
        for k_idx, (ap, w_k, loss_k, lbl_k) in enumerate(zip(
                aux_preds, self.aux_weights, self.aux_inner_losses, aux_labels)):
            view_k = k_idx + 1
            wm_k = (aux_wmaps[k_idx]
                    if aux_wmaps is not None else None)
            aux_l = self._compute_loss_fp32(
                loss_k, ap, lbl_k, weight_map=wm_k)
            total = total + w_k * aux_l
            if breakdown is not None:
                breakdown[f"L_aux_{view_k}"] = float(aux_l.detach().item())
                breakdown[f"w_aux_{view_k}"] = float(w_k)
        if breakdown is not None:
            breakdown["L_total"] = float(total.detach().item())
        return total

    def _squeeze_2_5d_keep_views(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        wmap: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Aux-supervision variant of :meth:`_squeeze_2_5d`.

        The image is reshaped exactly as in the legacy single-view path —
        ``(B, C_res, D, H, W) → (B, C_res*D, H, W)`` — but ``label`` and
        ``wmap`` are returned UNCHANGED at rank-5 ``(B, C_res, D, H, W)``.
        The training loop then slices view 0 for the main supervision and
        view k for the k-th auxiliary head, ensuring every view's
        physically-resampled label can drive its corresponding output.
        """
        if image.ndim != 5:
            raise ValueError(
                f"2.5D _squeeze_keep_views expects rank-5 image "
                f"(B, C_res, D, H, W); got shape={tuple(image.shape)}")
        if label.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"image / label batch+C_res mismatch: image="
                f"{tuple(image.shape)}, label={tuple(label.shape)}")
        B, C_res, D, H, W = image.shape
        image_2d = image.reshape(B, C_res * D, H, W).contiguous()
        return image_2d, label, wmap

    def _compute_loss_aux_fp32(
        self,
        pred,
        label_all: torch.Tensor,
        wmap_all: Optional[torch.Tensor],
        breakdown: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """Aux-aware loss aggregator for the 2.5D multi-FOV training step.

        Layout
        ------
        ``pred`` is normally a dict ``{"main": ..., "aux": [...]}`` produced
        by :class:`UNet3D` when aux supervision is active. As a defensive
        fallback we accept the legacy tensor / list form too — in that
        case only the main loss is computed (zero aux contribution), so
        a configuration drift can never silently inflate the loss.

        ``label_all`` is rank-5 ``(B, C_res, D, H, W)``: view 0 supervises
        the main head, view ``k`` supervises aux head ``k``. ``wmap_all``
        carries the per-view loss weight maps (or None).

        Loss formula
        ------------
            L = L_main(view_0) + Σ_{k=1..K} w_k * L_aux(view_k)

        where ``L_main`` is the full DS+SliceChannel pipeline and each
        ``L_aux`` is a single-resolution :class:`SliceChannelLoss` on the
        view-k label. Weights ``w_k`` come from ``self.aux_weights``.
        """
        # ---- Slice the supervision tensors per view --------------------
        label_main = label_all[:, 0]
        wmap_main = wmap_all[:, 0] if wmap_all is not None else None

        if isinstance(pred, dict):
            main_pred = pred["main"]
            aux_preds = pred.get("aux", []) or []
        else:
            # Defensive: model didn't produce aux outputs (e.g. eval mode
            # leakage or aux disabled at the model side). Compute main-only.
            main_pred, aux_preds = pred, []

        # Main path uses the full criterion (DS-wrapped if enabled). The
        # outer ``_compute_loss_fp32`` handles fp32 cast + logit clamp.
        main_l = self._compute_loss_fp32(
            self.criterion, main_pred, label_main, weight_map=wmap_main)
        total = main_l
        if breakdown is not None:
            # Detach to avoid keeping an extra autograd reference; ``.item``
            # synchronises but is acceptable at log_every cadence (and we
            # already do the same for the main loss meter).
            breakdown["L_main"] = float(main_l.detach().item())

        # Aux paths — each contributes ``w_k * L_aux(view_k)``.
        if not aux_preds or self.aux_inner_loss is None:
            if breakdown is not None:
                breakdown["L_total"] = float(total.detach().item())
            return total
        if len(aux_preds) != len(self.aux_weights):
            raise RuntimeError(
                f"Number of aux predictions ({len(aux_preds)}) does not "
                f"match number of aux weights ({len(self.aux_weights)}). "
                f"This indicates a model/config divergence.")
        for k_idx, (ap, w_k) in enumerate(zip(aux_preds, self.aux_weights)):
            view_k = k_idx + 1   # k=1..n_views-1 in label channel space
            lbl_k = label_all[:, view_k]
            wm_k = wmap_all[:, view_k] if wmap_all is not None else None
            aux_l = self._compute_loss_fp32(
                self.aux_inner_loss, ap, lbl_k, weight_map=wm_k)
            total = total + w_k * aux_l
            if breakdown is not None:
                # Log the RAW aux loss (un-weighted) so the user can see
                # whether the aux head is actually being learnt; the
                # contribution to the optimiser is ``w_k * L_aux``.
                breakdown[f"L_aux_{view_k}"] = float(aux_l.detach().item())
                breakdown[f"w_aux_{view_k}"] = float(w_k)
        if breakdown is not None:
            breakdown["L_total"] = float(total.detach().item())
        return total

    def _squeeze_2_5d(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        wmap: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Collapse the C_res (multi-FOV) axis for 2.5D mode.

        Input shapes (post-augment, post-crop):
          image: (B, C_res, D, H, W) → (B, C_res * D, H, W)
              Each FOV view contributes D channels stacked contiguously.
              View 0 occupies channels [0, D); view i occupies
              [i*D, (i+1)*D). The 2D model receives this layout directly;
              ``MultiStemProj`` knows the chunk boundaries via the
              ``in_ch_per_view`` it was built with (= D).
          label: (B, C_res, D, H, W) raw int labels → (B, D, H, W)
              Only view 0 (the 1× FOV, true geometry) is used for
              supervision and metrics. Wider-FOV labels would correspond
              to a different physical slab and have no defined alignment
              with the 1× output.
          wmap : (B, C_res, D, H, W) or None → (B, D, H, W)
              Same view-0 selection rule as label, for the same reason.

        ``D becomes the slice axis`` of ``SliceChannelLoss`` and the
        per-view input-channel slab of the 2D model.

        With ``C_res == 1`` (legacy single-FOV) the reshape collapses to
        the previous ``squeeze(1)`` and is bit-identical to pre-multi-FOV
        behaviour.
        """
        if image.ndim != 5:
            raise ValueError(
                f"2.5D _squeeze expects rank-5 image (B, C_res, D, H, W); "
                f"got shape={tuple(image.shape)}")
        if label.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"image / label batch+C_res mismatch: image="
                f"{tuple(image.shape)}, label={tuple(label.shape)}")
        B, C_res, D, H, W = image.shape
        # (B, C_res, D, H, W) → (B, C_res*D, H, W). ``contiguous`` keeps
        # the memory layout linear so downstream chunking (MultiStemProj
        # uses torch.split) is a zero-copy view.
        image = image.reshape(B, C_res * D, H, W).contiguous()
        # Supervision uses ONLY the 1× FOV view (channel 0). Wider-FOV
        # views feed the encoder but their resampled labels would not
        # be voxel-aligned with the model output's slab.
        label = label[:, 0]
        if wmap is not None:
            wmap = wmap[:, 0]
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