"""Per-class binary loss functions for 2D/3D segmentation.

All losses operate in per-class independent sigmoid mode:
  pred:   (B, num_fg, *spatial) — raw logits, one channel per foreground class
  target: (B, num_fg, *spatial) — binary masks, one channel per foreground class

Each channel is an independent binary segmentation problem.
Background is implicit (not predicted).

Optional spatial weight map:
  weight_map: (B, 1, *spatial) — per-voxel weight, broadcasts over channels.

Provides:
  - BinaryDiceLoss       (per-sample or batch-dice; optional ignore_empty)
  - BCELoss              (with consistent class-weight normalization)
  - BinaryFocalLoss      (proper alpha_t pos/neg balancing)
  - BinaryTverskyLoss    (per-sample or batch mode)
  - CompoundLoss         (weighted sum of losses)
  - DeepSupervisionLoss  (downsamples target to each pred scale by default)
  - build_loss(cfg)      (factory)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import LossConfig

logger = logging.getLogger(__name__)

EPS = 1e-8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _check_inputs(
    pred: torch.Tensor, target: torch.Tensor, weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Validate shapes and cast target to pred's dtype (AMP-safe)."""
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target shape mismatch: "
            f"{tuple(pred.shape)} vs {tuple(target.shape)}")
    if weight_map is not None:
        expected = (pred.shape[0], 1) + tuple(pred.shape[2:])
        if tuple(weight_map.shape) != expected:
            raise ValueError(
                f"weight_map must have shape {expected}, "
                f"got {tuple(weight_map.shape)}")
    if target.dtype != pred.dtype:
        target = target.to(pred.dtype)
    return target


def _register_class_weights(
    module: nn.Module, class_weights: Optional[Sequence[float]]) -> None:
    """Register class_weights as a buffer (or None) for consistent state_dict."""
    if class_weights:
        module.register_buffer(
            "class_weights",
            torch.tensor(list(class_weights), dtype=torch.float32))
    else:
        module.register_buffer("class_weights", None)


def _weighted_mean_over_classes(
    per_class: torch.Tensor, class_weights: Optional[torch.Tensor]) -> torch.Tensor:
    """Weighted mean over the last (class) dim.

    per_class: (..., C)  →  (...,)
    """
    if class_weights is None:
        return per_class.mean(dim=-1)
    w = class_weights.to(per_class.device).to(per_class.dtype)
    return (per_class * w).sum(dim=-1) / w.sum().clamp(min=EPS)


def _weighted_voxel_mean(
    per_voxel: torch.Tensor, weight_map: Optional[torch.Tensor], class_weights: Optional[torch.Tensor]) -> torch.Tensor:
    """Normalized weighted mean of a per-voxel loss tensor.

    per_voxel:     (B, C, *spatial)
    weight_map:    (B, 1, *spatial) or None
    class_weights: (C,) or None

    Returns a scalar = sum(loss * w) / sum(w), so the loss magnitude is
    invariant to the total weight.
    """
    if weight_map is None and class_weights is None:
        return per_voxel.mean()

    weight = per_voxel.new_ones(per_voxel.shape)
    if weight_map is not None:
        weight = weight * weight_map  # broadcast (B,1,*) → (B,C,*)
    if class_weights is not None:
        cw = class_weights.to(per_voxel.device).to(per_voxel.dtype)
        cw_shape = [1, -1] + [1] * (per_voxel.ndim - 2)
        weight = weight * cw.reshape(cw_shape)

    return (per_voxel * weight).sum() / weight.sum().clamp(min=EPS)


def _interp_mode_smooth(spatial_ndim: int) -> str:
    return {1: "linear", 2: "bilinear", 3: "trilinear"}[spatial_ndim]


# ---------------------------------------------------------------------------
# Binary Dice Loss
# ---------------------------------------------------------------------------
class BinaryDiceLoss(nn.Module):
    """Per-channel binary Dice loss using sigmoid.

    Args:
        smooth: smoothing term added to numerator and denominator.
        squared: V-Net style squared denominator (p**2 in denom). Target
            is binary so t**2 == t; only p is squared.
        batch_dice: if True, aggregate TP / denom across batch+spatial
            before the division. More stable for patches with sparse or
            empty foreground (nnU-Net default style).
        ignore_empty: per-sample mode only — exclude classes with no GT
            in the current sample from the mean. Prevents "correctly
            predicting empty" (dice≈1) from masking errors on other
            classes.
        class_weights: per-class weights for the final class-level mean.
    """

    def __init__(
        self,
        smooth: float = 1e-5,
        squared: bool = False,
        batch_dice: bool = False,
        ignore_empty: bool = False,
        class_weights: Optional[Sequence[float]] = None):
        super().__init__()
        self.smooth = smooth
        self.squared = squared
        self.batch_dice = batch_dice
        # ignore_empty only meaningful in per-sample mode
        self.ignore_empty = ignore_empty and not batch_dice
        _register_class_weights(self, class_weights)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        target = _check_inputs(pred, target, weight_map)
        pred_prob = torch.sigmoid(pred)
        B, C = pred.shape[:2]
        p = pred_prob.reshape(B, C, -1)
        t = target.reshape(B, C, -1)
        p_den = p * p if self.squared else p  # t is binary → t**2 == t

        sum_dims: Tuple[int, ...] = (0, 2) if self.batch_dice else (2,)

        if weight_map is not None:
            # weight_map acts as a SUMMATION weight: each voxel's contribution
            # to numerator and denominator is scaled by w consistently.
            w = weight_map.reshape(B, 1, -1)  # broadcasts over C
            intersection = (w * p * t).sum(dim=sum_dims)
            denom = (w * p_den).sum(dim=sum_dims) + (w * t).sum(dim=sum_dims)
        else:
            intersection = (p * t).sum(dim=sum_dims)
            denom = p_den.sum(dim=sum_dims) + t.sum(dim=sum_dims)

        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        # dice shape: (C,) in batch_dice mode, (B, C) in per-sample mode

        if self.batch_dice:
            return 1.0 - _weighted_mean_over_classes(dice, self.class_weights)

        per_class_loss = 1.0 - dice  # (B, C)

        if self.ignore_empty:
            has_gt = (t.sum(dim=2) > 0).to(per_class_loss.dtype)  # (B, C)
            if self.class_weights is not None:
                cw = self.class_weights.to(per_class_loss.device).to(
                    per_class_loss.dtype)
                w_cls = has_gt * cw.unsqueeze(0)
            else:
                w_cls = has_gt
            num = (per_class_loss * w_cls).sum(dim=1)
            den = w_cls.sum(dim=1).clamp(min=EPS)
            return (num / den).mean()

        return _weighted_mean_over_classes(
            per_class_loss, self.class_weights).mean()


# ---------------------------------------------------------------------------
# Binary Cross-Entropy Loss
# ---------------------------------------------------------------------------
class BCELoss(nn.Module):
    """Per-channel binary cross-entropy with logits.

    class_weights are applied in a normalized weighted-mean fashion so the
    loss magnitude stays comparable to the unweighted case (important when
    combined with Dice in CompoundLoss).
    """

    def __init__(self, class_weights: Optional[Sequence[float]] = None):
        super().__init__()
        _register_class_weights(self, class_weights)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        target = _check_inputs(pred, target, weight_map)
        per_voxel = F.binary_cross_entropy_with_logits(
            pred, target, reduction="none")
        return _weighted_voxel_mean(per_voxel, weight_map, self.class_weights)


# ---------------------------------------------------------------------------
# Binary Focal Loss
# ---------------------------------------------------------------------------
class BinaryFocalLoss(nn.Module):
    """Per-channel binary focal loss with proper positive/negative balancing.

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where
        alpha_t = alpha         for positives (target=1)
                = 1 - alpha     for negatives (target=0)

    The original code used alpha_t = alpha for both classes, which amounts
    to a constant scaling and provides no pos/neg balancing.
    """

    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, class_weights: Optional[Sequence[float]] = None):
        super().__init__()
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.gamma = gamma
        _register_class_weights(self, class_weights)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target = _check_inputs(pred, target, weight_map)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        # p_t = exp(-bce) — numerically stable, saves one sigmoid call.
        pt = torch.exp(-bce)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        per_voxel = alpha_t * (1.0 - pt).pow(self.gamma) * bce
        return _weighted_voxel_mean(per_voxel, weight_map, self.class_weights)


# ---------------------------------------------------------------------------
# Binary Tversky Loss
# ---------------------------------------------------------------------------
class BinaryTverskyLoss(nn.Module):
    """Per-channel binary Tversky loss (asymmetric Dice).

        TI = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)

    Default alpha=0.3, beta=0.7 emphasizes recall (penalizes FN more
    than FP). Useful for under-segmentation-sensitive tasks (small lesions,
    thin vessels).
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-5,
        batch_dice: bool = False,
        class_weights: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.batch_dice = batch_dice
        _register_class_weights(self, class_weights)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target = _check_inputs(pred, target, weight_map)
        pred_prob = torch.sigmoid(pred)
        B, C = pred.shape[:2]
        p = pred_prob.reshape(B, C, -1)
        t = target.reshape(B, C, -1)
        sum_dims: Tuple[int, ...] = (0, 2) if self.batch_dice else (2,)

        if weight_map is not None:
            w = weight_map.reshape(B, 1, -1)
            tp = (w * p * t).sum(dim=sum_dims)
            fp = (w * p * (1 - t)).sum(dim=sum_dims)
            fn = (w * (1 - p) * t).sum(dim=sum_dims)
        else:
            tp = (p * t).sum(dim=sum_dims)
            fp = (p * (1 - t)).sum(dim=sum_dims)
            fn = ((1 - p) * t).sum(dim=sum_dims)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        per_class_loss = 1.0 - tversky  # (C,) or (B, C)

        if self.batch_dice:
            return _weighted_mean_over_classes(per_class_loss, self.class_weights)
        return _weighted_mean_over_classes(
            per_class_loss, self.class_weights
        ).mean()


# ---------------------------------------------------------------------------
# Compound Loss
# ---------------------------------------------------------------------------
class CompoundLoss(nn.Module):
    """Weighted sum of multiple losses."""

    def __init__(
        self, losses: Sequence[nn.Module], weights: Sequence[float]
    ):
        super().__init__()
        if len(losses) != len(weights):
            raise ValueError(
                f"losses and weights length mismatch: "
                f"{len(losses)} vs {len(weights)}"
            )
        self.losses = nn.ModuleList(losses)
        self.weights = list(weights)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        total = pred.new_zeros(())
        for fn, w in zip(self.losses, self.weights):
            total = total + w * fn(pred, target, weight_map=weight_map)
        return total


# ---------------------------------------------------------------------------
# Deep Supervision Wrapper
# ---------------------------------------------------------------------------
class DeepSupervisionLoss(nn.Module):
    """Multi-scale loss for deep supervision.

    Default behavior (nnU-Net-style):
      - Downsample target (and weight_map) to each pred's resolution with
        nearest-neighbor. Memory-efficient: no need to upsample low-res
        logits back to full resolution.
      - Weights are normalized to sum to 1 so the total loss magnitude
        matches single-scale training when the main output has the
        largest weight.

    Args:
        base_loss: loss to apply at each scale.
        weights: one weight per prediction scale, highest-resolution first
                 (i.e. the full-res main output gets weights[0]).
        normalize_weights: if True, divide weights by their sum.
        upsample_pred: if True, upsample preds to target resolution
                       instead of downsampling target. Higher memory;
                       keeps smooth label gradients at every scale.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        weights: Sequence[float],
        normalize_weights: bool = True,
        upsample_pred: bool = False,
    ):
        super().__init__()
        self.base_loss = base_loss
        w = list(weights)
        if normalize_weights:
            s = sum(w)
            if s <= 0:
                raise ValueError(f"DS weights must sum to positive, got {s}")
            w = [wi / s for wi in w]
        self.weights = w
        self.upsample_pred = upsample_pred

    def forward(
        self,
        preds: Union[torch.Tensor, List[torch.Tensor]],
        target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Bypass: single tensor (e.g. DS disabled upstream, or at inference).
        if isinstance(preds, torch.Tensor):
            return self.base_loss(preds, target, weight_map=weight_map)

        if len(preds) != len(self.weights):
            raise ValueError(
                f"Number of predictions ({len(preds)}) must match number "
                f"of DS weights ({len(self.weights)})"
            )

        total = preds[0].new_zeros(())
        for w, pred in zip(self.weights, preds):
            tgt_i, wm_i = target, weight_map
            if pred.shape[2:] != target.shape[2:]:
                spatial_ndim = pred.ndim - 2
                if self.upsample_pred:
                    pred = F.interpolate(
                        pred,
                        size=target.shape[2:],
                        mode=_interp_mode_smooth(spatial_ndim),
                        align_corners=False,
                    )
                else:
                    tgt_i = F.interpolate(
                        target, size=pred.shape[2:], mode="nearest"
                    )
                    if weight_map is not None:
                        wm_i = F.interpolate(
                            weight_map, size=pred.shape[2:], mode="nearest"
                        )
            total = total + w * self.base_loss(pred, tgt_i, weight_map=wm_i)
        return total


# ---------------------------------------------------------------------------
# Loss factory
# ---------------------------------------------------------------------------
def _build_dice(cfg: LossConfig, cw: Optional[List[float]]) -> BinaryDiceLoss:
    return BinaryDiceLoss(
        smooth=cfg.dice_smooth,
        squared=cfg.dice_squared,
        batch_dice=getattr(cfg, "batch_dice", False),
        ignore_empty=getattr(cfg, "ignore_empty", False),
        class_weights=cw,
    )


def _build_bce(cfg: LossConfig, cw: Optional[List[float]]) -> BCELoss:
    return BCELoss(class_weights=cw)


def _build_focal(cfg: LossConfig, cw: Optional[List[float]]) -> BinaryFocalLoss:
    return BinaryFocalLoss(
        alpha=cfg.focal_alpha,
        gamma=cfg.focal_gamma,
        class_weights=cw,
    )


def _build_tversky(
    cfg: LossConfig, cw: Optional[List[float]]
) -> BinaryTverskyLoss:
    return BinaryTverskyLoss(
        alpha=cfg.tversky_alpha,
        beta=cfg.tversky_beta,
        smooth=cfg.dice_smooth,
        batch_dice=getattr(cfg, "batch_dice", False),
        class_weights=cw,
    )


_SINGLE_BUILDERS = {
    "dice": _build_dice,
    "bce": _build_bce,
    "focal": _build_focal,
    "tversky": _build_tversky,
}

_COMPOUND_BUILDERS = {
    "dice_bce": (_build_dice, _build_bce),
    "dice_focal": (_build_dice, _build_focal),
    "dice_tversky": (_build_dice, _build_tversky),
    "focal_tversky": (_build_focal, _build_tversky),
}


def _compound_weights(cfg: LossConfig, n: int) -> List[float]:
    ws = list(getattr(cfg, "compound_weights", None) or [])
    if len(ws) >= n:
        return ws[:n]
    logger.warning(
        "compound_weights has %d entries, need %d; defaulting missing to 1.0",
        len(ws),
        n,
    )
    return (ws + [1.0] * n)[:n]


class MultiResolutionLoss(nn.Module):
    """Wrapper that handles multi-resolution label format.

    When multi-resolution input is enabled:
      - Model output: (B, num_fg * C_res, D, H, W)
      - Label:        (B, C_res, D, H, W) with raw integer labels per channel

    This wrapper:
      1. Splits model output into C_res groups of num_fg channels
      2. Converts each label channel to per-fg binary masks (preprocess_label)
      3. Computes base_loss for each resolution independently
      4. Returns the average loss across resolutions

    Args:
        base_loss: The underlying loss function (e.g., CompoundLoss of Dice+BCE).
        num_fg_classes: Number of foreground classes.
        num_res: Number of resolution scales (C_res).
        label_values: [bg, fg1, fg2, ...] for preprocess_label.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        num_fg_classes: int,
        num_res: int,
        label_values: List[int],
    ):
        super().__init__()
        self.base_loss = base_loss
        self.num_fg = num_fg_classes
        self.num_res = num_res
        self.label_values = label_values
        self.fg_values = label_values[1:]  # exclude background

    def forward(
        self,
        pred: torch.Tensor,
        label_raw: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute loss across all resolutions.

        Args:
            pred: (B, num_fg * C_res, D, H, W) model logits.
            label_raw: (B, C_res, D, H, W) raw integer labels per resolution.
            weight_map: (B, C_res, D, H, W) per-res spatial weights, or None.

        Returns:
            Scalar loss averaged over resolutions.
        """
        total = pred.new_zeros(())

        for r in range(self.num_res):
            pred_r = pred[:, r * self.num_fg:(r + 1) * self.num_fg]
            lbl_r = label_raw[:, r]
            target_r = self._label_to_binary(lbl_r)

            # Per-resolution weight_map: (B, D, H, W) → (B, 1, D, H, W)
            wm_r = None
            if weight_map is not None:
                wm_r = weight_map[:, r:r + 1]  # (B, 1, D, H, W)

            total = total + self.base_loss(pred_r, target_r, weight_map=wm_r)

        return total / self.num_res

    def _label_to_binary(self, label: torch.Tensor) -> torch.Tensor:
        """Convert integer label (B, D, H, W) to binary masks (B, num_fg, D, H, W).

        Vectorized on GPU — no CPU round-trip.
        """
        # fg_values as tensor: (num_fg,)
        fg = torch.tensor(self.fg_values, device=label.device, dtype=label.dtype)
        # label: (B, D, H, W) → (B, 1, D, H, W)
        # fg:    (num_fg,)     → (1, num_fg, 1, 1, 1)
        label_exp = label.unsqueeze(1)
        fg_exp = fg.reshape(1, -1, *([1] * (label.ndim - 1)))
        return (label_exp == fg_exp).float()


def build_loss(cfg: LossConfig) -> nn.Module:
    """Build loss function from config.

    All losses use per-class independent sigmoid (binary mode).
    """
    cw = list(cfg.class_weights) if cfg.class_weights else None
    name = cfg.name.lower()

    if name in _SINGLE_BUILDERS:
        return _SINGLE_BUILDERS[name](cfg, cw)

    if name in _COMPOUND_BUILDERS:
        builders = _COMPOUND_BUILDERS[name]
        components = [b(cfg, cw) for b in builders]
        weights = _compound_weights(cfg, len(components))
        return CompoundLoss(components, weights)

    supported = sorted(
        list(_SINGLE_BUILDERS.keys()) + list(_COMPOUND_BUILDERS.keys())
    )
    raise ValueError(f"Unknown loss: {cfg.name!r}. Supported: {supported}")