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
# Generalized Dice Loss  (Sudre et al., DLMIA 2017)
# ---------------------------------------------------------------------------
class GeneralizedDiceLoss(nn.Module):
    """Generalized Dice Loss with automatic inverse-volume class weighting.

    Reference: Sudre et al., "Generalised Dice overlap as a deep learning
    loss function for highly unbalanced segmentations." DLMIA 2017.

        w_c  = 1 / (Σ t_c)^2                  (weight_type == "square")
        GDL  = 1 - 2 * Σ_c w_c * TP_c / Σ_c w_c * (P_c + T_c)

    The class weights compensate for volume imbalance automatically, making
    this loss a strong default when `class_weights` are not tuned by hand
    and the dataset has rare-class voxels.

    Args:
        smooth: numerator/denominator smoothing.
        batch_dice: aggregate over batch+spatial before dividing (nnU-Net
            default style, more stable for sparse patches).
        weight_type: "square" (paper), "simple" (w=1/Σt), or "uniform"
            (disables volume-based weighting; identical to batch_dice mode
            of BinaryDiceLoss when class_weights is None).
        class_weights: optional extra per-class multiplier applied AFTER
            the volume-based weight.
        w_max: clamp for 1/volume to avoid explosion on empty classes.
    """

    def __init__(
        self,
        smooth: float = 1e-5,
        batch_dice: bool = True,
        weight_type: str = "square",
        class_weights: Optional[Sequence[float]] = None,
        w_max: float = 1e5):
        super().__init__()
        if weight_type not in ("square", "simple", "uniform"):
            raise ValueError(
                f"weight_type must be one of square/simple/uniform, "
                f"got {weight_type!r}")
        self.smooth = smooth
        self.batch_dice = batch_dice
        self.weight_type = weight_type
        self.w_max = w_max
        _register_class_weights(self, class_weights)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        target = _check_inputs(pred, target, weight_map)
        pred_prob = torch.sigmoid(pred)
        B, C = pred.shape[:2]
        p = pred_prob.reshape(B, C, -1)
        t = target.reshape(B, C, -1)
        sum_dims: Tuple[int, ...] = (0, 2) if self.batch_dice else (2,)

        if weight_map is not None:
            w_vox = weight_map.reshape(B, 1, -1)
            t_vol = (w_vox * t).sum(dim=sum_dims)
            tp = (w_vox * p * t).sum(dim=sum_dims)
            denom = (w_vox * (p + t)).sum(dim=sum_dims)
        else:
            t_vol = t.sum(dim=sum_dims)
            tp = (p * t).sum(dim=sum_dims)
            denom = (p + t).sum(dim=sum_dims)

        # Volume-based class weights. Shapes: (C,) if batch_dice else (B, C).
        t_safe = t_vol.clamp(min=EPS)
        if self.weight_type == "square":
            wc = 1.0 / (t_safe * t_safe)
        elif self.weight_type == "simple":
            wc = 1.0 / t_safe
        else:  # uniform
            wc = torch.ones_like(t_safe)
        wc = wc.clamp(max=self.w_max)

        if self.class_weights is not None:
            cw = self.class_weights.to(wc.device).to(wc.dtype)
            wc = wc * cw  # broadcast over class axis (last)

        # Weighted aggregate along the class axis (last dim).
        num = (wc * tp).sum(dim=-1)
        den = (wc * denom).sum(dim=-1)
        gdl = 1.0 - (2.0 * num + self.smooth) / (den + self.smooth)
        # () if batch_dice else (B,)
        return gdl if gdl.ndim == 0 else gdl.mean()


# ---------------------------------------------------------------------------
# Focal Tversky Loss  (Abraham & Khan, ISBI 2019)
# ---------------------------------------------------------------------------
class BinaryFocalTverskyLoss(nn.Module):
    """Focal Tversky Loss — amplifies hard (low-TI) classes.

    Reference: Abraham & Khan, "A novel focal tversky loss function with
    improved attention U-Net for lesion segmentation." ISBI 2019.

        TI_c = (TP + s) / (TP + α FP + β FN + s)
        FTL  = mean_c( (1 - TI_c)^gamma )

    With α<β (default 0.3/0.7) the loss emphasises recall (penalises FN),
    and gamma>1 further concentrates gradient on hard classes. Empirically
    strong for small-lesion / thin-structure tasks.

    Note on gamma convention: our formulation is ``(1 - TI)^gamma`` with
    gamma >= 1 → harder classes dominate. Abraham & Khan's original paper
    writes ``(1 - TI)^(1/γ_paper)`` with γ_paper ∈ [1, 3]; our ``gamma``
    corresponds to ``1 / γ_paper`` of the paper inverted → the default
    1.333 (= 4/3) matches γ_paper = 0.75 which the authors use in their
    experiments.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 4.0 / 3.0,
        smooth: float = 1e-5,
        batch_dice: bool = False,
        class_weights: Optional[Sequence[float]] = None):
        super().__init__()
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.batch_dice = batch_dice
        _register_class_weights(self, class_weights)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
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
            tp + self.alpha * fp + self.beta * fn + self.smooth)
        # Clamp (1 - TI) into [0, 1] for numerical safety under fractional powers.
        focal = (1.0 - tversky).clamp(min=0.0, max=1.0).pow(self.gamma)

        if self.batch_dice:
            return _weighted_mean_over_classes(focal, self.class_weights)
        return _weighted_mean_over_classes(focal, self.class_weights).mean()


# ---------------------------------------------------------------------------
# Lovász-Hinge Loss  (Berman et al., CVPR 2018)
# ---------------------------------------------------------------------------
def _lovasz_grad_batched(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Gradient of the Lovász extension of the Jaccard loss — vectorised.

    gt_sorted: (..., L) binary ground-truth, sorted by descending error.
    Returns a tensor of the same shape whose dot-product with ReLU(errors)
    yields the Lovász-hinge loss for each leading slice independently.
    """
    # Σ t for each leading slice (keepdim for broadcasting).
    gts = gt_sorted.sum(dim=-1, keepdim=True)
    intersection = gts - gt_sorted.cumsum(dim=-1)
    union = gts + (1.0 - gt_sorted).cumsum(dim=-1)
    jaccard = 1.0 - intersection / union.clamp(min=EPS)
    # Difference along L (the step-function gradient of Lovász extension).
    if jaccard.shape[-1] > 1:
        shifted = jaccard[..., 1:] - jaccard[..., :-1]
        jaccard = torch.cat([jaccard[..., :1], shifted], dim=-1)
    return jaccard


class LovaszHingeLoss(nn.Module):
    """Per-class binary Lovász-Hinge loss — a direct IoU surrogate.

    Reference: Berman et al., "The Lovász-Softmax Loss: a tractable
    surrogate for the optimization of the intersection-over-union measure
    in neural networks." CVPR 2018.

    The Lovász-Hinge is the piecewise-linear convex surrogate of the
    Jaccard (IoU) loss. Unlike Dice/BCE, it directly minimises IoU by
    sorting per-voxel hinge errors and integrating the Jaccard gradient.

    Operates on RAW LOGITS (no sigmoid); targets are binary {0, 1}.

    weight_map is applied heuristically by multiplying the non-negative
    portion of hinge errors by the per-voxel weight before sort — retains
    the sort-based gradient structure while emphasising high-weight
    regions. Use sparingly; for strict theoretical fidelity, disable
    weight_map for this loss.

    Args:
        per_sample: if True, loss is averaged over (B, C) independent
            sorts; if False, concatenate across batch per channel before
            sort (batch-level Lovász — smoother for tiny patches).
        class_weights: optional weighted mean across channels.
    """

    def __init__(
        self, per_sample: bool = True,
        class_weights: Optional[Sequence[float]] = None):
        super().__init__()
        self.per_sample = per_sample
        _register_class_weights(self, class_weights)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        target = _check_inputs(pred, target, weight_map)
        B, C = pred.shape[:2]
        logits = pred.reshape(B, C, -1)   # operate on logits
        t = target.reshape(B, C, -1)

        # Hinge error in logit space: e_i = max(0, 1 - s_i * z_i)  where
        # s_i = 2 t_i - 1 ∈ {-1, +1}. We keep the raw (1 - s*z) because
        # Lovász requires the signed margin before ReLU.
        signed = 2.0 * t - 1.0
        errors = 1.0 - signed * logits

        if weight_map is not None:
            w_vox = weight_map.reshape(B, 1, -1).clamp(min=0)
            # Only scale the penalised side; negative margins mean correct
            # with margin, which should remain safely zero under ReLU.
            errors = torch.where(
                errors > 0, errors * w_vox, errors)

        if self.per_sample:
            # Sort along spatial axis (dim=-1), gather targets with same perm.
            err_sorted, perm = torch.sort(errors, dim=-1, descending=True)
            gt_sorted = t.gather(dim=-1, index=perm)
            grad = _lovasz_grad_batched(gt_sorted)
            per_class = (F.relu(err_sorted) * grad).sum(dim=-1)  # (B, C)
            return _weighted_mean_over_classes(
                per_class, self.class_weights).mean()

        # Batch-level Lovász: reshape (B*L) per channel, sort once.
        errors_bc = errors.permute(1, 0, 2).reshape(C, -1)  # (C, B*L)
        t_bc = t.permute(1, 0, 2).reshape(C, -1)
        err_sorted, perm = torch.sort(errors_bc, dim=-1, descending=True)
        gt_sorted = t_bc.gather(dim=-1, index=perm)
        grad = _lovasz_grad_batched(gt_sorted)
        per_class = (F.relu(err_sorted) * grad).sum(dim=-1)  # (C,)
        return _weighted_mean_over_classes(per_class, self.class_weights)


# ---------------------------------------------------------------------------
# Soft clDice  (Shit et al., CVPR 2021) — topology-preserving
# ---------------------------------------------------------------------------
def _soft_erode(x: torch.Tensor, spatial_ndim: int) -> torch.Tensor:
    """Differentiable morphological erosion via negated max-pool (kernel=3)."""
    if spatial_ndim == 3:
        return -F.max_pool3d(-x, kernel_size=3, stride=1, padding=1)
    if spatial_ndim == 2:
        return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
    raise ValueError(f"spatial_ndim must be 2 or 3, got {spatial_ndim}")


def _soft_dilate(x: torch.Tensor, spatial_ndim: int) -> torch.Tensor:
    """Differentiable morphological dilation via max-pool (kernel=3)."""
    if spatial_ndim == 3:
        return F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
    if spatial_ndim == 2:
        return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    raise ValueError(f"spatial_ndim must be 2 or 3, got {spatial_ndim}")


def _soft_skeletonize(
    img: torch.Tensor, n_iter: int, spatial_ndim: int) -> torch.Tensor:
    """Iterative soft skeletonization (Shit et al., Algorithm 1).

    img is expected in [0, 1] (e.g., sigmoid probabilities or binary mask).
    Returns a soft skeleton of the same shape.
    """
    def _open(y: torch.Tensor) -> torch.Tensor:
        return _soft_dilate(_soft_erode(y, spatial_ndim), spatial_ndim)

    skel = F.relu(img - _open(img))
    for _ in range(n_iter):
        img = _soft_erode(img, spatial_ndim)
        delta = F.relu(img - _open(img))
        # (1 - skel) gates to prevent re-counting voxels already in skel.
        skel = skel + (1.0 - skel).clamp(min=0.0) * delta
    return skel


class SoftCLDiceLoss(nn.Module):
    """Soft centerline (clDice) loss — topology-preserving.

    Reference: Shit et al., "clDice — a Novel Topology-Preserving Loss
    Function for Tubular Structure Segmentation." CVPR 2021.

    Measures overlap between the SOFT skeletons of prediction and target:

        Tprec = |skel_P ∩ T| / |skel_P|
        Tsens = |skel_T ∩ P| / |skel_T|
        clDice = 2 * Tprec * Tsens / (Tprec + Tsens)
        loss   = 1 - clDice

    Best used in compound with Dice (paper's recommendation):

        L = (1 - α) * dice + α * (1 - clDice),    α ∈ [0, 1]

    Use the `dice_cldice` builder for this canonical recipe.

    `weight_map` is accepted for API uniformity but ignored: clDice is a
    topological metric over whole-structure skeletons; voxel reweighting
    does not have a consistent interpretation within it.
    """

    def __init__(
        self,
        iter_: int = 3,
        smooth: float = 1.0,
        class_weights: Optional[Sequence[float]] = None):
        super().__init__()
        if iter_ < 1:
            raise ValueError(f"iter_ must be >= 1, got {iter_}")
        self.iter = iter_
        self.smooth = smooth
        _register_class_weights(self, class_weights)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        # weight_map intentionally ignored — see class docstring.
        del weight_map
        _check_inputs(pred, target)
        spatial_ndim = pred.ndim - 2
        if spatial_ndim not in (2, 3):
            raise ValueError(
                f"SoftCLDiceLoss expects 2D or 3D spatial input; got "
                f"pred.ndim={pred.ndim}")
        target = target.to(pred.dtype)
        pred_prob = torch.sigmoid(pred)

        skel_pred = _soft_skeletonize(pred_prob, self.iter, spatial_ndim)
        skel_t = _soft_skeletonize(target, self.iter, spatial_ndim)

        B, C = pred.shape[:2]
        sp = skel_pred.reshape(B, C, -1)
        st = skel_t.reshape(B, C, -1)
        p = pred_prob.reshape(B, C, -1)
        t = target.reshape(B, C, -1)

        tprec = ((sp * t).sum(dim=-1) + self.smooth) / (
            sp.sum(dim=-1) + self.smooth)
        tsens = ((st * p).sum(dim=-1) + self.smooth) / (
            st.sum(dim=-1) + self.smooth)
        cldice = 2.0 * tprec * tsens / (tprec + tsens + self.smooth)
        per_class_loss = 1.0 - cldice  # (B, C)
        return _weighted_mean_over_classes(
            per_class_loss, self.class_weights).mean()


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


def _build_gdl(
    cfg: LossConfig, cw: Optional[List[float]]) -> GeneralizedDiceLoss:
    return GeneralizedDiceLoss(
        smooth=cfg.dice_smooth,
        batch_dice=getattr(cfg, "batch_dice", True),
        weight_type=getattr(cfg, "gdl_weight_type", "square"),
        class_weights=cw,
        w_max=getattr(cfg, "gdl_w_max", 1e5),
    )


def _build_focal_tversky(
    cfg: LossConfig, cw: Optional[List[float]]) -> BinaryFocalTverskyLoss:
    return BinaryFocalTverskyLoss(
        alpha=cfg.tversky_alpha,
        beta=cfg.tversky_beta,
        gamma=getattr(cfg, "focal_tversky_gamma", 4.0 / 3.0),
        smooth=cfg.dice_smooth,
        batch_dice=getattr(cfg, "batch_dice", False),
        class_weights=cw,
    )


def _build_lovasz(
    cfg: LossConfig, cw: Optional[List[float]]) -> LovaszHingeLoss:
    return LovaszHingeLoss(
        per_sample=getattr(cfg, "lovasz_per_sample", True),
        class_weights=cw,
    )


def _build_cldice(
    cfg: LossConfig, cw: Optional[List[float]]) -> SoftCLDiceLoss:
    return SoftCLDiceLoss(
        iter_=getattr(cfg, "cldice_iter", 3),
        smooth=getattr(cfg, "cldice_smooth", 1.0),
        class_weights=cw,
    )


_SINGLE_BUILDERS = {
    "dice": _build_dice,
    "bce": _build_bce,
    "focal": _build_focal,
    "tversky": _build_tversky,
    # New (Round "high-quality losses")
    "gdl": _build_gdl,
    "focal_tversky": _build_focal_tversky,
    "lovasz": _build_lovasz,
    "cldice": _build_cldice,
}

_COMPOUND_BUILDERS = {
    "dice_bce": (_build_dice, _build_bce),
    "dice_focal": (_build_dice, _build_focal),
    "dice_tversky": (_build_dice, _build_tversky),
    # New compounds.
    # NOTE: "focal_tversky" by itself is a single loss now (the standalone
    # Focal-Tversky); to combine Focal + Tversky use "focal_plus_tversky".
    "focal_plus_tversky": (_build_focal, _build_tversky),
    "dice_cldice": (_build_dice, _build_cldice),          # Shit et al. recipe
    "dice_focal_tversky": (_build_dice, _build_focal_tversky),
    "dice_lovasz": (_build_dice, _build_lovasz),
    "bce_lovasz": (_build_bce, _build_lovasz),
    "gdl_bce": (_build_gdl, _build_bce),
    "gdl_focal": (_build_gdl, _build_focal),
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

    def split_for_metrics(
        self, pred: torch.Tensor, label_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert (pred, multi-res raw label) into per-class binary form
        suitable for ``compute_dice_per_class`` / ``dice_batch_stats``.

        Uses only the FIRST resolution (highest fidelity) for metrics —
        consistent with prior trainer behaviour.

        Args:
            pred:      (B, num_fg * C_res, *spatial) logits.
            label_raw: (B, C_res, *spatial) raw integer labels.

        Returns:
            (pred_1x, target_1x): both with shape (B, num_fg, *spatial).
        """
        pred_1x = pred[:, :self.num_fg]
        target_1x = self._label_to_binary(label_raw[:, 0])
        return pred_1x, target_1x


# ---------------------------------------------------------------------------
# 2.5D Slice-Channel Loss Wrapper
# ---------------------------------------------------------------------------
class SliceChannelLoss(nn.Module):
    """Wrapper for the 2.5D patch mode.

    Tensor contracts (after the trainer squeezes ``C_res=1`` away):
      Model output : (B, num_fg * D, H, W) logits.
      Raw label    : (B, D, H, W) integer labels (D slices stacked as channels).
      Weight map   : (B, D, H, W) per-voxel weights, or None.

    For each foreground class ``c ∈ [0, num_fg)``:
      pred_c   = pred[:, c*D:(c+1)*D]                  # (B, D, H, W)
      target_c = (label_raw == fg_values[c]).float()   # (B, D, H, W)

    The base 2D segmentation loss is applied per slice by reshaping these
    to ``(B*D, 1, H, W)`` (one binary 2D problem per slice). The final
    scalar loss is averaged across all foreground classes.

    This matches the documented design: each foreground class becomes its
    own binary segmentation problem at every slice; class_weights from
    LossConfig still control the per-class outer mean.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        num_fg_classes: int,
        num_slices: int,
        label_values: List[int],
    ):
        super().__init__()
        if num_slices < 1:
            raise ValueError(f"num_slices must be >= 1, got {num_slices}")
        self.base_loss = base_loss
        self.num_fg = num_fg_classes
        self.num_slices = num_slices
        self.label_values = label_values
        self.fg_values = label_values[1:]  # exclude background

    # -- internal helper ------------------------------------------------
    def _label_to_binary(self, label_raw: torch.Tensor) -> torch.Tensor:
        """(B, D, H, W) raw labels → (B*D, num_fg, H, W) binary masks.

        Vectorised on-device. The output rank is rank-4 so that the
        underlying base loss can run as a 2D binary segmentation.
        """
        if label_raw.ndim != 4:
            raise ValueError(
                f"SliceChannelLoss expects (B, D, H, W) raw label, "
                f"got rank-{label_raw.ndim}")
        B, D, H, W = label_raw.shape
        if D != self.num_slices:
            raise ValueError(
                f"label slice count {D} != configured num_slices "
                f"{self.num_slices}")
        fg = torch.tensor(self.fg_values,
                          device=label_raw.device, dtype=label_raw.dtype)
        flat = label_raw.reshape(B * D, H, W).unsqueeze(1)         # (B*D, 1, H, W)
        fg_b = fg.reshape(1, -1, 1, 1)                              # (1, num_fg, 1, 1)
        return (flat == fg_b).float()                               # (B*D, num_fg, H, W)

    def _split_pred(self, pred: torch.Tensor) -> torch.Tensor:
        """(B, num_fg*D, H, W) → (B*D, num_fg, H, W)."""
        if pred.ndim != 4:
            raise ValueError(
                f"SliceChannelLoss expects (B, num_fg*D, H, W) pred, "
                f"got rank-{pred.ndim}")
        B, total_c, H, W = pred.shape
        D = self.num_slices
        if total_c != self.num_fg * D:
            raise ValueError(
                f"pred channel count {total_c} != num_fg*D = "
                f"{self.num_fg}*{D} = {self.num_fg * D}")
        # (B, num_fg, D, H, W) → (B, D, num_fg, H, W) → (B*D, num_fg, H, W)
        return (pred.reshape(B, self.num_fg, D, H, W)
                    .permute(0, 2, 1, 3, 4)
                    .reshape(B * D, self.num_fg, H, W))

    @staticmethod
    def _flatten_weight_map(
        weight_map: Optional[torch.Tensor], num_slices: int,
    ) -> Optional[torch.Tensor]:
        """(B, D, H, W) → (B*D, 1, H, W) for base-loss broadcasting."""
        if weight_map is None:
            return None
        if weight_map.ndim != 4:
            raise ValueError(
                f"SliceChannelLoss expects (B, D, H, W) weight_map, "
                f"got rank-{weight_map.ndim}")
        B, D, H, W = weight_map.shape
        if D != num_slices:
            raise ValueError(
                f"weight_map slice count {D} != num_slices {num_slices}")
        return weight_map.reshape(B * D, 1, H, W)

    # -- forward ---------------------------------------------------------
    def forward(
        self,
        pred: torch.Tensor,
        label_raw: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the per-class averaged 2D binary loss across slices."""
        pred_flat = self._split_pred(pred)                  # (B*D, num_fg, H, W)
        target_flat = self._label_to_binary(label_raw)      # (B*D, num_fg, H, W)
        wm_flat = self._flatten_weight_map(weight_map, self.num_slices)

        # Per-class loop — mirrors MultiResolutionLoss but iterates over
        # fg classes instead of resolution scales. We pass single-channel
        # binary tensors per class so that base_loss treats each as a
        # standalone 2D binary segmentation problem (rank-4 input).
        total = pred.new_zeros(())
        for c in range(self.num_fg):
            pred_c = pred_flat[:, c:c + 1]                  # (B*D, 1, H, W)
            target_c = target_flat[:, c:c + 1]              # (B*D, 1, H, W)
            total = total + self.base_loss(pred_c, target_c, weight_map=wm_flat)
        return total / self.num_fg

    def split_for_metrics(
        self, pred: torch.Tensor, label_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return tensors suitable for ``compute_dice_per_class`` / \
        ``dice_batch_stats``.

        For 2.5D the per-slice dice is pooled across (B, D) by collapsing
        them into the leading dimension so existing dice utilities work
        unchanged.

        Args:
            pred:      (B, num_fg * D, H, W) logits.
            label_raw: (B, D, H, W) raw integer labels.

        Returns:
            (pred_flat, target_flat): both (B*D, num_fg, H, W).
        """
        return self._split_pred(pred), self._label_to_binary(label_raw)


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