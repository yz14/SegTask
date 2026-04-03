"""Segmentation loss functions.

All losses expect:
- pred: (B, num_classes, *spatial) — raw logits
- target: (B, num_classes, *spatial) — one-hot encoded labels

Provides:
- DiceLoss: soft Dice loss
- CrossEntropyLoss: pixel-wise cross entropy (from logits)
- FocalLoss: focal loss for class imbalance
- TverskyLoss: asymmetric Dice variant (tunable FP/FN weights)
- CompoundLoss: weighted combination of multiple losses
- Deep supervision wrapper
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import LossConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dice Loss
# ---------------------------------------------------------------------------
class DiceLoss(nn.Module):
    """Soft Dice loss.

    Computes per-class Dice coefficient and returns 1 - mean(Dice).
    Optionally uses squared denominators for sharper gradients.
    """

    def __init__(
        self,
        smooth: float = 1e-5,
        squared: bool = False,
        class_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.smooth = smooth
        self.squared = squared
        if class_weights:
            self.register_buffer(
                "class_weights", torch.tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.class_weights = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, *spatial) logits.
            target: (B, C, *spatial) one-hot labels.
        """
        pred_soft = torch.softmax(pred, dim=1)
        # Flatten spatial dims
        B, C = pred.shape[:2]
        pred_flat = pred_soft.reshape(B, C, -1)
        target_flat = target.reshape(B, C, -1)

        intersection = (pred_flat * target_flat).sum(dim=2)

        if self.squared:
            denom = (pred_flat ** 2).sum(dim=2) + (target_flat ** 2).sum(dim=2)
        else:
            denom = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)

        if self.class_weights is not None:
            w = self.class_weights.to(dice.device)
            dice = dice * w.unsqueeze(0)
            loss = 1.0 - dice.sum(dim=1) / w.sum()
        else:
            loss = 1.0 - dice.mean(dim=1)

        return loss.mean()


# ---------------------------------------------------------------------------
# Cross Entropy Loss
# ---------------------------------------------------------------------------
class CrossEntropyLoss(nn.Module):
    """Pixel-wise cross entropy loss from logits.

    Supports label smoothing and per-class weights.
    Expects one-hot target, converts to class indices internally.
    """

    def __init__(
        self,
        class_weights: Optional[List[float]] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        if class_weights:
            self.register_buffer(
                "class_weights", torch.tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.class_weights = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Convert one-hot target to class indices
        target_idx = target.argmax(dim=1)  # (B, *spatial)

        weight = self.class_weights.to(pred.device) if self.class_weights is not None else None

        loss = F.cross_entropy(
            pred, target_idx,
            weight=weight,
            label_smoothing=self.label_smoothing,
            reduction="mean",
        )
        return loss


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        class_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        if class_weights:
            self.register_buffer(
                "class_weights", torch.tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.class_weights = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_idx = target.argmax(dim=1)  # (B, *spatial)
        ce = F.cross_entropy(pred, target_idx, reduction="none")

        pred_soft = torch.softmax(pred, dim=1)
        # Gather the probability of the correct class
        pt = pred_soft.gather(1, target_idx.unsqueeze(1)).squeeze(1)

        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * ce

        if self.class_weights is not None:
            w = self.class_weights.to(pred.device)
            class_w = w[target_idx]
            loss = loss * class_w

        return loss.mean()


# ---------------------------------------------------------------------------
# Tversky Loss
# ---------------------------------------------------------------------------
class TverskyLoss(nn.Module):
    """Tversky loss: generalized Dice with asymmetric FP/FN weights.

    TI = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
    alpha > beta → penalize FP more; alpha < beta → penalize FN more.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-5,
        class_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        if class_weights:
            self.register_buffer(
                "class_weights", torch.tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.class_weights = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_soft = torch.softmax(pred, dim=1)
        B, C = pred.shape[:2]
        pred_flat = pred_soft.reshape(B, C, -1)
        target_flat = target.reshape(B, C, -1)

        tp = (pred_flat * target_flat).sum(dim=2)
        fp = (pred_flat * (1 - target_flat)).sum(dim=2)
        fn = ((1 - pred_flat) * target_flat).sum(dim=2)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        if self.class_weights is not None:
            w = self.class_weights.to(tversky.device)
            loss = 1.0 - (tversky * w.unsqueeze(0)).sum(dim=1) / w.sum()
        else:
            loss = 1.0 - tversky.mean(dim=1)

        return loss.mean()


# ---------------------------------------------------------------------------
# Compound Loss
# ---------------------------------------------------------------------------
class CompoundLoss(nn.Module):
    """Weighted combination of multiple losses."""

    def __init__(self, losses: List[nn.Module], weights: List[float]):
        super().__init__()
        assert len(losses) == len(weights)
        self.losses = nn.ModuleList(losses)
        self.weights = weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = torch.tensor(0.0, device=pred.device)
        for loss_fn, w in zip(self.losses, self.weights):
            total = total + w * loss_fn(pred, target)
        return total


# ---------------------------------------------------------------------------
# Deep Supervision Wrapper
# ---------------------------------------------------------------------------
class DeepSupervisionLoss(nn.Module):
    """Wrapper that computes loss at multiple scales for deep supervision.

    The main (full-resolution) output gets the highest weight.
    Lower-resolution outputs are upsampled to match the target before loss computation.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        weights: List[float] = None,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights or [1.0, 0.5, 0.25, 0.125]

    def forward(
        self,
        preds: Union[torch.Tensor, List[torch.Tensor]],
        target: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(preds, torch.Tensor):
            return self.base_loss(preds, target)

        total = torch.tensor(0.0, device=target.device)
        for i, pred in enumerate(preds):
            w = self.weights[i] if i < len(self.weights) else self.weights[-1]
            if pred.shape[2:] != target.shape[2:]:
                mode = "bilinear" if target.ndim == 4 else "trilinear"
                pred = F.interpolate(
                    pred, size=target.shape[2:], mode=mode, align_corners=False
                )
            total = total + w * self.base_loss(pred, target)

        return total


# ---------------------------------------------------------------------------
# Border-weighted loss wrapper
# ---------------------------------------------------------------------------
class BorderWeightedLoss(nn.Module):
    """Wraps a base loss with per-pixel border weighting.

    Computes a weight map that upweights pixels near class boundaries,
    following the U-Net paper (Ronneberger et al., 2015):
        w(x) = 1 + w0 * exp(-(d1(x) + d2(x))^2 / (2 * sigma^2))
    where d1, d2 are distances to the two nearest class boundaries.

    This encourages the model to learn sharp boundaries between classes.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        sigma: float = 5.0,
        w0: float = 10.0,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.sigma = sigma
        self.w0 = w0

    def _compute_border_weights(self, target: torch.Tensor) -> torch.Tensor:
        """Compute per-pixel border weight map from one-hot target.

        Args:
            target: (B, C, *spatial) one-hot labels.

        Returns:
            Weight map (B, 1, *spatial).
        """
        try:
            from scipy.ndimage import distance_transform_edt
        except ImportError:
            return torch.ones(target.shape[0], 1, *target.shape[2:],
                              device=target.device)

        target_np = target.detach().cpu().numpy()
        B = target_np.shape[0]
        spatial_shape = target_np.shape[2:]
        weights = torch.ones(B, 1, *spatial_shape, dtype=torch.float32)

        for b in range(B):
            # Get class label map
            label_map = target_np[b].argmax(axis=0)  # (*spatial)
            unique_labels = set(label_map.flat)

            if len(unique_labels) <= 1:
                continue

            # Compute distance to boundary for each class
            distances = []
            for c in sorted(unique_labels):
                mask = (label_map == c).astype(float)
                # Distance from non-mask pixels to mask boundary
                if mask.any() and (~mask.astype(bool)).any():
                    dist = distance_transform_edt(1 - mask)
                    distances.append(dist)

            if len(distances) < 2:
                continue

            # Sort distances at each pixel, take two smallest
            dist_stack = np.stack(distances, axis=0)  # (n_classes, *spatial)
            dist_stack.sort(axis=0)
            d1 = dist_stack[0]  # nearest boundary
            d2 = dist_stack[1] if dist_stack.shape[0] > 1 else dist_stack[0]

            w = 1.0 + self.w0 * np.exp(
                -((d1 + d2) ** 2) / (2 * self.sigma ** 2)
            )
            weights[b, 0] = torch.from_numpy(w.astype(np.float32))

        return weights.to(target.device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # For CE-based losses: compute weighted version
        border_w = self._compute_border_weights(target)  # (B, 1, *spatial)

        # Compute per-pixel CE loss
        target_idx = target.argmax(dim=1)  # (B, *spatial)
        per_pixel_loss = F.cross_entropy(pred, target_idx, reduction="none")  # (B, *spatial)

        weighted_loss = (per_pixel_loss * border_w.squeeze(1)).mean()

        # Also compute base loss (e.g. Dice) without border weighting
        base = self.base_loss(pred, target)

        return base + weighted_loss


# ---------------------------------------------------------------------------
# Loss factory
# ---------------------------------------------------------------------------
def build_loss(cfg: LossConfig) -> nn.Module:
    """Build loss function from config.

    Args:
        cfg: Loss configuration.

    Returns:
        Loss module.
    """
    cw = cfg.class_weights if cfg.class_weights else None

    if cfg.name == "dice":
        base = DiceLoss(smooth=cfg.dice_smooth, squared=cfg.dice_squared, class_weights=cw)
    elif cfg.name == "ce":
        base = CrossEntropyLoss(class_weights=cw, label_smoothing=cfg.label_smoothing)
    elif cfg.name == "focal":
        base = FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma, class_weights=cw)
    elif cfg.name == "tversky":
        base = TverskyLoss(
            alpha=cfg.tversky_alpha, beta=cfg.tversky_beta,
            smooth=cfg.dice_smooth, class_weights=cw,
        )
    elif cfg.name == "dice_ce":
        dice = DiceLoss(smooth=cfg.dice_smooth, squared=cfg.dice_squared, class_weights=cw)
        ce = CrossEntropyLoss(class_weights=cw, label_smoothing=cfg.label_smoothing)
        base = CompoundLoss(
            [dice, ce], cfg.compound_weights[:2] if len(cfg.compound_weights) >= 2 else [1.0, 1.0]
        )
    elif cfg.name == "dice_focal":
        dice = DiceLoss(smooth=cfg.dice_smooth, squared=cfg.dice_squared, class_weights=cw)
        focal = FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma, class_weights=cw)
        base = CompoundLoss(
            [dice, focal], cfg.compound_weights[:2] if len(cfg.compound_weights) >= 2 else [1.0, 1.0]
        )
    else:
        raise ValueError(f"Unknown loss: {cfg.name}")

    # Wrap with border weighting if configured
    if cfg.spatial_weight_mode == "border":
        base = BorderWeightedLoss(
            base, sigma=cfg.border_weight_sigma, w0=cfg.border_weight_w0,
        )

    # Wrap with deep supervision if weights are provided
    if cfg.deep_supervision_weights:
        return DeepSupervisionLoss(base, cfg.deep_supervision_weights)

    return base
