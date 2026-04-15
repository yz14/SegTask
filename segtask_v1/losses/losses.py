"""Per-class binary loss functions for 3D segmentation.

All losses operate in per-class independent sigmoid mode:
  pred:   (B, num_fg, D, H, W) — raw logits, one channel per foreground class
  target: (B, num_fg, D, H, W) — binary masks, one channel per foreground class

Optional spatial weight map:
  weight_map: (B, 1, D, H, W) — per-voxel weight, broadcasts over channels.
  Used to emphasize specific label regions (e.g., small structures).

Each channel is an independent binary segmentation problem.
Background is implicit (not predicted).

Provides:
- BinaryDiceLoss
- BCELoss (binary cross-entropy with logits)
- BinaryFocalLoss
- BinaryTverskyLoss
- CompoundLoss (weighted combination)
- DeepSupervisionLoss (multi-scale wrapper)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import LossConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Binary Dice Loss
# ---------------------------------------------------------------------------
class BinaryDiceLoss(nn.Module):
    """Per-channel binary Dice loss using sigmoid.

    Loss = 1 - mean(Dice_per_channel)
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
                "class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        pred_prob = torch.sigmoid(pred)
        B, C = pred.shape[:2]

        if weight_map is not None:
            # weight_map: (B, 1, D, H, W) → broadcast to (B, C, D, H, W)
            w_spatial = weight_map.expand_as(pred)
            p = (pred_prob * w_spatial).reshape(B, C, -1)
            t = (target * w_spatial).reshape(B, C, -1)
        else:
            p = pred_prob.reshape(B, C, -1)
            t = target.reshape(B, C, -1)

        intersection = (p * t).sum(dim=2)
        if self.squared:
            denom = (p ** 2).sum(dim=2) + (t ** 2).sum(dim=2)
        else:
            denom = p.sum(dim=2) + t.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)  # (B, C)

        if self.class_weights is not None:
            w = self.class_weights.to(dice.device)
            loss = 1.0 - (dice * w.unsqueeze(0)).sum(dim=1) / w.sum()
        else:
            loss = 1.0 - dice.mean(dim=1)

        return loss.mean()


# ---------------------------------------------------------------------------
# Binary Cross-Entropy Loss
# ---------------------------------------------------------------------------
class BCELoss(nn.Module):
    """Per-channel binary cross-entropy with logits."""

    def __init__(self, class_weights: Optional[List[float]] = None):
        super().__init__()
        if class_weights:
            self.register_buffer(
                "class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        per_voxel = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        if self.class_weights is not None:
            w = self.class_weights.to(pred.device)
            per_voxel = per_voxel * w.reshape(1, -1, *([1] * (pred.ndim - 2)))
        if weight_map is not None:
            per_voxel = per_voxel * weight_map
        return per_voxel.mean()


# ---------------------------------------------------------------------------
# Binary Focal Loss
# ---------------------------------------------------------------------------
class BinaryFocalLoss(nn.Module):
    """Per-channel binary focal loss.

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
                "class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt = torch.sigmoid(pred)
        pt = pt * target + (1 - pt) * (1 - target)  # p_t
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce

        if self.class_weights is not None:
            w = self.class_weights.to(pred.device)
            loss = loss * w.reshape(1, -1, *([1] * (pred.ndim - 2)))
        if weight_map is not None:
            loss = loss * weight_map

        return loss.mean()


# ---------------------------------------------------------------------------
# Binary Tversky Loss
# ---------------------------------------------------------------------------
class BinaryTverskyLoss(nn.Module):
    """Per-channel binary Tversky loss (asymmetric Dice).

    TI = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
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
                "class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        pred_prob = torch.sigmoid(pred)
        B, C = pred.shape[:2]

        if weight_map is not None:
            w_spatial = weight_map.expand_as(pred)
            p = (pred_prob * w_spatial).reshape(B, C, -1)
            t = (target * w_spatial).reshape(B, C, -1)
        else:
            p = pred_prob.reshape(B, C, -1)
            t = target.reshape(B, C, -1)

        tp = (p * t).sum(dim=2)
        fp = (p * (1 - t)).sum(dim=2)
        fn = ((1 - p) * t).sum(dim=2)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

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

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        total = torch.tensor(0.0, device=pred.device)
        for fn, w in zip(self.losses, self.weights):
            total = total + w * fn(pred, target, weight_map=weight_map)
        return total


# ---------------------------------------------------------------------------
# Deep Supervision Wrapper
# ---------------------------------------------------------------------------
class DeepSupervisionLoss(nn.Module):
    """Multi-scale loss for deep supervision.

    Main (full-res) output gets highest weight.
    Lower-res outputs are upsampled to match target before loss computation.
    """

    def __init__(self, base_loss: nn.Module, weights: List[float]):
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights

    def forward(
        self,
        preds: Union[torch.Tensor, List[torch.Tensor]],
        target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(preds, torch.Tensor):
            return self.base_loss(preds, target, weight_map=weight_map)

        total = torch.tensor(0.0, device=target.device)
        for i, pred in enumerate(preds):
            w = self.weights[i] if i < len(self.weights) else self.weights[-1]
            wm_i = weight_map
            if pred.shape[2:] != target.shape[2:]:
                pred = F.interpolate(
                    pred, size=target.shape[2:],
                    mode="trilinear", align_corners=False,
                )
            total = total + w * self.base_loss(pred, target, weight_map=wm_i)
        return total


# ---------------------------------------------------------------------------
# Loss factory
# ---------------------------------------------------------------------------
def build_loss(cfg: LossConfig) -> nn.Module:
    """Build loss function from config.

    All losses use per-class independent sigmoid (binary mode).
    """
    cw = cfg.class_weights if cfg.class_weights else None

    if   cfg.name == "dice":
        return BinaryDiceLoss(smooth=cfg.dice_smooth, squared=cfg.dice_squared, class_weights=cw)
    elif cfg.name == "bce":
        return BCELoss(class_weights=cw)
    elif cfg.name == "focal":
        return BinaryFocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma, class_weights=cw)
    elif cfg.name == "tversky":
        return BinaryTverskyLoss(
            alpha=cfg.tversky_alpha, beta=cfg.tversky_beta,
            smooth=cfg.dice_smooth, class_weights=cw)
    elif cfg.name == "dice_bce":
        dice = BinaryDiceLoss(smooth=cfg.dice_smooth, squared=cfg.dice_squared, class_weights=cw)
        bce = BCELoss(class_weights=cw)
        weights = cfg.compound_weights[:2] if len(cfg.compound_weights) >= 2 else [1.0, 1.0]
        return CompoundLoss([dice, bce], weights)
    elif cfg.name == "dice_focal":
        dice = BinaryDiceLoss(smooth=cfg.dice_smooth, squared=cfg.dice_squared, class_weights=cw)
        focal = BinaryFocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma, class_weights=cw)
        weights = cfg.compound_weights[:2] if len(cfg.compound_weights) >= 2 else [1.0, 1.0]
        return CompoundLoss([dice, focal], weights)
    else:
        raise ValueError(f"Unknown loss: {cfg.name}")
