"""Utility classes and functions for training.

Provides:
- AverageMeter: running average tracker
- ModelEMA: exponential moving average of model weights
- Timer: simple elapsed time tracker
- compute_dice_per_class: per-class Dice coefficient (sigmoid mode)
- seed_everything: reproducibility
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Running average
# ---------------------------------------------------------------------------
class AverageMeter:
    """Tracks running mean and count."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


# ---------------------------------------------------------------------------
# Exponential Moving Average
# ---------------------------------------------------------------------------
class ModelEMA:
    """Exponential moving average of model parameters.

    Maintains a shadow copy of parameters updated as:
        shadow = decay * shadow + (1 - decay) * param

    `apply_shadow` / `restore` do an in-place tensor swap (copy_ into the
    live parameters) instead of building a full deep copy every validation
    cycle. For large models this removes hundreds of MB of per-validation
    allocation and eliminates a CPU-side copy stall.

    Distributed training notes (ISSUE-M):
        This implementation is designed for SINGLE-GPU / single-process
        training. It stores a full parameter-sized shadow + backup on the
        live model's device (~2x model memory). Under DDP it would
        replicate shadow/backup per rank; under FSDP it would break because
        `state_dict()` returns sharded tensors. If the project later adopts
        DDP/FSDP, revisit with one of:
          * shadow only on rank 0 and broadcast swapped weights, or
          * use `torch.optim.swa_utils.AveragedModel` which handles DDP.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        # shadow: cloned (detached) tensors living on the same device as the
        # model's state_dict. Using clone() rather than deepcopy() avoids
        # re-serialising autograd metadata.
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }
        # Online backup: lazily allocated once, then reused across every
        # apply_shadow/restore cycle (in-place copy_). Keyed by param name.
        self._backup: Dict[str, torch.Tensor] = {}
        self._swapped: bool = False

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow parameters."""
        for k, param in model.state_dict().items():
            if param.is_floating_point():
                # lerp_ computes: self.shadow[k] += (1-decay) * (param - self.shadow[k])
                # — equivalent to the weighted average, executed in-place.
                self.shadow[k].mul_(self.decay).add_(
                    param, alpha=1.0 - self.decay)
            else:
                # Non-floating buffers (e.g. BN running counts as int) are
                # simply tracked by the most recent model value.
                self.shadow[k].copy_(param)

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module) -> None:
        """Swap shadow weights INTO the model in-place.

        The previous online weights are preserved in `self._backup` so that
        `restore` can undo the swap without allocating fresh tensors.
        """
        if self._swapped:
            return  # idempotent
        sd = model.state_dict()
        # Lazily allocate backup buffers (same shape/dtype/device as live).
        if not self._backup:
            self._backup = {k: torch.empty_like(v) for k, v in sd.items()}
        for k, live in sd.items():
            self._backup[k].copy_(live)
            live.copy_(self.shadow[k])
        self._swapped = True

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """Restore original model parameters from the in-place backup."""
        if not self._swapped:
            return
        sd = model.state_dict()
        for k, live in sd.items():
            live.copy_(self._backup[k])
        self._swapped = False

    def state_dict(self) -> Dict:
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state: Dict) -> None:
        # Copy values in-place into our shadow buffers so downstream pointers
        # (if any) stay valid. Accept either a dict of same keys or a legacy
        # plain dict.
        loaded = state["shadow"]
        if set(loaded.keys()) == set(self.shadow.keys()):
            for k, v in loaded.items():
                self.shadow[k].copy_(v)
        else:
            # First-time load from a different model layout: rebuild.
            self.shadow = {k: v.detach().clone() for k, v in loaded.items()}
            self._backup = {}
            self._swapped = False
        self.decay = state.get("decay", self.decay)


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------
class Timer:
    """Simple elapsed time tracker."""

    def __init__(self):
        self.start = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start

    def elapsed_str(self) -> str:
        s = int(self.elapsed())
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Dice computation (per-class, sigmoid mode)
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_dice_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-5,
    ignore_empty: bool = True,
) -> torch.Tensor:
    """Compute per-class Dice coefficient using sigmoid predictions.

    Args:
        pred: (B, C, D, H, W) logits.
        target: (B, C, D, H, W) binary masks.
        threshold: Binarization threshold for sigmoid output.
        ignore_empty: If True (default, nnU-Net convention), samples whose
            GT mask is empty for a given class are excluded from that
            class's mean. The previous behaviour averaged in smoothed
            "empty-matches-empty" scores ≈ 1.0 which artificially inflated
            the reported dice on classes that were rare in the validation
            batch.

    Returns:
        Tensor of shape (C,) with mean Dice per foreground class. Classes
        with no non-empty GT in the batch return 0 (clearly marks that
        the metric is undefined rather than silently returning 1).
    """
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    B, C = pred.shape[:2]
    p = pred_bin.reshape(B, C, -1)
    t = target.reshape(B, C, -1)

    intersection = (p * t).sum(dim=2)
    denom = p.sum(dim=2) + t.sum(dim=2)
    dice = (2.0 * intersection + smooth) / (denom + smooth)  # (B, C)

    if not ignore_empty:
        return dice.mean(dim=0)

    # Mask out samples with empty GT per class; average only over non-empty
    has_gt = (t.sum(dim=2) > 0).to(dice.dtype)          # (B, C)
    num = (dice * has_gt).sum(dim=0)                     # (C,)
    den = has_gt.sum(dim=0).clamp(min=1)                 # (C,)
    # Classes that were fully empty in this batch report 0.0 (flag).
    mean_dice = torch.where(
        has_gt.sum(dim=0) > 0, num / den, torch.zeros_like(num))
    return mean_dice


@torch.no_grad()
def dice_batch_stats(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Return per-class accumulation primitives (intersection, denom,
    has_gt count) for pooling dice across a full validation pass.

    This is the nnU-Net-style "pooled dice" primitive:
        final_dice[c] = 2 * Σ intersection[c] / Σ denom[c]
    where the sums run over all samples in the dataset. Pooling avoids
    the negative bias from per-batch averaging in the presence of
    per-class empty GT masks.

    Args:
        pred: (B, C, D, H, W) logits.
        target: (B, C, D, H, W) binary masks.

    Returns:
        Dict with tensors of shape (C,):
          - "inter":  Σ |P ∩ T|
          - "denom":  Σ (|P| + |T|)
          - "n_with_gt": number of samples in this batch whose GT for
                        that class is non-empty (used for diagnostic
                        coverage logging).
    """
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    B, C = pred.shape[:2]
    p = pred_bin.reshape(B, C, -1)
    t = target.reshape(B, C, -1)
    inter = (p * t).sum(dim=(0, 2))            # (C,)
    denom = p.sum(dim=(0, 2)) + t.sum(dim=(0, 2))
    n_with_gt = (t.sum(dim=2) > 0).sum(dim=0).float()
    return {"inter": inter, "denom": denom, "n_with_gt": n_with_gt}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    logger.info("Seed set to %d (deterministic=%s)", seed, deterministic)
