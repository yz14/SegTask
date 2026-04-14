"""Utility classes and functions for training.

Provides:
- AverageMeter: running average tracker
- ModelEMA: exponential moving average of model weights
- Timer: simple elapsed time tracker
- compute_dice_per_class: per-class Dice coefficient (sigmoid mode)
- seed_everything: reproducibility
"""

from __future__ import annotations

import copy
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
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())
        self._backup = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow parameters."""
        for k, param in model.state_dict().items():
            if param.is_floating_point():
                self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * param
            else:
                self.shadow[k] = param

    def apply_shadow(self, model: nn.Module) -> None:
        """Replace model parameters with shadow (for eval)."""
        self._backup = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.shadow)

    def restore(self, model: nn.Module) -> None:
        """Restore original model parameters."""
        if self._backup:
            model.load_state_dict(self._backup)
            self._backup = {}

    def state_dict(self) -> Dict:
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state: Dict) -> None:
        self.shadow = state["shadow"]
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
) -> torch.Tensor:
    """Compute per-class Dice coefficient using sigmoid predictions.

    Args:
        pred: (B, C, D, H, W) logits.
        target: (B, C, D, H, W) binary masks.
        threshold: Binarization threshold for sigmoid output.

    Returns:
        Tensor of shape (C,) with mean Dice per foreground class.
    """
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    B, C = pred.shape[:2]
    p = pred_bin.reshape(B, C, -1)
    t = target.reshape(B, C, -1)

    intersection = (p * t).sum(dim=2)
    denom = p.sum(dim=2) + t.sum(dim=2)
    dice = (2.0 * intersection + smooth) / (denom + smooth)  # (B, C)

    return dice.mean(dim=0)  # (C,)


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
