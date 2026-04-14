"""Utility functions: metrics, EMA, seeding, logging setup."""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    logger.info("Seed set to %d (deterministic=%s)", seed, deterministic)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging(output_dir: str, log_level: str = "INFO") -> None:
    """Configure logging to console and file."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "train.log")

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        ],
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class AverageMeter:
    """Tracks running average of a metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def compute_dice_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-5,
    output_mode: str = "softmax",
) -> torch.Tensor:
    """Compute per-class Dice coefficient.

    Args:
        pred: (B, C, *spatial) logits.
        target: (B, C, *spatial) one-hot or binary labels.
        output_mode: "softmax" → softmax activation, "per_class" → sigmoid activation.

    Returns:
        Tensor of shape (C,) with per-class Dice scores.
    """
    if output_mode == "per_class":
        pred_soft = torch.sigmoid(pred)
    elif pred.shape[1] > 1:
        pred_soft = torch.softmax(pred, dim=1)
    else:
        pred_soft = torch.sigmoid(pred)

    B, C = pred.shape[:2]
    pred_flat = pred_soft.reshape(B, C, -1)
    target_flat = target.reshape(B, C, -1)

    intersection = (pred_flat * target_flat).sum(dim=2)
    denom = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return dice.mean(dim=0)  # (C,)


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> Dict[str, float]:
    """Compute comprehensive segmentation metrics.

    Returns:
        Dict with per-class Dice and mean Dice (excluding background).
    """
    dice = compute_dice_per_class(pred, target)
    metrics = {}
    for c in range(num_classes):
        metrics[f"dice_class_{c}"] = dice[c].item()

    # Mean Dice excluding background (class 0)
    if num_classes > 1:
        metrics["mean_dice"] = dice[1:].mean().item()
    else:
        metrics["mean_dice"] = dice.mean().item()

    return metrics


# ---------------------------------------------------------------------------
# Exponential Moving Average
# ---------------------------------------------------------------------------
class ModelEMA:
    """Exponential Moving Average of model parameters.

    Maintains shadow copies of model parameters and updates them
    with a running average: shadow = decay * shadow + (1 - decay) * param.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self, model: nn.Module) -> None:
        """Replace model params with EMA shadow weights."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restore model params from backup (undo apply_shadow)."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> Dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: Dict) -> None:
        self.decay = state["decay"]
        self.shadow = state["shadow"]


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------
class Timer:
    """Simple timer for profiling."""

    def __init__(self):
        self.start_time = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def reset(self) -> None:
        self.start_time = time.time()

    def elapsed_str(self) -> str:
        elapsed = self.elapsed()
        if elapsed < 60:
            return f"{elapsed:.1f}s"
        elif elapsed < 3600:
            return f"{elapsed / 60:.1f}min"
        else:
            return f"{elapsed / 3600:.1f}h"
