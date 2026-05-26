"""Training utilities: AverageMeter, ModelEMA, Timer, dice metrics, seeding."""

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


class AverageMeter:
    """Running mean tracker."""

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


class ModelEMA:
    """Param EMA with in-place apply/restore (single-GPU only; not DDP/FSDP-safe)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }
        self._backup: Dict[str, torch.Tensor] = {}
        self._swapped: bool = False

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, param in model.state_dict().items():
            if param.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(param, alpha=1.0 - self.decay)
            else:
                # int buffers (e.g. BN num_batches_tracked) just follow latest
                self.shadow[k].copy_(param)

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module) -> None:
        """Swap shadow weights into model; live weights saved to backup for restore()."""
        if self._swapped:
            return
        sd = model.state_dict()
        if not self._backup:
            self._backup = {k: torch.empty_like(v) for k, v in sd.items()}
        for k, live in sd.items():
            self._backup[k].copy_(live)
            live.copy_(self.shadow[k])
        self._swapped = True

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if not self._swapped:
            return
        sd = model.state_dict()
        for k, live in sd.items():
            live.copy_(self._backup[k])
        self._swapped = False

    def state_dict(self) -> Dict:
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state: Dict) -> None:
        loaded = state["shadow"]
        if set(loaded.keys()) == set(self.shadow.keys()):
            for k, v in loaded.items():
                self.shadow[k].copy_(v)
        else:
            # mismatched keys: rebuild shadow from scratch
            self.shadow = {k: v.detach().clone() for k, v in loaded.items()}
            self._backup = {}
            self._swapped = False
        self.decay = state.get("decay", self.decay)


class Timer:
    """Elapsed time tracker."""

    def __init__(self):
        self.start = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start

    def elapsed_str(self) -> str:
        s = int(self.elapsed())
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


@torch.no_grad()
def compute_dice_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-5,
    ignore_empty: bool = True,
) -> torch.Tensor:
    """Per-class sigmoid Dice on (B,C,D,H,W). ignore_empty=True (nnU-Net): empty-GT samples
    excluded from mean; classes fully empty in batch return 0.0.
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

    has_gt = (t.sum(dim=2) > 0).to(dice.dtype)
    num = (dice * has_gt).sum(dim=0)
    den = has_gt.sum(dim=0).clamp(min=1)
    mean_dice = torch.where(
        has_gt.sum(dim=0) > 0, num / den, torch.zeros_like(num))
    return mean_dice


@torch.no_grad()
def dice_batch_stats(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Per-class primitives (inter, denom, n_with_gt) for nnU-Net-style pooled dice:
    final_dice[c] = 2*Σinter[c] / Σdenom[c] over the full validation set.
    """
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    B, C = pred.shape[:2]
    p = pred_bin.reshape(B, C, -1)
    t = target.reshape(B, C, -1)
    inter = (p * t).sum(dim=(0, 2))
    denom = p.sum(dim=(0, 2)) + t.sum(dim=(0, 2))
    n_with_gt = (t.sum(dim=2) > 0).sum(dim=0).float()
    return {"inter": inter, "denom": denom, "n_with_gt": n_with_gt}


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed RNGs. deterministic=True forces cudnn deterministic (slower)."""
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
