"""训练工具：AverageMeter、ModelEMA、Timer、dice 指标、随机性。"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)


class AverageMeter:
    """跟踪运行均值。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum   += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


class ModelEMA:
    """参数 EMA，支持原地 apply/restore（仅单卡，不兼容 DDP/FSDP）。"""

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
                # 整型 buffer（如 BN num_batches_tracked）直接跟随最新。
                self.shadow[k].copy_(param)

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module) -> None:
        """将 shadow 权重换入 model；live 存入 backup 供 restore()。"""
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
            # key 不一致：从零重建 shadow。
            self.shadow = {k: v.detach().clone() for k, v in loaded.items()}
            self._backup = {}
            self._swapped = False
        self.decay = state.get("decay", self.decay)


class Timer:
    """计时器。"""

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
    """(B,C,D,H,W) 逐类 sigmoid Dice。ignore_empty=True (nnU-Net)：空 GT 样本不入均；batch 全空类返 0。"""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    p = rearrange(pred_bin, 'b c ... -> b c (...)')
    t = rearrange(target, 'b c ... -> b c (...)')

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
    """逐类汇汇 (inter, denom, n_with_gt)，供 nnU-Net pooled dice：final_dice[c]=2·Σinter[c]/Σdenom[c]。"""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    p = rearrange(pred_bin, 'b c ... -> b c (...)')
    t = rearrange(target, 'b c ... -> b c (...)')
    inter = (p * t).sum(dim=(0, 2))
    denom = p.sum(dim=(0, 2)) + t.sum(dim=(0, 2))
    n_with_gt = (t.sum(dim=2) > 0).sum(dim=0).float()
    return {"inter": inter, "denom": denom, "n_with_gt": n_with_gt}


def _binary_erosion_pool(mask: torch.Tensor, ndim: int) -> torch.Tensor:
    """3x3(x3) 二值腐蚀。外侧按背景 0 处理（先 zero-pad 再 maxpool 实现 minpool）。"""
    pad_amt = [1] * (2 * ndim)
    m = F.pad(mask, pad_amt, mode="constant", value=0.0)
    pool = F.max_pool2d if ndim == 2 else F.max_pool3d
    return -pool(-m, kernel_size=3, stride=1, padding=0)


def _binary_dilate_pool(mask: torch.Tensor, ndim: int, tol: int) -> torch.Tensor:
    """Chebyshev-τ 膨胀（kernel=2τ+1 maxpool）。τ=0 直接返回。"""
    if tol <= 0:
        return mask
    k = 2 * int(tol) + 1
    pool = F.max_pool2d if ndim == 2 else F.max_pool3d
    return pool(mask, kernel_size=k, stride=1, padding=int(tol))


@torch.no_grad()
def surface_dice_batch_stats(
    pred: torch.Tensor,
    target: torch.Tensor,
    tolerance: int = 1,
    threshold: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """逐类汇汇 (sd_num, sd_denom, n_with_gt)，供 pooled surface-dice@τ：
    SD[c] = Σ(|B_p ∩ Dil_τ(B_t)| + |B_t ∩ Dil_τ(B_p)|) / Σ(|B_p|+|B_t|)。
    支持 2D (B,C,H,W) 与 3D (B,C,D,H,W)；外侧体素按背景计入边界。"""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    target_f = target.float()
    ndim = pred_bin.ndim - 2
    assert ndim in (2, 3), f"surface_dice expects 2D/3D spatial, got rank {pred_bin.ndim}"

    p_er = _binary_erosion_pool(pred_bin, ndim)
    t_er = _binary_erosion_pool(target_f, ndim)
    pb = pred_bin * (1.0 - p_er)
    tb = target_f * (1.0 - t_er)

    pb_dil = _binary_dilate_pool(pb, ndim, tolerance)
    tb_dil = _binary_dilate_pool(tb, ndim, tolerance)

    spatial_dims = tuple(range(2, pb.ndim))
    reduce_dims = (0,) + spatial_dims

    sd_num = (pb * tb_dil).sum(dim=reduce_dims) + (tb * pb_dil).sum(dim=reduce_dims)
    sd_denom = pb.sum(dim=reduce_dims) + tb.sum(dim=reduce_dims)
    n_with_gt = (target_f.flatten(2).sum(dim=2) > 0).sum(dim=0).float()
    return {"sd_num": sd_num, "sd_denom": sd_denom, "n_with_gt": n_with_gt}


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """设置随机种子。deterministic=True 强制 cudnn deterministic（较慢）。"""
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
