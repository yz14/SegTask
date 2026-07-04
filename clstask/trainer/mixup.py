"""Mixup / CutMix（仅 volume 粒度）。

标签一律软化：多标签 target (B, K) 线性混合；单标签硬标签先 one-hot 再混合
（损失侧 ``SingleLabelCELoss`` 支持软标签）。CutMix 的 λ 按实际裁剪体积
比例回算（Yun et al., ICCV 2019）。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def _to_soft(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """硬标签 (B,) → one-hot (B, K)；已是 (B, K) 原样返回 float。"""
    if target.ndim == 1 and not target.is_floating_point():
        return torch.nn.functional.one_hot(
            target.long(), num_classes).float()
    return target.float()


def _rand_uniform(generator: Optional[torch.Generator]) -> float:
    return float(torch.rand((), generator=generator))


def _sample_beta(alpha: float,
                 generator: Optional[torch.Generator]) -> float:
    """Beta(α, α) 采样（Jöhnk 拒绝法），随机源统一走 ``generator``。"""
    inv = 1.0 / max(alpha, 1e-8)
    while True:
        x = _rand_uniform(generator) ** inv
        y = _rand_uniform(generator) ** inv
        if 0.0 < x + y <= 1.0:
            return x / (x + y)


def _rand_box(shape: Tuple[int, ...], lam: float,
              generator: Optional[torch.Generator]) -> Tuple[slice, ...]:
    """在空间形状 ``shape`` 内取体积占比 (1-λ) 的随机 box。"""
    ratio = (1.0 - lam) ** (1.0 / len(shape))
    slices = []
    for dim in shape:
        cut = max(int(round(dim * ratio)), 1)
        lo = int(torch.randint(0, max(dim - cut + 1, 1), (1,),
                               generator=generator).item())
        slices.append(slice(lo, lo + cut))
    return tuple(slices)


def apply_mixup_cutmix(
    images: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    mixup_alpha: float,
    cutmix_alpha: float,
    prob: float,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """按配置对一个 batch 应用 mixup 或 cutmix；返回 (images, soft_targets)。

    两者都启用时每 batch 等概率二选一。``prob`` 为整体触发概率；未触发时
    target 仍统一软化（保证损失侧类型一致）。
    """
    soft = _to_soft(targets, num_classes)
    use_mix = mixup_alpha > 0
    use_cut = cutmix_alpha > 0
    if (not use_mix and not use_cut) or _rand_uniform(generator) >= prob:
        return images, soft
    if use_mix and use_cut:
        use_mix = _rand_uniform(generator) < 0.5
        use_cut = not use_mix

    b = images.shape[0]
    if b < 2:
        return images, soft
    perm = torch.randperm(b, device=images.device, generator=generator)
    if use_mix:
        lam = _sample_beta(mixup_alpha, generator)
        images = lam * images + (1 - lam) * images[perm]
    else:
        lam = _sample_beta(cutmix_alpha, generator)
        box = _rand_box(tuple(images.shape[2:]), lam, generator)
        images = images.clone()
        idx = (slice(None), slice(None)) + box
        images[idx] = images[perm][idx]
        # λ 按实际裁剪体积回算。
        cut_vol = math.prod(s.stop - s.start for s in box)
        lam = 1.0 - cut_vol / math.prod(images.shape[2:])
    soft = lam * soft + (1 - lam) * soft[perm]
    return images, soft


__all__ = ["apply_mixup_cutmix"]
