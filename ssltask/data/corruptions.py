"""Models Genesis 式自监督破坏变换（image-only，GPU 逐样本）。

输入是干净 image patch ``x: (B, C, *spatial)``（spatial 为 2 或 3 维），输出是
**破坏后**的同形张量；训练目标是用破坏版重建干净版（见 ``GenesisMethod``）。

四类破坏（Zhou et al., Models Genesis, MICCAI'19）：
- 非线性强度变换（随机 Bézier 曲线）——学 HU/对比剂等强度域不变性；
- 局部像素打乱——学局部纹理/结构；
- 内补全（inner-painting）——学局部细节补全（对"补断血管"对味）；
- 外补全（outer-painting）——学全局上下文。

破坏不依赖标签；强度变换是专用的可逆曲线映射。
"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

import torch

from ..config import SSLConfig


def _interp1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """1D 线性插值（``xp`` 升序）。x 任意形状，返回同形。"""
    xc = x.clamp(xp[0], xp[-1])
    idx = torch.searchsorted(xp, xc).clamp(1, xp.numel() - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = fp[idx - 1], fp[idx]
    w = (xc - x0) / (x1 - x0).clamp_min(1e-8)
    return y0 + w * (y1 - y0)


def _bezier_intensity(sample: torch.Tensor) -> torch.Tensor:
    """随机 Bézier 曲线强度变换（per-sample，保形）。"""
    lo = sample.min()
    hi = sample.max()
    rng = (hi - lo)
    if float(rng) < 1e-6:
        return sample
    xn = (sample - lo) / rng

    dev = sample.device
    u1, v1, u2, v2 = (torch.rand(4, device=dev)).tolist()
    if random.random() < 0.5:          # 随机反转，允许对比度翻转
        v1, v2 = 1.0 - v1, 1.0 - v2
    t = torch.linspace(0.0, 1.0, 1024, device=dev)
    mt = 1.0 - t
    bx = 3 * mt ** 2 * t * u1 + 3 * mt * t ** 2 * u2 + t ** 3
    by = 3 * mt ** 2 * t * v1 + 3 * mt * t ** 2 * v2 + t ** 3
    bx_sorted, order = torch.sort(bx)
    by_sorted = by[order]
    yn = _interp1d(xn, bx_sorted, by_sorted).clamp(0.0, 1.0)
    return yn * rng + lo


def _rand_box(spatial: Sequence[int], frac_lo: float, frac_hi: float
              ) -> Tuple[List[int], List[int]]:
    """每轴随机盒子（origin, size），边长 = frac∈[lo,hi] * dim。"""
    origins: List[int] = []
    sizes: List[int] = []
    for dim in spatial:
        s_lo = max(1, int(frac_lo * dim))
        s_hi = max(s_lo, int(frac_hi * dim))
        size = random.randint(s_lo, s_hi)
        size = min(size, dim)
        origin = random.randint(0, dim - size)
        origins.append(origin)
        sizes.append(size)
    return origins, sizes


def _box_slices(origins: Sequence[int], sizes: Sequence[int]) -> Tuple:
    """(slice(None), spatial slices...) for indexing (C, *spatial)。"""
    return (slice(None),) + tuple(
        slice(o, o + s) for o, s in zip(origins, sizes))


def _local_pixel_shuffle(sample: torch.Tensor, n_blocks: int,
                         max_block: Sequence[int]) -> torch.Tensor:
    """局部像素打乱：n_blocks 个随机小窗内对 spatial 位置重排（各通道同序）。"""
    spatial = sample.shape[1:]
    nd = len(spatial)
    mb = list(max_block)[-nd:]
    for _ in range(n_blocks):
        sizes = [random.randint(1, min(int(mb[i]), spatial[i])) for i in range(nd)]
        origins = [random.randint(0, spatial[i] - sizes[i]) for i in range(nd)]
        sl = _box_slices(origins, sizes)
        block = sample[sl]                      # (C, *win)
        C = block.shape[0]
        flat = block.reshape(C, -1)
        if flat.shape[1] < 2:
            continue
        perm = torch.randperm(flat.shape[1], device=sample.device)
        sample[sl] = flat[:, perm].reshape(block.shape)
    return sample


def _paint(sample: torch.Tensor, inner: bool, count: int,
           frac_lo: float, frac_hi: float) -> torch.Tensor:
    """内/外补全：以样本值域内的均匀噪声填充随机盒子（内补）或盒子之外（外补）。"""
    lo = float(sample.min())
    hi = float(sample.max())
    if hi - lo < 1e-6:
        return sample
    if inner:
        for _ in range(max(count, 1)):
            origins, sizes = _rand_box(sample.shape[1:], frac_lo, frac_hi)
            sl = _box_slices(origins, sizes)
            noise = torch.empty_like(sample[sl]).uniform_(lo, hi)
            sample[sl] = noise
    else:
        # 外补：整体噪声，保留一个随机窗口（窗口偏大以留住主体）。
        origins, sizes = _rand_box(
            sample.shape[1:], max(frac_lo, 0.5), max(frac_hi, 0.7))
        sl = _box_slices(origins, sizes)
        keep = sample[sl].clone()
        sample.uniform_(lo, hi)
        sample[sl] = keep
    return sample


class GenesisCorruptor:
    """Models Genesis 破坏管线。``__call__(x)`` 返回破坏后的张量副本。"""

    def __init__(self, cfg: SSLConfig, spatial_dims: int):
        self.spatial_dims = int(spatial_dims)
        self.nonlinear_prob = float(cfg.nonlinear_prob)
        self.shuffle_prob = float(cfg.local_shuffle_prob)
        self.shuffle_blocks = int(cfg.local_shuffle_blocks)
        self.shuffle_max_block = list(cfg.local_shuffle_max_block)
        self.paint_prob = float(cfg.paint_prob)
        self.inner_paint_prob = float(cfg.inner_paint_prob)
        self.paint_count = int(cfg.paint_count)
        self.paint_lo = float(cfg.paint_block_range[0])
        self.paint_hi = float(cfg.paint_block_range[1])

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != self.spatial_dims + 2:
            raise ValueError(
                f"GenesisCorruptor expects (B, C, *{self.spatial_dims}d); "
                f"got shape {tuple(x.shape)}.")
        out = x.clone()
        for b in range(out.shape[0]):
            s = out[b]
            if random.random() < self.nonlinear_prob:
                s = _bezier_intensity(s)
            if random.random() < self.shuffle_prob and self.shuffle_blocks > 0:
                s = _local_pixel_shuffle(
                    s, self.shuffle_blocks, self.shuffle_max_block)
            if random.random() < self.paint_prob:
                inner = random.random() < self.inner_paint_prob
                s = _paint(s, inner, self.paint_count,
                           self.paint_lo, self.paint_hi)
            out[b] = s
        return out


__all__ = ["GenesisCorruptor"]
