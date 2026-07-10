"""On-GPU 多裁剪视图生成器（DINO④ / iBOT⑥ / BYOL 等自蒸馏·对比方法共享，2D/3D）。

DINO 式 multi-crop：从一张干净 patch ``x: (B, C, *spatial)`` 生成若干**大**的 global
裁剪和若干**小**的 local 裁剪——每个裁剪是一次独立的"随机框裁剪 + 重采样到固定尺寸"
（random-resized-crop），并叠加轻量增广（随机翻转 + 强度缩放/平移），从而构造同一解剖
区域的不同"视图"。返回的同类裁剪尺寸固定，故每个裁剪一次前向即可在 batch 内对齐。

设计：放在 ``data/`` 而非某个方法内，使所有需要多视图的自监督方法（DINO④ 的 student
看全部裁剪、teacher 只看 global；iBOT⑥；对比基线 BYOL/MoCo）复用同一生成器，保持
"破坏/视图构造在 data 层、目标在 method 层"的一致分工（与 ``GenesisCorruptor`` 同构）。

约定：裁剪/增广全程 ``@torch.no_grad()``、不原地修改输入 ``x``；spatial_dims 由调用方
显式给出（2 或 3）。global/local 输出尺寸需是长度 == spatial_dims 的序列。
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from segtask_v1.models.blocks import INTERP_SMOOTH


def _sample_box(spatial: Sequence[int], scale_lo: float, scale_hi: float
                ) -> Tuple[List[int], List[int]]:
    """逐轴独立采样裁剪框：每轴边长 = U(lo,hi)*dim（>=1），起点在体内随机。"""
    origins: List[int] = []
    sizes: List[int] = []
    for dim in spatial:
        frac = random.uniform(scale_lo, scale_hi)
        size = max(1, min(int(round(frac * dim)), int(dim)))
        origin = random.randint(0, int(dim) - size)
        origins.append(origin)
        sizes.append(size)
    return origins, sizes


class MultiCropGenerator:
    """多裁剪视图生成器。``__call__(x)`` 返回 ``{'global': [...], 'local': [...]}``。

    每个列表元素形如 ``(B, C, *out_size)``：``global`` 共 ``n_global`` 个（尺寸
    ``global_size``），``local`` 共 ``n_local`` 个（尺寸 ``local_size``）。
    """

    def __init__(
        self,
        spatial_dims    : int,
        global_size     : Sequence[int],
        local_size      : Sequence[int],
        n_global        : int = 2,
        n_local         : int = 6,
        global_scale    : Sequence[float] = (0.5, 1.0),
        local_scale     : Sequence[float] = (0.15, 0.5),
        flip_prob       : float = 0.5,
        intensity_scale : float = 0.1,
        intensity_shift : float = 0.1):
        self.spatial_dims = int(spatial_dims)
        self.mode = INTERP_SMOOTH[self.spatial_dims]
        self.global_size = tuple(int(s) for s in global_size)
        self.local_size = tuple(int(s) for s in local_size)
        if len(self.global_size) != self.spatial_dims:
            raise ValueError(
                f"global_size length {len(self.global_size)} != spatial_dims "
                f"{self.spatial_dims}.")
        if len(self.local_size) != self.spatial_dims:
            raise ValueError(
                f"local_size length {len(self.local_size)} != spatial_dims "
                f"{self.spatial_dims}.")
        self.n_global = int(n_global)
        self.n_local = int(n_local)
        self.global_scale = (float(global_scale[0]), float(global_scale[1]))
        self.local_scale = (float(local_scale[0]), float(local_scale[1]))
        self.flip_prob = float(flip_prob)
        self.intensity_scale = float(intensity_scale)
        self.intensity_shift = float(intensity_shift)

    # ------------------------------------------------------------------
    def _crop_resize(self, sample: torch.Tensor, out_size: Tuple[int, ...],
                     scale: Tuple[float, float]) -> torch.Tensor:
        """单样本 (C, *spatial) → 随机框裁剪 → 重采样到 ``out_size`` → (C, *out_size)。"""
        origins, sizes = _sample_box(sample.shape[1:], scale[0], scale[1])
        sl = (slice(None),) + tuple(
            slice(o, o + s) for o, s in zip(origins, sizes))
        crop = sample[sl]                                   # (C, *box)
        crop = F.interpolate(
            crop.unsqueeze(0).float(), size=out_size,
            mode=self.mode, align_corners=False).squeeze(0)
        return crop.to(sample.dtype)

    def _augment(self, crop: torch.Tensor) -> torch.Tensor:
        """轻量增广：逐轴随机翻转 + 全局强度缩放/平移（不引入解剖以外先验）。"""
        for axis in range(self.spatial_dims):
            if random.random() < self.flip_prob:
                crop = torch.flip(crop, dims=[axis + 1])    # +1：跳过通道维
        if self.intensity_scale > 0:
            s = 1.0 + random.uniform(-self.intensity_scale, self.intensity_scale)
            crop = crop * s
        if self.intensity_shift > 0:
            crop = crop + random.uniform(-self.intensity_shift, self.intensity_shift)
        return crop

    def _make_crops(self, x: torch.Tensor, n: int, out_size: Tuple[int, ...],
                    scale: Tuple[float, float]) -> List[torch.Tensor]:
        B = x.shape[0]
        crops: List[torch.Tensor] = []
        for _ in range(n):
            per_sample = [
                self._augment(self._crop_resize(x[b], out_size, scale))
                for b in range(B)]
            crops.append(torch.stack(per_sample, dim=0))     # (B, C, *out_size)
        return crops

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        if x.dim() != self.spatial_dims + 2:
            raise ValueError(
                f"MultiCropGenerator expects (B, C, *{self.spatial_dims}d); "
                f"got shape {tuple(x.shape)}.")
        return {
            "global": self._make_crops(
                x, self.n_global, self.global_size, self.global_scale),
            "local": self._make_crops(
                x, self.n_local, self.local_size, self.local_scale),
        }


class PairedCropGenerator:
    """带坐标元数据的成对裁剪生成器（稠密对应类方法：VICRegL 等）。

    ``__call__(x)`` 返回 ``(views, metas)``：``views`` 是两个 ``(B, C, *out_size)``
    视图（各自独立的随机框裁剪 + 重采样 + 翻转 + 强度增广）；``metas`` 记录每个
    视图逐样本的裁剪框（``origin``/``size``，原体素坐标）与逐轴翻转位（``flip``），
    使方法层能把任一特征图位点映射回**原 patch 体素坐标**，据此做视图间的
    位置匹配（location-based matching）。

    与 :class:`MultiCropGenerator` 的分工一致：视图构造在 data 层、匹配与目标在
    method 层。全程 ``@torch.no_grad()``、不修改输入。
    """

    def __init__(
        self,
        spatial_dims    : int,
        out_size        : Sequence[int],
        scale           : Sequence[float] = (0.6, 1.0),
        flip_prob       : float = 0.5,
        intensity_scale : float = 0.1,
        intensity_shift : float = 0.1):
        self.spatial_dims = int(spatial_dims)
        self.mode = INTERP_SMOOTH[self.spatial_dims]
        self.out_size = tuple(int(s) for s in out_size)
        if len(self.out_size) != self.spatial_dims:
            raise ValueError(
                f"out_size length {len(self.out_size)} != spatial_dims "
                f"{self.spatial_dims}.")
        self.scale = (float(scale[0]), float(scale[1]))
        self.flip_prob = float(flip_prob)
        self.intensity_scale = float(intensity_scale)
        self.intensity_shift = float(intensity_shift)

    def _one_view(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = x.shape[0]
        crops: List[torch.Tensor] = []
        origins = torch.zeros(B, self.spatial_dims)
        sizes = torch.zeros(B, self.spatial_dims)
        flips = torch.zeros(B, self.spatial_dims, dtype=torch.bool)
        for b in range(B):
            o, s = _sample_box(x.shape[2:], self.scale[0], self.scale[1])
            sl = (slice(None),) + tuple(
                slice(oo, oo + ss) for oo, ss in zip(o, s))
            crop = x[b][sl]
            crop = F.interpolate(
                crop.unsqueeze(0).float(), size=self.out_size,
                mode=self.mode, align_corners=False).squeeze(0)
            for axis in range(self.spatial_dims):
                if random.random() < self.flip_prob:
                    crop = torch.flip(crop, dims=[axis + 1])
                    flips[b, axis] = True
            if self.intensity_scale > 0:
                crop = crop * (1.0 + random.uniform(
                    -self.intensity_scale, self.intensity_scale))
            if self.intensity_shift > 0:
                crop = crop + random.uniform(
                    -self.intensity_shift, self.intensity_shift)
            crops.append(crop.to(x.dtype))
            origins[b] = torch.tensor([float(v) for v in o])
            sizes[b] = torch.tensor([float(v) for v in s])
        meta = {"origin": origins.to(x.device), "size": sizes.to(x.device),
                "flip": flips.to(x.device)}
        return torch.stack(crops, dim=0), meta

    @torch.no_grad()
    def __call__(self, x: torch.Tensor
                 ) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
        if x.dim() != self.spatial_dims + 2:
            raise ValueError(
                f"PairedCropGenerator expects (B, C, *{self.spatial_dims}d); "
                f"got shape {tuple(x.shape)}.")
        v1, m1 = self._one_view(x)
        v2, m2 = self._one_view(x)
        return [v1, v2], [m1, m2]


def site_coords(feat_spatial: Sequence[int], meta: Dict[str, torch.Tensor]
                ) -> torch.Tensor:
    """特征图位点中心 → 原 patch 体素坐标 ``(B, N, D)``（N = 位点数，D = 轴数）。

    位点 i（轴长 n）在裁剪框内的归一化中心为 ``(i+0.5)/n``（翻转轴取
    ``1-(i+0.5)/n``），映射回原坐标 ``origin + frac*size``。供视图间
    location-based 匹配（如 VICRegL 的 top-γ 最近位置配对）。
    """
    origin, size, flip = meta["origin"], meta["size"], meta["flip"]
    B, D = origin.shape
    sp = [int(s) for s in feat_spatial]
    if len(sp) != D:
        raise ValueError(f"feat_spatial length {len(sp)} != spatial dims {D}.")
    axes = [(torch.arange(n, device=origin.device, dtype=origin.dtype) + 0.5) / n
            for n in sp]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)  # (*sp, D)
    frac = grid.reshape(1, -1, D).expand(B, -1, D).clone()            # (B, N, D)
    frac = torch.where(flip.unsqueeze(1), 1.0 - frac, frac)
    return origin.unsqueeze(1) + frac * size.unsqueeze(1)


__all__ = ["MultiCropGenerator", "PairedCropGenerator", "site_coords"]
