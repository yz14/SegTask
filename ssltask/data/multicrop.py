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
    """随机框采样，**体积一致**：``scale`` 解释为体积/面积占比（RandomResizedCrop
    约定），采一个体积占比 ``f ~ U(lo,hi)``，各轴边长占比 = ``f**(1/dims)``（各向同性），
    故实际裁剪体积占比 ≈ f 且跨样本可比。

    旧实现逐轴独立采 ``U(lo,hi)`` 作**边长**占比，实际体积占比 = 各轴之积（3D 下
    U(0.5,1) → 体积占比 [0.125,1] 严重偏小且方差极大，与 scale 语义不符）。"""
    ndim = len(spatial)
    vol_frac = random.uniform(scale_lo, scale_hi)
    edge_frac = vol_frac ** (1.0 / ndim)
    origins: List[int] = []
    sizes: List[int] = []
    for dim in spatial:
        size = max(1, min(int(round(edge_frac * dim)), int(dim)))
        origin = random.randint(0, int(dim) - size)
        origins.append(origin)
        sizes.append(size)
    return origins, sizes


def _affine_grid(spatial: Sequence[int], out_size: Sequence[int],
                 origins: torch.Tensor, sizes: torch.Tensor,
                 flips: torch.Tensor, spatial_dims: int,
                 device: torch.device) -> torch.Tensor:
    """批量随机框 → ``grid_sample`` 归一化采样网格（align_corners=False）。

    等价于「按 box 裁剪 + F.interpolate 到 out_size」：输出像素 j 沿某轴映射到输入
    像素 ``o + (j+0.5)*s/O - 0.5``，转 [-1,1] 归一化即 ``g = a*(j+0.5)+b``，其中
    ``a = 2s/(O·L)``、``b = 2o/L − 1``；``flips`` 为真时以 ``O−1−j`` 反向采样。

    参数张量形状 ``(B, dims)``（origins/sizes 体素单位，flips 布尔）；轴序 (D,H,W)/
    (H,W)，返回 grid 末维按 grid_sample 约定取反序 (x,y[,z])。"""
    B = origins.shape[0]
    coords: List[torch.Tensor] = []      # 逐轴 (B, O_axis)
    for axis in range(spatial_dims):
        L = int(spatial[axis])
        O = int(out_size[axis])
        j = torch.arange(O, dtype=torch.float32, device=device)
        o = origins[:, axis].to(device).unsqueeze(1)
        s = sizes[:, axis].to(device).unsqueeze(1)
        a = 2.0 * s / (O * L)
        b = 2.0 * o / L - 1.0
        jj = j.unsqueeze(0).expand(B, O)
        flip = flips[:, axis].to(device).view(B, 1)
        jj = torch.where(flip, (O - 1) - jj, jj)
        coords.append(a * (jj + 0.5) + b)
    if spatial_dims == 3:
        o0, o1, o2 = (int(out_size[0]), int(out_size[1]), int(out_size[2]))
        gz = coords[0].view(B, o0, 1, 1).expand(B, o0, o1, o2)
        gy = coords[1].view(B, 1, o1, 1).expand(B, o0, o1, o2)
        gx = coords[2].view(B, 1, 1, o2).expand(B, o0, o1, o2)
        return torch.stack([gx, gy, gz], dim=-1)       # (B,D,H,W,3)
    o0, o1 = int(out_size[0]), int(out_size[1])
    gy = coords[0].view(B, o0, 1).expand(B, o0, o1)
    gx = coords[1].view(B, 1, o1).expand(B, o0, o1)
    return torch.stack([gx, gy], dim=-1)               # (B,H,W,2)


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
    def _batched_crop_resize(self, x: torch.Tensor,
                             out_size: Tuple[int, ...],
                             scale: Tuple[float, float]) -> torch.Tensor:
        """整批一次性「随机框裁剪 + 重采样 + 翻转」：(B,C,*spatial)→(B,C,*out_size)。

        逐样本采框/翻转（纯 RNG，开销可忽略），折叠进单次 ``grid_sample``，取代旧
        实现的 B 次 ``F.interpolate``（大幅减少 kernel launch，GPU 上更快）。翻转直接
        编码进采样网格，强度增广逐样本向量化。"""
        B = x.shape[0]
        dims = self.spatial_dims
        spatial = [int(s) for s in x.shape[2:]]
        origins = torch.empty(B, dims, dtype=torch.float32)
        sizes = torch.empty(B, dims, dtype=torch.float32)
        flips = torch.zeros(B, dims, dtype=torch.bool)
        for b in range(B):
            o, s = _sample_box(spatial, scale[0], scale[1])
            origins[b] = torch.tensor(o, dtype=torch.float32)
            sizes[b] = torch.tensor(s, dtype=torch.float32)
            if self.flip_prob > 0:
                for axis in range(dims):
                    flips[b, axis] = random.random() < self.flip_prob
        grid = _affine_grid(spatial, out_size, origins, sizes, flips, dims,
                            x.device)
        out = F.grid_sample(
            x.float(), grid, mode="bilinear", align_corners=False,
            padding_mode="border")
        out = self._intensity_aug(out)
        return out.to(x.dtype)

    def _intensity_aug(self, out: torch.Tensor) -> torch.Tensor:
        """逐样本全局强度缩放/平移（向量化）；out: (B,C,*spatial)。"""
        B = out.shape[0]
        shape = [B] + [1] * (self.spatial_dims + 1)
        if self.intensity_scale > 0:
            s = 1.0 + (torch.rand(B, device=out.device) * 2 - 1) \
                * self.intensity_scale
            out = out * s.view(shape)
        if self.intensity_shift > 0:
            sh = (torch.rand(B, device=out.device) * 2 - 1) \
                * self.intensity_shift
            out = out + sh.view(shape)
        return out

    def _make_crops(self, x: torch.Tensor, n: int, out_size: Tuple[int, ...],
                    scale: Tuple[float, float]) -> List[torch.Tensor]:
        return [self._batched_crop_resize(x, out_size, scale)
                for _ in range(n)]

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
        """整批一次性「随机框裁剪 + 重采样 + 翻转 + 强度增广」（复用
        :func:`_affine_grid` 的单次 ``grid_sample`` 批量路径，同
        :meth:`MultiCropGenerator._batched_crop_resize`），并返回逐样本裁剪框/
        翻转元数据供位置匹配。逐样本采框/翻转仍是纯 RNG（开销可忽略）。"""
        B = x.shape[0]
        dims = self.spatial_dims
        spatial = [int(s) for s in x.shape[2:]]
        origins = torch.empty(B, dims, dtype=torch.float32)
        sizes = torch.empty(B, dims, dtype=torch.float32)
        flips = torch.zeros(B, dims, dtype=torch.bool)
        for b in range(B):
            o, s = _sample_box(spatial, self.scale[0], self.scale[1])
            origins[b] = torch.tensor(o, dtype=torch.float32)
            sizes[b] = torch.tensor(s, dtype=torch.float32)
            if self.flip_prob > 0:
                for axis in range(dims):
                    flips[b, axis] = random.random() < self.flip_prob
        grid = _affine_grid(spatial, self.out_size, origins, sizes, flips,
                            dims, x.device)
        out = F.grid_sample(
            x.float(), grid, mode="bilinear", align_corners=False,
            padding_mode="border")
        if self.intensity_scale > 0:
            sc = 1.0 + (torch.rand(B, device=out.device) * 2 - 1) \
                * self.intensity_scale
            out = out * sc.view([B] + [1] * (dims + 1))
        if self.intensity_shift > 0:
            sh = (torch.rand(B, device=out.device) * 2 - 1) \
                * self.intensity_shift
            out = out + sh.view([B] + [1] * (dims + 1))
        meta = {"origin": origins.to(x.device), "size": sizes.to(x.device),
                "flip": flips.to(x.device)}
        return out.to(x.dtype), meta

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
