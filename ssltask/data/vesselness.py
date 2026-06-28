"""Label-free 多尺度 Frangi vesselness（纯 torch，2D/3D，GPU 友好）。

用作自监督预训练（``ssl.method='prior'``）的**免标注**回归目标：从原始 CT 直接
算出"管状结构置信图"，让网络回归它。等于把"什么是血管"的经典几何先验灌进
encoder/decoder，直击分割的 precision 短板（学会只在管状证据处响应）。

按通道独立计算（depthwise）：3D cubic（C=1）→ 3D vesselness；2.5D（C=切片数）→
逐切片 2D vesselness。输出与输入同形、逐样本归一化到 [0,1]。

参考：Frangi et al., "Multiscale vessel enhancement filtering", MICCAI'98。
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


def _gaussian_kernel1d(sigma: float, dtype, device) -> Tuple[torch.Tensor, int]:
    ks = max(int(2 * round(3 * sigma) + 1), 3)
    x = torch.arange(ks, dtype=dtype, device=device) - ks // 2
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    return k, ks // 2


def _separable_gaussian(x: torch.Tensor, sigma: float, spatial_dims: int
                        ) -> torch.Tensor:
    """可分离高斯平滑（depthwise，逐 spatial 轴 1D 卷积）。x: (B,C,*spatial)。"""
    B, C = x.shape[:2]
    spatial = x.shape[2:]
    k, pad = _gaussian_kernel1d(sigma, x.dtype, x.device)
    conv = F.conv3d if spatial_dims == 3 else F.conv2d
    xr = x.reshape(B * C, 1, *spatial)
    for axis in range(spatial_dims):
        kshape = [1, 1] + [1] * spatial_dims
        kshape[2 + axis] = k.numel()
        ker = k.reshape(kshape)
        padcfg = [0] * (2 * spatial_dims)
        idx = (spatial_dims - 1 - axis) * 2     # F.pad 末维在前
        padcfg[idx] = pad
        padcfg[idx + 1] = pad
        xr = F.pad(xr, padcfg, mode="replicate")
        xr = conv(xr, ker)
    return xr.reshape(B, C, *spatial)


def _hessian_components(x: torch.Tensor, spatial_dims: int
                        ) -> Dict[Tuple[int, int], torch.Tensor]:
    """二阶偏导（中心差分），返回上三角 Hessian 分量 {(i,j): tensor}。"""
    dims = tuple(range(2, 2 + spatial_dims))
    first = torch.gradient(x, dim=dims)
    comps: Dict[Tuple[int, int], torch.Tensor] = {}
    for i, gi in enumerate(first):
        second = torch.gradient(gi, dim=dims)
        for j, gij in enumerate(second):
            if j >= i:
                comps[(i, j)] = gij
    return comps


def _eigvals_abs_sorted(comps: Dict[Tuple[int, int], torch.Tensor],
                        spatial_dims: int) -> torch.Tensor:
    """组装对称 Hessian → 特征值，按 |λ| 升序返回 (B,C,*spatial,spatial_dims)。"""
    if spatial_dims == 3:
        h00, h11, h22 = comps[(0, 0)], comps[(1, 1)], comps[(2, 2)]
        h01, h02, h12 = comps[(0, 1)], comps[(0, 2)], comps[(1, 2)]
        row0 = torch.stack([h00, h01, h02], dim=-1)
        row1 = torch.stack([h01, h11, h12], dim=-1)
        row2 = torch.stack([h02, h12, h22], dim=-1)
        mat = torch.stack([row0, row1, row2], dim=-2)
    else:
        h00, h11, h01 = comps[(0, 0)], comps[(1, 1)], comps[(0, 1)]
        row0 = torch.stack([h00, h01], dim=-1)
        row1 = torch.stack([h01, h11], dim=-1)
        mat = torch.stack([row0, row1], dim=-2)
    eig = torch.linalg.eigvalsh(mat)                       # 升序（按值）
    order = torch.argsort(eig.abs(), dim=-1)               # 按 |λ| 升序
    return torch.gather(eig, -1, order)


def _frangi_response(eig: torch.Tensor, spatial_dims: int,
                     alpha: float, beta: float,
                     black_vessels: bool) -> torch.Tensor:
    """由排序特征值算 Frangi 响应。eig: (B,C,*spatial,spatial_dims)。"""
    eps = 1e-10
    sign = -1.0 if not black_vessels else 1.0     # 亮血管：λ2,λ3<0
    if spatial_dims == 3:
        lam1 = eig[..., 0]
        lam2 = eig[..., 1]
        lam3 = eig[..., 2]
        ra = lam2.abs() / (lam3.abs() + eps)
        rb = lam1.abs() / (torch.sqrt((lam2 * lam3).abs()) + eps)
        s = torch.sqrt(lam1 ** 2 + lam2 ** 2 + lam3 ** 2)
        # c = 半最大 structureness（逐 B,C）
        c = 0.5 * _amax_spatial(s, spatial_dims)
        c = c.clamp_min(eps)
        v = ((1.0 - torch.exp(-ra ** 2 / (2 * alpha ** 2)))
             * torch.exp(-rb ** 2 / (2 * beta ** 2))
             * (1.0 - torch.exp(-s ** 2 / (2 * c ** 2))))
        invalid = (sign * lam2 < 0) | (sign * lam3 < 0)
    else:
        lam1 = eig[..., 0]
        lam2 = eig[..., 1]
        rb = lam1.abs() / (lam2.abs() + eps)
        s = torch.sqrt(lam1 ** 2 + lam2 ** 2)
        c = 0.5 * _amax_spatial(s, spatial_dims)
        c = c.clamp_min(eps)
        v = (torch.exp(-rb ** 2 / (2 * beta ** 2))
             * (1.0 - torch.exp(-s ** 2 / (2 * c ** 2))))
        invalid = (sign * lam2 < 0)
    return v.masked_fill(invalid, 0.0)


def _amax_spatial(x: torch.Tensor, spatial_dims: int) -> torch.Tensor:
    """对 spatial 维取最大，保留 (B,C) 并广播回 (B,C,*1)。"""
    dims = tuple(range(2, 2 + spatial_dims))
    m = x.amax(dim=dims, keepdim=True)
    return m


@torch.no_grad()
def frangi_vesselness(
    x: torch.Tensor,
    scales: Sequence[float],
    spatial_dims: int,
    alpha: float = 0.5,
    beta: float = 0.5,
    black_vessels: bool = False,
    normalize: bool = True) -> torch.Tensor:
    """多尺度 Frangi vesselness。x: (B,C,*spatial) → 同形 [0,1] 置信图。

    按通道独立处理；多尺度取逐体素最大；逐样本归一化到 [0,1]。"""
    if x.dim() != spatial_dims + 2:
        raise ValueError(
            f"frangi_vesselness expects (B, C, *{spatial_dims}d); "
            f"got shape {tuple(x.shape)}.")
    if not scales:
        raise ValueError("scales must be a non-empty list of sigmas.")
    xf = x.float()
    out = None
    for sigma in scales:
        sigma = float(sigma)
        smoothed = _separable_gaussian(xf, sigma, spatial_dims)
        comps = _hessian_components(smoothed, spatial_dims)
        # γ 归一化：二阶导 × σ²，使不同尺度响应可比。
        comps = {k: v * (sigma ** 2) for k, v in comps.items()}
        eig = _eigvals_abs_sorted(comps, spatial_dims)
        resp = _frangi_response(eig, spatial_dims, alpha, beta, black_vessels)
        out = resp if out is None else torch.maximum(out, resp)
    out = out.to(x.dtype)
    if normalize:
        m = _amax_spatial(out.float(), spatial_dims).to(x.dtype)
        out = out / m.clamp_min(1e-6)
    return out


__all__ = ["frangi_vesselness"]
