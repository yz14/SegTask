"""GPU 3D 分割数据增强。空间变换（flip/affine/elastic/grid-dropout）逐样本独立；强度变换仅作用于 image。weight_map 同步受空间变换，插值由 AugConfig.wmap_interp_mode 控制。"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from ..config import AugConfig


class GPUAugmentor:
    """GPU 3D 增强管道。max_scale 为输入多分辨率最大 scale，用于缩小 elastic_deform_alpha，使最大物理通道位移 ≤ alpha 体素。"""

    def __init__(self, cfg: AugConfig, max_scale: float = 1.0):
        self.cfg       = cfg
        self.enabled   = cfg.enabled
        self.max_scale = max(float(max_scale), 1.0)
        # wmap interp：'nearest' 保留离散权重（默认）；'bilinear' 适连续。仅 affine/elastic 动 wmap。
        wmode = getattr(cfg, "wmap_interp_mode", "nearest")
        if wmode not in ("nearest", "bilinear"):
            raise ValueError(
                f"AugConfig.wmap_interp_mode={wmode!r}; expected "
                "'nearest' or 'bilinear'.")
        self.wmap_interp_mode = wmode

    def __call__(
        self, image: torch.Tensor, label: torch.Tensor, weight_map: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """对 batch 应用增强；返回 (image, label, weight_map)。"""
        if not self.enabled:
            return image, label, weight_map

        c = self.cfg

        # Spatial: flip / affine / elastic / grid-dropout
        image, label, weight_map = _random_flip(
            image, label, c.random_flip_prob, c.random_flip_axes, weight_map=weight_map)
        image, label, weight_map = _random_affine(
            image, label, c.random_affine_prob, c.random_rotate_range,
            c.random_scale_range, weight_map=weight_map,
            wmap_mode=self.wmap_interp_mode)
        # 按 max_scale 缩小 alpha。
        effective_alpha = c.elastic_deform_alpha / self.max_scale
        image, label, weight_map = _elastic_deform(
            image, label, c.elastic_deform_prob, c.elastic_deform_sigma,
            effective_alpha, weight_map=weight_map,
            wmap_mode=self.wmap_interp_mode)
        image, label, weight_map = _grid_dropout(
            image, label, c.grid_dropout_prob, c.grid_dropout_ratio,
            c.grid_dropout_holes, weight_map=weight_map)

        # Intensity (image only)
        image = _random_brightness(image, c.random_brightness_prob, c.random_brightness_range)
        image = _random_contrast(image, c.random_contrast_prob, c.random_contrast_range)
        image = _random_gamma(image, c.random_gamma_prob, c.random_gamma_range)
        image = _gaussian_noise(image, c.gaussian_noise_prob, c.gaussian_noise_std)
        image = _gaussian_blur_3d(image, c.gaussian_blur_prob, c.gaussian_blur_sigma)
        image = _simulate_lowres(image, c.simulate_lowres_prob, c.simulate_lowres_zoom)

        return image, label, weight_map


# 空间增强（逐样本独立）。
def _random_flip(
    image: torch.Tensor, label: torch.Tensor,
    prob: float, axes: list,
    weight_map: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """逐样本随机翻转；每轴独立采样。"""
    B = image.shape[0]
    for axis in axes:
        mask = torch.rand(B, device=image.device) < prob  # (B,) bool
        if mask.any():
            idx        = mask.nonzero(as_tuple=True)[0]
            image[idx] = torch.flip(image[idx], [axis])  # axis indexes into (B,C,D,H,W)
            label[idx] = torch.flip(label[idx], [axis])
            if weight_map is not None:
                weight_map[idx] = torch.flip(weight_map[idx], [axis])
    return image, label, weight_map


def _random_affine(
    image: torch.Tensor, label: torch.Tensor,
    prob: float, rotate_range: list, scale_range: list,
    weight_map: Optional[torch.Tensor] = None,
    wmap_mode: str = "nearest",
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """逐样本 3D 仿射（欧拉旋转 + 各同性缩放）。三路均用 padding_mode='border' 保边界，避免将 weight_map 背景=1 归零。"""
    B, _, D, H, W = image.shape
    device = image.device

    # 选择被增强的样本。
    mask = torch.rand(B, device=device) < prob
    if not mask.any():
        return image, label, weight_map

    # 逐样本采样旋转角（弧度）与 scale。
    n = mask.sum().item()
    lo, hi = math.radians(rotate_range[0]), math.radians(rotate_range[1])
    angles = torch.empty(n, 3, device=device).uniform_(lo, hi)  # (n, 3) for x,y,z
    scales = torch.empty(n, 1, device=device).uniform_(scale_range[0], scale_range[1])

    # 构建逐样本 3x4 仿射 + grid。
    affines = _build_rotation_matrices(angles, scales)
    grid = F.affine_grid(affines, [n, 1, D, H, W], align_corners=False)
    idx = mask.nonzero(as_tuple=True)[0]
    image[idx] = F.grid_sample(
        image[idx], grid, mode="bilinear", padding_mode="border", align_corners=False)

    # label 用 nearest 保二值。
    label[idx] = F.grid_sample(label[idx], grid, mode="nearest", padding_mode="border", align_corners=False)

    # wmap：nearest 保离散权重；bilinear 平滑连续权重。
    if weight_map is not None:
        weight_map[idx] = F.grid_sample(
            weight_map[idx], grid, mode=wmap_mode,
            padding_mode="border", align_corners=False)

    return image, label, weight_map


def _build_rotation_matrices(
    angles: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """从欧拉角（N×3 rad）与 scales（N×1）构建 (N,3,4) 仿射矩阵。"""
    N = angles.shape[0]
    device = angles.device

    cx, cy, cz = angles[:, 0].cos(), angles[:, 1].cos(), angles[:, 2].cos()
    sx, sy, sz = angles[:, 0].sin(), angles[:, 1].sin(), angles[:, 2].sin()

    # R = Rz @ Ry @ Rx。
    zeros = torch.zeros(N, device=device)

    r00 = cy * cz
    r01 = sx * sy * cz - cx * sz
    r02 = cx * sy * cz + sx * sz
    r10 = cy * sz
    r11 = sx * sy * sz + cx * cz
    r12 = cx * sy * sz - sx * cz
    r20 = -sy
    r21 = sx * cy
    r22 = cx * cy

    s = scales.squeeze(-1)  # (N,)
    # 3x4 = [s*R | 0]。
    mat = torch.stack([
        s * r00, s * r01, s * r02, zeros,
        s * r10, s * r11, s * r12, zeros,
        s * r20, s * r21, s * r22, zeros,
    ], dim=-1).reshape(N, 3, 4)

    return mat


def _elastic_deform(
    image: torch.Tensor, label: torch.Tensor,
    prob: float, sigma: float, alpha: float,
    weight_map: Optional[torch.Tensor] = None,
    wmap_mode: str = "nearest",
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """逐样本 3D 弹性形变。sigma 控平滑度（常 4–9）；alpha 控位移幅度体素（常 3–12）。"""
    B, _, D, H, W = image.shape
    device = image.device

    mask = torch.rand(B, device=device) < prob
    if not mask.any():
        return image, label, weight_map

    idx = mask.nonzero(as_tuple=True)[0]
    n = idx.shape[0]

    # 粗采样位移，上采平滑。
    cD = max(int(round(D / sigma)), 4)
    cH = max(int(round(H / sigma)), 4)
    cW = max(int(round(W / sigma)), 4)
    disp = torch.randn(n, 3, cD, cH, cW, device=device)
    disp = F.interpolate(disp, size=(D, H, W), mode="trilinear", align_corners=False)

    # 体素位移→归一化 grid 坐标（1 voxel = 2/N）；permute 后通道 (0,1,2) 对应 grid 轴 (W,H,D)。
    voxel_to_grid = torch.tensor(
        [2.0 / W, 2.0 / H, 2.0 / D],
        dtype=disp.dtype, device=device,
    ).reshape(1, 3, 1, 1, 1)
    disp = disp * alpha * voxel_to_grid

    grid = _identity_grid(n, D, H, W, device) + disp.permute(0, 2, 3, 4, 1)

    image[idx] = F.grid_sample(
        image[idx], grid, mode="bilinear", padding_mode="border", align_corners=False)
    label[idx] = F.grid_sample(label[idx], grid, mode="nearest", padding_mode="border", align_corners=False)
    if weight_map is not None:
        weight_map[idx] = F.grid_sample(
            weight_map[idx], grid, mode=wmap_mode,
            padding_mode="border", align_corners=False)

    return image, label, weight_map


def _identity_grid(
    N: int, D: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    """grid_sample(align_corners=False) 用的单位网格，范围 [-1+1/s, 1-1/s]。"""
    vecs = [torch.linspace(-1 + 1/s, 1 - 1/s, s, device=device) for s in (D, H, W)]
    grids = torch.meshgrid(*vecs, indexing="ij")  # (D, H, W) each
    grid = torch.stack(grids[::-1], dim=-1)  # (D,H,W,3)；grid_sample 顺序 W,H,D
    return grid.unsqueeze(0).expand(N, -1, -1, -1, -1)


def _grid_dropout(
    image: torch.Tensor, label: torch.Tensor,
    prob: float, ratio: float, num_holes: int,
    weight_map: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """随机零掩 num_holes 个矩形区域。label/weight_map 不被掩。"""
    if prob <= 0 or ratio <= 0:
        return image, label, weight_map

    B, _, D, H, W = image.shape
    device = image.device

    selected = torch.rand(B, device=device) < prob  # (B,)
    if not selected.any():
        return image, label, weight_map

    frac = (ratio / max(num_holes, 1)) ** (1.0 / 3.0)
    hd = max(1, int(D * frac))
    hh = max(1, int(H * frac))
    hw = max(1, int(W * frac))

    # 逐样本 hole 左上角。
    d0 = torch.randint(0, max(D - hd, 1), (B, num_holes), device=device)
    h0 = torch.randint(0, max(H - hh, 1), (B, num_holes), device=device)
    w0 = torch.randint(0, max(W - hw, 1), (B, num_holes), device=device)

    hole_mask = torch.ones(B, 1, D, H, W, device=device, dtype=image.dtype)
    d_off = torch.arange(hd, device=device)
    h_off = torch.arange(hh, device=device)
    w_off = torch.arange(hw, device=device)
    for k in range(num_holes):
        ds = d0[:, k, None] + d_off[None, :]
        hs = h0[:, k, None] + h_off[None, :]
        ws = w0[:, k, None] + w_off[None, :]
        b_idx = torch.arange(B, device=device)
        hole_mask[
            b_idx[:, None, None, None], :,
            ds[:, :, None, None],
            hs[:, None, :, None],
            ws[:, None, None, :],
        ] = 0

    # effective = selected ? hole_mask : 1。
    gate = selected.reshape(B, 1, 1, 1, 1).to(image.dtype)
    effective = hole_mask * gate + (1.0 - gate)
    return image * effective, label, weight_map


# 强度增强（逐样本独立）。
def _random_brightness(
    image: torch.Tensor, prob: float, brange: list) -> torch.Tensor:
    """逐样本随机加性亮度偏移。"""
    if prob <= 0:
        return image
    B = image.shape[0]
    mask = torch.rand(B, device=image.device) < prob
    if not mask.any():
        return image
    shift = torch.empty(B, 1, 1, 1, 1, device=image.device).uniform_(brange[0], brange[1])
    shift[~mask] = 0
    return image + shift


def _random_contrast(
    image: torch.Tensor, prob: float, crange: list) -> torch.Tensor:
    """逐样本随机对比度，以逐通道均值为轴。"""
    if prob <= 0:
        return image
    B = image.shape[0]
    mask = torch.rand(B, device=image.device) < prob
    if not mask.any():
        return image
    spatial_dims = tuple(range(2, image.ndim))
    mean = image.mean(dim=spatial_dims, keepdim=True)
    factor = torch.ones(B, 1, 1, 1, 1, device=image.device)
    factor[mask] = torch.empty(
        mask.sum().item(), 1, 1, 1, 1, device=image.device
    ).uniform_(crange[0], crange[1])
    return (image - mean) * factor + mean


def _random_gamma(
    image: torch.Tensor, prob: float, grange: list) -> torch.Tensor:
    """逐样本随机 gamma：逐通道 minmax 归一→pow(gamma)→反归一。"""
    if prob <= 0:
        return image
    B = image.shape[0]
    device = image.device
    mask = torch.rand(B, device=device) < prob  # (B,)
    if not mask.any():
        return image

    # 仅在空间轴 reduce，通道独立（多分辨率安全）。
    reduce_dims = tuple(range(2, image.ndim))
    mn = image.amin(dim=reduce_dims, keepdim=True)  # (B,C,1,1,1)
    mx = image.amax(dim=reduce_dims, keepdim=True)
    rng = (mx - mn).clamp(min=1e-7)
    normed = ((image - mn) / rng).clamp(0.0, 1.0)

    # 未选中样本 gamma=1。
    gamma = torch.empty(B, device=device).uniform_(grange[0], grange[1])
    gamma = torch.where(mask, gamma, torch.ones_like(gamma))
    gshape = (B,) + (1,) * (image.ndim - 1)
    gamma = gamma.reshape(gshape).to(image.dtype)

    return normed.pow(gamma) * rng + mn


def _gaussian_noise(
    image: torch.Tensor, prob: float, std: float) -> torch.Tensor:
    """逐样本加性高斯噪声。"""
    if prob <= 0:
        return image
    B = image.shape[0]
    mask = torch.rand(B, device=image.device) < prob
    if not mask.any():
        return image
    idx = mask.nonzero(as_tuple=True)[0]
    image[idx] = image[idx] + torch.randn_like(image[idx]) * std
    return image


def _gaussian_blur_3d(
    image: torch.Tensor, prob: float, sigma_range: list) -> torch.Tensor:
    """批量可分离 3D 高斯模糊；同一调用共享 sigma。"""
    if prob <= 0:
        return image
    B = image.shape[0]
    mask = torch.rand(B, device=image.device) < prob
    if not mask.any():
        return image

    idx = mask.nonzero(as_tuple=True)[0]
    sigma = float(torch.empty(1).uniform_(sigma_range[0], sigma_range[1]))
    ks = max(int(2 * round(3 * sigma) + 1), 3)
    x = torch.arange(ks, dtype=image.dtype, device=image.device) - ks // 2
    k1d = torch.exp(-0.5 * (x / sigma) ** 2)
    k1d = k1d / k1d.sum()
    pad = ks // 2

    # 将 (B,C) 折入 conv3d batch 轴，1D 核作用于每个 (样本, 通道) 切片。
    sub = image[idx]
    n, C = sub.shape[:2]
    sub = sub.reshape(n * C, 1, *sub.shape[2:])

    for k_shape, pad_arg in (
        ((-1, 1, 1), [0, 0, 0, 0, pad, pad]),
        ((1, -1, 1), [0, 0, pad, pad, 0, 0]),
        ((1, 1, -1), [pad, pad, 0, 0, 0, 0]),
    ):
        k = k1d.reshape(1, 1, *k_shape)
        sub = F.pad(sub, pad_arg, mode="replicate")
        sub = F.conv3d(sub, k)

    image[idx] = sub.reshape(n, C, *sub.shape[2:])
    return image


def _simulate_lowres(
    image: torch.Tensor, prob: float, zoom_range: list) -> torch.Tensor:
    """trilinear 下采→上采模拟低分辨率；同一调用共享 zoom。"""
    if prob <= 0:
        return image
    B = image.shape[0]
    mask = torch.rand(B, device=image.device) < prob
    if not mask.any():
        return image
    _, _, D, H, W = image.shape
    z = float(torch.empty(1).uniform_(zoom_range[0], zoom_range[1]))
    if z >= 0.99:
        return image
    idx = mask.nonzero(as_tuple=True)[0]
    sub = image[idx]
    small = F.interpolate(
        sub,
        size=(max(1, int(D * z)), max(1, int(H * z)), max(1, int(W * z))),
        mode="trilinear", align_corners=False)
    image[idx] = F.interpolate(
        small, size=(D, H, W), mode="trilinear", align_corners=False)
    return image
