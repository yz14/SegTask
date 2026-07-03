"""GPU 3D 分割数据增强。空间变换（flip/affine/elastic/grid-dropout）逐样本独立；强度变换仅作用于 image。weight_map 同步受空间变换，插值由 AugConfig.wmap_interp_mode 控制。"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange

from ..config import AugConfig


class GPUAugmentor:
    """GPU 3D 增强管道。max_scale 为输入多分辨率最大 scale，用于缩小 elastic_deform_alpha，使最大物理通道位移 ≤ alpha 体素。"""

    def __init__(self, cfg: AugConfig, max_scale: float = 1.0):
        self.cfg       = cfg
        self.enabled   = cfg.enabled
        self.max_scale = max(float(max_scale), 1.0)
        # wmap interp：'nearest' 保留离散权重（默认）；'bilinear' 适连续。仅 affine/elastic 动 wmap。
        wmode = cfg.wmap_interp_mode
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

        # 克隆输入，避免原地修改污染调用方持有的张量。
        image = image.clone()
        label = label.clone()
        if weight_map is not None:
            weight_map = weight_map.clone()

        c = self.cfg

        # Spatial: flip / affine / elastic / grid-dropout
        image, label, weight_map = _random_flip(
            image, label, c.random_flip_prob, c.random_flip_axes, weight_map=weight_map)
        image, label, weight_map = _random_affine(
            image, label, c.random_affine_prob, c.random_rotate_range,
            c.random_scale_range, weight_map=weight_map,
            wmap_mode=self.wmap_interp_mode,
            translate_range=c.random_translate_range,
            rotate_range_per_axis=c.random_rotate_range_per_axis,
            aspect_correct=c.random_affine_aspect_correct)
        # 按 max_scale 缩小 alpha。
        effective_alpha = c.elastic_deform_alpha / self.max_scale
        image, label, weight_map = _elastic_deform(
            image, label, c.elastic_deform_prob, c.elastic_deform_sigma,
            effective_alpha, weight_map=weight_map,
            wmap_mode=self.wmap_interp_mode)
        image, label, weight_map = _grid_dropout(
            image, label, c.grid_dropout_prob, c.grid_dropout_ratio,
            c.grid_dropout_holes, weight_map=weight_map)

        # Intensity (image only)。开 intensity_clamp 时记录增强前逐样本逐通道
        # min/max，全部强度增强后夹回原范围，避免叠加越界。
        if c.intensity_clamp:
            reduce_dims = tuple(range(2, image.ndim))
            clamp_lo = image.amin(dim=reduce_dims, keepdim=True)
            clamp_hi = image.amax(dim=reduce_dims, keepdim=True)
        image = _random_brightness(image, c.random_brightness_prob, c.random_brightness_range)
        image = _random_contrast(image, c.random_contrast_prob, c.random_contrast_range)
        image = _random_gamma(image, c.random_gamma_prob, c.random_gamma_range)
        image = _gaussian_noise(image, c.gaussian_noise_prob, c.gaussian_noise_std)
        image = _gaussian_blur_3d(image, c.gaussian_blur_prob, c.gaussian_blur_sigma)
        image = _simulate_lowres(image, c.simulate_lowres_prob, c.simulate_lowres_zoom)
        if c.intensity_clamp:
            image = torch.maximum(torch.minimum(image, clamp_hi), clamp_lo)

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
    translate_range: Optional[list] = None,
    rotate_range_per_axis: Optional[list] = None,
    aspect_correct: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """逐样本 3D 仿射（欧拉旋转 + 各同性缩放 + 可选平移）。三路均用 padding_mode='border' 保边界，避免将 weight_map 背景=1 归零。

    rotate_range_per_axis：3 对 [lo,hi]（度），轴序 (x,y,z)=(W,H,D)；None 时三轴共用 rotate_range。
    aspect_correct：True 时在物理各向同性坐标里旋转（R←A⁻¹RA，A=diag(W,H,D)），
    消除各向异性 patch 上 affine_grid 归一化坐标旋转隐含的剪切/非均匀缩放。
    translate_range：[lo,hi]（归一化坐标，[-1,1] 跨整轴），逐轴独立采样；None/[0,0]=无平移。
    """
    B, _, D, H, W = image.shape
    device = image.device

    # 选择被增强的样本。
    mask = torch.rand(B, device=device) < prob
    if not mask.any():
        return image, label, weight_map

    # 逐样本采样旋转角（弧度）与 scale。
    n = mask.sum().item()
    if rotate_range_per_axis is not None:
        angles = torch.empty(n, 3, device=device)  # (n, 3) for x,y,z
        for ax in range(3):
            lo = math.radians(rotate_range_per_axis[ax][0])
            hi = math.radians(rotate_range_per_axis[ax][1])
            angles[:, ax].uniform_(lo, hi)
    else:
        lo, hi = math.radians(rotate_range[0]), math.radians(rotate_range[1])
        angles = torch.empty(n, 3, device=device).uniform_(lo, hi)  # (n, 3) for x,y,z
    scales = torch.empty(n, 1, device=device).uniform_(scale_range[0], scale_range[1])

    translations = None
    if translate_range is not None and (
            translate_range[0] != 0.0 or translate_range[1] != 0.0):
        translations = torch.empty(n, 3, device=device).uniform_(
            translate_range[0], translate_range[1])

    aspect = None
    if aspect_correct:
        # affine_grid 坐标轴序 (x,y,z)=(W,H,D)；只用比例，公共尺度无影响。
        aspect = torch.tensor(
            [float(W), float(H), float(D)], device=device, dtype=torch.float32)

    # 构建逐样本 3x4 仿射 + grid。
    affines = _build_rotation_matrices(angles, scales, translations, aspect)
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
    angles: torch.Tensor, scales: torch.Tensor,
    translations: Optional[torch.Tensor] = None,
    aspect: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """从欧拉角（N×3 rad）与 scales（N×1）构建 (N,3,4) 仿射矩阵。

    translations：(N,3) 归一化平移；None=无平移。
    aspect：(3,) 各轴物理尺度比例（轴序 (x,y,z)=(W,H,D)）；非 None 时对旋转
    做共轭校正 R←A⁻¹RA，使旋转在物理各向同性坐标里进行（各同性 scale
    与对角阵可交换，不受影响）。
    """
    N = angles.shape[0]
    device = angles.device

    cx, cy, cz = angles[:, 0].cos(), angles[:, 1].cos(), angles[:, 2].cos()
    sx, sy, sz = angles[:, 0].sin(), angles[:, 1].sin(), angles[:, 2].sin()

    # R = Rz @ Ry @ Rx。
    r00 = cy * cz
    r01 = sx * sy * cz - cx * sz
    r02 = cx * sy * cz + sx * sz
    r10 = cy * sz
    r11 = sx * sy * sz + cx * cz
    r12 = cx * sy * sz - sx * cz
    r20 = -sy
    r21 = sx * cy
    r22 = cx * cy

    rot = torch.stack([
        r00, r01, r02,
        r10, r11, r12,
        r20, r21, r22,
    ], dim=-1)
    rot = rearrange(rot, 'n (r c) -> n r c', r=3, c=3)

    if aspect is not None:
        a = aspect.to(device=device, dtype=rot.dtype)
        rot = torch.diag(1.0 / a) @ rot @ torch.diag(a)

    m = scales.view(N, 1, 1) * rot  # (N,3,3)

    if translations is None:
        t = torch.zeros(N, 3, 1, device=device, dtype=m.dtype)
    else:
        t = translations.view(N, 3, 1).to(m.dtype)

    return torch.cat([m, t], dim=-1)  # (N,3,4)


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
    voxel_to_grid = rearrange(
        torch.tensor([2.0 / W, 2.0 / H, 2.0 / D],
                     dtype=disp.dtype, device=device),
        'c -> 1 c 1 1 1')
    disp = disp * alpha * voxel_to_grid

    grid = _identity_grid(n, D, H, W, device) + rearrange(
        disp, 'b c d h w -> b d h w c')

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
    gate = rearrange(selected, 'b -> b 1 1 1 1').to(image.dtype)
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
    # 动态阐 (B,) → (B, 1, 1, ..., 1) 以适应 image.ndim。
    gpattern = 'b -> b' + ' 1' * (image.ndim - 1)
    gamma = rearrange(gamma, gpattern).to(image.dtype)

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
    """可分离 3D 高斯模糊；逐样本独立采样 sigma。

    全部选中样本一次性向量化处理：核长统一取 sigma 上界对应的 ks（小 sigma
    样本仅多出近零尾部，归一化后等价），逐样本核通过 grouped conv 并行。"""
    if prob <= 0:
        return image
    B, C = image.shape[:2]
    device = image.device
    mask = torch.rand(B, device=device) < prob
    if not mask.any():
        return image

    idx = mask.nonzero(as_tuple=True)[0]
    n = idx.numel()
    sigmas = torch.empty(n, device=device, dtype=image.dtype).uniform_(
        sigma_range[0], sigma_range[1])  # (n,)
    ks = max(int(2 * round(3 * float(sigma_range[1])) + 1), 3)
    pad = ks // 2
    x = torch.arange(ks, dtype=image.dtype, device=device) - pad  # (ks,)
    k1d = torch.exp(-0.5 * (x[None, :] / sigmas[:, None]) ** 2)   # (n, ks)
    k1d = k1d / k1d.sum(dim=1, keepdim=True)
    kc = k1d.repeat_interleave(C, dim=0)  # (n*C, ks)，同样本各通道共用同核

    # 将 (n,C) 折入通道轴，groups=n*C 使每通道用自己的核。
    sub = rearrange(image[idx], 'n c d h w -> 1 (n c) d h w')
    nc = n * C
    for axis_pat, pad_arg in (
        ('g k -> g 1 k 1 1', [0, 0, 0, 0, pad, pad]),
        ('g k -> g 1 1 k 1', [0, 0, pad, pad, 0, 0]),
        ('g k -> g 1 1 1 k', [pad, pad, 0, 0, 0, 0]),
    ):
        k = rearrange(kc, axis_pat)
        sub = F.pad(sub, pad_arg, mode="replicate")
        sub = F.conv3d(sub, k, groups=nc)

    image[idx] = rearrange(sub, '1 (n c) d h w -> n c d h w', n=n)
    return image


def _simulate_lowres(
    image: torch.Tensor, prob: float, zoom_range: list) -> torch.Tensor:
    """trilinear 下采→上采模拟低分辨率；逐样本独立采样 zoom。

    zoom 一次性批量采样（单次同步）；目标尺寸相同的样本分组后批量
    interpolate（不同尺寸无法单次处理，只能逐组）。"""
    if prob <= 0:
        return image
    B = image.shape[0]
    mask = torch.rand(B, device=image.device) < prob
    if not mask.any():
        return image
    _, _, D, H, W = image.shape
    idxs = mask.nonzero(as_tuple=True)[0].tolist()
    zooms = torch.empty(len(idxs)).uniform_(zoom_range[0], zoom_range[1]).tolist()
    groups: dict = {}
    for i, z in zip(idxs, zooms):
        if z >= 0.99:
            continue
        size = (max(1, int(D * z)), max(1, int(H * z)), max(1, int(W * z)))
        groups.setdefault(size, []).append(i)
    for size, members in groups.items():
        small = F.interpolate(
            image[members], size=size, mode="trilinear", align_corners=False)
        image[members] = F.interpolate(
            small, size=(D, H, W), mode="trilinear", align_corners=False)
    return image
