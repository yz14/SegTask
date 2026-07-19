"""GPU 3D 生成数据增强（segtask GPUAugmentor 的生成变体）。

与分割版差异：无 label；空间变换（flip/affine/elastic/grid-dropout）同步作用于
image + cond + weight_map（cond 是空间对齐的条件体，必须与 image 同 warp 保持
空间一致性）；强度变换仅作用于 image（cond 是独立模态、有自己的归一化）。

同步点约束：选样 Bernoulli 掩码与逐样本标量参数均在 CPU 上采样
（``_bernoulli_mask``），再异步搬到设备；避免对 CUDA RNG 结果的隐式
device→host 同步打断流水。元素级张量运算仍全部在 GPU 上。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange

from ..config import AugConfig


def _bernoulli_mask(
    n: int, prob: float,
    gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """CPU 上采样 (n,) bool 选样掩码；后续 any/sum/nonzero 均零同步。"""
    return torch.rand(n, generator=gen) < prob


class GPUAugmentor:
    """GPU 3D 增强管道（生成变体）。

    ``max_scale`` 为输入多分辨率最大 scale，用于缩小 elastic_deform_alpha，
    使最大物理通道位移 ≤ alpha 体素。

    输入均为 rank-5 ``(B, C, D, H, W)``（dataset 发出的 max-FOV cube 布局）；
    rank-4 2.5D 预打包输入自动升为 ``(B, 1, D, H, W)`` 处理后还原。
    """

    def __init__(self, cfg: AugConfig, max_scale: float = 1.0,
                 seed: Optional[int] = None,
                 inplace: Optional[bool] = None):
        self.cfg       = cfg
        self.enabled   = cfg.enabled
        self.max_scale = max(float(max_scale), 1.0)
        wmode = cfg.wmap_interp_mode
        if wmode not in ("nearest", "bilinear"):
            raise ValueError(
                f"AugConfig.wmap_interp_mode={wmode!r}; expected "
                "'nearest' or 'bilinear'.")
        self.wmap_interp_mode = wmode
        # inplace 覆写：调用方在自身拥有输入张量所有权时（如训练循环的 H2D
        # 私有拷贝）可显式传 True 跳过入口 clone；None 时沿用 cfg.inplace。
        self.inplace = bool(cfg.inplace if inplace is None else inplace)
        # 独立随机流（与 taskcore.data.augment 同构）：seed 非 None 时创建专属
        # CPU/设备 Generator；None 时沿用全局 RNG，行为与历史一致。
        self._seed: Optional[int] = None if seed is None else int(seed)
        self._gen_cpu: Optional[torch.Generator] = None
        self._gen_dev: Optional[torch.Generator] = None
        if self._seed is not None:
            self._gen_cpu = torch.Generator().manual_seed(self._seed)

    def _device_generator(
        self, device: torch.device) -> Optional[torch.Generator]:
        """返回与输入设备匹配的专属 Generator；未启用独立流时返 None。"""
        if self._seed is None:
            return None
        if device.type == "cpu":
            return self._gen_cpu
        if self._gen_dev is None or self._gen_dev.device != device:
            self._gen_dev = torch.Generator(device=device)
            self._gen_dev.manual_seed(self._seed + 1)
        return self._gen_dev

    def __call__(
        self,
        image: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """对 batch 应用增强；返回 (image, weight_map, cond)。"""
        if not self.enabled:
            return image, weight_map, cond

        squeeze_back = image.ndim == 4
        if squeeze_back:
            image = image.unsqueeze(1)
            if weight_map is not None and weight_map.ndim == 4:
                weight_map = weight_map.unsqueeze(1)
            if cond is not None and cond.ndim == 4:
                cond = cond.unsqueeze(1)

        if not self.inplace:
            image = image.clone()
            if weight_map is not None:
                weight_map = weight_map.clone()
            if cond is not None:
                cond = cond.clone()

        c = self.cfg
        gen_cpu = self._gen_cpu
        gen_dev = self._device_generator(image.device)

        # 开 intensity_clamp 时在任何增强前记录逐样本逐通道 min/max。
        if c.intensity_clamp:
            reduce_dims = tuple(range(2, image.ndim))
            clamp_lo = image.amin(dim=reduce_dims, keepdim=True)
            clamp_hi = image.amax(dim=reduce_dims, keepdim=True)

        # Spatial: flip / (affine+elastic 融合单次 warp) / grid-dropout
        image, weight_map, cond = _random_flip(
            image, c.random_flip_prob, c.random_flip_axes,
            weight_map=weight_map, cond=cond, gen_cpu=gen_cpu)
        effective_alpha = c.elastic_deform_alpha / self.max_scale
        image, weight_map, cond = _random_affine_elastic(
            image,
            affine_prob=c.random_affine_prob,
            rotate_range=c.random_rotate_range,
            scale_range=c.random_scale_range,
            elastic_prob=c.elastic_deform_prob,
            sigma=c.elastic_deform_sigma,
            alpha=effective_alpha,
            weight_map=weight_map,
            cond=cond,
            wmap_mode=self.wmap_interp_mode,
            translate_range=c.random_translate_range,
            rotate_range_per_axis=c.random_rotate_range_per_axis,
            aspect_correct=c.random_affine_aspect_correct,
            gen_cpu=gen_cpu, gen_dev=gen_dev)
        image = _grid_dropout(
            image, c.grid_dropout_prob, c.grid_dropout_ratio,
            c.grid_dropout_holes, gen_cpu=gen_cpu, gen_dev=gen_dev)

        # Intensity (image only)。
        image = _random_brightness(
            image, c.random_brightness_prob, c.random_brightness_range,
            gen_cpu=gen_cpu)
        image = _random_contrast(
            image, c.random_contrast_prob, c.random_contrast_range,
            gen_cpu=gen_cpu)
        image = _random_gamma(
            image, c.random_gamma_prob, c.random_gamma_range, gen_cpu=gen_cpu)
        image = _gaussian_noise(
            image, c.gaussian_noise_prob, c.gaussian_noise_std,
            gen_cpu=gen_cpu, gen_dev=gen_dev)
        image = _gaussian_blur_3d(
            image, c.gaussian_blur_prob, c.gaussian_blur_sigma,
            gen_cpu=gen_cpu)
        image = _simulate_lowres(
            image, c.simulate_lowres_prob, c.simulate_lowres_zoom,
            gen_cpu=gen_cpu)
        if c.intensity_clamp:
            image = torch.maximum(torch.minimum(image, clamp_hi), clamp_lo)

        if squeeze_back:
            image = image.squeeze(1)
            if weight_map is not None and weight_map.shape[1] == 1:
                weight_map = weight_map.squeeze(1)
            if cond is not None and cond.shape[1] == 1:
                cond = cond.squeeze(1)
        return image, weight_map, cond


# 空间增强（逐样本独立）。
def _random_flip(
    image: torch.Tensor,
    prob: float, axes: list,
    weight_map: Optional[torch.Tensor] = None,
    cond: Optional[torch.Tensor] = None,
    gen_cpu: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """逐样本随机翻转；每轴独立采样；image/cond/wmap 同步。"""
    B = image.shape[0]
    for axis in axes:
        mask = _bernoulli_mask(B, prob, gen_cpu)
        if mask.any():
            idx        = mask.nonzero(as_tuple=True)[0].to(image.device)
            image[idx] = torch.flip(image[idx], [axis])
            if weight_map is not None:
                weight_map[idx] = torch.flip(weight_map[idx], [axis])
            if cond is not None:
                cond[idx] = torch.flip(cond[idx], [axis])
    return image, weight_map, cond


def _build_rotation_matrices(
    angles: torch.Tensor, scales: torch.Tensor,
    translations: Optional[torch.Tensor] = None,
    aspect: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """从欧拉角（N×3 rad）与 scales（N×1）构建 (N,3,4) 仿射矩阵。"""
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


def _elastic_grid_disp(
    n: int, D: int, H: int, W: int,
    sigma: float, alpha: float, device: torch.device,
    gen_dev: Optional[torch.Generator] = None) -> torch.Tensor:
    """采样 n 个弹性位移场，返 (n,D,H,W,3) 归一化 grid 坐标位移（轴序 W,H,D）。"""
    cD = max(int(round(D / sigma)), 4)
    cH = max(int(round(H / sigma)), 4)
    cW = max(int(round(W / sigma)), 4)
    disp = torch.randn(n, 3, cD, cH, cW, device=device, generator=gen_dev)
    disp = F.interpolate(disp, size=(D, H, W), mode="trilinear", align_corners=False)

    voxel_to_grid = rearrange(
        torch.tensor([2.0 / W, 2.0 / H, 2.0 / D],
                     dtype=disp.dtype, device=device),
        'c -> 1 c 1 1 1')
    disp = disp * alpha * voxel_to_grid
    return rearrange(disp, 'b c d h w -> b d h w c')


def _random_affine_elastic(
    image: torch.Tensor,
    affine_prob: float, rotate_range: list, scale_range: list,
    elastic_prob: float, sigma: float, alpha: float,
    weight_map: Optional[torch.Tensor] = None,
    cond: Optional[torch.Tensor] = None,
    wmap_mode: str = "nearest",
    translate_range: Optional[list] = None,
    rotate_range_per_axis: Optional[list] = None,
    aspect_correct: bool = False,
    gen_cpu: Optional[torch.Generator] = None,
    gen_dev: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """仿射与弹性形变融合为单次 grid_sample；image/cond bilinear、wmap 按配置。"""
    B, _, D, H, W = image.shape
    device = image.device

    mask_a = _bernoulli_mask(B, affine_prob, gen_cpu)
    mask_e = _bernoulli_mask(B, elastic_prob, gen_cpu)
    mask = mask_a | mask_e
    if not mask.any():
        return image, weight_map, cond

    idx_cpu = mask.nonzero(as_tuple=True)[0]
    idx = idx_cpu.to(device)
    n = idx_cpu.shape[0]
    sel_a = mask_a[idx_cpu]
    sel_e = mask_e[idx_cpu]

    theta = torch.eye(3, 4, device=device).unsqueeze(0).repeat(n, 1, 1)
    na = int(sel_a.sum())
    if na:
        if rotate_range_per_axis is not None:
            angles = torch.empty(na, 3)
            for ax in range(3):
                lo = math.radians(rotate_range_per_axis[ax][0])
                hi = math.radians(rotate_range_per_axis[ax][1])
                angles[:, ax].uniform_(lo, hi, generator=gen_cpu)
        else:
            lo, hi = math.radians(rotate_range[0]), math.radians(rotate_range[1])
            angles = torch.empty(na, 3).uniform_(lo, hi, generator=gen_cpu)
        angles = angles.to(device, non_blocking=True)
        scales = torch.empty(na, 1).uniform_(
            scale_range[0], scale_range[1],
            generator=gen_cpu).to(device, non_blocking=True)

        translations = None
        if translate_range is not None and (
                translate_range[0] != 0.0 or translate_range[1] != 0.0):
            translations = torch.empty(na, 3).uniform_(
                translate_range[0],
                translate_range[1],
                generator=gen_cpu).to(device, non_blocking=True)

        aspect = None
        if aspect_correct:
            aspect = torch.tensor(
                [float(W), float(H), float(D)],
                device=device, dtype=torch.float32)
        theta[sel_a.to(device)] = _build_rotation_matrices(
            angles, scales, translations, aspect)

    grid = F.affine_grid(theta, [n, 1, D, H, W], align_corners=False)

    ne = int(sel_e.sum())
    if ne:
        sel_e_dev = sel_e.to(device)
        disp = _elastic_grid_disp(ne, D, H, W, sigma, alpha, device, gen_dev)
        m = theta[sel_e_dev][:, :, :3]  # (ne,3,3)
        grid[sel_e_dev] = grid[sel_e_dev] + torch.einsum(
            'n r c, n d h w c -> n d h w r', m, disp)

    image[idx] = F.grid_sample(
        image[idx], grid, mode="bilinear", padding_mode="border", align_corners=False)
    if weight_map is not None:
        weight_map[idx] = F.grid_sample(
            weight_map[idx], grid, mode=wmap_mode,
            padding_mode="border", align_corners=False)
    if cond is not None:
        cond[idx] = F.grid_sample(
            cond[idx], grid, mode="bilinear",
            padding_mode="border", align_corners=False)

    return image, weight_map, cond


def _grid_dropout(
    image: torch.Tensor,
    prob: float, ratio: float, num_holes: int,
    gen_cpu: Optional[torch.Generator] = None,
    gen_dev: Optional[torch.Generator] = None) -> torch.Tensor:
    """随机零掩 num_holes 个矩形区域（仅 image；cond/wmap 不被掩）。"""
    if prob <= 0 or ratio <= 0:
        return image

    B, _, D, H, W = image.shape
    device = image.device

    selected = _bernoulli_mask(B, prob, gen_cpu)
    if not selected.any():
        return image

    frac = (ratio / max(num_holes, 1)) ** (1.0 / 3.0)
    hd = min(D, max(1, int(D * frac)))
    hh = min(H, max(1, int(H * frac)))
    hw = min(W, max(1, int(W * frac)))

    # 逐样本 hole 左上角；合法起点为 0..axis-hole（randint 上界不含，故 +1）。
    d0 = torch.randint(0, D - hd + 1, (B, num_holes), device=device,
                       generator=gen_dev)
    h0 = torch.randint(0, H - hh + 1, (B, num_holes), device=device,
                       generator=gen_dev)
    w0 = torch.randint(0, W - hw + 1, (B, num_holes), device=device,
                       generator=gen_dev)

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

    gate = rearrange(selected.to(device), 'b -> b 1 1 1 1').to(image.dtype)
    effective = hole_mask * gate + (1.0 - gate)
    return image * effective


# 强度增强（逐样本独立；仅 image）。
def _random_brightness(
    image: torch.Tensor, prob: float, brange: list,
    gen_cpu: Optional[torch.Generator] = None) -> torch.Tensor:
    """逐样本随机加性亮度偏移。"""
    if prob <= 0:
        return image
    B = image.shape[0]
    mask = _bernoulli_mask(B, prob, gen_cpu)
    if not mask.any():
        return image
    shift = torch.empty(B, 1, 1, 1, 1).uniform_(
        brange[0], brange[1], generator=gen_cpu)
    shift[~mask] = 0
    return image + shift.to(image.device, non_blocking=True)


def _random_contrast(
    image: torch.Tensor, prob: float, crange: list,
    gen_cpu: Optional[torch.Generator] = None) -> torch.Tensor:
    """逐样本随机对比度，以逐通道均值为轴。"""
    if prob <= 0:
        return image
    B = image.shape[0]
    mask = _bernoulli_mask(B, prob, gen_cpu)
    if not mask.any():
        return image
    spatial_dims = tuple(range(2, image.ndim))
    mean = image.mean(dim=spatial_dims, keepdim=True)
    factor = torch.ones(B, 1, 1, 1, 1)
    factor[mask] = torch.empty(
        int(mask.sum()), 1, 1, 1, 1).uniform_(
            crange[0], crange[1], generator=gen_cpu)
    return (image - mean) * factor.to(image.device, non_blocking=True) + mean


def _random_gamma(
    image: torch.Tensor, prob: float, grange: list,
    gen_cpu: Optional[torch.Generator] = None) -> torch.Tensor:
    """逐样本随机 gamma：逐通道 minmax 归一→pow(gamma)→反归一。"""
    if prob <= 0:
        return image
    B = image.shape[0]
    mask = _bernoulli_mask(B, prob, gen_cpu)
    if not mask.any():
        return image

    reduce_dims = tuple(range(2, image.ndim))
    mn = image.amin(dim=reduce_dims, keepdim=True)
    mx = image.amax(dim=reduce_dims, keepdim=True)
    rng = (mx - mn).clamp(min=1e-7)
    normed = ((image - mn) / rng).clamp(0.0, 1.0)

    gamma = torch.empty(B).uniform_(grange[0], grange[1], generator=gen_cpu)
    gamma = torch.where(mask, gamma, torch.ones_like(gamma))
    gamma = gamma.to(image.device, non_blocking=True)
    gpattern = 'b -> b' + ' 1' * (image.ndim - 1)
    gamma = rearrange(gamma, gpattern).to(image.dtype)

    return normed.pow(gamma) * rng + mn


def _gaussian_noise(
    image: torch.Tensor, prob: float, std: float,
    gen_cpu: Optional[torch.Generator] = None,
    gen_dev: Optional[torch.Generator] = None) -> torch.Tensor:
    """逐样本加性高斯噪声。"""
    if prob <= 0:
        return image
    B = image.shape[0]
    mask = _bernoulli_mask(B, prob, gen_cpu)
    if not mask.any():
        return image
    idx = mask.nonzero(as_tuple=True)[0].to(image.device)
    sub = image[idx]
    noise = torch.randn(
        sub.shape, dtype=sub.dtype, device=sub.device, generator=gen_dev)
    image[idx] = sub + noise * std
    return image


def _gaussian_blur_3d(
    image: torch.Tensor, prob: float, sigma_range: list,
    gen_cpu: Optional[torch.Generator] = None) -> torch.Tensor:
    """可分离 3D 高斯模糊；逐样本独立采样 sigma。"""
    if prob <= 0:
        return image
    B, C = image.shape[:2]
    device = image.device
    mask = _bernoulli_mask(B, prob, gen_cpu)
    if not mask.any():
        return image

    idx = mask.nonzero(as_tuple=True)[0].to(device)
    n = idx.numel()
    sigmas = torch.empty(n, dtype=image.dtype).uniform_(
        sigma_range[0], sigma_range[1],
        generator=gen_cpu).to(device, non_blocking=True)
    ks = max(int(2 * round(3 * float(sigma_range[1])) + 1), 3)
    pad = ks // 2
    x = torch.arange(ks, dtype=image.dtype, device=device) - pad
    k1d = torch.exp(-0.5 * (x[None, :] / sigmas[:, None]) ** 2)
    k1d = k1d / k1d.sum(dim=1, keepdim=True)
    kc = k1d.repeat_interleave(C, dim=0)

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
    image: torch.Tensor, prob: float, zoom_range: list,
    gen_cpu: Optional[torch.Generator] = None) -> torch.Tensor:
    """trilinear 下采→上采模拟低分辨率；逐样本独立采样 zoom。"""
    if prob <= 0:
        return image
    B = image.shape[0]
    mask = _bernoulli_mask(B, prob, gen_cpu)
    if not mask.any():
        return image
    _, _, D, H, W = image.shape
    idxs = mask.nonzero(as_tuple=True)[0].tolist()
    zooms = torch.empty(len(idxs)).uniform_(
        zoom_range[0], zoom_range[1], generator=gen_cpu).tolist()
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


__all__ = ["GPUAugmentor"]
