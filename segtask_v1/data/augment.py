"""GPU-based 3D data augmentation for segmentation.

Spatial transforms (flip / affine / elastic / grid-dropout) are per-sample;
intensity transforms (brightness / contrast / gamma / noise / blur / lowres)
operate on image only. Inputs:
  image      (B, 1, D, H, W), label (B, C, D, H, W), weight_map (B, 1, D, H, W) optional.
weight_map receives the same spatial transforms as image/label; its
interpolation is set by ``AugConfig.wmap_interp_mode`` ("nearest" preserves
discrete fg/bg weights; "bilinear" for continuous hand-annotated weights).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from ..config import AugConfig


class GPUAugmentor:
    """GPU 3D augmentation pipeline (per-sample transforms).

    ``max_scale`` is the largest multi-res scale in the input; it scales
    down ``elastic_deform_alpha`` so the largest physical channel sees at
    most ``alpha`` voxels of displacement.
    """

    def __init__(self, cfg: AugConfig, max_scale: float = 1.0):
        self.cfg = cfg
        self.enabled = cfg.enabled
        self.max_scale = max(float(max_scale), 1.0)
        # wmap interp: "nearest" preserves discrete fg/bg weights (default);
        # "bilinear" for continuous weights. Only affine/elastic touch wmap.
        wmode = getattr(cfg, "wmap_interp_mode", "nearest")
        if wmode not in ("nearest", "bilinear"):
            raise ValueError(
                f"AugConfig.wmap_interp_mode={wmode!r}; expected "
                "'nearest' or 'bilinear'.")
        self.wmap_interp_mode = wmode

    def __call__(
        self, image: torch.Tensor, label: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Apply augmentations to a batch; returns ``(image, label, weight_map)``."""
        if not self.enabled:
            return image, label, weight_map

        c = self.cfg

        # Spatial: flip / affine / elastic / grid-dropout
        image, label, weight_map = _random_flip(
            image, label, c.random_flip_prob, c.random_flip_axes,
            weight_map=weight_map)
        image, label, weight_map = _random_affine(
            image, label, c.random_affine_prob, c.random_rotate_range,
            c.random_scale_range, weight_map=weight_map,
            wmap_mode=self.wmap_interp_mode)
        # Scale alpha down by max_scale so largest physical channel sees ≤ alpha voxels.
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


# ===========================================================================
# Spatial augmentations (per-sample independent)
# ===========================================================================
def _random_flip(
    image: torch.Tensor, label: torch.Tensor,
    prob: float, axes: list,
    weight_map: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Per-sample random flip; each sample and axis independently sampled."""
    B = image.shape[0]
    for axis in axes:
        mask = torch.rand(B, device=image.device) < prob  # (B,) bool
        if mask.any():
            idx = mask.nonzero(as_tuple=True)[0]
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
    """Per-sample random 3D affine (Euler rotation + isotropic scale) via grid_sample.

    All three streams use ``padding_mode="border"`` so out-of-bounds voxels
    replicate the boundary; this preserves the bg=1 contract of weight_map
    (produced by ``load_region_weight_volume``) instead of zeroing the loss.
    """
    B, _, D, H, W = image.shape
    device = image.device

    # Decide which samples get augmented
    mask = torch.rand(B, device=device) < prob
    if not mask.any():
        return image, label, weight_map

    # Sample rotation angles (radians) and scale per sample
    n = mask.sum().item()
    lo, hi = math.radians(rotate_range[0]), math.radians(rotate_range[1])
    angles = torch.empty(n, 3, device=device).uniform_(lo, hi)  # (n, 3) for x,y,z
    scales = torch.empty(n, 1, device=device).uniform_(scale_range[0], scale_range[1])

    # Build per-sample 3x4 affine matrices
    affines = _build_rotation_matrices(angles, scales)  # (n, 3, 4)

    # Generate grids
    grid = F.affine_grid(affines, [n, 1, D, H, W], align_corners=False)  # (n, D, H, W, 3)

    # Apply to selected samples
    idx = mask.nonzero(as_tuple=True)[0]
    image[idx] = F.grid_sample(
        image[idx], grid, mode="bilinear", padding_mode="border", align_corners=False)

    # Label: use nearest interpolation to preserve binary values
    label[idx] = F.grid_sample(label[idx], grid, mode="nearest", padding_mode="border", align_corners=False)

    # wmap: nearest=preserve discrete weights, bilinear=smooth continuous weights.
    if weight_map is not None:
        weight_map[idx] = F.grid_sample(
            weight_map[idx], grid, mode=wmap_mode,
            padding_mode="border", align_corners=False)

    return image, label, weight_map


def _build_rotation_matrices(
    angles: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Build (N, 3, 4) affine matrices from Euler angles (rad, N x 3) and scales (N x 1)."""
    N = angles.shape[0]
    device = angles.device

    cx, cy, cz = angles[:, 0].cos(), angles[:, 1].cos(), angles[:, 2].cos()
    sx, sy, sz = angles[:, 0].sin(), angles[:, 1].sin(), angles[:, 2].sin()

    # Rotation matrix R = Rz @ Ry @ Rx
    zeros = torch.zeros(N, device=device)

    # Row 0
    r00 = cy * cz
    r01 = sx * sy * cz - cx * sz
    r02 = cx * sy * cz + sx * sz
    # Row 1
    r10 = cy * sz
    r11 = sx * sy * sz + cx * cz
    r12 = cx * sy * sz - sx * cz
    # Row 2
    r20 = -sy
    r21 = sx * cy
    r22 = cx * cy

    s = scales.squeeze(-1)  # (N,)

    # Build 3x4: [s*R | 0]
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
    """Per-sample 3D elastic deformation via smooth random displacement field.

    sigma controls smoothness (typical 4–9, larger=smoother);
    alpha controls displacement magnitude in voxels (typical 3–12).
    """
    B, _, D, H, W = image.shape
    device = image.device

    mask = torch.rand(B, device=device) < prob
    if not mask.any():
        return image, label, weight_map

    idx = mask.nonzero(as_tuple=True)[0]
    n = idx.shape[0]

    # Coarse displacement, upsampled (trilinear acts as smoothing).
    cD = max(int(round(D / sigma)), 4)
    cH = max(int(round(H / sigma)), 4)
    cW = max(int(round(W / sigma)), 4)
    disp = torch.randn(n, 3, cD, cH, cW, device=device)
    disp = F.interpolate(disp, size=(D, H, W), mode="trilinear", align_corners=False)

    # Convert voxel displacement → normalised grid coords (1 voxel = 2/N for align_corners=False).
    # After permute, channels (0,1,2) map to grid axes (W, H, D).
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
    """Identity grid in [-1+1/s, 1-1/s] for grid_sample(align_corners=False)."""
    vecs = [torch.linspace(-1 + 1/s, 1 - 1/s, s, device=device) for s in (D, H, W)]
    grids = torch.meshgrid(*vecs, indexing="ij")  # (D, H, W) each
    grid = torch.stack(grids[::-1], dim=-1)  # (D, H, W, 3) — order: W, H, D for grid_sample
    return grid.unsqueeze(0).expand(N, -1, -1, -1, -1)


def _grid_dropout(
    image: torch.Tensor, label: torch.Tensor,
    prob: float, ratio: float, num_holes: int,
    weight_map: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Mask out ``num_holes`` rectangular sub-regions of the image with zeros.

    Label and weight_map are not masked (ground truth and loss weights are
    preserved inside dropped regions).
    """
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

    # Per-sample hole top-left corners (B, num_holes) — all sampled in one call.
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

    # effective = selected ? hole_mask : 1
    gate = selected.reshape(B, 1, 1, 1, 1).to(image.dtype)
    effective = hole_mask * gate + (1.0 - gate)
    return image * effective, label, weight_map


# ===========================================================================
# Intensity augmentations (per-sample independent)
# ===========================================================================
def _random_brightness(
    image: torch.Tensor, prob: float, brange: list) -> torch.Tensor:
    """Per-sample random additive brightness shift."""
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
    """Per-sample random multiplicative contrast around the per-channel mean pivot."""
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
    """Per-sample random gamma: minmax-normalise per-channel, pow(gamma), de-normalise."""
    if prob <= 0:
        return image
    B = image.shape[0]
    device = image.device
    mask = torch.rand(B, device=device) < prob  # (B,)
    if not mask.any():
        return image

    # Reduce over spatial dims only — keep channels independent (multi-res safe).
    reduce_dims = tuple(range(2, image.ndim))
    mn = image.amin(dim=reduce_dims, keepdim=True)  # (B,C,1,1,1)
    mx = image.amax(dim=reduce_dims, keepdim=True)
    rng = (mx - mn).clamp(min=1e-7)
    normed = ((image - mn) / rng).clamp(0.0, 1.0)

    # Identity gamma=1.0 for un-selected samples.
    gamma = torch.empty(B, device=device).uniform_(grange[0], grange[1])
    gamma = torch.where(mask, gamma, torch.ones_like(gamma))
    gshape = (B,) + (1,) * (image.ndim - 1)
    gamma = gamma.reshape(gshape).to(image.dtype)

    return normed.pow(gamma) * rng + mn


def _gaussian_noise(
    image: torch.Tensor, prob: float, std: float) -> torch.Tensor:
    """Per-sample additive Gaussian noise."""
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
    """Batched separable 3D Gaussian blur; one sigma shared across selected samples per call."""
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

    # Fold (B, C) into conv3d batch axis so the 1D kernel hits every (sample, channel) slice.
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
    """Simulate low-res by trilinear downsample→upsample; one zoom factor per call."""
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
