"""GPU-based 3D data augmentation for segmentation.

All augmentations operate on CUDA tensors for speed.
Spatial transforms are per-sample independent (not batch-level).
Intensity transforms are applied to image only.

Spatial augmentations:
  - Random flip (per-sample, per-axis independent)
  - Random affine (rotation + scale via grid_sample, per-sample)
  - Elastic deformation (smooth random displacement field, per-sample)
  - Grid dropout (mask out rectangular sub-regions)

Intensity augmentations:
  - Brightness, contrast, gamma, noise, blur, simulate low-res

Input shapes:
  image: (B, 1, D, H, W) float32
  label: (B, C, D, H, W) float32 binary masks (may include weight_map channel)
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F

from ..config import AugConfig


class GPUAugmentor:
    """GPU-based 3D data augmentation pipeline with per-sample transforms."""

    def __init__(self, cfg: AugConfig):
        self.cfg = cfg
        self.enabled = cfg.enabled

    def __call__(
        self, image: torch.Tensor, label: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentations to a batch.

        Args:
            image: (B, 1, D, H, W) on GPU.
            label: (B, C, D, H, W) on GPU. May include extra channels
                   (e.g., weight_map concatenated by trainer).

        Returns:
            Augmented (image, label).
        """
        if not self.enabled:
            return image, label

        c = self.cfg

        # --- Spatial augmentations (image + label, per-sample) ---
        image, label = _random_flip(
            image, label, c.random_flip_prob, c.random_flip_axes)
        image, label = _random_affine(
            image, label, c.random_affine_prob, c.random_rotate_range, c.random_scale_range)
        image, label = _elastic_deform(
            image, label, c.elastic_deform_prob, c.elastic_deform_sigma, c.elastic_deform_alpha)
        image, label = _grid_dropout(
            image, label, c.grid_dropout_prob, c.grid_dropout_ratio, c.grid_dropout_holes)

        # --- Intensity augmentations (image only, per-sample) ---
        image = _random_brightness(image, c.random_brightness_prob, c.random_brightness_range)
        image = _random_contrast(image, c.random_contrast_prob, c.random_contrast_range)
        image = _random_gamma(image, c.random_gamma_prob, c.random_gamma_range)
        image = _gaussian_noise(image, c.gaussian_noise_prob, c.gaussian_noise_std)
        image = _gaussian_blur_3d(image, c.gaussian_blur_prob, c.gaussian_blur_sigma)
        image = _simulate_lowres(image, c.simulate_lowres_prob, c.simulate_lowres_zoom)

        return image, label


# ===========================================================================
# Spatial augmentations (per-sample independent)
# ===========================================================================
def _random_flip(
    image: torch.Tensor, label: torch.Tensor,
    prob: float, axes: list) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-sample random flip. Each sample and axis is independently random."""
    B = image.shape[0]
    for axis in axes:
        mask = torch.rand(B, device=image.device) < prob  # (B,) bool
        if mask.any():
            idx = mask.nonzero(as_tuple=True)[0]
            image[idx] = torch.flip(image[idx], [axis])  # axis indexes into (B,C,D,H,W)
            label[idx] = torch.flip(label[idx], [axis])
    return image, label


def _random_affine(
    image: torch.Tensor, label: torch.Tensor,
    prob: float, rotate_range: list, scale_range: list) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-sample random 3D affine (rotation around all axes + scale).

    Uses F.grid_sample with a per-sample affine matrix for efficiency.
    Rotation angles are sampled uniformly in rotate_range (degrees).
    Scale is sampled uniformly in scale_range.
    """
    B, _, D, H, W = image.shape
    device = image.device

    # Decide which samples get augmented
    mask = torch.rand(B, device=device) < prob
    if not mask.any():
        return image, label

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
        image[idx], grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # Label: use nearest interpolation to preserve binary values
    label[idx] = F.grid_sample(label[idx], grid, mode="nearest", padding_mode="zeros", align_corners=False)

    return image, label


def _build_rotation_matrices(
    angles: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Build 3x4 affine matrices from Euler angles (x,y,z) and isotropic scale.

    Args:
        angles: (N, 3) rotation angles in radians.
        scales: (N, 1) scale factors.

    Returns:
        (N, 3, 4) affine matrices.
    """
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
    prob: float, sigma: float, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-sample 3D elastic deformation via smooth random displacement field.

    Algorithm:
    1. Generate random displacement on a coarse grid (N(0,1))
    2. Upsample to full resolution (trilinear = smooth interpolation)
    3. Scale so that ``alpha`` controls displacement in **voxels**
    4. Convert per-dimension to normalised grid coordinates for grid_sample
    5. Apply via grid_sample

    Args:
        sigma: Controls the *smoothness* of the deformation field.
            Larger → smoother (fewer coarse control points).
            Typical range: 4–9.
        alpha: Controls the *magnitude* of displacement in **voxels**.
            After interpolation the displacement std ≈ alpha voxels.
            Typical range: 3–12.  (95 % of displacements within ±2·alpha voxels.)
    """
    B, _, D, H, W = image.shape
    device = image.device

    mask = torch.rand(B, device=device) < prob
    if not mask.any():
        return image, label

    idx = mask.nonzero(as_tuple=True)[0]
    n = idx.shape[0]

    # Coarse grid size (controls smoothness — fewer points = smoother)
    cD = max(int(round(D / sigma)), 4)
    cH = max(int(round(H / sigma)), 4)
    cW = max(int(round(W / sigma)), 4)

    # Random displacement on coarse grid, then upsample (acts as smoothing)
    disp = torch.randn(n, 3, cD, cH, cW, device=device)
    disp = F.interpolate(disp, size=(D, H, W), mode="trilinear", align_corners=False)

    # Scale displacement to voxel-space magnitude ``alpha``, then convert
    # each channel to normalised grid coordinates independently.
    #
    # After permute(0,2,3,4,1) the 3 channels map to grid axes as:
    #   channel 0 → x (W-axis)
    #   channel 1 → y (H-axis)
    #   channel 2 → z (D-axis)
    #
    # For align_corners=False, 1 voxel = 2/N in grid coords (grid spans
    # [-1, 1] over N pixels).  So: grid_disp = voxel_disp × (2 / N).
    voxel_to_grid = torch.tensor(
        [2.0 / W, 2.0 / H, 2.0 / D],
        dtype=disp.dtype, device=device,
    ).reshape(1, 3, 1, 1, 1)
    disp = disp * alpha * voxel_to_grid

    # Build sampling grid: identity + displacement
    # grid_sample expects grid in [-1, 1], shape (N, D, H, W, 3)
    grid = _identity_grid(n, D, H, W, device)  # (n, D, H, W, 3)
    grid = grid + disp.permute(0, 2, 3, 4, 1)  # add displacement

    # Apply to image
    image[idx] = F.grid_sample(
        image[idx], grid, mode="bilinear", padding_mode="border", align_corners=False)

    # Apply to label (nearest to preserve discrete values)
    label[idx] = F.grid_sample(label[idx], grid, mode="nearest", padding_mode="border", align_corners=False)

    return image, label


def _identity_grid(
    N: int, D: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    """Create identity sampling grid in [-1, 1] for grid_sample (align_corners=False).

    For align_corners=False, pixel i (0-indexed) maps to coordinate (2i+1)/s - 1,
    i.e. coordinates span [-1+1/s, 1-1/s] instead of [-1, 1].
    """
    vecs = [torch.linspace(-1 + 1/s, 1 - 1/s, s, device=device) for s in (D, H, W)]
    grids = torch.meshgrid(*vecs, indexing="ij")  # (D, H, W) each
    grid = torch.stack(grids[::-1], dim=-1)  # (D, H, W, 3) — order: W, H, D for grid_sample
    return grid.unsqueeze(0).expand(N, -1, -1, -1, -1)


def _grid_dropout(
    image: torch.Tensor, label: torch.Tensor,
    prob: float, ratio: float, num_holes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Grid dropout: mask out rectangular sub-regions with zeros.

    Vectorized: generates all B × num_holes hole positions in a single
    batched `torch.randint` call and marks the boolean mask via advanced
    indexing. Label is NOT masked (to preserve ground truth).

    Samples not selected by the Bernoulli mask pass through unchanged
    (their hole-mask stays all-ones via `selected` gating).
    """
    if prob <= 0 or ratio <= 0:
        return image, label

    B, _, D, H, W = image.shape
    device = image.device

    selected = torch.rand(B, device=device) < prob  # (B,)
    if not selected.any():
        return image, label

    # Hole sizes are constant per call — ratio and num_holes are scalars.
    frac = (ratio / max(num_holes, 1)) ** (1.0 / 3.0)
    hd = max(1, int(D * frac))
    hh = max(1, int(H * frac))
    hw = max(1, int(W * frac))

    # Sample B × num_holes top-left corners in a single kernel launch.
    d0 = torch.randint(0, max(D - hd, 1), (B, num_holes), device=device)
    h0 = torch.randint(0, max(H - hh, 1), (B, num_holes), device=device)
    w0 = torch.randint(0, max(W - hw, 1), (B, num_holes), device=device)

    # Build per-sample hole mask without per-sample Python loops.
    hole_mask = torch.ones(B, 1, D, H, W, device=device, dtype=image.dtype)
    # Pre-compute axis offset vectors, then broadcast-set via advanced indexing
    # one hole at a time — keep the inner dim small (num_holes is typically < 8).
    d_off = torch.arange(hd, device=device)  # (hd,)
    h_off = torch.arange(hh, device=device)
    w_off = torch.arange(hw, device=device)
    for k in range(num_holes):
        # (B,), per-sample start indices for this hole
        ds = d0[:, k, None] + d_off[None, :]      # (B, hd)
        hs = h0[:, k, None] + h_off[None, :]      # (B, hh)
        ws = w0[:, k, None] + w_off[None, :]      # (B, hw)
        b_idx = torch.arange(B, device=device)
        # Fancy-index over (B, D, H, W): broadcast b_idx × ds × hs × ws
        # to a (B, hd, hh, hw) cartesian product.
        hole_mask[
            b_idx[:, None, None, None], :,
            ds[:, :, None, None],
            hs[:, None, :, None],
            ws[:, None, None, :],
        ] = 0

    # Only zero-out samples that were selected; un-selected keep identity mask
    gate = selected.reshape(B, 1, 1, 1, 1).to(image.dtype)
    # effective_mask = selected ? hole_mask : 1
    effective = hole_mask * gate + (1.0 - gate)
    return image * effective, label


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
    """Per-sample random multiplicative contrast."""
    if prob <= 0:
        return image
    B = image.shape[0]
    mask = torch.rand(B, device=image.device) < prob
    if not mask.any():
        return image
    mean = image.reshape(B, -1).mean(dim=1).reshape(B, 1, 1, 1, 1)
    factor = torch.ones(B, 1, 1, 1, 1, device=image.device)
    factor[mask] = torch.empty(mask.sum().item(), 1, 1, 1, 1, device=image.device).uniform_(crange[0], crange[1])
    return (image - mean) * factor + mean


def _random_gamma(
    image: torch.Tensor, prob: float, grange: list) -> torch.Tensor:
    """Per-sample random gamma correction, fully vectorized.

    For each sample independently:
      1. Min/max-normalize to [0, 1] per-sample (stats taken over C,D,H,W).
      2. pow(gamma_i) with gamma_i sampled per-sample from grange.
      3. De-normalize back to the original intensity range.

    Samples not selected by the Bernoulli mask are returned unchanged
    by setting their effective gamma to 1.0 (identity).
    """
    if prob <= 0:
        return image
    B = image.shape[0]
    device = image.device
    mask = torch.rand(B, device=device) < prob  # (B,)
    if not mask.any():
        return image

    # Per-sample min/max over non-batch dims. amin/amax with a tuple is the
    # standard vectorised path and needs a single kernel launch per reduce.
    reduce_dims = tuple(range(1, image.ndim))
    mn = image.amin(dim=reduce_dims, keepdim=True)  # (B,1,1,1,1)
    mx = image.amax(dim=reduce_dims, keepdim=True)
    rng = (mx - mn).clamp(min=1e-7)
    normed = ((image - mn) / rng).clamp(0.0, 1.0)

    # Per-sample gamma; identity (1.0) for samples not selected.
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
    """Per-sample 3D Gaussian blur via separable 1D convolutions."""
    if prob <= 0:
        return image
    B = image.shape[0]
    mask = torch.rand(B, device=image.device) < prob
    if not mask.any():
        return image

    idx = mask.nonzero(as_tuple=True)[0]
    for i in idx:
        sigma = torch.empty(1).uniform_(sigma_range[0], sigma_range[1]).item()
        ks = int(2 * round(3 * sigma) + 1)
        if ks < 3:
            ks = 3
        x = torch.arange(ks, dtype=image.dtype, device=image.device) - ks // 2
        k1d = torch.exp(-0.5 * (x / sigma) ** 2)
        k1d = k1d / k1d.sum()
        pad = ks // 2

        img = image[i:i + 1]  # (1, C, D, H, W)
        C = img.shape[1]
        img = img.reshape(C, 1, *img.shape[2:])

        for k_shape in [(-1, 1, 1), (1, -1, 1), (1, 1, -1)]:
            k = k1d.reshape(1, 1, *k_shape)
            pad_arg = [0, 0, 0, 0, 0, 0]
            if k_shape[0] != 1:
                pad_arg = [0, 0, 0, 0, pad, pad]
            elif k_shape[1] != 1:
                pad_arg = [0, 0, pad, pad, 0, 0]
            else:
                pad_arg = [pad, pad, 0, 0, 0, 0]
            img = F.pad(img, pad_arg, mode="replicate")
            img = F.conv3d(img, k)

        image[i] = img.reshape(1, C, *img.shape[2:])[0]
    return image


def _simulate_lowres(
    image: torch.Tensor, prob: float, zoom_range: list) -> torch.Tensor:
    """Per-sample simulate low resolution by downsample→upsample."""
    if prob <= 0:
        return image
    B = image.shape[0]
    mask = torch.rand(B, device=image.device) < prob
    if not mask.any():
        return image
    _, _, D, H, W = image.shape
    idx = mask.nonzero(as_tuple=True)[0]
    for i in idx:
        z = torch.empty(1).uniform_(zoom_range[0], zoom_range[1]).item()
        if z >= 0.99:
            continue
        small = F.interpolate(
            image[i:i + 1],
            size=(max(1, int(D * z)), max(1, int(H * z)), max(1, int(W * z))),
            mode="trilinear", align_corners=False)
        image[i:i + 1] = F.interpolate(
            small, size=(D, H, W), mode="trilinear", align_corners=False)
    return image
