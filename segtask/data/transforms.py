"""GPU-accelerated data augmentation for segmentation.

All transforms operate on GPU tensors for maximum performance.
Image and label are transformed together for spatial transforms,
while intensity transforms only affect the image.

Usage:
    augmentor = GPUAugmentor(cfg.augment, spatial_dims=2)
    image, label = augmentor(image_gpu, label_gpu)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..config import AugmentConfig


class GPUAugmentor:
    """GPU-based data augmentation pipeline.

    Supports both 2D (B, C, H, W) and 3D (B, C, D, H, W) tensors.
    Spatial transforms are applied jointly to image and label.
    Intensity transforms are applied only to image.
    """

    def __init__(self, cfg: AugmentConfig, spatial_dims: int = 2):
        self.cfg = cfg
        self.spatial_dims = spatial_dims
        self.enabled = cfg.enabled

    @torch.no_grad()
    def __call__(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentation pipeline.

        Args:
            image: (B, C, H, W) or (B, C, D, H, W)
            label: (B, num_classes, H, W) or (B, num_classes, D, H, W)

        Returns:
            Augmented (image, label) on the same device.
        """
        if not self.enabled:
            return image, label

        # --- Spatial transforms (applied to both image and label) ---
        image, label = self._random_flip(image, label)
        image, label = self._random_affine(image, label)

        # --- Intensity transforms (image only) ---
        image = self._random_brightness(image)
        image = self._random_contrast(image)
        image = self._random_gamma(image)
        image = self._random_noise(image)
        image = self._random_blur(image)

        return image, label

    # ------------------------------------------------------------------
    # Spatial transforms
    # ------------------------------------------------------------------
    def _random_flip(
        self, image: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random flip along specified axes."""
        cfg = self.cfg
        if not cfg.random_flip_axes:
            return image, label

        for ax in cfg.random_flip_axes:
            if torch.rand(1).item() < cfg.random_flip_prob:
                # ax=0 → H, ax=1 → W, ax=2 → D (for 3D)
                # In tensor layout: 2D=(B,C,H,W), flip_dim = ax+2
                # 3D=(B,C,D,H,W), flip_dim = ax+2
                flip_dim = ax + 2
                if flip_dim < image.ndim:
                    image = torch.flip(image, [flip_dim])
                    label = torch.flip(label, [flip_dim])

        return image, label

    def _random_affine(
        self, image: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random rotation + scaling using affine grid."""
        cfg = self.cfg
        do_rotate = torch.rand(1).item() < cfg.random_rotate_prob
        do_scale = torch.rand(1).item() < cfg.random_scale_prob

        if not do_rotate and not do_scale:
            return image, label

        B = image.shape[0]
        device = image.device

        if self.spatial_dims == 2:
            return self._affine_2d(image, label, do_rotate, do_scale)
        else:
            return self._affine_3d(image, label, do_rotate, do_scale)

    def _affine_2d(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        do_rotate: bool,
        do_scale: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """2D affine transform."""
        B = image.shape[0]
        device = image.device
        cfg = self.cfg

        # Build 2x3 affine matrices per batch element
        theta = torch.eye(2, 3, device=device).unsqueeze(0).expand(B, -1, -1).clone()

        if do_rotate:
            angle = (torch.rand(B, device=device) * 2 - 1) * cfg.random_rotate_range
            angle_rad = angle * math.pi / 180.0
            cos_a = torch.cos(angle_rad)
            sin_a = torch.sin(angle_rad)
            rot = torch.zeros(B, 2, 2, device=device)
            rot[:, 0, 0] = cos_a
            rot[:, 0, 1] = -sin_a
            rot[:, 1, 0] = sin_a
            rot[:, 1, 1] = cos_a
            theta[:, :2, :2] = rot

        if do_scale:
            lo, hi = cfg.random_scale_range
            scale = torch.rand(B, device=device) * (hi - lo) + lo
            theta[:, 0, 0] *= scale
            theta[:, 0, 1] *= scale
            theta[:, 1, 0] *= scale
            theta[:, 1, 1] *= scale

        grid = F.affine_grid(theta, image.shape, align_corners=False)
        image = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros",
                              align_corners=False)
        label = F.grid_sample(label, grid, mode="nearest", padding_mode="zeros",
                              align_corners=False)

        return image, label

    def _affine_3d(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        do_rotate: bool,
        do_scale: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """3D affine transform (rotation around Z-axis + scaling)."""
        B = image.shape[0]
        device = image.device
        cfg = self.cfg

        theta = torch.eye(3, 4, device=device).unsqueeze(0).expand(B, -1, -1).clone()

        if do_rotate:
            angle = (torch.rand(B, device=device) * 2 - 1) * cfg.random_rotate_range
            angle_rad = angle * math.pi / 180.0
            cos_a = torch.cos(angle_rad)
            sin_a = torch.sin(angle_rad)
            # Rotate around D-axis (first spatial dim)
            theta[:, 1, 1] = cos_a
            theta[:, 1, 2] = -sin_a
            theta[:, 2, 1] = sin_a
            theta[:, 2, 2] = cos_a

        if do_scale:
            lo, hi = cfg.random_scale_range
            scale = torch.rand(B, device=device) * (hi - lo) + lo
            for i in range(3):
                theta[:, i, i] *= scale

        grid = F.affine_grid(theta, image.shape, align_corners=False)
        image = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros",
                              align_corners=False)
        label = F.grid_sample(label, grid, mode="nearest", padding_mode="zeros",
                              align_corners=False)

        return image, label

    # ------------------------------------------------------------------
    # Intensity transforms (image only)
    # ------------------------------------------------------------------
    def _random_brightness(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.cfg.random_brightness_prob:
            return image
        lo, hi = self.cfg.random_brightness_range
        offset = torch.rand(1, device=image.device) * (hi - lo) + lo
        return image + offset

    def _random_contrast(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.cfg.random_contrast_prob:
            return image
        lo, hi = self.cfg.random_contrast_range
        factor = torch.rand(1, device=image.device) * (hi - lo) + lo
        # Per-sample mean (keep batch dim, reduce all spatial + channel dims)
        dims = list(range(1, image.ndim))
        mean = image.mean(dim=dims, keepdim=True)
        return (image - mean) * factor + mean

    def _random_gamma(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.cfg.random_gamma_prob:
            return image
        lo, hi = self.cfg.random_gamma_range
        gamma = torch.rand(1).item() * (hi - lo) + lo
        # Normalize to [0,1], apply gamma, then restore original range
        img_min = image.min()
        img_max = image.max()
        rng = img_max - img_min
        if rng < 1e-8:
            return image
        normalized = (image - img_min) / rng
        return normalized.pow(gamma) * rng + img_min

    def _random_noise(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.cfg.gaussian_noise_prob:
            return image
        noise = torch.randn_like(image) * self.cfg.gaussian_noise_std
        return image + noise

    def _random_blur(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.cfg.gaussian_blur_prob:
            return image
        lo, hi = self.cfg.gaussian_blur_sigma
        sigma = torch.rand(1).item() * (hi - lo) + lo
        return self._gaussian_blur(image, sigma)

    def _gaussian_blur(self, image: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply Gaussian blur using separable convolution."""
        kernel_size = int(2 * round(3 * sigma) + 1)
        if kernel_size < 3:
            kernel_size = 3

        # Create 1D Gaussian kernel
        x = torch.arange(kernel_size, dtype=image.dtype, device=image.device)
        x = x - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        C = image.shape[1]

        if self.spatial_dims == 2:
            # Separable 2D blur
            k_h = kernel_1d.view(1, 1, -1, 1).expand(C, -1, -1, -1)
            k_w = kernel_1d.view(1, 1, 1, -1).expand(C, -1, -1, -1)
            pad_h = kernel_size // 2
            pad_w = kernel_size // 2
            image = F.pad(image, (0, 0, pad_h, pad_h), mode="reflect")
            image = F.conv2d(image, k_h, groups=C)
            image = F.pad(image, (pad_w, pad_w, 0, 0), mode="reflect")
            image = F.conv2d(image, k_w, groups=C)
        else:
            # Separable 3D blur (apply along each spatial dim)
            pad = kernel_size // 2
            for dim in range(3):
                shape = [1, 1, 1, 1, 1]
                shape[dim + 2] = kernel_size
                k = kernel_1d.view(*shape).expand(C, 1, *shape[2:])
                pad_list = [0] * 6
                pad_list[2 * (2 - dim)] = pad
                pad_list[2 * (2 - dim) + 1] = pad
                image = F.pad(image, pad_list, mode="reflect")
                image = F.conv3d(image, k, groups=C)

        return image


class MixupCutmix:
    """Batch-level Mixup augmentation for segmentation.

    Mixes pairs of images and labels within the same batch.
    """

    def __init__(self, alpha: float = 0.2, prob: float = 0.0):
        self.alpha = alpha
        self.prob = prob

    @torch.no_grad()
    def __call__(
        self, image: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.prob <= 0 or torch.rand(1).item() >= self.prob:
            return image, label

        B = image.shape[0]
        if B < 2:
            return image, label

        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        lam = max(lam, 1 - lam)  # Ensure lam >= 0.5

        perm = torch.randperm(B, device=image.device)
        image = lam * image + (1 - lam) * image[perm]
        # Mixup on one-hot labels is semantically incorrect (results in non-one-hot).
        # Only apply to image; label remains unchanged.
        # For soft-label mixup, use a separate implementation.

        return image, label
