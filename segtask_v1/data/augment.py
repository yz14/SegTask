"""GPU-based data augmentation for 3D segmentation.

All augmentations operate on CUDA tensors for speed.
Spatial transforms are applied to both image and label.
Intensity transforms are applied to image only.

Input shapes:
  image: (B, 1, D, H, W) float32
  label: (B, C, D, H, W) float32 binary masks
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from ..config import AugConfig


class GPUAugmentor:
    """GPU-based 3D data augmentation pipeline."""

    def __init__(self, cfg: AugConfig):
        self.cfg = cfg
        self.enabled = cfg.enabled

    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentations to a batch.

        Args:
            image: (B, 1, D, H, W) on GPU.
            label: (B, C, D, H, W) on GPU.

        Returns:
            Augmented (image, label).
        """
        if not self.enabled:
            return image, label

        c = self.cfg

        # --- Spatial augmentations (image + label) ---
        image, label = self._random_flip(image, label, c.random_flip_prob, c.random_flip_axes)
        image, label = self._random_rotate90(image, label, c.random_rotate90_prob)
        image, label = self._random_scale(image, label, c.random_scale_prob, c.random_scale_range)

        # --- Intensity augmentations (image only) ---
        image = self._random_brightness(image, c.random_brightness_prob, c.random_brightness_range)
        image = self._random_contrast(image, c.random_contrast_prob, c.random_contrast_range)
        image = self._random_gamma(image, c.random_gamma_prob, c.random_gamma_range)
        image = self._gaussian_noise(image, c.gaussian_noise_prob, c.gaussian_noise_std)
        image = self._gaussian_blur(image, c.gaussian_blur_prob, c.gaussian_blur_sigma)

        return image, label

    # -----------------------------------------------------------------------
    # Spatial augmentations
    # -----------------------------------------------------------------------
    @staticmethod
    def _random_flip(image: torch.Tensor, label: torch.Tensor, prob: float, axes: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random flip along specified axes. Each axis flipped independently."""
        for axis in axes:
            if torch.rand(1).item() < prob:
                image = torch.flip(image, [axis])
                label = torch.flip(label, [axis])
        return image, label

    @staticmethod
    def _random_rotate90(image: torch.Tensor, label: torch.Tensor, prob: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random 90-degree rotation in the H-W plane (dims 3,4)."""
        if torch.rand(1).item() < prob:
            k = torch.randint(1, 4, (1,)).item()  # 1, 2, or 3 times
            image = torch.rot90(image, k, dims=[3, 4])
            label = torch.rot90(label, k, dims=[3, 4])
        return image, label

    @staticmethod
    def _random_scale(image: torch.Tensor, label: torch.Tensor, prob: float, scale_range: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random isotropic scaling via trilinear interpolation."""
        if torch.rand(1).item() >= prob:
            return image, label

        scale = torch.empty(1).uniform_(scale_range[0], scale_range[1]).item()
        _, _, D, H, W = image.shape
        new_D = max(1, int(round(D * scale)))
        new_H = max(1, int(round(H * scale)))
        new_W = max(1, int(round(W * scale)))

        image_s = F.interpolate(image, size=(new_D, new_H, new_W),
                                mode="trilinear", align_corners=False)
        label_s = F.interpolate(label, size=(new_D, new_H, new_W),
                                mode="nearest")

        # Crop or pad back to original size
        image = _crop_or_pad(image_s, D, H, W, value=0.0)
        label = _crop_or_pad(label_s, D, H, W, value=0.0)
        return image, label

    # -----------------------------------------------------------------------
    # Intensity augmentations
    # -----------------------------------------------------------------------
    @staticmethod
    def _random_brightness(
        image: torch.Tensor, prob: float, brange: list,
    ) -> torch.Tensor:
        """Per-sample random additive brightness shift."""
        if torch.rand(1).item() >= prob:
            return image
        B = image.shape[0]
        shift = torch.empty(B, 1, 1, 1, 1, device=image.device).uniform_(brange[0], brange[1])
        return image + shift

    @staticmethod
    def _random_contrast(
        image: torch.Tensor, prob: float, crange: list,
    ) -> torch.Tensor:
        """Per-sample random multiplicative contrast."""
        if torch.rand(1).item() >= prob:
            return image
        B = image.shape[0]
        # Compute per-sample mean
        mean = image.reshape(B, -1).mean(dim=1).reshape(B, 1, 1, 1, 1)
        factor = torch.empty(B, 1, 1, 1, 1, device=image.device).uniform_(crange[0], crange[1])
        return (image - mean) * factor + mean

    @staticmethod
    def _random_gamma(
        image: torch.Tensor, prob: float, grange: list,
    ) -> torch.Tensor:
        """Per-sample random gamma correction. Assumes image ∈ [0, 1]."""
        if torch.rand(1).item() >= prob:
            return image
        B = image.shape[0]
        gamma = torch.empty(B, 1, 1, 1, 1, device=image.device).uniform_(grange[0], grange[1])
        # Clamp to [0,1] before gamma, then restore range
        img_min = image.reshape(B, -1).min(dim=1).values.reshape(B, 1, 1, 1, 1)
        img_max = image.reshape(B, -1).max(dim=1).values.reshape(B, 1, 1, 1, 1)
        img_range = (img_max - img_min).clamp(min=1e-7)
        normalized = ((image - img_min) / img_range).clamp(0, 1)
        return normalized.pow(gamma) * img_range + img_min

    @staticmethod
    def _gaussian_noise(
        image: torch.Tensor, prob: float, std: float,
    ) -> torch.Tensor:
        """Additive Gaussian noise."""
        if torch.rand(1).item() >= prob:
            return image
        noise = torch.randn_like(image) * std
        return image + noise

    @staticmethod
    def _gaussian_blur(
        image: torch.Tensor, prob: float, sigma_range: list,
    ) -> torch.Tensor:
        """3D Gaussian blur via separable 1D convolutions."""
        if torch.rand(1).item() >= prob:
            return image

        sigma = torch.empty(1).uniform_(sigma_range[0], sigma_range[1]).item()
        kernel_size = int(2 * round(3 * sigma) + 1)
        if kernel_size < 3:
            kernel_size = 3

        # 1D Gaussian kernel
        x = torch.arange(kernel_size, dtype=image.dtype, device=image.device)
        x = x - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        B, C, D, H, W = image.shape
        pad = kernel_size // 2
        img = image.reshape(B * C, 1, D, H, W)

        # Separable: blur along D, H, W sequentially
        k_d = kernel_1d.reshape(1, 1, -1, 1, 1)
        k_h = kernel_1d.reshape(1, 1, 1, -1, 1)
        k_w = kernel_1d.reshape(1, 1, 1, 1, -1)

        img = F.pad(img, (0, 0, 0, 0, pad, pad), mode="replicate")
        img = F.conv3d(img, k_d)
        img = F.pad(img, (0, 0, pad, pad, 0, 0), mode="replicate")
        img = F.conv3d(img, k_h)
        img = F.pad(img, (pad, pad, 0, 0, 0, 0), mode="replicate")
        img = F.conv3d(img, k_w)

        return img.reshape(B, C, D, H, W)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _crop_or_pad(
    x: torch.Tensor,
    target_d: int,
    target_h: int,
    target_w: int,
    value: float = 0.0,
) -> torch.Tensor:
    """Center-crop or zero-pad a 5D tensor to (B, C, target_d, target_h, target_w)."""
    _, _, D, H, W = x.shape

    # Crop if larger
    if D > target_d:
        start = (D - target_d) // 2
        x = x[:, :, start:start + target_d, :, :]
    if H > target_h:
        start = (H - target_h) // 2
        x = x[:, :, :, start:start + target_h, :]
    if W > target_w:
        start = (W - target_w) // 2
        x = x[:, :, :, :, start:start + target_w]

    # Pad if smaller
    _, _, D, H, W = x.shape
    pad_d = max(target_d - D, 0)
    pad_h = max(target_h - H, 0)
    pad_w = max(target_w - W, 0)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        # F.pad order: (W_left, W_right, H_left, H_right, D_left, D_right)
        x = F.pad(x, (
            pad_w // 2, pad_w - pad_w // 2,
            pad_h // 2, pad_h - pad_h // 2,
            pad_d // 2, pad_d - pad_d // 2,
        ), value=value)

    return x
