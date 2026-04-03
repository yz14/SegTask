"""Common building blocks shared across encoder/decoder implementations.

Provides:
- Normalization layer factory
- Activation layer factory
- Convolution helpers (2D/3D unified)
- Downsampling / Upsampling modules
"""

from __future__ import annotations

from typing import Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Layer factories
# ---------------------------------------------------------------------------
def get_conv(spatial_dims: int) -> Type[nn.Module]:
    """Return Conv2d or Conv3d based on spatial dimensionality."""
    return nn.Conv2d if spatial_dims == 2 else nn.Conv3d


def get_conv_transpose(spatial_dims: int) -> Type[nn.Module]:
    return nn.ConvTranspose2d if spatial_dims == 2 else nn.ConvTranspose3d


def get_norm(
    norm_type: str,
    num_channels: int,
    spatial_dims: int,
    num_groups: int = 8,
) -> nn.Module:
    """Create a normalization layer."""
    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels) if spatial_dims == 2 else nn.BatchNorm3d(num_channels)
    elif norm_type == "instance":
        if spatial_dims == 2:
            return nn.InstanceNorm2d(num_channels, affine=True)
        else:
            return nn.InstanceNorm3d(num_channels, affine=True)
    elif norm_type == "group":
        # Ensure num_groups divides num_channels
        while num_channels % num_groups != 0 and num_groups > 1:
            num_groups //= 2
        return nn.GroupNorm(num_groups, num_channels)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


def get_activation(name: str) -> nn.Module:
    """Create an activation layer."""
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "leakyrelu":
        return nn.LeakyReLU(0.01, inplace=True)
    elif name == "gelu":
        return nn.GELU()
    elif name == "swish":
        return nn.SiLU(inplace=True)
    else:
        raise ValueError(f"Unknown activation: {name}")


def get_pool(spatial_dims: int, kernel_size: int = 2) -> nn.Module:
    """Create a max pooling layer."""
    if spatial_dims == 2:
        return nn.MaxPool2d(kernel_size)
    else:
        return nn.MaxPool3d(kernel_size)


def get_adaptive_pool(spatial_dims: int, output_size: int = 1) -> nn.Module:
    if spatial_dims == 2:
        return nn.AdaptiveAvgPool2d(output_size)
    else:
        return nn.AdaptiveAvgPool3d(output_size)


# ---------------------------------------------------------------------------
# Convolution block: Conv + Norm + Activation
# ---------------------------------------------------------------------------
class ConvNormAct(nn.Module):
    """Single convolution + normalization + activation block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        spatial_dims: int = 2,
        norm_type: str = "instance",
        norm_groups: int = 8,
        activation: str = "leakyrelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        Conv = get_conv(spatial_dims)
        self.conv = Conv(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = get_norm(norm_type, out_channels, spatial_dims, norm_groups)
        self.act = get_activation(activation)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.conv(x))))


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------
class Downsample(nn.Module):
    """Downsampling via strided convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dims: int = 2,
        norm_type: str = "instance",
        norm_groups: int = 8,
    ):
        super().__init__()
        Conv = get_conv(spatial_dims)
        self.conv = Conv(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        self.norm = get_norm(norm_type, out_channels, spatial_dims, norm_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


# ---------------------------------------------------------------------------
# Upsampling
# ---------------------------------------------------------------------------
class Upsample(nn.Module):
    """Upsampling via transposed convolution or interpolation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dims: int = 2,
        mode: str = "transpose",
    ):
        super().__init__()
        self.mode = mode
        self.spatial_dims = spatial_dims

        if mode == "transpose":
            ConvT = get_conv_transpose(spatial_dims)
            self.up = ConvT(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            Conv = get_conv(spatial_dims)
            self.up = Conv(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "transpose":
            return self.up(x)
        else:
            interp_mode = "bilinear" if self.spatial_dims == 2 else "trilinear"
            x = F.interpolate(x, scale_factor=2, mode=interp_mode, align_corners=False)
            return self.up(x)
