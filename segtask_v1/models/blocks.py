"""Common building blocks shared across encoder/decoder implementations.

Provides:
- Layer factories (conv, norm, activation)
- ConvNormAct: single conv + norm + activation
- SqueezeExcite3D: channel attention (SE block)
- Downsample: strided convolution
- Upsample: transposed convolution or trilinear interpolation
"""

from __future__ import annotations

from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Layer factories
# ---------------------------------------------------------------------------
def get_conv3d() -> Type[nn.Module]:
    return nn.Conv3d


def get_norm(
    norm_type: str,
    num_channels: int,
    num_groups: int = 8,
) -> nn.Module:
    """Create a 3D normalization layer."""
    if norm_type == "batch":
        return nn.BatchNorm3d(num_channels)
    elif norm_type == "instance":
        return nn.InstanceNorm3d(num_channels, affine=True)
    elif norm_type == "group":
        while num_channels % num_groups != 0 and num_groups > 1:
            num_groups //= 2
        return nn.GroupNorm(num_groups, num_channels)
    else:
        raise ValueError(f"Unknown norm: {norm_type}")


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


# ---------------------------------------------------------------------------
# Conv + Norm + Activation
# ---------------------------------------------------------------------------
class ConvNormAct(nn.Module):
    """3D convolution + normalization + activation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm_type: str = "instance",
        norm_groups: int = 8,
        activation: str = "leakyrelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.norm = get_norm(norm_type, out_ch, norm_groups)
        self.act = get_activation(activation)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.conv(x))))


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation (channel attention)
# ---------------------------------------------------------------------------
class SqueezeExcite3D(nn.Module):
    """3D Squeeze-and-Excitation block (Hu et al., 2018).

    Global average pool → FC reduce → ReLU → FC expand → Sigmoid → scale.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x * scale


# ---------------------------------------------------------------------------
# Downsampling (strided conv)
# ---------------------------------------------------------------------------
class Downsample(nn.Module):
    """Downsample by factor 2 via strided 3D convolution."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        norm_type: str = "instance",
        norm_groups: int = 8,
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.norm = get_norm(norm_type, out_ch, norm_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


# ---------------------------------------------------------------------------
# Upsampling
# ---------------------------------------------------------------------------
class Upsample(nn.Module):
    """Upsample by factor 2 via transposed conv or trilinear interpolation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        mode: str = "transpose",
    ):
        super().__init__()
        self.mode = mode
        if mode == "transpose":
            self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        else:
            self.up = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "transpose":
            return self.up(x)
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        return self.up(x)
