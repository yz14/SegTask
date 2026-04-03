"""ResNet-style encoder for UNet.

Each encoder level consists of N residual blocks with skip connections.
Supports both BasicBlock (2 convs) and the option for pre-activation style.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from ..blocks import ConvNormAct, Downsample, get_conv, get_norm, get_activation


class ResidualBlock(nn.Module):
    """A single residual block: Conv-Norm-Act-Conv-Norm + skip."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dims: int = 2,
        norm_type: str = "instance",
        norm_groups: int = 8,
        activation: str = "leakyrelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        Conv = get_conv(spatial_dims)

        self.conv1 = Conv(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = get_norm(norm_type, out_channels, spatial_dims, norm_groups)
        self.act1 = get_activation(activation)

        self.conv2 = Conv(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = get_norm(norm_type, out_channels, spatial_dims, norm_groups)
        self.act2 = get_activation(activation)

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Skip connection with 1x1 conv if channel mismatch
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                Conv(in_channels, out_channels, 1, bias=False),
                get_norm(norm_type, out_channels, spatial_dims, norm_groups),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.drop(out)
        out = self.norm2(self.conv2(out))
        return self.act2(out + residual)


class ResNetEncoder(nn.Module):
    """ResNet-style encoder with configurable depth and channels.

    Each level has N residual blocks followed by downsampling.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: List[int] = None,
        blocks_per_level: List[int] = None,
        spatial_dims: int = 2,
        norm_type: str = "instance",
        norm_groups: int = 8,
        activation: str = "leakyrelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256, 512]
        if blocks_per_level is None:
            blocks_per_level = [2] * len(channels)

        assert len(channels) == len(blocks_per_level)

        self.num_levels = len(channels)
        self.out_channels = channels

        self.levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i, (ch, n_blocks) in enumerate(zip(channels, blocks_per_level)):
            ch_in = in_channels if i == 0 else channels[i - 1]

            blocks = []
            for j in range(n_blocks):
                block_in = ch_in if j == 0 else ch
                blocks.append(
                    ResidualBlock(
                        block_in, ch,
                        spatial_dims=spatial_dims,
                        norm_type=norm_type,
                        norm_groups=norm_groups,
                        activation=activation,
                        dropout=dropout,
                    )
                )
            self.levels.append(nn.Sequential(*blocks))

            if i < len(channels) - 1:
                self.downsamples.append(
                    Downsample(ch, ch, spatial_dims=spatial_dims,
                               norm_type=norm_type, norm_groups=norm_groups)
                )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for i, level in enumerate(self.levels):
            x = level(x)
            features.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        return features
