"""VGG-style encoder for UNet.

Each encoder level consists of N stacked Conv-Norm-Act blocks
(the classic VGG design), followed by downsampling.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from ..blocks import ConvNormAct, Downsample


class VGGBlock(nn.Module):
    """A single VGG-style block: N x (Conv3x3 + Norm + Act)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_convs: int = 2,
        spatial_dims: int = 2,
        norm_type: str = "instance",
        norm_groups: int = 8,
        activation: str = "leakyrelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        for i in range(num_convs):
            ch_in = in_channels if i == 0 else out_channels
            layers.append(
                ConvNormAct(
                    ch_in, out_channels,
                    kernel_size=3, padding=1,
                    spatial_dims=spatial_dims,
                    norm_type=norm_type,
                    norm_groups=norm_groups,
                    activation=activation,
                    dropout=dropout,
                )
            )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VGGEncoder(nn.Module):
    """VGG-style encoder with configurable depth and channels.

    Outputs a list of feature maps at each level (for skip connections)
    plus the final bottleneck feature map.
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

        # Build encoder levels
        self.levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i, (ch, n_blocks) in enumerate(zip(channels, blocks_per_level)):
            ch_in = in_channels if i == 0 else channels[i - 1]
            self.levels.append(
                VGGBlock(
                    ch_in, ch,
                    num_convs=n_blocks,
                    spatial_dims=spatial_dims,
                    norm_type=norm_type,
                    norm_groups=norm_groups,
                    activation=activation,
                    dropout=dropout,
                )
            )
            # Downsample between levels (not after the last one)
            if i < len(channels) - 1:
                self.downsamples.append(
                    Downsample(ch, ch, spatial_dims=spatial_dims,
                               norm_type=norm_type, norm_groups=norm_groups)
                )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returning feature maps at each level.

        Returns:
            List of tensors [level_0, level_1, ..., level_N-1]
            where level_0 is the highest resolution and level_N-1 is the bottleneck.
        """
        features = []
        for i, level in enumerate(self.levels):
            x = level(x)
            features.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        return features
