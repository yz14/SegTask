"""ResNet-style decoder for UNet.

Each decoder level consists of N residual blocks with skip connections,
mirroring the ResNet encoder design.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from ..blocks import Upsample, get_conv, get_norm, get_activation
from ..encoders.resnet import ResidualBlock


class ResNetDecoder(nn.Module):
    """ResNet-style decoder with skip connections.

    At each level:
    1. Upsample from deeper level
    2. Fuse skip connection (cat or add)
    3. Apply N residual blocks
    """

    def __init__(
        self,
        encoder_channels: List[int] = None,
        decoder_channels: List[int] = None,
        blocks_per_level: List[int] = None,
        spatial_dims: int = 2,
        norm_type: str = "instance",
        norm_groups: int = 8,
        activation: str = "leakyrelu",
        dropout: float = 0.0,
        upsample_mode: str = "transpose",
        skip_mode: str = "cat",
    ):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [32, 64, 128, 256, 512]
        if decoder_channels is None:
            decoder_channels = list(reversed(encoder_channels[:-1]))
        if blocks_per_level is None:
            blocks_per_level = [2] * len(decoder_channels)

        self.num_levels = len(decoder_channels)
        self.skip_mode = skip_mode
        self.out_channels = decoder_channels

        self.upsamples = nn.ModuleList()
        self.blocks = nn.ModuleList()

        for i in range(self.num_levels):
            deeper_ch = encoder_channels[-1] if i == 0 else decoder_channels[i - 1]
            skip_ch = encoder_channels[-(i + 2)]

            self.upsamples.append(
                Upsample(deeper_ch, deeper_ch, spatial_dims=spatial_dims,
                         mode=upsample_mode)
            )

            if skip_mode == "cat":
                block_in = deeper_ch + skip_ch
            else:
                block_in = deeper_ch

            blocks = []
            for j in range(blocks_per_level[i]):
                b_in = block_in if j == 0 else decoder_channels[i]
                blocks.append(
                    ResidualBlock(
                        b_in, decoder_channels[i],
                        spatial_dims=spatial_dims,
                        norm_type=norm_type,
                        norm_groups=norm_groups,
                        activation=activation,
                        dropout=dropout,
                    )
                )
            self.blocks.append(nn.Sequential(*blocks))

            if skip_mode == "add" and deeper_ch != skip_ch:
                Conv = get_conv(spatial_dims)
                self.add_module(
                    f"skip_proj_{i}",
                    Conv(skip_ch, deeper_ch, 1, bias=False),
                )

    def forward(
        self, encoder_features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        x = encoder_features[-1]
        decoder_outputs = []

        for i in range(self.num_levels):
            x = self.upsamples[i](x)
            skip = encoder_features[-(i + 2)]

            if x.shape[2:] != skip.shape[2:]:
                x = _match_size(x, skip)

            if self.skip_mode == "cat":
                x = torch.cat([x, skip], dim=1)
            else:
                proj = getattr(self, f"skip_proj_{i}", None)
                if proj is not None:
                    skip = proj(skip)
                x = x + skip

            x = self.blocks[i](x)
            decoder_outputs.append(x)

        return decoder_outputs


def _match_size(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pad or crop x to match target's spatial dimensions."""
    slices = [slice(None), slice(None)]
    for d in range(2, x.ndim):
        if x.shape[d] > target.shape[d]:
            slices.append(slice(0, target.shape[d]))
        else:
            slices.append(slice(None))
    x = x[tuple(slices)]

    pad = []
    for d in range(x.ndim - 1, 1, -1):
        diff = target.shape[d] - x.shape[d]
        pad.extend([0, diff])
    if any(p > 0 for p in pad):
        x = nn.functional.pad(x, pad)
    return x
