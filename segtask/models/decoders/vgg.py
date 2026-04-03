"""VGG-style decoder for UNet.

Mirrors the VGG encoder: each level has N Conv-Norm-Act blocks,
with upsampling + skip connection fusion between levels.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from ..blocks import ConvNormAct, Upsample, get_conv


class VGGDecoderBlock(nn.Module):
    """VGG decoder block: N x (Conv3x3 + Norm + Act)."""

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


class VGGDecoder(nn.Module):
    """VGG-style decoder with skip connections.

    At each level:
    1. Upsample the feature map from the previous (deeper) level
    2. Concatenate (or add) the skip connection from the encoder
    3. Apply VGG block (N conv layers)
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
            # Input from deeper level
            if i == 0:
                deeper_ch = encoder_channels[-1]  # bottleneck
            else:
                deeper_ch = decoder_channels[i - 1]

            # Skip connection from encoder
            # Encoder features are indexed: encoder_channels[-2], [-3], ...
            skip_ch = encoder_channels[-(i + 2)]

            self.upsamples.append(
                Upsample(deeper_ch, deeper_ch, spatial_dims=spatial_dims,
                         mode=upsample_mode)
            )

            if skip_mode == "cat":
                block_in = deeper_ch + skip_ch
            else:  # add
                block_in = deeper_ch

            self.blocks.append(
                VGGDecoderBlock(
                    block_in, decoder_channels[i],
                    num_convs=blocks_per_level[i],
                    spatial_dims=spatial_dims,
                    norm_type=norm_type,
                    norm_groups=norm_groups,
                    activation=activation,
                    dropout=dropout,
                )
            )

            # 1x1 conv for channel matching when using add mode
            if skip_mode == "add" and deeper_ch != skip_ch:
                Conv = get_conv(spatial_dims)
                self.add_module(
                    f"skip_proj_{i}",
                    Conv(skip_ch, deeper_ch, 1, bias=False),
                )

    def forward(
        self, encoder_features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Decode using encoder features (from high-res to low-res).

        Args:
            encoder_features: [level_0, level_1, ..., level_N-1]
                where level_N-1 is the bottleneck (lowest resolution).

        Returns:
            List of decoder feature maps [dec_0, dec_1, ..., dec_N-2]
            from low-res to high-res. dec_0 corresponds to the highest
            resolution decoder output.
        """
        x = encoder_features[-1]  # bottleneck
        decoder_outputs = []

        for i in range(self.num_levels):
            x = self.upsamples[i](x)
            skip = encoder_features[-(i + 2)]

            # Handle spatial size mismatch due to odd dimensions
            if x.shape[2:] != skip.shape[2:]:
                x = self._match_size(x, skip)

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

    @staticmethod
    def _match_size(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Pad or crop x to match target's spatial dimensions."""
        slices = [slice(None), slice(None)]  # B, C
        for d in range(2, x.ndim):
            if x.shape[d] > target.shape[d]:
                slices.append(slice(0, target.shape[d]))
            else:
                slices.append(slice(None))
        x = x[tuple(slices)]

        # Pad if needed
        pad = []
        for d in range(x.ndim - 1, 1, -1):
            diff = target.shape[d] - x.shape[d]
            pad.extend([0, diff])
        if any(p > 0 for p in pad):
            x = nn.functional.pad(x, pad)
        return x
