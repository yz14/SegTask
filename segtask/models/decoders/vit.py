"""Vision Transformer decoder for UNet.

Each decoder level uses transformer blocks for feature refinement,
with upsampling and skip connection fusion between levels.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from ..blocks import Upsample, get_conv, get_norm, get_activation
from ..encoders.vit import TransformerBlock


class ViTDecoder(nn.Module):
    """ViT-style decoder with skip connections.

    At each level:
    1. Upsample from deeper level
    2. Fuse skip connection (cat or add)
    3. Project channels with 1x1 conv
    4. Apply N transformer blocks
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
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.1,
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

        total_blocks = sum(blocks_per_level)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        self.upsamples = nn.ModuleList()
        self.proj_layers = nn.ModuleList()
        self.blocks = nn.ModuleList()

        Conv = get_conv(spatial_dims)
        block_idx = 0

        for i in range(self.num_levels):
            deeper_ch = encoder_channels[-1] if i == 0 else decoder_channels[i - 1]
            skip_ch = encoder_channels[-(i + 2)]

            self.upsamples.append(
                Upsample(deeper_ch, deeper_ch, spatial_dims=spatial_dims,
                         mode=upsample_mode)
            )

            if skip_mode == "cat":
                fused_ch = deeper_ch + skip_ch
            else:
                fused_ch = deeper_ch

            # 1x1 projection to decoder channel dim
            self.proj_layers.append(nn.Sequential(
                Conv(fused_ch, decoder_channels[i], 1, bias=False),
                get_norm(norm_type, decoder_channels[i], spatial_dims, norm_groups),
                get_activation(activation),
            ))

            n_heads = min(num_heads, decoder_channels[i])
            while decoder_channels[i] % n_heads != 0 and n_heads > 1:
                n_heads -= 1

            level_blocks = []
            for j in range(blocks_per_level[i]):
                level_blocks.append(
                    TransformerBlock(
                        dim=decoder_channels[i],
                        num_heads=n_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        dropout=dropout,
                        drop_path=dpr[block_idx] if block_idx < len(dpr) else 0.0,
                        spatial_dims=spatial_dims,
                        norm_type=norm_type,
                        norm_groups=norm_groups,
                    )
                )
                block_idx += 1
            self.blocks.append(nn.Sequential(*level_blocks))

            if skip_mode == "add" and deeper_ch != skip_ch:
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

            x = self.proj_layers[i](x)
            x = self.blocks[i](x)
            decoder_outputs.append(x)

        return decoder_outputs


def _match_size(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
