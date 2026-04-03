"""Generic UNet architecture.

Combines any encoder + any decoder + segmentation head.
Supports deep supervision (multi-scale output) for training.

The architecture:
    Input → Encoder → [skip_0, skip_1, ..., bottleneck]
                         ↓
    Decoder (with skip connections) → [dec_0, dec_1, ..., dec_N]
                         ↓
    Segmentation Head → per-class output maps
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from .blocks import get_conv, get_norm, get_activation


class SegmentationHead(nn.Module):
    """Final 1x1 convolution to produce per-class logits."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        spatial_dims: int = 2,
    ):
        super().__init__()
        Conv = get_conv(spatial_dims)
        self.conv = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """Generic UNet with pluggable encoder and decoder.

    Features:
    - Any encoder/decoder combination (VGG, ResNet, ViT)
    - Deep supervision: output at multiple decoder levels
    - Per-class single-channel output with independent loss computation

    Args:
        encoder: Encoder module producing multi-scale features.
        decoder: Decoder module consuming encoder features.
        num_classes: Number of output classes (including background).
        spatial_dims: 2 or 3.
        deep_supervision: If True, output at multiple decoder levels.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        num_classes: int,
        spatial_dims: int = 2,
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        # Main segmentation head (highest resolution decoder output)
        decoder_out_ch = decoder.out_channels[-1]  # last = highest resolution
        self.seg_head = SegmentationHead(decoder_out_ch, num_classes, spatial_dims)

        # Deep supervision heads (for intermediate decoder levels)
        if deep_supervision:
            self.ds_heads = nn.ModuleList()
            for ch in decoder.out_channels[:-1]:
                self.ds_heads.append(
                    SegmentationHead(ch, num_classes, spatial_dims)
                )

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W) or (B, C, D, H, W).

        Returns:
            If deep_supervision=False:
                Single output tensor (B, num_classes, H, W) or (B, num_classes, D, H, W).
            If deep_supervision=True:
                List of tensors [main_output, ds_output_1, ds_output_2, ...]
                where ds outputs are at progressively lower resolutions.
        """
        # Encode
        encoder_features = self.encoder(x)

        # Decode
        decoder_features = self.decoder(encoder_features)

        # Main output (highest resolution)
        main_out = self.seg_head(decoder_features[-1])

        if not self.deep_supervision or not self.training:
            return main_out

        # Deep supervision outputs
        outputs = [main_out]
        for i, ds_head in enumerate(self.ds_heads):
            ds_out = ds_head(decoder_features[i])
            outputs.append(ds_out)

        return outputs

    def get_param_count(self) -> Dict[str, int]:
        """Get parameter counts by component."""
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        dec_params = sum(p.numel() for p in self.decoder.parameters())
        head_params = sum(p.numel() for p in self.seg_head.parameters())
        ds_params = 0
        if self.deep_supervision and hasattr(self, "ds_heads"):
            ds_params = sum(p.numel() for p in self.ds_heads.parameters())
        total = enc_params + dec_params + head_params + ds_params
        return {
            "encoder": enc_params,
            "decoder": dec_params,
            "seg_head": head_params,
            "deep_supervision_heads": ds_params,
            "total": total,
        }
