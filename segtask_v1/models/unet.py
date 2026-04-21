"""Generic 3D UNet architecture.

Structure:
  Input → Stem → [Encoder levels with Downsample] → Bottleneck
                      ↓ (skip connections)
  [Decoder levels with Upsample + skip fusion] → Segmentation Head

The encoder and decoder are symmetric in channel count.
Backbone blocks (ResNet / ConvNeXt) are injected via the factory.

Supports:
- Deep supervision (multi-scale outputs during training)
- Configurable skip connection mode (concatenate or add)
- Per-class independent sigmoid output
"""

from __future__ import annotations

from typing import Dict, List, Union

import torch
import torch.nn as nn

from .blocks import ConvNormAct, Downsample, Upsample, get_norm


class Encoder(nn.Module):
    """UNet encoder: stem + N stages with downsampling between them.

    Produces multi-scale features [level_0, level_1, ..., bottleneck].
    level_0 is at the highest resolution.
    """

    def __init__(
        self,
        in_channels: int,
        stage_channels: List[int],
        stage_builder,
        norm_type: str = "instance",
        norm_groups: int = 8,
        activation: str = "leakyrelu"):
        super().__init__()
        # Stem: project input to first channel count
        self.stem = ConvNormAct(
            in_channels, stage_channels[0],
            kernel_size=3, stride=1, padding=1,
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation)

        # Encoder stages and downsampling
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i, ch in enumerate(stage_channels):
            in_ch = stage_channels[i - 1] if i > 0 else stage_channels[0]
            self.stages.append(stage_builder(in_ch, ch))
            if i > 0:
                self.downsamples.append(
                    Downsample(stage_channels[i - 1], stage_channels[i - 1],
                               norm_type, norm_groups))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns features from each level: [level_0, level_1, ..., level_N]."""
        x = self.stem(x)
        features = []
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = self.downsamples[i - 1](x)
            x = stage(x)
            features.append(x)
        return features


class DecoderLevel(nn.Module):
    """Single decoder level: upsample + skip fusion + stage blocks."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        stage_builder,
        upsample_mode: str = "transpose",
        skip_mode: str = "cat"):
        super().__init__()
        self.skip_mode = skip_mode
        self.upsample  = Upsample(in_ch, out_ch, mode=upsample_mode)

        if skip_mode == "cat":
            fused_ch = out_ch + skip_ch
        else:  # add
            # Project skip to match out_ch if needed
            self.skip_proj = (
                nn.Conv3d(skip_ch, out_ch, 1, bias=False)
                if skip_ch != out_ch else nn.Identity())
            fused_ch = out_ch

        self.stage = stage_builder(fused_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # Handle size mismatch (can happen due to odd input sizes)
        if x.shape[2:] != skip.shape[2:]:
            x = _match_size(x, skip.shape[2:])

        if self.skip_mode == "cat":
            x = torch.cat([x, skip], dim=1)
        else:
            x = x + self.skip_proj(skip)

        return self.stage(x)


class Decoder(nn.Module):
    """UNet decoder: N levels of upsample + skip fusion + blocks.

    Takes encoder features [level_0, ..., bottleneck] and produces
    decoder features [dec_low_res, ..., dec_high_res].
    """

    def __init__(
        self,
        encoder_channels: List[int],
        stage_builder,
        upsample_mode: str = "transpose",
        skip_mode: str = "cat"):
        super().__init__()
        self.levels = nn.ModuleList()
        n = len(encoder_channels)

        # Decoder levels: from deepest to shallowest
        # Level i connects encoder[n-2-i] (skip) with previous decoder output
        for i in range(n - 1):
            in_ch = encoder_channels[n - 1 - i]  # from deeper level
            skip_ch = encoder_channels[n - 2 - i]  # skip connection
            out_ch = encoder_channels[n - 2 - i]   # symmetric output

            self.levels.append(
                DecoderLevel(in_ch, skip_ch, out_ch, stage_builder,
                             upsample_mode, skip_mode))

        # Output channels at each decoder level (low-res → high-res)
        self.out_channels = [encoder_channels[n - 2 - i] for i in range(n - 1)]

    def forward(self, encoder_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Decode features.

        Args:
            encoder_features: [level_0, level_1, ..., bottleneck]

        Returns:
            Decoder features [dec_low_res, ..., dec_high_res]
        """
        x = encoder_features[-1]  # bottleneck
        outputs = []
        for i, level in enumerate(self.levels):
            skip_idx = len(encoder_features) - 2 - i
            x = level(x, encoder_features[skip_idx])
            outputs.append(x)
        return outputs


class SegmentationHead(nn.Module):
    """1x1x1 convolution to produce per-class logits."""

    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet3D(nn.Module):
    """Generic 3D UNet with pluggable encoder/decoder stages.

    Args:
        encoder: Encoder module.
        decoder: Decoder module.
        num_fg_classes: Number of foreground classes (output channels).
        deep_supervision: Output at multiple decoder levels during training.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        num_fg_classes: int,
        deep_supervision: bool = False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_fg_classes   = num_fg_classes
        self.deep_supervision = deep_supervision

        # Main segmentation head (highest resolution decoder output)
        self.seg_head = SegmentationHead(decoder.out_channels[-1], num_fg_classes)

        # Deep supervision heads (lower-resolution outputs).
        # decoder.out_channels is [low, ..., high]; we want DS outputs ordered
        # from 2nd-highest to lowest resolution so that forward() can return
        # [main_out, 2nd_high, 3rd_high, ..., lowest] — matching the
        # DeepSupervisionLoss convention weights[0]=highest-res.
        if deep_supervision:
            self.ds_heads = nn.ModuleList()
            # Reverse [low..2nd-high] → [2nd-high..low]
            for ch in reversed(decoder.out_channels[:-1]):
                self.ds_heads.append(SegmentationHead(ch, num_fg_classes))

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward pass.

        Args:
            x: (B, 1, D, H, W) input.

        Returns:
            If deep_supervision=False or eval: (B, num_fg, D, H, W) logits.
            If deep_supervision=True and training: list of multi-scale logits
                ordered [main_out (highest-res), 2nd_high, 3rd_high, ..., lowest].
        """
        enc_features = self.encoder(x)
        dec_features = self.decoder(enc_features)

        main_out = self.seg_head(dec_features[-1])

        if not self.deep_supervision or not self.training:
            return main_out

        # dec_features = [low, ..., high]; dec_features[-1] is already used
        # as main_out. DS heads must consume features in decreasing resolution:
        # dec_features[-2] (2nd-highest), dec_features[-3], ..., dec_features[0] (lowest).
        outputs = [main_out]
        for i, head in enumerate(self.ds_heads):
            outputs.append(head(dec_features[-2 - i]))
        return outputs

    def param_count(self) -> Dict[str, int]:
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        head = sum(p.numel() for p in self.seg_head.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {"encoder": enc, "decoder": dec, "seg_head": head, "total": total}


def _match_size(x: torch.Tensor, target_size) -> torch.Tensor:
    """Pad or crop x to match target spatial size."""
    import torch.nn.functional as F
    return F.interpolate(x, size=target_size, mode="trilinear", align_corners=False)
