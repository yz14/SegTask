"""Stem / patch-embed builders for 3D UNet.

Provides the initial feature-extraction layer applied to raw input volumes.
The stem determines the resolution at which the encoder operates:

- "conv3"  : classical 3×3×3 stride-1 conv (default; preserves resolution).
- "conv7"  : ConvNeXt-style large-kernel 7×7×7 stride-1 conv (larger RF).
- "dual"   : nnU-Net-style two stacked 3×3×3 stride-1 convs.
- "patch2" : 2×2×2 stride-2 patch embedding (halves resolution).
- "patch4" : 4×4×4 stride-4 patch embedding (Swin / ConvNeXt standard).

Patch-embed stems (`patchN`) reduce spatial resolution by N, meaning the
encoder produces features starting at (input / N).  The UNet wrapper
(see `UNet3D`) restores the original resolution with a final learned
upsample applied only to the main segmentation output.
"""

from __future__ import annotations

from typing import Tuple

import torch.nn as nn

from .blocks import ConvNormAct, get_activation, get_norm


STEM_MODES = ("conv3", "conv7", "dual", "patch2", "patch4")


class DualConvStem(nn.Module):
    """Two stacked 3×3×3 conv-norm-act blocks (nnU-Net stem)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        norm_type: str = "instance",
        norm_groups: int = 8,
        activation: str = "leakyrelu",
    ):
        super().__init__()
        self.block1 = ConvNormAct(
            in_ch, out_ch, kernel_size=3, stride=1, padding=1,
            norm_type=norm_type, norm_groups=norm_groups, activation=activation)
        self.block2 = ConvNormAct(
            out_ch, out_ch, kernel_size=3, stride=1, padding=1,
            norm_type=norm_type, norm_groups=norm_groups, activation=activation)

    def forward(self, x):
        return self.block2(self.block1(x))


class PatchEmbedStem(nn.Module):
    """Patch-embedding stem: stride-N conv + norm + activation.

    Resolution is reduced by a factor of ``patch_size`` along every spatial
    axis.  Inspired by Swin Transformer and ConvNeXt patchify stems.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        patch_size: int,
        norm_type: str = "instance",
        norm_groups: int = 8,
        activation: str = "gelu",
    ):
        super().__init__()
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}")
        self.patch_size = patch_size
        self.conv = nn.Conv3d(in_ch, out_ch,
                              kernel_size=patch_size,
                              stride=patch_size,
                              bias=False)
        self.norm = get_norm(norm_type, out_ch, norm_groups)
        self.act = get_activation(activation)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


def build_stem(
    mode: str,
    in_ch: int,
    out_ch: int,
    norm_type: str = "instance",
    norm_groups: int = 8,
    activation: str = "leakyrelu",
) -> Tuple[nn.Module, int]:
    """Construct a stem module.

    Returns:
        (stem_module, stem_stride): ``stem_stride`` is the spatial
        downsampling factor introduced by the stem (1 for stride-1 stems,
        2 or 4 for patch-embed stems).  Callers use this to decide whether
        a matching final-upsample is required downstream.
    """
    if mode not in STEM_MODES:
        raise ValueError(f"Unknown stem mode: {mode!r}. Valid: {STEM_MODES}")

    if mode == "conv3":
        stem = ConvNormAct(
            in_ch, out_ch, kernel_size=3, stride=1, padding=1,
            norm_type=norm_type, norm_groups=norm_groups, activation=activation)
        return stem, 1

    if mode == "conv7":
        stem = ConvNormAct(
            in_ch, out_ch, kernel_size=7, stride=1, padding=3,
            norm_type=norm_type, norm_groups=norm_groups, activation=activation)
        return stem, 1

    if mode == "dual":
        stem = DualConvStem(
            in_ch, out_ch,
            norm_type=norm_type, norm_groups=norm_groups, activation=activation)
        return stem, 1

    # patch-embed variants
    patch_size = 2 if mode == "patch2" else 4
    # Patch-embed stems typically use GELU; expose activation for flexibility.
    stem = PatchEmbedStem(
        in_ch, out_ch, patch_size=patch_size,
        norm_type=norm_type, norm_groups=norm_groups,
        activation="gelu" if activation == "leakyrelu" else activation)
    return stem, patch_size
