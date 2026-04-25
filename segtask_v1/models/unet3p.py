"""UNet3+ Full-Scale Skip Decoder (3D).

Reference:
    Huang et al., "UNet 3+: A Full-Scale Connected UNet for Medical Image
    Segmentation", ICASSP 2020.

Motivation
----------
Classical UNet's decoder at level ``i`` only sees the encoder skip at the
same level ``i`` plus the upsampled feature from level ``i+1``. UNet3+
fuses **all** encoder and already-computed decoder features into every
decoder node, capturing both low-level fine details and high-level
semantics at every spatial scale.

Decoder node D_i aggregates ``n`` branches:
  * j < i: encoder E_j, max-pooled down to D_i's resolution.
  * j == i: encoder E_i, same resolution (just a 3×3×3 refinement conv).
  * i < j < n-1: decoder D_j (already computed, deeper node),
    trilinearly upsampled to D_i's resolution.
  * j == n-1: encoder E_{n-1} (bottleneck), trilinearly upsampled.

Each branch is passed through a ``cat_channels``-width Conv-Norm-Act,
the ``n`` branches are concatenated (→ ``n × cat_channels`` channels),
and a final Conv-Norm-Act fuses them back to ``fused_channels`` (default
``n × cat_channels``).  The fused channel count is uniform across all
decoder nodes so deep-supervision heads share a consistent input width.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import INTERP_SMOOTH, AttentionGate3D, ConvNormAct


class UNet3PDecoder(nn.Module):
    """Full-scale skip decoder.

    Args:
        encoder_channels: channel widths of the encoder, highest-res first
            and bottleneck last. Length = number of encoder levels (n).
        cat_channels:    per-branch channel count after the 3×3×3 conv.
            Paper default = 64.
        fused_channels:  output width of every decoder node. If 0, defaults
            to ``cat_channels * n`` (= 320 for n=5 in the paper).
        skip_attention:  if True, each encoder/decoder source branch is
            re-weighted by an ``AttentionGate3D`` driven by the same-level
            encoder feature E_i before the branch conv.

    Public attributes (UNet3D-compatible):
        ``out_channels``: list of output widths, one per decoder node,
            ordered ``[low_res, ..., high_res]`` (matching ``Decoder``).
    """

    def __init__(
        self,
        encoder_channels: List[int],
        cat_channels: int = 64,
        fused_channels: int = 0,
        norm_type: str = "instance",
        norm_groups: int = 8,
        activation: str = "leakyrelu",
        skip_attention: bool = False,
        spatial_dims: int = 3,
    ):
        super().__init__()
        n = len(encoder_channels)
        if n < 2:
            raise ValueError("UNet3PDecoder requires at least 2 encoder levels")
        self.n = n
        self.cat = cat_channels
        self.fused_ch = fused_channels if fused_channels > 0 else cat_channels * n
        self.skip_attention = skip_attention
        self.spatial_dims = spatial_dims

        def _cna(in_ch: int, out_ch: int) -> ConvNormAct:
            return ConvNormAct(
                in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                norm_type=norm_type, norm_groups=norm_groups,
                activation=activation, spatial_dims=spatial_dims)

        # For each decoder depth i ∈ [0, n-2] (0 = highest-res decoder), build
        # a list of n branch convs (one per source j ∈ [0, n-1]).
        self.branches = nn.ModuleList()
        self.fusions = nn.ModuleList()
        # Optional attention gates on each branch, driven by E_i.
        self.gates = nn.ModuleList() if skip_attention else None
        for i in range(n - 1):
            branch_convs = nn.ModuleList()
            branch_gates = nn.ModuleList() if skip_attention else None
            for j in range(n):
                if j <= i:
                    src_ch = encoder_channels[j]
                elif j < n - 1:
                    # Deeper decoder node D_j's width.
                    src_ch = self.fused_ch
                else:
                    # Bottleneck encoder feature.
                    src_ch = encoder_channels[n - 1]
                branch_convs.append(_cna(src_ch, cat_channels))
                if skip_attention:
                    branch_gates.append(
                        AttentionGate3D(
                            x_ch=src_ch, g_ch=encoder_channels[i],
                            spatial_dims=spatial_dims))
            self.branches.append(branch_convs)
            if skip_attention:
                self.gates.append(branch_gates)
            # Fusion conv — unifies the n concatenated branches.
            self.fusions.append(_cna(n * cat_channels, self.fused_ch))

        # UNet3D expects ``out_channels`` ordered low-res → high-res.
        self.out_channels = [self.fused_ch] * (n - 1)

    def _resize_to(self, src: torch.Tensor, target_shape, mode: str) -> torch.Tensor:
        if src.shape[2:] == target_shape:
            return src
        if mode == "down":
            # Adaptive max pool handles arbitrary ratios (robust to non-2^k
            # feature maps caused by odd patch sizes).
            if self.spatial_dims == 3:
                return F.adaptive_max_pool3d(src, target_shape)
            return F.adaptive_max_pool2d(src, target_shape)
        return F.interpolate(
            src, size=target_shape,
            mode=INTERP_SMOOTH[self.spatial_dims], align_corners=False)

    def forward(self, encoder_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Compute decoder features in ``[low_res, ..., high_res]`` order."""
        n = self.n
        # decoder_nodes[i] = D_i (highest-res decoder first). Not yet computed.
        decoder_nodes: List[torch.Tensor] = [None] * (n - 1)  # type: ignore

        # Iterate from deepest decoder (i = n-2) towards the highest-res (i = 0)
        # so that ``decoder_nodes[j]`` for j > i is already available.
        for i in range(n - 2, -1, -1):
            tgt_shape = encoder_features[i].shape[2:]
            gate_signal = encoder_features[i]  # drives optional attention.
            branches = []
            for j in range(n):
                if j < i:
                    src = self._resize_to(encoder_features[j], tgt_shape, "down")
                elif j == i:
                    src = encoder_features[i]
                elif j < n - 1:
                    src = self._resize_to(decoder_nodes[j], tgt_shape, "up")
                else:
                    src = self._resize_to(encoder_features[n - 1], tgt_shape, "up")

                if self.skip_attention:
                    src = self.gates[i][j](src, gate_signal)
                branches.append(self.branches[i][j](src))

            fused = torch.cat(branches, dim=1)
            decoder_nodes[i] = self.fusions[i](fused)

        # Standard UNet Decoder returns [low, ..., high]; UNet3D consumes
        # ``dec_features[-1]`` as the highest-resolution output.
        return list(reversed(decoder_nodes))
