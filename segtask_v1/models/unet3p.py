"""UNet3+ full-scale skip decoder (Huang 2020).

Each decoder node D_i fuses n branches: encoders E_j (j<i pooled, j=i same, j=n-1 upsampled)
and already-computed deeper decoders D_j (i<j<n-1, upsampled). Branches concat then fuse to a
uniform fused_channels width across all depths.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import INTERP_SMOOTH, AttentionGate3D, ConvNormAct


class UNet3PDecoder(nn.Module):
    """UNet3+ full-scale skip decoder; out_channels low-res → high-res.

    Args:
        encoder_channels: encoder widths, highest-res first, bottleneck last.
        cat_channels: per-branch width after 3x3 conv (paper: 64).
        fused_channels: per-node output width; 0 → cat_channels * n (e.g. 320 at n=5).
        skip_attention: gate each branch by same-level E_i.
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

        # For each decoder depth i, build n branch convs (one per source j) + a fusion conv.
        self.branches = nn.ModuleList()
        self.fusions = nn.ModuleList()
        self.gates = nn.ModuleList() if skip_attention else None
        for i in range(n - 1):
            branch_convs = nn.ModuleList()
            branch_gates = nn.ModuleList() if skip_attention else None
            for j in range(n):
                if j <= i:
                    src_ch = encoder_channels[j]
                elif j < n - 1:
                    src_ch = self.fused_ch       # deeper decoder node
                else:
                    src_ch = encoder_channels[n - 1]  # bottleneck
                branch_convs.append(_cna(src_ch, cat_channels))
                if skip_attention:
                    branch_gates.append(
                        AttentionGate3D(
                            x_ch=src_ch, g_ch=encoder_channels[i],
                            spatial_dims=spatial_dims))
            self.branches.append(branch_convs)
            if skip_attention:
                self.gates.append(branch_gates)
            self.fusions.append(_cna(n * cat_channels, self.fused_ch))

        self.out_channels = [self.fused_ch] * (n - 1)

    def _resize_to(self, src: torch.Tensor, target_shape, mode: str) -> torch.Tensor:
        if src.shape[2:] == target_shape:
            return src
        if mode == "down":
            # adaptive pool handles non-2^k feature maps (odd patch sizes)
            if self.spatial_dims == 3:
                return F.adaptive_max_pool3d(src, target_shape)
            return F.adaptive_max_pool2d(src, target_shape)
        return F.interpolate(
            src, size=target_shape,
            mode=INTERP_SMOOTH[self.spatial_dims], align_corners=False)

    def forward(self, encoder_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Return decoder features [low_res, ..., high_res]."""
        n = self.n
        decoder_nodes: List[torch.Tensor] = [None] * (n - 1)  # type: ignore

        # iterate deepest → shallowest so deeper D_j is ready when needed
        for i in range(n - 2, -1, -1):
            tgt_shape = encoder_features[i].shape[2:]
            gate_signal = encoder_features[i]
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

        return list(reversed(decoder_nodes))
