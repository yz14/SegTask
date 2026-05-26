"""UNet++ nested dense decoder (Zhou 2018/2020).

Nodes X[i,j]: i=depth, j=column. j=0 is encoder; j>=1 fuses all prior same-depth
nodes + upsampled X[i+1,j-1]. Exposes diagonal X[i,n-1-i] as decoder outputs
(low-res → high-res) to stay UNet3D-compatible.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

import torch.nn.functional as F

from .blocks import INTERP_SMOOTH, AttentionGate3D, Upsample


class UNetPPDecoder(nn.Module):
    """UNet++ nested decoder. Exposes diagonal X[i,n-1-i] as low-res→high-res outputs.

    Args:
        encoder_channels: encoder widths, highest-res first.
        stage_builder: (in_ch, out_ch) -> Module for each node's conv block.
        upsample_mode: e.g. 'transpose' (learned) | 'trilinear' (parameter-free).
        skip_attention: if True, gate upsampled branch by X[i,0] (Oktay 2018).
    """

    def __init__(
        self,
        encoder_channels: List[int],
        stage_builder,
        upsample_mode: str = "transpose",
        skip_attention: bool = False,
        spatial_dims: int = 3,
    ):
        super().__init__()
        n = len(encoder_channels)
        if n < 2:
            raise ValueError("UNetPPDecoder requires at least 2 encoder levels")
        self.n = n
        self.skip_attention = skip_attention
        self.spatial_dims = spatial_dims

        # Nested grid keyed by 'i_j' so ModuleDict registers deterministically.
        self.upsamples = nn.ModuleDict()
        self.blocks = nn.ModuleDict()
        self.gates = nn.ModuleDict() if skip_attention else None

        for i in range(n - 1):
            for j in range(1, n - i):
                key = f"{i}_{j}"
                # Upsample X[i+1, j-1] (channels enc[i+1]) → enc[i]
                self.upsamples[key] = Upsample(
                    encoder_channels[i + 1],
                    encoder_channels[i],
                    mode=upsample_mode,
                    spatial_dims=spatial_dims,
                )
                # Fused: j same-depth nodes + 1 upsampled = (j+1) * enc[i]
                fused_ch = (j + 1) * encoder_channels[i]
                self.blocks[key] = stage_builder(fused_ch, encoder_channels[i])

                if skip_attention:
                    self.gates[key] = AttentionGate3D(
                        x_ch=encoder_channels[i],
                        g_ch=encoder_channels[i],
                        spatial_dims=spatial_dims,
                    )

        # Diagonal X[i, n-1-i] widths (low-res → high-res), matches classical Decoder
        self.out_channels = [encoder_channels[n - 2 - k] for k in range(n - 1)]

    def forward(self, encoder_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """encoder_features high-res first; returns diagonal low-res → high-res."""
        n = self.n
        x: List[List[torch.Tensor]] = [[None] * (n - i) for i in range(n)]  # type: ignore
        for i in range(n):
            x[i][0] = encoder_features[i]

        # Fill columns left→right; column j depends only on column j-1
        for j in range(1, n):
            for i in range(n - j):
                key = f"{i}_{j}"
                up = self.upsamples[key](x[i + 1][j - 1])
                if up.shape[2:] != x[i][0].shape[2:]:
                    up = F.interpolate(
                        up, size=x[i][0].shape[2:],
                        mode=INTERP_SMOOTH[self.spatial_dims],
                        align_corners=False)
                if self.skip_attention:
                    up = self.gates[key](up, x[i][0])
                fused = torch.cat(x[i][:j] + [up], dim=1)
                x[i][j] = self.blocks[key](fused)

        return [x[n - 2 - k][1 + k] for k in range(n - 1)]
