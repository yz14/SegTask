"""UNet++ Nested Dense Decoder (3D).

Reference:
    Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image
    Segmentation", DLMIA 2018 / IEEE TMI 2020.

Motivation
----------
Classical UNet has a single decoder path that merges each encoder skip
exactly once. UNet++ replaces this with a *nested, densely connected*
grid of decoder nodes ``X[i,j]``:

* ``i`` is the depth index (0 = highest resolution, n-1 = bottleneck).
* ``j`` is the *column* index — ``j = 0`` is the encoder itself,
  ``j >= 1`` are the nested refinement columns.

At depth ``i`` the valid columns are ``j = 0, 1, ..., n-1-i``.
For a 5-level encoder this yields 15 nodes (5 + 4 + 3 + 2 + 1).

Node dependency rule (for ``j >= 1``):

    X[i, j]  =  Block( concat( X[i, 0], X[i, 1], ..., X[i, j-1],
                               Upsample( X[i+1, j-1] ) ) )

Every shorter pathway progressively re-injects encoder information into
the decoder, which empirically closes the semantic gap between encoder
and decoder features.

Interface compatibility
-----------------------
``UNet3D`` expects ``decoder.out_channels`` ordered low-res → high-res
and ``decoder(encoder_features)`` returning a same-ordered list of
feature maps. To keep existing seg_head / deep-supervision plumbing
unchanged, we expose the **diagonal** of the nested grid,
``X[i, n-1-i]`` — the last column available at each depth — as the
"decoder features". This gives exactly ``n-1`` outputs at the usual
multi-scale resolutions, and channel widths matching the classical
``Decoder`` (``encoder_channels[n-2-i]``).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

import torch.nn.functional as F

from .blocks import INTERP_SMOOTH, AttentionGate3D, Upsample


class UNetPPDecoder(nn.Module):
    """UNet++ nested dense decoder.

    Args:
        encoder_channels: channel widths of encoder levels, highest-res first.
        stage_builder:   callable ``(in_ch, out_ch) -> nn.Module`` used for
            each nested node's conv block. Same interface as the classical
            ``Decoder``'s ``stage_builder`` (usually a ResNet/ConvNeXt stage).
        upsample_mode:   passed to ``blocks.Upsample`` for the ``X[i+1,j-1]``
            → ``X[i,j]`` upsample branch.
        skip_attention:  if True, gate the upsampled branch by the same-depth
            encoder feature ``X[i, 0]`` via ``AttentionGate3D`` (Oktay 2018).

    Public attributes:
        ``out_channels``: ``[enc[n-2], enc[n-3], ..., enc[0]]`` — ordered
            low-res → high-res, matching the classical ``Decoder``.
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

        # Nested grid: nodes[i][j] for i in [0, n-2], j in [1, n-1-i].
        # Parameter containers are keyed by "i_j" strings so PyTorch's
        # ModuleDict can register them deterministically.
        self.upsamples = nn.ModuleDict()
        self.blocks = nn.ModuleDict()
        self.gates = nn.ModuleDict() if skip_attention else None

        for i in range(n - 1):
            for j in range(1, n - i):
                key = f"{i}_{j}"
                # Upsample from deeper column-(j-1) node.  When j == 1 the
                # source is the encoder feature X[i+1, 0] with channel
                # count encoder_channels[i+1]; for j > 1 it's the previously
                # computed X[i+1, j-1] which we keep at encoder_channels[i+1]
                # as well (symmetric widths).
                self.upsamples[key] = Upsample(
                    encoder_channels[i + 1],
                    encoder_channels[i],
                    mode=upsample_mode,
                    spatial_dims=spatial_dims,
                )
                # Fused input: j previous same-depth features (enc_ch[i] each)
                # + 1 upsampled feature (projected to enc_ch[i]) → (j+1) * enc_ch[i].
                fused_ch = (j + 1) * encoder_channels[i]
                self.blocks[key] = stage_builder(fused_ch, encoder_channels[i])

                if skip_attention:
                    # Gate the upsampled branch using X[i, 0] (raw encoder
                    # feature at this depth) as the gating signal.
                    self.gates[key] = AttentionGate3D(
                        x_ch=encoder_channels[i],
                        g_ch=encoder_channels[i],
                        spatial_dims=spatial_dims,
                    )

        # UNet3D-compatible decoder output channels (low-res → high-res).
        # We expose the diagonal X[i, n-1-i] for i in [0, n-2], which matches
        # the classical Decoder widths exactly.
        self.out_channels = [encoder_channels[n - 2 - k] for k in range(n - 1)]

    def forward(self, encoder_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Run the nested decoder and return the diagonal outputs.

        Args:
            encoder_features: ``[E_0, E_1, ..., E_{n-1}]`` highest-res first.

        Returns:
            ``[X[n-2, 1], X[n-3, 2], ..., X[0, n-1]]`` — ordered low-res
            to high-res (UNet3D convention).
        """
        n = self.n
        # x[i][j] = X[i, j] tensor. Only the columns we need survive.
        x: List[List[torch.Tensor]] = [[None] * (n - i) for i in range(n)]  # type: ignore
        for i in range(n):
            x[i][0] = encoder_features[i]

        # Compute columns left → right. Column j only depends on column j-1
        # at the same depth or one depth deeper.
        for j in range(1, n):
            for i in range(n - j):
                key = f"{i}_{j}"
                # Upsample deeper node X[i+1, j-1] and project channels.
                up = self.upsamples[key](x[i + 1][j - 1])
                # Resize if odd spatial dims caused a mismatch.
                if up.shape[2:] != x[i][0].shape[2:]:
                    up = F.interpolate(
                        up, size=x[i][0].shape[2:],
                        mode=INTERP_SMOOTH[self.spatial_dims],
                        align_corners=False)
                # Optional gating: drive by X[i, 0] (raw encoder feature).
                if self.skip_attention:
                    up = self.gates[key](up, x[i][0])

                # Concatenate all prior same-depth nodes + the upsampled branch.
                fused = torch.cat(x[i][:j] + [up], dim=1)
                x[i][j] = self.blocks[key](fused)

        # Diagonal: X[n-2, 1], X[n-3, 2], ..., X[0, n-1] — low-res to high-res.
        return [x[n - 2 - k][1 + k] for k in range(n - 1)]
