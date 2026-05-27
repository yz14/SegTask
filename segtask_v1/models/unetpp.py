"""UNet++ 嵌套稠密 decoder (Zhou 2018/2020)。节点 X[i,j]：i 深度、j 列；j=0 为 encoder，j>=1 融合同深度前列与上采样的 X[i+1,j-1]。对角线 X[i,n-1-i] 作为输出（low-res→high-res）。"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

import torch.nn.functional as F

from .blocks import INTERP_SMOOTH, AttentionGate3D, Upsample


class UNetPPDecoder(nn.Module):
    """UNet++ 嵌套 decoder。对角线 X[i,n-1-i] 为输出（low-res→high-res）。

    参数：upsample_mode示例 'transpose' 可学 或 'trilinear' 无参；
    skip_attention=True 时用 X[i,0] 对上采样分支作 gate (Oktay 2018)。
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

        # 以 'i_j' 为 key 保证 ModuleDict 注册顺序确定。
        self.upsamples = nn.ModuleDict()
        self.blocks = nn.ModuleDict()
        self.gates = nn.ModuleDict() if skip_attention else None

        for i in range(n - 1):
            for j in range(1, n - i):
                key = f"{i}_{j}"
                # X[i+1, j-1] (enc[i+1] 通道) 上采样到 enc[i]。
                self.upsamples[key] = Upsample(
                    encoder_channels[i + 1],
                    encoder_channels[i],
                    mode=upsample_mode,
                    spatial_dims=spatial_dims,
                )
                # 融合：j 个同深度节点 + 1 上采样 = (j+1)*enc[i]。
                fused_ch = (j + 1) * encoder_channels[i]
                self.blocks[key] = stage_builder(fused_ch, encoder_channels[i])

                if skip_attention:
                    self.gates[key] = AttentionGate3D(
                        x_ch=encoder_channels[i],
                        g_ch=encoder_channels[i],
                        spatial_dims=spatial_dims,
                    )

        # 对角线 X[i, n-1-i] 通道（low-res→high-res），与经典 Decoder 一致。
        self.out_channels = [encoder_channels[n - 2 - k] for k in range(n - 1)]

    def forward(self, encoder_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """encoder_features：high-res 优先；返回对角线 low-res→high-res。"""
        n = self.n
        x: List[List[torch.Tensor]] = [[None] * (n - i) for i in range(n)]  # type: ignore
        for i in range(n):
            x[i][0] = encoder_features[i]

        # 逐列填充（j 仅依赖 j-1）。
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
