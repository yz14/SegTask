"""UNet3+ 全尺度 skip decoder (Huang 2020)。每个节点 D_i 融合 n 分支：E_j (j<i 池化、j=i 同级、j=n-1 上采) 与已计 D_j (i<j<n-1 上采)。分支 cat 后融合为统一 fused_channels。"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import INTERP_SMOOTH, AttentionGate3D, ConvNormAct, checkpoint_if


class UNet3PDecoder(nn.Module):
    """UNet3+ 全尺度 skip decoder；out_channels low-res→high-res。

    参数：cat_channels 分支宽度（论文64）；fused_channels 节点输出宽（0=cat_channels*n）；
    skip_attention=True 时用同级 E_i 对各分支作 gate。
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
        grad_checkpointing: bool = False,
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
        # 梯度检查点：逐节点包裹融合卷积前向，反向重算以省激活显存。
        self.grad_checkpointing = bool(grad_checkpointing)

        def _cna(in_ch: int, out_ch: int) -> ConvNormAct:
            return ConvNormAct(
                in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                norm_type=norm_type, norm_groups=norm_groups,
                activation=activation, spatial_dims=spatial_dims)

        # 每个深度 i 构造 n 个分支卷积 + 1 个融合卷积。
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
                    src_ch = self.fused_ch       # 更深 decoder 节点
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
            # adaptive pool 处理非 2^k 特征图。
            if self.spatial_dims == 3:
                return F.adaptive_max_pool3d(src, target_shape)
            return F.adaptive_max_pool2d(src, target_shape)
        return F.interpolate(
            src, size=target_shape,
            mode=INTERP_SMOOTH[self.spatial_dims], align_corners=False)

    def forward(self, encoder_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """返回 [low_res, ..., high_res] decoder 特征。"""
        n = self.n
        decoder_nodes: List[torch.Tensor] = [None] * (n - 1)  # type: ignore

        # 自深至浅迭代，保证需要的 D_j 已就绪。
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
            decoder_nodes[i] = checkpoint_if(
                self.grad_checkpointing, self.fusions[i], fused)

        return list(reversed(decoder_nodes))
