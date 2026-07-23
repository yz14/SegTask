"""DenseNet-BC 特征提取器（2D/3D 通用，``spatial_dims`` 参数化）。

DenseNet（Huang et al., CVPR 2017 best paper）是医学影像分类最常用的 CNN 之一
（低参数量 + 特征复用 + 隐式深监督）。本实现为 BC 变体（bottleneck +
compression），接口与 taskcore ``Encoder`` 对齐：``forward(x) -> List[feat]``
（逐 stage 特征，最后一项为最深特征），供统一的分类头使用。

归一化/激活默认 instance + leakyrelu，与仓库 3D 小 batch 惯例一致（经典
DenseNet 为 BN+ReLU，可经参数切换）。
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from taskcore.models.blocks import checkpoint_if, get_activation, get_norm


def _conv_nd(spatial_dims: int):
    return nn.Conv3d if spatial_dims == 3 else nn.Conv2d


def _avgpool_nd(spatial_dims: int):
    return nn.AvgPool3d if spatial_dims == 3 else nn.AvgPool2d


class DenseLayer(nn.Module):
    """norm→act→1×1(4g) → norm→act→3×3(g)，输出与输入 cat。"""

    def __init__(self, in_ch: int, growth_rate: int, norm_type: str,
                 norm_groups: int, activation: str, spatial_dims: int):
        super().__init__()
        conv = _conv_nd(spatial_dims)
        inter = 4 * growth_rate
        self.net = nn.Sequential(
            get_norm(norm_type, in_ch, norm_groups, spatial_dims),
            get_activation(activation),
            conv(in_ch, inter, kernel_size=1, bias=False),
            get_norm(norm_type, inter, norm_groups, spatial_dims),
            get_activation(activation),
            conv(inter, growth_rate, kernel_size=3, padding=1, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.net(x)], dim=1)


class DenseBlock(nn.Sequential):
    def __init__(self, in_ch: int, num_layers: int, growth_rate: int,
                 norm_type: str, norm_groups: int, activation: str,
                 spatial_dims: int):
        layers = [
            DenseLayer(in_ch + i * growth_rate, growth_rate, norm_type,
                       norm_groups, activation, spatial_dims)
            for i in range(num_layers)]
        super().__init__(*layers)
        self.out_channels = in_ch + num_layers * growth_rate


class Transition(nn.Module):
    """norm→act→1×1 压缩 → avgpool(2)。"""

    def __init__(self, in_ch: int, out_ch: int, norm_type: str,
                 norm_groups: int, activation: str, spatial_dims: int):
        super().__init__()
        conv = _conv_nd(spatial_dims)
        self.net = nn.Sequential(
            get_norm(norm_type, in_ch, norm_groups, spatial_dims),
            get_activation(activation),
            conv(in_ch, out_ch, kernel_size=1, bias=False),
            _avgpool_nd(spatial_dims)(kernel_size=2, stride=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseNetEncoder(nn.Module):
    """DenseNet-BC 主干：stem → [DenseBlock → Transition]×N → 末 norm+act。

    ``forward(x) -> List[feat]``：每个 DenseBlock 之后（Transition 之前）的
    特征各一项；``out_channels_list`` 给出对应通道数。
    """

    def __init__(
        self,
        in_channels : int,
        growth_rate : int = 16,
        block_layers: List[int] = (4, 8, 12, 8),
        compression : float = 0.5,
        stem_channels: int = 32,
        norm_type   : str = "instance",
        norm_groups : int = 8,
        activation  : str = "leakyrelu",
        spatial_dims: int = 3,
        grad_checkpointing: bool = False):
        super().__init__()
        # 逐 DenseBlock 梯度检查点（dense cat 激活是显存大头）；eval/no_grad
        # 下零开销，语义见 taskcore.models.blocks.checkpoint_if。
        self.grad_checkpointing = bool(grad_checkpointing)
        if spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3; got {spatial_dims}")
        if not 0.0 < compression <= 1.0:
            raise ValueError(f"compression must be in (0, 1]; got {compression}")
        conv = _conv_nd(spatial_dims)
        self.spatial_dims = spatial_dims
        self.stem = conv(in_channels, stem_channels, kernel_size=3, padding=1,
                         bias=False)

        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        self.out_channels_list: List[int] = []
        ch = stem_channels
        for i, n_layers in enumerate(block_layers):
            block = DenseBlock(ch, int(n_layers), growth_rate, norm_type,
                               norm_groups, activation, spatial_dims)
            self.blocks.append(block)
            ch = block.out_channels
            self.out_channels_list.append(ch)
            if i < len(block_layers) - 1:
                out_ch = max(int(ch * compression), 1)
                self.transitions.append(Transition(
                    ch, out_ch, norm_type, norm_groups, activation,
                    spatial_dims))
                ch = out_ch
        self.final = nn.Sequential(
            get_norm(norm_type, self.out_channels_list[-1], norm_groups,
                     spatial_dims),
            get_activation(activation))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        features: List[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            x = checkpoint_if(self.grad_checkpointing, block, x)
            if i == len(self.blocks) - 1:
                x = self.final(x)
            features.append(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        return features


__all__ = ["DenseNetEncoder", "DenseBlock", "DenseLayer", "Transition"]
