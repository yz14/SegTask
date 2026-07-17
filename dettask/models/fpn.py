"""FPN 适配层：segtask Decoder 金字塔 → 通道对齐的检测特征金字塔。

分割 ``Decoder`` 输出 ``[dec_low_res, ..., dec_high_res]`` 天然就是特征金字塔
（Plan §3.2「Retina U-Net」思路）；本模块仅做 1×1 通道对齐 + 3×3 平滑，
dims 参数化 2D/3D 同一实现。层选择 ``levels``（0 = 最低分辨率）。
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn

from taskcore.models.blocks import _CONV

__all__ = ["FPNAdapter"]


class FPNAdapter(nn.Module):
    """Decoder 金字塔 → ``fpn_channels`` 统一通道的检测金字塔。

    输出保持 low-res → high-res 序（stride 递减），与 anchor 生成约定一致。
    """

    def __init__(self, decoder_channels: Sequence[int], fpn_channels: int,
                 levels: Sequence[int], spatial_dims: int):
        super().__init__()
        conv = _CONV[spatial_dims]
        self.levels = list(levels)
        self.laterals = nn.ModuleList(
            conv(decoder_channels[i], fpn_channels, kernel_size=1)
            for i in self.levels)
        self.smooth = nn.ModuleList(
            conv(fpn_channels, fpn_channels, kernel_size=3, padding=1)
            for _ in self.levels)
        self.out_channels = int(fpn_channels)

    def forward(self, pyramid: List[torch.Tensor]) -> List[torch.Tensor]:
        return [s(l(pyramid[i]))
                for i, l, s in zip(self.levels, self.laterals, self.smooth)]
