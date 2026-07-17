"""VICRegL 模块：共享 encoder + 全局 MLP 投影头 + 稠密 1×1 卷积投影头。

VICRegL（Bardes et al., NeurIPS 2022）= VICReg 全局项（invariance/variance/
covariance）+ 稠密局部项（位置/特征匹配位点对上的同套 VIC 损失）。孪生结构
（两视图共享同一套权重，无 EMA 教师/负样本队列）。

骨干与其它方法一致：``segtask_v1.models.factory.build_model(cfg).encoder``
（同名同形 → 下游 ``train.pretrain`` strict=False 仅命中 ``encoder.*``；
投影头为 SSL 专用，导出时丢弃）。小 batch 3D 场景不用 BatchNorm（与
``DINOHead`` 的取舍一致），全局头用 LayerNorm、稠密头无归一化。
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from taskcore.models.blocks import _CONV
from taskcore.models.factory import build_model


class GlobalProjector(nn.Module):
    """池化瓶颈特征 → VICReg 全局嵌入（Linear-LN-ReLU ×2 → Linear）。"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(out_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseProjector(nn.Module):
    """稠密特征 (B,C,*sp) → 局部嵌入 (B,out,*sp)（1×1 conv MLP，保分辨率）。"""

    def __init__(self, in_ch: int, hidden_ch: int, out_ch: int,
                 spatial_dims: int = 3):
        super().__init__()
        conv = _CONV[int(spatial_dims)]
        self.net = nn.Sequential(
            conv(int(in_ch), int(hidden_ch), kernel_size=1),
            nn.ReLU(inplace=True),
            conv(int(hidden_ch), int(out_ch), kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VICRegLNet(nn.Module):
    """encoder + 全局/稠密投影头。``forward(x)`` → ``(global_emb, dense_emb)``。

    * ``global_emb``: 瓶颈特征全局平均池化 → :class:`GlobalProjector` → (B, Dg)；
    * ``dense_emb``: ``feature_level`` 级特征 → :class:`DenseProjector` → (B, Dl, *sp)。
    """

    def __init__(self, encoder: nn.Module, global_proj: GlobalProjector,
                 dense_proj: DenseProjector, feature_level: int):
        super().__init__()
        self.encoder = encoder
        self.global_proj = global_proj
        self.dense_proj = dense_proj
        self.feature_level = int(feature_level)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats: List[torch.Tensor] = self.encoder(x)
        pooled = feats[-1].mean(dim=tuple(range(2, feats[-1].ndim)))
        return self.global_proj(pooled), self.dense_proj(feats[self.feature_level])


def build_vicregl_net(cfg, proj_dim: int, hidden_dim: int,
                      dense_proj_dim: int, feature_level: int) -> VICRegLNet:
    """由下游同一 ``build_model`` 的 encoder 构造 :class:`VICRegLNet`。"""
    encoder = build_model(cfg).encoder
    bott_ch = int(cfg.model.encoder_channels[-1])
    feat_ch = int(cfg.model.encoder_channels[int(feature_level)])
    global_proj = GlobalProjector(bott_ch, int(hidden_dim), int(proj_dim))
    dense_proj = DenseProjector(
        feat_ch, max(int(dense_proj_dim), feat_ch), int(dense_proj_dim),
        spatial_dims=int(cfg.model.spatial_dims))
    return VICRegLNet(encoder, global_proj, dense_proj, int(feature_level))


__all__ = ["GlobalProjector", "DenseProjector", "VICRegLNet",
           "build_vicregl_net"]
