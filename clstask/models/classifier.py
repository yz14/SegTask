"""分类模型：encoder（任意 backbone）+ 池化 + ``cls_head``。

命名约定：backbone 一律挂在 ``self.encoder``、任务头挂在 ``self.cls_head``——
与 segtask（``encoder``/``seg_head``）、ssltask 探针一致，保证 SSL/分割
checkpoint 的 ``encoder.*`` 权重可 strict=False 直接命中。

输出（logits）：

* ``label_granularity='volume'`` → ``(B, K)``（多标签）或单标签 CE 的 ``(B, K)``；
* ``label_granularity='slice'``  → ``(B, K, D)``，D = patch 深度：
  - 2.5D（spatial_dims=2）：深度折在通道里已被 encoder 混合，头输出 K×D 再
    reshape（与分割 2.5D 头 ``num_fg×D`` 同思想）；
  - 3D：只池化 H/W、保留 z 轴 → 逐深度共享 MLP（1×1 Conv1d）→ 沿 z 线性
    插值回 D（encoder 下采样后 z' < D）。
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pool(feat: torch.Tensor, mode: str, keep_axes: int = 0) -> torch.Tensor:
    """全局池化：聚合末 ``ndim-2-keep_axes`` 个空间轴。

    ``keep_axes=0`` → (B, C[, kept...]) 全空间池化；``keep_axes=1`` 且输入
    (B, C, D, H, W) → 只池化 H/W，返回 (B, C, D)。
    """
    dims = tuple(range(2 + keep_axes, feat.ndim))
    if mode == "avg":
        return feat.mean(dim=dims)
    if mode == "max":
        return feat.amax(dim=dims)
    if mode == "avgmax":
        return torch.cat([feat.mean(dim=dims), feat.amax(dim=dims)], dim=1)
    raise ValueError(f"Unknown pooling: {mode!r}")


def _mlp(in_dim: int, out_dim: int, hidden: int, dropout: float) -> nn.Module:
    if hidden <= 0:
        return nn.Linear(in_dim, out_dim)
    layers: List[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU(inplace=True)]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class VolumeClsHead(nn.Module):
    """全空间池化 → MLP → (B, K)。"""

    def __init__(self, in_dim: int, num_classes: int, hidden: int,
                 dropout: float, pooling: str):
        super().__init__()
        self.pooling = pooling
        eff = in_dim * (2 if pooling == "avgmax" else 1)
        self.mlp = _mlp(eff, num_classes, hidden, dropout)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.mlp(_pool(feat, self.pooling))


class SliceClsHead2D(nn.Module):
    """2.5D 逐 slice 头：池化 (H, W) → MLP 输出 K×D → (B, K, D)。"""

    def __init__(self, in_dim: int, num_classes: int, depth: int, hidden: int,
                 dropout: float, pooling: str):
        super().__init__()
        self.num_classes = num_classes
        self.depth = depth
        self.pooling = pooling
        eff = in_dim * (2 if pooling == "avgmax" else 1)
        self.mlp = _mlp(eff, num_classes * depth, hidden, dropout)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(_pool(feat, self.pooling))
        return logits.view(-1, self.num_classes, self.depth)


class SliceClsHead3D(nn.Module):
    """3D 逐 slice 头：池化 (H, W) 保 z → 逐深度共享 MLP → 线性插值回 D。"""

    def __init__(self, in_dim: int, num_classes: int, depth: int, hidden: int,
                 dropout: float, pooling: str):
        super().__init__()
        self.depth = depth
        eff = in_dim * (2 if pooling == "avgmax" else 1)
        self.pooling = pooling
        if hidden <= 0:
            self.mlp = nn.Conv1d(eff, num_classes, kernel_size=1)
        else:
            layers: List[nn.Module] = [
                nn.Conv1d(eff, hidden, kernel_size=1), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Conv1d(hidden, num_classes, kernel_size=1))
            self.mlp = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim != 5:
            raise ValueError(
                f"SliceClsHead3D expects (B, C, D, H, W); got {feat.shape}.")
        pooled = _pool(feat, self.pooling, keep_axes=1)   # (B, C', z')
        logits = self.mlp(pooled)                          # (B, K, z')
        if logits.shape[-1] != self.depth:
            logits = F.interpolate(logits, size=self.depth, mode="linear",
                                   align_corners=False)
        return logits                                      # (B, K, D)


class Classifier(nn.Module):
    """encoder + cls_head；``forward(x) -> logits``。"""

    def __init__(self, encoder: nn.Module, cls_head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.cls_head = cls_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        feat = feats[-1] if isinstance(feats, (list, tuple)) else feats
        return self.cls_head(feat)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = [
    "Classifier", "VolumeClsHead", "SliceClsHead2D", "SliceClsHead3D",
]
