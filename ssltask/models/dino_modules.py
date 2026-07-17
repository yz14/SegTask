"""DINO④ 自蒸馏专有模块：投影头 ``DINOHead`` + 编码器-投影网 ``DINONet``。

DINO 把 image-only 自监督建模为"学生网络匹配 EMA 教师网络的软分配"：两者都是
``encoder → 全局平均池化 → DINOHead``（MLP 投影 + L2 归一化 + 权重归一化的原型层），
输出 ``out_dim`` 个原型上的 logits。学生看全部裁剪、教师只看 global 裁剪，损失为
教师（centering+sharpening 后）与学生分布的交叉熵（见 ``methods/dino.py``）。

骨干复用：``DINONet.encoder`` 取自 ``segtask_v1.models.factory.build_model(cfg).encoder``
（保证与下游逐参数同名同形）；DINO 不预训练解码器，故下游 ``train.pretrain``
（strict=False）仅命中 ``encoder.*``、``decoder.*``/``seg_head.*`` 保持随机。``DINOHead``
与 student/teacher 前缀在导出时被丢弃（见 ``DINOMethod.export_backbone_state_dict``）。
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from taskcore.models.factory import build_model

logger = logging.getLogger(__name__)


class DINOHead(nn.Module):
    """DINO 投影头：MLP → L2 归一化 → 权重归一化原型层（输出 ``out_dim`` logits）。

    在 ``(B, in_dim)`` 的全局表征向量上工作（与 spatial_dims 无关）。末层用
    ``weight_norm`` 且把幅度 ``g`` 固定为 1（DINO 稳定化技巧，避免原型尺度漂移）。
    """

    def __init__(
        self,
        in_dim        : int,
        out_dim       : int,
        hidden_dim    : int = 2048,
        bottleneck_dim: int = 256,
        n_layers      : int = 3,
        use_bn        : bool = False):
        super().__init__()
        n_layers = max(int(n_layers), 1)
        if n_layers == 1:
            self.mlp: nn.Module = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False))
        # 幅度向量固定为 1 且不训练（仅训练方向 weight_v）。
        self.last_layer.weight_g.data.fill_(1.0)
        self.last_layer.weight_g.requires_grad = False

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class DINONet(nn.Module):
    """``encoder → 全局平均池化 → DINOHead``。forward: (B,C,*spatial) → (B, out_dim)。"""

    def __init__(self, encoder: nn.Module, head: DINOHead, spatial_dims: int):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.spatial_dims = int(spatial_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        z = feats[-1]                       # bottleneck (B, C, *grid)
        z = z.flatten(2).mean(dim=2)        # 全局平均池化 → (B, C)
        return self.head(z)


def build_dino_net(
    cfg,
    out_dim       : int,
    hidden_dim    : int = 2048,
    bottleneck_dim: int = 256,
    n_layers      : int = 3,
    use_bn        : bool = False) -> DINONet:
    """构造单个 ``DINONet``（复用 ``build_model`` 的 encoder，保证下游同名同形）。

    输入维度取 bottleneck 通道数 ``encoder_channels[-1]``。解码器/分割头被丢弃。
    """
    arch = str(cfg.model.arch).lower()
    if arch != "unet":
        raise ValueError(
            f"build_dino_net requires model.arch=='unet'; got {arch!r}.")
    seg_model = build_model(cfg)            # 同一构建路径，确保 encoder 同名同形
    encoder = seg_model.encoder
    in_dim = int(cfg.model.encoder_channels[-1])
    head = DINOHead(
        in_dim=in_dim, out_dim=int(out_dim), hidden_dim=int(hidden_dim),
        bottleneck_dim=int(bottleneck_dim), n_layers=int(n_layers),
        use_bn=bool(use_bn))
    return DINONet(encoder, head, int(cfg.model.spatial_dims))


__all__ = ["DINOHead", "DINONet", "build_dino_net"]
