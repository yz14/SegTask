"""JEPA⑦ 隐空间掩码预测专有模块：轻量卷积预测器 ``JEPAPredictor``。

JEPA 不做像素重建、不做实例对比，而是在**特征空间**预测被遮区域的表征：上下文编码器
（看 mask-token 遮挡后的输入）出上下文特征图，预测器据此预测被遮位点的特征，去逼近
EMA 目标编码器（看完整输入）在同位点的特征（目标侧 stop-grad）。

预测器刻意轻量且**非对称**（防坍缩的关键之一）：几层 3×3 ``ConvNormAct`` + 1×1 输出，
在 encoder 选定层级的特征图上逐位点工作。I-JEPA 原生用 ViT + token 上的 predictor 并显式
注入目标位置 query；CNN 适配里位置由卷积的平移等变性隐式承载（合理简化），预测器对整张
特征图输出预测、再仅在被遮位点计损。下游迁移的是编码器，预测器用完即弃。
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from taskcore.models.blocks import _CONV, ConvNormAct
from taskcore.models.factory import build_backbone

logger = logging.getLogger(__name__)


class JEPAPredictor(nn.Module):
    """轻量卷积预测器：(B, C, *grid) 上下文特征 → (B, C, *grid) 预测的目标特征。

    ``depth`` 个 3×3 ``ConvNormAct``（首层 C→hidden，其余 hidden→hidden）+ 1×1 输出
    hidden→C。保持空间分辨率不变（逐位点预测）。
    """

    def __init__(
        self,
        channels    : int,
        hidden      : int,
        depth       : int,
        spatial_dims: int = 3,
        norm_type   : str = "instance",
        norm_groups : int = 8,
        activation  : str = "leakyrelu"):
        super().__init__()
        depth = max(int(depth), 1)
        c_in = int(channels)
        layers = []
        for _ in range(depth):
            layers.append(ConvNormAct(
                c_in, int(hidden), kernel_size=3, stride=1, padding=1,
                norm_type=norm_type, norm_groups=norm_groups,
                activation=activation, spatial_dims=int(spatial_dims)))
            c_in = int(hidden)
        self.body = nn.Sequential(*layers)
        self.out = _CONV[int(spatial_dims)](int(hidden), int(channels),
                                            kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(x))


def build_jepa_encoder(cfg) -> nn.Module:
    """构造单个 encoder（复用 ``build_model``，保证与下游逐参数同名同形）。"""
    arch = str(cfg.model.arch).lower()
    if arch != "unet":
        raise ValueError(
            f"build_jepa_encoder requires model.arch=='unet'; got {arch!r}.")
    return build_backbone(cfg)


def build_jepa_predictor(cfg, channels: int, hidden: int, depth: int
                         ) -> JEPAPredictor:
    """构造预测器：``hidden<=0`` 时取特征通道数 ``channels``。"""
    h = int(hidden) if int(hidden) > 0 else int(channels)
    pred = JEPAPredictor(
        channels=int(channels), hidden=h, depth=int(depth),
        spatial_dims=int(cfg.model.spatial_dims),
        norm_type=cfg.model.norm_type, norm_groups=cfg.model.norm_groups,
        activation=cfg.model.activation)
    n = sum(p.numel() for p in pred.parameters())
    logger.info("Built JEPAPredictor: channels=%d, hidden=%d, depth=%d, "
                "params=%.3fM.", channels, h, depth, n / 1e6)
    return pred


__all__ = ["JEPAPredictor", "build_jepa_encoder", "build_jepa_predictor"]
