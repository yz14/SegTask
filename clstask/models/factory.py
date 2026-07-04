"""clstask 模型工厂：backbone 构建 + 分类头装配 + SSL/分割权重迁移。

四模板：

* ResNet / ConvNeXt —— ``cls.backbone='encoder'``，复用
  ``segtask_v1.models.factory.build_model(cfg).encoder``（具体由
  ``cfg.model.backbone`` 决定）。与 SSL 预训练**同一构建路径**，参数同名同形，
  ``cls.pretrained_ckpt`` 可 strict=False 直接命中 ``encoder.*``。
* DenseNet —— :class:`clstask.models.densenet.DenseNetEncoder`（2D/3D）。
* ViT —— :class:`clstask.models.vit.ViTEncoder`（2D/3D）。
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn

from segtask_v1.config import Config as SegConfig
from segtask_v1.models.factory import build_model as build_seg_model
from segtask_v1.trainer.checkpoint import (
    extract_model_state_dict,
    strip_common_prefixes,
)

from ..config import ClsConfig, resolve_num_classes
from .classifier import (
    Classifier,
    SliceClsHead2D,
    SliceClsHead3D,
    VolumeClsHead,
)
from .densenet import DenseNetEncoder
from .vit import ViTEncoder

logger = logging.getLogger(__name__)


def _build_encoder(cfg: SegConfig, cls: ClsConfig) -> Tuple[nn.Module, int]:
    """按 ``cls.backbone`` 构建特征提取器；返回 (encoder, 末级通道数)。"""
    sd = int(cfg.model.spatial_dims)
    in_ch = int(cfg.model.in_channels)
    if cls.backbone == "encoder":
        encoder = build_seg_model(cfg).encoder
        return encoder, int(cfg.model.encoder_channels[-1])
    if cls.backbone == "densenet":
        encoder = DenseNetEncoder(
            in_channels=in_ch,
            growth_rate=cls.densenet_growth_rate,
            block_layers=list(cls.densenet_block_layers),
            compression=cls.densenet_compression,
            stem_channels=cls.densenet_stem_channels,
            norm_type=cfg.model.norm_type,
            norm_groups=cfg.model.norm_groups,
            activation=cfg.model.activation,
            spatial_dims=sd)
        return encoder, encoder.out_channels_list[-1]
    if cls.backbone == "vit":
        encoder = ViTEncoder(
            in_channels=in_ch,
            embed_dim=cls.vit_embed_dim,
            depth=cls.vit_depth,
            num_heads=cls.vit_num_heads,
            mlp_ratio=cls.vit_mlp_ratio,
            drop_path_rate=cls.vit_drop_path_rate,
            patch_size=list(cls.vit_patch_size),
            input_size=list(cfg.data.patch_size),
            spatial_dims=sd)
        return encoder, cls.vit_embed_dim
    raise ValueError(f"Unknown cls.backbone: {cls.backbone!r}")


def _build_head(cfg: SegConfig, cls: ClsConfig, feat_dim: int) -> nn.Module:
    num_classes = resolve_num_classes(cls, cfg)
    depth = int(cfg.data.patch_size[0])
    sd = int(cfg.model.spatial_dims)
    if cls.label_granularity == "volume":
        return VolumeClsHead(feat_dim, num_classes, cls.head_hidden_dim,
                             cls.head_dropout, cls.pooling)
    if sd == 2:
        return SliceClsHead2D(feat_dim, num_classes, depth,
                              cls.head_hidden_dim, cls.head_dropout,
                              cls.pooling)
    return SliceClsHead3D(feat_dim, num_classes, depth, cls.head_hidden_dim,
                          cls.head_dropout, cls.pooling)


def load_pretrained_encoder(model: Classifier, ckpt_path: str) -> None:
    """从 SSL/分割 checkpoint 迁移 ``encoder.*`` 权重（strict=False）。

    命中率打日志（TODO.md 可验证性要求）；若一条都没命中直接报错——
    说明几何/backbone 与预训练不一致，静默跳过会掩盖配置错误。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    full_sd, source = extract_model_state_dict(ckpt, prefer_ema=False)
    sd = strip_common_prefixes(full_sd)
    logger.info("pretrained_ckpt state_dict source: %s", source)
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items()
              if k.startswith("encoder.")}
    if not enc_sd:
        raise KeyError(
            f"pretrained_ckpt {ckpt_path!r} has no 'encoder.*' keys; "
            "expected an ssltask/segtask checkpoint.")
    own = model.encoder.state_dict()
    matched = {k: v for k, v in enc_sd.items()
               if k in own and own[k].shape == v.shape}
    missing = [k for k in own if k not in matched]
    skipped = [k for k in enc_sd if k not in matched]
    if not matched:
        raise RuntimeError(
            f"pretrained_ckpt {ckpt_path!r}: 0/{len(own)} encoder tensors "
            "matched. Geometry (patch_mode/spatial_dims/in_channels) or "
            "backbone differs from the pretraining config.")
    model.encoder.load_state_dict(matched, strict=False)
    logger.info(
        "Pretrained encoder loaded from %s: matched %d/%d tensors "
        "(%d own-missing, %d ckpt-skipped).",
        ckpt_path, len(matched), len(own), len(missing), len(skipped))


def build_classifier(cfg: SegConfig, cls: ClsConfig) -> Classifier:
    """构建分类模型并（可选）加载预训练 encoder / 冻结 encoder。"""
    encoder, feat_dim = _build_encoder(cfg, cls)
    head = _build_head(cfg, cls, feat_dim)
    model = Classifier(encoder, head)
    if cls.pretrained_ckpt:
        load_pretrained_encoder(model, cls.pretrained_ckpt)
    if cls.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        logger.info("Encoder frozen (linear-probe mode).")
    logger.info(
        "Classifier built: backbone=%s (feat_dim=%d), granularity=%s, K=%d, "
        "spatial_dims=%d, params=%.2fM",
        cls.backbone, feat_dim, cls.label_granularity,
        resolve_num_classes(cls, cfg), cfg.model.spatial_dims,
        model.param_count() / 1e6)
    return model


__all__ = ["build_classifier", "load_pretrained_encoder"]
