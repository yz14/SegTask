"""dettask 模型工厂：共享 backbone 装配 + 四模板分派 + 预训练迁移。

DETR 头只用金字塔最低分辨率层（``det.fpn_levels`` 仍决定 FPN 输出，
DETR 取其第 0 个 = 最深层）。
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from taskcore.config.core import Config as SegConfig
from taskcore.models.factory import build_model as build_seg_model
from taskcore.engine.checkpoint import (
    extract_model_state_dict,
    strip_common_prefixes,
)

from ..config import DetConfig, resolve_num_classes
from .detector import DetectorModel
from .fpn import FPNAdapter
from .heads.detr import DETRHead
from .heads.fcos import FCOSHead
from .heads.frcnn import FasterRCNNHead
from .heads.retina import RetinaHead

logger = logging.getLogger(__name__)

_HEADS = {
    "retinanet": RetinaHead,
    "fcos": FCOSHead,
    "faster_rcnn": FasterRCNNHead,
    "detr": DETRHead,
}


def load_pretrained_backbone(model: DetectorModel, ckpt_path: str) -> None:
    """SSL/分割 ckpt → ``encoder.*``（重建式亦命中 ``decoder.*``），
    strict=False + 命中统计；0 命中报错（几何不一致不静默）。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    full_sd, source = extract_model_state_dict(ckpt, prefer_ema=False)
    sd = strip_common_prefixes(full_sd)
    logger.info("pretrained_ckpt state_dict source: %s", source)
    total_matched = 0
    for prefix, mod in (("encoder.", model.encoder),
                        ("decoder.", model.decoder)):
        sub = {k[len(prefix):]: v for k, v in sd.items()
               if k.startswith(prefix)}
        own = mod.state_dict()
        matched = {k: v for k, v in sub.items()
                   if k in own and own[k].shape == v.shape}
        if matched:
            mod.load_state_dict(matched, strict=False)
        logger.info("Pretrained %s* matched %d/%d tensors from %s.",
                    prefix, len(matched), len(own), ckpt_path)
        total_matched += len(matched)
    if total_matched == 0:
        raise RuntimeError(
            f"pretrained_ckpt {ckpt_path!r}: 0 encoder/decoder tensors "
            "matched. Geometry (patch_mode/spatial_dims/in_channels) or "
            "backbone differs from the pretraining config.")


def build_detector(cfg: SegConfig, det: DetConfig) -> DetectorModel:
    seg = build_seg_model(cfg)
    encoder, decoder = seg.encoder, seg.decoder
    dec_channels = list(decoder.out_channels)        # low-res → high-res
    levels = list(det.fpn_levels) or list(range(len(dec_channels)))
    for i in levels:
        if not 0 <= i < len(dec_channels):
            raise ValueError(
                f"det.fpn_levels {levels} out of range for decoder with "
                f"{len(dec_channels)} levels.")
    if det.arch in ("retinanet", "faster_rcnn") and det.anchor_sizes \
            and len(det.anchor_sizes) != len(levels):
        raise ValueError(
            f"det.anchor_sizes has {len(det.anchor_sizes)} entries but "
            f"{len(levels)} FPN levels are used ({levels}); lengths must "
            "match (or leave anchor_sizes empty for stride-based auto).")
    sd = int(cfg.model.spatial_dims)
    fpn = FPNAdapter(dec_channels, det.fpn_channels, levels, sd)
    K = resolve_num_classes(det, cfg)
    head = _HEADS[det.arch](det.fpn_channels, K, det, sd)
    model = DetectorModel(encoder, decoder, fpn, head)
    if det.pretrained_ckpt:
        load_pretrained_backbone(model, det.pretrained_ckpt)
    if det.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        logger.info("Encoder frozen.")
    logger.info(
        "Detector built: arch=%s, K=%d, spatial_dims=%d, fpn_levels=%s, "
        "params=%.2fM %s", det.arch, K, sd, levels,
        model.param_count() / 1e6, model.state_summary())
    return model


__all__ = ["build_detector", "load_pretrained_backbone"]
