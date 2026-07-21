"""根据 config 构建 gentask 的共享 2.5D/3D 图像到图像模型。"""

from __future__ import annotations

import logging
from typing import List

from ..config import Config
from taskcore.models.factory import build_model as build_taskcore_model
from taskcore.models.topology import build_topology

logger = logging.getLogger(__name__)


def build_model(cfg: Config):
    """顶层模型工厂。

    生成任务（``cfg.is_generation``）分派到回归 / 扩散生成模型；否则按
    ``cfg.model.arch`` 构造分割 backbone（见 ``build_backbone``）。
    """
    if getattr(cfg, "is_generation", False):
        from .generation import build_generation_model
        return build_generation_model(cfg)
    return build_backbone(cfg)


def _model_axis_scales(cfg: Config, spatial_dims: int) -> List[int]:
    """模型空间轴的逐轴超分倍率（与退化算子 axis_scales 一致）。"""
    per_axis = list(cfg.task.sr_scale_per_axis)
    if per_axis:
        return [int(s) for s in per_axis]
    return [int(cfg.task.sr_scale)] * spatial_dims


def _build_sisr_backbone(cfg: Config):
    """构建经典 SISR backbone（EDSR / RCAN，post-upsampling）。

    输入为真 LR 网格（配套 ``SuperResDegradation(keep_lr_size=True)``），
    上采头按逐轴倍率把特征放大回 HR 网格。
    """
    from .sisr import SISRNet

    mc = cfg.model
    topo = build_topology(cfg)
    factors = _model_axis_scales(cfg, topo.spatial_dims)
    model = SISRNet(
        in_channels  = mc.in_channels,
        out_channels = topo.out_classes,
        factors      = factors,
        arch         = str(mc.arch).lower(),
        channels     = mc.sisr.channels,
        num_blocks   = mc.sisr.num_blocks,
        num_groups   = mc.sisr.num_groups,
        res_scale    = mc.sisr.res_scale,
        activation   = mc.unet.activation,
        se_reduction = mc.unet.se_reduction,
        spatial_dims = topo.spatial_dims)
    logger.info(
        "Built SISRNet [%s]: total=%.2fM, channels=%d, blocks=%d, groups=%d, "
        "factors=%s, in_ch=%d, out_ch=%d, spatial_dims=%d",
        mc.arch, model.param_count()["total"] / 1e6, mc.sisr.channels,
        mc.sisr.num_blocks, mc.sisr.num_groups, factors,
        mc.in_channels, topo.out_classes, topo.spatial_dims)
    return model


def build_backbone(cfg: Config):
    """按 `model.arch` 构造共享 backbone：UNet、ADM、EDM2、EDSR/RCAN。"""
    arch = str(cfg.model.arch).lower()
    if cfg.model.grad_checkpointing and arch in ("edsr", "rcan"):
        logger.warning(
            "model.grad_checkpointing=True is not supported for arch=%r "
            "(only 'unet' | 'adm' | 'edm2'); ignored.", arch)
    if arch in ("edsr", "rcan"):
        return _build_sisr_backbone(cfg)
    if arch == "adm":
        from taskcore.models.adm_unet import build_adm_backbone
        return build_adm_backbone(cfg)
    if arch == "edm2":
        from taskcore.models.edm2_unet import build_edm2_backbone
        return build_edm2_backbone(cfg)
    if arch != "unet":
        raise ValueError(
            f"Unknown model.arch: {arch!r}. "
            f"Valid: 'unet' | 'adm' | 'edm2' | 'edsr' | 'rcan'.")
    # 生成/超分 UNet 主线：UNet++ 门控上采样分支（分割主线为门控 skips）。
    return build_taskcore_model(cfg, attn_gate_target="upsample")
