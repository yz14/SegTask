"""ADM / EDM2 装配诊断：被忽略的通用 model.* 字段统一告警（D1）。

ADM/EDM2 按论文固定结构（GN32+SiLU / MP），刻意不读通用 UNet 旋钮
（backbone / norm / activation / 上下采样 / skip / 注意力等）。用户在
arch='adm'|'edm2' 下设置这些字段不会报错、也不会生效——本模块在装配时
对「被忽略且非默认值」的字段发一条汇总 warning，提升可发现性。

不在此列的字段：
* 已被 validate 直接拒绝的（aux_topo_head / lift_2_5d_to_3d /
  selfattn_enabled / multirf_enabled）；
* 已有专项 warning 的（decoder_blocks_per_stage；edm2 的 stem_mode /
  aux_head_mode）；
* 经 sync 间接生效的（resenc_preset 展开进 encoder_blocks_per_stage）；
* 被 gate 字段控制的子字段（mednext_* 之于 backbone、aux_topo_* 之于
  aux_topo_head 等）——只告警 gate 本身，避免刷屏。
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from ..config.model_migration import FLAT_TO_NESTED

logger = logging.getLogger(__name__)

# 通用 UNet 旋钮：adm/edm2 都不消费（gate 层字段 + 独立字段）。
_IGNORED_UNET_FIELDS: Tuple[str, ...] = (
    "backbone",
    "block_type",
    "norm_type",
    "norm_groups",
    "activation",
    "drop_path_rate",
    "decoder_type",
    "unet3p_cat_channels",
    "downsample_mode",
    "upsample_mode",
    "upsample_norm_act",
    "anisotropic_pooling",
    "downsample_strides",
    "skip_mode",
    "attention_type",
    "se_reduction",
    "skip_attention",
    "attn_gate_norm",
    "grad_ckpt_encoder_stages",
    "grn_enabled",
    "convnext_layer_scale_init",
    "convnext_downsample_lnfirst",
    "mednext_expand_ratio",
    "mednext_kernel_size",
    "mednext_dilated_reparam",
)


def warn_ignored_model_fields(mc, arch: str) -> "List[str]":
    """对 arch='adm'|'edm2' 下被静默忽略且非默认值的 model 字段发汇总 warning。

    返回漂移字段名列表（供测试断言，旧扁平名口径）；无漂移时不发日志。
    D2 后扁平名经兼容 property 转发到嵌套段，读口径不变；默认值取同类
    全新实例（嵌套段各自的 dataclass 默认）。
    """
    arch = str(arch).lower()
    ignored = list(_IGNORED_UNET_FIELDS)
    # 对侧扩散家族的专属组同样被忽略（配置串台的常见来源）。
    other_prefix = "edm2_" if arch == "adm" else "adm_"
    ignored.extend(
        name for name in FLAT_TO_NESTED if name.startswith(other_prefix))

    pristine = type(mc)()
    drifted = [
        name for name in ignored
        if getattr(mc, name) != getattr(pristine, name)
    ]
    if drifted:
        logger.warning(
            "model.arch=%r uses a paper-faithful fixed structure and "
            "ignores these non-default model.* fields: %s. Remove them "
            "from the config (or switch to arch='unet') to avoid "
            "confusion.",
            arch,
            ", ".join(f"{n}={getattr(mc, n)!r}" for n in drifted))
    return drifted
