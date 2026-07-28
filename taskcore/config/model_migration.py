"""D2 迁移契约与兼容层：ModelConfig 扁平字段 → 「公共 + unet/adm/edm2 嵌套」。

本模块是 D2（ModelConfig 按 arch 拆嵌套）的单一真相源：

* 归属映射（``COMMON_FIELDS`` / ``UNET_FIELD_MAP`` / ``ADM_FIELD_MAP`` /
  ``EDM2_FIELD_MAP`` / 汇总 ``FLAT_TO_NESTED``）；
* 旧扁平 YAML dict → 嵌套 dict 的路由（:func:`route_legacy_model_dict`）；
* 旧 Python 接口兼容（:func:`install_flat_model_compat`：转发 property、
  扁平 kwargs 构造、老 checkpoint pickle ``__setstate__`` 迁移）。

归属原则（与 TODO S1-5 / 审查 S3-2 一致）：

* COMMON —— 三 arch（unet/adm/edm2）装配或 sync 都消费的字段，留 ModelConfig 顶层；
* UNET   —— 仅 arch=='unet' 消费（含 mednext/multirf/selfattn 模块子段）；
* ADM / EDM2 —— 各自 arch 专属组，嵌套后去 ``adm_`` / ``edm2_`` 前缀。

gentask 的 ``sisr_*`` 属 gentask 子类扩展（arch∈{edsr,rcan}）：见
:data:`SISR_FIELD_MAP`，由 gentask 在安装兼容层 / YAML 路由时传入
``extra_flat_to_nested``，不并入核心 :data:`FLAT_TO_NESTED`。

注意：本模块不得 import ``taskcore.config.core``（core 反向依赖本模块）。
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Dict, Mapping, Optional, Tuple, Type

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 公共字段：留在 ModelConfig 顶层，新旧路径一致。
# 依据：adm_unet/edm2_unet builder 实读集 ∩ factory 实读集 + sync 消费
# （resenc_preset 由 sync 展开为 blocks，对三 arch 都生效）。
# ---------------------------------------------------------------------------
COMMON_FIELDS: Tuple[str, ...] = (
    "arch",
    "encoder_channels",
    "blocks_per_level",
    "encoder_blocks_per_stage",
    "decoder_blocks_per_stage",
    "resenc_preset",
    "resenc_vram_budget_gb",
    "resenc_auto_batch_size",
    "dropout",
    "stem_mode",
    "stem_fusion_mode",
    "deep_supervision",
    "aux_seg_supervision",
    "aux_head_mode",
    "grad_checkpointing",
    "grad_ckpt_stem_downsample",
    "grad_ckpt_decoder_branches",
    "init_strategy",
)

# ---------------------------------------------------------------------------
# UNet 专属：旧扁平名 → ``model.unet.`` 段内的相对路径。
# 独立旋钮字段名不变；mednext_* / multirf_* / selfattn_* 模块组去前缀入子段。
# 与 models/arch_compat 告警表的差集说明：lift_2_5d_to_3d / aux_topo_* /
# selfattn_* / multirf_* 在 validate 层被 ADM/EDM2 直接拒绝（非静默忽略），
# 故不在告警表里，但归属上仍是 unet 专属。
# ---------------------------------------------------------------------------
UNET_FIELD_MAP: Dict[str, str] = {
    "backbone":                    "backbone",
    "block_type":                  "block_type",
    "norm_type":                   "norm_type",
    "norm_groups":                 "norm_groups",
    "activation":                  "activation",
    "drop_path_rate":              "drop_path_rate",
    "decoder_type":                "decoder_type",
    "unet3p_cat_channels":         "unet3p_cat_channels",
    "downsample_mode":             "downsample_mode",
    "upsample_mode":               "upsample_mode",
    "upsample_norm_act":           "upsample_norm_act",
    "upsample_interp_dtype":       "upsample_interp_dtype",
    "anisotropic_pooling":         "anisotropic_pooling",
    "downsample_strides":          "downsample_strides",
    "skip_mode":                   "skip_mode",
    "attention_type":              "attention_type",
    "se_reduction":                "se_reduction",
    "skip_attention":              "skip_attention",
    "attn_gate_norm":              "attn_gate_norm",
    "aux_topo_head":               "aux_topo_head",
    "aux_topo_target":             "aux_topo_target",
    "aux_topo_head_mode":          "aux_topo_head_mode",
    "lift_2_5d_to_3d":             "lift_2_5d_to_3d",
    "grad_ckpt_encoder_stages":    "grad_ckpt_encoder_stages",
    "grn_enabled":                 "grn_enabled",
    "convnext_layer_scale_init":   "convnext_layer_scale_init",
    "convnext_downsample_lnfirst": "convnext_downsample_lnfirst",
    # MedNeXt 模块子段。
    "mednext_expand_ratio":                 "mednext.expand_ratio",
    "mednext_kernel_size":                  "mednext.kernel_size",
    "mednext_dilated_reparam":              "mednext.dilated_reparam",
    "mednext_dilated_reparam_kernel_sizes": "mednext.dilated_reparam_kernel_sizes",
    "mednext_dilated_reparam_dilations":    "mednext.dilated_reparam_dilations",
    # MultiRF 模块子段。
    "multirf_enabled":         "multirf.enabled",
    "multirf_dilations":       "multirf.dilations",
    "multirf_mode":            "multirf.mode",
    "multirf_fusion":          "multirf.fusion",
    "multirf_axes":            "multirf.axes",
    "multirf_encoder_stages":  "multirf.encoder_stages",
    "multirf_decoder_stages":  "multirf.decoder_stages",
    "multirf_branch_norm_act": "multirf.branch_norm_act",
    # SelfAttention 模块子段。
    "selfattn_enabled":        "selfattn.enabled",
    "selfattn_type":           "selfattn.type",
    "selfattn_num_heads":      "selfattn.num_heads",
    "selfattn_head_dim":       "selfattn.head_dim",
    "selfattn_zero_init":      "selfattn.zero_init",
    "selfattn_rope":           "selfattn.rope",
    "selfattn_ffn":            "selfattn.ffn",
    "selfattn_ffn_ratio":      "selfattn.ffn_ratio",
    "selfattn_window_size":    "selfattn.window_size",
    "selfattn_grid_size":      "selfattn.grid_size",
    "selfattn_encoder_stages": "selfattn.encoder_stages",
    "selfattn_decoder_stages": "selfattn.decoder_stages",
}

# ---------------------------------------------------------------------------
# ADM / EDM2 专属：去组前缀入嵌套段（LinearAttention 再嵌一层）。
# ---------------------------------------------------------------------------
ADM_FIELD_MAP: Dict[str, str] = {
    "adm_attention_levels":           "attention_levels",
    "adm_num_heads":                  "num_heads",
    "adm_num_head_channels":          "num_head_channels",
    "adm_linear_attention_levels":    "linear_attention.levels",
    "adm_linear_attention_num_heads": "linear_attention.num_heads",
    "adm_linear_attention_head_dim":  "linear_attention.head_dim",
}

EDM2_FIELD_MAP: Dict[str, str] = {
    "edm2_attention_levels":  "attention_levels",
    "edm2_channels_per_head": "channels_per_head",
    "edm2_res_balance":       "res_balance",
    "edm2_attn_balance":      "attn_balance",
    "edm2_concat_balance":    "concat_balance",
    "edm2_clip_act":          "clip_act",
}

# gentask 专属（arch∈{edsr,rcan}）：不并入核心 FLAT_TO_NESTED。
SISR_FIELD_MAP: Dict[str, str] = {
    "sisr_channels":   "sisr.channels",
    "sisr_num_blocks": "sisr.num_blocks",
    "sisr_num_groups": "sisr.num_groups",
    "sisr_res_scale":  "sisr.res_scale",
}

# 旧扁平名 → 相对 ``model.`` 的完整新点路径（公共字段不在此表）。
FLAT_TO_NESTED: Dict[str, str] = {
    **{k: f"unet.{v}" for k, v in UNET_FIELD_MAP.items()},
    **{k: f"adm.{v}" for k, v in ADM_FIELD_MAP.items()},
    **{k: f"edm2.{v}" for k, v in EDM2_FIELD_MAP.items()},
}


def _merged_flat_map(
    extra: "Optional[Mapping[str, str]]" = None,
) -> Dict[str, str]:
    if not extra:
        return FLAT_TO_NESTED
    return {**FLAT_TO_NESTED, **dict(extra)}


def flat_to_nested_path(flat_name: str) -> str:
    """旧扁平字段名 → 新点路径（相对 ``model.``）。

    公共字段原样返回；其余查 :data:`FLAT_TO_NESTED`。未知字段名抛
    ``KeyError``（契约测试保证全集覆盖）。
    """
    if flat_name in COMMON_FIELDS:
        return flat_name
    return FLAT_TO_NESTED[flat_name]


# ---------------------------------------------------------------------------
# 旧 YAML dict → 嵌套 dict 路由
# ---------------------------------------------------------------------------
def _copy_tree(v: Any) -> Any:
    """dict 树深拷贝（叶子原样引用），避免路由改写调用方的 raw dict。"""
    if isinstance(v, dict):
        return {k: _copy_tree(x) for k, x in v.items()}
    return v


_MISSING_SENTINEL = object()


def flatten_model_dict(
    d: Mapping[str, Any],
    *,
    extra_flat_to_nested: "Optional[Mapping[str, str]]" = None,
) -> Dict[str, Any]:
    """嵌套 model 段 dict → 旧扁平形状（:func:`route_legacy_model_dict` 的逆）。

    供仍按扁平字段名消费的平面接口（launcher 表单等）使用；公共字段与
    未知键原样保留。``extra_flat_to_nested`` 供 gentask 等任务子类扩展
    （如 :data:`SISR_FIELD_MAP`）。
    """
    mapping = _merged_flat_map(extra_flat_to_nested)
    nested_roots = {"unet", "adm", "edm2", "sisr"}
    out: Dict[str, Any] = {
        k: v for k, v in d.items() if k not in nested_roots}
    for flat, path in mapping.items():
        cursor: Any = d
        for p in path.split("."):
            if isinstance(cursor, Mapping) and p in cursor:
                cursor = cursor[p]
            else:
                cursor = _MISSING_SENTINEL
                break
        if cursor is not _MISSING_SENTINEL:
            out[flat] = cursor
    return out


def route_legacy_model_dict(
    d: Mapping[str, Any],
    *,
    error_cls: Type[Exception] = ValueError,
    extra_flat_to_nested: "Optional[Mapping[str, str]]" = None,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """把 model 段 dict 中的旧扁平键路由进嵌套子段。

    返回 ``(routed, moved)``：``routed`` 为新形状 dict（输入不被修改）；
    ``moved`` 为 ``{旧扁平键: 新点路径}``（供调用方发一条汇总迁移提示）。

    新旧路径同时设置同一字段（如 ``backbone`` 与 ``unet.backbone``）时抛
    ``error_cls``（fail-fast，不做静默优先级）。
    """
    mapping = _merged_flat_map(extra_flat_to_nested)
    routed: Dict[str, Any] = {}
    moved: Dict[str, str] = {}
    for k, v in d.items():
        if k not in mapping:
            routed[k] = _copy_tree(v)
    for k, v in d.items():
        if k not in mapping:
            continue
        path = mapping[k]
        parts = path.split(".")
        cursor = routed
        for p in parts[:-1]:
            nxt = cursor.get(p)
            if nxt is None:
                nxt = {}
                cursor[p] = nxt
            elif not isinstance(nxt, dict):
                raise error_cls(
                    f"model.{p} must be a mapping to hold migrated flat key "
                    f"'{k}'; got {type(nxt).__name__}.")
            cursor = nxt
        leaf = parts[-1]
        if leaf in cursor:
            raise error_cls(
                f"model config sets both legacy flat key '{k}' and its nested "
                f"replacement 'model.{path}'; remove the flat key.")
        cursor[leaf] = v
        moved[k] = path
    return routed, moved


# ---------------------------------------------------------------------------
# Python 接口兼容：转发 property / 扁平 kwargs 构造 / 老 pickle 迁移
# ---------------------------------------------------------------------------
def _forwarding_property(flat_name: str, path: str) -> property:
    parts = tuple(path.split("."))

    def fget(self):
        obj = self
        for p in parts[:-1]:
            obj = getattr(obj, p)
        return getattr(obj, parts[-1])

    def fset(self, value):
        obj = self
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], value)

    fget.__name__ = flat_name
    fset.__name__ = flat_name
    return property(
        fget, fset,
        doc=f"[compat] 旧扁平字段，转发到 model.{path}（读写等价）。")


def _ensure_model_geometry_backing(obj) -> None:
    """补齐 ``_spatial_dims`` / ``_in_channels`` 私有 backing（只读 property 依赖）。

    老 checkpoint pickle 绕过 ``__init__``/``__post_init__``，可能只带扁平
    ``spatial_dims``/``in_channels`` 或二者皆无；property 是数据描述符，读
    ``obj.spatial_dims`` 会走 ``_spatial_dims``，缺失即 ``AttributeError``。
    """
    if "_spatial_dims" not in obj.__dict__:
        flat = obj.__dict__.pop("spatial_dims", 3)
        obj.__dict__["_spatial_dims"] = int(flat) if flat is not None else 3
    else:
        obj.__dict__.pop("spatial_dims", None)
    if "_in_channels" not in obj.__dict__:
        flat = obj.__dict__.pop("in_channels", 1)
        obj.__dict__["_in_channels"] = int(flat) if flat is not None else 1
    else:
        obj.__dict__.pop("in_channels", None)


def _model_setstate_factory(mapping: Dict[str, str], nested_sections: Tuple[str, ...]):
    def _model_setstate(self, state: Dict[str, Any]) -> None:
        """老 checkpoint pickle 迁移：旧版扁平 ``ModelConfig`` 状态自动路由。"""
        if "unet" in state:  # 新版 pickle：嵌套段就绪，原样恢复。
            self.__dict__.update(state)
            _ensure_model_geometry_backing(self)
            return
        legacy = {k: v for k, v in state.items() if k in mapping}
        for k, v in state.items():
            if k not in mapping:
                self.__dict__[k] = v
        for f in type(self).__dataclass_fields__.values():  # type: ignore[attr-defined]
            if f.name not in self.__dict__ and f.name in nested_sections:
                self.__dict__[f.name] = f.default_factory()  # type: ignore[misc]
        for k, v in legacy.items():
            setattr(self, k, v)
        _ensure_model_geometry_backing(self)
        if legacy:
            logger.info(
                "Migrated %d legacy flat ModelConfig fields from old checkpoint "
                "pickle into nested sections.", len(legacy))
    return _model_setstate


def _patch_flat_kwargs_init(cls, mapping: Dict[str, str]) -> None:
    """让 ``ModelConfig(backbone=..., adm_num_heads=...)`` 等旧构造继续可用。"""
    orig_init = cls.__init__
    if getattr(orig_init, "_flat_compat", False):
        return

    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs):
        legacy = {
            k: kwargs.pop(k) for k in list(kwargs) if k in mapping}
        orig_init(self, *args, **kwargs)
        for k, v in legacy.items():
            setattr(self, k, v)

    __init__._flat_compat = True  # type: ignore[attr-defined]
    cls.__init__ = __init__


def install_flat_model_compat(
    cls,
    *,
    extra_flat_to_nested: "Optional[Mapping[str, str]]" = None,
    nested_sections: Tuple[str, ...] = ("unet", "adm", "edm2"),
):
    """在（core 或任务子类）ModelConfig 上安装全部旧扁平接口兼容层。

    * 转发 property：``mc.backbone`` ↔ ``mc.unet.backbone`` 读写等价；
    * 扁平 kwargs 构造：``ModelConfig(backbone=...)``；
    * ``__setstate__``：老 checkpoint pickle 状态自动迁移。

    ``extra_flat_to_nested``：任务子类扩展映射（gentask 传
    :data:`SISR_FIELD_MAP`）；``nested_sections``：``__setstate__`` 补齐的
    嵌套段名（gentask 另加 ``sisr``）。

    幂等；可用作类装饰器。
    """
    mapping = _merged_flat_map(extra_flat_to_nested)
    for flat, path in mapping.items():
        if not isinstance(cls.__dict__.get(flat), property):
            setattr(cls, flat, _forwarding_property(flat, path))
    _patch_flat_kwargs_init(cls, mapping)
    cls.__setstate__ = _model_setstate_factory(mapping, nested_sections)
    return cls


__all__ = [
    "COMMON_FIELDS",
    "UNET_FIELD_MAP",
    "ADM_FIELD_MAP",
    "EDM2_FIELD_MAP",
    "SISR_FIELD_MAP",
    "FLAT_TO_NESTED",
    "flat_to_nested_path",
    "flatten_model_dict",
    "route_legacy_model_dict",
    "install_flat_model_compat",
]
