"""clstask 配置：复用 ``segtask_v1.config.Config`` + 独立 ``ClsConfig``。

设计（与 ssltask 同构）：**复用 ``segtask_v1.config.Config``** 作为
data/model/train/... 的载体（几何派生：``patch_mode`` × ``build_topology`` →
``spatial_dims`` / ``in_channels``，与下游分割/SSL 逐位一致，保证 SSL 预训练
encoder 权重可无缝迁移）；分类专属设置集中在 YAML 顶层 ``cls:`` 段：

* ``cfg`` —— ``segtask_v1.config.Config``，喂给数据管线 / encoder 工厂 / 优化器；
* ``cls`` —— :class:`ClsConfig`，仅分类任务读它。

几何支持（与分割同一套 patch_mode 语义）：

* 3D  —— ``patch_mode ∈ {whole, z_axis, cubic}``（spatial_dims=3，输入 (B,1,D,H,W)）；
* 2.5D —— ``patch_mode == '2_5d'``（spatial_dims=2，slab 深度 D 折进通道，
  输入 (B,D,H,W)）。当前仅支持单 FOV（``multi_res_scales == [1.0]``），与
  image-only SSL 预训练口径一致；多 FOV 折叠为后续扩展点。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import yaml

from segtask_v1.config import (
    Config as SegConfig,
    ConfigError,
    _dataclass_from_dict,
)

logger = logging.getLogger(__name__)

#: 支持的 backbone。'encoder' 复用 segtask Encoder（ResNet / ConvNeXt 由
#: ``cfg.model.backbone`` 决定），SSL 预训练权重可直接迁移；densenet / vit 为
#: clstask 自有实现（2D/3D 通用），无法吃 CNN Encoder 的 SSL 权重。
BACKBONES = ("encoder", "densenet", "vit")
GRANULARITIES = ("volume", "slice")
LABEL_SOURCES = ("mask", "table")
LOSS_TYPES = ("bce", "ce", "focal")
POOLINGS = ("avg", "max", "avgmax")
AGG_MODES = ("mean", "max", "lse", "topk")


@dataclass
class ClsConfig:
    """分类任务专属设置（YAML 顶层 ``cls:`` 段）。

    与训练/数据基建有关的字段（batch/lr/epochs/patch_mode/...）一律沿用
    ``segtask_v1.config.Config``（``cfg.model`` / ``cfg.train`` / ``cfg.data``）；
    此处只放分类语义相关字段。
    """

    # ---- 标签 ---------------------------------------------------------
    # 标签粒度：'volume' 对整个样本/patch 分类；'slice' 对每个 z 切片分类
    # （输出 (B, K, D)，D = patch_size[0]）。
    label_granularity: str = "volume"
    # 标签来源：
    #   'mask'  —— 由分割 mask 派生弱标签：每前景类"是否出现"（volume 粒度对
    #              patch any()；slice 粒度对每切片 any()）。类数 = num_fg_classes。
    #   'table' —— 显式标签表（csv/json，pid → 标签），仅 volume 粒度；
    #              patch 继承其所在卷的标签。
    label_source: str = "mask"
    # 标签表路径（label_source='table' 必填）。
    #   csv：表头 ``pid,label``（单标签整型类别）或 ``pid,c1,...,cK``（多热）。
    #   json：``{pid: int}`` 或 ``{pid: [0/1, ...]}``。
    label_table: str = ""
    # True = 多标签（一 vs 其余 sigmoid + BCE/focal）；False = 单标签 softmax CE
    # （仅 label_source='table' + volume 粒度支持）。
    multi_label: bool = True
    # 类数。0 = 自动：mask 源 = cfg.num_fg_classes；table 源须显式给出。
    num_classes: int = 0

    # ---- Backbone -----------------------------------------------------
    # 'encoder'（复用 segtask Encoder；ResNet/ConvNeXt 由 cfg.model.backbone 决定）
    # | 'densenet' | 'vit'。四模板 = resnet / convnext（经 encoder）+ densenet + vit。
    backbone: str = "encoder"

    # DenseNet-BC（backbone='densenet'）。
    densenet_growth_rate : int = 16
    densenet_block_layers: List[int] = field(default_factory=lambda: [4, 8, 12, 8])
    densenet_compression : float = 0.5
    densenet_stem_channels: int = 32

    # ViT（backbone='vit'）。
    vit_embed_dim     : int = 384
    vit_depth         : int = 8
    vit_num_heads     : int = 6
    vit_mlp_ratio     : float = 4.0
    vit_drop_path_rate: float = 0.1
    # token patch 大小 [pd, ph, pw]；2.5D（spatial_dims=2）只用 [ph, pw]。
    vit_patch_size    : List[int] = field(default_factory=lambda: [4, 16, 16])

    # ---- 分类头 -------------------------------------------------------
    head_hidden_dim: int = 256    # 0 = 单层 linear 头
    head_dropout   : float = 0.0
    pooling        : str = "avg"  # 'avg' | 'max' | 'avgmax'（cat 后过头）

    # ---- 损失 ---------------------------------------------------------
    loss_type      : str = "bce"  # 'bce' | 'ce' | 'focal'
    label_smoothing: float = 0.0
    focal_gamma    : float = 2.0
    focal_alpha    : float = -1.0  # <0 = 不用 alpha 平衡
    # 逐类权重（BCE 的 pos_weight / CE 的 class weight）；空 = 均匀。
    class_weights  : List[float] = field(default_factory=list)

    # ---- Mixup / CutMix（仅 volume 粒度）--------------------------------
    mixup_alpha : float = 0.0   # >0 启用
    cutmix_alpha: float = 0.0   # >0 启用；与 mixup 同时启用时每 batch 二选一
    mixup_prob  : float = 0.5   # 每 batch 应用增强的概率

    # ---- SSL / 预训练迁移 ----------------------------------------------
    # SSL / 分割 checkpoint 路径；只取 ``encoder.*`` 权重，strict=False 加载
    # 并打印命中/缺失统计。仅 backbone='encoder' 支持。
    pretrained_ckpt : str = ""
    freeze_encoder  : bool = False   # True = 只训头（linear probe）
    encoder_lr_mult : float = 1.0    # encoder 参数组学习率倍率（<1 = 微调更小 lr）

    # ---- 推理聚合（patch → volume 的 MIL 聚合）--------------------------
    agg_mode : str = "mean"   # 'mean' | 'max' | 'lse' | 'topk'
    agg_topk : int = 3        # 仅 'topk'
    agg_lse_r: float = 4.0    # 仅 'lse'（log-sum-exp 温度）

    # ---- 验证/推理抽样 --------------------------------------------------
    # 每卷 patch 数：验证集每卷采样数 = 该值（选模与部署同口径），推理
    # 铺格总数上限 = 该值（沿 z / 三轴均匀网格）。
    eval_patches_per_volume: int = 8
    # 整卷推理每次前向的 patch 数（micro-batch，防大卷 OOM）。
    infer_batch_size: int = 16
    # 推理翻转 TTA：3D 7 种轴组合翻转，2.5D 仅翻 H/W；各变体概率取平均。
    tta_flips: bool = False

    # ---- 选模 -----------------------------------------------------------
    # patch 级：'auc' | 'f1' | 'acc' | 'loss'；卷级 MIL（与推理 agg_mode 同
    # 口径）：'vol_auc' | 'vol_f1' | 'vol_acc'。
    save_best_metric: str = "auc"


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def validate_cls(cls: ClsConfig, cfg: SegConfig) -> None:
    """交叉校验 ``cls`` 与 ``cfg``（几何/标签/损失一致性）。"""
    _require(cls.label_granularity in GRANULARITIES,
             f"cls.label_granularity must be one of {GRANULARITIES}; "
             f"got {cls.label_granularity!r}.")
    _require(cls.label_source in LABEL_SOURCES,
             f"cls.label_source must be one of {LABEL_SOURCES}; "
             f"got {cls.label_source!r}.")
    _require(cls.backbone in BACKBONES,
             f"cls.backbone must be one of {BACKBONES}; got {cls.backbone!r}.")
    _require(cls.loss_type in LOSS_TYPES,
             f"cls.loss_type must be one of {LOSS_TYPES}; got {cls.loss_type!r}.")
    _require(cls.pooling in POOLINGS,
             f"cls.pooling must be one of {POOLINGS}; got {cls.pooling!r}.")
    _require(cls.agg_mode in AGG_MODES,
             f"cls.agg_mode must be one of {AGG_MODES}; got {cls.agg_mode!r}.")
    _require(cls.save_best_metric in ("auc", "f1", "acc", "loss",
                                      "vol_auc", "vol_f1", "vol_acc"),
             f"cls.save_best_metric must be auc|f1|acc|loss|vol_auc|vol_f1"
             f"|vol_acc; got {cls.save_best_metric!r}.")

    # ---- 几何 -----------------------------------------------------------
    pm = str(cfg.data.patch_mode).lower()
    _require(pm in ("whole", "z_axis", "cubic", "2_5d"),
             f"clstask supports patch_mode whole|z_axis|cubic|2_5d; got {pm!r}.")
    if pm == "2_5d":
        scales = list(cfg.data.multi_res_scales) or [1.0]
        _require(len(scales) == 1 and float(scales[0]) == 1.0,
                 "clstask 2.5D currently supports single-FOV only "
                 "(data.multi_res_scales == [1.0]), matching the image-only "
                 f"SSL pretraining geometry; got {scales}.")
        _require(not bool(cfg.model.lift_2_5d_to_3d),
                 "cls does not support model.lift_2_5d_to_3d yet; use plain "
                 "2.5D folding (spatial_dims=2) or a 3D patch_mode.")
    else:
        scales = list(cfg.data.multi_res_scales) or [1.0]
        _require(len(scales) == 1 and float(scales[0]) == 1.0,
                 "clstask currently supports single-resolution input "
                 f"(data.multi_res_scales == [1.0]); got {scales}.")

    # ---- 标签 -----------------------------------------------------------
    if cls.label_source == "table":
        _require(bool(cls.label_table),
                 "cls.label_source='table' requires cls.label_table (csv/json "
                 "path).")
        _require(cls.label_granularity == "volume",
                 "cls.label_source='table' supports label_granularity='volume' "
                 "only (per-slice tables are not supported yet; use "
                 "label_source='mask' for slice-level weak labels).")
        _require(cls.num_classes >= 1,
                 "cls.label_source='table' requires explicit cls.num_classes "
                 ">= 1 (cannot be derived from masks).")
    if not cls.multi_label:
        _require(cls.label_source == "table"
                 and cls.label_granularity == "volume",
                 "single-label mode (cls.multi_label=false) requires "
                 "label_source='table' + label_granularity='volume'.")
        _require(cls.loss_type == "ce",
                 "single-label mode requires cls.loss_type='ce'.")
        _require(cls.num_classes >= 2,
                 f"single-label CE needs cls.num_classes >= 2; got "
                 f"{cls.num_classes}.")
    else:
        _require(cls.loss_type in ("bce", "focal"),
                 "multi-label mode requires cls.loss_type in ('bce', 'focal'); "
                 "'ce' needs cls.multi_label=false.")

    if cls.class_weights:
        k = resolve_num_classes(cls, cfg)
        _require(len(cls.class_weights) == k,
                 f"cls.class_weights length ({len(cls.class_weights)}) must "
                 f"equal num_classes ({k}).")

    # ---- Mixup ----------------------------------------------------------
    if cls.mixup_alpha > 0 or cls.cutmix_alpha > 0:
        _require(cls.label_granularity == "volume",
                 "mixup/cutmix are supported for label_granularity='volume' "
                 "only.")
        _require(cls.multi_label or cls.loss_type == "ce",
                 "mixup/cutmix require soft-target-capable loss (bce/focal/ce).")

    # ---- 迁移 -----------------------------------------------------------
    if cls.pretrained_ckpt:
        _require(cls.backbone == "encoder",
                 "cls.pretrained_ckpt (SSL/seg encoder transfer) requires "
                 f"cls.backbone='encoder'; got {cls.backbone!r}. DenseNet/ViT "
                 "have different parameterizations and cannot load CNN "
                 "encoder weights.")
    _require(cls.encoder_lr_mult > 0,
             f"cls.encoder_lr_mult must be > 0; got {cls.encoder_lr_mult}.")

    # ---- ViT ------------------------------------------------------------
    if cls.backbone == "vit":
        _require(len(cls.vit_patch_size) == 3,
                 f"cls.vit_patch_size must be [pd, ph, pw]; got "
                 f"{cls.vit_patch_size}.")
        sd = int(cfg.model.spatial_dims)
        patch = [int(x) for x in cfg.data.patch_size]  # [D, H, W]
        tok = [int(x) for x in cls.vit_patch_size]
        dims = patch if sd == 3 else patch[1:]
        toks = tok if sd == 3 else tok[1:]
        for size, t in zip(dims, toks):
            _require(size % t == 0,
                     f"patch_size {patch} must be divisible by vit_patch_size "
                     f"{tok} on spatial axes (spatial_dims={sd}).")
        _require(cls.vit_embed_dim % cls.vit_num_heads == 0,
                 f"cls.vit_embed_dim ({cls.vit_embed_dim}) must be divisible "
                 f"by cls.vit_num_heads ({cls.vit_num_heads}).")

    # ---- 选模方向 × plateau scheduler --------------------------------------
    # plateau 的 mode 取 cfg.train.save_best_mode（seg 的单一真相源）；分类的
    # 选模指标方向由 cls.save_best_metric 决定（loss → min，其余 → max），
    # 二者不一致时 plateau 会在指标变好时误降 lr，这里显式拦截。
    if str(cfg.train.scheduler).lower() == "plateau":
        cls_mode = "min" if cls.save_best_metric == "loss" else "max"
        seg_mode = str(cfg.train.save_best_mode)
        _require(seg_mode == cls_mode,
                 f"train.scheduler='plateau' derives its direction from "
                 f"train.save_best_criterion (currently mode={seg_mode!r}), "
                 f"which conflicts with cls.save_best_metric="
                 f"{cls.save_best_metric!r} (mode={cls_mode!r}). Set "
                 f"train.save_best_criterion so its mode matches (e.g. "
                 f"'loss' for min / 'dice' for max).")

    _require(cls.eval_patches_per_volume >= 1,
             f"cls.eval_patches_per_volume must be >= 1; got "
             f"{cls.eval_patches_per_volume}.")
    _require(cls.infer_batch_size >= 1,
             f"cls.infer_batch_size must be >= 1; got {cls.infer_batch_size}.")
    _require(cls.agg_topk >= 1, f"cls.agg_topk must be >= 1; got {cls.agg_topk}.")


def resolve_num_classes(cls: ClsConfig, cfg: SegConfig) -> int:
    """有效类数：显式 ``cls.num_classes`` 优先，否则 mask 源 = num_fg_classes。"""
    if cls.num_classes >= 1:
        return int(cls.num_classes)
    return int(cfg.num_fg_classes)


# ---------------------------------------------------------------------------
# YAML I/O + overrides（与 ssltask.config 同构）
# ---------------------------------------------------------------------------
def load_config(path: Union[str, Path]) -> Tuple[SegConfig, ClsConfig]:
    """加载 clstask YAML，返回 ``(cfg, cls)``。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cls_raw = dict(raw.pop("cls", {}) or {})
    cfg = _dataclass_from_dict(SegConfig, raw)
    cfg.sync()
    cfg.validate()
    cls = _dataclass_from_dict(ClsConfig, cls_raw)
    validate_cls(cls, cfg)
    return cfg, cls


def save_config(cfg: SegConfig, cls: ClsConfig, path: Union[str, Path]) -> None:
    """把 ``(cfg, cls)`` 落盘为单个 YAML。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = asdict(cfg)
    blob["cls"] = asdict(cls)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(blob, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)


def _coerce(old: Any, val: str) -> Any:
    if isinstance(old, bool):
        return val.lower() in ("true", "1", "yes")
    if isinstance(old, int):
        return int(val)
    if isinstance(old, float):
        return float(val)
    if isinstance(old, list):
        return json.loads(val)
    return val


def _set_dotted(obj: Any, dotted: str, val: str) -> None:
    parts = dotted.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    attr = parts[-1]
    old = getattr(obj, attr)
    new = _coerce(old, val)
    setattr(obj, attr, new)
    logger.info("Override: %s = %s -> %s", dotted, old, new)


def apply_overrides(cfg: SegConfig, cls: ClsConfig,
                    overrides: List[str]) -> None:
    """点记法 override；``cls.*`` 路由到 ``cls``，其余路由到 ``cfg``。

    调用方应在其后自行 ``cfg.sync(); cfg.validate(); validate_cls(cls, cfg)``。
    """
    for ov in overrides:
        if "=" not in ov:
            continue
        key, val = ov.split("=", 1)
        if key.startswith("cls."):
            _set_dotted(cls, key[len("cls."):], val)
        else:
            _set_dotted(cfg, key, val)


__all__ = [
    "ClsConfig", "SegConfig", "ConfigError",
    "BACKBONES", "GRANULARITIES", "LABEL_SOURCES",
    "validate_cls", "resolve_num_classes",
    "load_config", "save_config", "apply_overrides",
]
