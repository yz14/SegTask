"""dettask 配置：复用 ``segtask_v1.config.Config`` + 独立 ``DetConfig``。

与 clstask 同构：``cfg``（segtask Config）承载 data/model/train 基建与几何派生
（``patch_mode`` × ``build_topology`` → ``spatial_dims`` / ``in_channels``），
检测专属设置集中在 YAML 顶层 ``det:`` 段。

双几何形态（Plan §3.3）：

* 3D（``patch_mode ∈ {whole, z_axis, cubic}``）—— 3D 框
  ``[z1,y1,x1,z2,y2,x2]``，3D 卷积头 + 自实现 3D NMS / ROIAlign；
* 2.5D（``patch_mode == '2_5d'``）—— slab 折叠为 2D，slab 内 2D 框
  ``[y1,x1,y2,x2]``；推理端跨 slab 按 IoU 链接拼接 3D 框。

框真值只存 3D 一份（npz ``boxes`` 键 (N, 7) = [z1,y1,x1,z2,y2,x2,cls]，或由
分割 mask 连通域自动派生）；2.5D 由 3D 框对 slab 切片自动派生。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Tuple, Union

import yaml

from segtask_v1.config import (
    Config as SegConfig,
    ConfigError,
    _dataclass_from_dict,
)

logger = logging.getLogger(__name__)

#: 四模板（Plan §3.5）。
ARCHS = ("retinanet", "fcos", "faster_rcnn", "detr")
ASSIGNERS = ("iou", "atss")
REG_LOSSES = ("giou", "l1", "smooth_l1")


@dataclass
class DetConfig:
    """检测任务专属设置（YAML 顶层 ``det:`` 段）。"""

    # ---- 模板 -----------------------------------------------------------
    arch: str = "retinanet"   # 'retinanet' | 'fcos' | 'faster_rcnn' | 'detr'

    # ---- 标签 -----------------------------------------------------------
    # 类数。0 = 自动（mask 源 = cfg.num_fg_classes）。
    num_classes: int = 0
    # 框来源：npz 'boxes' 键优先；否则由 mask 连通域派生（fg_values 逐类）。
    boxes_from_mask: bool = True
    # mask 派生连通域的最小体素数（滤噪点）。
    min_box_voxels: int = 8

    # ---- FPN -------------------------------------------------------------
    fpn_channels: int = 128
    # 用 decoder 金字塔的哪几层（0 = 最低分辨率）。空 = 全部层。
    fpn_levels: List[int] = field(default_factory=list)

    # ---- Anchor（retinanet / faster_rcnn）--------------------------------
    # 逐层基础尺寸（yx 向，体素）；长度须与使用的 FPN 层数一致，空 = 自动
    # （按各层步长 × 4）。
    anchor_sizes : List[float] = field(default_factory=list)
    anchor_ratios: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    anchor_scales: List[float] = field(default_factory=lambda: [1.0, 1.26, 1.587])
    # 3D 附加：z 向尺寸 = 基础尺寸 × z_scale。
    anchor_z_scales: List[float] = field(default_factory=lambda: [0.5])

    # ---- 分配器 -----------------------------------------------------------
    assigner : str = "iou"    # 'iou' | 'atss'
    pos_iou  : float = 0.4    # iou 分配器正样本阈值（医学小目标偏低）
    neg_iou  : float = 0.3
    atss_topk: int = 9

    # ---- 损失 --------------------------------------------------------------
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    reg_loss   : str = "giou"     # 'giou' | 'l1' | 'smooth_l1'
    reg_weight : float = 2.0
    # DETR 集合损失权重（cls / L1 / GIoU）与匹配代价同权。
    detr_cls_weight : float = 2.0
    detr_l1_weight  : float = 5.0
    detr_giou_weight: float = 2.0

    # ---- Faster R-CNN ------------------------------------------------------
    rpn_pre_nms_topk : int = 1000
    rpn_post_nms_topk: int = 256
    rpn_nms_iou      : float = 0.7
    rpn_pos_iou      : float = 0.5
    rpn_neg_iou      : float = 0.3
    rpn_batch_per_img: int = 64     # RPN 采样 anchor 数
    roi_batch_per_img: int = 32     # R-CNN 头采样 proposal 数
    roi_pos_fraction : float = 0.5
    roi_output_size  : int = 5      # ROIAlign 每轴输出网格

    # ---- DETR ---------------------------------------------------------------
    num_queries     : int = 64
    detr_hidden_dim : int = 128
    detr_num_heads  : int = 8
    detr_dec_layers : int = 3
    detr_num_points : int = 4       # 可变形注意力每 query 采样点数

    # ---- 推理 / NMS ----------------------------------------------------------
    score_thresh: float = 0.05
    nms_iou     : float = 0.3
    max_dets    : int = 50          # 每 patch/slab 检出上限
    infer_batch_size: int = 4       # 整卷滑窗推理每次前向的 patch/slab 数

    # ---- 2.5D 跨层拼接（Plan §3.3 stitching）----------------------------------
    stitch_link_iou: float = 0.3    # 相邻 slab 2D 框链接的最小 IoU
    stitch_min_span: int = 2        # 3D 框最少跨的 slab 数

    # ---- SSL / 预训练迁移 -------------------------------------------------------
    # 只取 encoder.*（重建式 SSL 亦可命中 decoder.*），strict=False。
    pretrained_ckpt: str = ""
    freeze_encoder : bool = False
    encoder_lr_mult: float = 1.0

    # ---- 评估 ---------------------------------------------------------------
    eval_iou_thresh: float = 0.1    # mAP / FROC 命中 IoU（医学小目标口径）
    froc_fp_per_vol: List[float] = field(
        default_factory=lambda: [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    save_best_metric: str = "map"   # 'map' | 'loss'


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def validate_det(det: DetConfig, cfg: SegConfig) -> None:
    """交叉校验 ``det`` 与 ``cfg``。"""
    _require(det.arch in ARCHS,
             f"det.arch must be one of {ARCHS}; got {det.arch!r}.")
    _require(det.assigner in ASSIGNERS,
             f"det.assigner must be one of {ASSIGNERS}; got {det.assigner!r}.")
    _require(det.reg_loss in REG_LOSSES,
             f"det.reg_loss must be one of {REG_LOSSES}; got {det.reg_loss!r}.")
    _require(det.save_best_metric in ("map", "loss"),
             f"det.save_best_metric must be map|loss; "
             f"got {det.save_best_metric!r}.")

    pm = str(cfg.data.patch_mode).lower()
    _require(pm in ("whole", "z_axis", "cubic", "2_5d"),
             f"dettask supports patch_mode whole|z_axis|cubic|2_5d; got {pm!r}.")
    scales = list(cfg.data.multi_res_scales) or [1.0]
    _require(len(scales) == 1 and float(scales[0]) == 1.0,
             "dettask currently supports single-FOV input "
             f"(data.multi_res_scales == [1.0]); got {scales}.")
    if pm == "2_5d":
        _require(not bool(cfg.model.lift_2_5d_to_3d),
                 "dettask 2.5D uses plain slab folding (2D boxes + stitching); "
                 "model.lift_2_5d_to_3d is not supported.")

    _require(0 <= det.neg_iou <= det.pos_iou <= 1,
             f"need 0 <= det.neg_iou <= det.pos_iou <= 1; got "
             f"neg={det.neg_iou}, pos={det.pos_iou}.")
    _require(det.detr_hidden_dim % det.detr_num_heads == 0,
             f"det.detr_hidden_dim ({det.detr_hidden_dim}) must be divisible "
             f"by det.detr_num_heads ({det.detr_num_heads}).")
    _require(det.stitch_min_span >= 1,
             f"det.stitch_min_span must be >= 1; got {det.stitch_min_span}.")
    _require(det.infer_batch_size >= 1,
             f"det.infer_batch_size must be >= 1; got {det.infer_batch_size}.")
    _require(det.fpn_channels >= 8,
             f"det.fpn_channels must be >= 8; got {det.fpn_channels}.")
    for name in ("anchor_ratios", "anchor_scales", "anchor_z_scales",
                 "froc_fp_per_vol"):
        _require(len(getattr(det, name)) >= 1, f"det.{name} must be non-empty.")
    if det.pretrained_ckpt:
        _require(det.encoder_lr_mult > 0,
                 f"det.encoder_lr_mult must be > 0; got {det.encoder_lr_mult}.")


def resolve_num_classes(det: DetConfig, cfg: SegConfig) -> int:
    if det.num_classes >= 1:
        return int(det.num_classes)
    return int(cfg.num_fg_classes)


# ---------------------------------------------------------------------------
# YAML I/O + overrides（与 clstask.config 同构）
# ---------------------------------------------------------------------------
def load_config(path: Union[str, Path]) -> Tuple[SegConfig, DetConfig]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    det_raw = dict(raw.pop("det", {}) or {})
    cfg = _dataclass_from_dict(SegConfig, raw)
    cfg.sync()
    cfg.validate()
    det = _dataclass_from_dict(DetConfig, det_raw)
    validate_det(det, cfg)
    return cfg, det


def save_config(cfg: SegConfig, det: DetConfig, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = asdict(cfg)
    blob["det"] = asdict(det)
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


def apply_overrides(cfg: SegConfig, det: DetConfig,
                    overrides: List[str]) -> None:
    """点记法 override；``det.*`` 路由到 ``det``，其余路由到 ``cfg``。"""
    for ov in overrides:
        if "=" not in ov:
            continue
        key, val = ov.split("=", 1)
        if key.startswith("det."):
            _set_dotted(det, key[len("det."):], val)
        else:
            _set_dotted(cfg, key, val)


__all__ = [
    "DetConfig", "SegConfig", "ConfigError", "ARCHS",
    "validate_det", "resolve_num_classes",
    "load_config", "save_config", "apply_overrides",
]
