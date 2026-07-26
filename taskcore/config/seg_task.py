"""Seg 专属配置段（P2a）：loss / predict 从 core Config 下沉。

组合式任务（cls / det / ssl）的 core Config 不再携带这两段；
分割经 :class:`SegTaskConfig` + :func:`validate_seg_task` 装配。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Union

from .core import ConfigError, LossConfig, PredictConfig, _require

if TYPE_CHECKING:
    from .core import Config

logger = logging.getLogger(__name__)


@dataclass
class SegTaskConfig:
    """分割任务专属段（YAML 顶层 ``seg:`` 或旧式顶层 ``loss``/``predict``）。"""

    loss: LossConfig = field(default_factory=LossConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)

    def validate(self, core: "Config") -> None:
        """校验 loss/predict 及依赖 core 几何/模型的交叉约束。"""
        self._validate_loss(core)
        self._validate_predict(core)
        self._validate_cross(core)

    def _validate_loss(self, core: "Config") -> None:
        loss = self.loss
        _require(
            loss.name in (
                "dice", "bce", "focal", "tversky",
                "gdl", "focal_tversky", "lovasz", "cldice",
                "dice_bce", "dice_focal", "dice_tversky",
                "focal_plus_tversky", "dice_cldice", "dice_focal_tversky",
                "dice_lovasz", "bce_lovasz",
                "gdl_bce", "gdl_focal",
            ),
            f"Invalid loss: {loss.name}")
        _require(
            loss.gdl_weight_type in ("square", "simple", "uniform"),
            f"Invalid gdl_weight_type: {loss.gdl_weight_type}")
        _require(
            loss.focal_tversky_gamma > 0,
            f"focal_tversky_gamma must be > 0, got {loss.focal_tversky_gamma}")
        _require(
            loss.cldice_iter >= 1,
            f"cldice_iter must be >= 1, got {loss.cldice_iter}")
        _require(
            loss.slice_loss_reduction in ("per_slice", "per_volume"),
            f"Invalid slice_loss_reduction: {loss.slice_loss_reduction!r}; "
            "expected 'per_slice' or 'per_volume'.")
        # region_weights 与 label_values 逐值对应（含 bg）；长度不符时
        # 运行期 zip 会静默截断，这里 fail-fast。
        if loss.region_weights and core.data.label_values:
            _require(
                len(loss.region_weights) == len(core.data.label_values),
                f"loss.region_weights must have one entry per label value "
                f"(incl. background): got {len(loss.region_weights)} "
                f"weights for {len(core.data.label_values)} label values.")
        if core.model.deep_supervision:
            _require(
                bool(loss.deep_supervision_weights),
                "model.deep_supervision=True requires non-empty "
                "loss.deep_supervision_weights：否则 pipeline 不会包装 "
                "DeepSupervisionLoss，而模型 forward 返回 list，首个训练 "
                "step 才会报错。")
        if core.model.deep_supervision and loss.deep_supervision_weights:
            ds_w = loss.deep_supervision_weights
            expected = len(core.model.encoder_channels) - 1
            if len(ds_w) != expected:
                logger.warning(
                    "loss.deep_supervision_weights 长度 %d 与深监督预测数 %d "
                    "(= len(model.encoder_channels) - 1，main + DS 头) 不符；"
                    "首个训练 step 将由 DeepSupervisionLoss 报错。weights=%s。",
                    len(ds_w), expected, ds_w)
            _require(
                all(float(w) >= 0.0 for w in ds_w) and sum(ds_w) > 0,
                f"loss.deep_supervision_weights must be non-negative with a "
                f"positive sum; got {ds_w}.")

    def _validate_predict(self, core: "Config") -> None:
        pred = self.predict
        thr_cfg = pred.threshold
        if isinstance(thr_cfg, (list, tuple)):
            _require(len(thr_cfg) > 0, "predict.threshold list must be non-empty.")
            _require(
                all(0.0 <= float(t) <= 1.0 for t in thr_cfg),
                f"predict.threshold entries must be in [0,1]; got {thr_cfg}.")
        else:
            _require(
                0.0 <= float(thr_cfg) <= 1.0,
                f"predict.threshold must be in [0,1]; got {thr_cfg}.")
        _require(
            0.0 <= float(pred.z_overlap) < 1.0,
            f"predict.z_overlap must be in [0, 1); got {pred.z_overlap}.")
        if pred.hw_overlap is not None:
            _require(
                0.0 <= float(pred.hw_overlap) < 1.0,
                f"predict.hw_overlap must be in [0, 1); "
                f"got {pred.hw_overlap}.")
            if core.data.patch_mode != "cubic":
                logger.warning(
                    "predict.hw_overlap is only used by patch_mode='cubic'; "
                    "current patch_mode=%r ignores it.", core.data.patch_mode)
        _require(
            pred.blend_mode in ("gaussian", "average"),
            f"predict.blend_mode must be 'gaussian' or 'average'; "
            f"got {pred.blend_mode!r}.")
        _require(
            str(pred.acc_dtype) in ("fp32", "fp16"),
            f"predict.acc_dtype must be 'fp32' or 'fp16'; "
            f"got {pred.acc_dtype!r}.")
        _require(
            str(pred.vol_dtype) in ("fp32", "fp16"),
            f"predict.vol_dtype must be 'fp32' or 'fp16'; "
            f"got {pred.vol_dtype!r}.")
        if pred.z_interleave_enabled:
            _require(
                core.data.patch_mode == "2_5d",
                f"predict.z_interleave_enabled=True requires patch_mode='2_5d'; "
                f"got {core.data.patch_mode!r}.")
            thr = pred.z_interleave_thresholds
            fac = pred.z_interleave_factors
            _require(
                len(fac) == len(thr) + 1,
                f"z_interleave_factors length must = len(thresholds)+1; "
                f"got thresholds={thr}, factors={fac}.")
            _require(
                all(t > 0 for t in thr),
                f"z_interleave_thresholds must all > 0; got {thr}.")
            _require(
                thr == sorted(thr),
                f"z_interleave_thresholds must be ascending; got {thr}.")
            _require(
                all(int(f) >= 1 for f in fac),
                f"z_interleave_factors must all >= 1; got {fac}.")
            if core.data.z_boundary_mode != "edge_pad":
                logger.warning(
                    "z_interleave_enabled=True with z_boundary_mode=%r: "
                    "short sub-streams will be stretched along z. Prefer 'edge_pad'.",
                    core.data.z_boundary_mode)
        if pred.adabn_enabled:
            _require(
                pred.adabn_mode in ("global", "per_volume"),
                f"predict.adabn_mode must be 'global' or 'per_volume'; "
                f"got {pred.adabn_mode!r}.")
            _require(
                int(pred.adabn_num_volumes) >= 1,
                f"predict.adabn_num_volumes must be >= 1; "
                f"got {pred.adabn_num_volumes}.")
            _require(
                0.0 < float(pred.adabn_sample_ratio) <= 1.0,
                f"predict.adabn_sample_ratio must be in (0, 1]; "
                f"got {pred.adabn_sample_ratio}.")
            if core.model.unet.norm_type != "batch":
                logger.warning(
                    "predict.adabn_enabled=True but model.norm_type=%r != "
                    "'batch'; AdaBN will be a no-op (no BatchNorm to adapt).",
                    core.model.unet.norm_type)

    def _validate_cross(self, core: "Config") -> None:
        """loss/predict 与 core model/data 的交叉约束（原 core._validate_model/_validate_2_5d 片段）。"""
        loss = self.loss
        if core.model.unet.aux_topo_head:
            _require(
                str(core.model.arch).lower() == "unet",
                "aux_topo_head=True is only supported with model.arch=='unet'; "
                f"got arch={core.model.arch!r}.")
            _require(
                loss.aux_topo_weight >= 0.0,
                f"loss.aux_topo_weight must be >= 0; got {loss.aux_topo_weight}.")
            _require(
                loss.aux_topo_iter >= 1,
                f"loss.aux_topo_iter must be >= 1; got {loss.aux_topo_iter}.")
            _require(
                loss.aux_topo_loss in (
                    "auto", "dice", "bce", "smooth_l1", "mse"),
                f"Invalid loss.aux_topo_loss: {loss.aux_topo_loss!r}. "
                "Valid: 'auto' | 'dice' | 'bce' | 'smooth_l1' | 'mse'.")
        if core.data.patch_mode == "2_5d" and core.model.aux_seg_supervision:
            n_views = len(core.data.multi_res_scales)
            aw = list(loss.aux_supervision_weights)
            if aw:
                _require(
                    len(aw) == n_views - 1,
                    f"aux_supervision_weights length must = n_views-1 ({n_views-1}); "
                    f"got {aw}.")
                _require(
                    all(w >= 0 for w in aw),
                    f"aux_supervision_weights must be non-negative; got {aw}.")


def validate_seg_task(seg: SegTaskConfig, core: "Config") -> None:
    """注册表 ``validate_task`` 回调。"""
    seg.validate(core)


def hoist_legacy_seg_sections(raw: dict) -> None:
    """把旧式顶层 ``loss``/``predict`` 迁入 ``seg`` 段（就地修改）。

    新旧并存（顶层 ``loss``/``predict`` 与 ``seg.loss``/``seg.predict``）时
    fail-fast，与 ``model_migration``「新旧同设即报错」范式对齐，避免静默覆盖。
    """
    from .core import ConfigError

    existing = raw.get("seg")
    seg = dict(existing) if isinstance(existing, dict) else {}
    changed = False
    for key in ("loss", "predict"):
        if key not in raw:
            continue
        if key in seg:
            raise ConfigError(
                f"Conflicting config: both top-level {key!r} and "
                f"seg.{key} are set. Keep only one (prefer seg.{key}).")
        seg[key] = raw.pop(key)
        changed = True
    if changed or seg:
        raw["seg"] = seg


__all__ = [
    "SegTaskConfig",
    "validate_seg_task",
    "hoist_legacy_seg_sections",
]
