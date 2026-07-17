"""预测流 builder：``整卷 → 预处理 → 滑窗/TTA → 模型 → 融合 → 阈值 → 输出``。

与模型流不同，这里把**整个模型抽象成一个框**（双击看推理关键参数），重点呈现推理
管线的几何与关键开关（patch_mode / overlap / blend / TTA / 阈值 / z-交错），便于核对
"用法是否与训练一致"。纯静态推导，不实例化 Predictor、不读盘。
"""

from __future__ import annotations

from typing import Optional, Tuple

from taskcore.config.core import Config
from taskcore.models.topology import ModelTopology, build_topology
from .graph import VisGraph, shape_str


def _patch_forward_shape(
    cfg: Config, topo: ModelTopology,
) -> Tuple[int, ...]:
    """推理单 batch 送入模型的 patch 形状（含 batch 维 = predict.batch_size）。"""
    B = int(cfg.predict.batch_size)
    pD, pH, pW = (int(x) for x in cfg.data.patch_size)
    if topo.spatial_dims == 2:
        return (B, topo.in_channels, pH, pW)
    return (B, topo.in_channels, pD, pH, pW)


def _threshold_str(thr) -> str:
    """阈值展示串：标量 → '0.5'；逐类列表 → '[0.5, 0.3]'。"""
    if isinstance(thr, (list, tuple)):
        return "[" + ", ".join(f"{float(t):g}" for t in thr) + "]"
    return f"{float(thr):g}"


def build_predict_flow(
    cfg: Config, topo: Optional[ModelTopology] = None,
) -> VisGraph:
    """构造预测流 ``VisGraph``。"""
    topo = topo or build_topology(cfg)
    pc = cfg.predict
    dc = cfg.data
    n_fg = int(cfg.num_fg_classes)
    pD, pH, pW = (int(x) for x in dc.patch_size)
    patch_fwd = _patch_forward_shape(cfg, topo)

    z_interleave = bool(pc.z_interleave_enabled and dc.patch_mode == "2_5d")
    # cubic 下 H/W 轴可单独设 overlap（hw_overlap=None 时三轴同 z_overlap）。
    overlap_str = (
        f"z={pc.z_overlap:g}, hw={pc.hw_overlap:g}"
        if (dc.patch_mode == "cubic" and pc.hw_overlap is not None)
        else f"{pc.z_overlap:g}")
    adabn_pv = bool(pc.adabn_enabled and pc.adabn_mode == "per_volume")

    g = VisGraph(title="预测流 Prediction Flow")
    g.meta = {
        "patch_mode": dc.patch_mode,
        "patch_size": shape_str((pD, pH, pW)),
        "overlap": overlap_str,
        "blend_mode": pc.blend_mode,
        "tta_flip": str(pc.tta_flip),
        "threshold": _threshold_str(pc.threshold),
    }

    # 1) 输入整卷 -------------------------------------------------------
    g.add_node(
        "volume", "输入整卷 (NIfTI)", kind="data",
        key_info={"shape": "(D, H, W)"},
        detail={"加载": "load_nifti / load_nifti_with_spacing（z-交错需物理 spacing）"})

    # 2) 可选 bbox 裁剪 -------------------------------------------------
    g.add_node(
        "bbox", "BBox 裁剪（可选）", kind="process",
        key_info={"启用": "bbox_path 提供时", "输出": "(D', H', W')"},
        detail={"说明": "在 ROI 内推理，推理后按偏移拼回原画布，外部填 0 概率"})
    g.add_edge("volume", "bbox")

    # 3) 预处理 ---------------------------------------------------------
    g.add_node(
        "preprocess", "预处理", kind="process",
        key_info={"输出": "(D, H, W) fp32"},
        detail={
            "强度窗": f"[{dc.intensity_min:g}, {dc.intensity_max:g}]",
            "normalize": dc.normalize,
            "说明": "与训练同口径（intensity window + normalize）",
        })
    g.add_edge("bbox", "preprocess")

    prev = "preprocess"

    # 3.5) 可选 per-volume AdaBN ---------------------------------------
    if adabn_pv:
        g.add_node(
            "adabn", "AdaBN 重估 (per_volume)", kind="process",
            key_info={"mode": "per_volume"},
            detail={"说明": "用本卷先跑一遍前向重估 BN running stats，再冻结预测"})
        g.add_edge(prev, "adabn")
        prev = "adabn"

    # 4) 滑窗取块 -------------------------------------------------------
    dispatch = {
        "whole": "whole_volume_forward（整卷一次）",
        "cubic": "sliding_window_cubic（三轴滑窗）",
        "z_axis": "sliding_window_z（z 轴滑窗）",
        "2_5d": ("sliding_window_z_interleaved（z 交错）"
                 if z_interleave else "sliding_window_z（2.5D forward）"),
    }.get(dc.patch_mode, dc.patch_mode)
    g.add_node(
        "sliding", "滑窗取块", kind="process",
        key_info={"patch": shape_str(patch_fwd),
                  "overlap": overlap_str,
                  "batch_size": str(pc.batch_size)},
        detail={
            "dispatch": dispatch,
            "z_interleave_enabled": str(z_interleave),
            "z_interleave_factors": str(list(pc.z_interleave_factors)),
            "说明": "按 patch_mode 滑窗切块，几何与训练一致",
        })
    g.add_edge(prev, "sliding")

    # 5) 可选 TTA flip --------------------------------------------------
    prev = "sliding"
    if pc.tta_flip:
        g.add_node(
            "tta", "TTA Flip", kind="process",
            key_info={"启用": "True",
                      "tta_batch_size": str(pc.tta_batch_size)},
            detail={"说明": ("flip 变体批量前向后平均（3D 7 种 / 2.5D 3 种），"
                             "提升鲁棒性")})
        g.add_edge("sliding", "tta")
        prev = "tta"

    # 6) 模型（抽象单框）----------------------------------------------
    g.add_node(
        "model", "模型 (前向)", kind="model",
        key_info={"in": shape_str(patch_fwd),
                  "out": shape_str((patch_fwd[0], topo.out_classes)
                                   + tuple(patch_fwd[2:]))},
        detail={
            "arch": str(cfg.model.arch),
            "in_channels": str(topo.in_channels),
            "out_classes": str(topo.out_classes),
            "AMP": f"use_amp={cfg.train.use_amp}, dtype={cfg.train.amp_dtype}",
            "说明": "推理期模型整体视为黑盒；结构细节见「模型流」标签页",
        })
    g.add_edge(prev, "model")

    # 7) 概率融合 -------------------------------------------------------
    g.add_node(
        "blend", "重叠融合", kind="process",
        key_info={"mode": pc.blend_mode,
                  "输出": shape_str((n_fg,)) + " → (num_fg, D, H, W)"},
        detail={
            "blend_mode": pc.blend_mode,
            "说明": "滑窗重叠区按高斯 / 均匀权重加权累积成整卷概率",
        })
    g.add_edge("model", "blend")

    # 8) 阈值 → 标签 ----------------------------------------------------
    g.add_node(
        "label", "阈值 → 标签", kind="process",
        key_info={"threshold": _threshold_str(pc.threshold),
                  "输出": "(D, H, W) int"},
        detail={
            "threshold": _threshold_str(pc.threshold),
            "label_values": str(list(dc.label_values)),
            "说明": "prob_to_label：逐前景类 sigmoid 概率阈值化为标签图",
        })
    g.add_edge("blend", "label")

    # 9) 输出 -----------------------------------------------------------
    g.add_node(
        "output", "输出", kind="output",
        key_info={"label_map": "(D, H, W)",
                  "probabilities": "(num_fg, D, H, W)"},
        detail={
            "save_probabilities": str(pc.save_probabilities),
            "说明": "拼回原尺寸 / 原 affine 后保存 NIfTI",
        })
    g.add_edge("label", "output")

    return g


__all__ = ["build_predict_flow"]
