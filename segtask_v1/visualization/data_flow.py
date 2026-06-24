"""数据流 builder：``npz → Dataset 取块 → GPU 增强 → 中心裁剪 → pipeline 视图重塑 → 模型输入``。

完全静态推导（cfg + topology + pipeline 几何），不读盘、不需真实数据。每个节点标注
该阶段张量形状，使人能核对"数据如何一步步变成模型输入、是否符合 yaml"。

形状口径与运行期严格对齐：
* z-cube（``z_axis`` / ``2_5d``）   —— dataset 发 ``(1, eD_max, pH, pW)``，仅 z 过采样 + 面内 resize；
* ``cubic``                          —— dataset 发 ``(1, eD_max, eH_max, eW_max)``，三轴过采样；
* ``whole``                          —— dataset 发 ``(1, eD, eH, eW)``，全卷 resize；
其中 ``eX = round(pX * aug_oversample)``，``eX_max = round(eX * max_scale)``。
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from ..config import Config
from ..models.topology import ModelTopology, build_topology
from .graph import VisGraph, shape_str

logger = logging.getLogger(__name__)


def _oversample(cfg: Config) -> float:
    """训练侧有效过采样比：``< 1.0`` 视为禁用（=1.0），与 dataset 行为一致。"""
    return max(float(cfg.data.aug_oversample_ratio), 1.0)


def _extract_cube_shape(
    cfg: Config, topo: ModelTopology,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], str]:
    """返回 ``(extract_size, cube_max_size, note)``。

    * ``extract_size`` —— 过采样后、未乘 max_scale 的基准尺寸（增强工作尺寸）。
    * ``cube_max_size`` —— dataset 实发的 max-FOV cube 空间尺寸 ``(eD_max, eH_max, eW_max)``。
    """
    pD, pH, pW = (int(x) for x in cfg.data.patch_size)
    ov = _oversample(cfg)
    ms = float(max(cfg.data.multi_res_scales or [1.0]))
    pm = topo.patch_mode

    if pm in ("z_axis", "2_5d"):
        eD = int(round(pD * ov))
        extract = (eD, pH, pW)               # 仅 z 过采样；面内一次 resize 到 (pH, pW)
        cube = (int(round(eD * ms)), pH, pW)  # 仅 z 维乘 max_scale
        note = "仅 z 轴过采样 + 多 FOV；面内直接 resize 到 (pH, pW)"
    elif pm == "cubic":
        eD, eH, eW = (int(round(p * ov)) for p in (pD, pH, pW))
        extract = (eD, eH, eW)               # 三轴同步过采样
        cube = (int(round(eD * ms)), int(round(eH * ms)), int(round(eW * ms)))
        note = "三轴同步过采样 + 多 FOV cube"
    else:  # whole
        eD, eH, eW = (int(round(p * ov)) for p in (pD, pH, pW))
        extract = (eD, eH, eW)
        cube = (eD, eH, eW)                   # whole 不支持多分辨率（max_scale==1.0）
        note = "全卷一次 resize 到 extract_size（无切块、无多分辨率）"
    return extract, cube, note


def _target_patch_size(cfg: Config) -> Tuple[int, int, int]:
    """中心裁剪后的目标尺寸：优先取真实 pipeline（含 native_3d 的 max-FOV target），
    构造失败时回退到 ``patch_size``。"""
    try:
        from ..losses.losses import build_loss
        from ..trainer.pipelines.factory import build_pipeline
        pipe = build_pipeline(cfg, build_loss(cfg.loss))
        tps = getattr(pipe, "target_patch_size", None)
        if tps is not None:
            return tuple(int(x) for x in tps)
    except Exception as e:  # 构造失败不影响数据流主体（回退默认）
        logger.debug("data_flow: pipeline 构造失败，target 回退 patch_size: %s", e)
    return tuple(int(x) for x in cfg.data.patch_size)


def _model_input_shape(
    cfg: Config, topo: ModelTopology, target: Tuple[int, int, int],
) -> Tuple[int, ...]:
    """pipeline 重塑后送入模型的张量形状（含 batch 维）。"""
    B = int(cfg.data.batch_size)
    tD, tH, tW = target
    if topo.spatial_dims == 2:
        # 2.5D 折叠：深度并入通道；空间仅 (H, W)。
        return (B, topo.in_channels, tH, tW)
    return (B, topo.in_channels, tD, tH, tW)


def _reshape_note(topo: ModelTopology) -> str:
    """pipeline 视图重塑阶段的人类可读说明。"""
    if topo.spatial_dims == 2:
        if topo.keep_native_view_depth:
            return ("2.5D native-d：逐视图原生深度 D_k 拼到通道维 "
                    "→ (B, ΣD_k, H, W)")
        return "2.5D：folded 深度并入通道 → (B, D×n_views, H, W)"
    if topo.lift_2_5d_to_3d:
        return "2.5D lift→3D：多 FOV 当通道 → (B, n_views, D, H, W)"
    if topo.keep_native_multi_res:
        return ("3D native multi-res：逐视图中心裁 + resize 回 patch，"
                "stack 到通道 → (B, n_views, D, H, W)")
    if topo.n_views > 1:
        return "3D 多分辨率：各 FOV cat 到通道 → (B, n_views, D, H, W)"
    return "3D 单分辨率：squeeze C_res 轴 → (B, 1, D, H, W)"


def build_data_flow(cfg: Config, topo: Optional[ModelTopology] = None) -> VisGraph:
    """构造数据流 ``VisGraph``。"""
    topo = topo or build_topology(cfg)
    dc = cfg.data
    B = int(dc.batch_size)
    pD, pH, pW = (int(x) for x in dc.patch_size)
    ov = _oversample(cfg)
    scales: List[float] = list(dc.multi_res_scales or [1.0])
    ms = float(max(scales))
    n_fg = int(cfg.num_fg_classes)

    extract, cube, cube_note = _extract_cube_shape(cfg, topo)
    target = _target_patch_size(cfg)
    model_in = _model_input_shape(cfg, topo, target)

    g = VisGraph(title="数据流 Data Flow")
    g.meta = {
        "patch_mode": topo.patch_mode,
        "patch_size": shape_str((pD, pH, pW)),
        "batch_size": str(B),
        "multi_res_scales": "[" + ", ".join(f"{s:g}" for s in scales) + "]",
        "aug_oversample": f"{ov:g}",
        "spatial_dims": f"{topo.spatial_dims}D",
    }

    # 1) 原始数据 npz ----------------------------------------------------
    g.add_node(
        "npz", "原始体数据 (npz)", kind="data",
        key_info={"image": "(D_vol, H_vol, W_vol)", "label": "(D_vol, H_vol, W_vol)"},
        detail={
            "来源": "make_data 预打包 npz（image / label / fg_coords）",
            "归一化": dc.normalize,
            "强度窗": f"[{dc.intensity_min:g}, {dc.intensity_max:g}]",
            "前景过采样": "fg_coords 驱动中心采样（z_axis / cubic）",
        })

    # 2) Dataset 取块（max-FOV cube）------------------------------------
    g.add_node(
        "dataset", "Dataset 取块", kind="process",
        key_info={
            "输出": shape_str((1,) + cube),
            "extract_size": shape_str(extract),
            "max_scale": f"{ms:g}",
        },
        detail={
            "策略": {"whole": "WholeSpec", "z_axis": "ZCubeSpec",
                     "2_5d": "ZCubeSpec", "cubic": "CubicSpec"}.get(
                         topo.patch_mode, topo.patch_mode),
            "说明": cube_note,
            "领头 1": "压叠 C_res 轴，与历史输出布局一致",
            "z_boundary_mode": dc.z_boundary_mode,
        })
    g.add_edge("npz", "dataset", "取块 / resize")

    # 3) collate 成 batch ------------------------------------------------
    g.add_node(
        "collate", "DataLoader collate", kind="process",
        key_info={"输出": shape_str((B, 1) + cube)},
        detail={"batch_size": str(B),
                "说明": "默认 collate 在最前堆叠 batch 维"})
    g.add_edge("dataset", "collate")

    # 4) GPU 增强（形状不变）-------------------------------------------
    aug_enabled = bool(cfg.augment.enabled)
    g.add_node(
        "augment", "GPU 增强" + ("" if aug_enabled else "（已禁用）"),
        kind="process",
        key_info={"输出": shape_str((B, 1) + cube),
                  "形状": "保持不变（仅改像素 / 几何）"},
        detail={
            "augment.enabled": str(aug_enabled),
            "说明": ("GPUAugmentor 在 batch 上做仿射 / 弹性 / 强度扰动，"
                     "不改变张量形状；过采样余量供随后中心裁剪吸收边缘伪影。"),
        })
    g.add_edge("collate", "augment")

    # 5) 中心裁剪（吸收过采样余量）------------------------------------
    crop_needed = (ov > 1.0) or (tuple(cube) != tuple(target))
    g.add_node(
        "crop", "中心裁剪", kind="process",
        key_info={"输出": shape_str((B, 1) + target),
                  "target": shape_str(target)},
        detail={
            "触发": ("过采样比 > 1 或 cube ≠ target 时裁剪；否则跳过"
                     if not crop_needed else "裁掉过采样 / 多 FOV 余量"),
            "生效": str(crop_needed),
            "说明": "把 max-FOV cube 中心裁回 pipeline 目标尺寸",
        })
    g.add_edge("augment", "crop")

    # 6) pipeline 视图重塑 → 模型输入 ----------------------------------
    g.add_node(
        "reshape", "Pipeline 视图重塑", kind="process",
        key_info={"输出": shape_str(model_in),
                  "n_views": str(topo.n_views)},
        detail={
            "说明": _reshape_note(topo),
            "num_res_groups": str(topo.num_res_groups),
            "slab_depth": str(topo.slab_depth),
            "per_view_depths": str(topo.per_view_depths) or "[]",
        })
    g.add_edge("crop", "reshape")

    # 7) 模型输入框 ------------------------------------------------------
    g.add_node(
        "model_input", "模型输入", kind="input",
        key_info={
            "shape": shape_str(model_in),
            "in_channels": str(topo.in_channels),
            "spatial_dims": f"{topo.spatial_dims}D",
        },
        detail={
            "out_classes（主头）": str(topo.out_classes),
            "num_fg_classes": str(n_fg),
            "布局": ("(B, C, H, W)" if topo.spatial_dims == 2
                     else "(B, C, D, H, W)"),
        })
    g.add_edge("reshape", "model_input", "→ 模型流")

    return g


__all__ = ["build_data_flow"]
