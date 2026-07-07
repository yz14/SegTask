"""按 ``ModelTopology`` 选择生成多视图管线。"""

from __future__ import annotations

from ...models.topology import build_topology
from .base import GenViewPipeline
from .native_d import NativeDPipeline
from .stacked import StackedMultiResPipeline
from .vanilla import VanillaPipeline


def build_pipeline(cfg) -> GenViewPipeline:
    """从 cfg（经 ``sync()``）构造 batch 几何准备管线。

    决策树（与 ``build_topology`` 对齐）：

    * n_views == 1                       → VanillaPipeline（仅裁过采样余量）
    * 2.5D + keep_native_view_depth      → NativeDPipeline（原生深度 slab 拼接）
    * 其余多视图（z_axis / cubic / 2.5D 统一深度 / lift）
                                         → StackedMultiResPipeline
    """
    topo = build_topology(cfg)
    patch_size = tuple(cfg.data.patch_size)
    scales = list(cfg.data.multi_res_scales) if cfg.data.multi_res_scales else [1.0]

    if topo.n_views <= 1:
        return VanillaPipeline(patch_size, scales)
    if topo.patch_mode == "2_5d" and topo.keep_native_view_depth:
        return NativeDPipeline(patch_size, scales, topo.per_view_depths)
    return StackedMultiResPipeline(
        patch_size, scales, cubic_fov=(topo.patch_mode == "cubic"))


__all__ = ["build_pipeline"]
