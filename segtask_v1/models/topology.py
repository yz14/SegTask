"""``ModelTopology`` —— 训练几何 / 通道布局派生量的单一真相源。

R5 引入。在此之前同一组派生量（``in_channels`` / ``out_classes`` /
``context_n_views`` / ``in_ch_per_view_list`` / ``aux_head_out_channels`` /
``aux_view_depths`` / ``spatial_dims``）由 ``Config.sync`` 与
``models.factory.build_model`` **各算一遍**，新增 patch_mode 时容易遗漏其中一处。

R5 后：

* ``build_topology(cfg)`` —— 唯一推导入口（``patch_mode`` × 5 个 mode flag → 全部派生量）
* ``Config.sync``         —— 调用 ``build_topology`` 写回 ``cfg.model.{in_channels, spatial_dims}``，保持旧 yaml / 旧外部代码读 ``cfg.model.in_channels`` 不破坏
* ``Config.aux_view_depths`` —— 委托 ``build_topology(self).aux_view_depths``
* ``models.factory.build_model`` —— 读 ``Topology`` 全字段，不再自行推导
* ``trainer.pipelines.factory.build_pipeline`` —— 读 ``Topology`` 决策（不再自行 ``len(cfg.data.multi_res_scales)``）

新增 patch_mode：仅需修改 ``build_topology`` 内的决策树。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelTopology:
    """模式无关地描述当前训练几何 / 通道布局。所有字段在 ``build_topology`` 中一次算齐。"""

    # ---- raw mode flags（mirror cfg；pipeline / dataset / model 决策直接读这里） ----
    patch_mode: str                     # "whole" | "z_axis" | "cubic" | "2_5d"
    lift_2_5d_to_3d: bool               # 仅 2.5D；其他模式恒 False
    aux_keep_native_d: bool             # 2.5D 懒视图；其他模式恒 False
    keep_native_multi_res: bool         # 3D 懒视图；其他模式恒 False

    # ---- 几何派生量 ----
    n_views: int                        # = max(len(multi_res_scales), 1)
    num_res_groups: int                 # 主路通道分组数：3D=n_views, 2.5D=1
    slab_depth: int                     # 2.5D 时 = D；其他 = 0
    aux_view_depths: List[int] = field(default_factory=list)
    # 仅 2.5D 非空：[D_0=D, D_1, ...]，强制 D_0=D。

    # ---- 模型 I/O 通道布局 ----
    in_channels: int = 1                # 模型输入通道数（写回 cfg.model.in_channels）
    out_classes: int = 1                # 主头输出通道（= num_fg × {1, D, n_views}）
    spatial_dims: int = 3               # 2 (2.5D folded) | 3 (其他)
    context_n_views: int = 1            # encoder stem 融合视图数（仅 2.5D=n_views）
    in_ch_per_view_list: Optional[List[int]] = None
    # 仅 ``aux_keep_native_d`` 启用时非 None；按 view 拆通道。

    # ---- Aux 监督拓扑 ----
    aux_seg_active: bool = False        # = aux_seg_supervision AND n_views > 1（已合并门控）
    aux_head_out_channels: Optional[List[int]] = None
    # 仅 ``aux_keep_native_d`` 启用时 = [num_fg * D_k for k in 1..]，否则 None=默认 num_fg。

    @property
    def num_fg_classes(self) -> int:
        """主头每 view / 每 slice 的前景类数（``out_classes // num_res_groups // (D if folded)``）。

        仅供日志/校验使用；构造模型时直接传 ``out_classes``。
        """
        return max(self.out_classes // max(self.num_res_groups, 1)
                   // max(self.slab_depth if self.slab_depth else 1, 1), 1)


# ---------------------------------------------------------------------------
# Single derivation entry point
# ---------------------------------------------------------------------------
def build_topology(cfg: "Config") -> ModelTopology:
    """从 ``cfg`` 一次性派生全部模型/训练几何字段。

    判定优先级（与重构前 ``Config.sync`` + ``models.factory.build_model`` 行为完全等价）：

    1. ``patch_mode == '2_5d'``
       a. ``lift_2_5d_to_3d=True``                → spatial_dims=3, in_ch=n_views, out_classes=num_fg, num_res_groups=1
       b. ``aux_keep_native_d=True`` & n_views>1  → spatial_dims=2, in_ch=Σ D_k,  out_classes=num_fg×D, num_res_groups=1
       c. otherwise                                → spatial_dims=2, in_ch=D×n_views, out_classes=num_fg×D, num_res_groups=1
    2. 3D ``patch_mode∈{whole, z_axis, cubic}``    → spatial_dims=3, in_ch=n_views, out_classes=num_fg×n_views, num_res_groups=n_views
    """
    dc = cfg.data
    mc = cfg.model
    pm = str(dc.patch_mode).lower()
    n_views = max(len(dc.multi_res_scales), 1)
    D = int(dc.patch_size[0])
    num_fg = cfg.num_fg_classes

    is_2_5d = pm == "2_5d"
    lift = bool(getattr(mc, "lift_2_5d_to_3d", False)) and is_2_5d
    native_d = (bool(getattr(dc, "aux_keep_native_d", False))
                and is_2_5d and n_views > 1)
    keep_native_3d = (bool(getattr(dc, "keep_native_multi_res", False))
                      and pm in ("z_axis", "cubic") and n_views > 1)
    aux_seg_active = (bool(getattr(mc, "aux_seg_supervision", False))
                      and n_views > 1)

    # ---- 通道 / 输出几何 -------------------------------------------------
    if is_2_5d and not lift:
        spatial_dims = 2
        num_res_groups = 1
        out_classes = num_fg * D
        if native_d:
            depths = [int(round(D * float(s))) for s in dc.multi_res_scales]
            depths[0] = D            # s_0 == 1.0
            in_channels = int(sum(depths))
        else:
            in_channels = D * n_views
    elif is_2_5d and lift:
        spatial_dims = 3
        num_res_groups = 1
        out_classes = num_fg
        in_channels = n_views          # rank-5 image, C_res = n_views
    else:                              # 3D（whole / z_axis / cubic）
        spatial_dims = 3
        num_res_groups = n_views
        out_classes = num_fg * num_res_groups
        in_channels = n_views          # 1 通道/视图（whole 时 n_views=1）

    # ---- 2.5D 专属 ------------------------------------------------------
    slab_depth = D if is_2_5d else 0
    aux_view_depths: List[int] = []
    if is_2_5d:
        ds = [int(round(D * float(s))) for s in dc.multi_res_scales]
        if ds:
            ds[0] = D
        aux_view_depths = ds

    context_n_views = n_views if is_2_5d else 1

    # ---- native_d 专属 --------------------------------------------------
    in_ch_per_view_list: Optional[List[int]] = None
    aux_head_out_channels: Optional[List[int]] = None
    if native_d:
        in_ch_per_view_list = list(aux_view_depths)
        aux_head_out_channels = [num_fg * d_k for d_k in aux_view_depths[1:]]

    return ModelTopology(
        patch_mode=pm,
        lift_2_5d_to_3d=lift,
        aux_keep_native_d=native_d,
        keep_native_multi_res=keep_native_3d,
        n_views=n_views,
        num_res_groups=num_res_groups,
        slab_depth=slab_depth,
        aux_view_depths=aux_view_depths,
        in_channels=in_channels,
        out_classes=out_classes,
        spatial_dims=spatial_dims,
        context_n_views=context_n_views,
        in_ch_per_view_list=in_ch_per_view_list,
        aux_seg_active=aux_seg_active,
        aux_head_out_channels=aux_head_out_channels,
    )


__all__ = ["ModelTopology", "build_topology"]
