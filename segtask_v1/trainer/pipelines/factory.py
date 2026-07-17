"""Pipeline 工厂：``cfg → ViewPipeline``。

整个 codebase 中**唯一允许 trainer 侧大段 if/elif 的地方**——所有 mode 派生量
（``n_views`` / ``keep_native_view_depth`` / ``keep_native_multi_res`` / ``aux_seg_active``
等）由 ``ModelTopology`` 统一供给（R5），本工厂仅做策略对象选择。
"""

from __future__ import annotations

import logging

from taskcore.config.core import Config
from ...losses.topo_aux import build_aux_topo_loss
from taskcore.models.topology import ModelTopology, build_topology
from .base import ViewPipeline
from .lift25d import Lift2_5DAuxPipeline, Lift2_5DPipeline
from .patch3d import Patch3DNativeMultiResPipeline
from .slab25d import (
    Slab2_5DAuxPipeline,
    Slab2_5DNativeDPipeline,
    Slab2_5DPipeline,
)
from .vanilla3d import Vanilla3DPipeline

logger = logging.getLogger(__name__)


def build_pipeline(cfg: Config, base_loss) -> ViewPipeline:
    """根据 ``cfg`` 选 pipeline。返回的对象自带 criterion / aux 损失栈。

    判定优先级（与历史 ``Trainer.__init__`` 行为等价；所有 mode 派生量来自
    ``ModelTopology``，与 ``models.factory.build_model`` 共用同一真相源）：

    1. ``patch_mode == '2_5d'``
       a. ``lift_2_5d_to_3d``                     → ``Lift2_5DAuxPipeline`` / ``Lift2_5DPipeline``
       b. ``aux_seg_active`` & ``keep_native_view_depth`` → ``Slab2_5DNativeDPipeline``
       c. ``aux_seg_active``                       → ``Slab2_5DAuxPipeline``
       d. otherwise                                → ``Slab2_5DPipeline``
    2. 3D ``patch_mode∈{whole, z_axis, cubic}``
       a. ``keep_native_multi_res`` & n_views>1 & mode∈{z_axis, cubic}
          → ``Patch3DNativeMultiResPipeline``
       b. otherwise → ``Vanilla3DPipeline``
    """
    topo: ModelTopology = build_topology(cfg)
    is_2_5d = topo.patch_mode == "2_5d"

    if is_2_5d:
        lift     = topo.lift_2_5d_to_3d
        aux      = topo.aux_seg_active
        native_d = topo.keep_native_view_depth

        if lift and aux:
            cls = Lift2_5DAuxPipeline
        elif lift:
            cls = Lift2_5DPipeline
        elif aux and native_d:
            cls = Slab2_5DNativeDPipeline
        elif aux:
            cls = Slab2_5DAuxPipeline
        else:
            cls = Slab2_5DPipeline
    else:
        cls = (Patch3DNativeMultiResPipeline if topo.keep_native_multi_res
               else Vanilla3DPipeline)

    logger.info("ViewPipeline selected: %s (patch_mode=%s, n_views=%d)",
                cls.__name__, topo.patch_mode, topo.n_views)
    pipeline = cls(cfg, base_loss)

    # 中心线/距离场辅助头损失：与具体 pipeline 无关，统一注入（compute_loss 内各加一项）。
    if cfg.model.aux_topo_head:
        pipeline.aux_topo_loss_fn = build_aux_topo_loss(cfg.model, cfg.loss)
        pipeline.aux_topo_weight = float(cfg.loss.aux_topo_weight)
        logger.info(
            "Aux topo head: ENABLED (target=%s, loss=%s, iter=%d, weight=%.3f)",
            cfg.model.aux_topo_target, pipeline.aux_topo_loss_fn.loss_name,
            cfg.loss.aux_topo_iter, pipeline.aux_topo_weight)
    return pipeline


__all__ = ["build_pipeline"]
