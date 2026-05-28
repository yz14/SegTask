"""Pipeline 工厂：``cfg → ViewPipeline``。

整个 codebase 中**唯一允许大段 if/elif 的地方**——把模式判断集中到这一处，
其他文件不再分支。
"""

from __future__ import annotations

import logging

from ...config import Config
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
    """根据 ``cfg`` 五个 flag 选 pipeline。返回的对象自带 criterion / aux 损失栈。

    判定优先级（与历史 ``Trainer.__init__`` 行为等价）：

    1. ``patch_mode == "2_5d"``
       a. ``lift_2_5d_to_3d``  → ``Lift2_5DAuxPipeline`` / ``Lift2_5DPipeline``
       b. ``aux_seg_supervision`` & ``aux_keep_native_d`` → ``Slab2_5DNativeDPipeline``
       c. ``aux_seg_supervision``                        → ``Slab2_5DAuxPipeline``
       d. otherwise                                      → ``Slab2_5DPipeline``
    2. 3D ``patch_mode∈{whole, z_axis, cubic}``
       a. ``keep_native_multi_res`` & n_views>1 & mode∈{z_axis, cubic}
          → ``Patch3DNativeMultiResPipeline``
       b. otherwise → ``Vanilla3DPipeline``
    """
    is_2_5d = cfg.data.patch_mode == "2_5d"
    n_views = len(cfg.data.multi_res_scales)

    if is_2_5d:
        lift = bool(getattr(cfg.model, "lift_2_5d_to_3d", False))
        aux = bool(getattr(cfg.model, "aux_seg_supervision", False)) and n_views > 1
        native_d = (bool(getattr(cfg.data, "aux_keep_native_d", False))
                    and n_views > 1)

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
        keep_native = (bool(getattr(cfg.data, "keep_native_multi_res", False))
                       and cfg.data.patch_mode in ("z_axis", "cubic")
                       and n_views > 1)
        cls = (Patch3DNativeMultiResPipeline if keep_native
               else Vanilla3DPipeline)

    logger.info("ViewPipeline selected: %s (patch_mode=%s, n_views=%d)",
                cls.__name__, cfg.data.patch_mode, n_views)
    return cls(cfg, base_loss)


__all__ = ["build_pipeline"]
