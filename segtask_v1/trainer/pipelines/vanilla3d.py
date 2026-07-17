"""Vanilla3D：``patch_mode∈{whole, z_axis, cubic}`` 且 **未启用** ``keep_native_multi_res``。

涵盖 3D 单分辨率以及 dataset 端 eager 多分辨率（``len(multi_res_scales)>1`` 但
按通道直接堆好），不做任何视图二次重塑。损失为 ``MultiResolutionLoss(num_res=n_views)``
（必要时被 ``DeepSupervisionLoss`` 包装）。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from taskcore.config.core import Config
from ...losses.losses import DeepSupervisionLoss, MultiResolutionLoss
from taskcore.engine.amp import compute_loss_fp32
from .base import SupervisionPack, ViewPipeline


class Vanilla3DPipeline(ViewPipeline):
    """3D pass-through pipeline（含 whole 与 patch3d-单/eager 多分辨率）。"""

    def __init__(self, cfg: Config, base_loss):
        self.cfg = cfg
        self.base_loss = base_loss

        n_views = len(cfg.data.multi_res_scales)
        self.n_views        = n_views
        self.n_aux_views    = 0  # 3D的不叫aux，叫res_group
        self.num_res_groups = n_views
        self.slab_depth     = 0

        self.main_loss_fn = MultiResolutionLoss(
            base_loss=base_loss,
            num_fg_classes=cfg.num_fg_classes,
            num_res=n_views,
            label_values=cfg.data.label_values)
        if cfg.model.deep_supervision and cfg.loss.deep_supervision_weights:
            self.criterion = DeepSupervisionLoss(
                self.main_loss_fn, cfg.loss.deep_supervision_weights)
        else:
            self.criterion = self.main_loss_fn

        self.aux_loss_fn     = None
        self.aux_loss_fns    = None
        self.aux_weights     = []
        self.mr_native_sizes = []
        self.per_view_depths = []

        self.target_patch_size = tuple(int(x) for x in cfg.data.patch_size)  # 3D单个整体

    def prepare_batch(self, image, label, wmap):
        return image, SupervisionPack(label_main=label, wmap_main=wmap)

    def prepare_val_batch(self, image, label):
        return image, label

    def compute_loss(self, pred, sup: SupervisionPack, breakdown=None):
        main_pred, _aux, topo_pred = self.split_pred(pred)
        loss = compute_loss_fp32(
            self.criterion, main_pred, sup.label_main, weight_map=sup.wmap_main)
        if breakdown is not None:
            breakdown["L_main"] = float(loss.detach().item())
        loss = self.add_topo_loss(loss, topo_pred, sup, breakdown)
        if breakdown is not None:
            breakdown["L_total"] = float(loss.detach().item())
        return loss


__all__ = ["Vanilla3DPipeline"]
