"""2.5D Slab pipelines（``patch_mode=2_5d`` 且未启用 ``lift_2_5d_to_3d``）。

三种子模式：

* ``Slab2_5DPipeline``         —— 无 aux。``(B,C_res,D,H,W) → (B,C_res*D,H,W)``。
* ``Slab2_5DAuxPipeline``      —— aux folded：image 折叠，label/wmap 保 rank-5
  以便逐视图取 ``[:, k]`` 作 aux 监督；aux 内损为共享 ``SliceChannelLoss``。
* ``Slab2_5DNativeDPipeline``  —— aux 异深：image 是 ``(B, ΣD_k, H, W)``；
  aux 内损为逐视图 ``SliceChannelLoss``（``num_slices=D_k``）。
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

from ...config import Config
from ...losses.losses import (
    DeepSupervisionLoss, MultiResolutionLoss, SliceChannelLoss,
)
from .. import views
from ..amp import compute_loss_fp32
from .base import SupervisionPack, ViewPipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------
def _resolve_aux_weights(cfg: Config, n_aux: int) -> List[float]:
    user_w = list(getattr(cfg.loss, "aux_supervision_weights", []))
    if not user_w:
        # 几何衰减：越宽 FOV 对齐越差，权重越小。
        user_w = [0.5 ** (k + 1) for k in range(n_aux)]
    elif len(user_w) != n_aux:
        raise ValueError(
            f"loss.aux_supervision_weights length ({len(user_w)}) "
            f"must equal n_views-1 ({n_aux}); got {user_w}.")
    return [float(w) for w in user_w]


def _accumulate_main(criterion, pred_main, sup, breakdown):
    """主路 fp32 损失 + breakdown ``L_main`` 标量。"""
    main_l = compute_loss_fp32(
        criterion, pred_main, sup.label_main, weight_map=sup.wmap_main)
    if breakdown is not None:
        breakdown["L_main"] = float(main_l.detach().item())
    return main_l


# ---------------------------------------------------------------------------
# 2.5D folded, no aux
# ---------------------------------------------------------------------------
class Slab2_5DPipeline(ViewPipeline):
    """2.5D 折叠（无 aux）：``image (B,C_res,D,H,W) → (B,C_res*D,H,W)``。"""

    def __init__(self, cfg: Config, base_loss):
        self.cfg = cfg
        self.base_loss = base_loss

        n_views = len(cfg.data.multi_res_scales)
        D = int(cfg.data.patch_size[0])
        self.n_views = n_views
        self.n_aux_views = 0
        self.num_res_groups = 1
        self.slab_depth = D

        self.inner_loss = SliceChannelLoss(
            base_loss=base_loss,
            num_fg_classes=cfg.num_fg_classes,
            num_slices=D,
            label_values=cfg.data.label_values,
            reduction=cfg.loss.slice_loss_reduction,
        )
        if cfg.model.deep_supervision and cfg.loss.deep_supervision_weights:
            self.criterion = DeepSupervisionLoss(
                self.inner_loss, cfg.loss.deep_supervision_weights)
        else:
            self.criterion = self.inner_loss

        self.aux_inner_loss = None
        self.aux_inner_losses = None
        self.aux_weights = []
        self.mr_native_sizes = []
        self.aux_view_depths = []
        self.target_patch_size = tuple(int(x) for x in cfg.data.patch_size)
        logger.info(
            "Loss: %s [2.5D, reduction=%s], num_slices=%d, fg_classes=%d",
            cfg.loss.name, cfg.loss.slice_loss_reduction,
            D, cfg.num_fg_classes)

    def prepare_batch(self, image, label, wmap):
        image, label, wmap = views.squeeze_2_5d(image, label, wmap)
        return image, SupervisionPack(label_main=label, wmap_main=wmap)

    def prepare_val_batch(self, image, label):
        image, label, _ = views.squeeze_2_5d(image, label, None)
        return image, label

    def compute_loss(self, pred, sup: SupervisionPack, breakdown=None):
        loss = compute_loss_fp32(
            self.criterion, pred, sup.label_main, weight_map=sup.wmap_main)
        if breakdown is not None:
            breakdown["L_total"] = float(loss.detach().item())
        return loss


# ---------------------------------------------------------------------------
# 2.5D folded + aux (shared SliceChannelLoss for aux)
# ---------------------------------------------------------------------------
class Slab2_5DAuxPipeline(ViewPipeline):
    """2.5D 折叠 + 多 FOV aux 监督。aux 内损共享单 ``SliceChannelLoss``。"""

    def __init__(self, cfg: Config, base_loss):
        self.cfg = cfg
        self.base_loss = base_loss

        n_views = len(cfg.data.multi_res_scales)
        if n_views < 2:
            raise ValueError(
                "Slab2_5DAuxPipeline requires n_views >= 2; "
                f"got {n_views}.")
        n_aux = n_views - 1
        D = int(cfg.data.patch_size[0])

        self.n_views = n_views
        self.n_aux_views = n_aux
        self.num_res_groups = 1
        self.slab_depth = D

        self.inner_loss = SliceChannelLoss(
            base_loss=base_loss,
            num_fg_classes=cfg.num_fg_classes,
            num_slices=D,
            label_values=cfg.data.label_values,
            reduction=cfg.loss.slice_loss_reduction,
        )
        if cfg.model.deep_supervision and cfg.loss.deep_supervision_weights:
            self.criterion = DeepSupervisionLoss(
                self.inner_loss, cfg.loss.deep_supervision_weights)
        else:
            self.criterion = self.inner_loss

        self.aux_inner_loss = SliceChannelLoss(
            base_loss=base_loss,
            num_fg_classes=cfg.num_fg_classes,
            num_slices=D,
            label_values=cfg.data.label_values,
            reduction=cfg.loss.slice_loss_reduction,
        )
        self.aux_inner_losses = None
        self.aux_weights = _resolve_aux_weights(cfg, n_aux)
        self.mr_native_sizes = []
        self.aux_view_depths = []
        self.target_patch_size = tuple(int(x) for x in cfg.data.patch_size)
        logger.info(
            "Aux seg supervision: ENABLED, n_aux_views=%d, weights=%s, "
            "fusion=%s",
            n_aux, self.aux_weights, cfg.model.context_fusion)

    def prepare_batch(self, image, label, wmap):
        # image 折叠；label/wmap 保 rank-5；主 label 取 view 0 用于 metrics。
        image, label_all, wmap_all = views.squeeze_2_5d_keep_views(
            image, label, wmap)
        return image, SupervisionPack(
            label_main=label_all[:, 0],
            wmap_main=wmap_all[:, 0] if wmap_all is not None else None,
            label_all_views=label_all,
            wmap_all_views=wmap_all,
        )

    def prepare_val_batch(self, image, label):
        # val 无 aux；折叠后仅 view-0 监督指标。
        image, label, _ = views.squeeze_2_5d(image, label, None)
        return image, label

    def compute_loss(self, pred, sup: SupervisionPack, breakdown=None):
        if isinstance(pred, dict):
            main_pred = pred["main"]
            aux_preds = pred.get("aux", []) or []
        else:
            main_pred, aux_preds = pred, []

        total = _accumulate_main(self.criterion, main_pred, sup, breakdown)
        if not aux_preds or self.aux_inner_loss is None:
            if breakdown is not None:
                breakdown["L_total"] = float(total.detach().item())
            return total
        if len(aux_preds) != len(self.aux_weights):
            raise RuntimeError(
                f"Number of aux predictions ({len(aux_preds)}) does not "
                f"match number of aux weights ({len(self.aux_weights)}).")

        label_all = sup.label_all_views
        wmap_all = sup.wmap_all_views
        for k_idx, (ap, w_k) in enumerate(zip(aux_preds, self.aux_weights)):
            view_k = k_idx + 1
            lbl_k = label_all[:, view_k]
            wm_k = wmap_all[:, view_k] if wmap_all is not None else None
            aux_l = compute_loss_fp32(
                self.aux_inner_loss, ap, lbl_k, weight_map=wm_k)
            total = total + w_k * aux_l
            if breakdown is not None:
                breakdown[f"L_aux_{view_k}"] = float(aux_l.detach().item())
                breakdown[f"w_aux_{view_k}"] = float(w_k)
        if breakdown is not None:
            breakdown["L_total"] = float(total.detach().item())
        return total


# ---------------------------------------------------------------------------
# 2.5D folded + aux + native depths
# ---------------------------------------------------------------------------
class Slab2_5DNativeDPipeline(ViewPipeline):
    """2.5D + aux + ``aux_keep_native_d``：逐视图深度 ``D_k`` 不同。"""

    def __init__(self, cfg: Config, base_loss):
        self.cfg = cfg
        self.base_loss = base_loss

        n_views = len(cfg.data.multi_res_scales)
        if n_views < 2:
            raise ValueError(
                "Slab2_5DNativeDPipeline requires n_views >= 2; "
                f"got {n_views}.")
        n_aux = n_views - 1
        D = int(cfg.data.patch_size[0])

        depths = list(cfg.aux_view_depths)
        assert depths and depths[0] == D, (
            "aux_view_depths[0] must equal patch_size[0]; "
            f"got {depths[0] if depths else None} vs {D}.")
        assert sum(depths) == int(cfg.model.in_channels), (
            f"sum(aux_view_depths)={sum(depths)} must equal "
            f"model.in_channels={cfg.model.in_channels}.")

        self.n_views = n_views
        self.n_aux_views = n_aux
        self.num_res_groups = 1
        self.slab_depth = D
        self.aux_view_depths = depths

        self.inner_loss = SliceChannelLoss(
            base_loss=base_loss,
            num_fg_classes=cfg.num_fg_classes,
            num_slices=D,
            label_values=cfg.data.label_values,
            reduction=cfg.loss.slice_loss_reduction,
        )
        if cfg.model.deep_supervision and cfg.loss.deep_supervision_weights:
            self.criterion = DeepSupervisionLoss(
                self.inner_loss, cfg.loss.deep_supervision_weights)
        else:
            self.criterion = self.inner_loss

        self.aux_inner_loss = None
        self.aux_inner_losses = [
            SliceChannelLoss(
                base_loss=base_loss,
                num_fg_classes=cfg.num_fg_classes,
                num_slices=int(d_k),
                label_values=cfg.data.label_values,
                reduction=cfg.loss.slice_loss_reduction,
            )
            for d_k in depths[1:]
        ]
        self.aux_weights = _resolve_aux_weights(cfg, n_aux)
        self.mr_native_sizes = []

        # 增强后的 max-FOV target（按 D 维 max_scale）
        max_scale = max(cfg.data.multi_res_scales)
        target_d_native = int(round(D * max_scale))
        self.target_patch_size = (
            target_d_native,
            int(cfg.data.patch_size[1]),
            int(cfg.data.patch_size[2]))
        logger.info(
            "Aux seg supervision: ENABLED (native depth), "
            "n_aux_views=%d, per-view depths=%s, weights=%s, "
            "fusion=%s",
            n_aux, depths[1:], self.aux_weights, cfg.model.context_fusion)
        logger.info(
            "Trainer aux_keep_native_d=True: max-FOV crop D=%d, "
            "per-view depths=%s, channel layout sum=%d.",
            target_d_native, depths, int(cfg.model.in_channels))

    def prepare_batch(self, image, label, wmap):
        image, label_main, wmap_main, aux_labels, aux_wmaps = (
            views.split_views_native_d(
                image, label, wmap,
                aux_view_depths=self.aux_view_depths,
                target_patch_size=self.target_patch_size))
        return image, SupervisionPack(
            label_main=label_main, wmap_main=wmap_main,
            aux_labels=aux_labels, aux_wmaps=aux_wmaps)

    def prepare_val_batch(self, image, label):
        image, label_main, _, _, _ = views.split_views_native_d(
            image, label, None,
            aux_view_depths=self.aux_view_depths,
            target_patch_size=self.target_patch_size)
        return image, label_main

    def compute_loss(self, pred, sup: SupervisionPack, breakdown=None):
        if isinstance(pred, dict):
            main_pred = pred["main"]
            aux_preds = pred.get("aux", []) or []
        else:
            main_pred, aux_preds = pred, []

        total = _accumulate_main(self.criterion, main_pred, sup, breakdown)
        if not aux_preds or not self.aux_inner_losses:
            if breakdown is not None:
                breakdown["L_total"] = float(total.detach().item())
            return total
        if sup.aux_labels is None:
            raise RuntimeError(
                "Slab2_5DNativeDPipeline.compute_loss requires sup.aux_labels "
                "but received None — likely a missing prepare_batch call.")
        if not (len(aux_preds) == len(self.aux_weights)
                == len(self.aux_inner_losses) == len(sup.aux_labels)):
            raise RuntimeError(
                "aux_keep_native_d arity mismatch: "
                f"preds={len(aux_preds)}, weights={len(self.aux_weights)}, "
                f"losses={len(self.aux_inner_losses)}, "
                f"labels={len(sup.aux_labels)}.")

        for k_idx, (ap, w_k, loss_k, lbl_k) in enumerate(zip(
                aux_preds, self.aux_weights, self.aux_inner_losses,
                sup.aux_labels)):
            view_k = k_idx + 1
            wm_k = (sup.aux_wmaps[k_idx] if sup.aux_wmaps is not None else None)
            aux_l = compute_loss_fp32(loss_k, ap, lbl_k, weight_map=wm_k)
            total = total + w_k * aux_l
            if breakdown is not None:
                breakdown[f"L_aux_{view_k}"] = float(aux_l.detach().item())
                breakdown[f"w_aux_{view_k}"] = float(w_k)
        if breakdown is not None:
            breakdown["L_total"] = float(total.detach().item())
        return total


__all__ = [
    "Slab2_5DPipeline",
    "Slab2_5DAuxPipeline",
    "Slab2_5DNativeDPipeline"]
