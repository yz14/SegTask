"""Lift2.5D：2.5D 数据但模型走真 3D（``(B, num_fg, D, H, W)``）。

* ``Lift2_5DPipeline``    —— 仅以 view 0 作主监督；image 保 rank-5。
* ``Lift2_5DAuxPipeline`` —— 多 FOV aux，aux 内损为 ``MultiResolutionLoss(num_res=1)``。

两者均强制 ``num_res_groups=1``（aux 视图不参与主路 num_res 复合）。
"""

from __future__ import annotations

import logging

import torch

from ...config import Config
from ...losses.losses import DeepSupervisionLoss, MultiResolutionLoss
from ..amp import compute_loss_fp32
from .base import SupervisionPack, ViewPipeline
from .slab25d import _accumulate_main, _resolve_aux_weights

logger = logging.getLogger(__name__)


class Lift2_5DPipeline(ViewPipeline):
    """2.5D Lift（无 aux）：image 保 rank-5；label/wmap 仅取 view 0 (``[:, :1]``)。"""

    def __init__(self, cfg: Config, base_loss):
        self.cfg = cfg
        self.base_loss = base_loss

        n_views = len(cfg.data.multi_res_scales)
        D = int(cfg.data.patch_size[0])
        self.n_views = n_views
        self.n_aux_views = 0
        self.num_res_groups = 1
        self.slab_depth = D

        self.inner_loss = MultiResolutionLoss(
            base_loss=base_loss,
            num_fg_classes=cfg.num_fg_classes,
            num_res=1,
            label_values=cfg.data.label_values,
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
        self.per_view_depths = []
        self.target_patch_size = tuple(int(x) for x in cfg.data.patch_size)
        logger.info(
            "Loss: %s, scales=%d, fg_classes=%d [2.5D LIFTED to 3D]",
            cfg.loss.name, 1, cfg.num_fg_classes)

    def prepare_batch(self, image, label, wmap):
        # image 不变；仅以 view 0 作监督，保留 C_res=1 以合 num_res=1。
        label = label[:, :1].contiguous()
        wmap = wmap[:, :1].contiguous() if wmap is not None else None
        return image, SupervisionPack(label_main=label, wmap_main=wmap)

    def prepare_val_batch(self, image, label):
        return image, label[:, :1].contiguous()

    def compute_loss(self, pred, sup: SupervisionPack, breakdown=None):
        loss = compute_loss_fp32(
            self.criterion, pred, sup.label_main, weight_map=sup.wmap_main)
        if breakdown is not None:
            breakdown["L_total"] = float(loss.detach().item())
        return loss


class Lift2_5DAuxPipeline(ViewPipeline):
    """2.5D Lift + aux：image 保 rank-5；label_all 整 (B,n_views,D,H,W) 留作 aux 索引。

    Aux head shape ``(B, num_fg, D, H, W)`` → ``MultiResolutionLoss(num_res=1)``。
    """

    def __init__(self, cfg: Config, base_loss):
        self.cfg = cfg
        self.base_loss = base_loss

        n_views = len(cfg.data.multi_res_scales)
        if n_views < 2:
            raise ValueError(
                "Lift2_5DAuxPipeline requires n_views >= 2; "
                f"got {n_views}.")
        n_aux = n_views - 1
        D = int(cfg.data.patch_size[0])

        self.n_views = n_views
        self.n_aux_views = n_aux
        self.num_res_groups = 1
        self.slab_depth = D

        self.inner_loss = MultiResolutionLoss(
            base_loss=base_loss,
            num_fg_classes=cfg.num_fg_classes,
            num_res=1,
            label_values=cfg.data.label_values,
        )
        if cfg.model.deep_supervision and cfg.loss.deep_supervision_weights:
            self.criterion = DeepSupervisionLoss(
                self.inner_loss, cfg.loss.deep_supervision_weights)
        else:
            self.criterion = self.inner_loss

        self.aux_inner_loss = MultiResolutionLoss(
            base_loss=base_loss,
            num_fg_classes=cfg.num_fg_classes,
            num_res=1,
            label_values=cfg.data.label_values,
        )
        self.aux_inner_losses = None
        self.aux_weights = _resolve_aux_weights(cfg, n_aux)
        self.mr_native_sizes = []
        self.per_view_depths = []
        self.target_patch_size = tuple(int(x) for x in cfg.data.patch_size)
        logger.info(
            "Aux seg supervision: ENABLED [LIFT], n_aux_views=%d, "
            "weights=%s, fusion=%s",
            n_aux, self.aux_weights, cfg.model.stem_fusion_mode)

    def prepare_batch(self, image, label, wmap):
        # image 保 rank-5；label/wmap 整存以便逐视图 [:, k:k+1] 取出（rank-5）
        return image, SupervisionPack(
            label_main=label[:, :1].contiguous(),
            wmap_main=wmap[:, :1].contiguous() if wmap is not None else None,
            label_all_views=label,
            wmap_all_views=wmap,
        )

    def prepare_val_batch(self, image, label):
        return image, label[:, :1].contiguous()

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
            lbl_k = label_all[:, view_k:view_k + 1]
            wm_k = (wmap_all[:, view_k:view_k + 1]
                    if wmap_all is not None else None)
            aux_l = compute_loss_fp32(
                self.aux_inner_loss, ap, lbl_k, weight_map=wm_k)
            total = total + w_k * aux_l
            if breakdown is not None:
                breakdown[f"L_aux_{view_k}"] = float(aux_l.detach().item())
                breakdown[f"w_aux_{view_k}"] = float(w_k)
        if breakdown is not None:
            breakdown["L_total"] = float(total.detach().item())
        return total


__all__ = ["Lift2_5DPipeline", "Lift2_5DAuxPipeline"]
