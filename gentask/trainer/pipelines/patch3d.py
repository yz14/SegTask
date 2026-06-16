"""Patch3DNativeMultiRes：3D ``z_axis`` / ``cubic`` 且 ``keep_native_multi_res=True``。

dataset 发送一份 max-FOV cube；本 pipeline 在增强后的 batch 上做"中心裁 + resize
回 patch_size"，把每个视图 stack 到通道维 → ``(B, n_views, pD, pH, pW)``。
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from ...config import Config
from ...losses.losses import DeepSupervisionLoss, MultiResolutionLoss
from .. import views
from ..amp import compute_loss_fp32
from .base import SupervisionPack, ViewPipeline


class Patch3DNativeMultiResPipeline(ViewPipeline):
    """3D 懒多分辨率：max-FOV cube → 逐视图通道堆叠。"""

    def __init__(self, cfg: Config, base_loss):
        self.cfg = cfg
        self.base_loss = base_loss

        n_views = len(cfg.data.multi_res_scales)
        if n_views < 2:
            raise ValueError(
                "Patch3DNativeMultiResPipeline requires len(multi_res_scales) >= 2; "
                f"got {n_views}.")

        self.n_views = n_views
        self.n_aux_views = 0
        self.num_res_groups = n_views
        self.slab_depth = 0

        self.main_loss_fn = MultiResolutionLoss(
            base_loss=base_loss,
            num_fg_classes=cfg.num_fg_classes,
            num_res=n_views,
            label_values=cfg.data.label_values,
        )
        if cfg.model.deep_supervision and cfg.loss.deep_supervision_weights:
            self.criterion = DeepSupervisionLoss(
                self.main_loss_fn, cfg.loss.deep_supervision_weights)
        else:
            self.criterion = self.main_loss_fn

        self.aux_loss_fn = None
        self.aux_loss_fns = None
        self.aux_weights = []
        self.per_view_depths = []

        # 每视图的原始尺寸：z_axis 仅缩 D；cubic 缩 3 轴。
        pD, pH, pW = (int(x) for x in cfg.data.patch_size)
        sizes: List[Tuple[int, int, int]] = []
        for s in cfg.data.multi_res_scales:
            D_k = int(round(pD * float(s)))
            if cfg.data.patch_mode == "z_axis":
                H_k, W_k = pH, pW
            else:  # cubic
                H_k = int(round(pH * float(s)))
                W_k = int(round(pW * float(s)))
            sizes.append((D_k, H_k, W_k))
        sizes[0] = (pD, pH, pW)  # view 0 强对齐 patch_size，防浮点漂移
        self.mr_native_sizes = sizes
        self._patch_size = (pD, pH, pW)

        # 增强后 max-FOV target
        max_scale = max(cfg.data.multi_res_scales)
        if cfg.data.patch_mode == "z_axis":
            self.target_patch_size = (int(round(pD * max_scale)), pH, pW)
        else:
            self.target_patch_size = (
                int(round(pD * max_scale)),
                int(round(pH * max_scale)),
                int(round(pW * max_scale)))
        # 交叉检查：逐视图原生尺寸不得超 max-FOV 目标。
        for k, (D_k, H_k, W_k) in enumerate(sizes):
            tD, tH, tW = self.target_patch_size
            if D_k > tD or H_k > tH or W_k > tW:
                raise ValueError(
                    f"keep_native_multi_res: view {k} native size "
                    f"({D_k},{H_k},{W_k}) exceeds max-FOV target "
                    f"{self.target_patch_size}. Check multi_res_scales / "
                    "patch_size for floating-point drift.")

    # ------------------------------------------------------------------
    def prepare_batch(self, image, label, wmap):
        image, label, wmap = views.split_views_native_3d(
            image, label, wmap,
            target_patch_size=self.target_patch_size,
            mr_native_sizes=self.mr_native_sizes,
            patch_size=self._patch_size,
        )
        return image, SupervisionPack(label_main=label, wmap_main=wmap)

    def prepare_val_batch(self, image, label):
        # val 无增强；cube 已到 max-FOV target，直接拆。
        image, label, _ = views.split_views_native_3d(
            image, label, None,
            target_patch_size=self.target_patch_size,
            mr_native_sizes=self.mr_native_sizes,
            patch_size=self._patch_size,
        )
        return image, label

    def compute_loss(self, pred, sup: SupervisionPack, breakdown=None):
        loss = compute_loss_fp32(
            self.criterion, pred, sup.label_main, weight_map=sup.wmap_main)
        if breakdown is not None:
            breakdown["L_total"] = float(loss.detach().item())
        return loss


__all__ = ["Patch3DNativeMultiResPipeline"]
