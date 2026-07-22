"""检测 patch 数据集：npz 卷 → (image patch, boxes, labels)。

框真值只存 3D 一份（Plan §3.4 / §8-2）：

* npz ``boxes`` 键 ``(N, 7) = [z1, y1, x1, z2, y2, x2, cls]``（cls ∈ [0, K)）
  优先；
* 否则由分割 mask 连通域派生（``det.boxes_from_mask``，scipy.ndimage.label
  逐前景类），弱标注冷启动。

patch 抽取口径与 segtask 各 ``patch_mode`` 逐位一致（保证 SSL / 分割预训练
encoder 看到的输入分布一致），框全程几何联动：

* ``cubic``  —— 3 轴随机中心 cube，越界 edge 复制（``crop_boxes`` 平移裁剪）；
* ``z_axis`` / ``2_5d`` —— z 轴滑窗（edge-padded）+ H/W 面内 resize 到
  patch_size（z 向 ``crop_boxes``、面内 ``scale_boxes``）；
* ``whole``  —— 全卷 resize 到 patch_size（``scale_boxes`` 三轴缩放）。

2.5D 折叠时由 3D 框对 slab 切片派生 2D 框（``slice_boxes_to_2d``）。输出：

* 3D  —— ``image (1, D, H, W)``，``boxes (N, 6)``；
* 2.5D —— ``image (D, H, W)``（折叠），``boxes (N, 4)``。

训练增强：``aug_flip_prob`` > 0 时逐空间轴独立随机翻转（``flip_boxes``
联动；强度增强由 trainer 在 GPU 上复用 seg 管道施加，不动框）。

验证采样：中心由 (seed, idx) 确定性派生；``val_grid_coverage=True`` 时改为
确定性网格覆盖（z 模式沿 z 等距 bin 中心，cubic 用 Halton(2,3,5)，口径同
segtask / clstask）——与推理滑窗铺点一致。

框数逐样本可变 → DataLoader 用 :func:`det_collate`（boxes/labels 保持 list）。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

from taskcore.data.dataset import (
    open_npz,
    extract_z_patch_padded,
    load_npz_image,
    load_npz_label,
    resize_3d,
)
from taskcore.data.patch_dataset_base import IndexScheme, NpzPatchDatasetBase
from taskcore.data.patch_extract import extract_patch_by_mode
from taskcore.data.patch_ops import (
    extract_cubic_patch_with_origin,
    safe_center_range,
)
from taskcore.data.sampling import (
    clip_center_to_ranges,
    halton_center,
    uniform_center,
    z_grid_center,
)

from ..targets import crop_boxes, flip_boxes, scale_boxes, slice_boxes_to_2d

logger = logging.getLogger(__name__)


def boxes_from_mask(lbl: np.ndarray, fg_values: Sequence[float],
                    min_voxels: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """mask 连通域 → 3D 框。返回 (boxes (N,6) float32, labels (N,) int64)。

    ``min_voxels`` 按连通域真实体素数过滤（非包围盒体积）。"""
    all_boxes: List[List[float]] = []
    all_labels: List[int] = []
    for k, v in enumerate(fg_values):
        comp, n = ndimage.label(lbl == v)
        if n == 0:
            continue
        objects = ndimage.find_objects(comp)
        for ci, sl in enumerate(objects, start=1):
            if sl is None:
                continue
            vox = int((comp[sl] == ci).sum())
            if vox < min_voxels:
                continue
            all_boxes.append([sl[0].start, sl[1].start, sl[2].start,
                              sl[0].stop, sl[1].stop, sl[2].stop])
            all_labels.append(k)
    if not all_boxes:
        return (np.zeros((0, 6), np.float32), np.zeros((0,), np.int64))
    return (np.asarray(all_boxes, np.float32),
            np.asarray(all_labels, np.int64))


def load_boxes(npz_path: str, fg_values: Sequence[float],
               allow_mask: bool, min_voxels: int
               ) -> Tuple[np.ndarray, np.ndarray]:
    """读 npz 框真值 → (boxes3d (N,6), labels (N,))。

    'boxes' 键（小数组）优先；否则由 label mask 连通域派生（label 走
    seg 的 memmap 零拷贝快路径，压缩 npz 自动回退）。不读 image。"""
    with open_npz(npz_path) as f:
        keys = list(f.files)
        if "boxes" in keys:
            raw = np.asarray(f["boxes"], np.float32).reshape(-1, 7)
            return raw[:, :6], raw[:, 6].astype(np.int64)
    if allow_mask and "label" in keys:
        return boxes_from_mask(
            np.asarray(load_npz_label(npz_path)), fg_values, min_voxels)
    raise KeyError(
        f"npz {npz_path!r} has neither 'boxes' (N,7) nor a 'label' "
        f"mask usable with det.boxes_from_mask (keys={keys}).")


def load_volume_boxes(npz_path: str, fg_values: Sequence[float],
                      allow_mask: bool, min_voxels: int
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读 npz → (原始 image, boxes3d (N,6), labels (N,))。"""
    with open_npz(npz_path) as f:
        if "image" not in f.files:
            raise KeyError(f"npz {npz_path!r} has no 'image' key "
                           f"(keys={list(f.files)}).")
        img = np.asarray(f["image"])
    boxes, labels = load_boxes(npz_path, fg_values, allow_mask, min_voxels)
    if img.ndim != 3:
        raise ValueError(f"expected 3D volume (D,H,W); got {img.shape} in "
                         f"{npz_path!r}.")
    return img, boxes, labels


class DetPatchDataset(NpzPatchDatasetBase):
    """检测 patch 数据集（patch 抽取口径由 ``patch_mode`` 决定，3D 或 2.5D
    折叠由 ``spatial_dims`` 决定）。

    训练随机中心抽 patch（``fg_oversample_ratio`` 概率以某 gt 框中心为锚，
    保证正样本供给）；验证中心确定性派生或网格覆盖。
    """

    def __init__(
        self,
        npz_paths          : Sequence[str],
        patch_size         : Sequence[int],
        fg_values          : Optional[Sequence[float]] = None,
        patch_mode         : str = "cubic",
        boxes_from_mask_ok : bool = True,
        min_box_voxels     : int = 8,
        intensity_min      : float = -1024.0,
        intensity_max      : float = 3071.0,
        normalize          : str = "minmax",
        global_mean        : float = 0.0,
        global_std         : float = 1.0,
        samples_per_volume : int = 8,
        spatial_dims       : int = 3,
        is_train           : bool = True,
        fg_oversample_ratio: float = 0.5,
        min_visibility     : float = 0.25,
        seed               : int = 42,
        aug_flip_prob      : float = 0.0,
        aug_flip_axes      : Sequence[int] = (),
        val_grid_coverage  : bool = False,
        cache_enabled      : bool = False,
        cache_max_volumes  : int = 0):
        super().__init__(
            npz_paths, patch_size, patch_mode, samples_per_volume,
            spatial_dims, is_train, seed, val_grid_coverage,
            intensity_min, intensity_max, normalize, global_mean, global_std,
            fg_oversample_ratio, cache_enabled, cache_max_volumes,
            index_scheme=IndexScheme.INTERLEAVED,
            dataset_name="DetPatchDataset",
        )
        self.fg_values = [float(v) for v in (fg_values or [1.0])]
        self.allow_mask = bool(boxes_from_mask_ok)
        self.min_box_voxels = int(min_box_voxels)
        self.min_vis = float(min_visibility)
        self.flip_prob = float(aug_flip_prob) if self.is_train else 0.0
        self.flip_axes = [int(a) for a in aug_flip_axes]
        if any(a not in (0, 1, 2) for a in self.flip_axes):
            raise ValueError(
                f"aug_flip_axes must be spatial axes in (0,1,2); "
                f"got {aug_flip_axes}.")
        self._box_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        logger.info(
            "DetPatchDataset(%s): %d volumes x %d samples, mode=%s, "
            "patch=%s, spatial_dims=%d (%s)%s",
            "train" if is_train else "val", len(self.paths), self.spv,
            self.mode, self.patch, self.spatial_dims,
            "2.5D folded" if self.fold_2_5d else "3D",
            ", val_grid_coverage" if self.val_grid_coverage else "")

    def _load(self, path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """→ (预处理后 image, boxes3d, labels)；image/框均逐 worker 缓存。"""
        cached = self._box_cache.get(path)
        if cached is None:
            cached = load_boxes(path, self.fg_values, self.allow_mask,
                                self.min_box_voxels)
            self._box_cache[path] = cached
        img = self._load_image_cached(path)
        boxes_np, labels_np = cached
        return img, boxes_np, labels_np

    # ------------------------------------------------------------------
    # 中心采样
    # ------------------------------------------------------------------
    def _sample_z(self, rng: np.random.Generator, D_vol: int,
                  boxes: np.ndarray, cov_j: Optional[int]) -> int:
        """z 模式中心 z：val 覆盖 → 等距 bin 中心；训练 fg 命中 → 某 gt 框
        z 中心；否则均匀采样。"""
        if cov_j is not None:
            return z_grid_center(cov_j, self.spv, D_vol)
        if (self.is_train and boxes.shape[0] > 0
                and rng.random() < self.fg_ratio):
            b = boxes[int(rng.integers(boxes.shape[0]))]
            return int(np.clip(round((b[0] + b[3]) / 2), 0, D_vol - 1))
        return int(rng.integers(0, D_vol))

    def _sample_center(self, rng: np.random.Generator,
                       shape: Tuple[int, ...], boxes: np.ndarray,
                       cov_j: Optional[int]) -> Tuple[int, int, int]:
        """cubic 模式中心 (d,h,w)：val 覆盖 → Halton(2,3,5) 铺满安全中心域；
        训练 fg 命中 → 某 gt 框中心夹取到安全范围；否则安全域内均匀采样。"""
        ranges = safe_center_range(shape, self.patch)
        if cov_j is not None:
            return halton_center(cov_j, ranges)
        if (self.is_train and boxes.shape[0] > 0
                and rng.random() < self.fg_ratio):
            b = boxes[int(rng.integers(boxes.shape[0]))]
            center = tuple(round((b[i] + b[i + 3]) / 2) for i in range(3))
            return clip_center_to_ranges(center, ranges)
        return uniform_center(rng, ranges)

    # ------------------------------------------------------------------
    # patch 抽取（image + boxes 同一几何）
    # ------------------------------------------------------------------
    def _extract_with_boxes(
        self, img: np.ndarray, boxes3d: torch.Tensor, labels: torch.Tensor,
        rng: np.random.Generator, boxes_np: np.ndarray,
        cov_j: Optional[int],
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """按 patch_mode 抽取严格 (pD,pH,pW) patch 并联动框到 patch 坐标。"""
        pD, pH, pW = self.patch
        D, H, W = img.shape
        if self.mode == "whole":
            center = (0, 0, 0)
            patch = extract_patch_by_mode(img, self.mode, center, self.patch)
            boxes = scale_boxes(boxes3d, (pD / D, pH / H, pW / W))
            return patch, boxes, labels
        if self.mode == "cubic":
            center = self._sample_center(rng, img.shape, boxes_np, cov_j)
            patch, lo = extract_cubic_patch_with_origin(img, center, self.patch)
            boxes, labels = crop_boxes(boxes3d, labels, lo, self.patch,
                                       self.min_vis)
            return patch, boxes, labels
        # z_axis / 2_5d：先在 (pD,H,W) 联动 crop，再面内 resize（框几何依赖此顺序）。
        zc = self._sample_z(rng, D, boxes_np, cov_j)
        patch = extract_z_patch_padded(img, zc, pD)
        z_lo = zc - pD // 2
        boxes, labels = crop_boxes(boxes3d, labels, (z_lo, 0, 0),
                                   (pD, H, W), self.min_vis)
        patch = resize_3d(patch, pD, pH, pW)
        boxes = scale_boxes(boxes, (1.0, pH / H, pW / W))
        return patch, boxes, labels

    def _apply_flips(self, patch: np.ndarray, boxes3d: torch.Tensor,
                     rng: np.random.Generator
                     ) -> Tuple[np.ndarray, torch.Tensor]:
        """训练随机翻转（3D patch 坐标系，逐轴独立；框同步联动）。"""
        for axis in self.flip_axes:
            if rng.random() >= self.flip_prob:
                continue
            patch = np.flip(patch, axis=axis)
            boxes3d = flip_boxes(boxes3d, axis, self.patch)
        return np.ascontiguousarray(patch), boxes3d

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        vol_idx = self._vol_idx(idx)
        img, boxes_np, labels_np = self._load(self.paths[vol_idx])
        rng, cov_j = self._item_rng_and_cov(idx)

        boxes3d = torch.from_numpy(boxes_np.astype(np.float32, copy=False))
        labels = torch.from_numpy(labels_np)
        patch, boxes3d, labels = self._extract_with_boxes(
            img, boxes3d, labels, rng, boxes_np, cov_j)
        if self.flip_prob > 0 and self.flip_axes:
            patch, boxes3d = self._apply_flips(patch, boxes3d, rng)

        img_t = torch.from_numpy(patch.astype(np.float32, copy=False))
        if self.fold_2_5d:
            # slab 折叠：全 slab 已在 patch 的 z 范围内 → 直接取 yx 范围。
            boxes, labels = slice_boxes_to_2d(
                boxes3d, labels, 0, self.patch[0], min_overlap=0.0)
            # (D, H, W)：深度折进通道。
        else:
            boxes = boxes3d
            img_t = img_t.unsqueeze(0)                     # (1, D, H, W)
        return {"image": img_t, "boxes": boxes, "labels": labels}


def det_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, object]:
    """images 堆叠；boxes/labels 逐样本变长 → 保持 list。"""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "boxes": [b["boxes"] for b in batch],
        "labels": [b["labels"] for b in batch],
    }


__all__ = ["DetPatchDataset", "det_collate", "boxes_from_mask",
           "load_boxes", "load_volume_boxes"]
