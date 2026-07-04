"""检测 patch 数据集：npz 卷 → (image patch, boxes, labels)。

框真值只存 3D 一份（Plan §3.4 / §8-2）：

* npz ``boxes`` 键 ``(N, 7) = [z1, y1, x1, z2, y2, x2, cls]``（cls ∈ [0, K)）
  优先；
* 否则由分割 mask 连通域派生（``det.boxes_from_mask``，scipy.ndimage.label
  逐前景类），弱标注冷启动。

patch 抽取后框同步联动（``crop_boxes``）；2.5D 折叠时由 3D 框对 slab 切片
派生 2D 框（``slice_boxes_to_2d``）。输出：

* 3D  —— ``image (1, D, H, W)``，``boxes (N, 6)``；
* 2.5D —— ``image (D, H, W)``（折叠），``boxes (N, 4)``。

框数逐样本可变 → DataLoader 用 :func:`det_collate`（boxes/labels 保持 list）。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

from segtask_v1.data.dataset import preprocess_image

from ..targets import crop_boxes, slice_boxes_to_2d

logger = logging.getLogger(__name__)


def boxes_from_mask(lbl: np.ndarray, fg_values: Sequence[float],
                    min_voxels: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """mask 连通域 → 3D 框。返回 (boxes (N,6) float32, labels (N,) int64)。"""
    all_boxes: List[List[float]] = []
    all_labels: List[int] = []
    for k, v in enumerate(fg_values):
        comp, n = ndimage.label(lbl == v)
        if n == 0:
            continue
        objects = ndimage.find_objects(comp)
        for sl in objects:
            if sl is None:
                continue
            vox = np.prod([s.stop - s.start for s in sl])
            if vox < min_voxels:
                continue
            all_boxes.append([sl[0].start, sl[1].start, sl[2].start,
                              sl[0].stop, sl[1].stop, sl[2].stop])
            all_labels.append(k)
    if not all_boxes:
        return (np.zeros((0, 6), np.float32), np.zeros((0,), np.int64))
    return (np.asarray(all_boxes, np.float32),
            np.asarray(all_labels, np.int64))


def load_volume_boxes(npz_path: str, fg_values: Sequence[float],
                      allow_mask: bool, min_voxels: int
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读 npz → (image, boxes3d (N,6), labels (N,))。"""
    with np.load(npz_path, allow_pickle=True) as f:
        if "image" not in f.files:
            raise KeyError(f"npz {npz_path!r} has no 'image' key "
                           f"(keys={list(f.files)}).")
        img = np.asarray(f["image"])
        if "boxes" in f.files:
            raw = np.asarray(f["boxes"], np.float32).reshape(-1, 7)
            boxes, labels = raw[:, :6], raw[:, 6].astype(np.int64)
        elif allow_mask and "label" in f.files:
            boxes, labels = boxes_from_mask(
                np.asarray(f["label"]), fg_values, min_voxels)
        else:
            raise KeyError(
                f"npz {npz_path!r} has neither 'boxes' (N,7) nor a 'label' "
                f"mask usable with det.boxes_from_mask "
                f"(keys={list(f.files)}).")
    if img.ndim != 3:
        raise ValueError(f"expected 3D volume (D,H,W); got {img.shape} in "
                         f"{npz_path!r}.")
    return img, boxes, labels


class DetPatchDataset(Dataset):
    """检测 patch 数据集（几何无关：3D / 2.5D 折叠由 ``spatial_dims`` 决定）。

    训练随机中心抽 patch（``fg_oversample_ratio`` 概率以某 gt 框中心为锚，
    保证正样本供给）；验证中心确定性派生。
    """

    def __init__(
        self,
        npz_paths          : Sequence[str],
        patch_size         : Sequence[int],
        fg_values          : Optional[Sequence[float]] = None,
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
        seed               : int = 42):
        self.paths = list(npz_paths)
        if not self.paths:
            raise ValueError("DetPatchDataset got empty npz_paths.")
        self.patch = tuple(int(s) for s in patch_size)
        if len(self.patch) != 3:
            raise ValueError(f"patch_size must be [D, H, W]; got {patch_size}.")
        if spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3; got {spatial_dims}.")
        self.spatial_dims = int(spatial_dims)
        self.fold_2_5d = self.spatial_dims == 2
        self.fg_values = [float(v) for v in (fg_values or [1.0])]
        self.allow_mask = bool(boxes_from_mask_ok)
        self.min_box_voxels = int(min_box_voxels)
        self.intensity_min = float(intensity_min)
        self.intensity_max = float(intensity_max)
        self.normalize = str(normalize)
        self.global_mean = float(global_mean)
        self.global_std = float(global_std)
        self.spv = max(int(samples_per_volume), 1)
        self.is_train = bool(is_train)
        self.fg_ratio = float(fg_oversample_ratio)
        self.min_vis = float(min_visibility)
        self.seed = int(seed)
        logger.info(
            "DetPatchDataset(%s): %d volumes x %d samples, patch=%s, "
            "spatial_dims=%d (%s)",
            "train" if is_train else "val", len(self.paths), self.spv,
            self.patch, self.spatial_dims,
            "2.5D folded" if self.fold_2_5d else "3D")

    def __len__(self) -> int:
        return len(self.paths) * self.spv

    # ------------------------------------------------------------------
    def _sample_offset(self, rng: np.random.Generator,
                       shape: Tuple[int, ...],
                       boxes: np.ndarray) -> Tuple[int, int, int]:
        """patch 左上角偏移（保证 patch 完整落在卷内或贴边）。"""
        if (self.is_train and boxes.shape[0] > 0
                and rng.random() < self.fg_ratio):
            b = boxes[int(rng.integers(boxes.shape[0]))]
            center = [(b[i] + b[i + 3]) / 2 for i in range(3)]
            off = [int(round(c - p / 2)) for c, p in zip(center, self.patch)]
        else:
            off = [int(rng.integers(0, max(dim - p, 0) + 1))
                   for dim, p in zip(shape, self.patch)]
        return tuple(int(np.clip(o, 0, max(d - p, 0)))
                     for o, d, p in zip(off, shape, self.patch))

    def _extract(self, vol: np.ndarray, off: Tuple[int, int, int]
                 ) -> np.ndarray:
        slices = tuple(slice(o, o + p) for o, p in zip(off, self.patch))
        out = vol[slices]
        pads = [(0, p - s) for p, s in zip(self.patch, out.shape)]
        if any(b for _, b in pads):
            out = np.pad(out, pads, mode="edge")
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        vol_idx = idx % len(self.paths)
        img, boxes_np, labels_np = load_volume_boxes(
            self.paths[vol_idx], self.fg_values, self.allow_mask,
            self.min_box_voxels)
        img = preprocess_image(
            img, self.intensity_min, self.intensity_max, self.normalize,
            self.global_mean, self.global_std, inplace=False)
        if self.is_train:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        off = self._sample_offset(rng, img.shape, boxes_np)
        patch = self._extract(img, off)

        boxes3d = torch.from_numpy(boxes_np.astype(np.float32, copy=False))
        labels = torch.from_numpy(labels_np)
        boxes3d, labels = crop_boxes(boxes3d, labels, off, self.patch,
                                     self.min_vis)

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
           "load_volume_boxes"]
