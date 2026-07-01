"""Image-only（无标注）npz patch 数据集 + dataloader 构造。

SSL 的价值来源是**大规模无标注**语料，故数据通路与分割的"标签耦合"管线解耦：
本数据集只读 npz 的 ``image`` 键（``make_ssl_data`` 产出的 image-only npz，或任何含
``image`` 的既有 npz 皆可），**不读** label / fg_coords / fg_slices，均匀随机抽取
``patch_size`` 立方体（越界 edge-pad），返回 ``{"image": (1, *patch)}``。

底层 IO / 预处理（``_open_npz`` / ``preprocess_image`` / ``_extract_cubic_patch``）直接
复用 ``segtask_v1.data.dataset``，不另造轮子。
"""

from __future__ import annotations

import glob
import logging
import os
import random
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from segtask_v1.data.dataset import (
    _extract_cubic_patch,
    _open_npz,
    preprocess_image,
)

logger = logging.getLogger(__name__)


def discover_image_npz(npz_dir: str, npz_suffix: str = ".npz") -> List[str]:
    """递归发现 ``npz_dir`` 下所有以 ``npz_suffix`` 结尾的 npz；按路径排序。"""
    if not npz_dir or not os.path.isdir(npz_dir):
        raise FileNotFoundError(
            f"SSL npz_dir not found or empty: {npz_dir!r}. Point data.npz_dir "
            f"to a directory of image npz packages (image-only or labelled).")
    paths = sorted(
        glob.glob(os.path.join(npz_dir, "**", f"*{npz_suffix}"), recursive=True))
    if not paths:
        raise RuntimeError(
            f"No '*{npz_suffix}' packages found under {npz_dir!r}.")
    return paths


def _rand_center(dim: int, p: int) -> int:
    """在 [0, dim) 取一个 center，使大小 p 的 patch 尽量落在体内（dim<=p 时取中点）。"""
    half = p // 2
    if dim <= p:
        return dim // 2
    lo = random.randint(0, dim - p)   # patch 起点 ∈ [0, dim-p]，保证不越界
    return lo + half


class ImageOnlyPatchDataset(Dataset):
    """从含 ``image`` 键的 npz 均匀随机抽 patch（image-only）。

    每个 epoch 的样本数 = ``len(paths) * samples_per_volume``；``__getitem__`` 内随机
    选体 + 随机中心抽 ``patch_size`` cube。

    输出布局随 ``spatial_dims`` 切换（与下游骨干一致）：

    * ``spatial_dims==3`` —— 3D 骨干：返回 ``{"image": (1, pD, pH, pW)} fp32``。
    * ``spatial_dims==2`` —— 2.5D 折叠骨干：把深度 D 折进通道，返回
      ``{"image": (pD, pH, pW)} fp32``（C=pD，spatial=(pH,pW)），与 segtask 2.5D
      ``squeeze_2_5d`` 折叠口径一致（单 FOV）。
    """

    def __init__(
        self,
        npz_paths        : Sequence[str],
        patch_size       : Sequence[int],
        intensity_min    : float,
        intensity_max    : float,
        normalize        : str = "minmax",
        samples_per_volume: int = 1,
        global_mean      : float = 0.0,
        global_std       : float = 1.0,
        spatial_dims     : int = 3):
        self.paths = list(npz_paths)
        if not self.paths:
            raise ValueError("ImageOnlyPatchDataset got empty npz_paths.")
        self.patch = tuple(int(s) for s in patch_size)  # (pD, pH, pW)
        if len(self.patch) != 3:
            raise ValueError(
                f"patch_size must be 3D (D,H,W) for SSL image-only dataset; "
                f"got {patch_size}.")
        self.spatial_dims = int(spatial_dims)
        if self.spatial_dims not in (2, 3):
            raise ValueError(
                f"spatial_dims must be 2 (2.5D folded) or 3; "
                f"got {self.spatial_dims}.")
        self.fold_2_5d = self.spatial_dims == 2
        self.intensity_min = float(intensity_min)
        self.intensity_max = float(intensity_max)
        self.normalize = str(normalize)
        self.global_mean = float(global_mean)
        self.global_std = float(global_std)
        self.spv = max(int(samples_per_volume), 1)
        logger.info(
            "ImageOnlyPatchDataset: %d volumes x %d samples = %d, patch=%s, "
            "spatial_dims=%d (%s)",
            len(self.paths), self.spv, len(self), self.patch,
            self.spatial_dims, "2.5D folded" if self.fold_2_5d else "3D")

    def __len__(self) -> int:
        return len(self.paths) * self.spv

    def _load_volume(self, path: str) -> np.ndarray:
        with _open_npz(path) as f:
            if "image" not in f.files:
                raise KeyError(
                    f"npz {path!r} has no 'image' key (keys={list(f.files)}).")
            img_int16 = f["image"]
            return preprocess_image(
                img_int16, self.intensity_min, self.intensity_max,
                self.normalize, self.global_mean, self.global_std,
                inplace=False)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.paths[idx % len(self.paths)]
        vol = self._load_volume(path)              # (D, H, W) fp32
        if vol.ndim != 3:
            raise ValueError(
                f"SSL expects 3D image volume (D,H,W); got {vol.shape} in {path!r}.")
        center = tuple(_rand_center(d, p) for d, p in zip(vol.shape, self.patch))
        patch = _extract_cubic_patch(vol, center, self.patch)  # (pD,pH,pW)
        t = torch.from_numpy(patch.astype(np.float32, copy=False))  # (pD,pH,pW)
        if not self.fold_2_5d:
            t = t.unsqueeze(0)                                      # (1,pD,pH,pW)
        # 2.5D：直接 (pD,pH,pW) = (C=D, H, W)，深度折进通道。
        return {"image": t}


class LabeledPatchDataset(Dataset):
    """从含 ``image`` + ``label`` 键的 npz 抽**配对** patch，供 §0.5 在线探针。

    与 :class:`ImageOnlyPatchDataset` 共用 IO/预处理与抽样逻辑，额外读 ``label`` 并以
    *同一中心* 抽取对齐的 label patch（``mode='edge'`` 越界复制，与 image 一致）。
    label 为原始取值，前景二值化在探针侧按 ``label_values`` 完成。仅用于轻量评测，
    不进 SSL 训练主路径。

    输出布局随 ``spatial_dims`` 切换（与 :class:`ImageOnlyPatchDataset` 折叠口径一致）：

    * ``spatial_dims==3`` —— ``{"image": (1, pD, pH, pW), "label": (1, pD, pH, pW)}``。
    * ``spatial_dims==2`` —— 2.5D 折叠：``{"image": (pD, pH, pW), "label": (pD, pH, pW)}``
      （C=pD 折进通道，spatial=(pH,pW)）；探针侧按 ``b (c d) h w`` 口径逐 (类,切片) 二值化。
    """

    def __init__(
        self,
        npz_paths         : Sequence[str],
        patch_size        : Sequence[int],
        intensity_min     : float,
        intensity_max     : float,
        normalize         : str = "minmax",
        samples_per_volume: int = 1,
        global_mean       : float = 0.0,
        global_std        : float = 1.0,
        spatial_dims      : int = 3,
        cls_label_key     : str = ""):
        self.paths = list(npz_paths)
        if not self.paths:
            raise ValueError("LabeledPatchDataset got empty npz_paths.")
        self.patch = tuple(int(s) for s in patch_size)
        if len(self.patch) != 3:
            raise ValueError(
                f"patch_size must be 3D (D,H,W) for the seg probe dataset; "
                f"got {patch_size}.")
        self.spatial_dims = int(spatial_dims)
        if self.spatial_dims not in (2, 3):
            raise ValueError(
                f"spatial_dims must be 2 (2.5D folded) or 3; "
                f"got {self.spatial_dims}.")
        self.fold_2_5d = self.spatial_dims == 2
        self.intensity_min = float(intensity_min)
        self.intensity_max = float(intensity_max)
        self.normalize = str(normalize)
        self.global_mean = float(global_mean)
        self.global_std = float(global_std)
        self.spv = max(int(samples_per_volume), 1)
        self.cls_label_key = str(cls_label_key)
        logger.info(
            "LabeledPatchDataset (probe): %d volumes x %d samples = %d, patch=%s, "
            "spatial_dims=%d (%s)",
            len(self.paths), self.spv, len(self), self.patch,
            self.spatial_dims, "2.5D folded" if self.fold_2_5d else "3D")

    def __len__(self) -> int:
        return len(self.paths) * self.spv

    def _load(self, path: str):
        with _open_npz(path) as f:
            if "image" not in f.files or "label" not in f.files:
                raise KeyError(
                    f"probe npz {path!r} must have both 'image' and 'label' "
                    f"keys (keys={list(f.files)}).")
            img = preprocess_image(
                f["image"], self.intensity_min, self.intensity_max,
                self.normalize, self.global_mean, self.global_std,
                inplace=False)
            lbl = np.asarray(f["label"])
            cls_label = None
            if self.cls_label_key:
                if self.cls_label_key not in f.files:
                    raise KeyError(
                        f"probe npz {path!r} has no {self.cls_label_key!r} "
                        f"key (keys={list(f.files)}).")
                cls_label = np.asarray(f[self.cls_label_key])
        if img.shape != lbl.shape:
            raise ValueError(
                f"image/label shape mismatch in {path!r}: "
                f"{img.shape} vs {lbl.shape}.")
        return img, lbl, cls_label

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.paths[idx % len(self.paths)]
        img, lbl, cls_label = self._load(path)            # (D,H,W) fp32 / raw
        if img.ndim != 3:
            raise ValueError(
                f"probe expects 3D volume (D,H,W); got {img.shape} in {path!r}.")
        center = tuple(_rand_center(d, p) for d, p in zip(img.shape, self.patch))
        img_patch = _extract_cubic_patch(img, center, self.patch)  # (pD,pH,pW)
        lbl_patch = _extract_cubic_patch(lbl, center, self.patch)  # (pD,pH,pW)
        img_t = torch.from_numpy(img_patch.astype(np.float32, copy=False))
        lbl_t = torch.from_numpy(lbl_patch.astype(np.float32, copy=False))
        if not self.fold_2_5d:
            img_t = img_t.unsqueeze(0)                             # (1,pD,pH,pW)
            lbl_t = lbl_t.unsqueeze(0)
        # 2.5D：(pD,pH,pW) = (C=D, H, W)，深度折进通道。
        out = {"image": img_t, "label": lbl_t}
        if cls_label is not None:
            out["cls_label"] = torch.from_numpy(np.asarray(cls_label))
        return out


def build_ssl_dataloader(cfg) -> DataLoader:
    """按 ``cfg.data`` 构造 image-only 训练 dataloader（无 val：见 §0.5 在线探针）。

    依据 ``cfg.model.spatial_dims`` 自动选 3D / 2.5D 折叠输出。2.5D（折叠 D 进通道）
    仅支持单 FOV：要求 ``in_channels == patch_size[0]``（即 ``multi_res_scales==[1.0]``）；
    多 FOV 2.5D 需要数据增强级的多分辨率裁剪（segtask ``split_views_native_d``），
    暂不在 image-only SSL 通路内支持。
    """
    dc = cfg.data
    spatial_dims = int(cfg.model.spatial_dims)
    if spatial_dims == 2:
        D = int(dc.patch_size[0])
        in_ch = int(cfg.model.in_channels)
        if in_ch != D:
            raise ValueError(
                f"2.5D SSL pretraining supports single-FOV only: model.in_channels "
                f"({in_ch}) must equal patch_size[0] (D={D}), i.e. "
                f"data.multi_res_scales==[1.0]. Multi-FOV 2.5D needs augmentation-"
                f"level multi-resolution cropping, not available in image-only SSL.")
    paths = discover_image_npz(dc.npz_dir, dc.npz_suffix)
    ds = ImageOnlyPatchDataset(
        npz_paths         = paths,
        patch_size        = dc.patch_size,
        intensity_min     = dc.intensity_min,
        intensity_max     = dc.intensity_max,
        normalize         = dc.normalize,
        samples_per_volume= dc.samples_per_volume,
        spatial_dims      = spatial_dims)
    num_workers = int(dc.num_workers)
    kwargs: Dict[str, object] = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(dc.persistent_workers)
        kwargs["prefetch_factor"] = int(dc.prefetch_factor)
    return DataLoader(
        ds,
        batch_size      = int(dc.batch_size),
        shuffle         = True,
        num_workers     = num_workers,
        pin_memory      = bool(dc.pin_memory),
        drop_last       = True,
        **kwargs)


__all__ = [
    "ImageOnlyPatchDataset", "LabeledPatchDataset",
    "build_ssl_dataloader", "discover_image_npz",
]
