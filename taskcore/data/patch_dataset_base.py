"""单 npz 路径列表的 patch 数据集模板基类（P2c）。

cls / det 共用：路径列表、patch_mode、LRU 图像缓存、worker RNG、
``__len__``、卷索引方案（blocked vs interleaved）、验证 RNG + 网格覆盖。

子类只实现 ``__getitem__`` 内的样本组装（label / boxes / target 等）。
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset

from .dataset import VolumeCache, load_npz_image
from .patch_extract import normalize_patch_mode
from .sampling import WorkerNumpyRng

logger = logging.getLogger(__name__)

Center3 = Tuple[int, int, int]


class IndexScheme:
    """全局样本 idx → 卷 idx 的两种排布（与 DataLoader 遍历顺序对齐）。"""

    BLOCKED = "blocked"
    """同卷样本连续：``vol_idx = idx // spv``（cls，利于 LRU 命中）。"""

    INTERLEAVED = "interleaved"
    """卷交错：``vol_idx = idx % n_vols``（det/seg）。"""


class NpzPatchDatasetBase(Dataset):
    """npz 单路径列表 patch 数据集基类。"""

    def __init__(
        self,
        npz_paths: Sequence[str],
        patch_size: Sequence[int],
        patch_mode: str,
        samples_per_volume: int,
        spatial_dims: int,
        is_train: bool,
        seed: int,
        val_grid_coverage: bool,
        intensity_min: float,
        intensity_max: float,
        normalize: str,
        global_mean: float,
        global_std: float,
        fg_oversample_ratio: float,
        cache_enabled: bool,
        cache_max_volumes: int,
        *,
        z_sampling_mode: str = "safe",
        index_scheme: str = IndexScheme.INTERLEAVED,
        dataset_name: str = "NpzPatchDatasetBase",
    ) -> None:
        super().__init__()
        self.paths = list(npz_paths)
        if not self.paths:
            raise ValueError(f"{dataset_name} got empty npz_paths.")
        self.patch = tuple(int(s) for s in patch_size)
        if len(self.patch) != 3:
            raise ValueError(
                f"patch_size must be [D, H, W]; got {patch_size}.")
        self.mode = normalize_patch_mode(patch_mode)
        if spatial_dims not in (2, 3):
            raise ValueError(
                f"spatial_dims must be 2 or 3; got {spatial_dims}.")
        if index_scheme not in (IndexScheme.BLOCKED, IndexScheme.INTERLEAVED):
            raise ValueError(
                f"bad index_scheme: {index_scheme!r}; expected "
                f"'{IndexScheme.BLOCKED}' or '{IndexScheme.INTERLEAVED}'.")
        self.spatial_dims = int(spatial_dims)
        self.fold_2_5d = self.spatial_dims == 2
        self.intensity_min = float(intensity_min)
        self.intensity_max = float(intensity_max)
        self.normalize = str(normalize)
        self.global_mean = float(global_mean)
        self.global_std = float(global_std)
        self.spv = max(int(samples_per_volume), 1)
        self.is_train = bool(is_train)
        self.fg_ratio = (
            float(fg_oversample_ratio) if self.mode != "whole" else 0.0)
        self.seed = int(seed)
        self.val_grid_coverage = (
            bool(val_grid_coverage) and not self.is_train)
        self.z_sampling_mode = str(z_sampling_mode).lower()
        if self.z_sampling_mode not in ("safe", "legacy"):
            raise ValueError(
                f"z_sampling_mode must be 'safe' or 'legacy'; "
                f"got {z_sampling_mode!r}")
        self._index_scheme = index_scheme
        self._worker_rng = WorkerNumpyRng()
        self._img_cache = VolumeCache(cache_enabled, cache_max_volumes)

    def __len__(self) -> int:
        return len(self.paths) * self.spv

    def _rng(self) -> np.random.Generator:
        return self._worker_rng.get()

    def _vol_idx(self, idx: int) -> int:
        if self._index_scheme == IndexScheme.BLOCKED:
            return idx // self.spv
        return idx % len(self.paths)

    def _item_rng_and_cov(
        self, idx: int,
    ) -> Tuple[np.random.Generator, Optional[int]]:
        """训练 worker RNG；验证 ``(seed,idx)`` 确定性 RNG + 可选网格 j。"""
        from .sampling import (
            deterministic_idx_rng,
            val_coverage_j_blocked,
            val_coverage_j_interleaved,
        )

        if self.is_train:
            rng = self._rng()
        else:
            rng = deterministic_idx_rng(self.seed, idx)
        cov_j: Optional[int] = None
        if self.val_grid_coverage:
            if self._index_scheme == IndexScheme.BLOCKED:
                cov_j = val_coverage_j_blocked(idx, self.spv)
            else:
                cov_j = val_coverage_j_interleaved(idx, len(self.paths))
        return rng, cov_j

    def _load_image_cached(self, path: str) -> np.ndarray:
        """预处理 image；逐 worker LRU 缓存。"""
        img = self._img_cache.get(path)
        if img is not None:
            return img
        img = load_npz_image(
            path, self.intensity_min, self.intensity_max, self.normalize,
            self.global_mean, self.global_std)
        if img.ndim != 3:
            raise ValueError(
                f"expected 3D volume (D,H,W); got {img.shape} in {path!r}.")
        self._img_cache.put(path, img)
        return img


__all__ = [
    "Center3",
    "IndexScheme",
    "NpzPatchDatasetBase",
]
