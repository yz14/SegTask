"""dettask DataLoader 工厂：npz 发现 → train/val 划分 → DetPatchDataset。"""

from __future__ import annotations

import logging
from typing import List, Tuple

from torch.utils.data import DataLoader

from segtask_v1.config import Config as SegConfig
from segtask_v1.data.loader import train_val_split

from clstask.data.loader import discover_npz

from ..config import DetConfig
from .det_dataset import DetPatchDataset, det_collate

logger = logging.getLogger(__name__)


def build_det_dataloaders(
    cfg: SegConfig, det: DetConfig) -> Tuple[DataLoader, DataLoader]:
    dc = cfg.data
    paths = discover_npz(dc.npz_dir, dc.npz_suffix)
    train_idx, val_idx = train_val_split(
        len(paths), dc.val_ratio, dc.split_seed)
    train_paths = [paths[i] for i in train_idx]
    val_paths = [paths[i] for i in val_idx]
    if not train_paths or not val_paths:
        raise RuntimeError(
            f"train/val split degenerate: {len(train_paths)} train / "
            f"{len(val_paths)} val from {len(paths)} volumes "
            f"(val_ratio={dc.val_ratio}).")

    fg_values = [float(v) for v in (dc.label_values[1:] if
                                    len(dc.label_values) > 1 else [1.0])]
    spatial_dims = int(cfg.model.spatial_dims)
    ac = cfg.augment
    # 空间翻转在 dataset 内施加（框联动）；AugConfig 的 flip 轴为张量轴
    # (B,C,D,H,W) 的 2/3/4，映射到空间轴 (z,y,x) = (0,1,2)。
    flip_prob = float(ac.random_flip_prob) if ac.enabled else 0.0
    flip_axes = [int(a) - 2 for a in ac.random_flip_axes]

    def _mk(split_paths: List[str], is_train: bool) -> DetPatchDataset:
        return DetPatchDataset(
            npz_paths=split_paths,
            patch_size=dc.patch_size,
            fg_values=fg_values,
            patch_mode=dc.patch_mode,
            boxes_from_mask_ok=det.boxes_from_mask,
            min_box_voxels=det.min_box_voxels,
            intensity_min=dc.intensity_min,
            intensity_max=dc.intensity_max,
            normalize=dc.normalize,
            global_mean=dc.global_mean,
            global_std=dc.global_std,
            samples_per_volume=(dc.samples_per_volume if is_train
                                else max(dc.samples_per_volume // 2, 1)),
            spatial_dims=spatial_dims,
            is_train=is_train,
            fg_oversample_ratio=(max(dc.foreground_oversample_ratio, 0.5)
                                 if is_train else 0.0),
            seed=dc.split_seed,
            aug_flip_prob=flip_prob if is_train else 0.0,
            aug_flip_axes=flip_axes,
            val_grid_coverage=dc.val_grid_coverage,
            cache_enabled=dc.cache_mode == "memory",
            cache_max_volumes=dc.cache_max_volumes)

    train_ds = _mk(train_paths, True)
    val_ds = _mk(val_paths, False)
    common = dict(
        num_workers=dc.num_workers,
        pin_memory=dc.pin_memory,
        collate_fn=det_collate,
        persistent_workers=dc.persistent_workers and dc.num_workers > 0)
    if dc.num_workers > 0:
        common["prefetch_factor"] = dc.prefetch_factor
    train_loader = DataLoader(
        train_ds, batch_size=dc.batch_size, shuffle=True, drop_last=False,
        **common)
    val_loader = DataLoader(
        val_ds, batch_size=dc.batch_size, shuffle=False, drop_last=False,
        **common)
    logger.info("dettask loaders: %d train / %d val volume(s)",
                len(train_paths), len(val_paths))
    return train_loader, val_loader


__all__ = ["build_det_dataloaders"]
