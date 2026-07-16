"""clstask DataLoader 工厂：npz 发现 → train/val 划分 → ClsPatchDataset。

复用 segtask 的划分工具（``train_val_split``），npz 发现口径与 ssltask 一致
（递归扫 ``data.npz_dir``）。
"""

from __future__ import annotations

import glob
import logging
import os
from typing import List, Tuple

from torch.utils.data import DataLoader

from segtask_v1.config import Config as SegConfig
from segtask_v1.data.loader import train_val_split

from ..config import ClsConfig, resolve_num_classes
from .cls_dataset import ClsPatchDataset, load_label_table, match_table_to_paths

logger = logging.getLogger(__name__)


def discover_npz(npz_dir: str, npz_suffix: str = ".npz") -> List[str]:
    """递归发现 ``npz_dir`` 下所有 ``*{npz_suffix}``，按路径排序。"""
    if not npz_dir or not os.path.isdir(npz_dir):
        raise FileNotFoundError(
            f"data.npz_dir not found: {npz_dir!r}. clstask trains on "
            f"pre-generated npz packages (image [+ label]).")
    paths = sorted(glob.glob(
        os.path.join(npz_dir, "**", f"*{npz_suffix}"), recursive=True))
    if not paths:
        raise RuntimeError(f"No '*{npz_suffix}' found under {npz_dir!r}.")
    return paths


def build_cls_dataloaders(
    cfg: SegConfig, cls: ClsConfig) -> Tuple[DataLoader, DataLoader]:
    """按 ``(cfg, cls)`` 构造 train/val DataLoader。"""
    dc = cfg.data
    paths = discover_npz(dc.npz_dir, dc.npz_suffix)
    num_classes = resolve_num_classes(cls, cfg)

    table = None
    if cls.label_source == "table":
        table = load_label_table(cls.label_table, num_classes, cls.multi_label)

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

    def _mk(split_paths: List[str], is_train: bool) -> ClsPatchDataset:
        targets = (match_table_to_paths(split_paths, table, dc.npz_suffix)
                   if table is not None else None)
        return ClsPatchDataset(
            npz_paths=split_paths,
            patch_size=dc.patch_size,
            num_classes=num_classes,
            patch_mode=dc.patch_mode,
            label_granularity=cls.label_granularity,
            label_source=cls.label_source,
            table_targets=targets,
            fg_values=fg_values,
            intensity_min=dc.intensity_min,
            intensity_max=dc.intensity_max,
            normalize=dc.normalize,
            global_mean=dc.global_mean,
            global_std=dc.global_std,
            # 验证每卷 patch 数与推理铺格上限同一来源（选模与部署同口径）。
            samples_per_volume=(dc.samples_per_volume if is_train
                                else max(int(cls.eval_patches_per_volume), 1)),
            spatial_dims=spatial_dims,
            is_train=is_train,
            fg_oversample_ratio=(dc.foreground_oversample_ratio
                                 if is_train else 0.0),
            seed=dc.split_seed,
            gpu_augment=(bool(cfg.augment.enabled) and is_train),
            val_grid_coverage=(bool(dc.val_grid_coverage)
                               and not is_train),
            cache_enabled=(dc.cache_mode == "memory"),
            cache_max_volumes=dc.cache_max_volumes)

    train_ds = _mk(train_paths, True)
    val_ds = _mk(val_paths, False)
    common = dict(
        num_workers=dc.num_workers,
        pin_memory=dc.pin_memory,
        persistent_workers=dc.persistent_workers and dc.num_workers > 0)
    if dc.num_workers > 0:
        common["prefetch_factor"] = dc.prefetch_factor
    train_loader = DataLoader(
        train_ds, batch_size=dc.batch_size, shuffle=True, drop_last=False,
        **common)
    val_loader = DataLoader(
        val_ds, batch_size=dc.batch_size, shuffle=False, drop_last=False,
        **common)
    logger.info("clstask loaders: %d train / %d val volume(s), K=%d",
                len(train_paths), len(val_paths), num_classes)
    return train_loader, val_loader


__all__ = ["discover_npz", "build_cls_dataloaders"]
