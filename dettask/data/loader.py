"""dettask DataLoader 工厂：npz 发现 → train/val 划分 → DetPatchDataset。"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from taskcore.config.core import Config as SegConfig
from taskcore.data.dataset import open_npz, derive_volume_targets
from taskcore.data.loader import (
    assemble_train_val_loaders,
    discover_npz_recursive as discover_npz,
    stratified_split_by_key as stratified_split,
    train_val_split,
)

from ..config import DetConfig
from .det_dataset import DetPatchDataset, det_collate

logger = logging.getLogger(__name__)


def _det_split_keys(paths: List[str], fg_values: List[float]) -> List[str]:
    """每卷的分层 key = 类存在集合（排序去重）。

    'boxes' 键（小数组）直接取 cls 列；mask 源复用 clstask
    ``derive_volume_targets``（优先读 meta.label_counts，免整卷解码），
    存在性口径与框派生一致（连通域必属某前景类）。"""
    keys: List[str] = [""] * len(paths)
    mask_pos: List[int] = []
    for i, p in enumerate(paths):
        with open_npz(p) as f:
            if "boxes" in f.files:
                cls = np.asarray(f["boxes"], np.float32).reshape(-1, 7)[:, 6]
                present = sorted(set(int(c) for c in cls))
                keys[i] = ",".join(map(str, present)) if present else "empty"
            else:
                mask_pos.append(i)
    if mask_pos:
        vt = derive_volume_targets([paths[i] for i in mask_pos],
                                   fg_values).numpy()
        for j, i in enumerate(mask_pos):
            present = [str(k) for k, v in enumerate(vt[j]) if v > 0]
            keys[i] = ",".join(present) if present else "empty"
    return keys


def build_det_dataloaders(
    cfg: SegConfig, det: DetConfig,
    rank: int = 0, world_size: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    """``world_size > 1``（DDP）时：训练集用 ``DistributedSampler`` 不相交
    切分到各 rank（每 epoch 需在外层 ``set_epoch``）；验证集用
    ``ValBatchShardSampler`` 按 batch 块不相交切分，指标在训练器内跨 rank
    聚合。单进程路径零变化。"""
    dc = cfg.data
    paths = discover_npz(dc.npz_dir, dc.npz_suffix)
    fg_values_split = [float(v) for v in (dc.label_values[1:] if
                                          len(dc.label_values) > 1 else [1.0])]
    if det.stratify_split:
        keys = _det_split_keys(paths, fg_values_split)
        train_idx, val_idx = stratified_split(
            keys, dc.val_ratio, dc.split_seed)
        logger.info("stratified split: %d strata over %d volumes",
                    len(set(keys)), len(paths))
    else:
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
    train_loader, val_loader = assemble_train_val_loaders(
        train_ds, val_ds, cfg, rank=rank, world_size=world_size,
        collate_fn=det_collate, log_prefix="dettask",
        train_drop_last=False)
    logger.info("dettask loaders: %d train / %d val volume(s)",
                len(train_paths), len(val_paths))
    return train_loader, val_loader


__all__ = ["build_det_dataloaders"]
