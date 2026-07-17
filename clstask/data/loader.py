"""clstask DataLoader 工厂：npz 发现 → train/val 划分 → ClsPatchDataset。

划分：分类默认按标签**分层**（``cls.stratify_split``，保证每个标签组合在
训练/验证集都有代表，避免小类全落单侧致验证指标无定义）；关闭时回退
segtask 的纯随机划分（``train_val_split``）。npz 发现口径与 ssltask 一致
（递归扫 ``data.npz_dir``）。
"""

from __future__ import annotations

import glob
import logging
import os
from typing import List, Sequence, Tuple

import numpy as np
from torch.utils.data import DataLoader, DistributedSampler

from taskcore.config.core import Config as SegConfig
from taskcore.data.loader import (
    ValBatchShardSampler,
    scaled_num_workers,
    train_val_split,
)

from ..config import ClsConfig, resolve_num_classes
from .cls_dataset import (
    ClsPatchDataset,
    derive_volume_targets,
    load_label_table,
    match_table_to_paths,
)

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


def stratified_split(keys: Sequence[str], val_ratio: float,
                     seed: int) -> Tuple[List[int], List[int]]:
    """按标签层（key）分层的 train/val 划分。

    逐层内部确定性 shuffle 后按 ``val_ratio`` 切分；层内 ≥2 个样本时
    train/val 各至少分到 1 个（保证小类两侧都有代表）；单样本层归
    训练集。同 (keys, val_ratio, seed) 下结果确定。
    """
    rng = np.random.RandomState(seed)
    by_key: "dict[str, List[int]]" = {}
    for i, k in enumerate(keys):
        by_key.setdefault(str(k), []).append(i)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for k in sorted(by_key):
        idx = by_key[k]
        perm = rng.permutation(len(idx))
        n = len(idx)
        if n == 1:
            train_idx.append(idx[0])
            continue
        n_val = min(max(int(round(n * val_ratio)), 1), n - 1)
        for j, p in enumerate(perm):
            (val_idx if j < n_val else train_idx).append(idx[p])
    return sorted(train_idx), sorted(val_idx)


def _split_keys(paths: Sequence[str], cfg: SegConfig, cls: ClsConfig,
                table, npz_suffix: str) -> List[str]:
    """每卷的分层 key：table 源用显式标签，mask 源用整卷多热真值
    （:func:`derive_volume_targets`，优先读 meta.label_counts，开销小）。"""
    if table is not None:
        targets = match_table_to_paths(paths, table, npz_suffix)
        return [",".join(f"{int(round(float(x)))}" for x in
                         np.atleast_1d(np.asarray(t)))
                for t in targets]
    fg_values = [float(v) for v in (cfg.data.label_values[1:] if
                                    len(cfg.data.label_values) > 1 else [1.0])]
    vt = derive_volume_targets(paths, fg_values).numpy()
    return [",".join(str(int(x)) for x in row) for row in vt]


def build_cls_dataloaders(
    cfg: SegConfig, cls: ClsConfig,
    rank: int = 0, world_size: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    """按 ``(cfg, cls)`` 构造 train/val DataLoader。

    ``world_size > 1``（DDP）时：训练集用 ``DistributedSampler`` 不相交切分
    到各 rank（每 epoch 需在外层 ``set_epoch``）；验证集用
    ``ValBatchShardSampler`` 按 batch 块不相交切分，指标在训练器内跨 rank
    聚合。单进程路径零变化。"""
    dc = cfg.data
    paths = discover_npz(dc.npz_dir, dc.npz_suffix)
    num_classes = resolve_num_classes(cls, cfg)

    table = None
    if cls.label_source == "table":
        table = load_label_table(cls.label_table, num_classes, cls.multi_label)

    if cls.stratify_split:
        keys = _split_keys(paths, cfg, cls, table, dc.npz_suffix)
        train_idx, val_idx = stratified_split(
            keys, dc.val_ratio, dc.split_seed)
        n_strata = len(set(keys))
        logger.info("stratified split: %d strata over %d volumes",
                    n_strata, len(paths))
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
    eff_num_workers = scaled_num_workers(
        dc.num_workers, world_size,
        bool(cfg.train.ddp_scale_dataloader_per_rank))
    common = dict(
        num_workers=eff_num_workers,
        pin_memory=dc.pin_memory,
        persistent_workers=dc.persistent_workers and eff_num_workers > 0)
    if eff_num_workers > 0:
        common["prefetch_factor"] = dc.prefetch_factor
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=True)
        train_loader = DataLoader(
            train_ds, batch_size=dc.batch_size, sampler=train_sampler,
            drop_last=True, **common)
        val_loader = DataLoader(
            val_ds, batch_size=dc.batch_size,
            sampler=ValBatchShardSampler(
                len(val_ds), dc.batch_size, rank, world_size),
            drop_last=False, **common)
        logger.info("clstask DDP samplers: rank=%d/%d, ~%d train samples/rank",
                    rank, world_size, len(train_sampler))
    else:
        train_loader = DataLoader(
            train_ds, batch_size=dc.batch_size, shuffle=True, drop_last=False,
            **common)
        val_loader = DataLoader(
            val_ds, batch_size=dc.batch_size, shuffle=False, drop_last=False,
            **common)
    logger.info("clstask loaders: %d train / %d val volume(s), K=%d",
                len(train_paths), len(val_paths), num_classes)
    return train_loader, val_loader


__all__ = ["discover_npz", "stratified_split", "build_cls_dataloaders"]
