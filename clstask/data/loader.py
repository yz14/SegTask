"""clstask DataLoader 工厂：npz 发现 → train/val 划分 → ClsPatchDataset。

划分：分类默认按标签**分层**（``cls.stratify_split``，保证每个标签组合在
训练/验证集都有代表，避免小类全落单侧致验证指标无定义）；关闭时回退
segtask 的纯随机划分（``train_val_split``）。npz 发现口径与 ssltask 一致
（递归扫 ``data.npz_dir``）。
"""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import numpy as np

from taskcore.config.core import Config as SegConfig
from taskcore.data.loader import (
    assemble_train_val_loaders,
    discover_npz_recursive,
    stratified_split_by_key,
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


# 通用件已上提 taskcore.data.loader；此处保留旧名别名（dettask/测试/外部脚本兼容）。
discover_npz = discover_npz_recursive
stratified_split = stratified_split_by_key


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
    train_loader, val_loader = assemble_train_val_loaders(
        train_ds, val_ds, cfg, rank=rank, world_size=world_size,
        log_prefix="clstask", train_drop_last=False)
    logger.info("clstask loaders: %d train / %d val volume(s), K=%d",
                len(train_paths), len(val_paths), num_classes)
    return train_loader, val_loader


__all__ = ["discover_npz", "stratified_split", "build_cls_dataloaders"]
