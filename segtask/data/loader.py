"""DataLoader factory: builds train/val/test loaders from config.

Handles dataset creation, sampler selection, and DataLoader construction
for all data modes (2D, 2.5D, 3D).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from ..config import Config
from .dataset import load_nifti, SegDataset2D, SegDataset3D
from .matching import SampleRecord, match_data, split_dataset
from .sampler import ClassBalancedSampler, compute_sample_weights_from_labels

logger = logging.getLogger(__name__)


def _detect_label_values(records: List[SampleRecord], max_samples: int = 5) -> List[int]:
    """Auto-detect unique label values from a few samples."""
    all_values = set()
    for rec in records[:max_samples]:
        lbl = load_nifti(rec.label_path, dtype=np.float32)
        unique = np.unique(np.round(lbl).astype(np.int32))
        all_values.update(unique.tolist())

    values = sorted(all_values)
    logger.info("Auto-detected label values: %s", values)
    return values


def _create_dataset(
    records: List[SampleRecord],
    cfg: Config,
    is_train: bool = True,
):
    """Create the appropriate dataset based on config data mode."""
    dc = cfg.data

    common_kwargs = dict(
        records=records,
        label_values=dc.label_values,
        intensity_min=dc.intensity_min,
        intensity_max=dc.intensity_max,
        normalize=dc.normalize,
        global_mean=dc.global_mean,
        global_std=dc.global_std,
        cache_enabled=(dc.cache_mode == "memory"),
        foreground_oversample_ratio=dc.foreground_oversample_ratio if is_train else 0.0,
        is_train=is_train,
    )

    # Remove is_train from common_kwargs — SegDataset3D takes it as a separate param
    kw_3d = {k: v for k, v in common_kwargs.items() if k != "is_train"}

    if dc.mode == "2d":
        return SegDataset2D(crop_size=tuple(dc.crop_size), **common_kwargs)
    elif dc.mode == "2.5d":
        # 2.5D directly reuses SegDataset3D — just set D = total_slices
        total_slices = 2 * dc.num_slices_per_side + 1
        patch_25d = (total_slices, dc.crop_size[0], dc.crop_size[1])
        # Estimate samples_per_volume from first record's depth for proper coverage
        first_vol = load_nifti(records[0].image_path, dtype=np.float32)
        avg_depth = first_vol.shape[0]  # after transpose: (D, H, W)
        spv = max(1, avg_depth // total_slices) if is_train else 1
        return SegDataset3D(
            patch_size=patch_25d,
            samples_per_volume=spv,
            is_train=is_train,
            **kw_3d,
        )
    elif dc.mode == "3d":
        return SegDataset3D(
            patch_size=tuple(dc.patch_size),
            samples_per_volume=4 if is_train else 1,
            is_train=is_train,
            **kw_3d,
        )
    else:
        raise ValueError(f"Unknown data mode: {dc.mode}")


def build_dataloaders(
    cfg: Config,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Build train, val, and test DataLoaders from config.

    Steps:
    1. Match image-label pairs
    2. Auto-detect label values if not specified
    3. Split into train/val/test
    4. Create datasets
    5. Optionally compute class-balanced sampling weights
    6. Build DataLoaders

    Args:
        cfg: Full configuration.

    Returns:
        (train_loader, val_loader, test_loader)
        test_loader is None if no test split exists.
    """
    dc = cfg.data

    # 1. Match data
    records = match_data(
        image_dir=dc.image_dir,
        label_dir=dc.label_dir,
        image_suffix=dc.image_suffix,
        label_suffix=dc.label_suffix)

    # 2. Auto-detect label values
    if not dc.label_values:
        dc.label_values = _detect_label_values(records)
    if dc.num_classes == 0:
        dc.num_classes = len(dc.label_values)
    cfg.sync()

    logger.info("Label values: %s, num_classes: %d", dc.label_values, dc.num_classes)

    # 3. Split
    train_recs, val_recs, test_recs = split_dataset(
        records,
        method=dc.split_method,
        meta_csv=dc.meta_csv,
        val_ratio=dc.val_ratio,
        test_ratio=dc.test_ratio,
        seed=dc.split_seed)

    # 4. Create datasets
    train_ds = _create_dataset(train_recs, cfg, is_train=True)
    val_ds = _create_dataset(val_recs, cfg, is_train=False)
    test_ds = _create_dataset(test_recs, cfg, is_train=False) if test_recs else None

    # 5. Sampler
    if dc.class_sample_weights:
        # Use provided class weights for balanced sampling
        sample_weights = compute_sample_weights_from_labels(
            train_ds, dc.num_classes, dc.class_sample_weights
        )
        train_sampler = ClassBalancedSampler(sample_weights, num_samples=len(train_ds))
    else:
        train_sampler = RandomSampler(train_ds)

    # 6. Build loaders
    loader_kwargs = dict(
        batch_size=dc.batch_size,
        num_workers=dc.num_workers,
        pin_memory=dc.pin_memory,
        prefetch_factor=dc.prefetch_factor if dc.num_workers > 0 else None,
        persistent_workers=dc.num_workers > 0,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_ds,
        sampler=train_sampler,
        drop_last=True,
        **{k: v for k, v in loader_kwargs.items() if k != "drop_last"},
    )
    val_loader = DataLoader(
        val_ds,
        sampler=SequentialSampler(val_ds),
        **loader_kwargs,
    )
    test_loader = None
    if test_ds is not None and len(test_ds) > 0:
        test_loader = DataLoader(
            test_ds,
            sampler=SequentialSampler(test_ds),
            **loader_kwargs,
        )

    logger.info(
        "DataLoaders: train=%d batches, val=%d batches, test=%s batches",
        len(train_loader),
        len(val_loader),
        len(test_loader) if test_loader else "N/A",
    )

    return train_loader, val_loader, test_loader
