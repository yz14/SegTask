"""DataLoader factory + train/val split.

Scans the data directories, splits into train/val, and creates DataLoaders.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import DataLoader

from ..config import Config
from .dataset import SegDataset3D, load_nifti

logger = logging.getLogger(__name__)


def discover_samples(
    image_dir: str,
    label_dir: str,
    image_suffix: str = ".nii.gz",
    label_suffix: str = ".nii.gz",
) -> Tuple[List[str], List[str]]:
    """Discover matched image-label pairs from directories.

    Returns:
        (image_paths, label_paths) sorted by filename.
    """
    img_dir = Path(image_dir)
    lbl_dir = Path(label_dir)
    assert img_dir.is_dir(), f"Image dir not found: {img_dir}"
    assert lbl_dir.is_dir(), f"Label dir not found: {lbl_dir}"

    img_files = {p.name: p for p in sorted(img_dir.glob(f"*{image_suffix}"))}
    lbl_files = {p.name: p for p in sorted(lbl_dir.glob(f"*{label_suffix}"))}

    # Match by filename
    common = sorted(set(img_files.keys()) & set(lbl_files.keys()))
    if not common:
        raise ValueError(
            f"No matched pairs found in {img_dir} and {lbl_dir}. "
            f"Images: {len(img_files)}, Labels: {len(lbl_files)}"
        )

    image_paths = [str(img_files[n]) for n in common]
    label_paths = [str(lbl_files[n]) for n in common]
    logger.info("Found %d matched image-label pairs.", len(common))
    return image_paths, label_paths


def detect_label_values(label_paths: List[str], max_scan: int = 5) -> List[int]:
    """Auto-detect unique label values from a subset of label files.

    Returns sorted list starting with background (0).
    """
    all_labels = set()
    for path in label_paths[:max_scan]:
        lbl = load_nifti(path)
        unique = np.unique(np.round(lbl).astype(np.int32)).tolist()
        all_labels.update(unique)
    result = sorted(all_labels)
    logger.info("Auto-detected label values: %s", result)
    return result


def train_val_split(n: int, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    """Random train/val split by index."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n).tolist()
    n_val = max(1, int(n * val_ratio))
    return indices[n_val:], indices[:n_val]


def build_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders from config.

    Steps:
    1. Discover image-label pairs
    2. Auto-detect label values if not specified
    3. Split into train/val
    4. Create SegDataset3D + DataLoader for each
    """
    dc = cfg.data

    # Discover samples, 数据配对
    image_paths, label_paths = discover_samples(
        dc.image_dir, dc.label_dir, dc.image_suffix, dc.label_suffix)

    # Auto-detect labels if needed, 标签值确认
    if not dc.label_values:
        dc.label_values = detect_label_values(label_paths)
        dc.num_classes = len(dc.label_values)
        cfg.sync()
    logger.info("Label values: %s, num_classes: %d, num_fg: %d",
                dc.label_values, dc.num_classes, cfg.num_fg_classes)

    # Split
    train_idx, val_idx = train_val_split(len(image_paths), dc.val_ratio, dc.split_seed)
    logger.info("Split: %d train, %d val", len(train_idx), len(val_idx))

    cache = dc.cache_mode == "memory"
    common_kwargs = dict(
        label_values=dc.label_values,
        patch_size=tuple(dc.patch_size),
        intensity_min=dc.intensity_min,
        intensity_max=dc.intensity_max,
        normalize=dc.normalize,
        global_mean=dc.global_mean,
        global_std=dc.global_std,
        cache_enabled=cache)

    train_ds = SegDataset3D(
        image_paths=[image_paths[i] for i in train_idx],
        label_paths=[label_paths[i] for i in train_idx],
        foreground_oversample_ratio=dc.foreground_oversample_ratio,
        samples_per_volume=dc.samples_per_volume,
        is_train=True,
        **common_kwargs)
    val_ds = SegDataset3D(
        image_paths=[image_paths[i] for i in val_idx],
        label_paths=[label_paths[i] for i in val_idx],
        foreground_oversample_ratio=0.0,
        samples_per_volume=max(dc.samples_per_volume // 2, 1),
        is_train=False,
        **common_kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=dc.batch_size,
        shuffle=True,
        num_workers=dc.num_workers,
        pin_memory=dc.pin_memory,
        drop_last=True)
    val_loader = DataLoader(
        val_ds,
        batch_size=dc.batch_size,
        shuffle=False,
        num_workers=dc.num_workers,
        pin_memory=dc.pin_memory,
        drop_last=False)

    return train_loader, val_loader