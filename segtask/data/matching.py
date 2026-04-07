"""Data matching: find paired image-label files and split into train/val/test.

Handles the case where image and label directories don't have 1:1 correspondence
by matching on the common subject ID extracted from filenames.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SampleRecord:
    """A single matched image-label pair."""

    subject_id: str
    image_path: str
    label_path: str
    split: str = ""  # "train", "val", "test"


def _extract_subject_id(filename: str) -> str:
    """Extract subject ID from filename by removing common suffixes.

    Examples:
        s0000.nii.gz     -> s0000
        s0001-seg.nii.gz -> s0001
        patient_01.nii   -> patient_01
    """
    stem = filename
    # Remove .nii.gz or .nii
    for ext in [".nii.gz", ".nii"]:
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break

    # Remove common label suffixes like -seg, _seg, _label, _mask
    for suffix in ["-seg", "_seg", "-label", "_label", "-mask", "_mask"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    return stem


def match_data(
    image_dir: str, label_dir: str,
    image_suffix: str = ".nii.gz", label_suffix: str = ".nii.gz") -> List[SampleRecord]:
    """Match image files with label files based on subject ID.

    Args:
        image_dir: Directory containing image files.
        label_dir: Directory containing label files.
        image_suffix: Image file extension filter.
        label_suffix: Label file extension filter.

    Returns:
        List of matched SampleRecord objects.
    """
    img_dir, lbl_dir = Path(image_dir), Path(label_dir)

    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not lbl_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {lbl_dir}")

    # Build lookup tables: subject_id -> filepath
    img_map: Dict[str, Path] = {}
    for f in sorted(img_dir.iterdir()):
        if f.name.endswith(image_suffix):
            sid = _extract_subject_id(f.name)
            img_map[sid] = f

    lbl_map: Dict[str, Path] = {}
    for f in sorted(lbl_dir.iterdir()):
        if f.name.endswith(label_suffix):
            sid = _extract_subject_id(f.name)
            lbl_map[sid] = f

    # Match
    matched_ids = sorted(set(img_map.keys()) & set(lbl_map.keys()))

    if not matched_ids:
        raise RuntimeError(
            f"No matched pairs found between {img_dir} ({len(img_map)} images) "
            f"and {lbl_dir} ({len(lbl_map)} labels). "
            "Check filename patterns and suffixes.")

    records = []
    for sid in matched_ids:
        records.append(
            SampleRecord(
                subject_id=sid,
                image_path=str(img_map[sid]),
                label_path=str(lbl_map[sid])))

    logger.info(
        "Data matching: %d images, %d labels → %d matched pairs",
        len(img_map),
        len(lbl_map),
        len(records))

    # Report unmatched
    unmatched_img = set(img_map.keys()) - set(lbl_map.keys())
    unmatched_lbl = set(lbl_map.keys()) - set(img_map.keys())
    if unmatched_img:
        logger.warning("%d images without labels: %s ...", len(unmatched_img),
                       list(unmatched_img)[:5])
    if unmatched_lbl:
        logger.warning("%d labels without images: %s ...", len(unmatched_lbl),
                       list(unmatched_lbl)[:5])

    return records


def split_dataset(
    records: List[SampleRecord], method: str = "random",
    meta_csv: str = "", val_ratio: float = 0.15,
    test_ratio: float = 0.15, seed: int = 42) -> Tuple[List[SampleRecord], List[SampleRecord], List[SampleRecord]]:
    """Split records into train/val/test sets.

    Args:
        records: List of SampleRecord.
        method: "meta" (use CSV split column) or "random".
        meta_csv: Path to meta.csv (required if method="meta").
        val_ratio: Validation set ratio (for random split).
        test_ratio: Test set ratio (for random split).
        seed: Random seed.

    Returns:
        (train_records, val_records, test_records)
    """
    if method == "meta" and meta_csv:
        return _split_by_meta(records, meta_csv)
    else:
        return _split_random(records, val_ratio, test_ratio, seed)


def _split_by_meta(
    records: List[SampleRecord], meta_csv: str
) -> Tuple[List[SampleRecord], List[SampleRecord], List[SampleRecord]]:
    """Split using the 'split' column from meta.csv."""
    csv_path = Path(meta_csv)
    if not csv_path.exists():
        logger.warning("Meta CSV not found: %s. Falling back to random split.", csv_path)
        return _split_random(records, 0.15, 0.15, 42)

    # Auto-detect delimiter
    with open(csv_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    sep = ";" if ";" in first_line else ","

    df = pd.read_csv(csv_path, sep=sep)

    if "image_id" not in df.columns or "split" not in df.columns:
        logger.warning(
            "Meta CSV missing 'image_id' or 'split' columns. "
            "Falling back to random split."
        )
        return _split_random(records, 0.15, 0.15, 42)

    split_map = dict(zip(df["image_id"].astype(str), df["split"].astype(str)))

    train, val, test = [], [], []
    unassigned = []

    for rec in records:
        s = split_map.get(rec.subject_id, "")
        if s == "train":
            rec.split = "train"
            train.append(rec)
        elif s == "val":
            rec.split = "val"
            val.append(rec)
        elif s == "test":
            rec.split = "test"
            test.append(rec)
        else:
            unassigned.append(rec)

    # Assign unassigned to train
    for rec in unassigned:
        rec.split = "train"
        train.append(rec)

    if unassigned:
        logger.warning(
            "%d subjects not found in meta CSV, assigned to train.", len(unassigned)
        )

    # If val or test is empty, split from train
    if not val and train:
        n_val = max(1, int(len(train) * 0.15))
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(train))
        val = [train[i] for i in indices[:n_val]]
        train = [train[i] for i in indices[n_val:]]
        for r in val:
            r.split = "val"
        logger.info("No val in meta CSV; split %d from train.", n_val)

    logger.info(
        "Split (meta): train=%d, val=%d, test=%d",
        len(train), len(val), len(test),
    )
    return train, val, test


def _split_random(
    records: List[SampleRecord], val_ratio: float,
    test_ratio: float, seed: int) -> Tuple[List[SampleRecord], List[SampleRecord], List[SampleRecord]]:
    """Random stratified split."""
    rng = np.random.RandomState(seed)
    n = len(records)
    indices = rng.permutation(n)

    n_test = max(1, int(n * test_ratio)) if test_ratio > 0 else 0
    n_val  = max(1, int(n * val_ratio)) if val_ratio > 0 else 0

    test_idx  = indices[:n_test]
    val_idx   = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]

    train = [records[i] for i in train_idx]
    val   = [records[i] for i in val_idx]
    test  = [records[i] for i in test_idx]

    for r in train:
        r.split = "train"
    for r in val:
        r.split = "val"
    for r in test:
        r.split = "test"

    logger.info(
        "Split (random, seed=%d): train=%d, val=%d, test=%d",
        seed, len(train), len(val), len(test))
    return train, val, test
