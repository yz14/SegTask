"""DataLoader factory + train/val split.

Scans the data directories, splits into train/val, and creates DataLoaders.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader

from ..config import Config
from .dataset import (SegDataset3D, SegDataset3DCubic, SegDataset3DWhole, load_nifti)

logger = logging.getLogger(__name__)


def _load_exclude_pids(exclude_list: str) -> set:
    """Load a set of pids / stems from a plain-text exclude list.

    Format: one pid per line. Blank lines and lines starting with ``#`` are
    ignored. A trailing ``.nii.gz`` / ``.nii`` is tolerated (and stripped)
    so users can paste raw filenames too. Returns an empty set when the
    path is empty or missing (with a warning in the latter case).
    """
    if not exclude_list:
        return set()
    p = Path(exclude_list)
    if not p.is_file():
        logger.warning("`data.exclude_list` set but file not found: %s — "
                       "no samples will be excluded.", p)
        return set()
    pids = set()
    with open(p, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            # Tolerate full filenames in the list.
            for suf in (".nii.gz", ".nii"):
                if s.endswith(suf):
                    s = s[: -len(suf)]
                    break
            pids.add(s)
    logger.info("Loaded %d pid(s) to exclude from %s", len(pids), p)
    return pids


def _filter_by_exclude(
    image_paths: List[str],
    label_paths: List[str],
    image_suffix: str,
    exclude_pids: set) -> Tuple[List[str], List[str], List[int]]:
    """Drop pairs whose image stem is in ``exclude_pids``.

    Returns the filtered ``(image_paths, label_paths, keep_idx)`` where
    ``keep_idx`` indexes back into the *original* lists — callers use this
    to also re-align any companion list (e.g. bbox paths built later).
    """
    if not exclude_pids:
        return image_paths, label_paths, list(range(len(image_paths)))

    keep_idx: List[int] = []
    dropped: List[str] = []
    for i, img_path in enumerate(image_paths):
        name = Path(img_path).name
        if name.endswith(image_suffix):
            base = name[: -len(image_suffix)]
        else:
            base = Path(name).stem
        if base in exclude_pids:
            dropped.append(base)
        else:
            keep_idx.append(i)

    if dropped:
        head = ", ".join(dropped[:10])
        more = f", ... (+{len(dropped) - 10} more)" if len(dropped) > 10 else ""
        logger.warning(
            "Excluded %d/%d sample(s) via `data.exclude_list`: [%s%s]",
            len(dropped), len(image_paths), head, more)

    image_paths = [image_paths[i] for i in keep_idx]
    label_paths = [label_paths[i] for i in keep_idx]
    return image_paths, label_paths, keep_idx


def discover_samples(
    image_dir: str, label_dir: str, image_suffix: str=".nii.gz", label_suffix: str=".nii.gz") -> Tuple[List[str], List[str]]:
    """Discover matched image-label pairs from directories.

    Returns:
        (image_paths, label_paths) sorted by filename.
    """
    img_dir, lbl_dir = Path(image_dir), Path(label_dir)
    assert img_dir.is_dir(), f"Image dir not found: {img_dir}"
    assert lbl_dir.is_dir(), f"Label dir not found: {lbl_dir}"

    img_files = {p.name: p for p in sorted(img_dir.glob(f"*{image_suffix}"))}
    lbl_files = {p.name: p for p in sorted(lbl_dir.glob(f"*{label_suffix}"))}

    # Match by filename
    common = sorted(set(img_files.keys()) & set(lbl_files.keys()))
    if not common:
        raise ValueError(
            f"No matched pairs found in {img_dir} and {lbl_dir}. "
            f"Images: {len(img_files)}, Labels: {len(lbl_files)}")

    image_paths = [str(img_files[n]) for n in common]
    label_paths = [str(lbl_files[n]) for n in common]
    logger.info("Found %d matched image-label pairs.", len(common))
    return image_paths, label_paths  # 匹配好的


def _match_per_sample_paths(
    image_paths: List[str],
    src_dir: str,
    image_suffix: str,
    out_suffix: str,
    kind: str) -> List[str]:
    """Generic 1:1 per-sample file matcher by *base name*.

    Used by both ``match_bbox_paths`` and ``match_region_weight_paths``
    — strips ``image_suffix`` from each image filename, appends
    ``out_suffix``, and looks the result up under ``src_dir``. Missing
    matches raise ``FileNotFoundError`` listing up to the first five
    offenders (strong contract: silently dropping samples biases batch
    statistics and hides data-layout bugs).

    ``kind`` is purely an English label used in log / error messages
    ("BBox", "RegionWeight", ...) so callers don't need to format their
    own error strings.
    """
    sdir = Path(src_dir)
    assert sdir.is_dir(), f"{kind} dir not found: {sdir}"

    out: List[str] = []
    missing: List[str] = []
    for img_path in image_paths:
        name = Path(img_path).name
        # Strip image_suffix to get the base name; tolerate names that
        # don't end with the suffix (use stem as fallback).
        if name.endswith(image_suffix):
            base = name[: -len(image_suffix)]
        else:
            base = Path(name).stem
        cand = sdir / f"{base}{out_suffix}"
        if not cand.is_file():
            missing.append(str(cand))
        else:
            out.append(str(cand))

    if missing:
        # Show only the first few to keep the message readable.
        head = "\n  ".join(missing[:5])
        more = f"\n  ... ({len(missing) - 5} more)" if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"{kind} files not found for {len(missing)}/{len(image_paths)} "
            f"samples under {sdir}:\n  {head}{more}")

    logger.info("Matched %d %s files under %s.", len(out), kind.lower(), sdir)
    return out


def match_bbox_paths(
    image_paths: List[str],
    bbox_dir: str,
    image_suffix: str,
    bbox_suffix: str) -> List[str]:
    """Resolve a per-sample list of ROI bbox NIfTI files matched 1:1 to
    ``image_paths`` by *base name* (i.e. filename with the image suffix
    stripped, then `bbox_suffix` appended).

    Errors out (rather than silently dropping samples) if any image
    lacks a matching bbox file — a missing bbox means we'd fall back
    to the whole volume for that sample, which is rarely intended and
    biases batch statistics.
    """
    return _match_per_sample_paths(
        image_paths, bbox_dir, image_suffix, bbox_suffix, kind="BBox")


def match_region_weight_paths(
    image_paths: List[str],
    region_weight_dir: str,
    image_suffix: str,
    region_weight_suffix: str) -> List[str]:
    """Resolve a per-sample list of region-weight NIfTI files matched 1:1
    to ``image_paths`` by *base name* (image suffix stripped, then
    ``region_weight_suffix`` appended).

    Strong contract: every sample must have a matching file
    (FileNotFoundError otherwise), mirroring ``match_bbox_paths``. See
    ``DataConfig.region_weight_dir`` for the semantics of the file
    (background=0, non-bg = weight value; dataset adds +1 on load).
    """
    return _match_per_sample_paths(
        image_paths, region_weight_dir, image_suffix, region_weight_suffix,
        kind="RegionWeight")


def detect_label_values(
    label_paths: List[str], max_scan: Optional[int] = None) -> List[int]:
    """Auto-detect unique label values from the label files.

    Scans ALL label files by default (previously only the first 5, which
    silently missed rare classes distributed across the dataset). Pass
    ``max_scan`` if the dataset is very large and a subset scan is
    acceptable — results will be logged with an explicit "partial scan"
    warning in that case.

    Returns a sorted list of integer label values starting with background.
    """
    n_total = len(label_paths)
    if max_scan is None or max_scan >= n_total:
        scan_paths = label_paths
        partial = False
    else:
        scan_paths = label_paths[:max_scan]
        partial = True

    all_labels = set()
    for path in scan_paths:
        lbl    = load_nifti(path)
        unique = np.unique(np.round(lbl).astype(np.int32)).tolist()
        all_labels.update(unique)

    result = sorted(all_labels)
    if partial:
        logger.warning(
            "Auto-detected label values from partial scan (%d/%d files): %s. "
            "Rare classes may be missed; pass max_scan=None to scan all.",
            len(scan_paths), n_total, result)
    else:
        logger.info(
            "Auto-detected label values (scanned %d files): %s",
            n_total, result)
    return result


def train_val_split(n: int, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    """Random (non-stratified) train/val split by index."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n).tolist()
    n_val = max(1, int(n * val_ratio))
    return indices[n_val:], indices[:n_val]


def _volume_primary_class(
    label_path: str, label_values: List[int]) -> int:
    """Return the label value that occupies the most voxels in the volume
    (background counted too). Ties break on the smallest label value.
    """
    lbl = load_nifti(label_path)
    lbl_int = np.round(lbl).astype(np.int32)
    # Count voxels per requested label value; ignore stray labels.
    counts = np.array(
        [(lbl_int == v).sum() for v in label_values], dtype=np.int64)
    if counts.sum() == 0:
        return label_values[0]
    return int(label_values[int(np.argmax(counts))])


def stratified_train_val_split(
    label_paths: List[str],
    label_values: List[int],
    val_ratio: float,
    seed: int,
    use_foreground_only: bool = True) -> Tuple[List[int], List[int]]:
    """Stratified split by each volume's primary label.

    Each volume is assigned a stratum equal to the most-frequent foreground
    label it contains (falling back to background for entirely-empty volumes).
    Within each stratum, samples are shuffled and split according to
    ``val_ratio``. When ``use_foreground_only`` is True (default), the
    primary label is restricted to foreground — this matches the typical
    medical-segmentation use case where the tumour / organ class distribution
    matters far more than "how much background a volume has".

    Falls back gracefully to a non-stratified split when the dataset is
    too small to stratify (fewer than 2 samples per stratum would leave
    some strata empty on one side).
    """
    n   = len(label_paths)
    rng = np.random.RandomState(seed)

    # Determine which label values are used as strata keys.
    fg_vals = label_values[1:] if use_foreground_only and len(label_values) > 1 else label_values
    strata_vals = fg_vals if fg_vals else label_values

    # Assign each volume to a stratum.
    strata: Dict[int, List[int]] = {v: [] for v in strata_vals}
    fallback: List[int] = []  # volumes with no voxel in any fg class
    for idx, path in enumerate(label_paths):
        lbl = load_nifti(path)
        lbl_int = np.round(lbl).astype(np.int32)
        counts = {v: int((lbl_int == v).sum()) for v in strata_vals}
        best = max(counts.values())
        if best == 0:
            fallback.append(idx)
        else:
            primary = min(v for v, c in counts.items() if c == best)  # tie-break by smallest label
            strata[primary].append(idx)

    # Determine per-stratum split. If any stratum has < 2 samples we cannot
    # stratify it cleanly; treat it as non-fractionable and put the entire
    # bucket into train (preferring training coverage).
    train_idx: List[int] = []
    val_idx: List[int] = []

    for key, members in strata.items():
        if not members:
            continue
        rng.shuffle(members)
        if len(members) < 2:
            train_idx.extend(members)
            continue
        n_val_k = max(1, int(round(len(members) * val_ratio)))
        # Never put every member of a stratum into val
        n_val_k = min(n_val_k, len(members) - 1)
        val_idx.extend(members[:n_val_k])
        train_idx.extend(members[n_val_k:])

    # Empty-label volumes are distributed by the same val_ratio, untouched
    # by stratification.
    rng.shuffle(fallback)
    n_val_f = int(round(len(fallback) * val_ratio))
    val_idx.extend(fallback[:n_val_f])
    train_idx.extend(fallback[n_val_f:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    # Safety net: if splitting produced an empty val set (e.g., every
    # stratum has 1 sample), fall back to random split so training can proceed.
    if not val_idx or not train_idx:
        logger.warning(
            "Stratified split produced degenerate sets "
            "(train=%d, val=%d); falling back to random split.",
            len(train_idx), len(val_idx))
        return train_val_split(n, val_ratio, seed)

    logger.info(
        "Stratified split: %d train, %d val (strata sizes: %s)",
        len(train_idx), len(val_idx),
        {str(k): len(v) for k, v in strata.items()})
    return train_idx, val_idx


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

    # Drop bad samples (e.g. SimpleITK-unreadable NIfTIs with non-orthonormal
    # direction cosines) listed in `data.exclude_list`. Done BEFORE label
    # auto-detection / stratified split so those stages never touch the
    # offending files. `keep_idx` later re-aligns the bbox list too.
    exclude_pids = _load_exclude_pids(getattr(dc, "exclude_list", ""))
    image_paths, label_paths, _ = _filter_by_exclude(
        image_paths, label_paths, dc.image_suffix, exclude_pids)

    # Auto-detect labels if needed, 标签值确认
    if not dc.label_values:
        dc.label_values = detect_label_values(label_paths)
        dc.num_classes  = len(dc.label_values)
        cfg.sync()
    logger.info("Label values: %s, num_classes: %d, num_fg: %d",
                dc.label_values, dc.num_classes, cfg.num_fg_classes)

    # Split — stratified by primary foreground class when requested.
    if getattr(dc, "stratified_split", True) and dc.num_classes >= 2:
        train_idx, val_idx = stratified_train_val_split(
            label_paths, dc.label_values, dc.val_ratio, dc.split_seed)
    else:
        train_idx, val_idx = train_val_split(
            len(image_paths), dc.val_ratio, dc.split_seed)
        logger.info("Split (random): %d train, %d val",
                    len(train_idx), len(val_idx))

    cache = dc.cache_mode == "memory"
    rw = cfg.loss.region_weights if cfg.loss.region_weights else None
    # `aug_oversample_ratio` now applies to BOTH z_axis and cubic modes (BUG-B).
    # Validation sets always use oversample=1.0 so val patches match the
    # physical patch size verbatim; no GPU augmentation runs on val data.
    train_oversample = max(dc.aug_oversample_ratio, 1.0)
    common_kwargs = dict(
        label_values=dc.label_values,
        patch_size=tuple(dc.patch_size),
        intensity_min=dc.intensity_min,
        intensity_max=dc.intensity_max,
        normalize=dc.normalize,
        global_mean=dc.global_mean,
        global_std=dc.global_std,
        cache_enabled=cache,
        cache_max_volumes=getattr(dc, "cache_max_volumes", 0),
        region_weights=rw)
    # ``z_boundary_mode`` is meaningful for the z-axis-style datasets
    # (z_axis / 2_5d) but not for cubic / whole. Inject it only there
    # to keep the cubic / whole dataset signatures untouched.
    z_kwargs = dict(z_boundary_mode=getattr(dc, "z_boundary_mode", "stretch"))

    # Optional ROI bbox paths — matched 1:1 to image_paths, then split
    # along the same train/val indices so each per-sample bbox stays
    # aligned with its image / label after the split.
    bbox_paths_all: Optional[List[str]] = None
    if getattr(dc, "bbox_dir", ""):
        bbox_paths_all = match_bbox_paths(
            image_paths, dc.bbox_dir, dc.image_suffix, dc.bbox_suffix)

    # Optional per-sample region-weight NIfTI paths — same 1:1 matching
    # as bbox_paths_all, strong-contract (error out on any missing file).
    # When set, takes precedence over ``loss.region_weights`` at runtime
    # inside the dataset (see ``SegDataset3D.__getitem__`` et al.).
    rw_paths_all: Optional[List[str]] = None
    if getattr(dc, "region_weight_dir", ""):
        rw_paths_all = match_region_weight_paths(
            image_paths, dc.region_weight_dir, dc.image_suffix,
            getattr(dc, "region_weight_suffix", ".nii.gz"))

    train_paths = dict(
        image_paths=[image_paths[i] for i in train_idx],
        label_paths=[label_paths[i] for i in train_idx])
    val_paths = dict(
        image_paths=[image_paths[i] for i in val_idx],
        label_paths=[label_paths[i] for i in val_idx])
    if bbox_paths_all is not None:
        train_paths["bbox_paths"] = [bbox_paths_all[i] for i in train_idx]
        val_paths["bbox_paths"] = [bbox_paths_all[i] for i in val_idx]
    if rw_paths_all is not None:
        train_paths["region_weight_paths"] = [rw_paths_all[i] for i in train_idx]
        val_paths["region_weight_paths"] = [rw_paths_all[i] for i in val_idx]

    if dc.patch_mode == "2_5d":
        # 2.5D mode reuses the z_axis dataset verbatim with a single
        # resolution (forced by Config.validate). The squeeze that turns
        # (C_res=1, D, H, W) into a 2D (D-channel) input happens inside
        # the trainer AFTER GPU augmentation, so the dataset itself is
        # bit-identical to the legacy z_axis single-res path.
        logger.info(
            "Using 2.5D patch mode (oversample=%.2f, z_boundary=%s) — "
            "z_axis dataset, trainer squeezes C_res=1 to feed a 2D model "
            "with D=%d input channels.",
            train_oversample, z_kwargs["z_boundary_mode"],
            int(dc.patch_size[0]))
        train_ds = SegDataset3D(
            **train_paths,
            aug_oversample_ratio=train_oversample,
            multi_res_scales=[1.0],
            foreground_oversample_ratio=dc.foreground_oversample_ratio,
            samples_per_volume=dc.samples_per_volume,
            is_train=True,
            **common_kwargs,
            **z_kwargs)
        val_ds = SegDataset3D(
            **val_paths,
            aug_oversample_ratio=1.0,
            multi_res_scales=[1.0],
            foreground_oversample_ratio=0.0,
            samples_per_volume=max(dc.samples_per_volume // 2, 1),
            is_train=False,
            **common_kwargs,
            **z_kwargs)
    elif dc.patch_mode == "whole":
        logger.info("Using WHOLE-VOLUME patch mode (oversample=%.2f)",
                    train_oversample)
        # Whole mode ignores `foreground_oversample_ratio` and
        # `multi_res_scales` (validated in Config). Builds a 1-channel
        # (eD, eH, eW) resize of the full volume.
        train_ds = SegDataset3DWhole(
            **train_paths,
            aug_oversample_ratio=train_oversample,
            samples_per_volume=dc.samples_per_volume,
            is_train=True,
            **common_kwargs)
        val_ds = SegDataset3DWhole(
            **val_paths,
            aug_oversample_ratio=1.0,
            samples_per_volume=max(dc.samples_per_volume // 2, 1),
            is_train=False,
            **common_kwargs)
    elif dc.patch_mode == "cubic":
        logger.info("Using CUBIC patch mode (oversample=%.2f, scales=%s)",
                     train_oversample, dc.multi_res_scales)
        train_ds = SegDataset3DCubic(
            **train_paths,
            aug_oversample_ratio=train_oversample,
            multi_res_scales=dc.multi_res_scales,
            foreground_oversample_ratio=dc.foreground_oversample_ratio,
            samples_per_volume=dc.samples_per_volume,
            is_train=True,
            **common_kwargs)
        val_ds = SegDataset3DCubic(
            **val_paths,
            aug_oversample_ratio=1.0,
            multi_res_scales=dc.multi_res_scales,
            foreground_oversample_ratio=0.0,
            samples_per_volume=max(dc.samples_per_volume // 2, 1),
            is_train=False,
            **common_kwargs)
    else:
        logger.info("Using Z_AXIS patch mode (oversample=%.2f, scales=%s, "
                    "z_boundary=%s)",
                    train_oversample, dc.multi_res_scales,
                    z_kwargs["z_boundary_mode"])
        train_ds = SegDataset3D(
            **train_paths,
            aug_oversample_ratio=train_oversample,
            multi_res_scales=dc.multi_res_scales,
            foreground_oversample_ratio=dc.foreground_oversample_ratio,
            samples_per_volume=dc.samples_per_volume,
            is_train=True,
            **common_kwargs,
            **z_kwargs)
        val_ds = SegDataset3D(
            **val_paths,
            aug_oversample_ratio=1.0,
            multi_res_scales=dc.multi_res_scales,
            foreground_oversample_ratio=0.0,
            samples_per_volume=max(dc.samples_per_volume // 2, 1),
            is_train=False,
            **common_kwargs,
            **z_kwargs)

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