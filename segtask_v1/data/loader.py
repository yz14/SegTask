"""DataLoader factory + train/val split.

Scans the data directories, splits into train/val, and creates DataLoaders.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

SuffixSpec = Union[str, Sequence[str]]

import numpy as np
from torch.utils.data import DataLoader

from ..config import Config
from .dataset import (
    SegDataset3D,
    SegDataset3DCubic,
    SegDataset3DWhole,
    load_nifti,
    load_npz_label_for_split)

logger = logging.getLogger(__name__)


def _load_exclude_pids(exclude_list: str) -> set:
    """Load pids from a plain-text exclude list (one per line; ``#`` comments allowed).

    Trailing ``.nii.gz``/``.nii`` is stripped so raw filenames are also accepted.
    Returns empty set if path is empty or missing.
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
    image_suffix: SuffixSpec,
    exclude_pids: set) -> Tuple[List[str], List[str], List[int]]:
    """Drop pairs whose image stem is in ``exclude_pids``; ``keep_idx`` re-aligns companion lists."""
    if not exclude_pids:
        return image_paths, label_paths, list(range(len(image_paths)))

    image_suffixes = _normalize_suffixes(image_suffix)
    keep_idx: List[int] = []
    dropped : List[str] = []
    for i, img_path in enumerate(image_paths):
        name = Path(img_path).name
        base = _strip_suffix(name, image_suffixes)
        if base is None:
            base = Path(name).stem
        if base in exclude_pids:
            dropped.append(base)
        else:
            keep_idx.append(i)

    if dropped:  # loggin
        head = ", ".join(dropped[:10])
        more = f", ... (+{len(dropped) - 10} more)" if len(dropped) > 10 else ""
        logger.warning(
            "Excluded %d/%d sample(s) via `data.exclude_list`: [%s%s]",
            len(dropped), len(image_paths), head, more)

    image_paths = [image_paths[i] for i in keep_idx]
    label_paths = [label_paths[i] for i in keep_idx]
    return image_paths, label_paths, keep_idx


def _normalize_suffixes(suffix: SuffixSpec) -> List[str]:
    """Normalise suffix spec to a de-duplicated list (string or sequence accepted)."""
    if isinstance(suffix, str):
        items = [suffix]
    else:
        items = list(suffix)
    out: List[str] = []
    seen = set()
    for s in items:
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    if not out:
        raise ValueError(
            "Suffix spec is empty; expected at least one non-empty string.")
    return out


def _strip_suffix(name: str, suffixes: Sequence[str]) -> Optional[str]:
    """Strip the first matching suffix; return ``None`` if no candidate matches."""
    for sfx in suffixes:
        if name.endswith(sfx):
            return name[: -len(sfx)]
    return None


def discover_samples(
    image_dir: str, label_dir: str,
    image_suffix: SuffixSpec = ".nii.gz",
    label_suffix: SuffixSpec = ".nii.gz",
) -> Tuple[List[str], List[str]]:
    """Pair images and labels by *base name* (suffix stripped); first matching label wins.

    Both suffix args accept a single string or a sequence of candidates,
    e.g. ``label_suffix=[".nii.gz", "-seg.nii.gz"]``.
    Returns ``(image_paths, label_paths)`` sorted by base name.
    """
    img_dir, lbl_dir = Path(image_dir), Path(label_dir)
    assert img_dir.is_dir(), f"Image dir not found: {img_dir}"
    assert lbl_dir.is_dir(), f"Label dir not found: {lbl_dir}"

    image_suffixes = _normalize_suffixes(image_suffix)
    label_suffixes = _normalize_suffixes(label_suffix)

    # Enumerate images by any accepted suffix; on collision earlier suffix wins.
    img_by_base: Dict[str, Path] = {}
    for sfx in image_suffixes:
        for p in sorted(img_dir.glob(f"*{sfx}")):
            base = _strip_suffix(p.name, [sfx])
            if base is None:
                continue
            img_by_base.setdefault(base, p)

    # For each image base: first existing ``<base><suffix>`` under lbl_dir wins.
    image_paths  : List[str] = []
    label_paths  : List[str] = []
    missing_bases: List[str] = []
    for base in sorted(img_by_base.keys()):
        chosen: Optional[Path] = None
        for sfx in label_suffixes:
            cand = lbl_dir / f"{base}{sfx}"
            if cand.is_file():
                chosen = cand
                break
        if chosen is None:
            missing_bases.append(base)
            continue
        image_paths.append(str(img_by_base[base]))
        label_paths.append(str(chosen))

    if not image_paths:
        raise ValueError(
            f"No matched pairs found in {img_dir} and {lbl_dir}. "
            f"Images: {len(img_by_base)} (suffixes={image_suffixes}), "
            f"label_suffixes tried={label_suffixes}.")

    if missing_bases:
        head = ", ".join(missing_bases[:5])
        more = f" ... (+{len(missing_bases) - 5} more)" \
            if len(missing_bases) > 5 else ""
        logger.warning(
            "discover_samples: %d/%d image bases have no matching label "
            "under %s for any of %s; dropping them. Missing bases: %s%s",
            len(missing_bases), len(img_by_base), lbl_dir,
            label_suffixes, head, more)

    logger.info(
        "Found %d matched image-label pairs (image_suffixes=%s, "
        "label_suffixes=%s).",
        len(image_paths), image_suffixes, label_suffixes)
    return image_paths, label_paths


def _match_per_sample_paths(
    image_paths: List[str],
    src_dir: str,
    image_suffix: SuffixSpec,
    out_suffix: SuffixSpec,
    kind: str) -> List[str]:
    """Strict 1:1 per-sample matcher by base name; raises on any missing match.

    Used by ``match_bbox_paths`` and ``match_region_weight_paths``.
    ``kind`` is the label used in log / error messages.
    """
    sdir = Path(src_dir)
    assert sdir.is_dir(), f"{kind} dir not found: {sdir}"

    image_suffixes = _normalize_suffixes(image_suffix)
    out_suffixes   = _normalize_suffixes(out_suffix)

    out: List[str] = []
    missing: List[str] = []
    for img_path in image_paths:
        name = Path(img_path).name
        base = _strip_suffix(name, image_suffixes) or Path(name).stem
        chosen: Optional[Path] = None
        for sfx in out_suffixes:
            cand = sdir / f"{base}{sfx}"
            if cand.is_file():
                chosen = cand
                break
        if chosen is None:
            attempts = ", ".join(f"{base}{sfx}" for sfx in out_suffixes)
            missing.append(f"{sdir}/[{attempts}]")
        else:
            out.append(str(chosen))

    if missing:
        head = "\n  ".join(missing[:5])
        more = f"\n  ... ({len(missing) - 5} more)" if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"{kind} files not found for {len(missing)}/{len(image_paths)} "
            f"samples (suffixes tried={out_suffixes}):\n  {head}{more}")

    logger.info(
        "Matched %d %s files under %s (suffixes=%s).",
        len(out), kind.lower(), sdir, out_suffixes)
    return out


def match_bbox_paths(
    image_paths: List[str],
    bbox_dir: str,
    image_suffix: SuffixSpec,
    bbox_suffix: SuffixSpec) -> List[str]:
    """Resolve per-sample bbox NIfTI paths 1:1 with ``image_paths``; raises on any miss."""
    return _match_per_sample_paths(
        image_paths, bbox_dir, image_suffix, bbox_suffix, kind="BBox")


def match_bbox_paths_lenient(
    image_paths: List[str],
    bbox_dir: str,
    image_suffix: SuffixSpec,
    bbox_suffix: SuffixSpec) -> Tuple[List[str], List[str]]:
    """Lenient bbox matcher for inference: samples without a bbox are dropped (warned), not raised.

    Returns ``(matched_image_paths, matched_bbox_paths)`` aligned 1:1.
    """
    sdir = Path(bbox_dir)
    assert sdir.is_dir(), f"BBox dir not found: {sdir}"

    image_suffixes = _normalize_suffixes(image_suffix)
    out_suffixes = _normalize_suffixes(bbox_suffix)

    matched_images: List[str] = []
    matched_bboxes: List[str] = []
    missing: List[str] = []
    for img_path in image_paths:
        name = Path(img_path).name
        base = _strip_suffix(name, image_suffixes)
        if base is None:
            base = Path(name).stem
        chosen: Optional[Path] = None
        for sfx in out_suffixes:
            cand = sdir / f"{base}{sfx}"
            if cand.is_file():
                chosen = cand
                break
        if chosen is None:
            missing.append(base)
        else:
            matched_images.append(img_path)
            matched_bboxes.append(str(chosen))

    if missing:
        head = ", ".join(missing[:5])
        more = f" ... (+{len(missing) - 5} more)" \
            if len(missing) > 5 else ""
        logger.warning(
            "match_bbox_paths_lenient: %d/%d samples have no matching "
            "bbox under %s (suffixes tried=%s) — they will be SKIPPED. "
            "Missing bases: %s%s",
            len(missing), len(image_paths), sdir, out_suffixes,
            head, more)

    logger.info(
        "Matched %d/%d bbox files under %s (suffixes=%s).",
        len(matched_bboxes), len(image_paths), sdir, out_suffixes)
    return matched_images, matched_bboxes


def match_region_weight_paths(
    image_paths: List[str],
    region_weight_dir: str,
    image_suffix: SuffixSpec,
    region_weight_suffix: SuffixSpec) -> List[str]:
    """Resolve per-sample region-weight NIfTI paths 1:1 with ``image_paths``; raises on any miss.

    File semantics: background=0, non-bg = weight; dataset adds +1 on load.
    """
    return _match_per_sample_paths(
        image_paths, region_weight_dir, image_suffix, region_weight_suffix,
        kind="RegionWeight")


def _default_label_loader(path: str) -> np.ndarray:
    """Default int16 NIfTI label reader (npz mode swaps in ``load_npz_label_for_split``)."""
    return load_nifti(path, dtype=np.int16)


def detect_label_values(
    label_paths: List[str],
    max_scan: Optional[int] = None,
    label_loader_fn=None) -> List[int]:
    """Auto-detect unique label values; scans all files by default.

    ``max_scan`` enables a partial scan with a warning. ``label_loader_fn``
    swaps the reader (NIfTI vs npz). Returns sorted ints starting with bg.
    """
    if label_loader_fn is None:
        label_loader_fn = _default_label_loader
    n_total = len(label_paths)
    if max_scan is None or max_scan >= n_total:
        scan_paths = label_paths
        partial    = False
    else:
        scan_paths = label_paths[:max_scan]
        partial = True

    all_labels = set()
    for path in scan_paths:
        # int16 decode keeps the startup scan light on RAM.
        lbl    = label_loader_fn(path)
        unique = np.unique(lbl.astype(np.int32, copy=False)).tolist()
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
    label_path: str, label_values: List[int],
    label_loader_fn=None) -> int:
    """Label value with the highest voxel count (ties broken by smallest label)."""
    if label_loader_fn is None:
        label_loader_fn = _default_label_loader
    lbl = label_loader_fn(label_path)
    lbl_int = lbl.astype(np.int32, copy=False)
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
    use_foreground_only: bool = True,
    label_loader_fn=None) -> Tuple[List[int], List[int]]:
    """Stratify train/val by each volume's primary fg label; falls back to random if degenerate.

    ``use_foreground_only=True`` ignores background frequency when picking
    the primary class (typical for medical segmentation).
    """
    n   = len(label_paths)
    rng = np.random.RandomState(seed)

    fg_vals = label_values[1:] if use_foreground_only and len(label_values) > 1 else label_values
    strata_vals = fg_vals if fg_vals else label_values

    strata: Dict[int, List[int]] = {v: [] for v in strata_vals}
    fallback: List[int] = []  # volumes with no fg voxel
    if label_loader_fn is None:
        label_loader_fn = _default_label_loader
    for idx, path in enumerate(label_paths):
        lbl = label_loader_fn(path)
        lbl_int = lbl.astype(np.int32, copy=False)
        counts = {v: int((lbl_int == v).sum()) for v in strata_vals}
        best = max(counts.values())
        if best == 0:
            fallback.append(idx)
        else:
            primary = min(v for v, c in counts.items() if c == best)  # tie → smallest
            strata[primary].append(idx)

    # Strata with <2 members go entirely to train (cannot fractionate cleanly).
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
        # Never put every member of a stratum into val.
        n_val_k = min(n_val_k, len(members) - 1)
        val_idx.extend(members[:n_val_k])
        train_idx.extend(members[n_val_k:])

    # Empty-label volumes split by the same val_ratio (no stratification).
    rng.shuffle(fallback)
    n_val_f = int(round(len(fallback) * val_ratio))
    val_idx.extend(fallback[:n_val_f])
    train_idx.extend(fallback[n_val_f:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    # Safety net: if either split is empty, fall back to random.
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


def discover_npz_samples(
    npz_dir: str, npz_suffix: str = ".npz") -> List[str]:
    """List ``make_data`` npz packages under ``npz_dir``; ignores ``_*`` / dot sidecars."""
    d = Path(npz_dir)
    assert d.is_dir(), f"NPZ dir not found: {d}"
    paths = sorted(
        p for p in d.glob(f"*{npz_suffix}")
        if not p.name.startswith(("_", ".")))
    if not paths:
        raise ValueError(
            f"No npz packages found under {d} (suffix={npz_suffix!r}). "
            f"Did you run `python -m segtask_v1.data.make_data` first?")
    logger.info("Discovered %d npz package(s) under %s.", len(paths), d)
    return [str(p) for p in paths]


def build_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders: discover → detect labels → split → dataset/loader.

    NPZ vs. NIfTI modes are mutually exclusive: a non-empty ``data.npz_dir``
    forces the npz path and ignores ``image_dir``/``label_dir``/``bbox_dir``/
    ``region_weight_dir`` (logged loudly).
    """
    dc = cfg.data

    # Source dispatch: npz packages vs. NIfTI.
    npz_dir = getattr(dc, "npz_dir", "")
    if npz_dir:
        npz_suffix = getattr(dc, "npz_suffix", ".npz")
        # Auto-build npz cache when missing/empty (one-time cost; partial dir is
        # treated as authoritative — user must re-run make_data with --overwrite).
        npz_p       = Path(npz_dir)
        npz_present = npz_p.is_dir() and any(
            x for x in npz_p.glob(f"*{npz_suffix}") if not x.name.startswith(("_", ".")))
        if not npz_present:
            if not bool(getattr(dc, "npz_auto_build", True)):
                raise FileNotFoundError(
                    f"data.npz_dir={npz_dir!r} is empty/missing and "
                    f"data.npz_auto_build is False. Run "
                    f"`python -m segtask_v1.data.make_data --config "
                    f"<yaml> --out {npz_dir}` first, or set "
                    f"data.npz_auto_build: true to build inline.")
            logger.info(
                "data.npz_dir=%s is empty/missing — auto-building via "
                "make_data.prepare_dataset (workers=%d). This is a "
                "one-time cost; subsequent train runs will reuse the "
                "npz cache.", npz_dir, max(dc.num_workers, 1))
            # Local import: keeps the make_data deps off the loader.py path.
            from .make_data import prepare_dataset
            counters = prepare_dataset(
                cfg, npz_dir,
                workers=max(dc.num_workers, 1),
                overwrite=False)
            logger.info(
                "Auto-build complete: written=%d, skipped=%d, "
                "failed=%d / total=%d.",
                counters["written"], counters["skipped"],
                counters["failed"], counters["total"])
            if counters["failed"] > 0:
                logger.warning(
                    "make_data reported %d failed sample(s). Inspect "
                    "%s/_failures.txt; affected pids will be missing "
                    "from the training set.",
                    counters["failed"], npz_dir)
            if counters["written"] + counters["skipped"] == 0:
                raise RuntimeError(
                    f"Auto-build produced 0 valid npz packages under "
                    f"{npz_dir}. Check the input image_dir / "
                    f"label_dir paths and the make_data error log.")
        logger.info(
            "DataConfig.npz_dir is set (%s, suffix=%s) — using "
            "pre-computed npz pipeline. The following NIfTI fields "
            "are IGNORED at runtime: image_dir=%r, label_dir=%r, "
            "bbox_dir=%r, region_weight_dir=%r.",
            npz_dir, npz_suffix,
            dc.image_dir, dc.label_dir,
            getattr(dc, "bbox_dir", ""),
            getattr(dc, "region_weight_dir", ""))
        npz_paths_all = discover_npz_samples(npz_dir, npz_suffix)
        # In npz mode image_paths/label_paths alias the npz list (used only for len/cache keys);
        # actual I/O routes through ``_npz_paths`` inside the dataset.
        image_paths = list(npz_paths_all)
        label_paths = list(npz_paths_all)
        exclude_pids = _load_exclude_pids(getattr(dc, "exclude_list", ""))
        image_paths, label_paths, keep_idx = _filter_by_exclude(
            image_paths, label_paths, npz_suffix, exclude_pids)
        if exclude_pids:
            npz_paths_all = [npz_paths_all[i] for i in keep_idx]
        label_loader_fn = load_npz_label_for_split
    else:
        npz_paths_all = None
        image_paths, label_paths = discover_samples(
            dc.image_dir, dc.label_dir, dc.image_suffix, dc.label_suffix)

        # Apply exclude_list before label scan/split so those stages skip bad files.
        exclude_pids = _load_exclude_pids(getattr(dc, "exclude_list", ""))
        image_paths, label_paths, _ = _filter_by_exclude(
            image_paths, label_paths, dc.image_suffix, exclude_pids)
        label_loader_fn = None  # default NIfTI int16 reader

    if not dc.label_values:
        dc.label_values = detect_label_values(
            label_paths, label_loader_fn=label_loader_fn)
        dc.num_classes  = len(dc.label_values)
        cfg.sync()
    logger.info("Label values: %s, num_classes: %d, num_fg: %d",
                dc.label_values, dc.num_classes, cfg.num_fg_classes)

    # Stratified split by primary fg class (random fallback below).
    if getattr(dc, "stratified_split", True) and dc.num_classes >= 2:
        train_idx, val_idx = stratified_train_val_split(
            label_paths, dc.label_values, dc.val_ratio, dc.split_seed,
            label_loader_fn=label_loader_fn)
    else:
        train_idx, val_idx = train_val_split(
            len(image_paths), dc.val_ratio, dc.split_seed)
        logger.info("Split (random): %d train, %d val",
                    len(train_idx), len(val_idx))

    cache = dc.cache_mode == "memory"
    rw    = cfg.loss.region_weights if cfg.loss.region_weights else None
    # Train uses oversample ≥ 1.0 (extra slack absorbed by augmentation);
    # val always uses 1.0 so patches match the physical patch_size verbatim.
    train_oversample = max(dc.aug_oversample_ratio, 1.0)
    common_kwargs = dict(
        label_values      = dc.label_values,
        patch_size        = tuple(dc.patch_size),
        intensity_min     = dc.intensity_min,
        intensity_max     = dc.intensity_max,
        normalize         = dc.normalize,
        global_mean       = dc.global_mean,
        global_std        = dc.global_std,
        cache_enabled     = cache,
        cache_max_volumes = getattr(dc, "cache_max_volumes", 0),
        region_weights    = rw)
    # z_boundary_mode applies only to z_axis / 2.5d datasets.
    z_kwargs = dict(z_boundary_mode=getattr(dc, "z_boundary_mode", "stretch"))

    # 2.5D-only switch — single max-FOV cube path; trainer center-crops per view.
    aux_native_kwargs = dict(
        aux_keep_native_d = bool(getattr(dc, "aux_keep_native_d", False))
        and dc.patch_mode == "2_5d"
        and len(dc.multi_res_scales) > 1)

    # 3D analogue of aux_keep_native_d for z_axis / cubic; defensively gated by mode.
    keep_native_kwargs_z = dict(
        keep_native_multi_res=bool(getattr(dc, "keep_native_multi_res", False))
        and dc.patch_mode == "z_axis"
        and len(dc.multi_res_scales) > 1)
    keep_native_kwargs_cubic = dict(
        keep_native_multi_res=bool(getattr(dc, "keep_native_multi_res", False))
        and dc.patch_mode == "cubic"
        and len(dc.multi_res_scales) > 1)

    # Per-sample ROI bbox paths (NIfTI mode only; npz already has bbox baked in).
    bbox_paths_all: Optional[List[str]] = None
    if npz_paths_all is None and getattr(dc, "bbox_dir", ""):
        bbox_paths_all = match_bbox_paths(
            image_paths, dc.bbox_dir, dc.image_suffix, dc.bbox_suffix)

    # Per-sample region-weight NIfTI paths (NIfTI mode only; npz embeds rw).
    # When set, overrides loss.region_weights inside the dataset.
    rw_paths_all: Optional[List[str]] = None
    if npz_paths_all is None and getattr(dc, "region_weight_dir", ""):
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
        val_paths["bbox_paths"]   = [bbox_paths_all[i] for i in val_idx]
    if rw_paths_all is not None:
        train_paths["region_weight_paths"] = [rw_paths_all[i] for i in train_idx]
        val_paths["region_weight_paths"]   = [rw_paths_all[i] for i in val_idx]
    # NPZ mode: pass the per-sample npz path slice; dataset routes I/O via npz readers.
    if npz_paths_all is not None:
        train_paths["npz_paths"] = [npz_paths_all[i] for i in train_idx]
        val_paths["npz_paths"]   = [npz_paths_all[i] for i in val_idx]

    if dc.patch_mode == "2_5d":
        # 2.5D reuses the z_axis dataset. Two layouts depending on aux_keep_native_d:
        #   False (legacy): each scale extracts round(eD*s) slices, resized to (eD,pH,pW);
        #     trainer collapses (B, n_views, D, H, W) → (B, n_views*D, H, W).
        #   True: single max-FOV cube of depth round(eD*max_scale); trainer
        #     center-crops per view at native depth before forward.
        n_views = max(len(dc.multi_res_scales), 1)
        if aux_native_kwargs["aux_keep_native_d"]:
            max_scale = max(dc.multi_res_scales)
            eD_max    = int(round(int(dc.patch_size[0]) * max_scale))
            logger.info(
                "Using 2.5D patch mode + aux_keep_native_d=True "
                "(oversample=%.2f, scales=%s, n_views=%d, max_scale=%.2f) "
                "— SINGLE max-FOV cube extraction (depth=%d), trainer "
                "center-crops per view at native depth before forward.",
                train_oversample, dc.multi_res_scales, n_views,
                max_scale, eD_max)
        else:
            logger.info(
                "Using 2.5D patch mode (oversample=%.2f, z_boundary=%s, "
                "scales=%s, n_views=%d) — z_axis dataset; trainer reshapes "
                "(B, %d, D=%d, H, W) → (B, %d, H, W) for the 2D model.",
                train_oversample, z_kwargs["z_boundary_mode"],
                dc.multi_res_scales, n_views,
                n_views, int(dc.patch_size[0]),
                n_views * int(dc.patch_size[0]))
        train_ds = SegDataset3D(
            **train_paths,
            aug_oversample_ratio=train_oversample,
            multi_res_scales=dc.multi_res_scales,
            foreground_oversample_ratio = dc.foreground_oversample_ratio,
            samples_per_volume          = dc.samples_per_volume,
            is_train=True,
            **common_kwargs,
            **z_kwargs,
            **aux_native_kwargs)
        val_ds = SegDataset3D(
            **val_paths,
            aug_oversample_ratio=1.0,
            multi_res_scales=dc.multi_res_scales,
            foreground_oversample_ratio=0.0,
            samples_per_volume=max(dc.samples_per_volume // 2, 1),
            is_train=False,
            **common_kwargs,
            **z_kwargs,
            **aux_native_kwargs)
    elif dc.patch_mode == "whole":
        logger.info("Using WHOLE-VOLUME patch mode (oversample=%.2f)",
                    train_oversample)
        # Whole mode ignores fg oversample / multi_res_scales (validated in Config).
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
        if keep_native_kwargs_cubic["keep_native_multi_res"]:
            logger.info(
                "Using CUBIC patch mode + keep_native_multi_res=True "
                "(oversample=%.2f, scales=%s, max_scale=%.2f) — SINGLE "
                "max-FOV cube extraction; trainer crops+resizes per "
                "view before the 3D forward.",
                train_oversample, dc.multi_res_scales,
                max(dc.multi_res_scales))
        else:
            logger.info(
                "Using CUBIC patch mode (oversample=%.2f, scales=%s)",
                train_oversample, dc.multi_res_scales)
        train_ds = SegDataset3DCubic(
            **train_paths,
            aug_oversample_ratio=train_oversample,
            multi_res_scales=dc.multi_res_scales,
            foreground_oversample_ratio=dc.foreground_oversample_ratio,
            samples_per_volume=dc.samples_per_volume,
            is_train=True,
            **common_kwargs,
            **keep_native_kwargs_cubic)
        val_ds = SegDataset3DCubic(
            **val_paths,
            aug_oversample_ratio=1.0,
            multi_res_scales=dc.multi_res_scales,
            foreground_oversample_ratio=0.0,
            samples_per_volume=max(dc.samples_per_volume // 2, 1),
            is_train=False,
            **common_kwargs,
            **keep_native_kwargs_cubic)
    else:
        if keep_native_kwargs_z["keep_native_multi_res"]:
            logger.info(
                "Using Z_AXIS patch mode + keep_native_multi_res=True "
                "(oversample=%.2f, scales=%s, max_scale=%.2f, "
                "z_boundary=%s) — SINGLE max-FOV z-cube extraction; "
                "trainer crops+resizes per view before the 3D forward.",
                train_oversample, dc.multi_res_scales,
                max(dc.multi_res_scales), z_kwargs["z_boundary_mode"])
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
            **z_kwargs,
            **keep_native_kwargs_z)
        val_ds = SegDataset3D(
            **val_paths,
            aug_oversample_ratio=1.0,
            multi_res_scales=dc.multi_res_scales,
            foreground_oversample_ratio=0.0,
            samples_per_volume=max(dc.samples_per_volume // 2, 1),
            is_train=False,
            **common_kwargs,
            **z_kwargs,
            **keep_native_kwargs_z)

    # persistent_workers / prefetch_factor only valid when num_workers > 0.
    loader_kwargs: Dict[str, object] = {}
    if dc.num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            getattr(dc, "persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(
            getattr(dc, "prefetch_factor", 4))

    train_loader = DataLoader(
        train_ds,
        batch_size=dc.batch_size,
        shuffle=True,
        num_workers=dc.num_workers,
        pin_memory=dc.pin_memory,
        drop_last=True,
        **loader_kwargs)
    val_loader = DataLoader(
        val_ds,
        batch_size=dc.batch_size,
        shuffle=False,
        num_workers=dc.num_workers,
        pin_memory=dc.pin_memory,
        drop_last=False,
        **loader_kwargs)

    logger.info(
        "DataLoader: batch_size=%d, num_workers=%d, pin_memory=%s, "
        "persistent_workers=%s, prefetch_factor=%s",
        dc.batch_size, dc.num_workers, dc.pin_memory,
        loader_kwargs.get("persistent_workers", "n/a"),
        loader_kwargs.get("prefetch_factor", "n/a"))

    # Memory-cache footprint estimate (purely diagnostic; per-worker caches multiply).
    if dc.cache_mode == "memory":
        try:
            # Estimate realistic post-crop bytes/vol from train_ds bboxes when present.
            bboxes = getattr(train_ds, "_bboxes", None)
            npz_paths_train = getattr(train_ds, "_npz_paths", None)
            has_rw_runtime = bool(getattr(dc, "region_weight_dir", ""))
            if npz_paths_train is not None:
                # NPZ: read shape + rw-key presence from the first npz, no NIfTI decode.
                from .dataset import _open_npz as _peek_npz  # local alias
                _f = _peek_npz(npz_paths_train[0])
                _shape = _f["image"].shape
                sample_voxels = int(np.prod(_shape))
                has_rw_runtime = "rw" in _f.files
            else:
                def _cached_voxels(i: int) -> int:
                    """Voxels per cached volume i (bbox-cropped when available)."""
                    bb = bboxes[i] if bboxes and i < len(bboxes) else None
                    if bb is not None:
                        (d0, d1), (h0, h1), (w0, w1) = bb
                        return (d1 - d0) * (h1 - h0) * (w1 - w0)
                    # Fallback: decode a single header-only-ish scan.
                    sample = load_nifti(image_paths[i])
                    return int(sample.size)
                sample_voxels = _cached_voxels(0)
            # image fp32 (4B), label int16 (2B), rw fp32 (4B, when configured).
            bytes_per_img = sample_voxels * 4
            bytes_per_lbl = sample_voxels * 2
            bytes_per_rw = sample_voxels * 4 if has_rw_runtime else 0
            per_vol_bytes = bytes_per_img + bytes_per_lbl + bytes_per_rw
            n_train_vols = len(train_idx)
            cap = int(dc.cache_max_volumes)
            # cap=0 means unbounded → worst-case: every volume cached.
            eff_cap = cap if cap > 0 else n_train_vols
            eff_cap = min(eff_cap, n_train_vols)
            workers = max(dc.num_workers, 1)
            total_gb = per_vol_bytes * eff_cap * workers / (1024 ** 3)
            logger.info(
                "Volume cache estimate: ~%.2f MiB per volume "
                "(image fp32 + label int16%s, bbox-cropped); effective "
                "cap=%d, num_workers=%d => up to ~%.2f GiB RAM (all "
                "workers, caches only; transient decode peaks add "
                "~%.2f MiB/worker).",
                per_vol_bytes / (1024 ** 2),
                " + region_weight fp32" if bytes_per_rw else "",
                eff_cap, workers, total_gb,
                bytes_per_img / (1024 ** 2))
            if cap == 0 and n_train_vols * workers >= 16:
                # Recommend a cap that fits under an 8 GiB budget heuristic.
                budget_gb = 8.0
                rec = max(
                    1,
                    int(budget_gb * (1024 ** 3)
                        / max(per_vol_bytes, 1) / workers))
                logger.warning(
                    "cache_max_volumes=0 (unbounded) with %d volumes and "
                    "%d workers is the likely OOM culprit on large "
                    "datasets. Consider setting "
                    "`data.cache_max_volumes: %d` (≈%.1f GiB budget) "
                    "or `data.cache_mode: \"none\"` to rely on the OS "
                    "page cache (shared across workers).",
                    n_train_vols, workers, rec, budget_gb)
        except Exception as exc:  # pragma: no cover — diagnostic only
            logger.debug("Could not estimate volume cache size: %s", exc)

    return train_loader, val_loader