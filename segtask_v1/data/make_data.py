"""Offline per-sample npz packager (image + label + optional rw + fg-index).

Decouples training from the gzip-decompress peak of NIfTI by writing one
bbox-cropped npz per sample. Runtime then mmaps these (uncompressed) npz
so multiple workers share the OS page cache.

Output: ``<out_dir>/<pid>.npz`` with keys:
  - ``image`` int16, ``label`` int16 (both bbox-cropped, raw HU/labels)
  - ``fg_slices`` int32 (M,), ``fg_coords`` int32 (N, 3) (cropped frame)
  - ``meta`` 0-d object (provenance: pid, src paths, bbox, label_values, ...)
  - ``rw`` int16 or fp32 (only if ``region_weight_dir`` set; +1-shifted)

Image stays raw HU because intensity windowing is a train-time hparam.
Default uses ``np.savez`` (uncompressed) so npy blobs can be memmap-shared;
``--compress`` opts into ``savez_compressed`` (smaller, slower, no sharing).

CLI: ``python -m segtask_v1.data.make_data --config <yaml> --out <dir> [--workers N]``.
Existing npz are skipped unless ``--overwrite``. Failures collected in
``<out_dir>/_failures.txt`` (compatible with ``data.exclude_list``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import Config, load_config
from .dataset import (
    BBox,
    compute_bbox_from_volume,
    load_nifti,
    load_nifti_cropped,
    load_region_weight_volume,
)
from .loader import (
    _filter_by_exclude,
    _load_exclude_pids,
    discover_samples,
    match_bbox_paths,
    match_region_weight_paths,
    detect_label_values,
)

logger = logging.getLogger(__name__)


_TOOL_VERSION = "make_data/1.0"

# Matches SegDataset3DCubic._build_index cap; override via CLI if needed.
_DEFAULT_FG_SUBSAMPLE = 50_000


# =============================================================================
# Per-sample worker
# =============================================================================
def _stem(path: str, suffix) -> str:
    """Return ``filename - suffix``; ``suffix`` may be a string or list of candidates."""
    name = Path(path).name
    suffixes = [suffix] if isinstance(suffix, str) else list(suffix)
    for sfx in suffixes:
        if sfx and name.endswith(sfx):
            return name[: -len(sfx)]
    return Path(name).stem


def _compute_fg_indices(
    label: np.ndarray,
    bg_val: int,
    fg_subsample: int,
    seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Compute fg_slices (z-indices) and fg_coords (sub-sampled to ``fg_subsample``).

    Mirrors ``SegDataset3D._build_index`` / ``SegDataset3DCubic._build_index``
    (seed=42) so any patch_mode can be served without rescanning at runtime.
    """
    if label.size == 0:
        return (np.zeros((0,), dtype=np.int32),
                np.zeros((0, 3), dtype=np.int32))
    label_int = label.astype(np.int32, copy=False)
    fg_mask = label_int != int(bg_val)
    if not fg_mask.any():
        return (np.zeros((0,), dtype=np.int32),
                np.zeros((0, 3), dtype=np.int32))
    fg_slices = np.where(np.any(fg_mask, axis=(1, 2)))[0].astype(np.int32)
    coords = np.argwhere(fg_mask).astype(np.int32)
    if fg_subsample > 0 and len(coords) > fg_subsample:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(coords), fg_subsample, replace=False)
        coords = coords[idx]
    return fg_slices, coords


def _bbox_from_mask_path(bbox_path: Optional[str]) -> Optional[BBox]:
    """Decode mask as int16 → ``compute_bbox_from_volume``; ``None`` if no path / empty mask."""
    if not bbox_path:
        return None
    mask = load_nifti(bbox_path, dtype=np.int16)
    return compute_bbox_from_volume(mask)


def prepare_one(
    pid: str,
    image_path: str,
    label_path: str,
    bbox_path: Optional[str],
    rw_path: Optional[str],
    out_path: str,
    label_values: List[int],
    fg_subsample: int = _DEFAULT_FG_SUBSAMPLE,
    compress: bool = False,
    overwrite: bool = False) -> Dict[str, object]:
    """Materialise the npz package for one sample; idempotent unless ``overwrite``.

    Returns a status dict (pid, status, size_bytes, elapsed_s, ...) used by
    the aggregate progress log.
    """
    out_p = Path(out_path)
    if out_p.is_file() and not overwrite:
        return {"pid": pid, "status": "skipped",
                "size_bytes": out_p.stat().st_size, "elapsed_s": 0.0}

    t0 = time.perf_counter()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # 1. Bbox from mask (None if no mask / empty).
    bbox = _bbox_from_mask_path(bbox_path)

    # 2-3. Load image (raw HU, int16) and label (int16), cropped to bbox.
    image = load_nifti_cropped(image_path, bbox=bbox, dtype=np.int16)
    label = load_nifti_cropped(label_path, bbox=bbox, dtype=np.int16)

    if image.shape != label.shape:
        raise ValueError(
            f"image shape {image.shape} != label shape {label.shape} for "
            f"pid={pid} (image={image_path}, label={label_path})")

    # 4. Region weight (+1-shifted): store int16 when integer-valued and in range,
    # else fp32 (runtime loader returns fp32 either way).
    rw: Optional[np.ndarray] = None
    rw_dtype_stored = None
    if rw_path:
        rw = load_region_weight_volume(rw_path, bbox=bbox)
        if rw.shape != image.shape:
            raise ValueError(
                f"region_weight shape {rw.shape} != image shape {image.shape} "
                f"for pid={pid} (rw={rw_path})")
        rw_min = float(rw.min())
        rw_max = float(rw.max())
        is_integer_valued = np.all(rw == np.round(rw))
        fits_int16 = (rw_min >= np.iinfo(np.int16).min
                      and rw_max <= np.iinfo(np.int16).max)
        if is_integer_valued and fits_int16:
            rw = rw.astype(np.int16, copy=False)
            rw_dtype_stored = "int16"
        else:
            rw_dtype_stored = "float32"
            logger.warning(
                "pid=%s rw has non-integer or out-of-int16 values "
                "(min=%.3f, max=%.3f, integer_valued=%s) — storing as float32.",
                pid, rw_min, rw_max, is_integer_valued)

    # 5. Foreground indices in the cropped frame.
    bg_val = int(label_values[0])
    fg_slices, fg_coords = _compute_fg_indices(label, bg_val, fg_subsample)

    # 6. Provenance metadata (self-describing for debugging).
    meta = {
        "pid": pid,
        "src_image": str(image_path),
        "src_label": str(label_path),
        "src_bbox": str(bbox_path) if bbox_path else "",
        "src_rw": str(rw_path) if rw_path else "",
        "bbox": (
            list(map(list, bbox)) if bbox is not None else None),
        "label_values": list(map(int, label_values)),
        "has_rw": rw is not None,
        "rw_shift": 1.0,
        "rw_dtype": rw_dtype_stored,    # int16 / float32 / None
        "image_dtype": str(image.dtype),
        "made_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool_version": _TOOL_VERSION,
    }
    meta_arr = np.array(meta, dtype=object)

    # 7. Atomic write (tmp + rename). Pass an open file handle to np.savez
    # so it does NOT auto-append ".npz" to our tmp name.
    tmp_path = out_p.with_name(out_p.name + ".tmp")
    save_fn = np.savez_compressed if compress else np.savez
    payload = {
        "image": image,
        "label": label,
        "fg_slices": fg_slices,
        "fg_coords": fg_coords,
        "meta": meta_arr,
    }
    if rw is not None:
        payload["rw"] = rw
    with open(tmp_path, "wb") as fh:
        save_fn(fh, **payload)
    # Windows: destination must not exist before rename.
    if out_p.exists():
        out_p.unlink()
    tmp_path.rename(out_p)

    elapsed = time.perf_counter() - t0
    return {
        "pid": pid,
        "status": "written",
        "size_bytes": out_p.stat().st_size,
        "elapsed_s": elapsed,
        "shape": tuple(image.shape),
        "n_fg_slices": int(fg_slices.size),
        "n_fg_coords": int(fg_coords.shape[0]),
    }


# =============================================================================
# Top-level driver (CLI + library entry point)
# =============================================================================
def _build_sample_table(cfg: Config) -> List[Dict[str, Optional[str]]]:
    """Discover and pair image/label/bbox/rw paths via loader helpers; honours exclude_list."""
    dc = cfg.data

    image_paths, label_paths = discover_samples(
        dc.image_dir, dc.label_dir, dc.image_suffix, dc.label_suffix)

    exclude_pids = _load_exclude_pids(getattr(dc, "exclude_list", ""))
    image_paths, label_paths, _ = _filter_by_exclude(
        image_paths, label_paths, dc.image_suffix, exclude_pids)

    bbox_paths_all: Optional[List[str]] = None
    if getattr(dc, "bbox_dir", ""):
        bbox_paths_all = match_bbox_paths(
            image_paths, dc.bbox_dir, dc.image_suffix, dc.bbox_suffix)

    rw_paths_all: Optional[List[str]] = None
    if getattr(dc, "region_weight_dir", ""):
        rw_paths_all = match_region_weight_paths(
            image_paths, dc.region_weight_dir, dc.image_suffix,
            getattr(dc, "region_weight_suffix", ".nii.gz"))

    samples: List[Dict[str, Optional[str]]] = []
    for i, (img, lbl) in enumerate(zip(image_paths, label_paths)):
        samples.append({
            "pid": _stem(img, dc.image_suffix),
            "image": img,
            "label": lbl,
            "bbox": bbox_paths_all[i] if bbox_paths_all else None,
            "rw": rw_paths_all[i] if rw_paths_all else None,
        })
    return samples


def _resolve_label_values(
    cfg: Config, samples: List[Dict[str, Optional[str]]]) -> List[int]:
    """Auto-detect label values if not configured (mirrors loader)."""
    dc = cfg.data
    if dc.label_values:
        return list(map(int, dc.label_values))
    label_paths = [s["label"] for s in samples]
    detected = detect_label_values(label_paths)
    return list(map(int, detected))


def prepare_dataset(
    cfg: Config,
    out_dir: str,
    workers: int = 4,
    fg_subsample: int = _DEFAULT_FG_SUBSAMPLE,
    compress: bool = False,
    overwrite: bool = False,
    limit: int = 0) -> Dict[str, int]:
    """Pre-compute npz packages for every sample under ``cfg.data``.

    ``workers``: parallel processes (0 = inline / single-process for debugging).
    ``compress``: ``np.savez_compressed`` (smaller disk, loses memmap sharing).
    ``limit > 0``: smoke-test on the first N samples only.
    Returns counters ``{written, skipped, failed, total}``.
    """
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    samples = _build_sample_table(cfg)
    if limit and limit > 0:
        samples = samples[:limit]
        logger.info("--limit %d: processing only the first %d samples.",
                    limit, len(samples))

    label_values = _resolve_label_values(cfg, samples)
    logger.info("Using label_values=%s (bg=%d)", label_values, label_values[0])

    tasks: List[Tuple[Dict[str, Optional[str]], str]] = []
    for s in samples:
        out_path = str(out_p / f"{s['pid']}.npz")
        tasks.append((s, out_path))

    n_total = len(tasks)
    counters = {"written": 0, "skipped": 0, "failed": 0, "total": n_total}
    failures: List[Tuple[str, str]] = []   # (pid, error)
    timings: List[float] = []
    sizes: List[int] = []

    logger.info(
        "Preparing %d samples → %s (workers=%d, compress=%s, "
        "overwrite=%s, fg_subsample=%d)",
        n_total, out_p, workers, compress, overwrite, fg_subsample)

    def _kwargs(sample: Dict[str, Optional[str]], out_path: str) -> dict:
        return dict(
            pid=sample["pid"],
            image_path=sample["image"],
            label_path=sample["label"],
            bbox_path=sample["bbox"],
            rw_path=sample["rw"],
            out_path=out_path,
            label_values=label_values,
            fg_subsample=fg_subsample,
            compress=compress,
            overwrite=overwrite,
        )

    t0 = time.perf_counter()

    if workers <= 0:
        # Inline: full traceback, no pickling — easier to debug.
        for i, (s, out_path) in enumerate(tasks):
            try:
                res = prepare_one(**_kwargs(s, out_path))
                _record(res, counters, timings, sizes)
                _log_progress(i + 1, n_total, res, t0)
            except Exception as exc:
                counters["failed"] += 1
                failures.append((s["pid"], _short_exc(exc)))
                logger.exception("FAILED pid=%s: %s", s["pid"], exc)
    else:
        # Process pool (spawn on Windows; SimpleITK import paid once per worker).
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_pid = {
                pool.submit(prepare_one, **_kwargs(s, out_path)): s["pid"]
                for s, out_path in tasks
            }
            for i, fut in enumerate(as_completed(future_to_pid)):
                pid = future_to_pid[fut]
                try:
                    res = fut.result()
                    _record(res, counters, timings, sizes)
                    _log_progress(i + 1, n_total, res, t0)
                except Exception as exc:
                    counters["failed"] += 1
                    failures.append((pid, _short_exc(exc)))
                    logger.error("FAILED pid=%s: %s", pid, exc)

    # Aggregate report.
    elapsed = time.perf_counter() - t0
    total_bytes = sum(sizes)
    total_gb = total_bytes / (1024 ** 3)
    mean_s = (sum(timings) / max(len(timings), 1)) if timings else 0.0
    logger.info(
        "Done in %.1fs: written=%d, skipped=%d, failed=%d / total=%d. "
        "Total npz size: %.2f GiB (mean per sample: %.1f MiB, "
        "mean compute: %.2fs).",
        elapsed, counters["written"], counters["skipped"],
        counters["failed"], counters["total"],
        total_gb,
        (total_bytes / max(len(sizes), 1)) / (1024 ** 2) if sizes else 0.0,
        mean_s)

    # _failures.txt is data.exclude_list-compatible; clear stale file on success.
    fail_path = out_p / "_failures.txt"
    if not failures and fail_path.is_file():
        fail_path.unlink()
    if failures:
        with open(fail_path, "w", encoding="utf-8") as f:
            f.write("# make_data failures — generated %s\n" %
                    datetime.now(timezone.utc).isoformat(timespec="seconds"))
            f.write("# Format: <pid>\\t<error>\n")
            for pid, err in failures:
                f.write(f"{pid}\t{err}\n")
        logger.warning(
            "Wrote %d failed pid(s) to %s — review before training, then "
            "either re-run with --overwrite for the affected files OR add "
            "them to data.exclude_list.", len(failures), fail_path)

    # Run manifest for downstream traceability.
    manifest = {
        "tool_version": _TOOL_VERSION,
        "made_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_paths": {
            "image_dir": cfg.data.image_dir,
            "label_dir": cfg.data.label_dir,
            "bbox_dir": getattr(cfg.data, "bbox_dir", ""),
            "region_weight_dir": getattr(cfg.data, "region_weight_dir", ""),
        },
        "label_values": label_values,
        "n_total": counters["total"],
        "n_written": counters["written"],
        "n_skipped": counters["skipped"],
        "n_failed": counters["failed"],
        "compress": compress,
        "fg_subsample": fg_subsample,
    }
    with open(out_p / "_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return counters


def _record(
    res: Dict[str, object],
    counters: Dict[str, int],
    timings: List[float],
    sizes: List[int]) -> None:
    """Bookkeeping for both the inline and pool paths."""
    status = res.get("status", "written")
    counters[status] = counters.get(status, 0) + 1
    timings.append(float(res.get("elapsed_s", 0.0)))
    sizes.append(int(res.get("size_bytes", 0)))


def _log_progress(
    done: int, total: int, res: Dict[str, object], t0: float) -> None:
    """Periodic per-sample / batched progress line."""
    if done == 1 or done == total or done % 10 == 0:
        elapsed = time.perf_counter() - t0
        rate = done / max(elapsed, 1e-6)
        eta = (total - done) / max(rate, 1e-6)
        size_mib = float(res.get("size_bytes", 0)) / (1024 ** 2)
        shape = res.get("shape", "-")
        logger.info(
            "[%d/%d] %s pid=%s  shape=%s  %.1f MiB  (%.2fs)  "
            "rate=%.2f sample/s  ETA=%.0fs",
            done, total, res.get("status", "?"),
            res.get("pid", "?"), shape, size_mib,
            float(res.get("elapsed_s", 0.0)),
            rate, eta)


def _short_exc(exc: BaseException, max_len: int = 200) -> str:
    """Single-line length-bounded exception summary for the failure file."""
    msg = f"{type(exc).__name__}: {exc}".replace("\n", " | ").replace("\t", " ")
    return msg[:max_len]


# =============================================================================
# CLI
# =============================================================================
def _setup_logging(level: str = "INFO") -> None:
    fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt, datefmt=datefmt,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-compute per-sample npz packages "
                    "(image+label+rw+fg-index, bbox-cropped).")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config (same one used by train.py).")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Config overrides (key=value, dot notation), "
                             "e.g. --override data.image_dir=F:/x data.label_values=[0,1].")
    parser.add_argument("--out", type=str, required=True,
                        help="Output directory for the npz packages.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel worker processes (0 = inline). "
                             "Each worker peaks at ~1 cropped sample's "
                             "RAM; tune to host memory.")
    parser.add_argument("--fg-subsample", type=int,
                        default=_DEFAULT_FG_SUBSAMPLE,
                        help="Max stored 3D fg coords per sample "
                             "(matches SegDataset3DCubic._build_index).")
    parser.add_argument("--compress", action="store_true",
                        help="Use np.savez_compressed (smaller disk, "
                             "but no shared OS page cache and slower load).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-write existing npz files.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N samples "
                             "(smoke-test; 0=all).")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    _setup_logging(args.log_level)

    cfg = load_config(args.config)
    logger.info("Config loaded from %s", args.config)
    if args.override:
        # Reuse train.py override semantics; lazy import keeps make_data lean.
        from ..train import apply_overrides
        apply_overrides(cfg, args.override)
        cfg.sync()
        cfg.validate()

    counters = prepare_dataset(
        cfg=cfg,
        out_dir=args.out,
        workers=args.workers,
        fg_subsample=args.fg_subsample,
        compress=args.compress,
        overwrite=args.overwrite,
        limit=args.limit,
    )
    return 0 if counters["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
