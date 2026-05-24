"""Pre-compute per-sample npz packages for fast / OOM-free training.

Motivation
----------
The runtime data pipeline (``SegDataset3D`` & friends) decodes
``image.nii.gz`` + ``label.nii.gz`` + ``region_weight.nii.gz`` PER
sample PER worker and applies the per-sample ROI bbox on the fly.
Even with the streaming ``load_nifti_cropped`` reader, ``.nii.gz`` is
not a seekable stream — ITK must decompress the FULL native-dtype
buffer inside ``Execute()`` before any ROI can be returned. With
``num_workers=4`` × (image, label, region_weight) = 12 concurrent
gzip-decompress peaks, that trivially OOMs a 16 GiB host on large
CTs (the symptom seen in production: ``RuntimeError: bad allocation``
inside ``SimpleITK ImageFileReader_Execute`` on the third concurrent
load).

This module materialises a one-shot offline pre-processing stage
that runs ONCE and writes a per-sample npz package. The training
pipeline then mmaps the npz directly, which:

  * removes the gzip-decompress peak (npz arrays are stored as raw
    ``.npy`` blobs — ``np.load(..., mmap_mode='r')`` returns a memmap
    backed by the OS page cache, SHARED across workers);
  * removes the per-sample bbox-mask read + ``compute_bbox_from_volume``
    pass (bbox is already applied to the stored arrays);
  * removes the dataset's ``_build_index`` / ``precompute_bboxes``
    startup scan (foreground slice indices and 3D coordinates are
    stored in the npz);
  * shrinks the per-sample on-disk footprint ~14× by storing only the
    ROI cube (image / label / rw all share the SAME bbox).

Output contract (per sample)
----------------------------
File: ``<out_dir>/<pid>.npz`` where ``pid = image_filename - image_suffix``.

Mandatory keys::

    image      int16   (D', H', W')   raw HU values, bbox-cropped
    label      int16   (D', H', W')   raw label values, bbox-cropped
    fg_slices  int32   (M,)           z-indices (in the cropped frame!)
                                      where ``label != background``;
                                      empty array when no fg present
    fg_coords  int32   (N, 3)         (d, h, w) coordinates of fg
                                      voxels in the cropped frame,
                                      sub-sampled to ``--fg-subsample``
                                      (default 50000) with seed=42;
                                      empty (0, 3) when no fg
    meta       object 0-d             pickled dict with provenance:
        {'pid', 'src_image', 'src_label', 'src_bbox', 'src_rw',
         'orig_shape':(D,H,W), 'bbox':((d0,d1),(h0,h1),(w0,w1))|None,
         'label_values': [...], 'has_rw': bool, 'rw_shift': 1.0,
         'image_dtype':'int16', 'made_at': iso8601, 'tool_version': str}

Optional key (present iff a ``region_weight_dir`` is configured)::

    rw         int16   (D', H', W')   already-+1-shifted weight map,
              OR fp32                 bbox-cropped (matches
                                      ``load_region_weight_volume``).
                                      int16 when source values fit
                                      (the common case for hand-
                                      annotated weights — saves ~50%
                                      of npz size); fp32 fallback
                                      for non-integer or out-of-
                                      int16-range sources. Runtime
                                      loader always returns fp32
                                      regardless of stored dtype.

Why ``image`` stays raw HU and not pre-normalised:
   ``intensity_min/max/normalize/global_mean/global_std`` are training-
   time hyper-parameters; locking them at preprocessing time would
   force a re-make whenever the windowing changes. The runtime
   ``preprocess_image(inplace=True)`` is one clip + one affine pass
   on the ROI cube — negligible cost on the worker side.

Why ``rw`` is pre-+1-shifted:
   Mirrors ``load_region_weight_volume`` semantics so the runtime
   loader is dtype-/value-equivalent to the legacy NIfTI path with
   no extra arithmetic.

Why ``np.savez`` (NOT ``savez_compressed``) by default:
   The whole point is to avoid decompress peaks. ``np.savez`` writes
   each array as a raw ``.npy`` inside an uncompressed zip, which
   ``np.load(..., mmap_mode='r')`` can memmap directly. With four
   workers reading the SAME npz, the OS page cache is shared — net
   RAM is one copy, not four. Pass ``--compress`` to opt into
   ``savez_compressed`` (smaller on disk; load path falls back to
   in-memory decode and loses the page-cache sharing benefit).

CLI
---
::

    python -m segtask_v1.data.make_data \
        --config configs/seg2_5d.yaml \
        --out F:/path/to/prepared \
        --workers 4

Increment-friendly: existing ``<pid>.npz`` files are skipped unless
``--overwrite`` is set. Failures are isolated per sample and
aggregated into ``<out_dir>/_failures.txt`` (one pid per line) so
they can be plugged straight into ``data.exclude_list``.
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

# Default subsample cap — matches ``SegDataset3DCubic._build_index``
# so the runtime fg-coord pool is identical regardless of the data
# source. Override via CLI for very dense / very sparse datasets.
_DEFAULT_FG_SUBSAMPLE = 50_000


# =============================================================================
# Per-sample worker
# =============================================================================
def _stem(path: str, suffix) -> str:
    """Return ``filename - suffix`` (or single-extension stem if no
    suffix matches). ``suffix`` accepts either a string or a sequence of
    candidate suffixes — the first one whose `name.endswith` matches
    wins. This mirrors the relaxed pairing rule in
    :mod:`segtask_v1.data.loader`, so a single ``pid`` is computed
    consistently across the make/train code paths even when images use
    suffixes like ``.nii``, ``.nii.gz`` or custom variants.
    """
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
    """Compute (fg_slices, fg_coords) from a (D, H, W) label volume.

    Mirrors the in-dataset index builders:
      * ``SegDataset3D._build_index`` records fg slice indices;
      * ``SegDataset3DCubic._build_index`` sub-samples 3D fg coords
        with a fixed RNG seed (42) to a maximum cap (default 50000).

    Both are computed unconditionally so the same npz feeds any
    patch_mode without re-scanning the label at runtime.
    """
    if label.size == 0:
        return (np.zeros((0,), dtype=np.int32),
                np.zeros((0, 3), dtype=np.int32))
    label_int = label.astype(np.int32, copy=False)
    fg_mask = label_int != int(bg_val)
    if not fg_mask.any():
        return (np.zeros((0,), dtype=np.int32),
                np.zeros((0, 3), dtype=np.int32))
    # z-axis index (cropped frame).
    fg_slices = np.where(np.any(fg_mask, axis=(1, 2)))[0].astype(np.int32)
    # Full 3D coords, sub-sampled with the same RNG / cap as
    # ``SegDataset3DCubic._build_index`` so the cubic index is
    # bit-equivalent to the legacy on-the-fly path.
    coords = np.argwhere(fg_mask).astype(np.int32)
    if fg_subsample > 0 and len(coords) > fg_subsample:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(coords), fg_subsample, replace=False)
        coords = coords[idx]
    return fg_slices, coords


def _bbox_from_mask_path(bbox_path: Optional[str]) -> Optional[BBox]:
    """Compute the ROI bbox from a mask NIfTI, mirroring
    ``precompute_bboxes`` (int16 decode → ``compute_bbox_from_volume``).

    Returns ``None`` when the mask is entirely empty or no path was
    supplied — the caller then stores the full volume.
    """
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
    """Materialise the npz package for ONE sample.

    Returns a small status dict with timing + disk-size info — the
    multiprocessing collector uses it for the aggregate progress log.

    Idempotent: if ``out_path`` already exists and ``overwrite`` is
    False, the function returns immediately with ``status='skipped'``.
    Errors are caught and re-raised by the caller (per-sample isolation
    is done at the executor level).
    """
    out_p = Path(out_path)
    if out_p.is_file() and not overwrite:
        return {"pid": pid, "status": "skipped",
                "size_bytes": out_p.stat().st_size, "elapsed_s": 0.0}

    t0 = time.perf_counter()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # --- 1. Compute bbox (independent NIfTI; small int16 decode) ---
    bbox = _bbox_from_mask_path(bbox_path)

    # --- 2. Load image as int16 HU (raw) cropped to bbox -----------
    # int16 is the native CT storage dtype; load_nifti_cropped reads
    # the stored dtype natively (no float promotion) when an integer
    # output is requested. Raw HU is preserved verbatim.
    image = load_nifti_cropped(image_path, bbox=bbox, dtype=np.int16)

    # --- 3. Load label int16 cropped to bbox -----------------------
    label = load_nifti_cropped(label_path, bbox=bbox, dtype=np.int16)

    if image.shape != label.shape:
        raise ValueError(
            f"image shape {image.shape} != label shape {label.shape} for "
            f"pid={pid} (image={image_path}, label={label_path})")

    # --- 4. Load region weight (+1 shifted) cropped to bbox -------
    # Source NIfTI is hand-annotated (background=0, fg=integer
    # weight) → after +1 shift the entire volume is small non-
    # negative integers. Storing as int16 instead of fp32 cuts the
    # rw payload 4× on disk (~50% of total npz size on lung_weight).
    # The runtime loader (``load_npz_region_weight``) casts back to
    # fp32 on read, so downstream behaviour is bit-equivalent.
    # Defensive guard: if a future weight source ever produces non-
    # integer or out-of-int16-range values, fall back to fp32 so we
    # never lose precision silently.
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
                "(min=%.3f, max=%.3f, integer_valued=%s) — storing "
                "as float32. Disk usage for this sample is 4x higher "
                "than int16-storable samples.",
                pid, rw_min, rw_max, is_integer_valued)

    # --- 5. Foreground indices in the CROPPED frame ----------------
    bg_val = int(label_values[0])
    fg_slices, fg_coords = _compute_fg_indices(label, bg_val, fg_subsample)

    # --- 6. Provenance metadata ------------------------------------
    # Original (uncropped) image shape — read from the SimpleITK header
    # via a header-only pass would be tidier, but we already have
    # ``bbox`` (the half-open cropped extents in the original frame)
    # which is sufficient for any reverse mapping the predictor
    # might want later. We also stash the source paths so the npz
    # is self-describing for debugging.
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
        "rw_dtype": rw_dtype_stored,    # 'int16' / 'float32' / None
        "image_dtype": str(image.dtype),
        "made_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool_version": _TOOL_VERSION,
    }
    meta_arr = np.array(meta, dtype=object)

    # --- 7. Write npz atomically (write-to-tmp + rename) -----------
    # ``np.savez(_compressed)`` does not stream — it builds the entire
    # zip in memory before flushing. For our ROI sizes (≤ a few 100
    # MiB) that is fine, and the in-process peak is bounded by the
    # arrays we already hold. We write to ``out_path.tmp`` then rename
    # so an interrupted run never leaves a half-written npz behind.
    # Pass an OPEN FILE HANDLE to np.savez — when given a string/Path,
    # numpy auto-appends ``.npz`` to the filename, which silently
    # turns ``foo.npz.tmp`` into ``foo.npz.tmp.npz`` on disk and
    # breaks the subsequent rename. The file-object overload writes
    # to exactly the path we open.
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
    # Windows requires the destination to NOT exist before rename.
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
    """Discover image/label/bbox/rw paths from a Config and return a
    list of per-sample dicts ready for ``prepare_one``.

    Re-uses the loader's discovery / matching helpers verbatim so the
    npz pipeline stays in lockstep with the runtime pipeline (same
    pid convention, same strong-contract bbox / rw matching, same
    exclude-list filtering).
    """
    dc = cfg.data

    # Pair up images and labels.
    image_paths, label_paths = discover_samples(
        dc.image_dir, dc.label_dir, dc.image_suffix, dc.label_suffix)

    # Honour the same exclude list as the runtime trainer.
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

    Args:
        cfg:           Config object (same one trainer uses).
        out_dir:       Output directory; created if missing.
        workers:       Parallel worker processes (each peaks at one
                       cropped sample's ROI buffer; tune to host RAM).
                       Pass 0 to run inline (single-process — useful
                       for debugging tracebacks).
        fg_subsample:  Cap on stored 3D fg coordinates per sample.
        compress:      Use ``np.savez_compressed`` (slower load,
                       smaller disk; loses memmap shareability).
        overwrite:     Re-write existing npz files.
        limit:         If > 0, only process the first N samples
                       (smoke-test mode).

    Returns:
        Counters dict ``{written, skipped, failed, total}``.
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

    # Pre-compute output paths and skip-list to make the progress log
    # accurate (skipped count visible from the start).
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
        # Inline path — easier to debug (full traceback, no pickling).
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
        # Process pool. ``ProcessPoolExecutor`` uses spawn on Windows;
        # the per-worker import cost (SimpleITK) is paid once per
        # worker and amortised across many samples.
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

    # Persist the failure list — directly compatible with
    # ``data.exclude_list`` (one pid per line, # comments allowed).
    # Always clear the stale file when the new run produces no
    # failures so a successful re-run leaves no misleading
    # ``_failures.txt`` behind.
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

    # Manifest: small json summarising the run for downstream traceability.
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
    """Single-line, length-bounded exception summary for the CSV-ish
    failure file."""
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
        # Reuse train.py's override semantics so users can swap
        # image_dir / label_dir / bbox_dir on the fly without
        # editing the yaml — useful for smoke-testing on a
        # different small_data folder. Imported lazily to keep the
        # ``make_data`` import path free of trainer dependencies.
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
