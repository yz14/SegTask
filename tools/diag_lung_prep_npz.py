"""Diagnose ``F:/BaiduNetdiskDownload/lung_prep`` npz contents.

TODO-2 hypothesis: in npz mode the runtime IGNORES ``data.bbox_dir`` and
``data.region_weight_dir``; rw / bbox are baked into the npz at
``make_data`` time. If the npz is stale (built before lung_weight was
set up) the per-sample rw is MISSING and the loss falls back to the
static ``loss.region_weights`` map, which is very different from the
user's hand-annotated 9/7/4/14 boundary weights → both 2.5D AND 3D see
the SAME wrong weight signal → identical bad cases.

This script opens one (or a few) npz files and prints exactly what
``SegDataset3D`` would consume at runtime, so the hypothesis can be
confirmed or rejected from data — not by reasoning.

Run:
    D:/miniconda/envs/torch27_env/python.exe tools/diag_lung_prep_npz.py

Optional: ``--n N`` inspects N samples (default 3); ``--all`` aggregates
rw / label histograms across the whole directory (slower, ~30s for 1k
files thanks to mmap).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import Counter

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

NPZ_DIR_DEFAULT = "F:/BaiduNetdiskDownload/lung_prep"


def _summarise_array(name: str, arr: np.ndarray, max_unique_print: int = 30) -> None:
    print(f"    {name}: shape={arr.shape} dtype={arr.dtype}")
    if arr.size == 0:
        print(f"      (empty)")
        return
    flat = arr.ravel()
    finite = flat[np.isfinite(flat)] if np.issubdtype(arr.dtype, np.floating) else flat
    if finite.size == 0:
        print(f"      all non-finite")
        return
    print(f"      min={float(finite.min()):.4f}  max={float(finite.max()):.4f}  "
          f"mean={float(finite.mean()):.4f}")
    # Cheap discrete-value detector (works on int and rounded float arrays).
    if np.issubdtype(arr.dtype, np.integer) or (
            np.issubdtype(arr.dtype, np.floating)
            and float(finite.max() - finite.min()) <= 50):
        rounded = np.round(finite).astype(np.int64)
        unique, counts = np.unique(rounded, return_counts=True)
        if len(unique) <= max_unique_print:
            total = counts.sum()
            print(f"      unique values (count, ratio):")
            for u, c in zip(unique.tolist(), counts.tolist()):
                print(f"        {u:>6d} : {c:>12d}  ({100.0 * c / total:6.3f}%)")
        else:
            print(f"      {len(unique)} unique values (too many to list)")


def _peek_one(npz_path: Path) -> dict:
    print("=" * 80)
    print(f"PID FILE : {npz_path.name}")
    print(f"FULL PATH: {npz_path}")
    print(f"SIZE     : {npz_path.stat().st_size / (1<<20):.2f} MiB")

    with np.load(str(npz_path), allow_pickle=True) as f:
        keys = list(f.files)
        print(f"NPZ KEYS : {keys}")

        # ---- image ----
        if "image" in f.files:
            img = f["image"]
            print("  [image]")
            _summarise_array("image", img)
        else:
            print("  [image]  MISSING — npz is broken")
            img = None

        # ---- label ----
        if "label" in f.files:
            lbl = f["label"]
            print("  [label]")
            _summarise_array("label", lbl)
        else:
            print("  [label]  MISSING — npz is broken")
            lbl = None

        # ---- rw (the smoking gun) ----
        has_rw = "rw" in f.files
        print(f"  [rw]    {'present' if has_rw else 'MISSING (will fall back to static loss.region_weights)'}")
        rw = None
        if has_rw:
            rw = f["rw"]
            _summarise_array("rw", rw)
            print("      NOTE: ``load_region_weight_volume`` does +1 shift at "
                  "make_data time. User-declared raw weights {0, 4, 7, 9, 14} "
                  "should appear here as {1, 5, 8, 10, 15}.")

        # ---- fg_slices / fg_coords ----
        if "fg_slices" in f.files:
            fs = f["fg_slices"]
            print(f"  [fg_slices] count={len(fs)}  "
                  f"range=[{int(fs.min()) if len(fs) else None}, "
                  f"{int(fs.max()) if len(fs) else None}]")
        if "fg_coords" in f.files:
            fc = f["fg_coords"]
            print(f"  [fg_coords] shape={fc.shape}")

        # ---- meta ----
        if "meta" in f.files:
            try:
                meta = f["meta"].item()
                print("  [meta]")
                for k, v in meta.items():
                    print(f"      {k} = {v}")
            except Exception as e:
                print(f"  [meta]  unreadable: {e}")

    # Return summary for aggregation
    return {
        "has_rw": has_rw,
        "label_unique": (np.unique(lbl).tolist() if lbl is not None else None),
        "rw_unique": (
            np.unique(np.round(rw).astype(np.int64)).tolist()
            if rw is not None and rw.size > 0 else None),
        "rw_min": float(rw.min()) if rw is not None and rw.size else None,
        "rw_max": float(rw.max()) if rw is not None and rw.size else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default=NPZ_DIR_DEFAULT)
    ap.add_argument("--n", type=int, default=3,
                    help="number of npz files to dump in full (default 3)")
    ap.add_argument("--all", action="store_true",
                    help="aggregate rw-presence stats across the whole dir")
    args = ap.parse_args()

    npz_dir = Path(args.npz_dir)
    if not npz_dir.is_dir():
        print(f"ERROR: npz_dir does not exist: {npz_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(npz_dir.glob("*.npz"))
    # Skip the make_data manifest if present.
    files = [p for p in files if not p.name.startswith("_")]
    print(f"npz_dir   = {npz_dir}")
    print(f"total npz = {len(files)}")
    if not files:
        return

    n_dump = min(args.n, len(files))
    summaries = []
    for i in range(n_dump):
        s = _peek_one(files[i])
        summaries.append(s)

    if args.all:
        print("=" * 80)
        print(f"AGGREGATE over {len(files)} npz files …")
        rw_present = 0
        rw_absent = 0
        label_val_counter: Counter = Counter()
        rw_val_counter: Counter = Counter()
        for p in files:
            with np.load(str(p), allow_pickle=True) as f:
                if "rw" in f.files:
                    rw_present += 1
                    rwv = np.round(f["rw"]).astype(np.int64)
                    rw_val_counter.update(np.unique(rwv).tolist())
                else:
                    rw_absent += 1
                if "label" in f.files:
                    lblv = np.unique(f["label"]).tolist()
                    label_val_counter.update(lblv)
        print(f"  rw present     : {rw_present} / {len(files)}")
        print(f"  rw absent      : {rw_absent} / {len(files)}")
        print(f"  union(label.unique) across all files: "
              f"{sorted(label_val_counter.keys())}")
        if rw_present:
            print(f"  union(rw.unique) across all files (post +1 shift):")
            for v, n in sorted(rw_val_counter.items()):
                print(f"    {v:>6d} : appears in {n:>6d} files")
    else:
        print("=" * 80)
        print("SUMMARY (first N samples)")
        for i, s in enumerate(summaries):
            print(f"  [{i}] has_rw={s['has_rw']}  "
                  f"label_unique={s['label_unique']}  "
                  f"rw_unique={s['rw_unique']}")

    print("=" * 80)
    print("INTERPRETATION GUIDE")
    print("  * If `has_rw=True` AND rw_unique includes {1, 5, 8, 10, 15} (= "
          "user's 0/4/7/9/14 + 1 shift), the rw signal is intact — bug is "
          "elsewhere; we move on to loss / training dynamics.")
    print("  * If `has_rw=False`, the npz is stale (built before lung_weight "
          "was wired in). At runtime the loss falls back to the YAML's "
          "`loss.region_weights: [1.0, 4.0, 2.0]` — which gives lung weight 4 "
          "and bone weight 2 EVERYWHERE, no edge / peri-lung / HU-close "
          "emphasis. This is almost certainly your shared bad case across "
          "2.5D and 3D. Fix: rebuild npz with the lung_weight dir wired in.")
    print("  * If `has_rw=True` but rw_unique is e.g. {1, 2, 3} (small "
          "integers ≠ 1/5/8/10/15), the npz was built against a DIFFERENT "
          "region_weight source (older / lighter labelling). Rebuild npz.")
    print("  * Cross-check `meta.src_rw` to see EXACTLY which "
          "region_weight NIfTI was used for that npz package.")


if __name__ == "__main__":
    main()
