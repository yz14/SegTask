"""Scan a dataset for NIfTI files that SimpleITK cannot read.

Typical symptom:
    ``RuntimeError: ITK only supports orthonormal direction cosines.
    No orthonormal definition found!``

This happens when a NIfTI's qform / sform matrix is *not* orthonormal
(numerical drift or non-standard orientations). SimpleITK aborts the
read outright, which crashes training. The cheapest fix is to exclude
those samples from training: run this script once, feed its
``bad_pids.txt`` into ``data.exclude_list`` in the YAML config.

Usage
-----
By config (recommended — matches exactly what training sees):
    python tools/scan_bad_nifti.py --config configs/seg2_5d.yaml \\
        --out tools/bad_seg2_5d

By explicit dirs (config not needed):
    python tools/scan_bad_nifti.py \\
        --image-dir F:/path/to/nii --label-dir F:/path/to/masks \\
        --bbox-dir  F:/path/to/bbox \\
        --image-suffix .nii.gz --label-suffix .nii.gz --bbox-suffix .nii.gz \\
        --out tools/bad_dataset

Outputs (under ``--out`` directory):
    bad_pids.txt   one pid per line (stem, `.nii.gz` stripped)
    bad_files.csv  role, pid, path, error   (role in {image,label,bbox})
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import SimpleITK as sitk

# Make the repo root importable so we can reuse the project config loader
# when --config is provided, without forcing the user to set PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("scan_bad_nifti")


# ---------------------------------------------------------------------------
# Core probe
# ---------------------------------------------------------------------------
def _try_read(path: str) -> Optional[str]:
    """Attempt to read a NIfTI with SimpleITK. Returns an error message on
    failure, or ``None`` on success.

    We call ``ReadImage(path)`` without forcing a pixel type — this is the
    cheapest probe that still triggers the header validation that raises
    the "orthonormal direction cosines" error. Decoding pixels into a
    specific dtype (as ``load_nifti`` does in training) would surface the
    same header errors, so there's no need to replicate it here.
    """
    try:
        sitk.ReadImage(path)
        return None
    except RuntimeError as exc:
        # Flatten the (often multi-line) ITK error to a single line so the
        # CSV stays grep-able.
        return " ".join(str(exc).split())
    except Exception as exc:  # pragma: no cover - defensive
        return f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def _stem(path: Path, suffix: str) -> str:
    """``<pid>{suffix}`` → ``<pid>``. Falls back to ``Path.stem`` when the
    filename does not end with the expected suffix."""
    name = path.name
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def _list_niftis(root: Path, suffix: str) -> List[Path]:
    return sorted(root.glob(f"*{suffix}"))


# ---------------------------------------------------------------------------
# Scan driver
# ---------------------------------------------------------------------------
def scan_role(
    role: str,
    paths: List[Path],
    suffix: str,
    workers: int) -> List[Tuple[str, str, str, str]]:
    """Return a list of ``(role, pid, path, error)`` tuples for all files
    under ``paths`` that fail to read."""
    bad: List[Tuple[str, str, str, str]] = []
    n = len(paths)
    logger.info("[%s] scanning %d file(s) with %d worker(s)...",
                role, n, workers)

    def _probe(p: Path) -> Tuple[Path, Optional[str]]:
        return p, _try_read(str(p))

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_probe, p): p for p in paths}
        for fut in as_completed(futs):
            p, err = fut.result()
            done += 1
            if err is not None:
                pid = _stem(p, suffix)
                bad.append((role, pid, str(p), err))
                logger.warning("[%s] BAD %s — %s", role, p.name, err[:180])
            # Light progress pulse every 5%.
            if done % max(1, n // 20) == 0:
                logger.info("[%s] %d / %d (%d bad so far)",
                            role, done, n, len(bad))
    logger.info("[%s] done. %d bad file(s) out of %d.", role, len(bad), n)
    return bad


# ---------------------------------------------------------------------------
# Config loader (optional)
# ---------------------------------------------------------------------------
def _load_from_config(cfg_path: str) -> Dict[str, str]:
    """Pull the relevant dirs / suffixes out of a project YAML config.
    Avoids running the full validate() (which may reject unrelated legacy
    keys) by parsing the YAML directly.
    """
    import yaml
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    data = raw.get("data", {}) or {}
    return {
        "image_dir":     data.get("image_dir", "") or "",
        "label_dir":     data.get("label_dir", "") or "",
        "bbox_dir":      data.get("bbox_dir", "")  or "",
        "image_suffix":  data.get("image_suffix", ".nii.gz"),
        "label_suffix":  data.get("label_suffix", ".nii.gz"),
        "bbox_suffix":   data.get("bbox_suffix", ".nii.gz"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Scan a dataset for SimpleITK-unreadable NIfTI files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--config", type=str, default="",
                    help="Path to a project YAML config. If set, image/label/"
                         "bbox dirs and suffixes are taken from `data.*`.")
    ap.add_argument("--image-dir", type=str, default="")
    ap.add_argument("--label-dir", type=str, default="")
    ap.add_argument("--bbox-dir",  type=str, default="")
    ap.add_argument("--image-suffix", type=str, default=".nii.gz")
    ap.add_argument("--label-suffix", type=str, default=".nii.gz")
    ap.add_argument("--bbox-suffix",  type=str, default=".nii.gz")
    ap.add_argument("--out", type=str, required=True,
                    help="Output directory (bad_pids.txt + bad_files.csv).")
    ap.add_argument("--workers", type=int,
                    default=max(4, min(16, (os.cpu_count() or 8))),
                    help="Thread-pool size for parallel probing.")
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S")

    # Resolve dirs: config takes precedence; CLI flags override per-field.
    cfg_dirs: Dict[str, str] = {}
    if args.config:
        cfg_dirs = _load_from_config(args.config)
        logger.info("Loaded dirs from config %s: %s", args.config, cfg_dirs)

    def pick(cli_val: str, cfg_key: str) -> str:
        if cli_val:
            return cli_val
        return cfg_dirs.get(cfg_key, "")

    image_dir    = pick(args.image_dir,    "image_dir")
    label_dir    = pick(args.label_dir,    "label_dir")
    bbox_dir     = pick(args.bbox_dir,     "bbox_dir")
    image_suffix = pick(args.image_suffix, "image_suffix") or ".nii.gz"
    label_suffix = pick(args.label_suffix, "label_suffix") or ".nii.gz"
    bbox_suffix  = pick(args.bbox_suffix,  "bbox_suffix")  or ".nii.gz"

    roles: List[Tuple[str, str, str]] = []  # (role, dir, suffix)
    if image_dir:
        roles.append(("image", image_dir, image_suffix))
    if label_dir:
        roles.append(("label", label_dir, label_suffix))
    if bbox_dir:
        roles.append(("bbox",  bbox_dir,  bbox_suffix))
    if not roles:
        ap.error("No directories specified. Pass --config or --image-dir / "
                 "--label-dir / --bbox-dir.")

    all_bad: List[Tuple[str, str, str, str]] = []
    for role, d, suf in roles:
        p = Path(d)
        if not p.is_dir():
            logger.warning("[%s] directory not found, skipping: %s", role, p)
            continue
        files = _list_niftis(p, suf)
        if not files:
            logger.warning("[%s] no *%s files under %s", role, suf, p)
            continue
        all_bad.extend(scan_role(role, files, suf, args.workers))

    # Aggregate pids (union across roles — excluding a sample if *any* of
    # its files is unreadable is the only safe policy for training).
    bad_pids = sorted({pid for _role, pid, _path, _err in all_bad})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pids_file = out_dir / "bad_pids.txt"
    with open(pids_file, "w", encoding="utf-8") as f:
        f.write("# Auto-generated by tools/scan_bad_nifti.py\n")
        f.write(f"# Unreadable NIfTI samples across roles: "
                f"{', '.join(r for r, _, _ in roles)}\n")
        for pid in bad_pids:
            f.write(f"{pid}\n")

    csv_file = out_dir / "bad_files.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["role", "pid", "path", "error"])
        for row in sorted(all_bad):
            w.writerow(row)

    logger.info("=" * 60)
    logger.info("Bad samples: %d unique pid(s) across %d file failure(s).",
                len(bad_pids), len(all_bad))
    logger.info("Wrote: %s", pids_file)
    logger.info("Wrote: %s", csv_file)
    logger.info("Next step: set `data.exclude_list: \"%s\"` in your YAML.",
                pids_file.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
