"""Diagnose the "lung-bottom is consistently under-predicted" symptom.

Given one or more npz packages from ``F:/BaiduNetdiskDownload/lung_prep``,
locate the most-caudal and most-cranial slice that contains lung
(``label == 1``) and dump per-slice ``(label × rw)`` joint histograms in
a small z-band on each end.

Two failure modes we are trying to disambiguate (TODO-2):

  Hypothesis E (label under-annotation at the diaphragm)
      The GT itself misses lung tissue near the costodiaphragmatic
      recess. We will see ``label=0`` in regions where the image HU
      and surrounding tissue suggest lung. Hard to diagnose from npz
      alone (need image inspection too).

  Hypothesis F (rw=20 zone suppresses lung predictions at the lung
  bottom)
      The high-emphasis "HU close to lung" weight zone (rw=20 after
      the +1 shift, raw 19 in the lung_weight NIfTI) overlaps the
      lung-bottom region. With rw=20 multiplier on a (label=0)
      target, the loss yells "do NOT predict lung here" 20× louder
      than ordinary background. If this band wraps around the
      true lung bottom, training systematically suppresses the
      model's lung output at the diaphragm.

Run:
    D:/miniconda/envs/torch27_env/python.exe tools/diag_lung_bottom_npz.py

Optional flags:
    --n N        : inspect N npz files (default 3, capped at total)
    --band K     : ± K slices around the lung-edge slice (default 4)
    --only-rw    : suppress per-(label, rw) cross-tab; show rw band stats only

Outputs are designed to be skim-friendly: each volume produces ~30 lines
of compact tables, no images, < 1 s per volume on lung_prep.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

NPZ_DIR_DEFAULT = "F:/BaiduNetdiskDownload/lung_prep"
LUNG_LABEL = 1     # cfg.data.label_values = [0, 1, 2] => 1=lung, 2=bone
BONE_LABEL = 2


def _xtab_label_rw(label_slc: np.ndarray, rw_slc: np.ndarray) -> str:
    """Compact ``(label × rw)`` cross-tabulation for one slice."""
    rw_int = np.round(rw_slc).astype(np.int32)
    rw_unique = np.unique(rw_int)
    lbl_unique = np.unique(label_slc)
    total = label_slc.size

    # Header
    cols = "  ".join(f"rw={int(v):>2d}" for v in rw_unique)
    out_lines = [f"      label \\ rw : {cols}    sum"]
    for lbl in lbl_unique:
        cells = []
        row_total = 0
        for rv in rw_unique:
            n = int(((label_slc == lbl) & (rw_int == rv)).sum())
            cells.append(f"{n:>6d}")
            row_total += n
        out_lines.append(
            f"      label={int(lbl):>2d}    : "
            + "  ".join(cells)
            + f"   {row_total:>7d} "
            f"({100.0 * row_total / total:5.2f}%)")
    # Column totals
    col_totals = []
    for rv in rw_unique:
        n = int((rw_int == rv).sum())
        col_totals.append(f"{n:>6d}")
    out_lines.append(
        "      sum         : "
        + "  ".join(col_totals)
        + f"   {total:>7d}")
    return "\n".join(out_lines)


def _zband_summary(
        label: np.ndarray, rw: np.ndarray, image: np.ndarray,
        z_indices: list[int], tag: str, only_rw: bool) -> None:
    print(f"  {tag}  (z indices: {z_indices})")
    for z in z_indices:
        if z < 0 or z >= label.shape[0]:
            print(f"    z={z}: out of range")
            continue
        l = label[z]
        r = rw[z]
        i = image[z]
        n_lung = int((l == LUNG_LABEL).sum())
        n_bone = int((l == BONE_LABEL).sum())
        n_bg = int((l == 0).sum())
        # rw=20 voxels at this slice (the high-weight "HU close to lung"
        # zone). Their label distribution is the smoking gun for hyp F.
        rw_int = np.round(r).astype(np.int32)
        m20 = (rw_int == 20)
        n_rw20 = int(m20.sum())
        n_rw20_lung = int((m20 & (l == LUNG_LABEL)).sum())
        n_rw20_bg = int((m20 & (l == 0)).sum())
        # Image stats inside the rw=20 band
        if n_rw20 > 0:
            i_mean_rw20 = float(i[m20].mean())
            i_min_rw20 = int(i[m20].min())
            i_max_rw20 = int(i[m20].max())
            rw20_str = (
                f"  rw=20 voxels: {n_rw20:>6d} "
                f"({100*n_rw20/l.size:5.2f}%)  "
                f"of which label=lung:{n_rw20_lung:>5d} "
                f"label=bg:{n_rw20_bg:>6d}  "
                f"img_HU[mean={i_mean_rw20:6.1f} "
                f"range={i_min_rw20:>5d}..{i_max_rw20:>5d}]")
        else:
            rw20_str = "  rw=20 voxels: 0"

        print(f"    z={z:>3d}  label_counts: bg={n_bg} lung={n_lung} "
              f"bone={n_bone}  {rw20_str}")
        if not only_rw:
            print(_xtab_label_rw(l, r))


def _peek_one(npz_path: Path, band: int, only_rw: bool) -> None:
    print("=" * 80)
    print(f"PID FILE : {npz_path.name}")
    with np.load(str(npz_path), allow_pickle=True) as f:
        if not all(k in f.files for k in ("image", "label", "rw")):
            print(f"  SKIP: missing one of (image, label, rw)")
            return
        image = f["image"][:]
        label = f["label"][:]
        rw = f["rw"][:].astype(np.float32)

    D, H, W = label.shape
    print(f"  shape (D, H, W) = {(D, H, W)}")

    # Per-slice lung-mass: number of lung voxels in each axial slice.
    lung_per_slice = ((label == LUNG_LABEL).reshape(D, -1).sum(axis=1))
    lung_slices = np.where(lung_per_slice > 0)[0]
    if lung_slices.size == 0:
        print(f"  no lung voxels found — skipping bottom/top inspection")
        return

    z_lung_min = int(lung_slices[0])      # most-caudal lung slice
    z_lung_max = int(lung_slices[-1])     # most-cranial lung slice
    print(f"  lung exists on z ∈ [{z_lung_min}, {z_lung_max}]  "
          f"(span = {z_lung_max - z_lung_min + 1} of {D} total)")
    print(f"  lung voxels per slice — first / last 6:")
    for z in lung_slices[:6]:
        print(f"    z={int(z):>3d}: {int(lung_per_slice[z]):>7d}")
    print(f"    ...")
    for z in lung_slices[-6:]:
        print(f"    z={int(z):>3d}: {int(lung_per_slice[z]):>7d}")

    # ---- Bottom band (the symptom is here) ----------------------------
    # Most-caudal: z_lung_min and the ``band`` slices on either side. We
    # look BELOW the lung start (z < z_lung_min) too, because if rw=20
    # leaks into z < z_lung_min and labels there are bg, that's exactly
    # the suppression band that eats into lung-bottom predictions.
    bottom_z = list(range(
        max(0, z_lung_min - band),
        min(D, z_lung_min + band + 1)))
    _zband_summary(
        label, rw, image, bottom_z,
        f"=== LUNG BOTTOM band (z = lung_min ± {band}) ===", only_rw)

    # ---- Top band (control) -------------------------------------------
    top_z = list(range(
        max(0, z_lung_max - band),
        min(D, z_lung_max + band + 1)))
    _zband_summary(
        label, rw, image, top_z,
        f"=== LUNG TOP band (z = lung_max ± {band}, control) ===", only_rw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default=NPZ_DIR_DEFAULT)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--band", type=int, default=4)
    ap.add_argument("--only-rw", action="store_true")
    args = ap.parse_args()

    npz_dir = Path(args.npz_dir)
    files = [p for p in sorted(npz_dir.glob("*.npz"))
             if not p.name.startswith("_")]
    print(f"npz_dir   = {npz_dir}")
    print(f"total npz = {len(files)}")
    if not files:
        return

    n = min(args.n, len(files))
    for i in range(n):
        _peek_one(files[i], band=args.band, only_rw=args.only_rw)

    print("=" * 80)
    print("INTERPRETATION")
    print("  * If the LUNG BOTTOM band shows non-trivial rw=20 voxels with "
          "`label=bg` immediately ABOVE z_lung_min (i.e., inside what *should* "
          "be lung but is annotated as bg with weight-20 suppression):")
    print("       → Hypothesis F is confirmed: rw=20 labelling stamps over "
          "true lung tissue at the lung-bottom; loss penalises lung "
          "predictions there 20× harder than elsewhere → systematic "
          "under-prediction. Same for both 2.5D and 3D.")
    print("  * If the rw=20 band sits OUTSIDE z_lung_min (only below it) and "
          "the lung-bottom slice itself has rich lung labels, F is rejected. "
          "Move on to checking GT under-annotation (E) by visualising the "
          "actual NIfTI overlay near the diaphragm.")
    print("  * Cross-check the LUNG TOP band: if rw=20 behaves DIFFERENTLY at "
          "the top (e.g., far less rw=20 there), it confirms the rw=20 "
          "annotation is anatomy-asymmetric — possibly a labelling protocol "
          "asymmetry rather than a code bug.")


if __name__ == "__main__":
    main()
