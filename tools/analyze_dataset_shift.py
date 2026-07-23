"""数据集域差异分析（TODO #5）。

对比"训练集"与"推理集"两个 NIfTI 目录在 spacing / 物理 FOV / 强度分布 三个维度的差异，
用于定位跨数据集推理假阳偏多的根因。

为什么需要它：SegTask 的 z_axis / 2.5D 管线**不做物理 spacing 重采样**——z 轴按体素取
patch_D 张切片，面内只 resize 到 patch_H×patch_W。因此模型学到的是训练集的"解剖体素尺度"。
推理集若 spacing / FOV / 强度单位不同，进模型前的实际解剖尺度与强度分布就会偏移，导致假阳。

本脚本复用 taskcore.data.dataset 的 preprocess_image / resize_3d（或本地等价实现），模拟"进模型前"的有效尺度，量化两集差异。

用法::

    D:\\miniconda\\envs\\torch27_env\\python.exe scripts/analyze_dataset_shift.py \\
        --train-dir F:/Totalsegmentator_dataset_v201/nii \\
        --infer-dir F:/airway_segment_with_img/imgs \\
        --out-dir   scripts/dataset_shift_report \\
        --workers 8

默认强度窗 / 归一化 / patch_size 取自 configs/segtest0.yaml。
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger("analyze_dataset_shift")

# ---------------------------------------------------------------------------
# 直方图固定 bin（强度，落在窗内）——所有文件共用以便逐文件直方图可平均聚合。
# ---------------------------------------------------------------------------
HIST_BINS = 64

# 强度子采样上限（每卷展平后最多取这么多体素做统计，控制内存/耗时）。
INTENSITY_SUBSAMPLE_CAP = 2_000_000


@dataclass
class FileStats:
    """单卷的几何 + 强度统计（可序列化为 CSV 行）。"""

    dataset: str
    pid: str
    path: str
    # 几何（sitk spacing/size 顺序为 (x, y, z)）
    sx: float
    sy: float
    sz: float
    nx: int
    ny: int
    nz: int
    fov_x: float  # 物理 FOV = size * spacing (mm)
    fov_y: float
    fov_z: float
    # 强度（窗内不裁剪的原始 HU 统计）
    raw_min: float
    raw_max: float
    mean: float
    std: float
    q01: float
    q50: float
    q99: float
    frac_below: float  # < intensity_min 的体素占比（窗下饱和）
    frac_above: float  # > intensity_max 的体素占比（窗上饱和）
    # 派生：进模型前的"有效"物理尺度
    eff_spacing_h: float  # 面内 resize 到 pH 后单像素物理尺寸 (mm) = fov_y / pH
    eff_spacing_w: float  # = fov_x / pW
    slab_z_mm: float      # patch_D 张切片覆盖的物理厚度 = patch_D * sz


def _subsample_flat(arr: np.ndarray, cap: int) -> np.ndarray:
    """展平 + 等距抽样到 <= cap 个体素，避免大卷统计 OOM。"""
    flat = arr.reshape(-1)
    if flat.size <= cap:
        return flat
    step = int(flat.size // cap) + 1
    return flat[::step]


def scan_one(
    path_str: str,
    dataset: str,
    image_suffix: str,
    intensity_min: float,
    intensity_max: float,
    patch_d: int,
    patch_h: int,
    patch_w: int,
) -> Tuple[Optional[Dict], Optional[np.ndarray], Optional[str]]:
    """扫描单卷。返回 (stats_dict, hist_counts, error)。

    * stats_dict: FileStats 的 asdict（成功时）
    * hist_counts: 窗内 HIST_BINS 长直方图计数（density 归一化前的原始计数）
    * error: 失败原因字符串（成功时 None）
    """
    path = Path(path_str)
    pid = path.name
    if image_suffix and pid.endswith(image_suffix):
        pid = pid[: -len(image_suffix)]
    try:
        img = sitk.ReadImage(path_str)
    except Exception as e:  # noqa: BLE001 — 单卷失败不应中断整批
        return None, None, f"{path_str}: read failed: {e}"

    sx, sy, sz = (float(v) for v in img.GetSpacing())   # (x, y, z)
    nx, ny, nz = (int(v) for v in img.GetSize())          # (x, y, z)

    arr = sitk.GetArrayViewFromImage(img)                 # (z, y, x)
    sample = _subsample_flat(np.asarray(arr), INTENSITY_SUBSAMPLE_CAP)
    sample = sample.astype(np.float32, copy=False)

    raw_min = float(sample.min())
    raw_max = float(sample.max())
    mean = float(sample.mean())
    std = float(sample.std())
    q01, q50, q99 = (float(v) for v in np.quantile(sample, [0.01, 0.5, 0.99]))
    frac_below = float((sample < intensity_min).mean())
    frac_above = float((sample > intensity_max).mean())

    # 窗内直方图（裁剪到窗，超出归入边界 bin）——逐文件可平均聚合。
    hist_counts, _ = np.histogram(
        np.clip(sample, intensity_min, intensity_max),
        bins=HIST_BINS, range=(intensity_min, intensity_max))

    stats = FileStats(
        dataset=dataset, pid=pid, path=path_str,
        sx=sx, sy=sy, sz=sz, nx=nx, ny=ny, nz=nz,
        fov_x=nx * sx, fov_y=ny * sy, fov_z=nz * sz,
        raw_min=raw_min, raw_max=raw_max, mean=mean, std=std,
        q01=q01, q50=q50, q99=q99,
        frac_below=frac_below, frac_above=frac_above,
        eff_spacing_h=(ny * sy) / float(patch_h),
        eff_spacing_w=(nx * sx) / float(patch_w),
        slab_z_mm=patch_d * sz,
    )
    return asdict(stats), hist_counts.astype(np.float64), None


def _gather_niftis(directory: str) -> List[str]:
    """递归收集目录下所有 .nii / .nii.gz。"""
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(directory)
    files = sorted(
        str(p) for p in root.rglob("*")
        if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz")))
    if not files:
        raise FileNotFoundError(f"No .nii/.nii.gz under {directory}")
    return files


def scan_dataset(
    directory: str,
    dataset: str,
    image_suffix: str,
    intensity_min: float,
    intensity_max: float,
    patch_d: int,
    patch_h: int,
    patch_w: int,
    workers: int,
    limit: int,
) -> Tuple[List[Dict], np.ndarray, List[str]]:
    """并行扫描一个数据集。返回 (rows, mean_hist_density, errors)。"""
    files = _gather_niftis(directory)
    if limit > 0:
        files = files[:limit]
    logger.info("[%s] scanning %d files (workers=%d) under %s",
                dataset, len(files), workers, directory)

    rows: List[Dict] = []
    hist_accum = np.zeros(HIST_BINS, dtype=np.float64)
    n_hist = 0
    errors: List[str] = []

    def _handle(stats, hist, err):
        nonlocal hist_accum, n_hist
        if err is not None:
            errors.append(err)
            return
        rows.append(stats)
        # 逐文件 density 归一化后累加，避免大卷主导聚合直方图。
        total = hist.sum()
        if total > 0:
            hist_accum += hist / total
            n_hist += 1

    common = dict(
        dataset=dataset, image_suffix=image_suffix,
        intensity_min=intensity_min, intensity_max=intensity_max,
        patch_d=patch_d, patch_h=patch_h, patch_w=patch_w)

    if workers <= 1:
        for i, f in enumerate(files, 1):
            _handle(*scan_one(f, **common))
            if i % 50 == 0 or i == len(files):
                logger.info("[%s] %d/%d", dataset, i, len(files))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(scan_one, f, **common): f for f in files}
            done = 0
            for fut in as_completed(futs):
                _handle(*fut.result())
                done += 1
                if done % 50 == 0 or done == len(files):
                    logger.info("[%s] %d/%d", dataset, done, len(files))

    mean_hist = hist_accum / n_hist if n_hist > 0 else hist_accum
    if errors:
        logger.warning("[%s] %d files failed to read (see errors CSV).",
                       dataset, len(errors))
    return rows, mean_hist, errors


# ---------------------------------------------------------------------------
# 聚合 + 输出
# ---------------------------------------------------------------------------
_NUMERIC_METRICS = [
    "sx", "sy", "sz", "fov_x", "fov_y", "fov_z",
    "raw_min", "raw_max", "mean", "std", "q01", "q50", "q99",
    "frac_below", "frac_above",
    "eff_spacing_h", "eff_spacing_w", "slab_z_mm",
]


def _summarize(rows: List[Dict], dataset: str) -> List[Dict]:
    """对每个数值指标算 mean/std/median/q05/q95。"""
    out: List[Dict] = []
    for m in _NUMERIC_METRICS:
        vals = np.asarray([r[m] for r in rows], dtype=np.float64)
        if vals.size == 0:
            continue
        out.append({
            "metric": m, "dataset": dataset, "n": int(vals.size),
            "mean": float(vals.mean()), "std": float(vals.std()),
            "median": float(np.median(vals)),
            "q05": float(np.quantile(vals, 0.05)),
            "q95": float(np.quantile(vals, 0.95)),
        })
    return out


def _write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote %s (%d rows)", path, len(rows))


def _make_plots(
    out_dir: Path,
    train_rows: List[Dict],
    infer_rows: List[Dict],
    train_hist: np.ndarray,
    infer_hist: np.ndarray,
    train_name: str,
    infer_name: str,
    intensity_min: float,
    intensity_max: float,
) -> None:
    """spacing / FOV / 有效尺度 / 强度直方图 对比图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _arr(rows, key):
        return np.asarray([r[key] for r in rows], dtype=np.float64)

    def _overlay_hist(ax, key, title, xlabel):
        a = _arr(train_rows, key)
        b = _arr(infer_rows, key)
        lo = float(min(a.min(), b.min()))
        hi = float(max(a.max(), b.max()))
        bins = np.linspace(lo, hi, 40) if hi > lo else 40
        ax.hist(a, bins=bins, alpha=0.5, density=True, label=train_name)
        ax.hist(b, bins=bins, alpha=0.5, density=True, label=infer_name)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)

    # 图 1：几何（spacing 三轴 + 有效面内 spacing + slab 物理厚度 + 面内 FOV）
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    _overlay_hist(axes[0, 0], "sx", "In-plane spacing X (mm)", "mm/voxel")
    _overlay_hist(axes[0, 1], "sz", "Slice spacing Z (mm)", "mm/voxel")
    _overlay_hist(axes[0, 2], "fov_x", "In-plane FOV X (mm)", "mm")
    _overlay_hist(axes[1, 0], "eff_spacing_h",
                  "Effective in-plane spacing after resize->H (mm)", "mm/voxel")
    _overlay_hist(axes[1, 1], "slab_z_mm",
                  "Physical z-thickness of patch_D slab (mm)", "mm")
    _overlay_hist(axes[1, 2], "fov_z", "Z FOV (mm)", "mm")
    fig.suptitle("Geometry comparison (key scale-shift indicators)")
    fig.tight_layout()
    p1 = out_dir / "geometry_comparison.png"
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    logger.info("Wrote %s", p1)

    # 图 2：强度（窗内平均直方图 + 窗外饱和占比）
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    centers = np.linspace(intensity_min, intensity_max, HIST_BINS)
    axes[0].plot(centers, train_hist, label=train_name)
    axes[0].plot(centers, infer_hist, label=infer_name)
    axes[0].set_title("Mean intensity histogram (within window)")
    axes[0].set_xlabel("HU")
    axes[0].set_ylabel("mean density")
    axes[0].legend(fontsize=8)

    _overlay_hist(axes[1], "frac_below",
                  f"Frac voxels < {intensity_min:.0f} HU (window saturation)",
                  "fraction")
    _overlay_hist(axes[2], "q50", "Per-volume median HU", "HU")
    fig.suptitle("Intensity comparison")
    fig.tight_layout()
    p2 = out_dir / "intensity_comparison.png"
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    logger.info("Wrote %s", p2)


def _log_verdict(
    train_rows: List[Dict], infer_rows: List[Dict],
    train_name: str, infer_name: str,
) -> None:
    """打印关键差异结论。"""
    def _med(rows, key):
        return float(np.median([r[key] for r in rows]))

    logger.info("=" * 72)
    logger.info("VERDICT — median comparison (%s vs %s)", train_name, infer_name)
    logger.info("-" * 72)
    for key, unit, desc in [
        ("sz", "mm", "slice spacing Z"),
        ("sx", "mm", "in-plane spacing X"),
        ("fov_x", "mm", "in-plane FOV X"),
        ("eff_spacing_h", "mm/vox", "effective in-plane spacing after resize"),
        ("slab_z_mm", "mm", "physical z-thickness of patch_D slab"),
        ("q50", "HU", "per-volume median intensity"),
        ("frac_below", "", "frac voxels below window"),
    ]:
        t, i = _med(train_rows, key), _med(infer_rows, key)
        ratio = (i / t) if t != 0 else float("inf")
        logger.info("  %-42s train=%9.3f %-6s infer=%9.3f  (infer/train=%.2fx)",
                    desc, t, unit, i, ratio)
    logger.info("=" * 72)
    logger.info(
        "解读：eff_spacing_h / slab_z_mm 的 infer/train 比值越偏离 1.0，说明进模型前的"
        "解剖体素尺度差异越大——这是 z_axis/2.5D 无 spacing 重采样导致跨域假阳的直接量化。")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze train vs inference dataset distribution shift.")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--infer-dir", required=True)
    parser.add_argument("--out-dir", default="scripts/dataset_shift_report")
    parser.add_argument("--train-name", default="train(totalseg)")
    parser.add_argument("--infer-name", default="infer(airway)")
    parser.add_argument("--image-suffix", default=".nii.gz")
    # 默认取自 configs/segtest0.yaml
    parser.add_argument("--intensity-min", type=float, default=-1024.0)
    parser.add_argument("--intensity-max", type=float, default=1024.0)
    parser.add_argument("--patch-size", type=int, nargs=3, default=[12, 256, 256],
                        metavar=("D", "H", "W"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0,
                        help="Only scan first N files per dataset (0=all).")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pD, pH, pW = args.patch_size

    scan_kwargs = dict(
        image_suffix=args.image_suffix,
        intensity_min=args.intensity_min, intensity_max=args.intensity_max,
        patch_d=pD, patch_h=pH, patch_w=pW,
        workers=args.workers, limit=args.limit)

    train_rows, train_hist, train_err = scan_dataset(
        args.train_dir, args.train_name, **scan_kwargs)
    infer_rows, infer_hist, infer_err = scan_dataset(
        args.infer_dir, args.infer_name, **scan_kwargs)

    if not train_rows or not infer_rows:
        logger.error("One dataset produced 0 valid rows; aborting "
                     "(train=%d, infer=%d).", len(train_rows), len(infer_rows))
        return 1

    # 逐文件 CSV
    fieldnames = list(train_rows[0].keys())
    _write_csv(out_dir / "per_file_stats.csv",
               train_rows + infer_rows, fieldnames)

    # 汇总 CSV
    summary = _summarize(train_rows, args.train_name) + \
        _summarize(infer_rows, args.infer_name)
    _write_csv(out_dir / "summary_stats.csv", summary,
               ["metric", "dataset", "n", "mean", "std", "median", "q05", "q95"])

    # 失败 CSV
    if train_err or infer_err:
        _write_csv(
            out_dir / "read_errors.csv",
            [{"error": e} for e in (train_err + infer_err)], ["error"])

    _make_plots(out_dir, train_rows, infer_rows, train_hist, infer_hist,
                args.train_name, args.infer_name,
                args.intensity_min, args.intensity_max)

    _log_verdict(train_rows, infer_rows, args.train_name, args.infer_name)
    logger.info("Done. Report written to %s", out_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
