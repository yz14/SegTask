"""逐样本 npz 预烘包（生成任务）：委托 ``taskcore.data.make_data.prepare_one``。

与分割差异仅在样本发现：可选 ``cond_dirs`` 条件体配对。spacing 归一化、
逐类 fg 索引、meta skip 校验与 core 口径一致。

CLI：`python -m gentask.data.make_data --config <yaml> --out <dir> [--workers N]`。
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import Config, load_config
from taskcore.data.make_data import (
    _DEFAULT_FG_SUBSAMPLE,
    _TOOL_VERSION,
    _resolve_target_spacing,
    prepare_one,
)

from .loader import (
    _filter_by_exclude,
    _load_exclude_pids,
    discover_samples,
    match_condition_paths,
    match_bbox_paths,
    match_region_weight_paths,
    detect_label_values,
)

logger = logging.getLogger(__name__)


def _stem(path: str, suffix) -> str:
    """返回去后缀的文件名；suffix 可为 str 或候选列表。"""
    name = Path(path).name
    suffixes = [suffix] if isinstance(suffix, str) else list(suffix)
    for sfx in suffixes:
        if sfx and name.endswith(sfx):
            return name[: -len(sfx)]
    return Path(name).stem


def _build_sample_table(cfg: Config) -> List[Dict[str, Optional[str]]]:
    """通过 loader helpers 发现/配对 image/label/bbox/rw/cond 路径；遵守 exclude_list。"""
    dc = cfg.data

    image_paths, label_paths = discover_samples(
        dc.image_dir, dc.label_dir, dc.image_suffix, dc.label_suffix)

    exclude_pids = _load_exclude_pids(dc.exclude_list)
    image_paths, label_paths, _ = _filter_by_exclude(
        image_paths, label_paths, dc.image_suffix, exclude_pids)

    bbox_paths_all: Optional[List[str]] = None
    if dc.bbox_dir:
        bbox_paths_all = match_bbox_paths(
            image_paths, dc.bbox_dir, dc.image_suffix, dc.bbox_suffix)

    rw_paths_all: Optional[List[str]] = None
    if dc.region_weight_dir:
        rw_paths_all = match_region_weight_paths(
            image_paths, dc.region_weight_dir, dc.image_suffix,
            dc.region_weight_suffix)

    cond_paths_all: Optional[List[List[str]]] = None
    if dc.cond_dirs:
        cond_paths_all = []
        for cond_dir in dc.cond_dirs:
            cond_paths_all.append(match_condition_paths(
                image_paths, cond_dir, dc.image_suffix, dc.cond_suffixes))

    samples: List[Dict[str, Optional[str]]] = []
    for i, (img, lbl) in enumerate(zip(image_paths, label_paths)):
        samples.append({
            "pid"  : _stem(img, dc.image_suffix),
            "image": img,
            "label": lbl,
            "bbox" : bbox_paths_all[i] if bbox_paths_all else None,
            "rw"   : rw_paths_all[i] if rw_paths_all else None,
            "cond" : ([paths[i] for paths in cond_paths_all]
                      if cond_paths_all else None)})
    return samples


def _resolve_label_values(
    cfg: Config, samples: List[Dict[str, Optional[str]]]) -> List[int]:
    """未配置时自动探测 label values（同 loader）。"""
    dc = cfg.data
    if dc.label_values:
        return list(map(int, dc.label_values))
    label_paths = [s["label"] for s in samples]
    detected    = detect_label_values(label_paths)
    return list(map(int, detected))


def prepare_dataset(
    cfg         : Config,
    out_dir     : str,
    workers     : int = 4,
    fg_subsample: int = _DEFAULT_FG_SUBSAMPLE,
    compress    : bool = False,
    overwrite   : bool = False,
    limit       : int = 0) -> Dict[str, int]:
    """为 cfg.data 下的所有样本生成 npz（含可选 cond；spacing/fg/meta 走 core）。"""
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    samples = _build_sample_table(cfg)
    if limit and limit > 0:
        samples = samples[:limit]
        logger.info("--limit %d: processing only the first %d samples.",
                    limit, len(samples))

    label_values = _resolve_label_values(cfg, samples)
    logger.info("Using label_values=%s (bg=%d)", label_values, label_values[0])

    spacing_norm = bool(getattr(cfg.data, "spacing_normalization", False))
    target_spacing = (
        _resolve_target_spacing(cfg, samples) if spacing_norm else None)
    if spacing_norm:
        logger.info(
            "spacing_normalization=True: resampling every volume to "
            "target_spacing=%s mm (numpy axis order (D,H,W)).", target_spacing)

    tasks: List[Tuple[Dict[str, Optional[str]], str]] = []
    for s in samples:
        out_path = str(out_p / f"{s['pid']}.npz")
        tasks.append((s, out_path))

    n_total  = len(tasks)
    counters = {"written": 0, "skipped": 0, "failed": 0, "total": n_total}
    failures : List[Tuple[str, str]] = []
    timings  : List[float] = []
    sizes    : List[int] = []

    logger.info(
        "Preparing %d samples → %s (workers=%d, compress=%s, "
        "overwrite=%s, fg_subsample=%d)",
        n_total, out_p, workers, compress, overwrite, fg_subsample)

    def _kwargs(sample: Dict[str, Optional[str]], out_path: str) -> dict:
        return dict(
            pid          = sample["pid"],
            image_path   = sample["image"],
            label_path   = sample["label"],
            bbox_path    = sample["bbox"],
            rw_path      = sample["rw"],
            cond_paths   = sample["cond"],
            out_path     = out_path,
            label_values = label_values,
            fg_subsample = fg_subsample,
            compress     = compress,
            overwrite    = overwrite,
            spacing_normalization = spacing_norm,
            target_spacing = target_spacing)

    t0 = time.perf_counter()

    if workers <= 0:
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
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_pid = {
                pool.submit(prepare_one, **_kwargs(s, out_path)): s["pid"]
                for s, out_path in tasks}
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

    elapsed     = time.perf_counter() - t0
    total_bytes = sum(sizes)
    total_gb    = total_bytes / (1024 ** 3)
    mean_s      = (sum(timings) / max(len(timings), 1)) if timings else 0.0
    logger.info(
        "Done in %.1fs: written=%d, skipped=%d, failed=%d / total=%d. "
        "Total npz size: %.2f GiB (mean per sample: %.1f MiB, "
        "mean compute: %.2fs).",
        elapsed, counters["written"], counters["skipped"],
        counters["failed"], counters["total"], total_gb,
        (total_bytes / max(len(sizes), 1)) / (1024 ** 2) if sizes else 0.0,
        mean_s)

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

    manifest = {
        "tool_version": _TOOL_VERSION,
        "made_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_paths": {
            "image_dir": cfg.data.image_dir,
            "label_dir": cfg.data.label_dir,
            "bbox_dir": cfg.data.bbox_dir,
            "region_weight_dir": cfg.data.region_weight_dir,
            "cond_dirs": list(cfg.data.cond_dirs),
        },
        "label_values": label_values,
        "spacing_normalization": spacing_norm,
        "target_spacing": ([float(s) for s in target_spacing]
                           if target_spacing is not None else None),
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
    status = res.get("status", "written")
    counters[status] = counters.get(status, 0) + 1
    timings.append(float(res.get("elapsed_s", 0.0)))
    sizes.append(int(res.get("size_bytes", 0)))


def _log_progress(
    done: int, total: int, res: Dict[str, object], t0: float) -> None:
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
    msg = f"{type(exc).__name__}: {exc}"
    return msg if len(msg) <= max_len else msg[: max_len - 3] + "..."


def _setup_logging(level: str = "INFO") -> None:
    from taskcore.utils.logging_utils import setup_logging
    setup_logging(output_dir=None, level=level, log_filename=None)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="gentask make_data: bake per-sample npz (delegates to "
                    "taskcore; supports cond_dirs).")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--out", required=True, help="Output npz directory")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fg-subsample", type=int,
                        default=_DEFAULT_FG_SUBSAMPLE)
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    _setup_logging(args.log_level)
    cfg = load_config(args.config)
    if args.override:
        from ..train import apply_overrides
        apply_overrides(cfg, args.override)
        cfg.sync()
        cfg.validate()

    try:
        counters = prepare_dataset(
            cfg, args.out,
            workers=args.workers,
            fg_subsample=args.fg_subsample,
            compress=args.compress,
            overwrite=args.overwrite,
            limit=args.limit)
    except Exception:
        logger.error("make_data aborted:\n%s", traceback.format_exc())
        return 2
    return 1 if counters.get("failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
