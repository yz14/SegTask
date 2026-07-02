"""DataLoader 工厂 + 训/验划分。扫描数据目录、划分 train/val、创建 DataLoader，供 gentask 的共享 data 层使用。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

SuffixSpec = Union[str, Sequence[str]]

import numpy as np
from torch.utils.data import DataLoader

from ..config import Config
from .dataset import load_nifti, load_npz_label_for_split
from .specs import DatasetCommonCfg, SplitPaths, build_data_spec

logger = logging.getLogger(__name__)


def _load_exclude_pids(exclude_list: str) -> set:
    """从文本读排除 pid 列表（每行一个，'#' 为注释）；可含 .nii(.gz) 后缀。路径为空或不存在返回空集。"""
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
    """丢弃 image 基名在 exclude_pids 中的对；keep_idx 用于同步同名列表。"""
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

    if dropped:  # logging
        head = ", ".join(dropped[:10])
        more = f", ... (+{len(dropped) - 10} more)" if len(dropped) > 10 else ""
        logger.warning(
            "Excluded %d/%d sample(s) via `data.exclude_list`: [%s%s]",
            len(dropped), len(image_paths), head, more)

    image_paths = [image_paths[i] for i in keep_idx]
    label_paths = [label_paths[i] for i in keep_idx]
    return image_paths, label_paths, keep_idx


def _normalize_suffixes(suffix: SuffixSpec) -> List[str]:
    """将后缀规范为去重列表（接受 str 或序列）。"""
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
    """剔除首个匹配的后缀；无匹配返 None。"""
    for sfx in suffixes:
        if name.endswith(sfx):
            return name[: -len(sfx)]
    return None


def discover_samples(
    image_dir: str, label_dir: str,
    image_suffix: SuffixSpec = ".nii.gz",
    label_suffix: SuffixSpec = ".nii.gz",
) -> Tuple[List[str], List[str]]:
    """按基名配对 image/label（首个匹配胜出）。后缀接受单个或候选序列。按基名排序返回。"""
    img_dir, lbl_dir = Path(image_dir), Path(label_dir)
    assert img_dir.is_dir(), f"Image dir not found: {img_dir}"
    assert lbl_dir.is_dir(), f"Label dir not found: {lbl_dir}"

    image_suffixes = _normalize_suffixes(image_suffix)
    label_suffixes = _normalize_suffixes(label_suffix)

    # 按接受后缀枚举 image；冲突时首项胜。
    img_by_base: Dict[str, Path] = {}
    for sfx in image_suffixes:
        for p in sorted(img_dir.glob(f"*{sfx}")):
            base = _strip_suffix(p.name, [sfx])
            if base is None:
                continue
            img_by_base.setdefault(base, p)

    # 每个 base：lbl_dir 下首个存在的 <base><suffix> 胜。
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
    """严格 1:1 按基名匹配；任意缺失报错。供 match_bbox_paths / match_region_weight_paths 复用；kind 仅为日志标签。"""
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
    """与 image_paths 1:1 解析 bbox NIfTI 路径；缺失报错。"""
    return _match_per_sample_paths(
        image_paths, bbox_dir, image_suffix, bbox_suffix, kind="BBox")


def match_bbox_paths_lenient(
    image_paths: List[str],
    bbox_dir: str,
    image_suffix: SuffixSpec,
    bbox_suffix: SuffixSpec) -> Tuple[List[str], List[str]]:
    """宽容 bbox 匹配（推理专用）：无 bbox 的样本被丢弃并警告。返回 1:1 对齐的 (image, bbox) 路径。"""
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
    """与 image_paths 1:1 解析 region-weight NIfTI 路径；缺失报错。文件语义：bg=0、非 bg=权重；dataset 加载时 +1。"""
    return _match_per_sample_paths(
        image_paths, region_weight_dir, image_suffix, region_weight_suffix,
        kind="RegionWeight")


def _default_label_loader(path: str) -> np.ndarray:
    """默认 int16 NIfTI label reader；npz 模式使用 load_npz_label_for_split。"""
    return load_nifti(path, dtype=np.int16)


def detect_label_values(
    label_paths: List[str],
    max_scan: Optional[int] = None,
    label_loader_fn=None,
    *,
    return_primaries: bool = False,
) -> Union[List[int], Tuple[List[int], List[Dict[int, int]]]]:
    """自动探测标签取值；默认扫描全部。max_scan 指定部分扫描（会警告）；label_loader_fn 切换读器（NIfTI vs npz）。返按升序整数，含 bg。

    ``return_primaries=True`` 时额外返回每个样本的 ``{label_value: voxel_count}``
    字典列表，供 ``stratified_train_val_split`` 直接使用，避免重复扫描。"""
    if label_loader_fn is None:
        label_loader_fn = _default_label_loader
    n_total = len(label_paths)
    if max_scan is None or max_scan >= n_total:
        scan_paths = label_paths
        partial    = False
    else:
        scan_paths = label_paths[:max_scan]
        partial = True

    all_labels: set = set()
    per_sample_counts: List[Dict[int, int]] = []
    for path in scan_paths:
        lbl    = label_loader_fn(path)
        lbl_int = lbl.astype(np.int32, copy=False)
        unique = np.unique(lbl_int).tolist()
        all_labels.update(unique)
        if return_primaries:
            per_sample_counts.append(
                {int(v): int((lbl_int == v).sum()) for v in unique})

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
    if return_primaries:
        return result, per_sample_counts
    return result


def train_val_split(n: int, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    """随机（非分层）按索引划分 train/val。"""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n).tolist()
    n_val = max(1, int(n * val_ratio))
    return indices[n_val:], indices[:n_val]


def _volume_primary_class(
    label_path: str, label_values: List[int],
    label_loader_fn=None) -> int:
    """返回体素数最多的标签（同分取最小）。"""
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
    label_loader_fn=None,
    per_sample_counts: Optional[List[Dict[int, int]]] = None,
) -> Tuple[List[int], List[int]]:
    """按主前景标签分层划分；退化时回退随机。use_foreground_only=True 时忽略背景频率。

    ``per_sample_counts`` 可由 ``detect_label_values(return_primaries=True)``
    预先生成，避免重复扫描标签文件。"""
    n   = len(label_paths)
    rng = np.random.RandomState(seed)

    fg_vals = label_values[1:] if use_foreground_only and len(label_values) > 1 else label_values
    strata_vals = fg_vals if fg_vals else label_values

    strata: Dict[int, List[int]] = {v: [] for v in strata_vals}
    fallback: List[int] = []  # 无前景体素
    if label_loader_fn is None:
        label_loader_fn = _default_label_loader
    for idx, path in enumerate(label_paths):
        if per_sample_counts is not None and idx < len(per_sample_counts):
            counts = {v: per_sample_counts[idx].get(v, 0) for v in strata_vals}
        else:
            lbl = label_loader_fn(path)
            lbl_int = lbl.astype(np.int32, copy=False)
            counts = {v: int((lbl_int == v).sum()) for v in strata_vals}
        best = max(counts.values())
        if best == 0:
            fallback.append(idx)
        else:
            primary = min(v for v, c in counts.items() if c == best)  # 同分取最小
            strata[primary].append(idx)

    # 成员<2 的层全入 train（无法干净划分）。
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
        # 避免整层都进 val。
        n_val_k = min(n_val_k, len(members) - 1)
        val_idx.extend(members[:n_val_k])
        train_idx.extend(members[n_val_k:])

    # 空 label 体同 val_ratio 划分（不分层）。
    rng.shuffle(fallback)
    n_val_f = int(round(len(fallback) * val_ratio))
    val_idx.extend(fallback[:n_val_f])
    train_idx.extend(fallback[n_val_f:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    # 任一为空时回退随机。
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
    """列出 npz_dir 下的 make_data npz 包；忽略 '_' / '.' 附件。"""
    d = Path(npz_dir)
    assert d.is_dir(), f"NPZ dir not found: {d}"
    paths = sorted(
        p for p in d.glob(f"*{npz_suffix}")
        if not p.name.startswith(("_", ".")))
    if not paths:
        raise ValueError(
            f"No npz packages found under {d} (suffix={npz_suffix!r}). "
            f"Did you run `python -m gentask.data.make_data` first?")
    logger.info("Discovered %d npz package(s) under %s.", len(paths), d)
    return [str(p) for p in paths]


def build_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    """构建 train/val DataLoader。训练仅读 npz：data.npz_dir 必须设。
    目录为空且 npz_auto_build=True 时，从 NIfTI 目录内联调 make_data.prepare_dataset 生成。"""
    dc = cfg.data

    npz_dir = dc.npz_dir
    if not npz_dir:
        raise ValueError(
            "data.npz_dir is required for training (npz-only data path). "
            "(or should be created); see gentask.data.make_data.")
    npz_suffix = dc.npz_suffix

    # 缺失/空时自建 npz 缓存（一次性；部分目录被视为权威，重生请 make_data --overwrite）。
    npz_p       = Path(npz_dir)
    npz_present = npz_p.is_dir() and any(
        x for x in npz_p.glob(f"*{npz_suffix}") if not x.name.startswith(("_", ".")))
    if not npz_present:
        if not bool(dc.npz_auto_build):
            raise FileNotFoundError(
                f"data.npz_dir={npz_dir!r} is empty/missing and "
                f"data.npz_auto_build is False. Run "
                f"`python -m gentask.data.make_data --config "
                f"<yaml> --out {npz_dir}` first, or set "
                f"data.npz_auto_build: true to build inline.")
        logger.info(
            "data.npz_dir=%s is empty/missing — auto-building via "
            "make_data.prepare_dataset (workers=%d). One-time cost; ",
            npz_dir, max(dc.num_workers, 1))
        from .make_data import prepare_dataset
        counters = prepare_dataset(
            cfg, npz_dir, workers=max(dc.num_workers, 1), overwrite=False)
        logger.info(
            "Auto-build complete: written=%d, skipped=%d, failed=%d / total=%d.",
            counters["written"], counters["skipped"],
            counters["failed"], counters["total"])
        if counters["failed"] > 0:
            logger.warning(
                "make_data reported %d failed sample(s). Inspect "
                "%s/_failures.txt; affected pids will be missing from the "
                "training set.", counters["failed"], npz_dir)
        if counters["written"] + counters["skipped"] == 0:
            raise RuntimeError(
                f"Auto-build produced 0 valid npz packages under "
                f"{npz_dir}. Check input image_dir / label_dir paths "
                f"and the make_data error log.")

    logger.info(
        "Training source: npz packages under %s (suffix=%s). "
        "NIfTI fields image_dir/label_dir/bbox_dir/region_weight_dir are "
        "consumed only by make_data when the npz cache must be built.",
        npz_dir, npz_suffix)
    npz_paths_all = discover_npz_samples(npz_dir, npz_suffix)
    # image_paths/label_paths 仅为别名（计 len/缓存键）；实际 I/O 由 dataset._npz_paths。
    image_paths  = list(npz_paths_all)
    label_paths  = list(npz_paths_all)
    exclude_pids = _load_exclude_pids(dc.exclude_list)
    image_paths, label_paths, keep_idx = _filter_by_exclude(
        image_paths, label_paths, npz_suffix, exclude_pids)
    if exclude_pids:
        npz_paths_all = [npz_paths_all[i] for i in keep_idx]
    label_loader_fn = load_npz_label_for_split

    # 自动探测时顺便记录逐样本体素计数，供分层划分复用（避免二次全量扫描）。
    per_sample_counts: Optional[List[Dict[int, int]]] = None
    if not dc.label_values:
        dc.label_values, per_sample_counts = detect_label_values(
            label_paths, label_loader_fn=label_loader_fn,
            return_primaries=True)
        dc.num_classes  = len(dc.label_values)
        cfg.sync()
    logger.info("Label values: %s, num_classes: %d, num_fg: %d",
                dc.label_values, dc.num_classes, cfg.num_fg_classes)

    # 按主前景类分层划分（不可行时回退随机）。
    if dc.stratified_split and dc.num_classes >= 2:
        train_idx, val_idx = stratified_train_val_split(
            label_paths, dc.label_values, dc.val_ratio, dc.split_seed,
            label_loader_fn=label_loader_fn,
            per_sample_counts=per_sample_counts)
    else:
        train_idx, val_idx = train_val_split(
            len(image_paths), dc.val_ratio, dc.split_seed)
        logger.info("Split (random): %d train, %d val",
                    len(train_idx), len(val_idx))

    # 模式无关的公共构造参数 + 单 split 路径包装。
    common_cfg  = DatasetCommonCfg.from_cfg(cfg)
    train_paths = SplitPaths(
        image_paths = [image_paths[i] for i in train_idx],
        label_paths = [label_paths[i] for i in train_idx],
        npz_paths   = [npz_paths_all[i] for i in train_idx])
    val_paths = SplitPaths(
        image_paths = [image_paths[i] for i in val_idx],
        label_paths = [label_paths[i] for i in val_idx],
        npz_paths   = [npz_paths_all[i] for i in val_idx])

    # 唯一的 patch_mode 决策点；所有"split-dependent kwargs"（aug_oversample
    # / samples_per_volume / fg_ratio）由 spec 内部按 is_train 切换。
    spec = build_data_spec(cfg)
    spec.log_summary()
    train_ds = spec.make_split(train_paths, is_train=True, common=common_cfg)
    val_ds   = spec.make_split(val_paths, is_train=False, common=common_cfg)

    # persistent_workers / prefetch_factor 仅 num_workers>0 时有效。
    loader_kwargs: Dict[str, object] = {}
    if dc.num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(dc.persistent_workers)
        loader_kwargs["prefetch_factor"] = int(dc.prefetch_factor)

    train_loader = DataLoader(
        train_ds,
        batch_size  = dc.batch_size,
        shuffle     = True,
        num_workers = dc.num_workers,
        pin_memory  = dc.pin_memory,
        drop_last   = True,
        **loader_kwargs)
    val_loader = DataLoader(
        val_ds,
        batch_size  = dc.batch_size,
        shuffle     = False,
        num_workers = dc.num_workers,
        pin_memory  = dc.pin_memory,
        drop_last   = False,
        **loader_kwargs)

    logger.info(
        "DataLoader: batch_size=%d, num_workers=%d, pin_memory=%s, "
        "persistent_workers=%s, prefetch_factor=%s",
        dc.batch_size, dc.num_workers, dc.pin_memory,
        loader_kwargs.get("persistent_workers", "n/a"),
        loader_kwargs.get("prefetch_factor", "n/a"))

    # 内存缓存足迹估计（仅诊断；逐 worker 倍增）。
    if dc.cache_mode == "memory":
        try:
            # NPZ-only：读首个 npz 的 shape 与 rw 存在性。
            from .dataset import _open_npz as _peek_npz
            npz_paths_train = train_ds._npz_paths
            _f = _peek_npz(npz_paths_train[0])
            sample_voxels  = int(np.prod(_f["image"].shape))
            has_rw_runtime = "rw" in _f.files
            # image fp32 (4B)、label int16 (2B)、rw fp32 (4B 可选)。
            bytes_per_img = sample_voxels * 4
            bytes_per_lbl = sample_voxels * 2
            bytes_per_rw = sample_voxels * 4 if has_rw_runtime else 0
            per_vol_bytes = bytes_per_img + bytes_per_lbl + bytes_per_rw
            n_train_vols = len(train_idx)
            cap = int(dc.cache_max_volumes)
            # cap=0 为无上限 → 最坏情况缓存全部。
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
                # 在 8 GiB 预算下建议一个合适的 cap。
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
        except Exception as exc:  # pragma: no cover — 仅诊断
            logger.debug("Could not estimate volume cache size: %s", exc)

    return train_loader, val_loader