"""DataLoader 工厂 + 训/验划分。扫描数据目录、划分 train/val、创建 DataLoader，供 gentask 的共享 data 层使用。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

SuffixSpec = Union[str, Sequence[str]]

import numpy as np
from torch.utils.data import DataLoader, DistributedSampler

from taskcore.data.loader import ValBatchShardSampler, scaled_num_workers

from ..config import Config
from .dataset import load_nifti, load_npz_label_for_split
from .specs import DatasetCommonCfg, SplitPaths, build_data_spec
# 公共纯函数工具从 taskcore.data.loader 复用（与迁移前逐字符一致）。
from taskcore.data.loader import (  # noqa: F401  (re-export，保持旧 import 路径可用)
    _default_label_loader,
    _filter_by_exclude,
    _load_exclude_pids,
    _match_per_sample_paths,
    _normalize_suffixes,
    _strip_suffix,
    _volume_primary_class,
    discover_samples,
    match_bbox_paths,
    match_bbox_paths_lenient,
    match_region_weight_paths,
    stratified_train_val_split,
)
from taskcore.data.loader import (  # noqa: F401  (re-export)
    detect_label_values,
    discover_npz_samples,
    train_val_split,
)

logger = logging.getLogger(__name__)


def match_condition_paths(
    image_paths: List[str],
    cond_dir: str,
    image_suffix: SuffixSpec,
    cond_suffix: SuffixSpec) -> List[str]:
    """与 image_paths 1:1 解析条件体 NIfTI 路径；缺失报错。"""
    return _match_per_sample_paths(
        image_paths, cond_dir, image_suffix, cond_suffix, kind="Condition")


# detect_label_values / train_val_split / discover_npz_samples 已与 taskcore
# 合流（taskcore 版含 meta.label_counts 快路与 n==1 边界修正），见上方 re-export。


def build_dataloaders(
    cfg: Config, rank: int = 0, world_size: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    """构建 train/val DataLoader。训练仅读 npz：data.npz_dir 必须设。
    目录为空且 npz_auto_build=True 时，从 NIfTI 目录内联调 make_data.prepare_dataset 生成。

    ``world_size > 1``（DDP）时：训练集用 ``DistributedSampler`` 不相交切分
    到各 rank（每 epoch 需在外层 ``set_epoch``）；验证集用
    ``ValBatchShardSampler`` 按 batch 块不相交切分，指标在训练器内跨 rank
    聚合。单进程路径零变化。"""
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

    # drop_last=True 下不足一个 batch 会静默零批次空转，显式拦截。
    if len(train_ds) < dc.batch_size:
        raise ValueError(
            f"Train dataset yields only {len(train_ds)} samples but "
            f"batch_size={dc.batch_size} with drop_last=True would produce "
            "zero batches; lower data.batch_size or raise "
            "data.samples_per_volume.")

    # persistent_workers / prefetch_factor 仅 num_workers>0 时有效。
    eff_num_workers = scaled_num_workers(
        dc.num_workers, world_size,
        bool(cfg.train.ddp_scale_dataloader_per_rank))
    loader_kwargs: Dict[str, object] = {}
    if eff_num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(dc.persistent_workers)
        loader_kwargs["prefetch_factor"] = int(dc.prefetch_factor)

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=True)
        train_loader = DataLoader(
            train_ds,
            batch_size  = dc.batch_size,
            sampler     = train_sampler,
            num_workers = eff_num_workers,
            pin_memory  = dc.pin_memory,
            drop_last   = True,
            **loader_kwargs)
        val_loader = DataLoader(
            val_ds,
            batch_size  = dc.batch_size,
            sampler     = ValBatchShardSampler(
                len(val_ds), dc.batch_size, rank, world_size),
            num_workers = eff_num_workers,
            pin_memory  = dc.pin_memory,
            drop_last   = False,
            **loader_kwargs)
        logger.info(
            "gentask DDP samplers: rank=%d/%d, ~%d train samples/rank",
            rank, world_size, len(train_sampler))
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size  = dc.batch_size,
            shuffle     = True,
            num_workers = eff_num_workers,
            pin_memory  = dc.pin_memory,
            drop_last   = True,
            **loader_kwargs)
        val_loader = DataLoader(
            val_ds,
            batch_size  = dc.batch_size,
            shuffle     = False,
            num_workers = eff_num_workers,
            pin_memory  = dc.pin_memory,
            drop_last   = False,
            **loader_kwargs)

    logger.info(
        "DataLoader: batch_size=%d, num_workers=%d, pin_memory=%s, "
        "persistent_workers=%s, prefetch_factor=%s",
        dc.batch_size, eff_num_workers, dc.pin_memory,
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