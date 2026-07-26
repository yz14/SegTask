"""DataLoader 工厂 + 训/验划分。扫描数据目录、划分 train/val、创建 DataLoader。"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

SuffixSpec = Union[str, Sequence[str]]

import numpy as np
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    DistributedSampler,
    Sampler,
)

from ..config import Config
from .dataset import (
    load_nifti,
    load_npz_label_counts,
    load_npz_label_for_split,
)
from .mixed_sampler import (
    SOURCE_PRIMARY,
    SOURCE_SECONDARY,
    MixedBatchSampler,
    SourceTaggedDataset,
    resolve_per_batch_counts,
)
from .specs import DatasetCommonCfg, SplitPaths, build_data_spec

logger = logging.getLogger(__name__)


class ValBatchShardSampler(Sampler):
    """DDP 验证采样器：按 batch 块把 val 样本不相交地切给各 rank。

    全集按 ``batch_size`` 顺序分块，块序号 ``i % world_size == rank`` 的块归当前
    rank，块内样本顺序不变。与"逐 rank 完整迭代 val_loader、按 batch 序号跳过"
    的切分严格同构，但 worker 只生产本 rank 的 batch，验证阶段 DataLoader CPU
    开销不随卡数翻倍。无 padding / 无重复（各 rank 计数可不等长；指标经
    all-reduce 汇总，与单进程全集累加严格相等）。
    """

    def __init__(self, num_samples: int, batch_size: int,
                 rank: int, world_size: int):
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        n_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
        self._blocks = list(range(self.rank, n_batches, self.world_size))

    def __iter__(self):
        for b in self._blocks:
            start = b * self.batch_size
            end = min(start + self.batch_size, self.num_samples)
            yield from range(start, end)

    def __len__(self) -> int:
        total = 0
        for b in self._blocks:
            start = b * self.batch_size
            total += min(start + self.batch_size, self.num_samples) - start
        return total


def _load_exclude_pids(exclude_list: str) -> set:
    """从文本读排除 pid 列表（每行一个，'#' 为注释）；可含 .nii(.gz) 后缀，
    也兼容 make_data 的 ``_failures.txt``（``pid\t<error>``，取首列）。
    路径为空或不存在返回空集。"""
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
            s = s.split("\t", 1)[0].strip()
            if not s:
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
    *,
    allow_unpaired: bool = False,
) -> Tuple[List[str], List[str]]:
    """按基名配对 image/label（首个匹配胜出）。后缀接受单个或候选序列。按基名排序返回。

    ``allow_unpaired=False``（默认）：任一 image 无匹配 label 即 ``FileNotFoundError``，
    与 bbox/rw 强配对口径一致。``True`` 时降级为 warning + 丢弃缺配对样本。
    """
    img_dir, lbl_dir = Path(image_dir), Path(label_dir)
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image dir not found: {img_dir}")
    if not lbl_dir.is_dir():
        raise FileNotFoundError(f"Label dir not found: {lbl_dir}")

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
        detail = (
            f"discover_samples: {len(missing_bases)}/{len(img_by_base)} "
            f"image bases have no matching label under {lbl_dir} for any of "
            f"{label_suffixes}. Missing bases: {head}{more}")
        if allow_unpaired:
            logger.warning("%s; dropping them (data.allow_unpaired=True).",
                           detail)
        else:
            raise FileNotFoundError(
                f"{detail}. Set data.allow_unpaired: true to drop unpaired "
                f"images with a warning instead.")

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
    if not sdir.is_dir():
        raise FileNotFoundError(f"{kind} dir not found: {sdir}")

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
    if not sdir.is_dir():
        raise FileNotFoundError(f"BBox dir not found: {sdir}")

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
    label_counts_fn=None,
) -> Union[List[int], Tuple[List[int], List[Dict[int, int]]]]:
    """自动探测标签取值；默认扫描全部。max_scan 指定部分扫描（会警告）；label_loader_fn 切换读器（NIfTI vs npz）。返按升序整数，含 bg。

    ``return_primaries=True`` 时额外返回每个样本的 ``{label_value: voxel_count}``
    字典列表，供 ``stratified_train_val_split`` 直接使用，避免重复扫描。

    ``label_counts_fn``：可选快路，返该样本的 ``{label_value: voxel_count}``
    或 ``None``（不可用）。可用时跳过全量解码 label（如从 npz meta 读
    预计算计数），启动期由逐卷 I/O+扫描降为仅读小体积 meta。"""
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
    n_fast = 0
    for path in scan_paths:
        counts = label_counts_fn(path) if label_counts_fn is not None else None
        if counts is None:
            lbl    = label_loader_fn(path)
            lbl_int = lbl.astype(np.int32, copy=False)
            unique, ucounts = np.unique(lbl_int, return_counts=True)
            counts = {int(v): int(c) for v, c in zip(unique, ucounts)}
        else:
            n_fast += 1
        all_labels.update(counts.keys())
        if return_primaries:
            per_sample_counts.append(counts)

    if n_fast:
        logger.info(
            "Label stats from precomputed npz meta for %d/%d files "
            "(no full label decode).", n_fast, len(scan_paths))
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


def finalize_from_data(
    cfg: Config,
    label_values: Sequence[int],
    *,
    per_sample_counts: Optional[List[Dict[int, int]]] = None,
) -> Tuple[Config, Optional[List[Dict[int, int]]]]:
    """显式把数据探测结果写入配置并同步派生字段。

    数据 loader 只能通过本函数提交 ``label_values`` / ``num_classes``；
    这样配置生命周期不再依赖某个 loader 是否恰好被先调用。返回值保留
    逐样本计数，供后续分层划分复用。
    """
    values = [int(v) for v in label_values]
    if not values:
        raise ValueError("finalize_from_data requires at least one label value.")
    cfg.data.label_values = values
    cfg.data.num_classes = len(values)
    cfg.sync()
    return cfg, per_sample_counts


def _half_up_count(n: int, val_ratio: float) -> int:
    return int(np.floor(float(n) * float(val_ratio) + 0.5))


def _random_split_val_count(
    n: int, val_ratio: float, rounding_mode: str) -> int:
    """随机 split：legacy 截断后 clamp 到 [1, n-1]。"""
    if n <= 1:
        return 0
    raw = (int(n * val_ratio) if rounding_mode == "legacy"
           else _half_up_count(n, val_ratio))
    return min(max(raw, 1), n - 1)


def _stratified_split_val_count(
    n: int, val_ratio: float, rounding_mode: str) -> int:
    """分层成员：legacy round()，至少 1，再限制不能整层进 val。"""
    raw = (int(round(n * val_ratio)) if rounding_mode == "legacy"
           else _half_up_count(n, val_ratio))
    return min(max(raw, 1), n - 1)


def _fallback_split_val_count(
    n: int, val_ratio: float, rounding_mode: str) -> int:
    """空 label fallback：legacy round()，不做上下界修正。"""
    return (int(round(n * val_ratio)) if rounding_mode == "legacy"
            else _half_up_count(n, val_ratio))


def _check_split_manifest_drift(
    path: Path,
    *,
    train: Sequence[str],
    val: Sequence[str],
) -> None:
    """读回既有 manifest 与本次划分比对：val 集变化时高声告警。

    样本增删会改变随机划分的排列结果：旧 val 样本可能进入新 train 集，
    选模指标被污染。manifest 损坏/不可读时跳过（随后会被重写）。"""
    if not path.is_file():
        return
    try:
        prev = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning("Split manifest %s unreadable (%s); rewriting.", path, e)
        return
    prev_val = set(prev.get("val", []))
    prev_train = set(prev.get("train", []))
    cur_val, cur_train = set(val), set(train)
    if prev_val == cur_val and prev_train == cur_train:
        return
    leaked = sorted(prev_val & cur_train)
    logger.warning(
        "Train/val split drifted from manifest %s: val %d->%d, train %d->%d "
        "sample(s).%s Check data roster / split_seed if this resume was "
        "expected to reuse the previous split.",
        path, len(prev_val), len(cur_val), len(prev_train), len(cur_train),
        (f" {len(leaked)} previous val sample(s) moved into train "
         f"(metric leakage risk), first 3: {leaked[:3]}.") if leaked else "")


def _write_split_manifest(
    path: Path,
    *,
    seed: int,
    val_ratio: float,
    rounding_mode: str,
    train: Sequence[str],
    val: Sequence[str],
    rank: int,
) -> None:
    """rank0 原子写入 split manifest，避免 DDP 并发覆盖。"""
    if int(rank) != 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "rounding_mode": str(rounding_mode),
        "train": list(train),
        "val": list(val),
    }, ensure_ascii=False, indent=2)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def train_val_split(
    n: int, val_ratio: float, seed: int, rounding_mode: str = "legacy"
) -> Tuple[List[int], List[int]]:
    """随机（非分层）按索引划分 train/val。"""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n).tolist()
    # 至少留 1 个训练样本（n==1 或 val_ratio 过大时防止 train 集为空）。
    n_val = _random_split_val_count(n, val_ratio, rounding_mode)
    if n_val == 0:
        logger.warning(
            "train_val_split: only %d sample(s); validation set is empty.", n)
    return indices[n_val:], indices[:n_val]


def extract_group_ids(
    paths: Sequence[str], group_id_regex: str) -> List[str]:
    """从 npz 文件名 stem 提取 group id（首个捕获组，无捕获组取整个匹配）。

    任一样本匹配失败即 fail-fast：静默回退会让患者级隔离契约无声失效。"""
    pat = re.compile(group_id_regex)
    gids: List[str] = []
    for p in paths:
        stem = Path(p).stem
        m = pat.search(stem)
        if m is None:
            raise ValueError(
                f"data.group_id_regex={group_id_regex!r} does not match "
                f"npz stem {stem!r} ({p}). Fix the regex or the file "
                f"naming; group-aware split requires every sample to "
                f"resolve a group id.")
        gids.append(m.group(1) if m.groups() else m.group(0))
    return gids


def grouped_train_val_split(
    paths: Sequence[str], group_id_regex: str,
    val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    """组级（患者级）随机划分：同 group id 的样本整体进 train 或 val。

    按组数而非样本数应用 ``val_ratio``（至少 1 组进 train；仅 1 组时 val
    为空并告警）；划分后断言 train/val 组集合互斥。"""
    gids = extract_group_ids(paths, group_id_regex)
    groups = sorted(set(gids))
    rng = np.random.RandomState(seed)
    perm = [groups[i] for i in rng.permutation(len(groups))]
    n_g = len(groups)
    n_val_g = min(max(1, int(n_g * val_ratio)), n_g - 1) if n_g > 1 else 0
    if n_val_g == 0:
        logger.warning(
            "grouped_train_val_split: only %d group(s); validation set is "
            "empty.", n_g)
    val_groups = set(perm[:n_val_g])
    train_idx = [i for i, g in enumerate(gids) if g not in val_groups]
    val_idx = [i for i, g in enumerate(gids) if g in val_groups]
    # 患者级隔离不变量：train/val 组集合必须互斥。
    if {gids[i] for i in train_idx} & {gids[i] for i in val_idx}:
        raise RuntimeError(
            "grouped_train_val_split invariant broken: train/val groups overlap")
    logger.info(
        "Split (group-aware, regex=%r): %d train / %d val samples from "
        "%d / %d groups (%d groups total).",
        group_id_regex, len(train_idx), len(val_idx),
        n_g - n_val_g, n_val_g, n_g)
    return train_idx, val_idx


def group_aware_train_val_split(
    paths: Sequence[str],
    val_ratio: float,
    seed: int,
    group_id_regex: str = "",
    stratified_keys: Optional[Sequence[str]] = None,
) -> Tuple[List[int], List[int]]:
    """按 group → stratified → random 的顺序选择 train/val 划分。

    ``group_id_regex`` 为空时不改变各任务原有的分层/随机分支；非空时
    患者（或其他业务组）隔离优先，分层参数仅作为无 group 配置时的回退。
    """
    if group_id_regex:
        return grouped_train_val_split(
            paths, group_id_regex, val_ratio, seed)
    if stratified_keys is not None:
        return stratified_split_by_key(
            stratified_keys, val_ratio, seed)
    return train_val_split(len(paths), val_ratio, seed)


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
    rounding_mode: str = "legacy",
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
        n_val_k = _stratified_split_val_count(
            len(members), val_ratio, rounding_mode)
        # 避免整层都进 val。
        n_val_k = min(n_val_k, len(members) - 1)
        val_idx.extend(members[:n_val_k])
        train_idx.extend(members[n_val_k:])

    # 空 label 体同 val_ratio 划分（不分层）。
    rng.shuffle(fallback)
    n_val_f = _fallback_split_val_count(
        len(fallback), val_ratio, rounding_mode)
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
    if not d.is_dir():
        raise FileNotFoundError(f"NPZ dir not found: {d}")
    paths = sorted(
        p for p in d.glob(f"*{npz_suffix}")
        if not p.name.startswith(("_", ".")))
    if not paths:
        raise ValueError(
            f"No npz packages found under {d} (suffix={npz_suffix!r}). "
            f"Did you run `python -m taskcore.data.make_data` first?")
    logger.info("Discovered %d npz package(s) under %s.", len(paths), d)
    return [str(p) for p in paths]


def discover_npz_recursive(npz_dir: str, npz_suffix: str = ".npz") -> List[str]:
    """递归发现 ``npz_dir`` 下所有 ``*{npz_suffix}``，按路径排序。

    与 :func:`discover_npz_samples`（仅顶层、忽略 '_'/'.' 附件）互补：
    cls/det/ssl 的 npz 数据目录允许按子目录组织，用本函数递归扫描。"""
    if not npz_dir or not os.path.isdir(npz_dir):
        raise FileNotFoundError(
            f"data.npz_dir not found: {npz_dir!r}. Expected a directory of "
            f"pre-generated npz packages (image [+ label]).")
    paths = sorted(glob.glob(
        os.path.join(npz_dir, "**", f"*{npz_suffix}"), recursive=True))
    if not paths:
        raise RuntimeError(f"No '*{npz_suffix}' found under {npz_dir!r}.")
    return paths


def stratified_split_by_key(keys: Sequence[str], val_ratio: float,
                            seed: int) -> Tuple[List[int], List[int]]:
    """按标签层（key）分层的 train/val 划分。

    逐层内部确定性 shuffle 后按 ``val_ratio`` 切分；层内 ≥2 个样本时
    train/val 各至少分到 1 个（保证小类两侧都有代表）；单样本层归
    训练集。同 (keys, val_ratio, seed) 下结果确定。
    """
    rng = np.random.RandomState(seed)
    by_key: "dict[str, List[int]]" = {}
    for i, k in enumerate(keys):
        by_key.setdefault(str(k), []).append(i)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for k in sorted(by_key):
        idx = by_key[k]
        perm = rng.permutation(len(idx))
        n = len(idx)
        if n == 1:
            train_idx.append(idx[0])
            continue
        n_val = min(max(int(round(n * val_ratio)), 1), n - 1)
        for j, p in enumerate(perm):
            (val_idx if j < n_val else train_idx).append(idx[p])
    return sorted(train_idx), sorted(val_idx)


def _resolve_npz_paths(
    cfg: Config, npz_dir: str, *, allow_auto_build: bool,
    rank: int = 0, world_size: int = 1) -> List[str]:
    """扫描 ``npz_dir``（必要且允许时内联自建）+ exclude 过滤，返回 npz 路径列表。

    ``allow_auto_build=True``（主源）时，目录缺失/空且 ``data.npz_auto_build`` 为真，
    则从 cfg 的 NIfTI 目录调 ``make_data.prepare_dataset`` 生成；副源恒为 False
    （cfg 仅描述一套 NIfTI 输入，副源必须事先用 make_data 离线生成）。

    DDP（``world_size > 1``）下仅 rank0 执行 auto-build，随后 ``dist.barrier()``，
    避免多 rank 交错写同一 pid 包导致损坏且被 skip 永久固化。
    """
    dc         = cfg.data
    npz_suffix = dc.npz_suffix
    npz_p      = Path(npz_dir)
    npz_present = npz_p.is_dir() and any(
        x for x in npz_p.glob(f"*{npz_suffix}") if not x.name.startswith(("_", ".")))

    if not npz_present:
        if not allow_auto_build:
            raise FileNotFoundError(
                f"Secondary npz dir {npz_dir!r} is empty/missing. The "
                f"secondary (coarse) source must be pre-built offline: run "
                f"`python -m taskcore.data.make_data --config <yaml> "
                f"--out {npz_dir}` against the coarse-label NIfTI set first.")
        if not bool(dc.npz_auto_build):
            raise FileNotFoundError(
                f"data.npz_dir={npz_dir!r} is empty/missing and "
                f"data.npz_auto_build is False. Run "
                f"`python -m taskcore.data.make_data --config "
                f"<yaml> --out {npz_dir}` first, or set "
                f"data.npz_auto_build: true to build inline.")
        build_error: Optional[str] = None
        if int(rank) == 0:
            try:
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
                    build_error = (
                        f"Auto-build produced 0 valid npz packages under "
                        f"{npz_dir}. Check input image_dir / label_dir paths "
                        f"and the make_data error log.")
            except Exception as exc:
                build_error = f"{type(exc).__name__}: {exc}"
        if int(world_size) > 1:
            import torch
            import torch.distributed as dist
            if not (dist.is_available() and dist.is_initialized()):
                raise RuntimeError(
                    "npz auto-build with world_size>1 requires an initialized "
                    "torch.distributed process group (rank0 builds, peers barrier).")
            # 1=ok, 0=fail — rank0 失败时仍 barrier，避免 peer 永久挂死。
            flag = torch.tensor(
                [0 if build_error else 1], dtype=torch.int32,
                device="cuda" if torch.cuda.is_available() else "cpu")
            dist.broadcast(flag, src=0)
            dist.barrier()
            if int(flag.item()) == 0:
                raise RuntimeError(
                    build_error or
                    f"npz auto-build failed on rank0 (reported to rank {rank}).")
            if int(rank) != 0:
                npz_present = npz_p.is_dir() and any(
                    x for x in npz_p.glob(f"*{npz_suffix}")
                    if not x.name.startswith(("_", ".")))
                if not npz_present:
                    raise RuntimeError(
                        f"npz auto-build on rank0 finished but {npz_dir!r} "
                        f"is still empty/missing on rank {rank}.")
        elif build_error:
            raise RuntimeError(build_error)
    paths        = discover_npz_samples(npz_dir, npz_suffix)
    exclude_pids = _load_exclude_pids(dc.exclude_list)
    kept, _, keep_idx = _filter_by_exclude(
        list(paths), list(paths), npz_suffix, exclude_pids)
    if exclude_pids:
        paths = [paths[i] for i in keep_idx]
    return list(paths)


def _split_paths_from(npz_paths: List[str], idxs: Sequence[int]) -> SplitPaths:
    """从 npz 路径列表按索引切出一个 split。

    npz-only 模式下 image/label 路径仅作别名（计 len / 缓存键），实际 I/O 走
    ``dataset._npz_paths``，故三者同源。
    """
    sel = [npz_paths[i] for i in idxs]
    return SplitPaths(
        image_paths = list(sel), label_paths = list(sel), npz_paths = list(sel))


def scaled_num_workers(num_workers: int, world_size: int, enabled: bool) -> int:
    """DDP 下每卡 DataLoader 的 ``num_workers``。

    ``world_size>1`` 且 ``enabled`` 时按卡数平摊（向下取整、至少 1），使**全机聚合**
    worker 进程数与逐 worker LRU 缓存 RAM 与单卡基线一致；否则原样返回（每卡满额）。
    """
    if world_size > 1 and enabled and num_workers > 0:
        return max(1, num_workers // world_size)
    return num_workers


def resolve_dataloader_workers(
    cfg: Config, world_size: int = 1,
) -> int:
    """按 ``train.ddp_scale_dataloader_per_rank`` 解析每卡 ``num_workers``。"""
    dc = cfg.data
    return scaled_num_workers(
        dc.num_workers, world_size,
        bool(cfg.train.ddp_scale_dataloader_per_rank))


def worker_loader_kwargs(cfg: Config, num_workers: int) -> Dict[str, object]:
    """``persistent_workers`` / ``prefetch_factor``（仅 ``num_workers>0`` 时有效）。"""
    kwargs: Dict[str, object] = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(cfg.data.persistent_workers)
        kwargs["prefetch_factor"] = int(cfg.data.prefetch_factor)
    return kwargs


def ensure_train_batch_capacity(train_ds, batch_size: int) -> None:
    """``drop_last=True`` 下训练集不足一个 batch 会静默零批次，显式拦截。"""
    if len(train_ds) < int(batch_size):
        raise ValueError(
            f"Train dataset yields only {len(train_ds)} samples but "
            f"batch_size={batch_size} with drop_last=True would produce "
            "zero batches; lower data.batch_size or raise "
            "data.samples_per_volume.")


def assemble_train_val_loaders(
    train_ds,
    val_ds,
    cfg: Config,
    *,
    rank: int = 0,
    world_size: int = 1,
    collate_fn=None,
    log_prefix: str = "DataLoader",
    train_drop_last: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """单源 train/val DataLoader 装配（DDP sampler + drop_last 契约）。

    任务侧负责构造 ``train_ds`` / ``val_ds``；本函数收敛：
    workers 平摊、loader kwargs、零批次拦截、
    ``DistributedSampler`` / ``ValBatchShardSampler``。

    ``train_drop_last`` 仅作用于单进程训练 loader（seg/gen=True 保持步数
    稳定；cls/det=False 保留尾批）。DDP 恒为 drop_last=True——各 rank 步数
    必须一致，否则集合通信错位。
    """
    dc = cfg.data
    batch_size = int(dc.batch_size)
    eff_num_workers = resolve_dataloader_workers(cfg, world_size)
    extra = worker_loader_kwargs(cfg, eff_num_workers)
    if collate_fn is not None:
        extra = {**extra, "collate_fn": collate_fn}

    # drop_last 生效处（DDP 恒开；单进程按 train_drop_last）不足一个 batch
    # 会静默零批次空转，装配前显式拦截。
    if world_size > 1 or train_drop_last:
        ensure_train_batch_capacity(train_ds, batch_size)

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=eff_num_workers,
            pin_memory=dc.pin_memory,
            drop_last=True,
            **extra)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            sampler=ValBatchShardSampler(
                len(val_ds), batch_size, rank, world_size),
            num_workers=eff_num_workers,
            pin_memory=dc.pin_memory,
            drop_last=False,
            **extra)
        logger.info(
            "%s DDP samplers: rank=%d/%d, ~%d train samples/rank",
            log_prefix, rank, world_size, len(train_sampler))
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=eff_num_workers,
            pin_memory=dc.pin_memory,
            drop_last=train_drop_last,
            **extra)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=eff_num_workers,
            pin_memory=dc.pin_memory,
            drop_last=False,
            **extra)

    logger.info(
        "%s: batch_size=%d, num_workers=%d (per rank), pin_memory=%s, "
        "persistent_workers=%s, prefetch_factor=%s",
        log_prefix, batch_size, eff_num_workers, dc.pin_memory,
        extra.get("persistent_workers", "n/a"),
        extra.get("prefetch_factor", "n/a"))
    return train_loader, val_loader


def log_volume_cache_estimate(
    cfg: Config,
    train_ds,
    *,
    n_train_vols: int,
    num_workers: int,
    world_size: int = 1,
    open_npz=None,
) -> None:
    """内存缓存足迹估计（仅诊断；失败不影响训练）。

    ``train_ds`` 需暴露 ``_npz_paths``；``open_npz`` 缺省用 taskcore 的
    ``open_npz``（gen fork 可传入自身实现）。
    """
    dc = cfg.data
    if str(dc.cache_mode) != "memory":
        return
    try:
        if open_npz is None:
            from .dataset import open_npz
        from .dataset import _read_npz_image_shape
        npz_paths_train = train_ds._npz_paths
        # 免整卷解码读形状；with 确保句柄及时释放。
        with open_npz(npz_paths_train[0]) as _f:
            sample_voxels = int(np.prod(
                _read_npz_image_shape(_f, npz_paths_train[0])))
            has_rw_runtime = "rw" in _f.files
        img_b = 2 if str(dc.cache_dtype) == "int16" else 4
        bytes_per_img = sample_voxels * img_b
        bytes_per_lbl = sample_voxels * 2
        bytes_per_rw = sample_voxels * 4 if has_rw_runtime else 0
        index_bytes = 0
        seen = set()
        for name in ("_vol_fg_slices", "_vol_fg_coords",
                     "_vol_fg_slices_by_cls", "_vol_fg_coords_by_cls"):
            for value in getattr(train_ds, name, ()):
                values = value if isinstance(value, (list, tuple)) else (value,)
                for array in values:
                    if hasattr(array, "nbytes") and id(array) not in seen:
                        seen.add(id(array))
                        index_bytes += int(array.nbytes)
        per_vol_bytes = bytes_per_img + bytes_per_lbl + bytes_per_rw
        cap = int(dc.cache_max_volumes)
        eff_cap = cap if cap > 0 else n_train_vols
        eff_cap = min(eff_cap, n_train_vols)
        workers = max(int(num_workers), 1)
        total_gb = (
            per_vol_bytes * eff_cap + index_bytes) * workers / (1024 ** 3)
        agg_note = (
            "" if world_size <= 1 else
            " [per rank; x%d ranks => ~%.2f GiB machine-wide aggregate]"
            % (world_size, total_gb * world_size))
        logger.info(
            "Volume cache estimate: ~%.2f MiB per volume across image/"
            "label/weight caches (image %s + label int16%s, bbox-cropped); "
            "cap=%d volume(s) per cache, num_workers=%d => up to ~%.2f GiB "
            "RAM (all "
            "workers, caches + foreground indices; transient decode "
            "peaks add ~%.2f MiB/worker)%s.",
            per_vol_bytes / (1024 ** 2),
            "int16" if img_b == 2 else "fp32",
            " + region_weight fp32" if bytes_per_rw else "",
            eff_cap, workers, total_gb,
            sample_voxels * 4 / (1024 ** 2), agg_note)
        if index_bytes:
            logger.info(
                "Foreground index footprint: %.2f MiB (one copy per dataset "
                "worker process; not multiplied by cache_max_volumes).",
                index_bytes / (1024 ** 2))
        agg_workers = workers * max(world_size, 1)
        if cap == 0 and n_train_vols * agg_workers >= 16:
            budget_gb = 8.0
            rec = max(
                1,
                int(budget_gb * (1024 ** 3)
                    / max(per_vol_bytes, 1) / agg_workers))
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


def build_dataloaders(
    cfg: Config,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    """构建 train/val DataLoader。训练仅读 npz：data.npz_dir 必须设。
    目录为空且 npz_auto_build=True 时，从 NIfTI 目录内联调 make_data.prepare_dataset 生成。

    ``data.npz_dir_secondary`` 非空时启用双批混合：第二批（粗标）仅用于训练，
    与第一批（金标准）经 ``ConcatDataset`` + ``MixedBatchSampler`` 在每个 train
    batch 内按 ``data.mix_ratio`` 混合；验证集始终仅取金标准。DDP 下
    ``MixedBatchSampler`` 自身按 rank 对全局 batch 序列不相交切分（各 rank
    同 seed+epoch，每 epoch 需在外层 ``set_epoch``）。

    ``world_size > 1`` 时为多卡 DDP：单源训练集用 ``DistributedSampler`` 不相交
    切分到各 rank（每 epoch 需在外层 ``set_epoch`` 以重新洗牌）。验证集用
    ``ValBatchShardSampler`` 按 batch 块不相交切分（worker 只生产本 rank 的
    batch，无 padding / 无重复）；整卷(high)验证不走 val_loader 的 batch，仍按
    ``_npz_paths`` 在验证器内逐 rank 切。"""
    dc = cfg.data

    # DDP 下按 world_size 平摊每卡 DataLoader 的 num_workers（向下取整、至少 1）。
    # 每个 rank 是独立进程，各自 fork ``num_workers`` 个 worker 且各持一份逐 worker
    # LRU 卷缓存；不分摊则 worker 进程数与缓存 RAM 随卡数线性翻倍（混用机上 CPU 超额
    # 订阅 / 换页抖动 / 内核 soft-lockup 的根因）。分摊后**全机聚合**量与单卡基线一致。
    eff_num_workers = scaled_num_workers(
        dc.num_workers, world_size,
        bool(cfg.train.ddp_scale_dataloader_per_rank))
    if world_size > 1 and bool(cfg.train.ddp_scale_dataloader_per_rank):
        if eff_num_workers != dc.num_workers:
            logger.info(
                "DDP dataloader scaling: num_workers %d -> %d per rank "
                "(world_size=%d; aggregate %d workers across ranks matches "
                "the single-GPU baseline). Per-worker LRU cache is unchanged, "
                "so aggregate cache RAM also matches single-GPU. Set "
                "train.ddp_scale_dataloader_per_rank=false to keep full "
                "num_workers on every rank.",
                dc.num_workers, eff_num_workers, world_size,
                eff_num_workers * world_size)

    npz_dir = dc.npz_dir
    if not npz_dir:
        raise ValueError(
            "data.npz_dir is required for training (npz-only data path). "
            "(or should be created); see taskcore.data.make_data.")
    npz_suffix = dc.npz_suffix

    logger.info(
        "Primary (gold) training source: npz packages under %s (suffix=%s). "
        "NIfTI fields image_dir/label_dir/bbox_dir/region_weight_dir are "
        "consumed only by make_data when the npz cache must be built.",
        npz_dir, npz_suffix)
    primary_paths = _resolve_npz_paths(
        cfg, npz_dir, allow_auto_build=True, rank=rank, world_size=world_size)

    use_mixed       = bool(dc.npz_dir_secondary)
    secondary_paths : List[str] = []
    if use_mixed:
        secondary_paths = _resolve_npz_paths(
            cfg, dc.npz_dir_secondary, allow_auto_build=False,
            rank=rank, world_size=world_size)
        logger.info(
            "Secondary (coarse) training source: %d npz package(s) under %s "
            "(train-only; validation always uses gold only).",
            len(secondary_paths), dc.npz_dir_secondary)
        # 主/副源 pid 重叠即 fail-fast：同一病例的粗标副本全量进 train，
        # 若其金标进 val，验证指标被同病例训练数据污染。
        primary_pids = {Path(p).stem for p in primary_paths}
        overlap = sorted(
            Path(p).stem for p in secondary_paths
            if Path(p).stem in primary_pids)
        if overlap:
            raise ValueError(
                f"{len(overlap)} pid(s) appear in BOTH the primary and "
                f"secondary npz sources (first 5: {overlap[:5]}). The "
                "secondary (coarse) copies always train while their gold "
                "twins may fall into validation — remove the duplicates "
                "from one source.")

    label_loader_fn = load_npz_label_for_split

    # label_values：在主源探测（顺带记录逐样本体素计数供分层划分复用，避免重扫）；
    # 副源仅取并集补充可能新增的标签值。npz meta 含 label_counts（make_data≥1.3）
    # 时走快路，启动期不解码任何 label 卷。
    per_sample_counts: Optional[List[Dict[int, int]]] = None
    if not dc.label_values:
        detected_values, per_sample_counts = detect_label_values(
            primary_paths, label_loader_fn=label_loader_fn,
            return_primaries=True, label_counts_fn=load_npz_label_counts)
        if secondary_paths:
            sec_values = detect_label_values(
                secondary_paths, label_loader_fn=label_loader_fn,
                label_counts_fn=load_npz_label_counts)
            merged = sorted(set(detected_values) | set(sec_values))
            if merged != list(detected_values):
                logger.info(
                    "Label values extended by secondary source: %s -> %s",
                    detected_values, merged)
            detected_values = merged
        finalize_from_data(
            cfg, detected_values, per_sample_counts=per_sample_counts)
    logger.info("Label values: %s, num_classes: %d, num_fg: %d",
                dc.label_values, dc.num_classes, cfg.num_fg_classes)

    # 主源 train/val 划分（粗标不参与划分，整批用于训练）。
    if dc.group_id_regex:
        # 患者/组级划分优先：同组样本不得跨 train/val（防泄漏）。
        if dc.stratified_split:
            logger.warning(
                "data.group_id_regex is set; group-aware split overrides "
                "stratified_split (stratification within group constraints "
                "is not supported).")
        train_idx, val_idx = grouped_train_val_split(
            primary_paths, dc.group_id_regex, dc.val_ratio, dc.split_seed)
    elif dc.stratified_split and dc.num_classes >= 2:
        if per_sample_counts is None:
            # label_values 显式配置时未走探测；尝试从 npz meta 取计数，
            # 全部可用才使用（部分缺失则整体回退逐卷解码，保证划分口径一致）。
            metas = [load_npz_label_counts(p) for p in primary_paths]
            if all(m is not None for m in metas):
                per_sample_counts = metas
        train_idx, val_idx = stratified_train_val_split(
            primary_paths, dc.label_values, dc.val_ratio, dc.split_seed,
            label_loader_fn=label_loader_fn,
            per_sample_counts=per_sample_counts,
            rounding_mode=dc.split_rounding_mode)
    else:
        train_idx, val_idx = train_val_split(
            len(primary_paths), dc.val_ratio, dc.split_seed,
            rounding_mode=dc.split_rounding_mode)
        logger.info("Split (random): %d train, %d val",
                    len(train_idx), len(val_idx))
    if dc.split_manifest_path:
        if rank == 0:
            _check_split_manifest_drift(
                Path(dc.split_manifest_path),
                train=[primary_paths[i] for i in train_idx],
                val=[primary_paths[i] for i in val_idx])
        _write_split_manifest(
            Path(dc.split_manifest_path),
            seed=dc.split_seed,
            val_ratio=dc.val_ratio,
            rounding_mode=dc.split_rounding_mode,
            train=[primary_paths[i] for i in train_idx],
            val=[primary_paths[i] for i in val_idx],
            rank=rank)

    # 模式无关的公共构造参数 + 单 split 路径包装。
    common_cfg          = DatasetCommonCfg.from_cfg(cfg)
    primary_train_paths = _split_paths_from(primary_paths, train_idx)
    val_paths           = _split_paths_from(primary_paths, val_idx)

    # 唯一的 patch_mode 决策点；所有"split-dependent kwargs"（aug_oversample
    # / samples_per_volume / fg_ratio）由 spec 内部按 is_train 切换。
    spec = build_data_spec(cfg)
    spec.log_summary()
    primary_train_ds = spec.make_split(
        primary_train_paths, is_train=True, common=common_cfg)
    val_ds = spec.make_split(val_paths, is_train=False, common=common_cfg)

    # drop_last=True 下不足一个 batch 会静默零批次；与 assemble_train_val_loaders
    # 对齐，装配前显式拦截（混采由 MixedBatchSampler 自带等长守卫）。
    if not use_mixed:
        ensure_train_batch_capacity(primary_train_ds, int(dc.batch_size))

    # persistent_workers / prefetch_factor 仅 num_workers>0 时有效。
    loader_kwargs: Dict[str, object] = {}
    if eff_num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(dc.persistent_workers)
        loader_kwargs["prefetch_factor"] = int(dc.prefetch_factor)

    # 缓存估计与日志用的代表性 train dataset / 体积数。
    train_ds_for_estimate = primary_train_ds
    n_train_vols          = len(train_idx)

    if use_mixed:
        # 粗标整批用于训练（不划分）；金/粗各打 source 标记后串联。
        secondary_train_paths = SplitPaths(
            image_paths = list(secondary_paths),
            label_paths = list(secondary_paths),
            npz_paths   = list(secondary_paths))
        secondary_train_ds = spec.make_split(
            secondary_train_paths, is_train=True, common=common_cfg)
        tagged_primary   = SourceTaggedDataset(primary_train_ds, SOURCE_PRIMARY)
        tagged_secondary = SourceTaggedDataset(secondary_train_ds, SOURCE_SECONDARY)
        n_primary_samples   = len(tagged_primary)
        n_secondary_samples = len(tagged_secondary)
        gold_per_batch, coarse_per_batch = resolve_per_batch_counts(
            dc.mix_ratio, dc.batch_size)
        # DDP：sampler 自身按 rank 切分全局 batch 序列（各 rank 同排列、
        # 不相交、等长）；外层每 epoch 需 set_epoch 对齐重洗。
        sampler = MixedBatchSampler(
            n_primary        = n_primary_samples,
            n_secondary      = n_secondary_samples,
            gold_per_batch   = gold_per_batch,
            coarse_per_batch = coarse_per_batch,
            seed             = dc.split_seed,
            rank             = rank,
            world_size       = world_size)
        concat_ds = ConcatDataset([tagged_primary, tagged_secondary])
        train_loader = DataLoader(
            concat_ds,
            batch_sampler = sampler,
            num_workers   = eff_num_workers,
            pin_memory    = dc.pin_memory,
            **loader_kwargs)
        n_train_vols = len(train_idx) + len(secondary_paths)
        logger.info(
            "Mixed two-source training enabled: mix_ratio(gold:coarse)=%s -> "
            "per-batch %d gold + %d coarse (batch_size=%d), %d batches/epoch"
            "%s.",
            dc.mix_ratio, gold_per_batch, coarse_per_batch,
            dc.batch_size, len(sampler),
            f" per rank (world_size={world_size})" if world_size > 1 else "")
    elif world_size > 1:
        # DDP：训练样本经 DistributedSampler 不相交切分到各 rank（shuffle 由
        # sampler 负责，外层每 epoch set_epoch 重洗）。drop_last 保持各 rank 等长。
        train_sampler = DistributedSampler(
            primary_train_ds, num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=True)
        train_loader = DataLoader(
            primary_train_ds,
            batch_size  = dc.batch_size,
            sampler     = train_sampler,
            num_workers = eff_num_workers,
            pin_memory  = dc.pin_memory,
            drop_last   = True,
            **loader_kwargs)
        logger.info(
            "DDP DistributedSampler: rank=%d/%d, ~%d samples/rank (train).",
            rank, world_size, len(train_sampler))
    else:
        train_loader = DataLoader(
            primary_train_ds,
            batch_size  = dc.batch_size,
            shuffle     = True,
            num_workers = eff_num_workers,
            pin_memory  = dc.pin_memory,
            drop_last   = True,
            **loader_kwargs)

    # DDP：val 在采样器层按 batch 块切给各 rank，worker 只生产本 rank 的 batch
    # （否则每 rank 完整生产全集、验证 CPU 开销随卡数线性翻倍）。单进程时保持
    # shuffle=False 顺序全集。
    val_sampler = (
        ValBatchShardSampler(len(val_ds), dc.batch_size, rank, world_size)
        if world_size > 1 else None)
    val_loader = DataLoader(
        val_ds,
        batch_size  = dc.batch_size,
        shuffle     = False,
        sampler     = val_sampler,
        num_workers = eff_num_workers,
        pin_memory  = dc.pin_memory,
        drop_last   = False,
        **loader_kwargs)

    logger.info(
        "DataLoader: batch_size=%d, num_workers=%d (per rank), pin_memory=%s, "
        "persistent_workers=%s, prefetch_factor=%s",
        dc.batch_size, eff_num_workers, dc.pin_memory,
        loader_kwargs.get("persistent_workers", "n/a"),
        loader_kwargs.get("prefetch_factor", "n/a"))

    # 内存缓存足迹估计（仅诊断；逐 worker 倍增）。
    log_volume_cache_estimate(
        cfg, train_ds_for_estimate,
        n_train_vols=n_train_vols,
        num_workers=eff_num_workers,
        world_size=world_size)

    return train_loader, val_loader