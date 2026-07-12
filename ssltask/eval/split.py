"""组级（患者级）train/val 划分，供在线探针与离线评测共用。

npz 文件名 stem 即 make_data 的 ``pid``；同一患者可能有多个序列
（如 ``P001_a.npz`` / ``P001_b.npz``）。``group_regex`` 的第一个捕获组把
文件名 stem 归并为组键，保证同组样本不跨 train/val 泄漏。
"""

from __future__ import annotations

import logging
import os
import random
import re
from typing import Dict, List, Sequence, Tuple

logger = logging.getLogger(__name__)


def group_key(path: str, group_regex: str = "") -> str:
    """文件路径 → 组键：默认文件名 stem；``group_regex`` 非空时取其第一个
    捕获组（不匹配则退回 stem 并告警一次由调用方汇总）。"""
    stem = os.path.basename(path)
    if stem.endswith(".npz"):
        stem = stem[: -len(".npz")]
    if not group_regex:
        return stem
    m = re.search(group_regex, stem)
    if m is None or not m.groups():
        return stem
    return str(m.group(1))


def group_split(
    paths: Sequence[str],
    val_ratio: float,
    seed: int,
    group_regex: str = "",
    allow_single_group: bool = False,
) -> Tuple[List[str], List[str]]:
    """按组（患者）划分 train/val；同组文件绝不同时出现在两侧。

    - 组按 ``seed`` 洗牌后从头取 val 组，直到覆盖 ≥ ``val_ratio`` 的文件数
      （至少 1 组），且保证 train 至少 1 组。
    - 只有 1 个组时：``allow_single_group=False`` 抛错（train==val 的读数
      无效）；``True`` 则告警后 train==val（仅调试用）。
    """
    paths = list(paths)
    if not paths:
        raise ValueError("group_split got empty paths.")
    groups: Dict[str, List[str]] = {}
    unmatched = 0
    for p in paths:
        k = group_key(p, group_regex)
        if group_regex and k == group_key(p, ""):
            m = re.search(group_regex, k)
            if m is None or not m.groups():
                unmatched += 1
        groups.setdefault(k, []).append(p)
    if unmatched:
        logger.warning(
            "group_split: %d/%d file(s) did not match group_regex %r; "
            "falling back to filename stem for those.",
            unmatched, len(paths), group_regex)
    keys = sorted(groups)
    if len(keys) == 1:
        msg = (
            f"group_split: only one group ({keys[0]!r}) across {len(paths)} "
            f"file(s) — a train/val split is impossible without leakage.")
        if not allow_single_group:
            raise ValueError(
                msg + " Provide more volumes/groups, or set "
                "allow_single_group=True (debug only: train==val, metrics "
                "are NOT valid for model selection).")
        logger.warning("%s Reusing the same data for train and val "
                       "(allow_single_group=True; metrics are optimistic).", msg)
        return list(paths), list(paths)
    rng = random.Random(int(seed))
    rng.shuffle(keys)
    n_files = len(paths)
    target = max(int(round(n_files * float(val_ratio))), 1)
    val_keys: List[str] = []
    n_val_files = 0
    for k in keys:
        if len(val_keys) >= len(keys) - 1:      # train 至少留 1 组
            break
        if n_val_files >= target and val_keys:
            break
        val_keys.append(k)
        n_val_files += len(groups[k])
    val_set = set(val_keys)
    train_paths = [p for k in keys if k not in val_set for p in groups[k]]
    val_paths = [p for k in val_keys for p in groups[k]]
    assert train_paths and val_paths
    logger.info(
        "group_split: %d group(s) -> train %d group(s)/%d file(s), "
        "val %d group(s)/%d file(s) (seed=%d, regex=%r).",
        len(keys), len(keys) - len(val_keys), len(train_paths),
        len(val_keys), len(val_paths), int(seed), group_regex)
    return train_paths, val_paths


__all__ = ["group_key", "group_split"]
