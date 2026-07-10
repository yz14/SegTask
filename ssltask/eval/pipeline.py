"""P6 离线评测/对比 harness（§0.4）。

说明：
* segmentation 的 ``finetune`` 读数指 **encoder-finetune + 线性探针头**，并非完整
  segtask 全 UNet 微调；这样可与分类读数保持一致的轻量比较口径。
* B2 from-scratch 通过 ``entry=None`` / ``entry='from_scratch'`` 走同一探针代码路径，
  唯一差异是不载入预训练 encoder 权重。
"""

from __future__ import annotations

import csv
import copy
import json
import logging
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from segtask_v1.trainer.checkpoint import extract_model_state_dict

from ..data.ssl_dataset import LabeledPatchDataset, discover_image_npz
from .cls_probe import ClsProbe
from .probe import SegProbe

logger = logging.getLogger(__name__)

EntrySpec = Union[None, str, Path, Dict[str, torch.Tensor], Tuple[str, object], Dict[str, object]]


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_entries(entries: Optional[Sequence[EntrySpec]]) -> List[Tuple[str, EntrySpec]]:
    if not entries:
        return [("B2-from-scratch", None)]
    out: List[Tuple[str, EntrySpec]] = []
    for idx, entry in enumerate(entries):
        if isinstance(entry, tuple) and len(entry) == 2:
            name, spec = entry
        elif isinstance(entry, dict) and "name" in entry:
            name = str(entry["name"])
            spec = entry.get("ckpt", entry.get("path", entry.get("spec")))
        elif isinstance(entry, str) and ("=" in entry or ":" in entry):
            if "=" in entry:
                name, rhs = entry.split("=", 1)
            else:
                name, rhs = entry.split(":", 1)
            name, spec = name.strip(), rhs.strip()
            if str(spec).lower() in ("from_scratch", "none", "null"):
                spec = None
        else:
            name = f"entry{idx}"
            spec = entry
        out.append((str(name), spec))
    return out


def _resolve_state_dict(spec: EntrySpec) -> Optional[Dict[str, torch.Tensor]]:
    if spec is None:
        return None
    if isinstance(spec, str) and spec.lower() == "from_scratch":
        return None
    if isinstance(spec, Path) or isinstance(spec, str):
        blob = torch.load(str(spec), map_location="cpu", weights_only=False)
        sd, _ = extract_model_state_dict(blob, prefer_ema=True)
        return sd
    if isinstance(spec, dict):
        if all(isinstance(v, torch.Tensor) for v in spec.values()):
            return spec  # raw state_dict
        if "model_state_dict" in spec or "state_dict" in spec:
            sd, _ = extract_model_state_dict(spec, prefer_ema=True)
            return sd
    raise TypeError(f"Unsupported entry spec: {type(spec)!r}")


def _split_holdout(paths: Sequence[str], seed: int, holdout_ratio: float) -> Tuple[List[str], List[str]]:
    paths = list(paths)
    if not paths:
        raise ValueError("No labelled npz found for offline evaluation.")
    rng = random.Random(int(seed))
    rng.shuffle(paths)
    n_val = max(1, int(round(len(paths) * float(holdout_ratio))))
    if len(paths) > 1:
        n_val = min(n_val, len(paths) - 1)
    val_paths = paths[:n_val]
    train_paths = paths[n_val:] or list(paths)
    return train_paths, val_paths


def build_nested_shot_splits(train_pool: Sequence[str], shots: Sequence[int], seed: int) -> Dict[int, List[str]]:
    """按固定种子生成嵌套 few-shot 子集；大 shots 包含小 shots。"""
    pool = list(train_pool)
    rng = random.Random(int(seed))
    rng.shuffle(pool)
    out: Dict[int, List[str]] = {}
    for shot in sorted({max(int(s), 1) for s in shots}):
        out[shot] = pool[:min(shot, len(pool))]
    return out


def _build_loader(cfg, ssl, paths: Sequence[str], task: str, batch_size: int,
                  samples_per_volume: Optional[int] = None,
                  shuffle: bool = False) -> DataLoader:
    dc = cfg.data
    if samples_per_volume is None:
        samples_per_volume = int(ssl.probe_samples_per_volume)
    ds = LabeledPatchDataset(
        npz_paths=paths,
        patch_size=dc.patch_size,
        intensity_min=dc.intensity_min,
        intensity_max=dc.intensity_max,
        normalize=dc.normalize,
        samples_per_volume=int(samples_per_volume),
        global_mean=dc.global_mean,
        global_std=dc.global_std,
        spatial_dims=int(cfg.model.spatial_dims),
        cls_label_key=str(ssl.cls_label_key) if task == "cls" else "",
        cache_enabled=dc.cache_mode == "memory",
        cache_max_volumes=dc.cache_max_volumes,
    )
    return DataLoader(ds, batch_size=max(int(batch_size), 1), shuffle=bool(shuffle),
                      num_workers=0, drop_last=False)


def _make_probe(cfg, ssl, device: torch.device, task: str, finetune: bool):
    if task == "seg":
        return SegProbe(cfg, ssl, device, finetune=finetune)
    if task == "cls":
        return ClsProbe(cfg, ssl, device, finetune=finetune)
    raise ValueError(f"Invalid task: {task!r}")


def run_eval_pipeline(cfg, ssl, entries: Optional[Sequence[EntrySpec]] = None,
                      shots: Optional[Sequence[int]] = None,
                      readouts: Optional[Sequence[str]] = None,
                      tasks: Optional[Sequence[str]] = None,
                      out_dir: Optional[Union[str, Path]] = None) -> Dict[str, object]:
    """运行 P6 离线评测，返回嵌套结果、扁平 rows，以及落盘路径。"""
    data_dir = ssl.eval_data_dir or ssl.probe_data_dir
    if not data_dir:
        raise ValueError("eval requires ssl.eval_data_dir or ssl.probe_data_dir.")
    probe_ssl = copy.copy(ssl)
    probe_ssl.probe_data_dir = data_dir
    paths = discover_image_npz(data_dir, cfg.data.npz_suffix)
    train_pool, val_pool = _split_holdout(paths, int(ssl.eval_seed), float(ssl.eval_holdout_ratio))

    shot_list = sorted({max(int(s), 1) for s in (shots or ssl.eval_shots)})
    readout_list = [str(r) for r in (readouts or ssl.eval_readouts)]
    task_list = [str(t) for t in (tasks or ssl.eval_tasks)]
    shot_splits = build_nested_shot_splits(train_pool, shot_list, int(ssl.eval_seed))
    entries_norm = _normalize_entries(entries or ssl.__dict__.get("eval_entries"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(out_dir or ssl.eval_out_dir or Path(cfg.train.output_dir) / "eval")
    out_path.mkdir(parents=True, exist_ok=True)

    nested: Dict[str, Dict[str, Dict[str, Dict[int, Dict[str, float]]]]] = {}
    rows: List[Dict[str, object]] = []

    for e_idx, (entry_name, entry_spec) in enumerate(entries_norm):
        full_sd = _resolve_state_dict(entry_spec)
        nested[entry_name] = {}
        for task in task_list:
            if task not in ("seg", "cls"):
                raise ValueError(f"Unsupported task: {task!r}")
            nested[entry_name][task] = {}
            for readout in readout_list:
                if readout not in ("linear", "finetune"):
                    raise ValueError(f"Unsupported readout: {readout!r}")
                finetune = readout == "finetune"
                nested[entry_name][task][readout] = {}
                for shot in shot_list:
                    seed = int(ssl.eval_seed) + e_idx * 1000 + shot * 10 + (0 if task == "seg" else 1) + (2 if finetune else 0)
                    _seed_all(seed)
                    probe = _make_probe(cfg, probe_ssl, device, task, finetune=finetune)
                    train_paths = shot_splits[int(shot)]
                    probe.train_loader = _build_loader(cfg, probe_ssl, train_paths, task, cfg.data.batch_size,
                                                        shuffle=True)
                    probe.val_loader = _build_loader(cfg, probe_ssl, val_pool, task, cfg.data.batch_size,
                                                     samples_per_volume=max(int(probe_ssl.probe_samples_per_volume) // 2, 1),
                                                     shuffle=False)
                    metrics = probe.evaluate(full_sd)
                    metrics = {k: float(v) for k, v in metrics.items()}
                    nested[entry_name][task][readout][int(shot)] = metrics
                    row = {
                        "entry": entry_name,
                        "task": task,
                        "readout": readout,
                        "shots": int(shot),
                        "dice": metrics.get("probe_dice"),
                        "hd95": metrics.get("probe_hd95"),
                        "auc": metrics.get("cls_auc"),
                        "f1": metrics.get("cls_f1"),
                    }
                    rows.append(row)

    json_path = out_path / "eval_summary.json"
    csv_path = out_path / "eval_summary.csv"
    payload = {
        "nested": nested,
        "rows": rows,
        "entries": [name for name, _ in entries_norm],
        "shots": shot_list,
        "readouts": readout_list,
        "tasks": task_list,
        "train_pool": train_pool,
        "val_pool": val_pool,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["entry", "task", "readout", "shots", "dice", "hd95", "auc", "f1"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in writer.fieldnames})

    return {
        "nested": nested,
        "rows": rows,
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "train_pool": train_pool,
        "val_pool": val_pool,
    }


__all__ = [
    "build_nested_shot_splits",
    "run_eval_pipeline",
]
