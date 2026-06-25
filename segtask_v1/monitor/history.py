"""训练过程指标的持久化与读取（数据层，零外部依赖）。

职责单一：把训练循环每个 epoch 产出的标量指标追加写入 ``metrics.jsonl``，
并维护 ``metrics_summary.json``（run 元信息 + 当前 best）。``MetricsHistory``
负责读回历史、按指标名取序列，供仪表盘渲染（``dashboard.py``）与多 run 对比
使用。本模块**不做任何绘图 / HTML**，仅 I/O 与数据模型。

文件布局（位于 ``output_dir/monitor/``）：

* ``metrics.jsonl`` —— 每行一个 epoch 记录（``EpochRecord`` 的 JSON）。整体
  每 epoch 原子重写一次，故续训 / 崩溃重跑都不会产生重复或半行。
* ``metrics_summary.json`` —— run 级元信息（选模标准、计划 epoch 数、类别数）
  与派生的 best 信息、运行状态。

设计取舍：JSONL 体量小（每 run 至多数千行），每 epoch 全量重写既能彻底规避
续训时的重复行，又保持崩溃安全（写临时文件后原子 ``os.replace``）。
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

_METRICS_FILENAME = "metrics.jsonl"
_SUMMARY_FILENAME = "metrics_summary.json"


def _finite_scalars(d: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """仅保留可转 float 且有限（非 NaN/Inf）的标量，丢弃其余。"""
    out: Dict[str, float] = {}
    if not d:
        return out
    for k, v in d.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            out[str(k)] = fv
    return out


def _finite_or_none(v: Any) -> Optional[float]:
    """转 float；非有限或不可转则 None。"""
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    return fv if math.isfinite(fv) else None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class EpochRecord:
    """单个 epoch 的指标快照（epoch 为 0-based 内部索引）。"""

    epoch: int
    train: Dict[str, float] = field(default_factory=dict)
    val: Dict[str, float] = field(default_factory=dict)
    lr: Optional[float] = None
    gpu_peak_mib: Optional[float] = None
    wall_time_s: Optional[float] = None
    is_best: bool = False
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": int(self.epoch),
            "train": dict(self.train),
            "val": dict(self.val),
            "lr": self.lr,
            "gpu_peak_mib": self.gpu_peak_mib,
            "wall_time_s": self.wall_time_s,
            "is_best": bool(self.is_best),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EpochRecord":
        return cls(
            epoch=int(d.get("epoch", 0)),
            train=_finite_scalars(d.get("train")),
            val=_finite_scalars(d.get("val")),
            lr=_finite_or_none(d.get("lr")),
            gpu_peak_mib=_finite_or_none(d.get("gpu_peak_mib")),
            wall_time_s=_finite_or_none(d.get("wall_time_s")),
            is_best=bool(d.get("is_best", False)),
            timestamp=float(d.get("timestamp", 0.0) or 0.0),
        )

    def get(self, key: str, source: str = "val") -> Optional[float]:
        """取某指标值；``source`` ∈ {"val","train","top"}。

        ``top`` 指记录顶层标量（lr / gpu_peak_mib / wall_time_s）。
        """
        if source == "val":
            return self.val.get(key)
        if source == "train":
            return self.train.get(key)
        if source == "top":
            return getattr(self, key, None)
        return None


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------
@dataclass
class MetricsHistory:
    """已落盘训练历史的只读模型，供渲染 / 对比使用。"""

    run_name: str
    records: List[EpochRecord] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    source_dir: Optional[str] = None

    # -- loading --------------------------------------------------------
    @classmethod
    def from_dir(cls, path: Union[str, Path]) -> "MetricsHistory":
        """从一个 run 目录加载。``path`` 既可指向含 ``metrics.jsonl`` 的目录，
        也可指向其父级 ``output_dir``（自动下探 ``monitor/`` 子目录）。"""
        d = Path(path)
        jsonl = d / _METRICS_FILENAME
        if not jsonl.exists() and (d / "monitor" / _METRICS_FILENAME).exists():
            d = d / "monitor"
            jsonl = d / _METRICS_FILENAME
        summary = cls._read_summary(d / _SUMMARY_FILENAME)
        records = cls._read_jsonl(jsonl)
        run_name = str(summary.get("run_name") or d.parent.name or d.name)
        return cls(run_name=run_name, records=records, summary=summary,
                   source_dir=str(d))

    @staticmethod
    def _read_jsonl(path: Path) -> List[EpochRecord]:
        if not path.exists():
            return []
        recs: "OrderedDict[int, EpochRecord]" = OrderedDict()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = EpochRecord.from_dict(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    continue
                recs[rec.epoch] = rec
        return [recs[e] for e in sorted(recs)]

    @staticmethod
    def _read_summary(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    # -- queries --------------------------------------------------------
    def series(self, key: str, source: str = "val") -> Tuple[List[int], List[float]]:
        """返回 ``(epochs, values)``，自动跳过缺失该指标的 epoch。

        epochs 为内部 0-based 索引；展示层负责 +1。
        """
        xs: List[int] = []
        ys: List[float] = []
        for rec in self.records:
            v = rec.get(key, source)
            if v is None:
                continue
            xs.append(rec.epoch)
            ys.append(v)
        return xs, ys

    def metric_keys(self, source: str = "val") -> List[str]:
        """该 source 下出现过的全部指标名（排序去重）。"""
        keys = set()
        for rec in self.records:
            d = rec.val if source == "val" else rec.train
            keys.update(d.keys())
        return sorted(keys)

    def per_class_keys(self, prefix: str, source: str = "val") -> List[str]:
        """按数值后缀排序返回形如 ``{prefix}{c}`` 的逐类指标名。"""
        out = [k for k in self.metric_keys(source) if k.startswith(prefix)]

        def _idx(k: str) -> int:
            tail = k[len(prefix):]
            return int(tail) if tail.isdigit() else 1 << 30

        return sorted(out, key=_idx)

    @property
    def best(self) -> Dict[str, Any]:
        return dict(self.summary.get("best") or {})

    @property
    def last_epoch(self) -> Optional[int]:
        return self.records[-1].epoch if self.records else None

    @property
    def num_classes(self) -> int:
        return int(self.summary.get("num_classes", 0) or 0)

    def __len__(self) -> int:
        return len(self.records)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
class MetricsLogger:
    """训练期每 epoch 落盘指标 + 维护 summary。``enabled=False`` 由调用方负责，
    本类一经实例化即认为启用。"""

    def __init__(
        self,
        output_dir: Union[str, Path],
        *,
        run_name: Optional[str] = None,
        save_best_metric: str = "mean_dice",
        save_best_mode: str = "max",
        save_best_criterion: str = "",
        num_classes: int = 0,
        total_epochs: int = 0,
        config_meta: Optional[Dict[str, Any]] = None,
        resume: bool = False,
    ):
        self.dir = Path(output_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.dir / _METRICS_FILENAME
        self.summary_path = self.dir / _SUMMARY_FILENAME

        self.run_name = str(run_name or self.dir.parent.name or self.dir.name)
        self.save_best_metric = save_best_metric
        self.save_best_mode = save_best_mode if save_best_mode in ("max", "min") else "max"
        self.save_best_criterion = save_best_criterion
        self.num_classes = int(num_classes)
        self.total_epochs = int(total_epochs)
        self.config_meta = dict(config_meta or {})

        self._records: "OrderedDict[int, EpochRecord]" = OrderedDict()
        self._created_at = time.time()
        self._status = "running"

        if resume and self.jsonl_path.exists():
            for rec in MetricsHistory._read_jsonl(self.jsonl_path):
                self._records[rec.epoch] = rec
            prior = MetricsHistory._read_summary(self.summary_path)
            self._created_at = float(prior.get("created_at", self._created_at)
                                     or self._created_at)
            logger.info("MetricsLogger resumed: %d prior epoch record(s) at %s",
                        len(self._records), self.jsonl_path)
        else:
            # 全新 run：清掉可能存在的上轮残留，写出初始空历史 + summary。
            self._write_jsonl()
            self._write_summary()

    # -- public API -----------------------------------------------------
    def log_epoch(
        self,
        epoch: int,
        *,
        train: Optional[Dict[str, Any]] = None,
        val: Optional[Dict[str, Any]] = None,
        lr: Optional[float] = None,
        gpu_peak_mib: Optional[float] = None,
        wall_time_s: Optional[float] = None,
        is_best: bool = False,
    ) -> EpochRecord:
        """记录一个 epoch（覆盖同号 epoch），落盘 jsonl + summary。"""
        rec = EpochRecord(
            epoch=int(epoch),
            train=_finite_scalars(train),
            val=_finite_scalars(val),
            lr=_finite_or_none(lr),
            gpu_peak_mib=_finite_or_none(gpu_peak_mib),
            wall_time_s=_finite_or_none(wall_time_s),
            is_best=bool(is_best),
            timestamp=time.time(),
        )
        self._records[rec.epoch] = rec
        self._write_jsonl()
        self._write_summary()
        return rec

    def finalize(self, status: str = "completed") -> None:
        """训练正常结束 / 早停 / 中断时更新 run 状态并落盘 summary。"""
        self._status = status
        self._write_summary()

    @property
    def records(self) -> List[EpochRecord]:
        return [self._records[e] for e in sorted(self._records)]

    # -- best computation ----------------------------------------------
    def _compute_best(self) -> Optional[Dict[str, Any]]:
        """据 ``save_best_metric`` / mode 从已记录的验证指标里选 best epoch。"""
        best_rec: Optional[EpochRecord] = None
        best_val: Optional[float] = None
        for rec in self._records.values():
            v = rec.val.get(self.save_best_metric)
            if v is None:
                continue
            if best_val is None or (
                v > best_val if self.save_best_mode == "max" else v < best_val
            ):
                best_val, best_rec = v, rec
        if best_rec is None:
            return None
        return {
            "epoch": best_rec.epoch,
            "metric_name": self.save_best_metric,
            "metric_value": best_val,
            "val": dict(best_rec.val),
            "train": dict(best_rec.train),
            "lr": best_rec.lr,
        }

    # -- atomic writers -------------------------------------------------
    def _atomic_write(self, path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _write_jsonl(self) -> None:
        lines = [json.dumps(self._records[e].to_dict(), ensure_ascii=False)
                 for e in sorted(self._records)]
        self._atomic_write(self.jsonl_path,
                            "\n".join(lines) + ("\n" if lines else ""))

    def _write_summary(self) -> None:
        recs = sorted(self._records)
        summary: Dict[str, Any] = {
            "run_name": self.run_name,
            "created_at": self._created_at,
            "updated_at": time.time(),
            "status": self._status,
            "save_best_metric": self.save_best_metric,
            "save_best_mode": self.save_best_mode,
            "save_best_criterion": self.save_best_criterion,
            "num_classes": self.num_classes,
            "total_epochs_planned": self.total_epochs,
            "epochs_recorded": len(recs),
            "last_epoch": recs[-1] if recs else None,
            "best": self._compute_best(),
            "config": self.config_meta,
        }
        self._atomic_write(self.summary_path,
                           json.dumps(summary, ensure_ascii=False, indent=2))


__all__ = ["EpochRecord", "MetricsHistory", "MetricsLogger"]
