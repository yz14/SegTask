"""SSL 评测子包（下游评测 / 训练期探针）。

- :class:`SegProbe`：训练期**在线**分割线性探针，可驱动 best ckpt 选择。
- :class:`ClsProbe`：**离线**分类探针，经 ``python -m ssltask.evaluate`` 使用，
  不接入 ``SSLTrainer`` 训练环。
"""

from __future__ import annotations

from .cls_probe import ClsProbe, build_cls_probe_loaders
from .metrics import hd95
from .pipeline import build_nested_shot_splits, run_eval_pipeline
from .probe import SegProbe, build_probe_loaders

__all__ = [
    "ClsProbe",
    "build_cls_probe_loaders",
    "SegProbe",
    "build_probe_loaders",
    "build_nested_shot_splits",
    "run_eval_pipeline",
    "hd95",
]
