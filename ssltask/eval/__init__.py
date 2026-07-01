"""SSL 评测子包（§0.4 下游评测 / §0.5 在线探针）。

当前提供在线分割线性探针 :class:`SegProbe` 与在线分类探针 :class:`ClsProbe`，
用于在预训练过程中以"真表征质量"（小标注集上冻结 encoder 的下游读数）驱动 best
ckpt 选择，避免按自监督代理损失选模（DINO/JEPA 等损失与表征质量不单调）。
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
