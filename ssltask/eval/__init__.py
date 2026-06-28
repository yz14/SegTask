"""SSL 评测子包（§0.4 下游评测 / §0.5 在线探针）。

当前提供在线分割线性探针 :class:`SegProbe`，用于在预训练过程中以"真表征质量"
（小标注集上冻结 encoder 的线性探针 Dice）驱动 best ckpt 选择，避免按自监督代理
损失选模（DINO/JEPA 等损失与表征质量不单调）。
"""

from __future__ import annotations

from .probe import SegProbe, build_probe_loaders

__all__ = ["SegProbe", "build_probe_loaders"]
