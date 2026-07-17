"""五任务推理器共用工程件基类（``BasePredictor``）。

与 ``BaseTrainer`` 同一设计取向：各任务推理几何（滑窗 / 网格 / 整卷复原）差异
大，基类不吞并推理主流程；只把逐字重复的工程 blocks 收敛为 protected helpers，
子类按原有顺序显式调用：

* ``_setup_infer_amp``  —— 推理 autocast dtype 解析（口径同训练
  ``train.amp_dtype``，含 ``'auto'``）+ use_amp 开关；
* ``_autocast``         —— 依据上面两项构造 autocast 上下文；
* ``flip_tta_combos``   —— 翻转 TTA 的轴组合枚举（可含/不含恒等组合）。
"""

from __future__ import annotations

import itertools
import logging
from typing import List, Sequence, Tuple

import torch

from .amp import _AMP_DTYPES, resolve_auto_amp_dtype

logger = logging.getLogger(__name__)


class BasePredictor:
    """共用推理工程件；子类实现自己的 ``predict_volume`` 等主流程。"""

    def _setup_infer_amp(self, use_amp: bool) -> None:
        """解析推理 autocast dtype（``train.amp_dtype``，含 'auto'）。

        ``use_amp``：任务侧开关（seg/cls/det 沿用 ``train.use_amp``，gen 用
        ``predict.use_amp``）；仅 CUDA 设备实际启用。
        """
        amp_name = str(self.cfg.train.amp_dtype)
        if amp_name == "auto":
            amp_name = resolve_auto_amp_dtype(self.device)
        if amp_name not in _AMP_DTYPES:
            raise ValueError(
                f"Unknown amp_dtype: {self.cfg.train.amp_dtype!r}. "
                f"Expected one of {sorted(_AMP_DTYPES) + ['auto']}.")
        self.amp_dtype = _AMP_DTYPES[amp_name]
        self.use_amp = bool(use_amp) and self.device.type == "cuda"

    def _autocast(self):
        """推理前向的 autocast 上下文（enabled=False 时为零开销直通）。"""
        return torch.autocast(device_type=self.device.type,
                              enabled=self.use_amp, dtype=self.amp_dtype)

    @staticmethod
    def flip_tta_combos(
        axes: Sequence[int], include_identity: bool = False,
    ) -> List[Tuple[int, ...]]:
        """翻转 TTA 的轴组合全枚举。

        ``axes``：可翻的空间轴索引；``include_identity=True`` 时首项为空组合
        （原图直通），否则仅返回非空组合（调用方自行叠加原始预测）。
        """
        start = 0 if include_identity else 1
        return [c for r in range(start, len(axes) + 1)
                for c in itertools.combinations(axes, r)]


__all__ = ["BasePredictor"]
