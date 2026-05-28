"""Predictor 包：3D 分割滑窗推理（R6 起从单文件升为包）。

历史上 ``segtask_v1.predictor`` 是一个单文件 1412 行的 God Module，混合了：

* `Predictor` 类：滑窗 / 多分辨率窗口建造 / forward / TTA / 概率体后处理
* `run_inference` 顶层入口、checkpoint 加载、precision 选择

R6 把它拆为子模块（``predictor.py`` / ``io.py`` / ...），但保持
**外部 API 100% 兼容**——下面的 re-export 让

    from segtask_v1.predictor import Predictor, run_inference

继续工作；同时 ``Predictor.<private_method>`` 在 ``Predictor.__new__(Predictor)``
风格的单元测试中也照常可用。
"""

from __future__ import annotations

from .predictor import Predictor, _AMP_DTYPES  # noqa: F401
from .io import (  # noqa: F401  (re-export)
    run_inference,
    _strip_compile_prefix,
    _unwrap_ema_state,
    _select_state_dict,
    _resolve_inference_precision,
    _PRECISION_CHOICES,
)

__all__ = [
    "Predictor",
    "run_inference",
]
