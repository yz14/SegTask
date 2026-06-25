"""训练过程监测工具（TODO #2）。

与 ``segtask_v1.visualization``（静态架构 / 数据流图）正交：本包关注**训练时序**
—— 逐 epoch 的损失、各项指标、学习率、显存等随训练演进的曲线，支持实时更新、
训练后 best 模型指标汇总，以及多 run 对比。产物为自包含、零外部依赖的 HTML
仪表盘。

数据层（``history.py``）与渲染层（``dashboard.py``）均可独立使用；Trainer 实时
集成见 ``trainer.trainer``，离线（重）渲染与多 run 对比 CLI 见 ``__main__``
（``python -m segtask_v1.monitor``）。
"""

from __future__ import annotations

from .dashboard import (
    render_comparison,
    render_dashboard,
    write_comparison,
    write_dashboard,
)
from .history import EpochRecord, MetricsHistory, MetricsLogger

__all__ = [
    "EpochRecord",
    "MetricsHistory",
    "MetricsLogger",
    "render_dashboard",
    "render_comparison",
    "write_dashboard",
    "write_comparison",
]
