"""segtask_v1.trainer 包入口。

向后兼容老的扁平 ``segtask_v1.trainer`` 模块：
``from segtask_v1.trainer import Trainer, build_scheduler, build_optimizer,
WarmupScheduler`` 仍然可用。

子模块布局（Round 1 拆分结果）：
    * ``trainer.optim``      —— ``build_optimizer`` / ``build_scheduler`` / ``WarmupScheduler``
    * ``trainer.amp``        —— ``GradScaler`` shim / autocast / fp32 包装
    * ``trainer.memory``     —— GPU 持久内存估计
    * ``trainer.breakdown``  —— per-step 损失分量收集与渲染
    * ``trainer.checkpoint`` —— ckpt state_dict 解析、前缀剥离、compile 拆包
    * ``trainer.trainer``    —— ``Trainer`` 主类（fit / _train_epoch / _validate / view 操作 / ckpt I/O）
"""

from __future__ import annotations

from taskcore.engine.optim import build_optimizer, build_scheduler, WarmupScheduler
from .trainer import Trainer

__all__ = [
    "Trainer",
    "build_optimizer",
    "build_scheduler",
    "WarmupScheduler",
]
