"""taskcore.engine — 训练 / 推理工程件。

* ``amp``        —— autocast / GradScaler / fp32 损失封装；
* ``optim``      —— 优化器 / 调度器 / warmup 工厂；
* ``checkpoint`` —— checkpoint I/O、前缀剥离、compile 解包；
* ``dist_utils`` —— 分布式辅助；
* ``memory``     —— 显存预算与统计；
* ``prefetch``   —— CUDA 预取；
* ``views``      —— 2.5D 折叠原语与折叠契约；
* ``base_trainer``   —— 五任务训练器共用工程件基类 ``BaseTrainer``；
* ``base_predictor`` —— 五任务推理器共用工程件基类 ``BasePredictor``。
"""

from . import (  # noqa: F401
    amp, base_predictor, base_trainer, checkpoint, dist_utils, memory,
    optim, prefetch, views,
)
from .base_predictor import BasePredictor  # noqa: F401
from .base_trainer import (  # noqa: F401
    BaseTrainer, OptimStepResult, reseed_rank_rng, _reseed_rank_rng)
