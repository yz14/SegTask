"""Generation trainer utilities.

The super-resolution path reuses the shared optimizer / scheduler / AMP /
checkpoint helpers and exposes :class:`GenerationTrainer`.
"""

from __future__ import annotations

from taskcore.engine.checkpoint import _select_state_dict, _strip_compile_prefix, _unwrap_ema_state, unwrap_compile
from taskcore.engine.optim import build_optimizer, build_scheduler, WarmupScheduler
from .gen_trainer import GenerationTrainer

__all__ = [
    "GenerationTrainer",
    "build_optimizer",
    "build_scheduler",
    "WarmupScheduler",
    "unwrap_compile",
    "_strip_compile_prefix",
    "_unwrap_ema_state",
    "_select_state_dict",
]
