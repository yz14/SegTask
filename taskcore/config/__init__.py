"""taskcore.config —— 五任务共用配置层（核心段：data/aug/model/loss/train/predict/vis/monitor）。"""

from .core import *  # noqa: F401,F403
from . import core as _core

__all__ = list(getattr(_core, "__all__", [])) or [n for n in dir(_core) if not n.startswith("_")]
