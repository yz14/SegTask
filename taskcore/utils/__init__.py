"""taskcore.utils — 通用工具层。

* ``common``        —— seed / AverageMeter / ModelEMA / Timer 等通用工具
  （dice 等指标数学在 ``taskcore.metrics``，此处经 re-export 兼容）；
* ``logging_utils`` —— 彩色日志初始化。
"""

from . import common, logging_utils
from .common import *  # noqa: F401,F403
from .logging_utils import setup_logging  # noqa: F401
