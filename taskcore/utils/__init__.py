"""taskcore.utils — 通用工具层。

* ``common``        —— seed / AverageMeter / ModelEMA / Timer / dice 等通用工具；
* ``logging_utils`` —— 彩色日志初始化。
"""

from . import common, logging_utils
from .common import *  # noqa: F401,F403
from .logging_utils import setup_logging  # noqa: F401
