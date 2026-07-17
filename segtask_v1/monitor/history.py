"""[shim] 已迁移至 ``taskcore.monitor.history``；此处保留旧路径别名，行为不变。"""

import sys

from taskcore.monitor import history as _impl

sys.modules[__name__] = _impl
