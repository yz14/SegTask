"""[shim] 已迁移至 ``taskcore.monitor.dashboard``；此处保留旧路径别名，行为不变。"""

import sys

from taskcore.monitor import dashboard as _impl

sys.modules[__name__] = _impl
