"""[shim] 已迁移至 ``taskcore.engine.amp``；此处保留旧路径别名，行为不变。"""

import sys

from taskcore.engine import amp as _impl

sys.modules[__name__] = _impl
