"""[shim] 已迁移至 ``taskcore.models.topology``；此处保留旧路径别名，行为不变。"""

import sys

from taskcore.models import topology as _impl

sys.modules[__name__] = _impl
