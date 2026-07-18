"""[shim] 已迁移至 ``taskcore.utils.common``；此处保留旧路径别名，行为不变。"""

import sys

from taskcore.utils import common as _impl

sys.modules[__name__] = _impl
