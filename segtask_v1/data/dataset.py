"""[shim] 已迁移至 ``taskcore.data.dataset``；此处保留旧路径别名，行为不变。"""

import sys

from taskcore.data import dataset as _impl

sys.modules[__name__] = _impl
