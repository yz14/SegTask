"""[shim] 已迁移至 ``taskcore.data.augment``；此处保留旧路径别名，行为不变。"""

import sys

from taskcore.data import augment as _impl

sys.modules[__name__] = _impl
