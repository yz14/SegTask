"""[shim] 已迁移至 ``taskcore.data.mixed_sampler``；此处保留旧路径别名，行为不变。"""

import sys

from taskcore.data import mixed_sampler as _impl

sys.modules[__name__] = _impl
