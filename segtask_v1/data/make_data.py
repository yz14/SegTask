"""[shim] 已迁移至 ``taskcore.data.make_data``；此处保留旧路径别名，行为不变。

``python -m segtask_v1.data.make_data`` 入口同样保留。
"""

import sys

from taskcore.data import make_data as _impl

if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
