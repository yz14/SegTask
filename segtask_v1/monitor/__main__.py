"""[shim] 已迁移至 ``taskcore.monitor.__main__``；``python -m segtask_v1.monitor`` 入口保留。"""

from taskcore.monitor.__main__ import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
