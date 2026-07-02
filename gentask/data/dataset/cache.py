"""In-memory cache helpers for gentask.data.dataset."""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import numpy as np

class VolumeCache:
    """内存卷 LRU 缓存。max_volumes=0 不限容量；enabled=False 禁用。"""

    def __init__(self, enabled: bool = False, max_volumes: int = 0):
        self._enabled = enabled
        self._max = max(int(max_volumes), 0)
        self._store: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def get(self, path: str) -> Optional[np.ndarray]:
        if not self._enabled:
            return None
        data = self._store.get(path)
        if data is not None:
            # Mark as most-recently-used.
            self._store.move_to_end(path)
        return data

    def put(self, path: str, data: np.ndarray) -> None:
        if not self._enabled:
            return
        if path in self._store:
            self._store.move_to_end(path)
            self._store[path] = data
            return
        self._store[path] = data
        if self._max > 0:
            while len(self._store) > self._max:
                # popitem(last=False) pops the LEAST-recently-used entry.
                self._store.popitem(last=False)

    @property
    def size(self) -> int:
        return len(self._store)

    # Pickling：传到 DataLoader worker 时丢弃缓存内容（Windows spawn 下防管道超限，
    # 并且 worker 间不共享内存，传输是纯开销）。
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_store"] = OrderedDict()
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if not isinstance(self._store, OrderedDict):
            self._store = OrderedDict()


# ---------------------------------------------------------------------------
# Common npz-backed Dataset base
# ---------------------------------------------------------------------------
