"""[shim] ``VolumeCache`` 已与 ``taskcore.data.dataset`` 合流；保留旧路径 re-export。"""

from __future__ import annotations

from taskcore.data.dataset import VolumeCache  # noqa: F401

__all__ = ["VolumeCache"]
