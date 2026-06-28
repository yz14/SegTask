"""SSL 方法注册表。

新增方法：在此 import 子类并加入 ``_REGISTRY``，同时把注册键同步到
``ssltask.config.METHODS``（供配置校验）。``build_method`` 按 ``ssl.method`` 分派。
"""

from __future__ import annotations

from typing import Dict, Type

import torch

from .base import SSLMethod
from .dino import DINOMethod
from .genesis import GenesisMethod
from .jepa import JEPAMethod
from .prior import PriorMethod
from .simmim import SimMIMMethod
from .spark import SparKMethod

_REGISTRY: Dict[str, Type[SSLMethod]] = {
    GenesisMethod.name: GenesisMethod,
    PriorMethod.name: PriorMethod,
    SimMIMMethod.name: SimMIMMethod,
    DINOMethod.name: DINOMethod,
    SparKMethod.name: SparKMethod,
    JEPAMethod.name: JEPAMethod,
}


def build_method(cfg, ssl, device: torch.device) -> SSLMethod:
    """按 ``ssl.method`` 构造方法实例。"""
    cls = _REGISTRY.get(ssl.method)
    if cls is None:
        raise ValueError(
            f"Unknown ssl.method {ssl.method!r}; "
            f"registered: {sorted(_REGISTRY)}.")
    return cls(cfg, ssl, device)


__all__ = ["SSLMethod", "build_method"]
