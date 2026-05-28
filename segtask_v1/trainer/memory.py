"""GPU 内存静态估计：params / grads / optimizer / EMA 持久占用（MiB）。

不含激活与 cuDNN workspace —— 后者由 epoch 内 ``max_memory_allocated`` 报。
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


def estimate_train_memory(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: Optional[object] = None,
) -> Dict[str, float]:
    """静态估计持久 GPU 内存（MiB）。``ema`` 期望具 ``shadow: dict``（``ModelEMA``）。"""
    MIB = 1 << 20
    params = list(model.parameters())

    param_bytes = sum(p.numel() * p.element_size() for p in params)
    grad_bytes = sum(p.numel() * p.element_size()
                     for p in params if p.requires_grad)

    optim_name = type(optimizer).__name__
    n_train = sum(p.numel() for p in params if p.requires_grad)
    adam_family = {"Adam", "AdamW", "RAdam", "NAdam", "Adamax"}
    if optim_name in adam_family:
        optim_mult = 2
    elif optim_name == "SGD":
        has_momentum = any(g.get("momentum", 0) > 0
                           for g in optimizer.param_groups)
        optim_mult = 1 if has_momentum else 0
    elif optim_name == "Lion":
        optim_mult = 1
    else:
        optim_mult = 2  # 保守默认
    optim_bytes = optim_mult * n_train * 4

    ema_bytes = 0
    if ema is not None:
        shadow = getattr(ema, "shadow", None)
        if shadow is not None:
            ema_bytes = sum(t.numel() * t.element_size()
                            for t in shadow.values())

    persistent = param_bytes + grad_bytes + optim_bytes + ema_bytes
    return {
        "param_mib": param_bytes / MIB,
        "grad_mib": grad_bytes / MIB,
        "optim_mib": optim_bytes / MIB,
        "optim_mult": optim_mult,
        "optim_name": optim_name,
        "ema_mib": ema_bytes / MIB,
        "persistent_mib": persistent / MIB,
    }


__all__ = ["estimate_train_memory"]
