"""Checkpoint 工具：state_dict 解析、前缀剥离、compile 包装拆解。

主流程方法（``_build_state_dict`` / ``_save_checkpoint`` / ``_load_checkpoint``
/ ``_load_pretrain``）保留在 ``Trainer`` 类上，便于现有测试通过
``inspect.getsource(Trainer._build_state_dict)`` 校验关键 token；本模块仅承载
完全静态的辅助函数。
"""

from __future__ import annotations

import torch
import torch.nn as nn


def unwrap_compile(m: nn.Module) -> nn.Module:
    """剥 ``torch.compile`` 的 ``_orig_mod`` 包装。"""
    return getattr(m, "_orig_mod", m)


def extract_model_state_dict(ckpt, prefer_ema: bool):
    """定位 ckpt 里的 model state_dict，兼容 3 种布局：

    * 本 trainer ckpt（含 ``model_state_dict`` / ``model_online_state_dict``）
    * 第三方 ``{"state_dict": ...}``
    * 裸 ``OrderedDict``

    Returns
    -------
    (state_dict, source_label)
    """
    # 裸 state_dict
    if not isinstance(ckpt, dict) or all(
            isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt, "raw_state_dict"

    # 优先 EMA shadow
    if prefer_ema and "ema_state_dict" in ckpt:
        ema_state = ckpt["ema_state_dict"]
        if isinstance(ema_state, dict) and "shadow" in ema_state:
            return ema_state["shadow"], "ema_shadow"

    # trainer-format 在线权重
    if "model_online_state_dict" in ckpt:
        return ckpt["model_online_state_dict"], "model_online_state_dict"
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], "model_state_dict"

    # 第三方 state_dict
    if "state_dict" in ckpt:
        return ckpt["state_dict"], "state_dict"

    raise KeyError(
        "Pretrain checkpoint does not contain a recognisable model "
        "state_dict. Expected one of: 'model_state_dict', "
        "'model_online_state_dict', 'state_dict', or a raw OrderedDict.")


def strip_common_prefixes(sd):
    """剥去 ``module.``（DDP）与 ``_orig_mod.``（torch.compile）前缀。"""
    if not isinstance(sd, dict):
        return sd
    prefixes = ("module.", "_orig_mod.")
    out = {}
    changed = False
    for k, v in sd.items():
        new_k = k
        # 反复剥防嵌套包装。
        while new_k.startswith(prefixes):
            for p in prefixes:
                if new_k.startswith(p):
                    new_k = new_k[len(p):]
                    changed = True
                    break
        out[new_k] = v
    return out if changed else sd


__all__ = [
    "unwrap_compile",
    "extract_model_state_dict",
    "strip_common_prefixes",
]
