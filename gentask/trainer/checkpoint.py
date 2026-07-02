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


def _strip_compile_prefix(sd):
    """剥去 torch.compile 添加的 ``_orig_mod.`` 前缀。"""
    prefix = "_orig_mod."
    if isinstance(sd, dict) and any(k.startswith(prefix) for k in sd):
        return {(k[len(prefix):] if k.startswith(prefix) else k): v
                for k, v in sd.items()}
    return sd


def _unwrap_ema_state(ema_sd):
    """将 ``{shadow, decay}`` 拆为普通 state_dict；已是拆过的旧格式原返。"""
    if isinstance(ema_sd, dict) and "shadow" in ema_sd and isinstance(
            ema_sd["shadow"], dict):
        return ema_sd["shadow"]
    return ema_sd


def _select_state_dict(ckpt, variant: str):
    """从 ckpt 选权重。``variant``: ``'auto'`` (优 EMA) / ``'ema'`` / ``'online'``。

    返 ``(state_dict, label)``，``label`` 用于日志。
    """
    has_online = "model_online_state_dict" in ckpt
    has_ema = "ema_state_dict" in ckpt
    primary = ckpt["model_state_dict"]

    if variant == "online":
        return (ckpt["model_online_state_dict"] if has_online else primary,
                "online")
    if variant == "ema":
        if has_ema:
            return _unwrap_ema_state(ckpt["ema_state_dict"]), "ema"
        return (ckpt["model_online_state_dict"] if has_online else primary,
                "online")
    if has_ema:
        return _unwrap_ema_state(ckpt["ema_state_dict"]), "ema"
    return primary, "online"


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
    "_strip_compile_prefix",
    "_unwrap_ema_state",
    "_select_state_dict",
    "extract_model_state_dict",
    "strip_common_prefixes",
]
