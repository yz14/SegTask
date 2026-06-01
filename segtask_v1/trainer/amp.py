"""AMP / autocast / GradScaler 适配与 fp32 损失包装。

与具体训练流程解耦的纯工具：
    * ``GradScaler`` 版本无关 shim（PyTorch ≥2.3 接受 device 首参）
    * ``_cuda_supports_bf16`` 设备能力探测
    * ``resolve_auto_amp_dtype`` 解析 ``amp_dtype='auto'``
    * ``compute_loss_fp32`` 在 autocast 外以 fp32 调用任意 loss
    * ``_AMP_DTYPES`` / ``_LOGIT_CLAMP`` 常量
"""

from __future__ import annotations

import inspect as _inspect
from typing import Optional

import torch
import torch.nn as nn

# PyTorch ≥2.3：torch.amp.GradScaler 接受 device 首参；旧版需走 torch.cuda.amp。
try:
    from torch.amp import GradScaler as _GradScaler  # type: ignore
    from torch.amp import autocast  # type: ignore
except ImportError:  # pragma: no cover
    from torch.cuda.amp import GradScaler as _GradScaler  # type: ignore
    from torch.amp import autocast  # type: ignore


_AMP_DTYPES = {
    "float16" : torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}


# logit 夹匯上限：fp16 可能产生 ±inf —— 导致 BCEWithLogits 出 NaN。
# sigmoid(50) ≈ 1.0，不影响正常训练（|x|≪20）行为。
_LOGIT_CLAMP: float = 50.0


def GradScaler(device: str = "cuda", **kwargs):  # noqa: N802
    """版本无关 GradScaler：新 API 才传 device。"""
    try:
        params = _inspect.signature(_GradScaler).parameters
    except (TypeError, ValueError):
        params = {}
    if "device" in params:
        return _GradScaler(device, **kwargs)
    return _GradScaler(**kwargs)


def _cuda_supports_bf16() -> bool:
    """当前 CUDA 设备是否原生支持 bf16（Ampere+/ROCm）。"""
    if not torch.cuda.is_available():
        return False
    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(is_bf16_supported):
        try:
            return bool(is_bf16_supported())
        except Exception:  # pragma: no cover - defensive
            pass
    try:
        major, _minor = torch.cuda.get_device_capability()
        return major >= 8
    except Exception:  # pragma: no cover - defensive
        return False


def resolve_auto_amp_dtype(device: torch.device) -> str:
    """``amp_dtype='auto'`` 解析：CUDA 支持 bf16 返 'bfloat16'，否则 'float16'。"""
    if device.type == "cuda" and _cuda_supports_bf16():
        return "bfloat16"
    return "float16"


def compute_loss_fp32(
    loss_fn: nn.Module,
    pred,
    target: torch.Tensor,
    weight_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """在 autocast 外以 fp32 调用 ``loss_fn``；pred 裁裁防 ±inf，整型 label 保持。

    Dice/BCE 在 fp16 下汇总易溢出 → NaN，因此即使 forward 在 AMP 下，
    损失计算亦强制 fp32。
    """
    c = _LOGIT_CLAMP
    if isinstance(pred, list):  # deep supervision
        pred_fp32 = [p.float().clamp(-c, c) for p in pred]
    else:
        pred_fp32 = pred.float().clamp(-c, c)
    target_fp32 = target.float() if target.is_floating_point() else target
    wmap_fp32   = weight_map.float() if weight_map is not None else None
    with autocast(device_type="cuda", enabled=False):
        if wmap_fp32 is None:
            return loss_fn(pred_fp32, target_fp32)
        return loss_fn(pred_fp32, target_fp32, weight_map=wmap_fp32)


__all__ = [
    "_AMP_DTYPES",
    "_LOGIT_CLAMP",
    "GradScaler",
    "autocast",
    "_cuda_supports_bf16",
    "resolve_auto_amp_dtype",
    "compute_loss_fp32",
]
