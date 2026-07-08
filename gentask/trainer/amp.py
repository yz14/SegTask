"""AMP / autocast / GradScaler 适配。

与具体训练流程解耦的纯工具：
    * ``GradScaler`` 版本无关 shim（PyTorch ≥2.3 接受 device 首参）
    * ``_cuda_supports_bf16`` 设备能力探测
    * ``resolve_auto_amp_dtype`` 解析 ``amp_dtype='auto'``
    * ``_AMP_DTYPES`` 常量
"""

from __future__ import annotations

import inspect as _inspect

import torch

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


__all__ = [
    "_AMP_DTYPES",
    "GradScaler",
    "autocast",
    "_cuda_supports_bf16",
    "resolve_auto_amp_dtype",
]
