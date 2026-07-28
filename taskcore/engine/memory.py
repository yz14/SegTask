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
    # ZeRO-1 分片：每卡仅持有 1/world_size 的优化器状态；内层优化器类型取自
    # ZeroRedundancyOptimizer 的构造参数（未取到时按 Adam 族保守估）。
    zero_shard_div = 1
    if optim_name == "ZeroRedundancyOptimizer":
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            zero_shard_div = max(dist.get_world_size(), 1)
        inner_cls = getattr(optimizer, "_optim_constructor", None)
        if inner_cls is not None:
            optim_name = getattr(inner_cls, "__name__", optim_name)
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
    optim_bytes = optim_mult * n_train * 4 // zero_shard_div

    # EMA shadow 只计 GPU 常驻部分（CPU offload 时不占显存）。
    ema_bytes = 0
    ema_backup_bytes = 0
    if ema is not None:
        shadow = getattr(ema, "shadow", None)
        if shadow is not None:
            ema_bytes = sum(t.numel() * t.element_size()
                            for t in shadow.values() if t.is_cuda)
            # apply_shadow 首次调用时会为 GPU EMA 分配一份 live backup；
            # 将这份临时但可观测的峰值纳入预算，即使尚未发生 swap。
            backup = getattr(ema, "_backup", None) or {}
            if backup:
                ema_backup_bytes = sum(
                    t.numel() * t.element_size()
                    for t in backup.values() if t.is_cuda)
            elif any(t.is_cuda for t in shadow.values()):
                ema_backup_bytes = sum(
                    t.numel() * t.element_size()
                    for t in model.state_dict().values()
                    if t.is_cuda)

    persistent = (param_bytes + grad_bytes + optim_bytes + ema_bytes
                  + ema_backup_bytes)
    return {
        "param_mib": param_bytes / MIB,
        "grad_mib": grad_bytes / MIB,
        "optim_mib": optim_bytes / MIB,
        "optim_mult": optim_mult,
        "optim_name": optim_name,
        "ema_mib": ema_bytes / MIB,
        "ema_backup_mib": ema_backup_bytes / MIB,
        "persistent_mib": persistent / MIB,
    }


def _resenc_level_volumes(patch_size, n_levels):
    """逐级空间体素数（模拟各向异性自动减半：偶数、减半后 >=4、相对最大轴不过小）。"""
    sizes = [int(x) for x in patch_size]
    vols = []
    for _ in range(n_levels):
        v = 1
        for s in sizes:
            v *= max(int(s), 1)
        vols.append(v)
        ref = max(sizes)
        sizes = [s // 2 if (s % 2 == 0 and s // 2 >= 4 and s * 2 > ref)
                 else s for s in sizes]
    return vols


def estimate_resenc_train_memory_gb(
    patch_size,
    encoder_channels,
    encoder_blocks,
    decoder_blocks,
    batch_size: int,
    *,
    amp: bool = True,
    optim_mult: int = 2,
) -> Dict[str, float]:
    """ResEnc 训练显存粗估（GB），纯算术、不实例化模型（2-6 显存分档用）。

    组成：
    * 训练态（params + grads + optimizer）——按 3×3×3 conv 残差块解析估算；
    * 激活——按 backward 需保留的逐块特征图估算（AMP 半字节宽），乘 1.3
      安全系数覆盖 cuDNN workspace / 碎片。

    这是**分档预算估计**（±30% 量级），非精确 profiler；实测以
    ``torch.cuda.max_memory_allocated`` 为准。
    """
    GIB = float(1 << 30)
    chans = [int(c) for c in encoder_channels]
    n_levels = len(chans)
    enc_b = list(encoder_blocks) or [2] * n_levels
    dec_b = list(decoder_blocks) or [1] * max(n_levels - 1, 1)
    if len(dec_b) < n_levels - 1:
        dec_b = dec_b + [dec_b[-1]] * (n_levels - 1 - len(dec_b))

    # ---- 参数量（3^3 conv 残差块：每 block 2 conv C→C；stage 首 conv 升通道）----
    k3 = 27
    n_params = 0
    prev = 1
    for i, c in enumerate(chans):
        n_params += k3 * prev * c                    # stage 首 conv / 下采样
        n_params += enc_b[i] * 2 * k3 * c * c        # 残差 blocks
        prev = c
    for i in range(n_levels - 2, -1, -1):            # decoder（镜像，含上采样）
        c = chans[i]
        n_params += k3 * chans[i + 1] * c
        n_params += dec_b[i] * 2 * k3 * (2 * c) * c  # skip cat 后首 conv 2C→C 粗估
    state_gb = n_params * 4.0 * (2 + optim_mult) / GIB   # params+grads+optim(fp32)

    # ---- 激活（backward 保留量）----
    bytes_el = 2.0 if amp else 4.0
    vols = _resenc_level_volumes(patch_size, n_levels)
    act = 0.0
    for i, c in enumerate(chans):
        keep = 2 * enc_b[i] + 2                      # 每 block 2 conv 输出 + stage IO
        if i < n_levels - 1:
            keep += 2 * dec_b[i] + 2                 # decoder 镜像
        act += keep * c * vols[i]
    act_gb = act * bytes_el * float(batch_size) * 1.3 / GIB

    total = state_gb + act_gb
    return {
        "params_m": n_params / 1e6,
        "state_gb": state_gb,
        "activation_gb": act_gb,
        "total_gb": total,
    }


__all__ = ["estimate_train_memory", "estimate_resenc_train_memory_gb"]
