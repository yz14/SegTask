"""iBOT⑥ 专有模块：密集投影头构造 + 逐位点应用工具。

iBOT 在 ④DINO 的全局投影头之外，再对**密集**（逐位点）特征施加一个投影头，把每个空间
位点的特征向量映射到共享原型上的 logits（与全局头同构）。这里直接复用 ``DINOHead``
（它在 ``(M, in_dim)`` 上工作、与 spatial_dims 无关），把 ``(B, C, *grid)`` 展平成
``(B*N, C)`` 逐位点投影后再 reshape 回 ``(B, N, out_dim)``。
"""

from __future__ import annotations

import torch

from .dino_modules import DINOHead


def build_ibot_head(
    in_dim        : int,
    out_dim       : int,
    hidden_dim    : int = 2048,
    bottleneck_dim: int = 256,
    n_layers      : int = 3,
    use_bn        : bool = False) -> DINOHead:
    """构造 iBOT 密集投影头（复用 ``DINOHead``，逐位点作用于密集特征）。"""
    return DINOHead(
        in_dim=int(in_dim), out_dim=int(out_dim), hidden_dim=int(hidden_dim),
        bottleneck_dim=int(bottleneck_dim), n_layers=int(n_layers),
        use_bn=bool(use_bn))


def dense_head_forward(head: DINOHead, feat: torch.Tensor) -> torch.Tensor:
    """逐位点应用投影头：``(B, C, *grid)`` 密集特征 → ``(B, N, out_dim)`` logits。"""
    b, c = feat.shape[0], feat.shape[1]
    x = feat.flatten(2).transpose(1, 2)            # (B, N, C)
    n = x.shape[1]
    x = head(x.reshape(b * n, c))                  # (B*N, out_dim)
    return x.view(b, n, -1)                        # (B, N, out_dim)


__all__ = ["build_ibot_head", "dense_head_forward"]
