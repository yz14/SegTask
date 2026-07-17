"""ViT 特征提取器（2D/3D 通用，``spatial_dims`` 参数化）。

Vision Transformer（Dosovitskiy et al., ICLR 2021）标准实现：patch-embed
（Conv stride=patch）→ 可学习位置编码 → Pre-LN Transformer block ×N →
末 LN。接口与 segtask ``Encoder`` 对齐：``forward(x) -> List[feat]``，把
token 序列 reshape 回特征图 ``(B, C, *grid)``（仅单尺度，list 长度 1），供
统一的池化分类头使用（等价于 mean-token 池化，无 [CLS]）。

drop-path 复用 segtask ``blocks.DropPath``。
"""

from __future__ import annotations

import math
from typing import List, Sequence

import torch
import torch.nn as nn

from taskcore.models.blocks import DropPath


class PatchEmbed(nn.Module):
    """Conv(k=stride=patch) patch embedding；记录网格形状供 reshape。"""

    def __init__(self, in_channels: int, embed_dim: int,
                 patch_size: Sequence[int], spatial_dims: int):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.patch_size = tuple(int(p) for p in patch_size)
        conv = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        self.proj = conv(in_channels, embed_dim,
                         kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for ax, p in enumerate(self.patch_size):
            size = x.shape[2 + ax]
            if size % p != 0:
                raise ValueError(
                    f"input spatial size {tuple(x.shape[2:])} not divisible "
                    f"by vit patch_size {self.patch_size} (axis {ax}).")
        x = self.proj(x)                      # (B, C, *grid)
        self.grid = tuple(x.shape[2:])
        return x.flatten(2).transpose(1, 2)   # (B, N, C)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Pre-LN：x + DropPath(MHSA(LN x)) → x + DropPath(MLP(LN x))。"""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float,
                 drop_path: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        self.drop_path = (DropPath(drop_path) if drop_path > 0
                          else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop_path(h)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViTEncoder(nn.Module):
    """标准 ViT 主干；输出单尺度特征图 ``[(B, C, *grid)]``。"""

    def __init__(
        self,
        in_channels : int,
        embed_dim   : int = 384,
        depth       : int = 8,
        num_heads   : int = 6,
        mlp_ratio   : float = 4.0,
        drop_path_rate: float = 0.1,
        patch_size  : Sequence[int] = (4, 16, 16),
        input_size  : Sequence[int] = (16, 128, 128),
        spatial_dims: int = 3):
        super().__init__()
        if spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3; got {spatial_dims}")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads "
                f"({num_heads}).")
        ps = list(int(p) for p in patch_size)
        if spatial_dims == 2:
            ps = ps[-2:]
        if len(ps) != spatial_dims:
            raise ValueError(
                f"patch_size {patch_size} incompatible with spatial_dims="
                f"{spatial_dims}.")
        self.spatial_dims = spatial_dims
        self.patch_embed = PatchEmbed(in_channels, embed_dim, ps, spatial_dims)
        insz = [int(s) for s in input_size]
        if spatial_dims == 2:
            insz = insz[-2:]
        if len(insz) != spatial_dims:
            raise ValueError(
                f"input_size {input_size} incompatible with spatial_dims="
                f"{spatial_dims}.")
        for size, p in zip(insz, ps):
            if size % p != 0:
                raise ValueError(
                    f"input_size {input_size} must be divisible by "
                    f"patch_size {patch_size} on spatial axes.")
        # 位置编码按构造时的输入网格建立；推理时网格变化则按轴插值适配。
        self._pos_grid: tuple = tuple(size // p for size, p in zip(insz, ps))
        n_tokens = int(math.prod(self._pos_grid))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, dpr[i])
            for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.out_channels_list = [embed_dim]

    def _positional(self, grid: tuple) -> torch.Tensor:
        if grid == self._pos_grid:
            return self.pos_embed
        # 网格变化（如推理 patch 尺寸不同）：按轴插值位置编码。
        c = self.pos_embed.shape[-1]
        pe = self.pos_embed.reshape(1, *self._pos_grid, c)
        perm = (0, self.spatial_dims + 1, *range(1, self.spatial_dims + 1))
        pe = pe.permute(*perm)  # (1, C, *old_grid)
        mode = "trilinear" if self.spatial_dims == 3 else "bilinear"
        pe = nn.functional.interpolate(pe, size=grid, mode=mode,
                                       align_corners=False)
        pe = pe.flatten(2).transpose(1, 2)  # (1, N, C)
        return pe

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        tokens = self.patch_embed(x)                    # (B, N, C)
        grid = self.patch_embed.grid
        tokens = tokens + self._positional(grid)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        feat = tokens.transpose(1, 2).reshape(
            tokens.shape[0], tokens.shape[2], *grid)    # (B, C, *grid)
        return [feat]


__all__ = ["ViTEncoder", "PatchEmbed", "Block"]
