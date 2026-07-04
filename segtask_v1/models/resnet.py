"""UNet stage 用 ResNet 块。block_type 示例：'basic' 轻量后置激活 、'r2plus1d' (1,3,3)+(3,1,1) 仅 3D。还有 'preact'/'bottleneck'。下采样由 blocks.Downsample 外部完成。"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .blocks import (
    DropPath, _CONV, _DROP, SqueezeExcite3D,
    get_activation, get_norm, make_attention)


class ResNetBlock(nn.Module):
    """后置激活 ResNet 块（可选 attention）。attention_type='se' 时启用 SE。"""

    def __init__(
        self,
        in_ch         : int,
        out_ch        : int,
        norm_type     : str = "instance",
        norm_groups   : int = 8,
        activation    : str = "leakyrelu",
        dropout       : float = 0.0,
        se_reduction  : int = 16,
        attention_type: str = "none",
        drop_path     : float = 0.0,
        spatial_dims  : int = 3):
        super().__init__()
        d = spatial_dims
        self.conv1 = _CONV[d](in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = get_norm(norm_type, out_ch, norm_groups, spatial_dims=d)
        self.act1  = get_activation(activation)

        self.conv2 = _CONV[d](out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = get_norm(norm_type, out_ch, norm_groups, spatial_dims=d)
        self.act2  = get_activation(activation)

        self.drop = _DROP[d](dropout) if dropout > 0 else nn.Identity()

        self.attn = make_attention(attention_type, out_ch, spatial_dims=d, reduction=se_reduction)
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0
            else nn.Identity())

        self.shortcut = (
            nn.Sequential(_CONV[d](in_ch, out_ch, 1, bias=False),
                          get_norm(norm_type, out_ch, norm_groups, spatial_dims=d))
            if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.drop(out)
        out = self.norm2(self.conv2(out))
        out = self.attn(out)
        return self.act2(res + self.drop_path(out))


class PreActResNetBlock(nn.Module):
    """预激活 ResNet 块 (He 2016)：norm-act-conv × 2 + 残差；适合深 encoder。"""

    def __init__(
        self,
        in_ch         : int,
        out_ch        : int,
        norm_type     : str = "instance",
        norm_groups   : int = 8,
        activation    : str = "leakyrelu",
        dropout       : float = 0.0,
        se_reduction  : int = 16,
        attention_type: str = "none",
        drop_path     : float = 0.0,
        spatial_dims  : int = 3):
        super().__init__()
        d = spatial_dims
        self.norm1 = get_norm(norm_type, in_ch, norm_groups, spatial_dims=d)
        self.act1  = get_activation(activation)
        self.conv1 = _CONV[d](in_ch, out_ch, 3, padding=1, bias=False)

        self.norm2 = get_norm(norm_type, out_ch, norm_groups, spatial_dims=d)
        self.act2  = get_activation(activation)
        self.conv2 = _CONV[d](out_ch, out_ch, 3, padding=1, bias=False)

        self.drop = _DROP[d](dropout) if dropout > 0 else nn.Identity()

        self.attn = make_attention(attention_type, out_ch, spatial_dims=d, reduction=se_reduction)
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0
            else nn.Identity())

        # shortcut 作用于原 x；通道不匹配时用 1×1 投影（标准 pre-act）。
        self.shortcut = (
            _CONV[d](in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.conv1(self.act1(self.norm1(x)))
        out = self.drop(out)
        out = self.conv2(self.act2(self.norm2(out)))
        out = self.attn(out)
        return self.drop_path(out) + res


class BottleneckBlock(nn.Module):
    """ResNet-50 风 bottleneck：1×1 压 → 3×3 → 1×1 扩（expansion=4，适 ResEnc-XL）。"""

    def __init__(
        self,
        in_ch         : int,
        out_ch        : int,
        expansion     : int = 4,
        norm_type     : str = "instance",
        norm_groups   : int = 8,
        activation    : str = "leakyrelu",
        dropout       : float = 0.0,
        se_reduction  : int = 16,
        attention_type: str = "none",
        drop_path     : float = 0.0,
        spatial_dims  : int = 3):
        super().__init__()
        d = spatial_dims
        mid = max(out_ch // expansion, 1)  # 压缩

        self.conv1 = _CONV[d](in_ch, mid, 1, bias=False)
        self.norm1 = get_norm(norm_type, mid, norm_groups, spatial_dims=d)
        self.act1  = get_activation(activation)

        self.conv2 = _CONV[d](mid, mid, 3, padding=1, bias=False)
        self.norm2 = get_norm(norm_type, mid, norm_groups, spatial_dims=d)
        self.act2  = get_activation(activation)

        self.conv3 = _CONV[d](mid, out_ch, 1, bias=False)
        self.norm3 = get_norm(norm_type, out_ch, norm_groups, spatial_dims=d)
        self.act3  = get_activation(activation)

        self.drop = _DROP[d](dropout) if dropout > 0 else nn.Identity()

        self.attn = make_attention(attention_type, out_ch, spatial_dims=d, reduction=se_reduction)
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0
            else nn.Identity())

        self.shortcut = (
            nn.Sequential(_CONV[d](in_ch, out_ch, 1, bias=False),
                          get_norm(norm_type, out_ch, norm_groups, spatial_dims=d))
            if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.act2(self.norm2(self.conv2(out)))
        out = self.drop(out)
        out = self.norm3(self.conv3(out))
        out = self.attn(out)
        return self.act3(res + self.drop_path(out))


class R2Plus1DBlock(nn.Module):
    """R(2+1)D 残差块 (Tran 2018)，仅 3D。每个 3×3×3 拆为 (1,3,3) 空间 + norm + act + (3,1,1) 时间；中间非线性不可省，mid_ch=out_ch。"""

    def __init__(
        self,
        in_ch          : int,
        out_ch         : int,
        norm_type      : str = "instance",
        norm_groups    : int = 8,
        activation     : str = "leakyrelu",
        dropout        : float = 0.0,
        se_reduction   : int = 16,
        attention_type : str = "none",
        drop_path      : float = 0.0,
        spatial_dims   : int = 3,
        temporal_kernel: int = 3):
        super().__init__()
        if spatial_dims != 3:
            # D 必须是真空间轴；2.5D 中 D 被折叠到通道。
            raise ValueError(
                "R2Plus1DBlock requires spatial_dims=3 (D must be a real "
                "spatial axis). For 2.5D mode (spatial_dims=2), use "
                "block_type='basic'/'preact'/'bottleneck' instead, or "
                "switch your config to a 3D patch_mode (z_axis / cubic / "
                "whole) where the depth axis is preserved.")
        if temporal_kernel < 1 or temporal_kernel % 2 == 0:
            raise ValueError(
                f"temporal_kernel must be a positive odd integer, "
                f"got {temporal_kernel}")
        d = 3
        t_pad = temporal_kernel // 2

        # 第一组 (2+1)D：in_ch → out_ch。
        self.spatial1 = nn.Conv3d(
            in_ch, out_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        self.norm_s1  = get_norm(norm_type, out_ch, norm_groups, spatial_dims=d)
        self.act_s1   = get_activation(activation)

        self.temporal1 = nn.Conv3d(
            out_ch, out_ch, kernel_size=(temporal_kernel, 1, 1), padding=(t_pad, 0, 0), bias=False)
        self.norm_t1   = get_norm(norm_type, out_ch, norm_groups, spatial_dims=d)
        self.act_t1    = get_activation(activation)

        # 第二组 (2+1)D：out_ch → out_ch。
        self.spatial2 = nn.Conv3d(
            out_ch, out_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        self.norm_s2  = get_norm(norm_type, out_ch, norm_groups, spatial_dims=d)
        self.act_s2   = get_activation(activation)

        self.temporal2 = nn.Conv3d(
            out_ch, out_ch, kernel_size=(temporal_kernel, 1, 1), padding=(t_pad, 0, 0), bias=False)
        self.norm_t2   = get_norm(norm_type, out_ch, norm_groups, spatial_dims=d)
        self.act_out   = get_activation(activation)  # 残差相加后再激活

        self.drop = _DROP[d](dropout) if dropout > 0 else nn.Identity()

        self.attn = make_attention(attention_type, out_ch, spatial_dims=d, reduction=se_reduction)
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0
            else nn.Identity())

        self.shortcut = (
            nn.Sequential(
                _CONV[d](in_ch, out_ch, 1, bias=False),
                get_norm(norm_type, out_ch, norm_groups, spatial_dims=d))
            if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                f"R2Plus1DBlock expects rank-5 input (B, C, D, H, W); "
                f"got shape={tuple(x.shape)}.")
        res = self.shortcut(x)
        out = self.act_s1(self.norm_s1(self.spatial1(x)))
        out = self.act_t1(self.norm_t1(self.temporal1(out)))
        out = self.drop(out)
        # 后置激活：残差前不加 act（对齐 ResNetBlock）。
        out = self.act_s2(self.norm_s2(self.spatial2(out)))
        out = self.norm_t2(self.temporal2(out))
        out = self.attn(out)
        return self.act_out(res + self.drop_path(out))


_BLOCK_REGISTRY = {
    "basic"     : ResNetBlock,
    "preact"    : PreActResNetBlock,
    "bottleneck": BottleneckBlock,
    "r2plus1d"  : R2Plus1DBlock}

BLOCK_TYPES = tuple(_BLOCK_REGISTRY.keys())


def _make_block(block_type: str, in_ch: int, out_ch: int, **kwargs) -> nn.Module:
    if block_type not in _BLOCK_REGISTRY:
        raise ValueError(
            f"Unknown block_type: {block_type!r}. Valid: {BLOCK_TYPES}")
    return _BLOCK_REGISTRY[block_type](in_ch, out_ch, **kwargs)


def _mrf_dilation_padding(dilation: int, spatial_dims: int, axes: str):
    """3×3(×3) 卷积在给定膨胀率/作用轴下的 per-axis (dilation, padding)。

    kernel=3 时 padding=dilation 可保持空间尺寸不变。``axes='hw'`` 时 3D 的 z 轴
    （首空间轴）恒 dilation=1，仅在 H/W 膨胀；2.5D（spatial_dims=2）天然只有 H/W。
    """
    if spatial_dims == 2:
        return (dilation, dilation), (dilation, dilation)
    if axes == "hw":
        return (1, dilation, dilation), (1, dilation, dilation)
    return (dilation, dilation, dilation), (dilation, dilation, dilation)


class MultiRFBlock(nn.Module):
    """多感受野残差块：第一卷积换成「多膨胀率并行分支 → 融合」，再接一层标准 3×3。

    分支永远包含一条 dilation=1 支路（守门，抗网格效应/保细管）。``mode='split'`` 时各
    分支均分 ``out_ch``（≈等成本）；``'parallel'`` 时各分支均为 ``out_ch``（≈N×成本）。
    融合：``concat_proj``（concat→1×1）/ ``sum``（需 parallel）/ ``se``（concat→SE→1×1）。
    其余（norm/act/dropout/attention/残差）与 ``ResNetBlock`` 一致（后置激活）。
    """

    def __init__(
        self,
        in_ch         : int,
        out_ch        : int,
        dilations     : List[int],
        mode          : str = "split",
        fusion        : str = "concat_proj",
        axes          : str = "hw",
        norm_type     : str = "instance",
        norm_groups   : int = 8,
        activation    : str = "leakyrelu",
        dropout       : float = 0.0,
        se_reduction  : int = 16,
        attention_type: str = "none",
        drop_path     : float = 0.0,
        spatial_dims  : int = 3,
        branch_norm_act: bool = False):
        super().__init__()
        d = spatial_dims
        dils = [int(x) for x in dilations]
        n = len(dils)
        if n < 1:
            raise ValueError("MultiRFBlock requires at least one dilation.")
        if 1 not in dils:
            raise ValueError(
                "MultiRFBlock requires a dilation=1 branch (anti-gridding).")
        if mode not in ("split", "parallel"):
            raise ValueError(f"Unknown MultiRF mode: {mode!r}.")
        if fusion not in ("concat_proj", "sum", "se"):
            raise ValueError(f"Unknown MultiRF fusion: {fusion!r}.")
        if fusion == "sum" and mode != "parallel":
            raise ValueError(
                "MultiRF fusion='sum' requires mode='parallel' "
                "(branches must share channel count).")
        self.fusion = fusion

        # 各分支输出通道数。
        if mode == "split":
            base = out_ch // n
            if base < 1:
                raise ValueError(
                    f"MultiRF mode='split' needs out_ch ({out_ch}) >= number "
                    f"of branches ({n}).")
            branch_ch = [base] * n
            rem = out_ch - base * n
            # 余数补给 dilation=1 支路（守门支路最厚）。
            id1 = dils.index(1)
            branch_ch[id1] += rem
        else:  # parallel
            branch_ch = [out_ch] * n

        self.branches = nn.ModuleList()
        for dil, c in zip(dils, branch_ch):
            dilation, padding = _mrf_dilation_padding(dil, d, axes)
            self.branches.append(
                _CONV[d](in_ch, c, kernel_size=3, padding=padding,
                         dilation=dilation, bias=False))
        total = sum(branch_ch)

        # 可选：ASPP 风格的 per-branch norm+act —— 每条膨胀卷积分支在 concat/相加
        # 融合「之前」各自接一层 norm+act，使各感受野分支成为独立的非线性特征提取
        # 器（否则多分支+融合在第一个激活前整体仍是线性映射）。默认关、向后兼容。
        self.branch_norm_act = bool(branch_norm_act)
        if self.branch_norm_act:
            self.branch_post = nn.ModuleList()
            for c in branch_ch:
                # split 模式下分支通道 = out_ch//n，GroupNorm 需 c 能被 norm_groups
                # 整除。仓库的 get_norm 会静默把 groups 折半直至整除（最坏退化为 1 组），
                # 这里改为显式报错，让用户据此决定改通道数或换 norm（不做自适配）。
                if norm_type == "group" and c % norm_groups != 0:
                    raise ValueError(
                        f"MultiRF branch_norm_act=True with norm_type='group': "
                        f"branch channel count {c} (out_ch={out_ch}, "
                        f"mode={mode!r}, {n} branches) is not divisible by "
                        f"norm_groups={norm_groups}. Per-branch GroupNorm "
                        f"requires each branch's channels to be a multiple of "
                        f"norm_groups. Fix by one of: change out_ch so each "
                        f"branch (out_ch//{n}) is divisible by {norm_groups}; "
                        f"set norm_groups to a divisor of {c}; switch norm_type "
                        f"to 'instance'/'batch'; or use multirf_mode='parallel' "
                        f"(each branch = out_ch).")
                self.branch_post.append(nn.Sequential(
                    get_norm(norm_type, c, norm_groups, spatial_dims=d),
                    get_activation(activation)))
        else:
            self.branch_post = None

        # 融合层。
        if fusion == "sum":
            self.se   = None
            self.fuse = None  # 逐元素相加，各分支均 out_ch
        elif fusion == "se":
            self.se = SqueezeExcite3D(total, reduction=se_reduction,
                                      spatial_dims=d)
            self.fuse = _CONV[d](total, out_ch, kernel_size=1, bias=False)
        else:  # concat_proj
            self.se = None
            self.fuse = _CONV[d](total, out_ch, kernel_size=1, bias=False)

        self.norm1 = get_norm(norm_type, out_ch, norm_groups, spatial_dims=d)
        self.act1  = get_activation(activation)
        self.drop  = _DROP[d](dropout) if dropout > 0 else nn.Identity()

        # 第二层标准 3×3（dilation=1），守细节、与 ResNetBlock 对齐。
        self.conv2 = _CONV[d](out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = get_norm(norm_type, out_ch, norm_groups, spatial_dims=d)

        self.attn = make_attention(attention_type, out_ch, spatial_dims=d,
                                   reduction=se_reduction)
        self.act2 = get_activation(activation)
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0
            else nn.Identity())

        self.shortcut = (
            nn.Sequential(_CONV[d](in_ch, out_ch, 1, bias=False),
                          get_norm(norm_type, out_ch, norm_groups, spatial_dims=d))
            if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        feats = [branch(x) for branch in self.branches]
        if self.branch_post is not None:
            feats = [post(f) for post, f in zip(self.branch_post, feats)]
        if self.fusion == "sum":
            out = feats[0]
            for f in feats[1:]:
                out = out + f
        else:
            out = torch.cat(feats, dim=1)
            if self.se is not None:
                out = self.se(out)
            out = self.fuse(out)
        out = self.act1(self.norm1(out))
        out = self.drop(out)
        out = self.norm2(self.conv2(out))
        out = self.attn(out)
        return self.act2(res + self.drop_path(out))


class MultiRFStage(nn.Module):
    """同分辨率下的 N 个 ``MultiRFBlock``，首块可变通道（与 ``ResNetStage`` 等价接口）。"""

    def __init__(
        self,
        in_ch         : int,
        out_ch        : int,
        num_blocks    : int,
        dilations     : List[int],
        mode          : str = "split",
        fusion        : str = "concat_proj",
        axes          : str = "hw",
        norm_type     : str = "instance",
        norm_groups   : int = 8,
        activation    : str = "leakyrelu",
        dropout       : float = 0.0,
        se_reduction  : int = 16,
        attention_type: str = "none",
        drop_path_rates: List[float] = None,
        spatial_dims  : int = 3,
        branch_norm_act: bool = False):
        super().__init__()
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
        if drop_path_rates is None:
            drop_path_rates = [0.0] * num_blocks
        kwargs = dict(
            dilations=dilations, mode=mode, fusion=fusion, axes=axes,
            norm_type=norm_type, norm_groups=norm_groups, activation=activation,
            dropout=dropout, se_reduction=se_reduction,
            attention_type=attention_type, drop_path=drop_path_rates[0],
            spatial_dims=spatial_dims,
            branch_norm_act=branch_norm_act)
        blocks = [MultiRFBlock(in_ch, out_ch, **kwargs)]
        for i in range(1, num_blocks):
            kwargs["drop_path"] = drop_path_rates[i] if i < len(drop_path_rates) else 0.0
            blocks.append(MultiRFBlock(out_ch, out_ch, **kwargs))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ResNetStage(nn.Module):
    """同分辨率下的 N 个残差块，首块可变通道。block_type：'basic'/'preact'/'bottleneck'/'r2plus1d'。"""

    def __init__(
        self,
        in_ch         : int,
        out_ch        : int,
        num_blocks    : int = 2,
        norm_type     : str = "instance",
        norm_groups   : int = 8,
        activation    : str = "leakyrelu",
        dropout       : float = 0.0,
        se_reduction  : int = 16,
        attention_type: str = "none",
        block_type    : str = "basic",
        spatial_dims  : int = 3,
        drop_path_rates: List[float] = None,
    ):
        super().__init__()
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
        if drop_path_rates is None:
            drop_path_rates = [0.0] * num_blocks
        kwargs = dict(
            norm_type=norm_type, norm_groups=norm_groups, activation=activation,
            dropout=dropout, se_reduction=se_reduction,
            attention_type=attention_type, spatial_dims=spatial_dims,
            drop_path=drop_path_rates[0])
        blocks = [_make_block(block_type, in_ch, out_ch, **kwargs)]
        for i in range(1, num_blocks):
            kwargs["drop_path"] = drop_path_rates[i] if i < len(drop_path_rates) else 0.0
            blocks.append(_make_block(block_type, out_ch, out_ch, **kwargs))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
