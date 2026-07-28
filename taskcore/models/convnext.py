"""ConvNeXt blocks for 3D UNet (Liu 2022, dim-agnostic 2D/3D).

Block: dwconv7 → LN(C) → pwconv(4x) → GELU → pwconv → LayerScale → residual+DropPath.
Downsample: LN → Conv(k=2,s=2), paper-faithful inter-stage.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from .blocks import DropPath, GlobalResponseNorm, _CONV, make_attention
from .init_contract import declare_no_reinit


class LayerNorm3d(nn.Module):
    """Channel-first LayerNorm (2D/3D); stats over C only. Name kept for API stability."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias   = nn.Parameter(torch.zeros(num_channels))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, *spatial)；统计量 fp32 累加（AMP 下 fp16/bf16 求均值/方差
        # 精度不足），归一后转回输入 dtype（同 adm_unet fp32 范式）。
        dtype = x.dtype
        xf = x.float()
        u = xf.mean(dim=1, keepdim=True)
        s = (xf - u).pow(2).mean(dim=1, keepdim=True)
        x = ((xf - u) / torch.sqrt(s + self.eps)).type(dtype)
        # 动态阐广播形：(C,) → (1, C, 1, ..., 1)。
        pat = 'c -> 1 c' + ' 1' * (x.ndim - 2)
        return x * rearrange(self.weight, pat) + rearrange(self.bias, pat)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block: dwconv7 → LN → pw(4x) → GELU → pw → attn? → LayerScale → residual."""

    def __init__(
        self,
        dim                   : int,
        expand_ratio          : float = 4.0,
        drop_path             : float = 0.0,
        attention_type        : str = "none",
        use_grn               : bool = False,
        spatial_dims          : int = 3,
        layer_scale_init_value: float = 1e-6,
        attn_reduction        : int = 16):
        super().__init__()
        d = spatial_dims
        self.spatial_dims = d
        hidden = int(dim * expand_ratio)

        self.dwconv  = _CONV[d](dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        self.norm    = LayerNorm3d(dim)
        self.pwconv1 = _CONV[d](dim, hidden, kernel_size=1, bias=True)
        self.act     = nn.GELU()
        self.grn     = GlobalResponseNorm(hidden, spatial_dims=d) if use_grn else nn.Identity()
        self.pwconv2 = _CONV[d](hidden, dim, kernel_size=1, bias=True)
        # reduction 跟随 config（model.se_reduction，与 ResNet 系一致）；coord 内部
        # 归一化保持其默认 group/8（ConvNeXt 块内 norm 固定为 LN，不跟随全局 norm_type）。
        self.attn    = make_attention(attention_type, dim, spatial_dims=d,
                                      reduction=attn_reduction)
        # LayerScale: init small → near-identity start; <=0 disables
        if layer_scale_init_value > 0.0:
            self.gamma = nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True)
            # LayerScale 小值初始化契约：全局 init_strategy 不得覆盖（3-2）。
            declare_no_reinit(self.gamma)
        else:
            self.register_parameter("gamma", None)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.dwconv(x)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.grn(out)
        out = self.pwconv2(out)
        out = self.attn(out)
        if self.gamma is not None:
            pat = 'c -> 1 c' + ' 1' * self.spatial_dims
            out = out * rearrange(self.gamma, pat)
        return res + self.drop_path(out)


class ConvNeXtAdaptBlock(nn.Module):
    """ConvNeXt block with optional 1x1 channel projection (in_ch != out_ch)."""

    def __init__(
        self,
        in_ch                 : int,
        out_ch                : int,
        expand_ratio          : float = 4.0,
        drop_path             : float = 0.0,
        attention_type        : str = "none",
        use_grn               : bool = False,
        spatial_dims          : int = 3,
        layer_scale_init_value: float = 1e-6,
        attn_reduction        : int = 16):
        super().__init__()
        d = spatial_dims
        self.proj = (
            nn.Sequential(
                _CONV[d](in_ch, out_ch, 1, bias=False),
                LayerNorm3d(out_ch))
            if in_ch != out_ch else nn.Identity())
        self.block = ConvNeXtBlock(
            out_ch, expand_ratio, drop_path,
            attention_type=attention_type,
            use_grn=use_grn,
            spatial_dims=d,
            layer_scale_init_value=layer_scale_init_value,
            attn_reduction=attn_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.proj(x))


class ConvNeXtStage(nn.Module):
    """N ConvNeXt blocks at one resolution (first block may change channels)."""

    def __init__(
        self,
        in_ch                 : int,
        out_ch                : int,
        num_blocks            : int = 2,
        expand_ratio          : float = 4.0,
        drop_path_rates       : list = None,
        attention_type        : str = "none",
        use_grn               : bool = False,
        spatial_dims          : int = 3,
        layer_scale_init_value: float = 1e-6,
        attn_reduction        : int = 16):
        super().__init__()
        d = spatial_dims
        if not drop_path_rates:
            # None 与空列表同判：空列表下 [0] 取值会 IndexError。
            drop_path_rates = [0.0] * num_blocks
        blocks = [ConvNeXtAdaptBlock(
            in_ch, out_ch, expand_ratio,
            drop_path_rates[0], attention_type,
            use_grn=use_grn,
            spatial_dims=d,
            layer_scale_init_value=layer_scale_init_value,
            attn_reduction=attn_reduction)]
        for i in range(1, num_blocks):
            dp = drop_path_rates[i] if i < len(drop_path_rates) else 0.0
            blocks.append(ConvNeXtAdaptBlock(
                out_ch, out_ch, expand_ratio,
                dp, attention_type,
                use_grn=use_grn,
                spatial_dims=d,
                attn_reduction=attn_reduction,
                layer_scale_init_value=layer_scale_init_value))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ConvNeXtDownsample(nn.Module):
    """Paper-faithful ConvNeXt inter-stage downsample: LN → Conv(k=2,s=2). LN-first."""

    def __init__(self, in_ch: int, out_ch: int, spatial_dims: int = 3):
        super().__init__()
        d = spatial_dims
        self.norm = LayerNorm3d(in_ch)
        self.conv = _CONV[d](in_ch, out_ch, kernel_size=2, stride=2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x))
