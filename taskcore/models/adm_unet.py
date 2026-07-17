"""ADM U-Net 共享 backbone (Dhariwal & Nichol 2021)。与论文 enc/mid/dec 块一致 (GN32+SiLU, ResBlock 无 emb, 多头 QKVAttention, stride-2 Downsample, nearest+conv Upsample, 逐块 skip)。仅 2.5D；DS/aux/lucidrains LinearAttention 为附加扩展。forward 返回与 UNet3D 一致。"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .blocks import ConvNormAct
from .stem import build_context_stem, build_stem
from .topology import build_topology
from .unet import SegmentationHead, build_head

logger = logging.getLogger(__name__)


# ============================================================================
# Paper-faithful primitives (no-emb variants).
# ============================================================================


class _GroupNorm32(nn.GroupNorm):
    """ADM GroupNorm32：fp32 forward 后转回原 dtype。"""

    def forward(self, x):  # type: ignore[override]
        return super().forward(x.float()).type(x.dtype)


def _norm(channels: int) -> nn.Module:
    """ADM normalization：默认 32 groups；不整除时退到 gcd(channels,32)。"""
    g = 32 if channels % 32 == 0 else math.gcd(channels, 32) or 1
    return _GroupNorm32(g, channels)


def _conv2d(*args, **kwargs) -> nn.Conv2d:
    return nn.Conv2d(*args, **kwargs)


def _zero_(module: nn.Module) -> nn.Module:
    """ADM zero_module：将所有参数 in-place 零初始化。"""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class _Upsample(nn.Module):
    """ADM Upsample：nearest 2× + 可选 conv。"""

    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = _conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        # 旧 PyTorch (<2.1) 上 upsample_nearest2d 缺 bf16/fp16 kernel；cast→interp→cast 回。
        if x.dtype in (torch.bfloat16, torch.float16):
            orig_dtype = x.dtype
            x = F.interpolate(x.float(), scale_factor=2.0,
                              mode="nearest").to(orig_dtype)
        else:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class _Downsample(nn.Module):
    """ADM Downsample：stride-2 conv（use_conv=False 时为 avg pool）。"""

    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.op = _conv2d(channels, channels, 3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.op(x)


class _ResBlockBase(nn.Module):
    """ADM ResBlock 基类。子类 ``_ResBlockNoEmb``（分割，无 emb）/ ``_ResBlockEmb``
    （扩散，AdaGN time-embed）共享。``forward(x, emb=None)`` 统一签名，使
    encoder/middle/decoder 的逐块循环对两种块一视同仁地传 ``emb``。"""


class _ResBlockNoEmb(_ResBlockBase):
    """ADM ResBlock（去 time-embed）。
    in: norm→silu→conv3；out: norm→silu→dropout→conv3(zero-init)；skip: Identity / conv1 / conv3。。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        use_conv_skip: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_layers = nn.Sequential(
            _norm(in_channels),
            nn.SiLU(),
            _conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.out_layers = nn.Sequential(
            _norm(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            _zero_(_conv2d(out_channels, out_channels, 3, padding=1)),
        )
        if in_channels == out_channels:
            self.skip_connection = nn.Identity()
        elif use_conv_skip:
            self.skip_connection = _conv2d(in_channels, out_channels, 3, padding=1)
        else:
            self.skip_connection = _conv2d(in_channels, out_channels, 1)

    def forward(self, x, emb=None):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class _ResBlockEmb(_ResBlockBase):
    """ADM ResBlock（论文忠实 time-embed，AdaGN scale-shift）。

    与 ``_ResBlockNoEmb`` 结构一致，仅在 out 分支的 GroupNorm 之后注入
    ``h = norm(h) * (1 + scale) + shift``，其中 ``(scale, shift)`` 由
    ``SiLU + Linear(emb)`` 给出（Dhariwal & Nichol 2021 的 ``use_scale_shift_norm``）。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        dropout: float = 0.0,
        use_conv_skip: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_layers = nn.Sequential(
            _norm(in_channels),
            nn.SiLU(),
            _conv2d(in_channels, out_channels, 3, padding=1),
        )
        # emb → 2*out_channels（scale, shift），AdaGN。
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * out_channels),
        )
        self.out_norm = _norm(out_channels)
        self.out_rest = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            _zero_(_conv2d(out_channels, out_channels, 3, padding=1)),
        )
        if in_channels == out_channels:
            self.skip_connection = nn.Identity()
        elif use_conv_skip:
            self.skip_connection = _conv2d(in_channels, out_channels, 3, padding=1)
        else:
            self.skip_connection = _conv2d(in_channels, out_channels, 1)

    def forward(self, x, emb=None):
        if emb is None:
            raise ValueError("_ResBlockEmb.forward requires emb.")
        h = self.in_layers(x)
        scale_shift = self.emb_layers(emb).type(h.dtype)
        scale, shift = scale_shift[:, :, None, None].chunk(2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_rest(h)
        return self.skip_connection(x) + h


def _make_resblock(
    in_channels: int, out_channels: int, dropout: float, emb_channels: int):
    """按 ``emb_channels`` 选择 ResBlock：0→分割无 emb；>0→扩散 AdaGN。"""
    if emb_channels and emb_channels > 0:
        return _ResBlockEmb(in_channels, out_channels, emb_channels, dropout=dropout)
    return _ResBlockNoEmb(in_channels, out_channels, dropout=dropout)


class _QKVAttentionLegacy(nn.Module):
    """ADM QKVAttentionLegacy：先拆 head 再拆 qkv。"""

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        # (B, 3*h*ch, L) → (B*h, 3*ch, L) 后拆 q/k/v。
        qkv_h = rearrange(
            qkv, 'b (h c) l -> (b h) c l', h=self.n_heads)
        q, k, v = qkv_h.split(ch, dim=1)
        scale = 1.0 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return rearrange(a, '(b h) c l -> b (h c) l', h=self.n_heads)


class _LinearAttention(nn.Module):
    """O(N) 线性注意（Shen 2021，lucidrains 实现）：KᵀV 技巧，复杂度 O(D²N) 代替 O(DN²)。
    1×1 QKV(no bias)、Q/K 在 feat/spatial 上 softmax、Q 乘 head_dim**-0.5、出口 Conv1×1+GN(1)。。"""

    def __init__(self, channels: int, num_heads: int = 4, head_dim: int = 32):
        super().__init__()
        self.scale = float(head_dim) ** -0.5
        self.num_heads = num_heads
        hidden = num_heads * head_dim
        self.to_qkv = nn.Conv2d(channels, hidden * 3, 1, bias=False)
        # GroupNorm(1, channels) ≡ 逐样本通道 LN；non zero-init。
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden, channels, 1),
            nn.GroupNorm(1, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = (
            rearrange(t, 'b (nh d) hh ww -> b nh d (hh ww)', nh=self.num_heads)
            for t in qkv
        )  # 各 [B, H, head_dim, N]
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        # context: [B, H, head_dim_v, head_dim_k]。
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(out, 'b nh d (hh ww) -> b (nh d) hh ww', hh=h, ww=w)
        return self.to_out(out)


class _LinearAttentionBlock(nn.Module):
    """Residual(PreNorm(LinearAttention))，预 norm 为 GroupNorm(1)。与 softmax attn 叠加，逐级末位置一次。。"""

    def __init__(self, channels: int, num_heads: int = 4, head_dim: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.attn = _LinearAttention(channels, num_heads=num_heads,
                                      head_dim=head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(self.norm(x)) + x


class _AttentionBlock(nn.Module):
    """ADM AttentionBlock：逐位置多头自注意（拍平空间轴）。"""

    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_head_channels: int = -1,
    ):
        super().__init__()
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, (
                f"channels {channels} not divisible by num_head_channels "
                f"{num_head_channels}")
            self.num_heads = channels // num_head_channels
        self.norm = _norm(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = _QKVAttentionLegacy(self.num_heads)
        self.proj_out = _zero_(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        x_flat = rearrange(x, 'b c ... -> b c (...)')
        qkv = self.qkv(self.norm(x_flat))
        h = self.attention(qkv)
        h = self.proj_out(h)
        # (B, C, prod(spatial)) → 原 shape；unflatten 非 reshape API。
        return (x_flat + h).unflatten(-1, x.shape[2:])


# ============================================================================
# ADM shared U-Net backbone wrapper.
# ============================================================================


def _resolve_attention_levels(
    n_levels: int, attn_levels: Optional[Sequence[int]]
) -> List[int]:
    """解析注意级别索引；默认最深两级（与 ADM 默认 attention_resolutions=[16,8] 一致）"""
    if attn_levels is None:
        if n_levels >= 2:
            return [n_levels - 2, n_levels - 1]  # 最后两层
        return [n_levels - 1]
    out = sorted({int(v) for v in attn_levels})
    for v in out:
        if v < 0 or v >= n_levels:
            raise ValueError(
                f"adm.attn_levels entry {v} out of range [0, {n_levels - 1}]")
    return out


class _ADMEncoder(nn.Module):
    """ADM encoder: stem + L 级 (nb× ResBlock + opt Attn) + Downsample。
    返 {enc_features, enc_skips}：enc_skips 为完整 ADM 堆（每 ResBlock+每 Downsample+stem）。。"""

    def __init__(
        self,
        stem: nn.Module,
        encoder_channels: List[int],
        encoder_blocks_per_stage: List[int],
        attention_levels: List[int],
        num_heads: int,
        num_head_channels: int,
        dropout: float,
        linear_attention_levels: Optional[List[int]] = None,
        linear_attention_num_heads: int = 4,
        linear_attention_head_dim: int = 32,
        emb_channels: int = 0,
        cond_stem: Optional[nn.Module] = None,
        cond_fuse: Optional[nn.Module] = None,
        cond_in_channels: int = 0,
    ):
        super().__init__()
        self.stem = stem
        self.cond_stem = cond_stem
        self.cond_fuse = cond_fuse
        self.cond_in_channels = int(cond_in_channels)
        self.emb_channels = int(emb_channels)
        self.encoder_channels = list(encoder_channels)
        n_levels = len(encoder_channels)
        self.n_levels = n_levels
        self.attention_levels = list(attention_levels)
        self.linear_attention_levels = list(linear_attention_levels or [])
        self.encoder_blocks_per_stage = list(encoder_blocks_per_stage)

        self.levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        # 记录 skip 通道布局供 decoder 算 cat-fusion in_ch。
        self.skip_channels: List[int] = [encoder_channels[0]]  # stem skip

        for level, ch_out in enumerate(encoder_channels):
            ch_in = encoder_channels[0] if level == 0 else encoder_channels[level - 1]
            blocks_this_level: List[nn.Module] = []
            n_blocks = encoder_blocks_per_stage[level]
            for i in range(n_blocks):
                blocks_this_level.append(
                    _make_resblock(
                        ch_in if i == 0 else ch_out,
                        ch_out,
                        dropout,
                        self.emb_channels))
                if level in self.attention_levels:
                    blocks_this_level.append(
                        _AttentionBlock(
                            ch_out,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels))
                self.skip_channels.append(ch_out)
            # 逐级末一个 LinearAttention（在 ResBlocks + softmax-attn 后），重写最后一个 skip。
            if level in self.linear_attention_levels:
                blocks_this_level.append(
                    _LinearAttentionBlock(
                        ch_out,
                        num_heads=linear_attention_num_heads,
                        head_dim=linear_attention_head_dim))
            self.levels.append(nn.ModuleList(blocks_this_level))
            if level < n_levels - 1:
                self.downsamples.append(_Downsample(ch_out, use_conv=True))
                self.skip_channels.append(ch_out)
            else:
                self.downsamples.append(None)  # type: ignore[arg-type]

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """返 {enc_features, enc_skips}。"""
        cond = None
        if self.cond_in_channels > 0:
            x, cond = torch.split(x, [x.shape[1] - self.cond_in_channels, self.cond_in_channels], dim=1)
        x = self.stem(x)
        if cond is not None:
            cond = self.cond_stem(cond)
            x = self.cond_fuse(torch.cat([x, cond], dim=1))
        enc_skips: List[torch.Tensor] = [x]
        enc_features: List[torch.Tensor] = []
        for level, blocks in enumerate(self.levels):
            for blk in blocks:
                if isinstance(blk, _ResBlockBase):
                    x = blk(x, emb)
                    enc_skips.append(x)
                else:  # AttentionBlock
                    x = blk(x)
                    # ADM 顺序：attended 特征替换上一个 skip，而非追加。
                    enc_skips[-1] = x
            enc_features.append(x)
            if level < self.n_levels - 1:
                x = self.downsamples[level](x)
                enc_skips.append(x)
        return {
            "bottleneck"  : x,
            "enc_features": enc_features,
            "enc_skips"   : enc_skips}


class _ADMMiddle(nn.Module):
    """ADM middle：ResBlock → AttentionBlock → ResBlock。"""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        num_head_channels: int,
        dropout: float,
        emb_channels: int = 0,
    ):
        super().__init__()
        self.r1 = _make_resblock(channels, channels, dropout, emb_channels)
        self.a = _AttentionBlock(
            channels, num_heads=num_heads, num_head_channels=num_head_channels)
        self.r2 = _make_resblock(channels, channels, dropout, emb_channels)

    def forward(self, x, emb: Optional[torch.Tensor] = None):
        return self.r2(self.a(self.r1(x, emb)), emb)


class _ADMDecoder(nn.Module):
    """ADM decoder：逐级 (nb+1) × (cat skip → ResBlock + opt Attn) + Upsample。
    dec_features[i] = level (L-2-i) post-blocks/pre-upsample，长 L-1，low→high。。"""

    def __init__(
        self,
        encoder_channels: List[int],
        skip_channels: List[int],
        decoder_blocks_per_stage: List[int],
        attention_levels: List[int],
        num_heads: int,
        num_head_channels: int,
        dropout: float,
        linear_attention_levels: Optional[List[int]] = None,
        linear_attention_num_heads: int = 4,
        linear_attention_head_dim: int = 32,
        emb_channels: int = 0,
    ):
        super().__init__()
        self.emb_channels = int(emb_channels)
        self.encoder_channels = list(encoder_channels)
        n_levels = len(encoder_channels)
        self.n_levels = n_levels
        # 每级 (decoder_blocks_per_stage[level]+1) 个 ResBlock；长度同 encoder（最深多出一项作 bottleneck refinement）。
        if len(decoder_blocks_per_stage) != n_levels:
            raise ValueError(
                f"decoder_blocks_per_stage length {len(decoder_blocks_per_stage)} "
                f"!= expected {n_levels}")
        self.decoder_blocks_per_stage = list(decoder_blocks_per_stage)

        # 深→浅构造；反序 pop skip_channels。
        skip_stack = list(skip_channels)
        self.levels = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        # Start ch (entering level L-1) = bottleneck = encoder_channels[-1].
        ch = encoder_channels[-1]
        for ridx, level in enumerate(reversed(range(n_levels))):
            n_blocks = self.decoder_blocks_per_stage[level] + 1
            blocks_this_level: List[nn.Module] = []
            for i in range(n_blocks):
                ich = skip_stack.pop()
                blocks_this_level.append(
                    _make_resblock(
                        ch + ich,
                        encoder_channels[level],
                        dropout,
                        self.emb_channels,
                    )
                )
                ch = encoder_channels[level]
                if level in attention_levels:
                    blocks_this_level.append(
                        _AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                        )
                    )
            # 逐级末的 LinearAttention：不动 skip 栈（pop 数 = ResBlock 数）。
            if linear_attention_levels and level in linear_attention_levels:
                blocks_this_level.append(
                    _LinearAttentionBlock(
                        ch,
                        num_heads=linear_attention_num_heads,
                        head_dim=linear_attention_head_dim,
                    )
                )
            self.levels.append(nn.ModuleList(blocks_this_level))
            if level > 0:
                # Upsample 保持通道数（ADM 约定）。
                self.upsamples.append(_Upsample(ch, use_conv=True))
            else:
                self.upsamples.append(None)  # type: ignore[arg-type]
        if skip_stack:
            raise RuntimeError(
                f"ADM decoder skip-stack mismatch: {len(skip_stack)} "
                f"skip(s) left after build (expected 0).")

        # 逐级 out_channels (low→high)供 DS/aux 头，与 UNet3D 合同。
        self.out_channels: List[int] = [
            encoder_channels[level] for level in reversed(range(n_levels - 1))
        ]

    def forward(
        self, bottleneck: torch.Tensor, enc_skips: List[torch.Tensor],
        emb: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """返 dec_features，长 n_levels-1，low→high。"""
        x = bottleneck
        skip_stack = list(enc_skips)
        # bottleneck 是最后 push 的 post-Downsample 特征；ADM 在 level L-1 首块首个 pop。
        dec_features: List[torch.Tensor] = []
        for ridx, level in enumerate(reversed(range(self.n_levels))):
            for blk in self.levels[ridx]:
                if isinstance(blk, _ResBlockBase):
                    skip = skip_stack.pop()
                    if skip.shape[2:] != x.shape[2:]:
                        # 防御：输入空间与 encoder stride 对齐时不会触发。
                        skip = F.interpolate(
                            skip, size=x.shape[2:],
                            mode="bilinear", align_corners=False)
                    x = blk(torch.cat([x, skip], dim=1), emb)
                else:  # AttentionBlock
                    x = blk(x)
            # 除最深级外均抓取，以合 UNet3D 约定。
            if level < self.n_levels - 1:
                dec_features.append(x)
            if self.upsamples[ridx] is not None:
                x = self.upsamples[ridx](x)
        if skip_stack:
            raise RuntimeError(
                f"ADM decoder leftover skips at runtime: {len(skip_stack)}")
        return dec_features


class ADMSegModel(nn.Module):
    """ADM U-Net 适配为分割模型。forward 返回合同与 UNet3D 一致（见模块顶部 docstring）。"""

    def __init__(
        self,
        # Stem / multi-FOV.
        stem: nn.Module,
        stem_stride: int,
        num_stem_fusion_views: int,
        stem_fusion_mode: str,
        # Encoder topology.
        encoder_channels: List[int],
        encoder_blocks_per_stage: List[int],
        decoder_blocks_per_stage: List[int],
        attention_levels: List[int],
        num_heads: int,
        num_head_channels: int,
        dropout: float,
        # Output / heads.
        num_fg_classes: int,
        deep_supervision: bool,
        aux_seg_supervision: bool,
        aux_head_mode: str,
        # Optional lucidrains-style linear attention (composes with softmax attn).
        linear_attention_levels: Optional[List[int]] = None,
        linear_attention_num_heads: int = 4,
        linear_attention_head_dim: int = 32,
        aux_head_out_channels: Optional[List[int]] = None,
        cond_stem: Optional[nn.Module] = None,
        cond_fuse: Optional[nn.Module] = None,
        cond_in_channels: int = 0,
    ):
        super().__init__()
        self.spatial_dims = 2
        self.stem_stride = int(stem_stride)
        self.num_stem_fusion_views = int(num_stem_fusion_views)
        self.stem_fusion_mode = stem_fusion_mode
        self.deep_supervision = bool(deep_supervision)
        self.num_fg_classes = int(num_fg_classes)

        self.encoder = _ADMEncoder(
            stem=stem,
            cond_stem=cond_stem,
            cond_fuse=cond_fuse,
            cond_in_channels=cond_in_channels,
            encoder_channels=encoder_channels,
            encoder_blocks_per_stage=encoder_blocks_per_stage,
            attention_levels=attention_levels,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            dropout=dropout,
            linear_attention_levels=linear_attention_levels,
            linear_attention_num_heads=linear_attention_num_heads,
            linear_attention_head_dim=linear_attention_head_dim,
        )
        self.middle = _ADMMiddle(
            channels=encoder_channels[-1],
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            dropout=dropout,
        )
        self.decoder = _ADMDecoder(
            encoder_channels=encoder_channels,
            skip_channels=self.encoder.skip_channels,
            decoder_blocks_per_stage=decoder_blocks_per_stage,
            attention_levels=attention_levels,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            dropout=dropout,
            linear_attention_levels=linear_attention_levels,
            linear_attention_num_heads=linear_attention_num_heads,
            linear_attention_head_dim=linear_attention_head_dim,
        )

        # 1×1 logit conv（论文为 Conv3+GN+SiLU），与 UNet3D / DS 约定一致。
        self.seg_head = SegmentationHead(
            self.decoder.out_channels[-1],
            num_fg_classes,
            spatial_dims=2,
        )

        # DS 头位于低分辨率 decoder 特征（与 UNet3D 一致）。
        self.ds_heads = nn.ModuleList()
        if self.deep_supervision:
            for ch in reversed(self.decoder.out_channels[:-1]):
                self.ds_heads.append(
                    SegmentationHead(ch, num_fg_classes, spatial_dims=2)
                )

        # Aux 分割监督（仅 Plan A：所有 aux 头以最高分辨率 dec feat）。
        n_views = self.num_stem_fusion_views
        self.aux_seg_supervision = bool(aux_seg_supervision and n_views > 1)
        n_aux_expected = max(n_views - 1, 0) if self.aux_seg_supervision else 0
        if aux_head_out_channels is None:
            aux_out = [num_fg_classes] * n_aux_expected
        else:
            if len(aux_head_out_channels) != n_aux_expected:
                raise ValueError(
                    f"aux_head_out_channels length "
                    f"{len(aux_head_out_channels)} != expected "
                    f"{n_aux_expected}")
            aux_out = [int(c) for c in aux_head_out_channels]
        self.aux_head_out_channels = aux_out
        self.aux_head_mode = aux_head_mode
        self.aux_heads = nn.ModuleList()
        self.aux_feat_indices: List[int] = []
        if self.aux_seg_supervision:
            in_ch = self.decoder.out_channels[-1]
            n_dec = len(self.decoder.out_channels)
            for k in range(1, n_views):
                self.aux_feat_indices.append(n_dec - 1)
                self.aux_heads.append(
                    build_head(
                        mode=aux_head_mode,
                        in_ch=in_ch,
                        num_classes=aux_out[k - 1],
                        spatial_dims=2,
                        # GN+SiLU 与 ADM 一致（仅 mode='conv' 时生效）。
                        norm_type="group",
                        norm_groups=32 if in_ch % 32 == 0 else 8,
                        activation="swish",  # SiLU
                    )
                )

    # ----- Forward --------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, Any]]:
        target_size = x.shape[2:]
        enc_out = self.encoder(x)
        bottleneck = self.middle(enc_out["bottleneck"])
        dec_features = self.decoder(bottleneck, enc_out["enc_skips"])

        # 主头（总位于 decoder 最高分辨率特征）。
        main_out = self.seg_head(dec_features[-1])
        if main_out.shape[2:] != target_size:
            main_out = F.interpolate(
                main_out, size=target_size,
                mode="bilinear", align_corners=False)

        # Aux 头（仅训练，与 UNet3D 合同）。
        aux_outs: List[torch.Tensor] = []
        if self.aux_seg_supervision and self.training:
            for head, feat_idx in zip(self.aux_heads, self.aux_feat_indices):
                ao = head(dec_features[feat_idx])
                if ao.shape[2:] != target_size:
                    ao = F.interpolate(
                        ao, size=target_size,
                        mode="bilinear", align_corners=False)
                aux_outs.append(ao)

        # 主路 + 可选深监督。
        if self.deep_supervision and self.training:
            main_path: Union[torch.Tensor, List[torch.Tensor]] = [main_out]
            for i, head in enumerate(self.ds_heads):
                main_path.append(head(dec_features[-2 - i]))
        else:
            main_path = main_out

        if aux_outs:
            return {"main": main_path, "aux": aux_outs}
        return main_path

    def param_count(self) -> Dict[str, int]:
        enc = sum(p.numel() for p in self.encoder.parameters())
        mid = sum(p.numel() for p in self.middle.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        head = sum(p.numel() for p in self.seg_head.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {
            "encoder": enc, "middle": mid, "decoder": dec,
            "seg_head": head, "total": total,
        }


# ============================================================================
# Public factory.
# ============================================================================


def build_adm_seg_model(cfg) -> ADMSegModel:
    """从 Config 构造 ADMSegModel。读标准 data/model 字段 + ADM 专有：
    adm_attention_levels (默认 [L-2, L-1])、adm_num_heads(4)、adm_num_head_channels(-1)。
    model.dropout 复用为 ADM ResBlock dropout。。"""
    mc = cfg.model
    enc_channels = list(mc.encoder_channels)
    n_levels = len(enc_channels)
    num_fg = cfg.num_fg_classes
    topo = build_topology(cfg)

    # 仅 2.5D（其他模式需额外布线）。显式 raise（assert 在 python -O 下失效）。
    if cfg.data.patch_mode != "2_5d":
        raise ValueError(
            "arch='adm' is currently only wired for patch_mode='2_5d'; "
            f"got {cfg.data.patch_mode!r}.")
    D = int(cfg.data.patch_size[0])
    out_classes = num_fg * D
    n_views = max(len(cfg.data.multi_res_scales), 1)

    # 逐级块计数。
    enc_bps = list(mc.encoder_blocks_per_stage)
    if not enc_bps:
        enc_bps = [int(mc.blocks_per_level)] * n_levels
    if len(enc_bps) != n_levels:
        raise ValueError(
            f"encoder_blocks_per_stage length {len(enc_bps)} "
            f"!= len(encoder_channels) {n_levels}")
    # skip-stack 平衡要求 dec_bps_full[k]==enc_bps[k]；用户 decoder_blocks_per_stage 忽略，不一致告警。
    if mc.decoder_blocks_per_stage and (
            list(mc.decoder_blocks_per_stage) != enc_bps[:-1]):
        logger.warning(
            "model.arch='adm' ignores model.decoder_blocks_per_stage=%s; "
            "ADM's per-level decoder count is fixed by the encoder's "
            "skip-stack topology (nb+1 ResBlocks per level, paper-faithful). "
            "Using encoder_blocks_per_stage=%s instead.",
            list(mc.decoder_blocks_per_stage), enc_bps)
    dec_bps_full = list(enc_bps)

    # 显式空列表 [] = 不加注意力（保持历史默认行为）；传 None 才会用"最深两级"默认。
    attn_levels = _resolve_attention_levels(
        n_levels, mc.adm_attention_levels)

    # 可选 lucidrains 式线性注意（默认关）。
    raw_lin = mc.adm_linear_attention_levels or []
    lin_attn_levels: List[int] = sorted({int(v) for v in raw_lin})
    for v in lin_attn_levels:
        if v < 0 or v >= n_levels:
            raise ValueError(
                f"adm_linear_attention_levels entry {v} out of range "
                f"[0, {n_levels - 1}]")
    lin_num_heads = int(mc.adm_linear_attention_num_heads)
    lin_head_dim = int(mc.adm_linear_attention_head_dim)

    # 仅支持 shared_stem / multi_stem_proj；hierarchical 需 mid-encoder 注入。
    if mc.stem_fusion_mode == "hierarchical":
        raise ValueError(
            "model.arch='adm' does not yet support stem_fusion_mode="
            "'hierarchical'. Use 'shared_stem' or 'multi_stem_proj' "
            "for ADM. (Hierarchical fusion will be added in a follow-up.)")

    # 原生深度 ON：逐视图 D_k 可变；OFF：统一 D。
    in_ch_per_view_list = None
    aux_head_out_channels = None
    if bool(cfg.data.keep_native_view_depth) and n_views > 1:
        depths = list(cfg.per_view_depths)
        in_ch_per_view_list = depths
        aux_head_out_channels = [num_fg * d_k for d_k in depths[1:]]
        in_channels = sum(depths)
    else:
        in_channels = D * n_views
    base_ch_per_view = D  # 统一深度（无列表时的默认值）
    cond_in_channels = int(topo.cond_in_channels)

    stem, stem_stride = build_context_stem(
        mode=mc.stem_mode,
        fusion=mc.stem_fusion_mode,
        n_views=n_views,
        base_ch_per_view=base_ch_per_view,
        out_ch=enc_channels[0],
        # ADM 风格 stem：GroupNorm + SiLU。
        norm_type="group",
        norm_groups=32 if enc_channels[0] % 32 == 0 else 8,
        activation="swish",  # SiLU
        spatial_dims=2,
        stage_channels=enc_channels,
        in_ch_per_view_list=in_ch_per_view_list)
    cond_stem = None
    cond_fuse = None
    if cond_in_channels > 0:
        cond_stem, cond_stride = build_stem(
            mode=mc.stem_mode,
            in_ch=cond_in_channels,
            out_ch=enc_channels[0],
            norm_type="group",
            norm_groups=32 if enc_channels[0] % 32 == 0 else 8,
            activation="swish",
            spatial_dims=2)
        if cond_stride != stem_stride:
            raise RuntimeError(
                f"cond stem stride {cond_stride} != main stem stride {stem_stride}.")
        cond_fuse = ConvNormAct(
            enc_channels[0] * 2, enc_channels[0],
            kernel_size=1, stride=1, padding=0,
            norm_type="group",
            norm_groups=32 if enc_channels[0] % 32 == 0 else 8,
            activation="swish",
            spatial_dims=2)

    aux_seg = bool(mc.aux_seg_supervision) and n_views > 1

    model = ADMSegModel(
        stem=stem,
        cond_stem=cond_stem,
        cond_fuse=cond_fuse,
        cond_in_channels=cond_in_channels,
        stem_stride=stem_stride,
        num_stem_fusion_views=n_views,
        stem_fusion_mode=mc.stem_fusion_mode,
        encoder_channels=enc_channels,
        encoder_blocks_per_stage=enc_bps,
        decoder_blocks_per_stage=dec_bps_full,
        attention_levels=attn_levels,
        num_heads=int(mc.adm_num_heads),
        num_head_channels=int(mc.adm_num_head_channels),
        dropout=float(mc.dropout),
        linear_attention_levels=lin_attn_levels,
        linear_attention_num_heads=lin_num_heads,
        linear_attention_head_dim=lin_head_dim,
        num_fg_classes=out_classes,
        deep_supervision=bool(mc.deep_supervision),
        aux_seg_supervision=aux_seg,
        aux_head_mode=str(mc.aux_head_mode),
        aux_head_out_channels=aux_head_out_channels,
    )

    pc = model.param_count()
    logger.info(
        "Built ADMSegModel: enc=%.2fM, mid=%.2fM, dec=%.2fM, total=%.2fM, "
        "channels=%s, enc_blocks=%s, dec_blocks=%s, attn_levels=%s, "
        "lin_attn_levels=%s (heads=%d, head_dim=%d), "
        "in_ch=%d (per_view=%s), out_classes=%d (fg=%d, D=%d), "
        "stem=%s(stride=%d, n_views=%d, fusion=%s), ds=%s, aux_seg=%s "
        "(n_aux=%d, mode=%s)",
        pc["encoder"] / 1e6, pc["middle"] / 1e6, pc["decoder"] / 1e6,
        pc["total"] / 1e6,
        enc_channels, enc_bps, dec_bps_full, attn_levels,
        lin_attn_levels, lin_num_heads, lin_head_dim,
        in_channels,
        in_ch_per_view_list if in_ch_per_view_list is not None
        else [base_ch_per_view] * n_views,
        out_classes, num_fg, D,
        mc.stem_mode, stem_stride, n_views, mc.stem_fusion_mode,
        bool(mc.deep_supervision), aux_seg,
        len(model.aux_heads), mc.aux_head_mode)

    return model


# 旧名兼容：生成侧调用方使用 build_adm_backbone（同一构建函数）。
build_adm_backbone = build_adm_seg_model


# ============================================================================
# Diffusion backbone（论文忠实：重新启用 timestep / σ 条件，AdaGN scale-shift）
# ============================================================================


def _sinusoidal_embedding(values: torch.Tensor, dim: int, max_period: float = 10000.0):
    """标准正弦时间步嵌入（Transformer / ADM）。``values`` 形如 ``(B,)``。"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=values.device, dtype=torch.float32) / half)
    args = values.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ADMDiffusionUNet(nn.Module):
    """ADM U-Net 的条件扩散 backbone（2.5D）。

    复用 ``_ADMEncoder / _ADMMiddle / _ADMDecoder``，但以 ``emb_channels>0`` 启用
    每个 ResBlock 的 AdaGN scale-shift（Dhariwal & Nichol 2021）。输入为
    ``cat([x_noisy, cond], dim=1)``（cond=低分条件图），输出 ``out_channels`` 通道的
    原始网络预测 ``F_θ``（EDM/DDPM 预条件由 ``DiffusionTrainWrapper`` 在外层施加）。

    forward(x_cat, c_noise)：``c_noise`` 形如 ``(B,)``（EDM 取 ``0.25*ln σ``，DDPM 取
    时间步索引），内部正弦嵌入 + MLP 得到 ``emb`` 并贯穿编/中/解码器。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        encoder_channels: List[int],
        encoder_blocks_per_stage: List[int],
        attention_levels: List[int],
        num_heads: int = 4,
        num_head_channels: int = -1,
        dropout: float = 0.0):
        super().__init__()
        self.spatial_dims = 2
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        model_ch = int(encoder_channels[0])
        self.model_ch = model_ch
        emb_ch = model_ch * 4
        self.emb_ch = emb_ch

        # 时间步 MLP（正弦嵌入 → 2 层 Linear）。
        self.time_mlp = nn.Sequential(
            nn.Linear(model_ch, emb_ch),
            nn.SiLU(),
            nn.Linear(emb_ch, emb_ch),
        )
        # 输入卷积充当 stem（ADM 原始即单 Conv）。
        input_conv = _conv2d(self.in_channels, model_ch, 3, padding=1)

        self.encoder = _ADMEncoder(
            stem=input_conv,
            encoder_channels=encoder_channels,
            encoder_blocks_per_stage=encoder_blocks_per_stage,
            attention_levels=attention_levels,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            dropout=dropout,
            emb_channels=emb_ch)
        self.middle = _ADMMiddle(
            channels=encoder_channels[-1],
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            dropout=dropout,
            emb_channels=emb_ch)
        self.decoder = _ADMDecoder(
            encoder_channels=encoder_channels,
            skip_channels=self.encoder.skip_channels,
            decoder_blocks_per_stage=list(encoder_blocks_per_stage),
            attention_levels=attention_levels,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            dropout=dropout,
            emb_channels=emb_ch)
        # 输出：GroupNorm + SiLU + zero-init conv（ADM 风格）。
        self.out_norm = _norm(self.decoder.out_channels[-1])
        self.out_conv = _zero_(
            _conv2d(self.decoder.out_channels[-1], self.out_channels, 3, padding=1))

    def forward(self, x_cat: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        target_size = x_cat.shape[2:]
        emb = self.time_mlp(_sinusoidal_embedding(c_noise, self.model_ch))
        enc_out = self.encoder(x_cat, emb)
        bottleneck = self.middle(enc_out["bottleneck"], emb)
        dec_features = self.decoder(bottleneck, enc_out["enc_skips"], emb)
        h = dec_features[-1]
        out = self.out_conv(F.silu(self.out_norm(h)))
        if out.shape[2:] != target_size:
            out = F.interpolate(
                out, size=target_size, mode="bilinear", align_corners=False)
        return out

    def param_count(self) -> Dict[str, int]:
        return {"total": sum(p.numel() for p in self.parameters())}


def build_adm_diffusion_unet(
    cfg, in_channels: int, out_channels: int) -> ADMDiffusionUNet:
    """从 Config 构造 ADM 扩散 backbone（2.5D）。``in/out_channels`` 已折叠 D。"""
    mc = cfg.model
    enc_channels = list(mc.encoder_channels)
    n_levels = len(enc_channels)
    enc_bps = list(mc.encoder_blocks_per_stage) or [int(mc.blocks_per_level)] * n_levels
    if len(enc_bps) != n_levels:
        raise ValueError(
            f"encoder_blocks_per_stage length {len(enc_bps)} != {n_levels}")
    attn_levels = _resolve_attention_levels(n_levels, mc.adm_attention_levels)
    model = ADMDiffusionUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        encoder_channels=enc_channels,
        encoder_blocks_per_stage=enc_bps,
        attention_levels=attn_levels,
        num_heads=int(mc.adm_num_heads),
        num_head_channels=int(mc.adm_num_head_channels),
        dropout=float(mc.dropout))
    logger.info(
        "Built ADMDiffusionUNet: total=%.2fM, channels=%s, enc_blocks=%s, "
        "attn_levels=%s, in_ch=%d, out_ch=%d",
        model.param_count()["total"] / 1e6, enc_channels, enc_bps,
        attn_levels, in_channels, out_channels)
    return model
