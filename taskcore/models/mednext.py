"""MedNeXt blocks for 3D/2D UNet (Roy et al., MICCAI 2023, dim-agnostic 2D/3D).

档位 A（本文件）：实现 MedNeXt 的核心**残差倒瓶颈块**，复用框架既有的
``Downsample`` / ``Upsample`` 做重采样（``downsample_mode`` / ``upsample_mode`` 仍生效，
且与 ``anisotropic_pooling`` 兼容）。MedNeXt 原生的「重采样残差块（Up/Down block 把 stride
融入深度卷积 + 1×1 残差）」与 UpKern 大核权重迁移为后续档位 B。

Block（C 通道输入，参照论文 §2.1，3 层 mirror Transformer）:
  1. Depthwise Conv k³（groups=C）→ 通道级 GroupNorm（num_groups=C；小 batch 稳定，
     替代原 ConvNeXt 的 LayerNorm）。
  2. Expansion: 1×1 Conv（C → C·R）→ GELU。
  3. Compression: 1×1 Conv（C·R → C）。
  + 残差（in==out, stride=1）。
与 ConvNeXt 的差异：GroupNorm（非 LN）、核 3/5（非 7）、扩张比 R 可配（非固定 4）、无 LayerScale。
"""

from __future__ import annotations

import logging
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DropPath, GlobalResponseNorm, _BN, _CONV, make_attention

logger = logging.getLogger(__name__)


def _channelwise_groupnorm(num_channels: int) -> nn.GroupNorm:
    """通道级 GroupNorm（num_groups == num_channels）：MedNeXt 原作选型，
    等价逐通道按空间统计，小 batch 比 LayerNorm/BatchNorm 更稳。"""
    return nn.GroupNorm(num_groups=num_channels, num_channels=num_channels)


def _default_dilated_reparam_branches(kernel_size: int) -> list[tuple[int, int]]:
    """按 K 生成默认 UniRepLKNet 风格分支（仅小集合，便于调参与验证）。"""
    base = {
        3: [(3, 1)],
        5: [(3, 1), (3, 2)],
        7: [(3, 1), (5, 2), (3, 3)],
    }
    specs = base.get(int(kernel_size), [(3, 1), (3, 2), (3, 3)])
    out = []
    for k, d in specs:
        eff = (k - 1) * d + 1
        if eff <= kernel_size:
            out.append((k, d))
    return out


def _validate_dilated_reparam_branches(
    kernel_size: int,
    branch_kernel_sizes: list[int] | None,
    branch_dilations: list[int] | None,
) -> list[tuple[int, int]]:
    """规整/校验显式分支配置；空列表则回退默认分支集。"""
    k = int(kernel_size)
    if branch_kernel_sizes is None and branch_dilations is None:
        return _default_dilated_reparam_branches(k)
    if not branch_kernel_sizes and not branch_dilations:
        return _default_dilated_reparam_branches(k)
    if not branch_kernel_sizes or not branch_dilations:
        raise ValueError(
            "dilated_reparam branch override requires both "
            "branch_kernel_sizes and branch_dilations.")
    if len(branch_kernel_sizes) != len(branch_dilations):
        raise ValueError(
            "dilated_reparam branch override lists must have the same length.")
    specs: list[tuple[int, int]] = []
    for bk, bd in zip(branch_kernel_sizes, branch_dilations):
        bk = int(bk)
        bd = int(bd)
        eff = (bk - 1) * bd + 1
        if bk % 2 != 1:
            raise ValueError(
                f"dilated_reparam branch kernel must be odd, got {bk}.")
        if bd < 1:
            raise ValueError(
                f"dilated_reparam branch dilation must be >= 1, got {bd}.")
        if eff % 2 != 1:
            raise ValueError(
                f"dilated_reparam effective kernel must be odd, got "
                f"kernel={bk}, dilation={bd}, effective={eff}.")
        if eff > k:
            raise ValueError(
                f"dilated_reparam effective kernel {eff} exceeds target "
                f"kernel_size={k}.")
        specs.append((bk, bd))
    return specs


def _fold_conv_bn(
    conv: nn.Module,
    bn: nn.modules.batchnorm._BatchNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把 Conv + BN 折叠成等价的卷积权重和偏置。"""
    weight = conv.weight
    bias = conv.bias
    if bias is None:
        bias = torch.zeros(weight.shape[0], device=weight.device,
                           dtype=weight.dtype)
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    std = torch.sqrt(var + eps)
    scale = gamma / std
    view_shape = (weight.shape[0],) + (1,) * (weight.ndim - 1)
    weight = weight * scale.reshape(view_shape)
    bias = beta + (bias - mean) * scale
    return weight, bias


def _expand_dilated_kernel(
    weight: torch.Tensor,
    target_kernel_size: int,
    dilation: int,
) -> torch.Tensor:
    """把 dilated depthwise kernel 展开到目标大核尺寸。"""
    spatial_dims = weight.ndim - 2
    k = int(weight.shape[-1])
    eff = (k - 1) * int(dilation) + 1
    if eff > target_kernel_size:
        raise ValueError(
            f"effective kernel {eff} exceeds target kernel_size="
            f"{target_kernel_size}.")
    expanded = weight.new_zeros(
        (weight.shape[0], weight.shape[1]) + (eff,) * spatial_dims)
    for idx in product(range(k), repeat=spatial_dims):
        target_idx = tuple(i * dilation for i in idx)
        expanded[(slice(None), slice(None)) + target_idx] = weight[
            (slice(None), slice(None)) + idx]
    pad = (target_kernel_size - eff) // 2
    if pad > 0:
        pad_spec = []
        for _ in range(spatial_dims):
            pad_spec.extend((pad, pad))
        expanded = F.pad(expanded, tuple(reversed(pad_spec)))
    return expanded


class DilatedReparamBlock(nn.Module):
    """UniRepLKNet/RepLK 风格的可重参数化深度卷积块。

    训练态：大核 depthwise conv + BN 与若干 dilated depthwise 分支 + 各自 BN
    并行求和。
    推理态：把所有分支的 Conv+BN 折叠并展开到同一大核后，重参数化为单个
    depthwise Conv，零额外推理开销。

    这里内部使用 BatchNorm 仅为精确 fold 需要；它只存在于训练/折叠前的模块
    结构中，和 MedNeXt 主干里沿用的通道级 GroupNorm 互不冲突。它更贴近大
    batch 的 fold 路径；3D 小 batch 消融时 running stats 可能更噪。
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        *,
        branch_kernel_sizes: list[int] | None = None,
        branch_dilations: list[int] | None = None,
        spatial_dims: int = 3):
        super().__init__()
        d = spatial_dims
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}.")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.spatial_dims = d
        self.branch_specs = _validate_dilated_reparam_branches(
            self.kernel_size, branch_kernel_sizes, branch_dilations)
        Conv = _CONV[d]
        BN = _BN[d]

        self.lk = Conv(
            channels, channels, kernel_size=kernel_size, padding=kernel_size // 2,
            groups=channels, bias=False)
        self.lk_bn = BN(channels)
        self.branches = nn.ModuleList()
        for bk, bd in self.branch_specs:
            eff = (bk - 1) * bd + 1
            self.branches.append(nn.Sequential(
                Conv(
                    channels, channels, kernel_size=bk, padding=eff // 2,
                    dilation=bd, groups=channels, bias=False),
                BN(channels),
            ))
        self.deploy = False
        self.reparam = None

    def _branch_to_kernel_bias(self, branch: nn.Sequential) -> tuple[torch.Tensor, torch.Tensor]:
        conv = branch[0]
        bn = branch[1]
        weight, bias = _fold_conv_bn(conv, bn)
        if conv.dilation[0] != 1:
            weight = _expand_dilated_kernel(weight, self.kernel_size, conv.dilation[0])
        elif weight.shape[-1] != self.kernel_size:
            pad = (self.kernel_size - weight.shape[-1]) // 2
            if pad > 0:
                pad_spec = []
                for _ in range(self.spatial_dims):
                    pad_spec.extend((pad, pad))
                weight = F.pad(weight, tuple(reversed(pad_spec)))
        return weight, bias

    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """返回折叠后的等效大核和 bias。"""
        weight, bias = _fold_conv_bn(self.lk, self.lk_bn)
        if weight.shape[-1] != self.kernel_size:
            pad = (self.kernel_size - weight.shape[-1]) // 2
            if pad > 0:
                pad_spec = []
                for _ in range(self.spatial_dims):
                    pad_spec.extend((pad, pad))
                weight = F.pad(weight, tuple(reversed(pad_spec)))
        for branch in self.branches:
            bw, bb = self._branch_to_kernel_bias(branch)
            weight = weight + bw
            bias = bias + bb
        return weight, bias

    def switch_to_deploy(self) -> "DilatedReparamBlock":
        """重参数化为单个 depthwise Conv；重复调用保持幂等。"""
        if self.deploy:
            return self
        weight, bias = self.get_equivalent_kernel_bias()
        Conv = _CONV[self.spatial_dims]
        reparam = Conv(
            self.channels, self.channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=self.channels,
            bias=True,
            device=weight.device,
            dtype=weight.dtype)
        reparam.weight.data.copy_(weight)
        reparam.bias.data.copy_(bias)
        self.reparam = reparam
        self.deploy = True
        del self.lk
        del self.lk_bn
        del self.branches
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.reparam(x)
        out = self.lk_bn(self.lk(x))
        for branch in self.branches:
            out = out + branch(x)
        return out


def reparameterize_model(model: nn.Module) -> nn.Module:
    """递归调用所有子模块的 ``switch_to_deploy``。"""
    for module in list(model.modules()):
        switch = getattr(module, "switch_to_deploy", None)
        if callable(switch):
            switch()
    return model


def upkern_remap_state_dict(src_sd: dict, target_model: nn.Module) -> dict:
    """把小核 MedNeXt checkpoint 的深度卷积权重插值到目标大核。

    仅处理与目标参数同名、同 rank、同通道形状的 depthwise-conv-like 权重：
    ``(C, 1, k, k[, k])``。当仅空间核尺寸不一致时，按空间维做
    ``bilinear``/``trilinear`` 插值并保留其余张量不变；不在目标模型中的键
    直接丢弃，无法对齐的张量也保持目标模型初始化值。这里沿用
    ``align_corners=True`` 的现有 UpKern 迁移行为；它与 MedNeXt 官方默认的
    ``False`` 不同，但对 3→5 这类小核影响很小。

    Parameters
    ----------
    src_sd:
        源 checkpoint 的 state_dict。
    target_model:
        目标 MedNeXt 模型，用于提供目标形状。
    """
    target_sd = target_model.state_dict()
    remapped = {}

    def _prefix_and_kind(key: str) -> tuple[str | None, str | None]:
        if key.endswith(".dwconv.weight"):
            return key[:-len(".dwconv.weight")], "plain"
        if key.endswith(".dwconv.lk.weight"):
            return key[:-len(".dwconv.lk.weight")], "reparam"
        return None, None

    def _preview(keys: list[str], n: int = 8) -> str:
        if not keys:
            return "[]"
        head = ", ".join(keys[:n])
        if len(keys) > n:
            head += f", ... (+{len(keys) - n} more)"
        return f"[{head}]"

    def _depthwise_like_tensor(
        src_tensor: torch.Tensor,
        tgt_tensor: torch.Tensor,
        key: str,
    ) -> torch.Tensor | None:
        if src_tensor.shape == tgt_tensor.shape:
            return src_tensor
        if (src_tensor.ndim not in (4, 5)
                or tgt_tensor.ndim != src_tensor.ndim
                or src_tensor.shape[:2] != tgt_tensor.shape[:2]):
            return None
        if src_tensor.shape[1] != 1:
            logger.warning(
                "UpKern remap: skipping non-depthwise tensor %s "
                "(shape %s -> %s); target keeps its initialization.",
                key, tuple(src_tensor.shape), tuple(tgt_tensor.shape))
            return None
        if src_tensor.shape[2:] == tgt_tensor.shape[2:]:
            return src_tensor
        # 以 float32 完成插值，再回到源 dtype。
        mode = "bilinear" if src_tensor.ndim == 4 else "trilinear"
        spatial = tuple(int(s) for s in tgt_tensor.shape[2:])
        work = src_tensor.detach().to(dtype=torch.float32)
        work = work.reshape(work.shape[0] * work.shape[1], 1, *work.shape[2:])
        work = F.interpolate(work, size=spatial, mode=mode, align_corners=True)
        work = work.reshape(*tgt_tensor.shape).to(dtype=src_tensor.dtype)
        return work

    src_prefixes = {"plain": set(), "reparam": set()}
    tgt_prefixes = {"plain": set(), "reparam": set()}
    for key, tensor in src_sd.items():
        prefix, kind = _prefix_and_kind(key)
        if prefix and torch.is_tensor(tensor) and tensor.ndim in (4, 5):
            src_prefixes[kind].add(prefix)
    for key, tensor in target_sd.items():
        prefix, kind = _prefix_and_kind(key)
        if prefix and torch.is_tensor(tensor) and tensor.ndim in (4, 5):
            tgt_prefixes[kind].add(prefix)

    for prefix in sorted(src_prefixes["plain"] & tgt_prefixes["reparam"]):
        target_keys = [
            k for k in target_sd
            if k.startswith(prefix + ".dwconv.")
            and not k.endswith(".dwconv.lk.weight")
        ]
        logger.warning(
            "UpKern remap: plain checkpoint -> reparameterized target for "
            "module prefix %s; target-init keys stay random: %s",
            prefix, _preview(sorted(target_keys)))
    for prefix in sorted(src_prefixes["reparam"] & tgt_prefixes["plain"]):
        source_keys = [
            k for k in src_sd
            if k.startswith(prefix + ".dwconv.")
            and not k.endswith(".dwconv.lk.weight")
        ]
        logger.warning(
            "UpKern remap: reparameterized checkpoint -> plain target for "
            "module prefix %s; discarded reparam-only keys: %s",
            prefix, _preview(sorted(source_keys)))

    for key, src_tensor in src_sd.items():
        tgt_tensor = target_sd.get(key)
        mapped_key = key
        if tgt_tensor is None and torch.is_tensor(src_tensor):
            if key.endswith(".dwconv.weight"):
                candidate = key[:-len(".weight")] + ".lk.weight"
                tgt_tensor = target_sd.get(candidate)
                if tgt_tensor is not None:
                    mapped_key = candidate
            elif key.endswith(".dwconv.lk.weight"):
                candidate = key[:-len(".lk.weight")] + ".weight"
                tgt_tensor = target_sd.get(candidate)
                if tgt_tensor is not None:
                    mapped_key = candidate
        if tgt_tensor is None or not torch.is_tensor(src_tensor):
            continue
        if not torch.is_tensor(tgt_tensor):
            continue
        mapped = _depthwise_like_tensor(src_tensor, tgt_tensor, mapped_key)
        if mapped is not None:
            remapped[mapped_key] = mapped
            continue
    return remapped


class MedNeXtBlock(nn.Module):
    """MedNeXt 残差倒瓶颈块（stride=1, in==out）。

    dwconv(k) → 通道级 GroupNorm → pwconv↑(×R) → GELU → pwconv↓ → attn? → +residual。
    """

    def __init__(
        self,
        dim           : int,
        expand_ratio  : int = 4,
        kernel_size   : int = 3,
        drop_path     : float = 0.0,
        attention_type: str = "none",
        use_grn       : bool = False,
        spatial_dims  : int = 3,
        dilated_reparam: bool = False,
        dilated_reparam_branch_kernel_sizes: list[int] | None = None,
        dilated_reparam_branch_dilations: list[int] | None = None,
        attn_reduction: int = 16):
        super().__init__()
        d = spatial_dims
        self.spatial_dims = d
        hidden  = int(dim * expand_ratio)
        if dilated_reparam:
            self.dwconv = DilatedReparamBlock(
                dim, kernel_size,
                branch_kernel_sizes=dilated_reparam_branch_kernel_sizes,
                branch_dilations=dilated_reparam_branch_dilations,
                spatial_dims=d)
        else:
            padding = kernel_size // 2
            self.dwconv  = _CONV[d](
                dim, dim, kernel_size=kernel_size, padding=padding,
                groups=dim, bias=True)
        self.norm    = _channelwise_groupnorm(dim)
        self.pwconv1 = _CONV[d](dim, hidden, kernel_size=1, bias=True)
        self.act     = nn.GELU()
        self.grn     = GlobalResponseNorm(hidden, spatial_dims=d) if use_grn else nn.Identity()
        self.pwconv2 = _CONV[d](hidden, dim, kernel_size=1, bias=True)
        # reduction 跟随 config（model.se_reduction，与 ResNet 系一致）；coord 内部
        # 归一化保持其默认 group/8（MedNeXt 块内 norm 固定为通道级 GroupNorm）。
        self.attn    = make_attention(attention_type, dim, spatial_dims=d,
                                      reduction=attn_reduction)
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
        return res + self.drop_path(out)


class MedNeXtAdaptBlock(nn.Module):
    """通道适配版：in_ch != out_ch 时先 1×1 投影（+GroupNorm）再走标准 MedNeXt 块。

    本框架在「stage 首个 block」处升通道（stage 间下采样保持通道），故 stage 起始块需此适配
    （与 ConvNeXtAdaptBlock 同构）。投影后残差在 out_ch 维度内闭合。
    """

    def __init__(
        self,
        in_ch         : int,
        out_ch        : int,
        expand_ratio  : int = 4,
        kernel_size   : int = 3,
        drop_path     : float = 0.0,
        attention_type: str = "none",
        use_grn       : bool = False,
        spatial_dims  : int = 3,
        dilated_reparam: bool = False,
        dilated_reparam_branch_kernel_sizes: list[int] | None = None,
        dilated_reparam_branch_dilations: list[int] | None = None,
        attn_reduction: int = 16):
        super().__init__()
        d = spatial_dims
        self.proj = (
            nn.Sequential(
                _CONV[d](in_ch, out_ch, 1, bias=False),
                _channelwise_groupnorm(out_ch))
            if in_ch != out_ch else nn.Identity())
        self.block = MedNeXtBlock(
            out_ch, expand_ratio=expand_ratio, kernel_size=kernel_size,
            drop_path=drop_path, attention_type=attention_type,
            use_grn=use_grn, spatial_dims=d, attn_reduction=attn_reduction,
            dilated_reparam=dilated_reparam,
            dilated_reparam_branch_kernel_sizes=
            dilated_reparam_branch_kernel_sizes,
            dilated_reparam_branch_dilations=
            dilated_reparam_branch_dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.proj(x))


class MedNeXtStage(nn.Module):
    """单分辨率 N 个 MedNeXt 块（首块可改通道）。接口与 ConvNeXtStage/ResNetStage 一致。"""

    def __init__(
        self,
        in_ch         : int,
        out_ch        : int,
        num_blocks    : int = 2,
        expand_ratio  : int = 4,
        kernel_size   : int = 3,
        drop_path_rates: list = None,
        attention_type: str = "none",
        use_grn       : bool = False,
        spatial_dims  : int = 3,
        dilated_reparam: bool = False,
        dilated_reparam_branch_kernel_sizes: list[int] | None = None,
        dilated_reparam_branch_dilations: list[int] | None = None,
        attn_reduction: int = 16):
        super().__init__()
        d = spatial_dims
        if drop_path_rates is None:
            drop_path_rates = [0.0] * num_blocks
        blocks = [MedNeXtAdaptBlock(
            in_ch, out_ch, expand_ratio=expand_ratio, kernel_size=kernel_size,
            drop_path=drop_path_rates[0], attention_type=attention_type,
            use_grn=use_grn, spatial_dims=d, attn_reduction=attn_reduction,
            dilated_reparam=dilated_reparam,
            dilated_reparam_branch_kernel_sizes=
            dilated_reparam_branch_kernel_sizes,
            dilated_reparam_branch_dilations=
            dilated_reparam_branch_dilations)]
        for i in range(1, num_blocks):
            dp = drop_path_rates[i] if i < len(drop_path_rates) else 0.0
            blocks.append(MedNeXtBlock(
                out_ch, expand_ratio=expand_ratio, kernel_size=kernel_size,
                drop_path=dp, attention_type=attention_type,
                use_grn=use_grn, spatial_dims=d, attn_reduction=attn_reduction,
                dilated_reparam=dilated_reparam,
                dilated_reparam_branch_kernel_sizes=
                dilated_reparam_branch_kernel_sizes,
                dilated_reparam_branch_dilations=
                dilated_reparam_branch_dilations))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


__all__ = [
    "DilatedReparamBlock",
    "MedNeXtBlock",
    "MedNeXtAdaptBlock",
    "MedNeXtStage",
    "reparameterize_model",
    "upkern_remap_state_dict",
]
