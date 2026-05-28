"""3D UNet 的 stem / patch-embed 构建器。

stem 模式：conv3/conv7/dual stride=1；patch2/patch4 stride=N 降分辨率（UNet 末尾上采补回）。
多 FOV 上下文融合 (2.5D, n_views>1)：示例 'multi_stem_proj' 逐 view stem + 3×3 融合、'hierarchical' aux 逐级注入。还有 'shared_stem'（全部走同一 stem，最轻）。
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from .blocks import _CONV, ConvNormAct, get_activation, get_norm


STEM_MODES = ("conv3", "conv7", "dual", "patch2", "patch4")


class DualConvStem(nn.Module):
    """两个堆叠 3×3×3 conv-norm-act（nnU-Net stem）。"""

    def __init__(
        self,
        in_ch       : int,
        out_ch      : int,
        norm_type   : str = "instance",
        norm_groups : int = 8,
        activation  : str = "leakyrelu",
        spatial_dims: int = 3):
        super().__init__()
        self.block1 = ConvNormAct(
            in_ch, out_ch, kernel_size=3, stride=1, padding=1,
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation, spatial_dims=spatial_dims)
        self.block2 = ConvNormAct(
            out_ch, out_ch, kernel_size=3, stride=1, padding=1,
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation, spatial_dims=spatial_dims)

    def forward(self, x):
        return self.block2(self.block1(x))


class PatchEmbedStem(nn.Module):
    """Patch-embed stem：stride-N conv+norm+act，分辨率除以 patch_size。"""

    def __init__(
        self,
        in_ch       : int,
        out_ch      : int,
        patch_size  : int,
        norm_type   : str = "instance",
        norm_groups : int = 8,
        activation  : str = "gelu",
        spatial_dims: int = 3):
        super().__init__()
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}")
        self.patch_size = patch_size
        self.conv       = _CONV[spatial_dims](
            in_ch, out_ch,
            kernel_size = patch_size,
            stride      = patch_size,
            bias=False)
        self.norm = get_norm(norm_type, out_ch, norm_groups,
                             spatial_dims=spatial_dims)
        self.act = get_activation(activation)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


def build_stem(
    mode        : str,
    in_ch       : int,
    out_ch      : int,
    norm_type   : str = "instance",
    norm_groups : int = 8,
    activation  : str = "leakyrelu",
    spatial_dims: int = 3) -> Tuple[nn.Module, int]:
    """构建 stem；返回 (module, stem_stride)。conv3/7/dual=1；patchN=N。"""
    if mode not in STEM_MODES:
        raise ValueError(f"Unknown stem mode: {mode!r}. Valid: {STEM_MODES}")

    if mode == "conv3":
        # 单层 3×3：最薄方案。2.5D 折叠下 z 关系仅靠这一层非线性提取，
        # 容量偏弱；多切片输入推荐改用 'dual'。
        stem = ConvNormAct(
            in_ch, out_ch, kernel_size=3, stride=1, padding=1,
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation, spatial_dims=spatial_dims)
        return stem, 1

    if mode == "conv7":
        # 单层 7×7：感受野大但只有一次非线性；与 conv3 同样容量受限。
        stem = ConvNormAct(
            in_ch, out_ch, kernel_size=7, stride=1, padding=3,
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation, spatial_dims=spatial_dims)
        return stem, 1

    if mode == "dual":
        stem = DualConvStem(
            in_ch, out_ch,
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation, spatial_dims=spatial_dims)
        return stem, 1

    # patch-embed 变体：调用者保留 leakyrelu 默认时自动换为 GELU。
    patch_size = 2 if mode == "patch2" else 4
    stem = PatchEmbedStem(
        in_ch, out_ch, patch_size=patch_size,
        norm_type=norm_type, norm_groups=norm_groups,
        activation="gelu" if activation == "leakyrelu" else activation,
        spatial_dims=spatial_dims)
    return stem, patch_size


# 2.5D 多 z-FOV 下上下文融合 stem。
CONTEXT_FUSION_MODES = ("shared_stem", "multi_stem_proj", "hierarchical")


class MultiStemProj(nn.Module):
    """n_views 个独立 stem → 通道 cat → 3×3 融合为 out_ch；逐 view 学 FOV 专属滤波器。

    融合层用 3×3 (而非 1×1) 是为给跨 view 的特征融合一次空间能力——多 FOV 的核心
    差异是 z-axis 几何不同，1×1 仅做 per-pixel 通道线性投影，无法对齐/重组 view
    间的空间结构。3×3 在 stem 后 (输入分辨率最高) 代价小但提供 3 邻域感受野，
    与 nnU-Net stem block 的卷积核同构。"""

    def __init__(
        self,
        mode               : str,
        n_views            : int,
        in_ch_per_view     : int,
        out_ch             : int,
        norm_type          : str = "instance",
        norm_groups        : int = 8,
        activation         : str = "leakyrelu",
        spatial_dims       : int = 3,
        in_ch_per_view_list: List[int] = None):
        super().__init__()
        if n_views < 1:
            raise ValueError(f"n_views must be >= 1, got {n_views}")
        self.n_views = n_views

        if in_ch_per_view_list is not None:
            if len(in_ch_per_view_list) != n_views:
                raise ValueError(
                    f"in_ch_per_view_list length ({len(in_ch_per_view_list)}) "
                    f"must equal n_views ({n_views})")
            self.in_ch_per_view_list: List[int] = [int(c) for c in in_ch_per_view_list]
        else:
            self.in_ch_per_view_list = [int(in_ch_per_view)] * n_views
        # 后兼容：仅首 view 计数（完整信息请读 in_ch_per_view_list）。
        self.in_ch_per_view = self.in_ch_per_view_list[0]

        stems  : List[nn.Module] = []
        strides: List[int]       = []
        for c_v in self.in_ch_per_view_list:
            s, stride = build_stem(
                mode, c_v, out_ch,
                norm_type=norm_type, norm_groups=norm_groups,
                activation=activation, spatial_dims=spatial_dims)
            stems.append(s)
            strides.append(stride)
        if len(set(strides)) != 1:
            # 同 mode 下子 stem stride 应一致；防御性检查。
            raise RuntimeError(
                f"MultiStemProj sub-stems disagree on stride: {strides}")
        self.stems       = nn.ModuleList(stems)
        self.stem_stride = strides[0]

        # 3×3 融合回 out_ch：保留下游通道契约，同时给跨 view 一次空间感受野
        # (1×1 仅做通道线性投影，无法对齐 view 间空间结构)。
        self.proj = ConvNormAct(
            n_views * out_ch, out_ch,
            kernel_size=3, stride=1, padding=1,
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation, spatial_dims=spatial_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, sum(in_ch_per_view_list), *spatial)。"""
        expected_c = sum(self.in_ch_per_view_list)
        if x.shape[1] != expected_c:
            raise ValueError(
                f"MultiStemProj expects {expected_c} input channels "
                f"(per-view={self.in_ch_per_view_list}); got {x.shape[1]}")
        # 逐 view 通道拆分（零拷贝）。
        chunks = torch.split(x, self.in_ch_per_view_list, dim=1)
        feats  = [stem(c) for stem, c in zip(self.stems, chunks)]
        return self.proj(torch.cat(feats, dim=1))


class HierarchicalStems(nn.Module):
    """逐 FOV stem，stride 与阶对齐。view 0 → main_stem (native stride s0)；view k≥1 → aux_stems[k-1] PatchEmbed (stride=s0*2^k，out=stage_channels[k-1])。encoder 逐级 cat 融合 (2*ch→ch)。调用者需分别调 forward_main / forward_aux。"""

    def __init__(
        self,
        mode: str,
        n_views: int,
        in_ch_per_view: int,
        stage_channels: List[int],
        norm_type: str = "instance",
        norm_groups: int = 8,
        activation: str = "leakyrelu",
        spatial_dims: int = 3,
        aux_channels: List[int] = None,
        in_ch_per_view_list: List[int] = None):
        """通道布局同 MultiStemProj：均分或逐 view 列表。"""
        super().__init__()
        if n_views < 1:
            raise ValueError(f"n_views must be >= 1, got {n_views}")
        n_aux = n_views - 1
        if n_aux > 0 and len(stage_channels) <= n_aux:
            raise ValueError(
                f"hierarchical fusion requires at least n_views={n_views} "
                f"encoder stages (one stage per aux injection level + the "
                f"main path); got {len(stage_channels)} stages.")
        self.n_views = n_views
        if in_ch_per_view_list is not None:
            if len(in_ch_per_view_list) != n_views:
                raise ValueError(
                    f"in_ch_per_view_list length ({len(in_ch_per_view_list)}) "
                    f"must equal n_views ({n_views})")
            self.in_ch_per_view_list: List[int] = [int(c) for c in in_ch_per_view_list]
        else:
            self.in_ch_per_view_list = [int(in_ch_per_view)] * n_views
        # 后兼容：仅首 view 计数。
        self.in_ch_per_view = self.in_ch_per_view_list[0]

        # Main stem (view 0)
        self.main_stem, self.stem_stride = build_stem(
            mode, self.in_ch_per_view_list[0], stage_channels[0],
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation, spatial_dims=spatial_dims)

        # Aux stems：stride = main_stride*2^k；out_ch 默认 stage_channels[k-1]。
        if aux_channels is None:
            aux_channels = [stage_channels[k - 1] for k in range(1, n_views)]
        if len(aux_channels) != n_aux:
            raise ValueError(
                f"aux_channels length ({len(aux_channels)}) must equal "
                f"n_views - 1 ({n_aux})")

        self.aux_stems = nn.ModuleList()
        self.aux_levels: List[int] = []
        self.aux_strides: List[int] = []
        self.aux_out_channels: List[int] = list(aux_channels)
        for k in range(1, n_views):
            stride = self.stem_stride * (2 ** k)
            self.aux_levels.append(k)
            self.aux_strides.append(stride)
            self.aux_stems.append(
                PatchEmbedStem(
                    in_ch=self.in_ch_per_view_list[k],
                    out_ch=aux_channels[k - 1],
                    patch_size=stride,
                    norm_type=norm_type, norm_groups=norm_groups,
                    activation=("gelu" if activation == "leakyrelu"
                                else activation),
                    spatial_dims=spatial_dims))

    def split_views(self, x: torch.Tensor) -> List[torch.Tensor]:
        expected_c = sum(self.in_ch_per_view_list)
        if x.shape[1] != expected_c:
            raise ValueError(
                f"HierarchicalStems expects {expected_c} input channels "
                f"(per-view={self.in_ch_per_view_list}); got {x.shape[1]}")
        return list(torch.split(x, self.in_ch_per_view_list, dim=1))

    def forward_main(self, x_view0: torch.Tensor) -> torch.Tensor:
        return self.main_stem(x_view0)

    def forward_aux(
        self, chunks: List[torch.Tensor],
    ) -> "OrderedDict[int, torch.Tensor]":
        """逐 aux stem 作用于对应 view chunk；返回有序 {level: aux_feature}。"""
        from collections import OrderedDict
        out: "OrderedDict[int, torch.Tensor]" = OrderedDict()
        for k, stem in enumerate(self.aux_stems):
            level = self.aux_levels[k]
            out[level] = stem(chunks[k + 1])  # +1 跳过 view 0
        return out


def build_context_stem(
    mode               : str,
    fusion             : str,
    n_views            : int,
    in_ch_per_view     : int,
    out_ch             : int,
    norm_type          : str = "instance",
    norm_groups        : int = 8,
    activation         : str = "leakyrelu",
    spatial_dims       : int = 3,
    stage_channels     : List[int] = None,
    in_ch_per_view_list: List[int] = None) -> Tuple[nn.Module, int]:
    """分派 2.5D 多 FOV stem，返回 (module, stem_stride)。

    n_views==1 或 'shared_stem'：单 stem。'multi_stem_proj'：逐 view stem + 3×3 融合。'hierarchical'：需 stage_channels，encoder 逐级 cat 融合。
    """
    if fusion not in CONTEXT_FUSION_MODES:
        raise ValueError(
            f"Unknown context_fusion: {fusion!r}. Valid: {CONTEXT_FUSION_MODES}")
    # 校验逐 view / 均分布局总通道一致。
    if in_ch_per_view_list is not None and len(in_ch_per_view_list) != n_views:
        raise ValueError(
            f"in_ch_per_view_list length ({len(in_ch_per_view_list)}) "
            f"must equal n_views ({n_views})")

    if n_views == 1 or fusion == "shared_stem":
        # 总输入通道：逐 view 列表求和 或 均分。
        total_in = (sum(in_ch_per_view_list)
                    if in_ch_per_view_list is not None else n_views * in_ch_per_view)
        return build_stem(  # 一个stem
            mode, total_in, out_ch,
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation, spatial_dims=spatial_dims)
    if fusion == "multi_stem_proj":
        msp = MultiStemProj(  # 每个FOV一个stem
            mode=mode, n_views=n_views,
            in_ch_per_view=in_ch_per_view, out_ch=out_ch,
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation, spatial_dims=spatial_dims,
            in_ch_per_view_list=in_ch_per_view_list)
        return msp, msp.stem_stride
    # hierarchical
    if stage_channels is None:
        raise ValueError(
            "hierarchical fusion requires stage_channels (encoder channel "
            "list) so aux stems can size their output channels to match "
            "each injection level. Pass stage_channels=encoder_channels.")
    if stage_channels[0] != out_ch:
        # 契约：out_ch 必须等于 stage_channels[0]（main stem 输出）。
        raise ValueError(
            f"hierarchical fusion: out_ch ({out_ch}) must equal "
            f"stage_channels[0] ({stage_channels[0]}).")

    hier = HierarchicalStems(
        mode=mode, n_views=n_views,
        in_ch_per_view=in_ch_per_view,
        stage_channels=stage_channels,
        norm_type=norm_type, norm_groups=norm_groups,
        activation=activation, spatial_dims=spatial_dims,
        in_ch_per_view_list=in_ch_per_view_list)
    return hier, hier.stem_stride
