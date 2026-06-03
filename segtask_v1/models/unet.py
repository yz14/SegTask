"""通用 UNet：对称 enc/dec，backbone block 由 factory 注入。支持深监督、cat/add skip、多 FOV 融合及辅助分割头。"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

import torch.nn.functional as F

from .blocks import (
    _CONV, INTERP_SMOOTH,
    AttentionGate3D, ConvNormAct, Downsample, Upsample, get_norm)
from .stem import HierarchicalStems, build_context_stem, build_stem


class Encoder(nn.Module):
    """UNet encoder：stem + N 个 stage（中间下采样）。返回 [level_0, ..., bottleneck]，level_0 最高分辨率。"""

    def __init__(
        self,
        in_channels          : int,
        stage_channels       : List[int],
        stage_builder,
        norm_type            : str = "instance",
        norm_groups          : int = 8,
        activation           : str = "leakyrelu",
        downsample_mode      : str = "conv",
        stem_mode            : str = "conv3",
        spatial_dims         : int = 3,
        num_stem_fusion_views: int = 1,
        stem_fusion_mode     : str = "shared_stem",
        in_ch_per_view_list  : List[int] = None,
        downsample_builder   : Optional[Callable[[int, int], nn.Module]] = None,
        downsample_strides   : Optional[List] = None):
        super().__init__()

        self.spatial_dims = spatial_dims
        if num_stem_fusion_views < 1:        # 输入不可为空
            raise ValueError(
                f"num_stem_fusion_views must be >= 1, got {num_stem_fusion_views}")
        if in_ch_per_view_list is not None:  # 2.5D多分辨率输入
            if len(in_ch_per_view_list) != num_stem_fusion_views:
                raise ValueError(
                    f"in_ch_per_view_list length "
                    f"({len(in_ch_per_view_list)}) must equal "
                    f"num_stem_fusion_views ({num_stem_fusion_views})")
            if sum(in_ch_per_view_list) != in_channels:  # 输入总通道数
                raise ValueError(
                    f"sum(in_ch_per_view_list)={sum(in_ch_per_view_list)} "
                    f"must equal in_channels ({in_channels})")
            base_ch_per_view = int(in_ch_per_view_list[0])
        else:
            if in_channels % num_stem_fusion_views != 0:
                raise ValueError(
                    f"in_channels ({in_channels}) must be divisible by "
                    f"num_stem_fusion_views ({num_stem_fusion_views})")
            base_ch_per_view = in_channels // num_stem_fusion_views
        self.num_stem_fusion_views = num_stem_fusion_views
        # UNet3D 构建 aux head 时需读此字段对齐 stem 拓扑。
        self.stem_fusion_mode = stem_fusion_mode
        self.in_ch_per_view_list: List[int] = (
            list(in_ch_per_view_list) if in_ch_per_view_list is not None
            else [base_ch_per_view] * num_stem_fusion_views)
        self.stem, self.stem_stride = build_context_stem(
            mode=stem_mode, fusion=stem_fusion_mode,
            n_views=num_stem_fusion_views,
            base_ch_per_view=base_ch_per_view,
            out_ch=stage_channels[0],
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation, spatial_dims=spatial_dims,
            stage_channels=stage_channels,
            in_ch_per_view_list=in_ch_per_view_list)

        # Encoder stages and downsampling
        self.stages      = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        # 逐级下采样 stride（各向异性）。None → 全程各向同性 2。长度需 = 级数-1。
        n_down = len(stage_channels) - 1
        if downsample_strides is not None and len(downsample_strides) != n_down:
            raise ValueError(
                f"downsample_strides length ({len(downsample_strides)}) must "
                f"equal len(stage_channels)-1 ({n_down}).")
        self.downsample_strides: List = (
            list(downsample_strides) if downsample_strides is not None
            else [2] * n_down)

        for i, ch in enumerate(stage_channels):
            in_ch = stage_channels[i - 1] if i > 0 else stage_channels[0]
            self.stages.append(stage_builder(in_ch, ch))
            if i > 0:
                # stage 间下采样（通道不变，下一 stage 首 block 升通道）
                ds_in = stage_channels[i - 1]
                if downsample_builder is not None:  # 允许自定义
                    self.downsamples.append(downsample_builder(ds_in, ds_in))
                else:
                    self.downsamples.append(Downsample(  # TODO 这里最后一层是norm，确定没有问题吗？需要加act吗？
                        ds_in, ds_in,
                        norm_type=norm_type, norm_groups=norm_groups,
                        mode=downsample_mode, spatial_dims=spatial_dims,
                        stride=self.downsample_strides[i - 1]))

        # 仅 hierarchical stem：每级 cat(main, aux) → 1×1 → stage_channels[k-1]。key 为 str(level)。
        self.aux_fuse = nn.ModuleDict()
        if isinstance(self.stem, HierarchicalStems):
            hs = self.stem
            for idx, level in enumerate(hs.aux_levels):
                main_ch = stage_channels[level - 1]
                aux_ch = hs.aux_out_channels[idx]
                self.aux_fuse[str(level)] = ConvNormAct(
                    main_ch + aux_ch, main_ch,
                    kernel_size=1, stride=1, padding=0,
                    norm_type=norm_type, norm_groups=norm_groups,
                    activation=activation, spatial_dims=spatial_dims)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """返回 [level_0, ..., level_N]；hierarchical stem 在下采样后 cat 注入 aux 特征。"""
        if isinstance(self.stem, HierarchicalStems):
            chunks    = self.stem.split_views(x)
            x         = self.stem.forward_main(chunks[0])
            aux_feats = self.stem.forward_aux(chunks)
        else:
            x         = self.stem(x)
            aux_feats = {}

        features: List[torch.Tensor] = []
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = self.downsamples[i - 1](x)
                if i in aux_feats:
                    aux = aux_feats[i]
                    if aux.shape[2:] != x.shape[2:]:
                        # aux stem stride 已对齐 main；不匹配说明输入未按 aux stem stride 整除。
                        raise RuntimeError(
                            f"Plan C aux feature spatial mismatch at "
                            f"level {i}: main={tuple(x.shape[2:])}, "
                            f"aux={tuple(aux.shape[2:])}. Check that "
                            f"input spatial dims are divisible by the "
                            f"aux stem stride.")
                    x = self.aux_fuse[str(i)](torch.cat([x, aux], dim=1))
            x = stage(x)
            features.append(x)
        return features


class DecoderLevel(nn.Module):
    """单层 decoder：上采样 → (可选 attention-gate) skip 融合 → stage block。skip_attention 启用 Attention U-Net (Oktay 2018)。"""

    def __init__(
        self,
        in_ch         : int,
        skip_ch       : int,
        out_ch        : int,
        stage_builder,
        upsample_mode : str = "transpose",
        skip_mode     : str = "cat",
        skip_attention: bool = False,
        spatial_dims  : int = 3,
        upsample_stride = 2):
        super().__init__()

        self.skip_mode    = skip_mode
        self.spatial_dims = spatial_dims
        self.upsample     = Upsample(in_ch, out_ch, mode=upsample_mode,
                                     spatial_dims=spatial_dims,
                                     stride=upsample_stride)

        if skip_mode == "cat":
            fused_ch = out_ch + skip_ch
        else:  # add：必要时投影 skip 通道。
            self.skip_proj = (
                _CONV[spatial_dims](skip_ch, out_ch, 1, bias=False)
                if skip_ch != out_ch else nn.Identity())
            fused_ch = out_ch

        self.attn_gate = (
            AttentionGate3D(x_ch=skip_ch, g_ch=out_ch,
                            spatial_dims=spatial_dims)
            if skip_attention else None)
        self.stage = stage_builder(fused_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        if x.shape[2:] != skip.shape[2:]:  # 上采样后必须与 skip 同尺寸
            raise RuntimeError(
                f"DecoderLevel size mismatch after upsample: "
                f"x={tuple(x.shape[2:])} vs skip={tuple(skip.shape[2:])}. "
                f"Check input spatial dims are divisible by total encoder stride.")

        if self.attn_gate is not None:
            # 用上采样后的 decoder 特征作 gate。
            skip = self.attn_gate(skip, x)

        if self.skip_mode == "cat":
            x = torch.cat([x, skip], dim=1)
        else:
            x = x + self.skip_proj(skip)

        return self.stage(x)


class Decoder(nn.Module):
    """UNet decoder：N 层上采样+skip 融合。输入 [level_0,...,bottleneck] → 输出 [dec_low_res,...,dec_high_res]。"""

    def __init__(
        self,
        encoder_channels  : List[int],
        stage_builder,
        upsample_mode     : str = "transpose",
        skip_mode         : str = "cat",
        skip_attention    : bool = False,
        spatial_dims      : int = 3,
        downsample_strides: Optional[List] = None):
        super().__init__()

        self.levels       = nn.ModuleList()
        self.spatial_dims = spatial_dims
        n = len(encoder_channels)

        # 镜像 encoder 的逐级下采样 stride：decoder level i 还原 encoder
        n_down = n - 1
        if downsample_strides is not None and len(downsample_strides) != n_down:
            raise ValueError(
                f"downsample_strides length ({len(downsample_strides)}) must "
                f"equal len(encoder_channels)-1 ({n_down}).")
        ds_strides: List = (
            list(downsample_strides) if downsample_strides is not None
            else [2] * n_down)

        # 自深至浅
        for i in range(n - 1):
            in_ch     = encoder_channels[n - 1 - i]  # from deeper level
            skip_ch   = encoder_channels[n - 2 - i]  # skip connection
            out_ch    = encoder_channels[n - 2 - i]  # symmetric output
            up_stride = ds_strides[n - 2 - i]        # 镜像对应下采样 stride

            self.levels.append(DecoderLevel(
                in_ch, skip_ch, out_ch, stage_builder,
                upsample_mode  = upsample_mode,
                skip_mode      = skip_mode,
                skip_attention = skip_attention,
                spatial_dims   = spatial_dims,
                upsample_stride = up_stride))

        # 各 decoder 层的输出通道（low-res → high-res）。
        self.out_channels = [encoder_channels[n - 2 - i] for i in range(n - 1)]

    def forward(self, encoder_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """[level_0, ..., bottleneck] → [dec_low_res, ..., dec_high_res]。"""
        x = encoder_features[-1]  # bottleneck
        outputs = []
        for i, level in enumerate(self.levels):
            skip_idx = len(encoder_features) - 2 - i
            x        = level(x, encoder_features[skip_idx])
            outputs.append(x)
        return outputs


class SegmentationHead(nn.Module):
    """1×1 卷积输出逐类 logits。"""

    def __init__(self, in_ch: int, num_classes: int, spatial_dims: int = 3):
        super().__init__()
        self.conv = _CONV[spatial_dims](in_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConvSegmentationHead(nn.Module):
    """3×3 ConvNormAct → 1×1 logits；比 SegmentationHead 更厚的 aux 头（约 +1% 参数）。"""

    def __init__(
        self,
        in_ch       : int,
        num_classes : int,
        spatial_dims: int = 3,
        norm_type   : str = "instance",
        norm_groups : int = 8,
        activation  : str = "leakyrelu"):
        super().__init__()
        self.conv = ConvNormAct(
            in_ch, in_ch, kernel_size=3, stride=1, padding=1,
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation, spatial_dims=spatial_dims)
        self.classifier = _CONV[spatial_dims](in_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv(x))


def build_head(
    mode        : str,
    in_ch       : int,
    num_classes : int,
    spatial_dims: int,
    norm_type   : str = "instance",
    norm_groups : int = 8,
    activation  : str = "leakyrelu") -> nn.Module:
    """分割头工厂：'linear' (1×1) | 'conv' (3×3+1×1)。"""
    if mode == "linear":
        return SegmentationHead(in_ch, num_classes, spatial_dims=spatial_dims)
    if mode == "conv":
        return ConvSegmentationHead(
            in_ch, num_classes, spatial_dims=spatial_dims,
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation)
    raise ValueError(
        f"Unknown aux_head_mode: {mode!r}. Valid: 'linear' | 'conv'.")


class UNet3D(nn.Module):
    """通用 UNet。aux_seg_supervision 在 num_stem_fusion_views>1 时构建逐 FOV 辅助头

    forward 返回：
      - eval / 无 aux：Tensor 或 [main, ds1, ...]（深监督）。
      - train + aux：{"main": ..., "aux": [...]}，aux 上采样到 main 尺寸。
    """

    def __init__(
        self,
        encoder              : Encoder,
        decoder,
        num_fg_classes       : int,
        deep_supervision     : bool = False,
        spatial_dims         : int = 3,
        aux_seg_supervision  : bool = False,
        aux_head_mode        : str = "linear",
        norm_type            : str = "instance",
        norm_groups          : int = 8,
        activation           : str = "leakyrelu",
        aux_head_out_channels: List[int] = None):
        super().__init__()

        self.encoder          = encoder
        self.decoder          = decoder
        self.num_fg_classes   = num_fg_classes
        self.deep_supervision = deep_supervision
        self.spatial_dims     = spatial_dims

        # 主头读最高分辨率 decoder 特征；stem_stride>1 时 forward 末尾上采回输入分辨率。DS 头保留各自分辨率。
        self.stem_stride = getattr(encoder, "stem_stride", 1)
        self.seg_head    = SegmentationHead(  # TODO 这里单层是否过于简单？是否可以选择多层？像ConvSegmentationHead
            decoder.out_channels[-1], num_fg_classes, spatial_dims=spatial_dims)

        # DS 头按分辨率递减：forward 返回 [main, 2nd, ..., lowest]，对齐 DeepSupervisionLoss。
        if deep_supervision:
            self.ds_heads = nn.ModuleList()
            for ch in reversed(decoder.out_channels[:-1]):
                self.ds_heads.append(ConvSegmentationHead(ch, num_fg_classes, spatial_dims=spatial_dims))  # TODO 是否需要多层conv

        # Aux 头镜像 stem 拓扑：Plan A (shared_stem/multi_stem_proj) 全部读 dec[-1]；
        # Plan C (hierarchical) aux k 读 dec[-1-k]（对齐 view k 注入的 encoder 深度）。aux 上采到 main 尺寸。
        n_views = int(getattr(encoder, "num_stem_fusion_views", 1))
        fusion  = str(getattr(encoder, "stem_fusion_mode", "shared_stem"))
        
        if bool(aux_seg_supervision) and n_views <= 1:  # 对多分辨率监督
            raise ValueError(
                f"UNet3D got aux_seg_supervision=True but encoder."
                f"num_stem_fusion_views={n_views} (<=1). The caller should "
                "gate this via ModelTopology.aux_seg_active "
                "(= aux_seg_supervision AND n_views > 1).")
        self.aux_seg_supervision = bool(aux_seg_supervision)
        self.aux_n_views         = n_views
        # aux_feat_indices[k-1]：aux 头 k 用的 decoder 特征索引（避免运行时分支）。
        self.aux_feat_indices = []
        self.aux_heads        = nn.ModuleList()
        # aux 通道默认 num_fg；native-depth 路径显式传 [num_fg*D_1, ..., num_fg*D_{K-1}]。
        n_aux_expected = max(n_views - 1, 0) if self.aux_seg_supervision else 0
        if aux_head_out_channels is None:
            self.aux_head_out_channels: List[int] = ([num_fg_classes] * n_aux_expected)
        else:
            if len(aux_head_out_channels) != n_aux_expected:
                raise ValueError(
                    f"aux_head_out_channels length "
                    f"({len(aux_head_out_channels)}) must equal "
                    f"n_views - 1 ({n_aux_expected}).")
            self.aux_head_out_channels = [int(c) for c in aux_head_out_channels]
        # 'conv' 含 norm/act；'linear' 仅 1×1。
        def _build_head(in_ch: int, out_ch: int) -> nn.Module:
            return build_head(
                mode         = aux_head_mode,
                in_ch        = in_ch,
                num_classes  = out_ch,
                spatial_dims = spatial_dims,
                norm_type    = norm_type,
                norm_groups  = norm_groups,
                activation   = activation)
        self.aux_head_mode = aux_head_mode
        if self.aux_seg_supervision:
            n_dec = len(decoder.out_channels)
            if fusion == "hierarchical":
                # dec_features：[-1] 最高分辨率，[-1-k] 镜像 stage k。
                if n_views > n_dec:
                    raise ValueError(
                        f"aux_seg_supervision (hierarchical) requires "
                        f"len(decoder.out_channels) >= n_views; got "
                        f"n_dec={n_dec}, n_views={n_views}.")
                for k in range(1, n_views):
                    feat_idx = n_dec - 1 - k
                    self.aux_feat_indices.append(feat_idx)
                    self.aux_heads.append(
                        _build_head(decoder.out_channels[feat_idx],
                                    self.aux_head_out_channels[k - 1]))
            else:
                in_ch = decoder.out_channels[-1]
                for k in range(1, n_views):
                    self.aux_feat_indices.append(n_dec - 1)  # 用最后一个dec特征
                    self.aux_heads.append(
                        _build_head(in_ch, self.aux_head_out_channels[k - 1]))

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, Any]]:
        """x: (B, in_channels, *spatial)；2.5D 多 FOV 为 (B, n_views*D, H, W)。"""
        enc_features = self.encoder(x)
        dec_features = self.decoder(enc_features)
        target_size  = x.shape[2:]

        main_out = self.seg_head(dec_features[-1])
        if main_out.shape[2:] != target_size:
            raise RuntimeError(
                f"Main seg head output size mismatch: "
                f"got {tuple(main_out.shape[2:])}, expected {tuple(target_size)}. "
                f"Check stem_stride / encoder downsampling vs input spatial dims.")

        # aux 仅训练时输出
        aux_outs: List[torch.Tensor] = []
        if self.aux_seg_supervision and self.training:
            for head, feat_idx in zip(self.aux_heads, self.aux_feat_indices):
                ao = head(dec_features[feat_idx])
                if ao.shape[2:] != target_size:
                    raise RuntimeError(
                        f"Aux seg head (feat_idx={feat_idx}) output size mismatch: "
                        f"got {tuple(ao.shape[2:])}, expected {tuple(target_size)}. "
                        f"Check stem_stride / encoder downsampling vs input spatial dims.")
                aux_outs.append(ao)

        if self.deep_supervision and self.training:
            # dec_features=[low,...,high]；main 用 [-1]，DS 头用 [-2]..[low]。
            main_path: Union[torch.Tensor, List[torch.Tensor]] = [main_out]
            for i, head in enumerate(self.ds_heads):
                main_path.append(head(dec_features[-2 - i]))
        else:
            main_path = main_out

        if aux_outs:
            return {"main": main_path, "aux": aux_outs}
        return main_path

    def param_count(self) -> Dict[str, int]:
        enc   = sum(p.numel() for p in self.encoder.parameters())
        dec   = sum(p.numel() for p in self.decoder.parameters())
        head  = sum(p.numel() for p in self.seg_head.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {"encoder": enc, "decoder": dec, "seg_head": head, "total": total}