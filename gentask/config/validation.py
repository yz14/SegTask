"""Validation and top-level Config for gentask."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

from .dataclasses import (
    AugConfig, ConfigError, DataConfig, LossConfig, ModelConfig, PredictConfig,
    TaskConfig, TrainConfig, _require,
)

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """顶层配置，聚合所有子配置。"""

    data   : DataConfig    = field(default_factory=DataConfig)
    model  : ModelConfig   = field(default_factory=ModelConfig)
    loss   : LossConfig    = field(default_factory=LossConfig)
    train  : TrainConfig   = field(default_factory=TrainConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)
    task   : TaskConfig    = field(default_factory=TaskConfig)
    augment: AugConfig     = field(default_factory=AugConfig)

    def sync(self) -> None:
        """同步跨子配置的对应字段。

        所有"模型几何派生量"（``in_channels`` / ``spatial_dims``）由
        ``gentask.models.topology.build_topology(self)`` 一次性算出，写入
        ``ModelConfig`` 的私有 backing 字段（对外是只读 property）。本方法仅保留
        "非派生"职责（``num_classes`` 推断、``z_boundary_mode`` 自动升级、resenc
        preset）。
        """
        if self.data.label_values and self.data.num_classes == 0:
            self.data.num_classes = len(self.data.label_values)

        # z_boundary_mode 自动升级（lazy multi-res 隐式要求 edge_pad）—— 此为
        # *data 侧* 副作用，不属 ModelTopology 范畴。
        n_views = max(len(self.data.multi_res_scales), 1)
        if self.data.patch_mode == "2_5d":
            if (self.data.keep_native_view_depth and n_views > 1
                    and self.data.z_boundary_mode != "edge_pad"):
                logger.info(
                    "keep_native_view_depth=True implies z_boundary_mode='edge_pad'; "
                    "auto-upgraded from %r.", self.data.z_boundary_mode)
                self.data.z_boundary_mode = "edge_pad"
        elif (self.data.keep_native_multi_res
                and self.data.patch_mode == "z_axis"
                and n_views > 1
                and self.data.z_boundary_mode != "edge_pad"):
            logger.info(
                "keep_native_multi_res=True implies z_boundary_mode="
                "'edge_pad'; auto-upgraded from %r.",
                self.data.z_boundary_mode)
            self.data.z_boundary_mode = "edge_pad"

        # 单一真相源：所有通道/几何派生量在 build_topology 中一次算齐。
        # 局部 import 避免 models 包顶层依赖 config（消除循环 import 风险）。
        from ..models.topology import build_topology
        topo = build_topology(self)
        self.model._in_channels = topo.in_channels
        self.model._spatial_dims = topo.spatial_dims

        self._apply_resenc_preset()

    def _apply_resenc_preset(self) -> None:
        """将 model.resenc_preset 展开为逐级 block 数；用户显式传入优先。"""
        mc     = self.model
        preset = (mc.resenc_preset or "none").lower()
        if preset == "none":
            return
        if mc.encoder_blocks_per_stage and mc.decoder_blocks_per_stage:
            return

        n_levels = len(mc.encoder_channels)
        templates = {
            "s":  [1, 2, 2, 2, 2, 2],
            "m":  [1, 3, 4, 6, 6, 6],
            "l":  [1, 3, 4, 6, 6, 6, 6],
            "xl": [1, 4, 6, 8, 8, 10, 10, 10],
        }
        if preset not in templates:
            return

        tpl = templates[preset]
        # 裁剪或拓展（重复最深级计数）以匹配 n_levels。
        if n_levels <= len(tpl):
            enc_blocks = tpl[:n_levels]
        else:
            enc_blocks = tpl + [tpl[-1]] * (n_levels - len(tpl))

        if not mc.encoder_blocks_per_stage:
            mc.encoder_blocks_per_stage = enc_blocks
        if not mc.decoder_blocks_per_stage:
            mc.decoder_blocks_per_stage = [1] * (n_levels - 1)

    def validate(self) -> None:
        """校验配置一致性（按 section 拆分；非法配置抛 ConfigError）。"""
        self._validate_model()
        self._validate_loss()
        self._validate_data()
        self._validate_2_5d()
        self._validate_train()
        self._validate_predict()
        self._validate_task()
        self._validate_augment()
        if not self.is_generation and self.data.num_classes < 2:
            logger.warning("num_classes=%d < 2, will auto-detect from data.",
                           self.data.num_classes)

    def _validate_task(self) -> None:
        """task.* 校验：仅 type=='generation' 时检查生成相关字段。"""
        t = self.task
        ttype = str(t.type).lower()
        _require(
            ttype in ("segmentation", "generation"),
            f"Invalid task.type: {t.type!r}. Valid: 'segmentation' | 'generation'.")
        if ttype != "generation":
            return
        _require(
            str(t.algorithm).lower() in ("regression", "diffusion"),
            f"Invalid task.algorithm: {t.algorithm!r}. Valid: 'regression' | 'diffusion'.")
        _require(
            str(t.degradation).lower() == "superres",
            f"Invalid task.degradation: {t.degradation!r}. Only 'superres' supported.")
        _require(t.out_channels >= 1, f"task.out_channels must be >= 1; got {t.out_channels}.")
        _require(t.sr_scale >= 1, f"task.sr_scale must be >= 1; got {t.sr_scale}.")
        _require(
            str(t.sr_kernel).lower() in ("trilinear", "area", "nearest"),
            f"Invalid task.sr_kernel: {t.sr_kernel!r}. Valid: 'trilinear' | 'area' | 'nearest'.")
        _require(t.sr_noise_std >= 0.0, "task.sr_noise_std must be >= 0.")
        _require(
            str(t.sr_sampling).lower() in ("blur", "decimate"),
            f"Invalid task.sr_sampling: {t.sr_sampling!r}. Valid: 'blur' | 'decimate'.")
        _require(
            str(t.sr_kernel_up).lower() in ("trilinear", "area", "nearest"),
            f"Invalid task.sr_kernel_up: {t.sr_kernel_up!r}. "
            "Valid: 'trilinear' | 'area' | 'nearest'.")
        if t.sr_scale_per_axis:
            # 退化在模型空间轴上施加：用 topology 派生的 model.spatial_dims
            # （2.5D+lift 为 3），与 build_degradation 的调用方保持一致。
            sdims = int(self.model.spatial_dims)
            _require(
                len(t.sr_scale_per_axis) == sdims,
                f"task.sr_scale_per_axis length must equal model.spatial_dims "
                f"({sdims}); got {t.sr_scale_per_axis}.")
            _require(
                all(int(s) >= 1 for s in t.sr_scale_per_axis),
                f"each task.sr_scale_per_axis entry must be >= 1; got {t.sr_scale_per_axis}.")
            _require(
                any(int(s) > 1 for s in t.sr_scale_per_axis) or t.sr_noise_std > 0.0,
                "task.sr_scale_per_axis must have at least one axis with scale > 1 "
                f"(else the degradation is a no-op); got {t.sr_scale_per_axis}.")
        if str(t.algorithm).lower() == "regression":
            _require(
                str(t.recon_loss).lower() in ("charbonnier", "l1", "mse"),
                f"Invalid task.recon_loss: {t.recon_loss!r}. Valid: 'charbonnier' | 'l1' | 'mse'.")
            _require(t.ssim_weight >= 0.0 and t.grad_weight >= 0.0,
                     "task.ssim_weight / grad_weight must be >= 0.")
            _require(t.ssim_window >= 3 and t.ssim_window % 2 == 1,
                     f"task.ssim_window must be odd and >= 3; got {t.ssim_window}.")
        else:  # diffusion
            _require(
                str(self.model.arch).lower() in ("adm", "edm2"),
                "task.algorithm='diffusion' requires model.arch in "
                "('adm','edm2') (paper-faithful \u03c3/timestep conditioning); "
                f"got {self.model.arch!r}.")
            _require(
                not self.model.lift_2_5d_to_3d,
                "task.algorithm='diffusion' is 2.5D-only (ADM/EDM2 nets are 2D); "
                "incompatible with model.lift_2_5d_to_3d=True.")
            _require(
                not self.model.deep_supervision,
                "model.deep_supervision is only supported for generation "
                "algorithm='regression' (diffusion uses adm/edm2 nets without "
                "multi-scale heads); set deep_supervision=False or algorithm='regression'.")
            _require(
                str(t.parameterization).lower() in ("edm", "ddpm_eps"),
                f"Invalid task.parameterization: {t.parameterization!r}. Valid: 'edm' | 'ddpm_eps'.")
            _require(
                str(t.beta_schedule).lower() in ("linear", "cosine"),
                f"Invalid task.beta_schedule: {t.beta_schedule!r}. Valid: 'linear' | 'cosine'.")
            _require(
                str(t.sampler).lower() in ("edm_heun", "ddpm", "ddim"),
                f"Invalid task.sampler: {t.sampler!r}. Valid: 'edm_heun' | 'ddpm' | 'ddim'.")
            _require(t.num_train_timesteps >= 1, "task.num_train_timesteps must be >= 1.")
            _require(t.sample_steps >= 1, "task.sample_steps must be >= 1.")
            _require(0.0 < t.sigma_min < t.sigma_max, "require 0 < sigma_min < sigma_max.")
            _require(t.sigma_data > 0.0, "task.sigma_data must be > 0.")
            _require(t.rho > 0.0, "task.rho must be > 0.")
            # ddpm_eps 采样必须用 ddpm/ddim；edm_heun 仅适用于 edm 预条件。
            if str(t.parameterization).lower() == "ddpm_eps":
                _require(
                    str(t.sampler).lower() in ("ddpm", "ddim"),
                    "parameterization='ddpm_eps' requires sampler in ('ddpm','ddim').")
            else:
                _require(
                    str(t.sampler).lower() == "edm_heun",
                    "parameterization='edm' requires sampler='edm_heun'.")

    def _validate_model(self) -> None:
        """model.* 架构选项与逐级拓扑长度校验。"""
        arch = str(self.model.arch).lower()
        _require(
            arch in ("unet", "adm", "edm2"),
            f"Invalid model.arch: {arch!r}. Valid: 'unet' | 'adm' | 'edm2'.")
        if arch == "unet":
            _require(
                self.model.backbone in ("resnet", "convnext"),
                f"Invalid backbone: {self.model.backbone}")
            _require(
                self.model.norm_type in ("batch", "instance", "group"),
                f"Invalid norm: {self.model.norm_type}")
            _require(
                self.model.activation in (
                "relu", "leakyrelu", "gelu", "swish",
            ),
                f"Invalid activation: {self.model.activation}")
            _require(
                self.model.downsample_mode in (
                "conv", "maxpool", "avgpool", "blurpool", "pixelunshuffle",
            ),
                f"Invalid downsample_mode: {self.model.downsample_mode}")
            _require(
                self.model.upsample_mode in (
                "transpose", "trilinear", "nearest", "pixelshuffle",
                "carafe", "dysample",
            ),
                f"Invalid upsample_mode: {self.model.upsample_mode}")
            _require(
                self.model.skip_mode in ("cat", "add"),
                f"Invalid skip_mode: {self.model.skip_mode}")
        else:
            # ADM / EDM2 仅支持 2.5D + Plan A（shared_stem / multi_stem_proj）。
            _require(
                self.data.patch_mode == "2_5d",
                f"model.arch={arch!r} requires data.patch_mode='2_5d'; got {self.data.patch_mode!r}.")
            _require(
                self.model.stem_fusion_mode in (
                "shared_stem", "multi_stem_proj",
            ),
                f"model.arch={arch!r} only supports stem_fusion_mode in "
                f"('shared_stem','multi_stem_proj'); got {self.model.stem_fusion_mode!r}.")
        _require(
            self.model.spatial_dims in (2, 3),
            f"Invalid spatial_dims: {self.model.spatial_dims} (must be 2 or 3)")
        # 下面三项适用于所有 arch（ADM/EDM2 也读取）。
        _require(
            self.model.stem_mode in (
            "conv3", "conv7", "dual", "patch2", "patch4",
        ),
            f"Invalid stem_mode: {self.model.stem_mode}")
        _require(
            self.model.stem_fusion_mode in (
            "shared_stem", "multi_stem_proj", "hierarchical",
        ),
            f"Invalid stem_fusion_mode: {self.model.stem_fusion_mode!r}")
        _require(
            self.model.aux_head_mode in (
            "linear", "conv",
        ),
            f"Invalid aux_head_mode: {self.model.aux_head_mode!r}")
        # 仅 arch=='unet' 使用以下 backbone/block/decoder/r2plus1d/ResEnc/注意力选项。
        if arch == "unet":
            _require(
                self.model.attention_type in (
                "none", "se", "eca", "cbam", "coord",
            ),
                f"Invalid attention_type: {self.model.attention_type}")
            _require(
                self.model.decoder_type in ("unet", "unetpp", "unet3p"),
                f"Invalid decoder_type: {self.model.decoder_type}")
            _require(
                self.model.unet3p_cat_channels > 0,
                "unet3p_cat_channels must be > 0")
            _require(
                self.model.block_type in (
                "basic", "preact", "bottleneck", "r2plus1d"),
                f"Invalid block_type: {self.model.block_type}")
            # r2plus1d 需 D 为真空间轴；2.5D 下 D 在通道轴，拒绝。
            if self.model.block_type == "r2plus1d":
                _require(
                    self.model.spatial_dims == 3,
                    "model.block_type='r2plus1d' requires spatial_dims=3; "
                    "incompatible with 2.5D (D folded into channel axis). "
                    "Use patch_mode='z_axis' for Plan A on z-slab data.")
            _require(
                self.model.resenc_preset in ("none", "S", "M", "L", "XL"),
                f"Invalid resenc_preset: {self.model.resenc_preset}")
        # 逐级 block 数长度需与 encoder 深度对齐。
        n_levels = len(self.model.encoder_channels)
        ebps = self.model.encoder_blocks_per_stage
        dbps = self.model.decoder_blocks_per_stage
        if ebps:
            _require(
                len(ebps) == n_levels,
                f"encoder_blocks_per_stage must have {n_levels} entries "
                f"(= len(encoder_channels)); got {len(ebps)}")
            _require(
                all(b >= 1 for b in ebps),
                "encoder_blocks_per_stage entries must all be >= 1")
        if dbps:
            _require(
                len(dbps) == n_levels - 1,
                f"decoder_blocks_per_stage must have {n_levels - 1} entries "
                f"(= len(encoder_channels) - 1); got {len(dbps)}")
            _require(
                all(b >= 1 for b in dbps),
                "decoder_blocks_per_stage entries must all be >= 1")
        # 显式各向异性下采样 stride 校验（自动模式 anisotropic_pooling 无需在此校验）。
        sds = self.model.downsample_strides
        if sds:
            sd_dim = int(self.model.spatial_dims)
            _require(
                len(sds) == n_levels - 1,
                f"downsample_strides must have {n_levels - 1} entries "
                f"(= len(encoder_channels) - 1); got {len(sds)}")
            for s in sds:
                _require(
                    len(s) == sd_dim,
                    f"each downsample_strides entry must have "
                    f"spatial_dims={sd_dim} values; got {list(s)}")
                _require(
                    all(int(v) in (1, 2) for v in s),
                    f"downsample_strides values must be 1 or 2; got {list(s)}")

    def _validate_loss(self) -> None:
        """loss.* 校验。"""
        _require(
            all(w >= 0 for w in self.loss.deep_supervision_weights),
            f"Invalid deep_supervision_weights: {self.loss.deep_supervision_weights}")
        _require(
            all(w >= 0 for w in self.loss.aux_recon_weights),
            f"Invalid aux_recon_weights: {self.loss.aux_recon_weights}")

    def _validate_data(self) -> None:
        """data.* patch/multi-res/keep_native 校验。"""
        _require(
            len(self.data.patch_size) == 3,
            "patch_size must be [D, H, W]")
        _require(
            self.data.patch_mode in ("z_axis", "cubic", "whole", "2_5d"),
            f"Invalid patch_mode: {self.data.patch_mode}")
        _require(
            self.data.z_boundary_mode in ("stretch", "edge_pad"),
            f"Invalid z_boundary_mode: {self.data.z_boundary_mode!r}; "
            "expected 'stretch' or 'edge_pad'.")
        if self.data.patch_mode == "whole":
            _require(
                len(self.data.multi_res_scales) == 1 and self.data.multi_res_scales[0] == 1.0,
                f"whole-volume mode requires multi_res_scales=[1.0]; got {self.data.multi_res_scales}.")
        if self.data.keep_native_view_depth:
            _require(
                self.data.patch_mode == "2_5d",
                f"data.keep_native_view_depth=True requires patch_mode='2_5d'; got {self.data.patch_mode!r}.")
            _require(
                len(self.data.multi_res_scales) > 1,
                "data.keep_native_view_depth=True requires len(multi_res_scales) > 1; "
                f"got {self.data.multi_res_scales}.")
        if self.data.keep_native_multi_res:
            _require(
                self.data.patch_mode in ("z_axis", "cubic"),
                "data.keep_native_multi_res=True requires patch_mode in "
                "('z_axis','cubic'); got " + repr(self.data.patch_mode) + ". Use "
                "data.keep_native_view_depth for the 2.5D analogue.")
            _require(
                len(self.data.multi_res_scales) > 1,
                "data.keep_native_multi_res=True requires len(multi_res_scales) > 1; "
                f"got {self.data.multi_res_scales}.")
            _require(
                float(self.data.multi_res_scales[0]) == 1.0,
                "data.keep_native_multi_res=True requires multi_res_scales[0]==1.0; "
                f"got {self.data.multi_res_scales}.")
            _require(
                not self.data.keep_native_view_depth,
                "keep_native_multi_res and keep_native_view_depth are mutually exclusive (3D vs 2.5D analogues).")
            if self.data.patch_mode == "z_axis":
                _require(
                    self.data.z_boundary_mode == "edge_pad",
                    "keep_native_multi_res=True (z_axis) requires z_boundary_mode='edge_pad' "
                    f"(auto-set by sync()); got {self.data.z_boundary_mode!r}.")
        _require(
            self.data.aug_oversample_ratio >= 1.0,
            "aug_oversample_ratio must be >= 1.0")
        _require(
            len(self.data.multi_res_scales) >= 1,
            "multi_res_scales must have at least one scale (e.g. [1.0])")
        _require(
            all(s >= 1.0 for s in self.data.multi_res_scales),
            "All multi_res_scales must be >= 1.0")
        if self.data.cond_dirs:
            _require(
                self.data.cond_normalize in ("minmax", "zscore"),
                f"Invalid data.cond_normalize: {self.data.cond_normalize!r}")

    def _validate_2_5d(self) -> None:
        """2.5D 专属不变式（折叠通道 / lift / Plan A·C / aux 监督）。"""
        if self.data.patch_mode == "2_5d":
            _require(
                len(self.data.multi_res_scales) >= 1,
                "2.5D mode requires at least one entry in multi_res_scales.")
            _require(
                self.data.multi_res_scales[0] == 1.0,
                "2.5D mode requires multi_res_scales[0]==1.0 (view 0 = prediction target); "
                f"got {self.data.multi_res_scales}.")
            n_views = len(self.data.multi_res_scales)
            lift = bool(self.model.lift_2_5d_to_3d)
            if lift:
                _require(
                    self.model.spatial_dims == 3,
                    "lift_2_5d_to_3d=True requires model.spatial_dims=3 (auto-set by sync()).")
                _require(
                    self.model.in_channels == n_views,
                    f"lift_2_5d_to_3d=True requires in_channels == n_views ({n_views}); "
                    f"got {self.model.in_channels}.")
                _require(
                    not self.data.keep_native_view_depth,
                    "lift_2_5d_to_3d and keep_native_view_depth are mutually exclusive.")
                n_levels = len(self.model.encoder_channels)
                D = int(self.data.patch_size[0])
                req = 1 << (n_levels - 1)
                if D < req or D % req != 0:
                    raise ConfigError(
                        f"lift_2_5d_to_3d=True with {n_levels} encoder stages requires "
                        f"patch_size[0] (D={D}) divisible by 2**(n_levels-1)={req}. "
                        f"Increase D to a multiple of {req}, or reduce len(encoder_channels).")
            else:
                _require(
                    self.model.spatial_dims == 2,
                    "2.5D mode requires model.spatial_dims=2 (auto-set by sync()). "
                    "For Plan A 3D lift, set model.lift_2_5d_to_3d=True.")
            if (not lift) and self.data.keep_native_view_depth and n_views > 1:
                depths = self.per_view_depths
                _require(
                    len(depths) == n_views,
                    f"per_view_depths length must equal n_views ({n_views}); got {len(depths)}.")
                _require(
                    depths[0] == self.data.patch_size[0],
                    f"per_view_depths[0] must equal patch_size[0]={self.data.patch_size[0]}; got {depths[0]}.")
                from ..models.topology import build_topology
                _require(
                    build_topology(self).in_ch_per_view_list is not None,
                    "keep_native_view_depth=True requires in_ch_per_view_list "
                    "(derived by build_topology).")
            if self.model.aux_seg_supervision:
                _require(
                    n_views > 1,
                    "aux_seg_supervision=True requires n_views > 1; got 1.")
                if self.model.stem_fusion_mode == "hierarchical":
                    n_levels = len(self.model.encoder_channels)
                    _require(
                        n_views < n_levels,
                        f"aux_seg_supervision + hierarchical requires n_views < n_levels; "
                        f"got n_views={n_views}, n_levels={n_levels}.")

    def _validate_augment(self) -> None:
        """augment.* 校验：概率界、区间合法性、插值模式与翻转轴。"""
        a = self.augment
        probs = {
            "random_flip_prob": a.random_flip_prob,
            "random_affine_prob": a.random_affine_prob,
            "elastic_deform_prob": a.elastic_deform_prob,
            "grid_dropout_prob": a.grid_dropout_prob,
            "random_brightness_prob": a.random_brightness_prob,
            "random_contrast_prob": a.random_contrast_prob,
            "random_gamma_prob": a.random_gamma_prob,
            "gaussian_noise_prob": a.gaussian_noise_prob,
            "gaussian_blur_prob": a.gaussian_blur_prob,
            "simulate_lowres_prob": a.simulate_lowres_prob,
        }
        for name, p in probs.items():
            _require(0.0 <= float(p) <= 1.0,
                     f"augment.{name} must be in [0,1]; got {p}.")
        ranges = {
            "random_rotate_range": a.random_rotate_range,
            "random_scale_range": a.random_scale_range,
            "random_translate_range": a.random_translate_range,
            "random_brightness_range": a.random_brightness_range,
            "random_contrast_range": a.random_contrast_range,
            "random_gamma_range": a.random_gamma_range,
            "gaussian_blur_sigma": a.gaussian_blur_sigma,
            "simulate_lowres_zoom": a.simulate_lowres_zoom,
        }
        for name, r in ranges.items():
            _require(
                len(r) == 2 and float(r[0]) <= float(r[1]),
                f"augment.{name} must be [lo, hi] with lo <= hi; got {r}.")
        if a.random_rotate_range_per_axis is not None:
            _require(
                len(a.random_rotate_range_per_axis) == 3
                and all(len(rr) == 2 and float(rr[0]) <= float(rr[1])
                        for rr in a.random_rotate_range_per_axis),
                "augment.random_rotate_range_per_axis must be 3 pairs of "
                f"[lo, hi]; got {a.random_rotate_range_per_axis}.")
        _require(
            a.wmap_interp_mode in ("nearest", "bilinear"),
            f"Invalid augment.wmap_interp_mode: {a.wmap_interp_mode!r}; "
            "expected 'nearest' or 'bilinear'.")
        _require(
            all(int(ax) in (2, 3, 4) for ax in a.random_flip_axes),
            f"augment.random_flip_axes entries must be in (2,3,4) "
            f"(axes of (B,C,D,H,W)); got {a.random_flip_axes}.")
        _require(a.elastic_deform_sigma > 0.0,
                 "augment.elastic_deform_sigma must be > 0.")
        _require(a.elastic_deform_alpha >= 0.0,
                 "augment.elastic_deform_alpha must be >= 0.")
        _require(0.0 <= a.grid_dropout_ratio <= 1.0,
                 "augment.grid_dropout_ratio must be in [0,1].")
        _require(a.grid_dropout_holes >= 1,
                 "augment.grid_dropout_holes must be >= 1.")
        _require(a.gaussian_noise_std >= 0.0,
                 "augment.gaussian_noise_std must be >= 0.")

    def _validate_train(self) -> None:
        """train.* 优化器/调度器校验。"""
        _require(
            self.train.optimizer in ("adam", "adamw", "sgd"),
            f"Invalid optimizer: {self.train.optimizer}")
        _require(
            self.train.scheduler in (
            "cosine", "cosine_warm_restarts", "poly", "step", "plateau", "one_cycle",
        ),
            f"Invalid scheduler: {self.train.scheduler}")

    def _validate_predict(self) -> None:
        """predict.* 校验。"""
        _require(
            bool(self.predict.output_dir),
            "predict.output_dir must be non-empty.")


    @property
    def is_generation(self) -> bool:
        """True 当 task.type=='generation'。"""
        return str(self.task.type).lower() == "generation"

    @property
    def gen_out_channels(self) -> int:
        """生成任务每空间位置的输出通道数（CT 灰度=1）。"""
        return int(self.task.out_channels)

    @property
    def num_fg_classes(self) -> int:
        """Number of foreground classes (excluding background).

        Generation reuses this shared API to mean ``task.out_channels`` so the
        topology layer and model heads can stay uniform across modes.
        """
        if self.is_generation:
            return int(self.task.out_channels)
        return max(self.data.num_classes - 1, 1)

    @property
    def per_view_depths(self) -> List[int]:
        """2.5D 下每视图原生深度 D_k = round(D * s_k)，强制 D_0 = D。非 2.5D 返回空列表。

        R5：委托给 ``build_topology`` 以保持单一真相源；仅形状计算，不依赖
        ``data.keep_native_view_depth``，调用方自行根据该标志决定是否使用。
        """
        from ..models.topology import build_topology
        return list(build_topology(self).per_view_depths)
