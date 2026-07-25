"""Validation and top-level Config for gentask."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

from taskcore.config.core import MonitorConfig

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
    # 训练监测仪表盘（公用工程件，见 taskcore.monitor）。
    monitor: MonitorConfig = field(default_factory=MonitorConfig)

    def sync(self) -> None:
        """同步跨子配置的对应字段。

        所有"模型几何派生量"（``in_channels`` / ``spatial_dims``）由
        ``taskcore.models.topology.build_topology(self)`` 一次性算出，写入
        ``ModelConfig`` 的私有 backing 字段（对外是只读 property）。本方法仅保留
        "非派生"职责（``num_classes`` 推断、``z_boundary_mode`` 自动升级、resenc
        preset）。
        """
        if self.data.label_values and self.data.num_classes == 0:
            self.data.num_classes = len(self.data.label_values)

        # 与 taskcore Config.sync 对齐：stretch 已废弃，训练侧恒走 edge-pad，
        # 仅推理生效会在薄卷上造成训推几何 desync。
        if self.data.z_boundary_mode == "stretch":
            logger.warning(
                "data.z_boundary_mode='stretch' is deprecated: training-side "
                "extraction always uses edge-pad geometry, so stretch would "
                "only take effect at inference and desync train/infer "
                "geometry for volumes thinner than the patch depth. "
                "Auto-upgraded to 'edge_pad'.")
            self.data.z_boundary_mode = "edge_pad"

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
        from taskcore.models.topology import build_topology
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
        _kernels = ("trilinear", "area", "nearest", "gauss", "tri")
        _require(
            str(t.sr_kernel).lower() in _kernels,
            f"Invalid task.sr_kernel: {t.sr_kernel!r}. Valid: {list(_kernels)}.")
        _require(t.sr_noise_std >= 0.0, "task.sr_noise_std must be >= 0.")
        _require(
            str(t.sr_sampling).lower() in ("blur", "decimate"),
            f"Invalid task.sr_sampling: {t.sr_sampling!r}. Valid: 'blur' | 'decimate'.")
        is_blur = str(t.sr_sampling).lower() == "blur"
        if str(t.sr_kernel).lower() in ("gauss", "tri"):
            _require(
                is_blur,
                "SSP kernels ('gauss'/'tri') require sr_sampling=='blur'; "
                f"got sr_sampling={t.sr_sampling!r}.")
        if t.sr_kernel_pool:
            _require(
                is_blur,
                "task.sr_kernel_pool only applies to sr_sampling=='blur' "
                "(decimate ignores the down kernel); got "
                f"sr_sampling={t.sr_sampling!r}.")
            for k in t.sr_kernel_pool:
                _require(
                    str(k).lower() in _kernels,
                    f"Invalid task.sr_kernel_pool entry: {k!r}. "
                    f"Valid: {list(_kernels)}.")
        if t.sr_noise_std_range:
            _require(
                len(t.sr_noise_std_range) == 2,
                "task.sr_noise_std_range must be [lo, hi]; got "
                f"{t.sr_noise_std_range}.")
            lo, hi = float(t.sr_noise_std_range[0]), float(t.sr_noise_std_range[1])
            _require(
                0.0 <= lo <= hi,
                f"task.sr_noise_std_range must satisfy 0 <= lo <= hi; got {t.sr_noise_std_range}.")
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
                not self.model.unet.lift_2_5d_to_3d,
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
                str(t.sampler).lower() in (
                    "edm_heun", "edm_euler", "ddpm", "ddim"),
                f"Invalid task.sampler: {t.sampler!r}. Valid: "
                "'edm_heun' | 'edm_euler' | 'ddpm' | 'ddim'.")
            _require(t.num_train_timesteps >= 1, "task.num_train_timesteps must be >= 1.")
            _require(t.sample_steps >= 1, "task.sample_steps must be >= 1.")
            _require(0.0 < t.sigma_min < t.sigma_max, "require 0 < sigma_min < sigma_max.")
            _require(t.sigma_data > 0.0, "task.sigma_data must be > 0.")
            _require(t.rho > 0.0, "task.rho must be > 0.")
            # zscore 数据 std=1，EDM 预条件的 sigma_data=0.5（minmax 默认）
            # 会系统性偏置 c_skip/c_out/loss 权重；提示改为 1.0。
            if (str(self.data.normalize).lower() == "zscore"
                    and str(t.parameterization).lower() == "edm"
                    and abs(float(t.sigma_data) - 0.5) < 1e-9):
                logger.warning(
                    "data.normalize='zscore' (data std≈1.0) with EDM default "
                    "task.sigma_data=0.5 (tuned for minmax [0,1]); consider "
                    "task.sigma_data=1.0 to match the data scale.")
            # ddpm_eps 采样必须用 ddpm/ddim；edm_* 仅适用于 edm 预条件。
            if str(t.parameterization).lower() == "ddpm_eps":
                _require(
                    str(t.sampler).lower() in ("ddpm", "ddim"),
                    "parameterization='ddpm_eps' requires sampler in ('ddpm','ddim').")
            else:
                _require(
                    str(t.sampler).lower() in ("edm_heun", "edm_euler"),
                    "parameterization='edm' requires sampler in "
                    "('edm_heun','edm_euler').")

    def _validate_sisr_arch(self, arch: str) -> None:
        """经典 SISR（EDSR/RCAN，post-upsampling）的组合约束。"""
        _require(
            self.is_generation
            and str(self.task.algorithm).lower() == "regression",
            f"model.arch={arch!r} requires task.type='generation' with "
            f"task.algorithm='regression'.")
        _require(
            len(self.data.multi_res_scales) <= 1,
            f"model.arch={arch!r} does not support multi-view "
            f"(data.multi_res_scales); use a single view.")
        _require(
            not self.model.unet.lift_2_5d_to_3d,
            f"model.arch={arch!r} does not support lift_2_5d_to_3d.")
        _require(
            not self.model.deep_supervision,
            f"model.arch={arch!r} does not support deep_supervision "
            "(single full-resolution head).")
        _require(
            not self.data.cond_dirs,
            f"model.arch={arch!r} does not support cond_dirs: conditioning "
            "images live on the HR grid while the net input is the true LR "
            "grid.")
        _require(
            self.model.sisr.channels > 0 and self.model.sisr.num_blocks > 0
            and self.model.sisr.num_groups > 0,
            "sisr.channels/sisr.num_blocks/sisr.num_groups must be > 0.")
        # 模型空间轴 patch 尺寸须能被逐轴倍率整除（LR = patch/scale 为整）。
        sdims = 2 if self.data.patch_mode == "2_5d" else 3
        per_axis = list(self.task.sr_scale_per_axis)
        scales = ([int(s) for s in per_axis] if per_axis
                  else [int(self.task.sr_scale)] * sdims)
        if self.data.patch_mode != "whole":
            patch = [int(p) for p in self.data.patch_size]
            spatial = patch[1:] if sdims == 2 else patch
            for n, sc in zip(spatial, scales):
                _require(
                    n % sc == 0,
                    f"model.arch={arch!r} requires model-space patch sizes "
                    f"divisible by sr scales; got size {n} vs scale {sc}.")

    def _validate_model(self) -> None:
        """model.* 架构选项与逐级拓扑长度校验。"""
        arch = str(self.model.arch).lower()
        _require(
            arch in ("unet", "adm", "edm2", "edsr", "rcan"),
            f"Invalid model.arch: {arch!r}. "
            f"Valid: 'unet' | 'adm' | 'edm2' | 'edsr' | 'rcan'.")
        if arch in ("edsr", "rcan"):
            self._validate_sisr_arch(arch)
        elif arch == "unet":
            _require(
                self.model.unet.backbone in ("resnet", "convnext"),
                f"Invalid backbone: {self.model.unet.backbone}")
            _require(
                self.model.unet.norm_type in ("batch", "instance", "group"),
                f"Invalid norm: {self.model.unet.norm_type}")
            _require(
                self.model.unet.activation in (
                "relu", "leakyrelu", "gelu", "swish",
            ),
                f"Invalid activation: {self.model.unet.activation}")
            _require(
                self.model.unet.downsample_mode in (
                "conv", "maxpool", "avgpool", "blurpool", "pixelunshuffle",
            ),
                f"Invalid downsample_mode: {self.model.unet.downsample_mode}")
            _require(
                self.model.unet.upsample_mode in (
                "transpose", "trilinear", "nearest", "pixelshuffle",
                "carafe", "dysample",
            ),
                f"Invalid upsample_mode: {self.model.unet.upsample_mode}")
            _require(
                self.model.unet.skip_mode in ("cat", "add"),
                f"Invalid skip_mode: {self.model.unet.skip_mode}")
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
        # stem / stage 长度与 core 共享（arch allowlist 故意分叉：gen 多 edsr/rcan）。
        from taskcore.config.section_validators import (
            validate_encoder_decoder_stage_lengths,
            validate_stem_modes,
        )
        validate_stem_modes(self)
        # 仅 arch=='unet' 使用以下 backbone/block/decoder/r2plus1d/ResEnc/注意力选项。
        if arch == "unet":
            _require(
                self.model.unet.attention_type in (
                "none", "se", "eca", "cbam", "coord",
            ),
                f"Invalid attention_type: {self.model.unet.attention_type}")
            _require(
                self.model.unet.decoder_type in ("unet", "unetpp", "unet3p"),
                f"Invalid decoder_type: {self.model.unet.decoder_type}")
            _require(
                self.model.unet.unet3p_cat_channels > 0,
                "unet3p_cat_channels must be > 0")
            _require(
                self.model.unet.block_type in (
                "basic", "preact", "bottleneck", "r2plus1d"),
                f"Invalid block_type: {self.model.unet.block_type}")
            # r2plus1d 需 D 为真空间轴；2.5D 下 D 在通道轴，拒绝。
            if self.model.unet.block_type == "r2plus1d":
                _require(
                    self.model.spatial_dims == 3,
                    "model.block_type='r2plus1d' requires spatial_dims=3; "
                    "incompatible with 2.5D (D folded into channel axis). "
                    "Use patch_mode='z_axis' for Plan A on z-slab data.")
            # 与 sync()._apply_resenc_preset / core 校验一致：大小写不敏感。
            _require(
                str(self.model.resenc_preset or "none").lower()
                in ("none", "s", "m", "l", "xl"),
                f"Invalid resenc_preset: {self.model.resenc_preset}")
        validate_encoder_decoder_stage_lengths(self)

    def _validate_loss(self) -> None:
        """loss.* 校验。"""
        _require(
            all(w >= 0 for w in self.loss.deep_supervision_weights),
            f"Invalid deep_supervision_weights: {self.loss.deep_supervision_weights}")
        _require(
            all(w >= 0 for w in self.loss.aux_recon_weights),
            f"Invalid aux_recon_weights: {self.loss.aux_recon_weights}")

    def _validate_data(self) -> None:
        """data.* patch/multi-res/keep_native 校验。

        委托 ``taskcore`` 同名实现（duck-typed），再追加 gen 专属 ``cond_*``。
        """
        from taskcore.config.core import Config as _CoreConfig
        _CoreConfig._validate_data(self)
        if self.data.cond_dirs:
            _require(
                self.data.cond_normalize in ("minmax", "zscore"),
                f"Invalid data.cond_normalize: {self.data.cond_normalize!r}")

    def _validate_2_5d(self) -> None:
        """2.5D 几何不变式 —— 委托 core，跳过 seg 通道布局（gen 含 cond 扩展）。"""
        from taskcore.config.core import Config as _CoreConfig
        _CoreConfig._validate_2_5d(self, check_channel_layout=False)

    def _validate_augment(self) -> None:
        """augment.* —— 与 core 单一真相源对齐。"""
        from taskcore.config.core import Config as _CoreConfig
        _CoreConfig._validate_augment(self)

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
        _require(
            int(self.train.val_full_volume_max) >= 0,
            "train.val_full_volume_max must be >= 0; got "
            f"{self.train.val_full_volume_max}.")
        if bool(self.train.val_full_volume):
            _require(
                self.is_generation,
                "train.val_full_volume is only supported for "
                "task.type='generation'.")
        # prefetch_to_gpu 依赖 pinned host 内存才能真正异步（否则正确但无收益）。
        if bool(self.train.prefetch_to_gpu) and not bool(self.data.pin_memory):
            logger.warning(
                "train.prefetch_to_gpu=True but data.pin_memory=False: "
                "async H2D copy degrades to synchronous, prefetch overlap "
                "has no effect. Enable data.pin_memory.")

    def _validate_predict(self) -> None:
        """predict.* 校验。"""
        _require(
            bool(self.predict.output_dir),
            "predict.output_dir must be non-empty.")
        _require(
            str(self.predict.input_grid).lower() in ("hr", "lr"),
            f"Invalid predict.input_grid: {self.predict.input_grid!r}. "
            "Valid: 'hr' | 'lr'.")
        _require(
            0.0 <= float(self.predict.overlap) < 1.0,
            f"predict.overlap must be in [0, 1); got {self.predict.overlap}.")
        _require(
            str(self.predict.blend).lower() in ("gaussian", "uniform"),
            f"Invalid predict.blend: {self.predict.blend!r}. "
            "Valid: 'gaussian' | 'uniform'.")
        _require(
            int(self.predict.batch_size) >= 1,
            f"predict.batch_size must be >= 1; got {self.predict.batch_size}.")
        tz = float(self.predict.target_z_spacing)
        _require(tz >= 0.0,
                 f"predict.target_z_spacing must be >= 0; got {tz}.")
        if tz > 0.0:
            _require(
                str(self.predict.input_grid).lower() == "lr",
                "predict.target_z_spacing requires predict.input_grid='lr' "
                "(spacing-aware resampling only applies to true-LR input).")
            _require(
                int(self.model.spatial_dims) == 3,
                "predict.target_z_spacing requires a 3D model space "
                "(z axis degraded); 2.5D folds z into channels.")
            _require(
                str(self.model.arch).lower() not in ("edsr", "rcan"),
                "predict.target_z_spacing is incompatible with "
                "post-upsampling SISR (fixed upsample-head factors).")


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
        from taskcore.models.topology import build_topology
        return list(build_topology(self).per_view_depths)
