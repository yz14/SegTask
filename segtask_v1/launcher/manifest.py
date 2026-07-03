"""每个模式（2.5D / 3D）下"真正可调且生效"的参数清单（effective-parameter manifest）。

设计原则（对应用户需求"只展示该模式下有效、可调、能生效的参数"）：

* **剔除派生只读量**：``model.in_channels`` / ``model.spatial_dims`` 是
  ``@property``，``schema`` 本就不含；``data.num_classes`` 由 ``label_values``
  自动推断（``sync()`` 覆盖），故不暴露。
* **剔除该模式下无实质效果的项**：
    - 2.5D 下 ``model.anisotropic_pooling``（z 已折进通道，仅作用 H/W，"无实质变化"）
      与 ``model.multirf_axes``（自动等价 'hw'）被排除；
    - 3D 下 2.5D 专属项（``aux_seg_supervision`` / ``stem_fusion_mode`` /
      ``lift_2_5d_to_3d`` / ``adm_*`` / ``edm2_*`` / ``slice_loss_reduction`` /
      ``z_interleave_*`` / ``keep_native_view_depth`` 等）被排除。
* **条件生效项用 ``depends_on`` 联动**：仅当依赖满足时控件才启用/可见（如
  ``multirf_*`` 仅 ``multirf_enabled=True``、SGD 项仅 ``optimizer=='sgd'``）。
* **SSL 段整体排除**：自监督预训练是独立 task（``python -m segtask_v1.pretrain``），
  不属分割 train/predict 范畴。

每个字段项是 ``Field``：``ref="section.field"``，可选 ``enum`` / ``depends_on``。
``depends_on`` 是条件列表（AND 关系），每个条件支持：
    in / contains / truthy / nonempty / len_gt 五种谓词（见前端 ``conditionMet``）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

Cond = Dict[str, Any]


@dataclass
class Field:
    ref: str                              # "section.field"
    enum: Optional[List[str]] = None      # 提供则渲染为下拉
    depends_on: List[Cond] = field(default_factory=list)
    label: str = ""                       # 覆盖默认显示名（默认用字段名）
    readonly: bool = False                # 锁定显示（如 2.5D 的 patch_mode）


@dataclass
class Group:
    title: str                            # 分组标题
    section_tag: str                      # 所属 task 过滤标签（见 TASK_SECTIONS）
    fields: List[Field] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 枚举（来自 config.py 注释 + validate() 白名单，单一真相源的人工固化）。
# ---------------------------------------------------------------------------
ENUMS: Dict[str, List[str]] = {
    "data.normalize":              ["minmax", "zscore"],
    "data.cache_mode":             ["none", "memory"],
    "data.z_boundary_mode":        ["stretch", "edge_pad"],
    "augment.wmap_interp_mode":    ["nearest", "bilinear"],
    "model.backbone":              ["resnet", "convnext", "mednext"],
    "model.block_type":            ["basic", "preact", "bottleneck", "r2plus1d"],
    "model.resenc_preset":         ["none", "S", "M", "L", "XL"],
    "model.norm_type":             ["batch", "instance", "group"],
    "model.activation":            ["relu", "leakyrelu", "gelu", "swish"],
    "model.attention_type":        ["none", "se", "eca", "cbam", "coord"],
    "model.aux_head_mode":         ["linear", "conv"],
    "model.aux_topo_target":       ["centerline", "distance"],
    "model.aux_topo_head_mode":    ["linear", "conv"],
    "model.stem_mode":             ["conv3", "conv7", "dual", "patch2", "patch4"],
    "model.decoder_type":          ["unet", "unetpp", "unet3p"],
    "model.downsample_mode":       ["conv", "maxpool", "avgpool", "blurpool",
                                    "pixelunshuffle"],
    "model.upsample_mode":         ["transpose", "trilinear", "nearest",
                                    "pixelshuffle", "carafe", "dysample"],
    "model.skip_mode":             ["cat", "add"],
    "model.multirf_mode":          ["split", "parallel"],
    "model.multirf_fusion":        ["concat_proj", "sum", "se"],
    "model.multirf_axes":          ["all", "hw"],
    "model.selfattn_type":         ["softmax", "linear"],
    "loss.name":                   ["dice", "bce", "focal", "tversky", "gdl",
                                    "focal_tversky", "lovasz", "cldice",
                                    "dice_bce", "dice_focal", "dice_tversky",
                                    "focal_plus_tversky", "dice_cldice",
                                    "dice_focal_tversky", "dice_lovasz",
                                    "bce_lovasz", "gdl_bce", "gdl_focal"],
    "loss.gdl_weight_type":        ["square", "simple", "uniform"],
    "loss.slice_loss_reduction":   ["per_slice", "per_volume"],
    "loss.aux_topo_loss":          ["auto", "dice", "bce", "smooth_l1", "mse"],
    "train.optimizer":             ["adam", "adamw", "sgd"],
    "train.scheduler":             ["cosine", "cosine_warm_restarts", "poly",
                                    "step", "plateau", "one_cycle"],
    "train.amp_dtype":             ["float16", "bfloat16", "auto"],
    "train.compile_mode":          ["none", "default", "reduce-overhead",
                                    "max-autotune"],
    "train.save_best_criterion":   ["loss", "dice", "dice+surface_dice", "iou",
                                    "mcc", "min_dice", "balanced"],
    "train.save_best_preset":      ["", "lung", "vessel", "airway", "bone_multi",
                                    "lymph_node", "lesion_small", "oar_multi",
                                    "heart_chamber", "bone_lung_combined"],
    "train.val_metric_mode":       ["medium", "high"],
    "predict.blend_mode":          ["gaussian", "average"],
    "predict.adabn_mode":          ["global", "per_volume"],
    "vis.flows":                   None,  # list 控件，不下拉
}


# ---------------------------------------------------------------------------
# task → 显示哪些分组（按 section_tag 过滤）。
# ---------------------------------------------------------------------------
# 推理沿用训练所用配置（model/data 几何必须与 checkpoint 一致，不可在此改动），
# 故 predict 页仅暴露"推理专属旋钮"(predict 组) + CLI 运行参数(run 组)，其余字段
# 由所选基础配置（训练配置）继承。
TASK_SECTIONS: Dict[str, List[str]] = {
    "train":   ["data", "augment", "model", "loss", "train", "vismon"],
    "predict": ["predict", "run"],
}


# ---------------------------------------------------------------------------
# 常用 depends_on 条件构造器（保持可读）。
# ---------------------------------------------------------------------------
def _in(ref: str, values: List[str]) -> Cond:
    return {"f": ref, "in": values}


def _truthy(ref: str) -> Cond:
    return {"f": ref, "truthy": True}


def _nonempty(ref: str) -> Cond:
    return {"f": ref, "nonempty": True}


def _len_gt(ref: str, n: int) -> Cond:
    return {"f": ref, "len_gt": n}


def _contains(ref: str, token: str) -> Cond:
    return {"f": ref, "contains": token}


# ---------------------------------------------------------------------------
# 各 section 的字段清单构造（参数化以复用于 2.5D / 3D）。
# ---------------------------------------------------------------------------
def _data_fields(mode: str) -> List[Field]:
    is25 = mode == "2_5d"
    fs: List[Field] = [
        Field("data.npz_dir"),
        Field("data.npz_suffix"),
        Field("data.npz_auto_build"),
        Field("data.image_dir"),
        Field("data.label_dir"),
        Field("data.image_suffix"),
        Field("data.label_suffix"),
        Field("data.bbox_dir"),
        Field("data.bbox_suffix"),
        Field("data.region_weight_dir"),
        Field("data.region_weight_suffix"),
        Field("data.npz_dir_secondary"),
        Field("data.mix_ratio", depends_on=[_nonempty("data.npz_dir_secondary")]),
        Field("data.exclude_list"),
        Field("data.label_values"),
        Field("data.patch_size"),
    ]
    if is25:
        fs.append(Field("data.patch_mode", enum=["2_5d"], readonly=True))
    else:
        fs.append(Field("data.patch_mode", enum=["z_axis", "cubic", "whole"]))
    fs += [
        Field("data.multi_res_scales"),
        Field("data.aug_oversample_ratio"),
        Field("data.intensity_min"),
        Field("data.intensity_max"),
        Field("data.normalize"),
        Field("data.global_mean", depends_on=[_in("data.normalize", ["zscore"])]),
        Field("data.global_std", depends_on=[_in("data.normalize", ["zscore"])]),
        Field("data.val_ratio"),
        Field("data.split_seed"),
        Field("data.stratified_split"),
        Field("data.batch_size"),
        Field("data.num_workers"),
        Field("data.pin_memory"),
        Field("data.persistent_workers"),
        Field("data.prefetch_factor"),
        Field("data.foreground_oversample_ratio"),
        Field("data.samples_per_volume"),
        Field("data.cache_mode"),
        Field("data.cache_max_volumes"),
    ]
    if is25:
        # z_boundary_mode 在 2.5D 始终生效；keep_native_view_depth 为 2.5D 专属。
        fs.append(Field("data.z_boundary_mode"))
        fs.append(Field("data.keep_native_view_depth",
                        depends_on=[_len_gt("data.multi_res_scales", 1),
                                    _truthy("model.aux_seg_supervision")]))
    else:
        # z_boundary_mode 仅 z_axis 生效；keep_native_multi_res 为 3D（z_axis/cubic）专属。
        fs.append(Field("data.z_boundary_mode",
                        depends_on=[_in("data.patch_mode", ["z_axis"])]))
        fs.append(Field("data.keep_native_multi_res",
                        depends_on=[_in("data.patch_mode", ["z_axis", "cubic"]),
                                    _len_gt("data.multi_res_scales", 1)]))
    return fs


def _augment_fields() -> List[Field]:
    # 增强全部为 image+label GPU 变换，2.5D / 3D 同样生效。
    names = [
        "enabled",
        "random_flip_prob", "random_flip_axes",
        "random_affine_prob", "random_rotate_range", "random_scale_range",
        "elastic_deform_prob", "elastic_deform_sigma", "elastic_deform_alpha",
        "grid_dropout_prob", "grid_dropout_ratio", "grid_dropout_holes",
        "random_brightness_prob", "random_brightness_range",
        "random_contrast_prob", "random_contrast_range",
        "random_gamma_prob", "random_gamma_range",
        "gaussian_noise_prob", "gaussian_noise_std",
        "gaussian_blur_prob", "gaussian_blur_sigma",
        "simulate_lowres_prob", "simulate_lowres_zoom",
        "wmap_interp_mode",
    ]
    dep = [_truthy("augment.enabled")]
    return [Field("augment.enabled")] + [
        Field(f"augment.{n}", depends_on=dep) for n in names[1:]]


def _model_fields(mode: str) -> List[Field]:
    is25 = mode == "2_5d"
    unet = [_in("model.arch", ["unet"])]
    resnet = unet + [_in("model.backbone", ["resnet"])]
    fs: List[Field] = []
    # 架构族：2.5D 支持 unet/adm/edm2；3D 仅 unet（adm/edm2 要求 patch_mode='2_5d'）。
    if is25:
        fs.append(Field("model.arch", enum=["unet", "adm", "edm2"]))
    else:
        fs.append(Field("model.arch", enum=["unet"], readonly=True))
    fs += [
        Field("model.encoder_channels"),
        Field("model.backbone", depends_on=unet),
        Field("model.blocks_per_level", depends_on=unet),
        Field("model.encoder_blocks_per_stage", depends_on=unet),
        Field("model.decoder_blocks_per_stage", depends_on=unet),
        Field("model.resenc_preset", depends_on=unet),
    ]
    # block_type：仅 resnet；r2plus1d 需 spatial_dims=3（3D 或 2.5D+lift）。
    if is25:
        fs.append(Field("model.block_type",
                        enum=["basic", "preact", "bottleneck"],
                        depends_on=resnet))
    else:
        fs.append(Field("model.block_type", depends_on=resnet))
    fs += [
        Field("model.norm_type", depends_on=unet),
        Field("model.norm_groups", depends_on=unet + [_in("model.norm_type", ["group"])]),
        Field("model.activation", depends_on=unet),
        Field("model.dropout", depends_on=unet),
        Field("model.attention_type", depends_on=unet),
        Field("model.se_reduction", depends_on=unet + [_in("model.attention_type", ["se"])]),
        Field("model.skip_attention", depends_on=unet),
        Field("model.deep_supervision", depends_on=unet),
        Field("model.stem_mode"),
        Field("model.decoder_type", depends_on=unet),
        Field("model.unet3p_cat_channels", depends_on=unet + [_in("model.decoder_type", ["unet3p"])]),
        Field("model.downsample_mode", depends_on=unet),
        # 2.5D 默认 spatial_dims=2，carafe/dysample 为 3D 专属上采样，故剔除。
        Field("model.upsample_mode",
              enum=(["transpose", "trilinear", "nearest", "pixelshuffle"]
                    if is25 else None),
              depends_on=unet),
        Field("model.upsample_norm_act", depends_on=unet + [_in("model.upsample_mode", ["trilinear", "nearest"])]),
        Field("model.downsample_strides", depends_on=unet),
        Field("model.skip_mode", depends_on=unet),
        # 梯度检查点（所有 backbone 通用）。
        Field("model.grad_checkpointing", depends_on=unet),
        # ConvNeXt 专属。
        Field("model.drop_path_rate", depends_on=unet + [_in("model.backbone", ["convnext"])]),
        Field("model.convnext_layer_scale_init", depends_on=unet + [_in("model.backbone", ["convnext"])]),
        Field("model.convnext_downsample_lnfirst", depends_on=unet + [_in("model.backbone", ["convnext"])]),
        # MedNeXt 专属（仅 backbone=='mednext'）。
        Field("model.mednext_expand_ratio", depends_on=unet + [_in("model.backbone", ["mednext"])]),
        Field("model.mednext_kernel_size", depends_on=unet + [_in("model.backbone", ["mednext"])]),
        # 拓扑辅助头（2D/3D 通用，仅 unet）。
        Field("model.aux_topo_head", depends_on=unet),
        Field("model.aux_topo_target", depends_on=unet + [_truthy("model.aux_topo_head")]),
        Field("model.aux_topo_head_mode", depends_on=unet + [_truthy("model.aux_topo_head")]),
    ]
    # 各向异性下采样：2.5D 下"无实质变化" → 仅 3D 暴露。
    if not is25:
        fs.append(Field("model.anisotropic_pooling",
                        depends_on=unet + [_in("model.decoder_type", ["unet"])]))
    # MultiRF（仅 resnet）。multirf_axes 仅 3D 有区别（2.5D 自动 hw）。
    fs += [
        Field("model.multirf_enabled", depends_on=resnet),
        Field("model.multirf_dilations", depends_on=resnet + [_truthy("model.multirf_enabled")]),
        Field("model.multirf_mode", depends_on=resnet + [_truthy("model.multirf_enabled")]),
        Field("model.multirf_fusion", depends_on=resnet + [_truthy("model.multirf_enabled")]),
    ]
    if not is25:
        fs.append(Field("model.multirf_axes",
                        depends_on=resnet + [_truthy("model.multirf_enabled")]))
    fs += [
        Field("model.multirf_encoder_stages", depends_on=resnet + [_truthy("model.multirf_enabled")]),
        Field("model.multirf_decoder_stages", depends_on=resnet + [_truthy("model.multirf_enabled")]),
        Field("model.multirf_branch_norm_act", depends_on=resnet + [_truthy("model.multirf_enabled")]),
        # SelfAttention（仅 resnet+unet，2D/3D 通用）。
        Field("model.selfattn_enabled", depends_on=resnet),
        Field("model.selfattn_type", depends_on=resnet + [_truthy("model.selfattn_enabled")]),
        Field("model.selfattn_num_heads", depends_on=resnet + [_truthy("model.selfattn_enabled")]),
        Field("model.selfattn_head_dim", depends_on=resnet + [_truthy("model.selfattn_enabled")]),
        Field("model.selfattn_zero_init", depends_on=resnet + [_truthy("model.selfattn_enabled")]),
        Field("model.selfattn_encoder_stages", depends_on=resnet + [_truthy("model.selfattn_enabled")]),
        Field("model.selfattn_decoder_stages", depends_on=resnet + [_truthy("model.selfattn_enabled")]),
    ]
    # 2.5D 专属：多 FOV 上下文融合 / 辅助分割监督 / Plan A lift / ADM / EDM2。
    if is25:
        multi = [_len_gt("data.multi_res_scales", 1)]
        fs += [
            Field("model.stem_fusion_mode", depends_on=multi),
            Field("model.aux_seg_supervision", depends_on=multi),
            Field("model.aux_head_mode", depends_on=multi + [_truthy("model.aux_seg_supervision")]),
            Field("model.lift_2_5d_to_3d"),
            # ADM（arch=='adm'）。
            Field("model.adm_attention_levels", depends_on=[_in("model.arch", ["adm"])]),
            Field("model.adm_num_heads", depends_on=[_in("model.arch", ["adm"])]),
            Field("model.adm_num_head_channels", depends_on=[_in("model.arch", ["adm"])]),
            Field("model.adm_linear_attention_levels", depends_on=[_in("model.arch", ["adm"])]),
            Field("model.adm_linear_attention_num_heads", depends_on=[_in("model.arch", ["adm"])]),
            Field("model.adm_linear_attention_head_dim", depends_on=[_in("model.arch", ["adm"])]),
            # EDM2（arch=='edm2'）。
            Field("model.edm2_attention_levels", depends_on=[_in("model.arch", ["edm2"])]),
            Field("model.edm2_channels_per_head", depends_on=[_in("model.arch", ["edm2"])]),
            Field("model.edm2_res_balance", depends_on=[_in("model.arch", ["edm2"])]),
            Field("model.edm2_attn_balance", depends_on=[_in("model.arch", ["edm2"])]),
            Field("model.edm2_concat_balance", depends_on=[_in("model.arch", ["edm2"])]),
            Field("model.edm2_clip_act", depends_on=[_in("model.arch", ["edm2"])]),
        ]
    return fs


def _loss_fields(mode: str) -> List[Field]:
    is25 = mode == "2_5d"
    fs = [
        Field("loss.name"),
        Field("loss.compound_weights"),
        Field("loss.class_weights"),
        Field("loss.region_weights"),
        Field("loss.dice_smooth", depends_on=[_contains("loss.name", "dice")]),
        Field("loss.dice_squared", depends_on=[_contains("loss.name", "dice")]),
        Field("loss.focal_alpha", depends_on=[_contains("loss.name", "focal")]),
        Field("loss.focal_gamma", depends_on=[_contains("loss.name", "focal")]),
        Field("loss.tversky_alpha", depends_on=[_contains("loss.name", "tversky")]),
        Field("loss.tversky_beta", depends_on=[_contains("loss.name", "tversky")]),
        Field("loss.batch_dice"),
        Field("loss.ignore_empty"),
        Field("loss.gdl_weight_type", depends_on=[_contains("loss.name", "gdl")]),
        Field("loss.gdl_w_max", depends_on=[_contains("loss.name", "gdl")]),
        Field("loss.focal_tversky_gamma", depends_on=[_contains("loss.name", "focal_tversky")]),
        Field("loss.lovasz_per_sample", depends_on=[_contains("loss.name", "lovasz")]),
        Field("loss.cldice_iter", depends_on=[_contains("loss.name", "cldice")]),
        Field("loss.cldice_smooth", depends_on=[_contains("loss.name", "cldice")]),
        Field("loss.deep_supervision_weights", depends_on=[_truthy("model.deep_supervision")]),
        Field("loss.aux_topo_weight", depends_on=[_truthy("model.aux_topo_head")]),
        Field("loss.aux_topo_iter", depends_on=[_truthy("model.aux_topo_head")]),
        Field("loss.aux_topo_loss", depends_on=[_truthy("model.aux_topo_head")]),
    ]
    # slice_loss_reduction / aux_supervision_weights 仅 2.5D 生效。
    if is25:
        fs.append(Field("loss.slice_loss_reduction"))
        fs.append(Field("loss.aux_supervision_weights",
                        depends_on=[_truthy("model.aux_seg_supervision")]))
    return fs


def _train_fields() -> List[Field]:
    sgd = [_in("train.optimizer", ["sgd"])]
    return [
        Field("train.epochs"),
        Field("train.optimizer"),
        Field("train.lr"),
        Field("train.weight_decay"),
        Field("train.momentum", depends_on=sgd),
        Field("train.nesterov", depends_on=sgd),
        Field("train.scheduler"),
        Field("train.warmup_epochs"),
        Field("train.warmup_lr"),
        Field("train.cosine_min_lr", depends_on=[_in("train.scheduler", ["cosine", "cosine_warm_restarts", "one_cycle"])]),
        Field("train.cosine_restart_period", depends_on=[_in("train.scheduler", ["cosine_warm_restarts"])]),
        Field("train.cosine_restart_mult", depends_on=[_in("train.scheduler", ["cosine_warm_restarts"])]),
        Field("train.poly_power", depends_on=[_in("train.scheduler", ["poly"])]),
        Field("train.step_size", depends_on=[_in("train.scheduler", ["step"])]),
        Field("train.step_gamma", depends_on=[_in("train.scheduler", ["step"])]),
        Field("train.plateau_patience", depends_on=[_in("train.scheduler", ["plateau"])]),
        Field("train.plateau_factor", depends_on=[_in("train.scheduler", ["plateau"])]),
        Field("train.grad_accum_steps"),
        Field("train.grad_clip_norm"),
        Field("train.use_amp"),
        Field("train.amp_dtype", depends_on=[_truthy("train.use_amp")]),
        Field("train.compile_mode"),
        Field("train.use_ema"),
        Field("train.ema_decay", depends_on=[_truthy("train.use_ema")]),
        Field("train.output_dir"),
        Field("train.save_every"),
        Field("train.save_keep_last"),
        Field("train.save_best_criterion"),
        Field("train.surface_dice_tolerance", depends_on=[_in("train.save_best_criterion", ["dice+surface_dice", "balanced"])]),
        Field("train.surface_dice_weight", depends_on=[_in("train.save_best_criterion", ["dice+surface_dice"])]),
        Field("train.save_best_preset"),
        Field("train.val_metric_mode"),
        Field("train.early_stopping"),
        Field("train.log_every"),
        Field("train.val_every"),
        Field("train.vis_every"),
        Field("train.seed"),
        Field("train.deterministic"),
        Field("train.resume"),
        Field("train.pretrain"),
        Field("train.pretrain_strict", depends_on=[_nonempty("train.pretrain")]),
        Field("train.pretrain_load_ema", depends_on=[_nonempty("train.pretrain")]),
        Field("train.gpus"),
        Field("train.ddp_find_unused_parameters", depends_on=[_len_gt("train.gpus", 1)]),
        Field("train.ddp_master_port", depends_on=[_len_gt("train.gpus", 1)]),
        Field("train.ddp_scale_dataloader_per_rank", depends_on=[_len_gt("train.gpus", 1)]),
    ]


def _predict_fields(mode: str) -> List[Field]:
    is25 = mode == "2_5d"
    fs = [
        Field("predict.z_overlap"),
        Field("predict.blend_mode"),
        Field("predict.batch_size"),
        Field("predict.tta_flip"),
        Field("predict.tta_batch_size", depends_on=[_truthy("predict.tta_flip")]),
        Field("predict.threshold"),
        # 注：predict.output_dir 始终被 predict.py 以 --output 或派生值覆盖，属
        # "被覆盖字段"，故不在此暴露——输出目录改用 run 组的 --output。
        Field("predict.save_probabilities"),
        Field("predict.adabn_enabled"),
        Field("predict.adabn_mode", depends_on=[_truthy("predict.adabn_enabled")]),
        Field("predict.adabn_num_volumes", depends_on=[_truthy("predict.adabn_enabled"), _in("predict.adabn_mode", ["global"])]),
    ]
    # z 轴交错多流推理：仅 2.5D。
    if is25:
        fs += [
            Field("predict.z_interleave_enabled"),
            Field("predict.z_interleave_thresholds", depends_on=[_truthy("predict.z_interleave_enabled")]),
            Field("predict.z_interleave_factors", depends_on=[_truthy("predict.z_interleave_enabled")]),
        ]
    return fs


def _vismon_fields() -> List[Field]:
    return [
        Field("vis.enabled"),
        Field("vis.output_dir", depends_on=[_truthy("vis.enabled")]),
        Field("vis.filename", depends_on=[_truthy("vis.enabled")]),
        Field("vis.flows", depends_on=[_truthy("vis.enabled")]),
        Field("vis.trace_shapes", depends_on=[_truthy("vis.enabled")]),
        Field("vis.max_detail_params", depends_on=[_truthy("vis.enabled")]),
        Field("monitor.enabled"),
        Field("monitor.output_dir", depends_on=[_truthy("monitor.enabled")]),
        Field("monitor.filename", depends_on=[_truthy("monitor.enabled")]),
        Field("monitor.update_every", depends_on=[_truthy("monitor.enabled")]),
        Field("monitor.auto_reload_seconds", depends_on=[_truthy("monitor.enabled")]),
        Field("monitor.run_name", depends_on=[_truthy("monitor.enabled")]),
        Field("monitor.compare_runs", depends_on=[_truthy("monitor.enabled")]),
        Field("monitor.health_monitor", depends_on=[_truthy("monitor.enabled")]),
        Field("monitor.health_grad_norm_when_no_clip", depends_on=[_truthy("monitor.enabled")]),
        Field("monitor.health_update_ratio", depends_on=[_truthy("monitor.enabled")]),
    ]


# ---------------------------------------------------------------------------
# 推理 CLI-only 运行参数（不在 YAML 内；映射为 predict.py 命令行开关）。
# ---------------------------------------------------------------------------
@dataclass
class RunArg:
    name: str                 # 前端字段名
    flag: str                 # 命令行 flag，如 "--checkpoint"
    control: str              # bool / str
    default: Any
    tooltip: str
    required: bool = False


def predict_run_args() -> List[RunArg]:
    return [
        RunArg("checkpoint", "--checkpoint", "str", "",
               "Checkpoint 路径；留空默认 <train.output_dir>/best_model.pth。"),
        RunArg("input", "--input", "str", "",
               "待推理的 NIfTI 文件或目录；留空默认 cfg.data.image_dir。"),
        RunArg("output", "--output", "str", "",
               "输出目录；留空默认 <input_parent>/<task_name>_pred。"),
        RunArg("bbox", "--bbox", "str", "",
               "ROI bbox 文件/目录；填 '' 显式禁用；留空回退 cfg.data.bbox_dir。"),
        RunArg("weights", "--weights", "enum", "auto",
               "加载哪份权重：auto / ema / online。"),
        RunArg("precision", "--precision", "enum", "auto",
               "推理精度：auto（跟随 train.amp_dtype）/ fp32 / bf16 / fp16。"),
        RunArg("no_recursive", "--no-recursive", "bool", False,
               "不递归搜索 --input 的子目录。"),
    ]


RUN_ARG_ENUMS: Dict[str, List[str]] = {
    "weights":   ["auto", "ema", "online"],
    "precision": ["auto", "fp32", "bf16", "fp16"],
}


# ---------------------------------------------------------------------------
# 组装每个模式的分组清单。
# ---------------------------------------------------------------------------
def build_groups(mode: str) -> List[Group]:
    """返回某模式（'2_5d' / '3d'）下的有序分组清单。"""
    return [
        Group("数据 Data", "data", _data_fields(mode)),
        Group("数据增强 Augmentation", "augment", _augment_fields()),
        Group("模型 Model", "model", _model_fields(mode)),
        Group("损失 Loss", "loss", _loss_fields(mode)),
        Group("训练 Training", "train", _train_fields()),
        Group("推理 Inference", "predict", _predict_fields(mode)),
        Group("可视化与监测 Vis & Monitor", "vismon", _vismon_fields()),
    ]
