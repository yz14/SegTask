"""Dataclass + YAML config. Each YAML file maps directly to nested dataclasses."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    """Data paths and preprocessing."""

    image_dir: str = ""
    label_dir: str = ""
    # 后缀：单值或候选列表（取首个存在）。例：".nii.gz" 或 [".nii.gz", "-seg.nii.gz"]。
    image_suffix: Union[str, List[str]] = ".nii.gz"
    label_suffix: Union[str, List[str]] = ".nii.gz"

    # 可选 ROI bbox 掩码目录；设置后按 bbox 裁剪。空=禁用。
    bbox_dir   : str = ""
    bbox_suffix: Union[str, List[str]] = ".nii.gz"

    # 可选逐样本区域权重目录（值 +1）。优先级：此目录 > loss.region_weights。
    region_weight_dir   : str = ""
    region_weight_suffix: Union[str, List[str]] = ".nii.gz"

    # 预生成 npz 包目录；设置后忽略上述 NIfTI 目录，避免多 worker gzip OOM。
    npz_dir   : str = ""
    npz_suffix: str = ".npz"

    # True=启动时自动调用 make_data 生成；False=要求手动预生成。
    npz_auto_build: bool = True

    # 样本排除清单路径（每行一个 pid）。空=不过滤。
    exclude_list: str = ""

    # 标签取值集（0=背景）。空=从数据自动探测。
    label_values: List[int] = field(default_factory=list)
    num_classes : int = 0  # 由 label_values 自动设置

    # 3D patch 尺寸 [D, H, W]。
    patch_size: List[int] = field(default_factory=lambda: [64, 128, 128])

    # Patch 抽取模式。示例："z_axis"（仅 z 滑块，H/W 全尺寸）、"2_5d"（D 折叠为通道驱动 2D UNet）。
    # 其他："cubic" 3 轴中心抽取；"whole" 整体 resize。
    patch_mode: str = "z_axis"

    # 增强过采样比：先抽 round(patch_size*ratio)，增强后中心裁回。1.0=禁用；affine/elastic 建议 1.4–1.5。
    aug_oversample_ratio: float = 1.0

    # 多分辨率 FOV：各 scale 同中心抽更宽 FOV，resize 后作额外输入通道。
    # 示例：[1.0] 单通道；[1.0, 1.5, 2.0] 3 通道。cubic 作用 3 轴，z_axis 仅 z 轴。
    multi_res_scales: List[float] = field(default_factory=lambda: [1.0])

    # 强度窗（CT HU）。
    intensity_min: float = -1024.0
    intensity_max: float = 1024.0
    # 归一化："minmax"→[0,1]；"zscore"→零均值单位方差。
    normalize  : str = "minmax"
    global_mean: float = 0.0
    global_std : float = 1.0

    # 训/验划分。
    val_ratio : float = 0.2
    split_seed: int = 42
    # 按首个前景类分层；样本太少时回退随机。
    stratified_split: bool = True

    # DataLoader。
    batch_size        : int = 2
    num_workers       : int = 4
    pin_memory        : bool = True
    persistent_workers: bool = True
    prefetch_factor   : int = 4

    # 前景过采样：中心点落在前景上的概率。
    foreground_oversample_ratio: float = 0.5

    # 每体积每 epoch 采样次数。
    samples_per_volume: int = 8

    # 缓存："none" 或 "memory"（每 worker LRU）。cache_max_volumes=0 不限（OOM 风险）。
    cache_mode       : str = "memory"
    cache_max_volumes: int = 1

    # z 轴边界填充（z_axis/2.5D）："stretch" 范围内拉伸；"edge_pad" 边缘复制后 resize（推荐）。
    z_boundary_mode: str = "edge_pad"

    # 2.5D 多视图保持原生深度。True 时 dataset 抽最大 FOV cube，trainer 按 D_k 中心裁；强制 edge_pad。
    # 仅在 patch_mode='2_5d' + len(scales)>1 + aux_seg_supervision=True 生效。
    aux_keep_native_d: bool = False

    # 3D 多 FOV 懒加载单 cube（z_axis/cubic）。True：dataset 发单 cube，trainer 逐视图裁剪/重采样。
    # 约束：scales[0]==1.0；与 aux_keep_native_d 互斥；z_axis 强制 edge_pad。
    keep_native_multi_res: bool = False


# ---------------------------------------------------------------------------
# Augmentation configuration
# ---------------------------------------------------------------------------
@dataclass
class AugConfig:
    """GPU 数据增强。所有空间变换逐样本独立。"""

    enabled: bool = True

    # --- 空间变换（image + label 同步） ---
    random_flip_prob: float = 0.2
    random_flip_axes: List[int] = field(default_factory=lambda: [2, 3, 4])

    # Affine：小角旋转 + 缩放，合成单次 grid_sample。
    random_affine_prob : float = 0.3
    random_rotate_range: List[float] = field(default_factory=lambda: [-15.0, 15.0])
    random_scale_range : List[float] = field(default_factory=lambda: [0.85, 1.15])

    # 弹性形变（B-spline 随机位移场）。
    elastic_deform_prob : float = 0.2
    elastic_deform_sigma: float = 5.0   # 位移平滑度
    elastic_deform_alpha: float = 7.0   # 位移幅度（voxel）

    # Grid dropout：随机遮挡矩形子区域。
    grid_dropout_prob : float = 0.0
    grid_dropout_ratio: float = 0.3
    grid_dropout_holes: int = 4

    # --- 强度变换（仅 image） ---
    random_brightness_prob : float = 0.3
    random_brightness_range: List[float] = field(default_factory=lambda: [-0.1, 0.1])

    random_contrast_prob : float = 0.3
    random_contrast_range: List[float] = field(default_factory=lambda: [0.8, 1.2])

    random_gamma_prob : float = 0.2
    random_gamma_range: List[float] = field(default_factory=lambda: [0.8, 1.2])

    gaussian_noise_prob: float = 0.15
    gaussian_noise_std : float = 0.05

    gaussian_blur_prob : float = 0.1
    gaussian_blur_sigma: List[float] = field(default_factory=lambda: [0.5, 1.5])

    # 模拟低分辨率（下采样后上采样）。
    simulate_lowres_prob: float = 0.1
    simulate_lowres_zoom: List[float] = field(default_factory=lambda: [0.5, 1.0])

    # weight_map 插值模式："nearest" 保持离散权重（默认）；"bilinear" 仅在连续手标 wmap 时用。  TODO 连续手标 wmap 时也要用nearest
    wmap_interp_mode: str = "nearest"


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """模型架构设置。"""

    # 架构族。示例："unet"（本项目 UNet，读下面 backbone/block/norm 等）、"adm"（ADM U-Net，仅 2.5D，读 adm_*）。
    # 还有 "edm2"（EDM2 U-Net，仅 2.5D，读 edm2_*）。
    arch: str = "unet"

    # Backbone："resnet" 或 "convnext"。
    backbone: str = "resnet"

    # 3 = 3D UNet（z_axis/cubic/whole）；2 = 2D UNet（2.5D）。
    spatial_dims: int = 3

    # 输入通道数（单模态 3D 为 1）。
    in_channels: int = 1

    # 每级 encoder 通道数，决定深度。例：[32, 64, 128, 256, 512] = 5 级。
    encoder_channels: List[int] = field(
        default_factory=lambda: [32, 64, 128, 256, 512])

    # 每级 block 数默认值（仅在 encoder/decoder_blocks_per_stage 都为空时使用）。
    blocks_per_level: int = 2

    # 残差块变体（仅 resnet）。示例："basic" 标准 ResNet；"r2plus1d" (1,3,3)+(3,1,1) 分解卷积（需 spatial_dims=3）。
    # 还有 "preact" / "bottleneck"。
    block_type: str = "basic"

    # 逐级 block 数（nnU-Net ResEncUNet 风格）。非空时长度须与网络深度匹配。
    encoder_blocks_per_stage: List[int] = field(default_factory=list)
    decoder_blocks_per_stage: List[int] = field(default_factory=list)

    # nnU-Net ResEnc 预设："none" | "S" | "M" | "L" | "XL"。非 none 且 *_blocks_per_stage 为空时 sync() 自填。
    resenc_preset: str = "none"

    # 归一化："batch" | "instance" | "group"。
    norm_type  : str = "instance"
    norm_groups: int = 8

    # 激活："relu" | "leakyrelu" | "gelu" | "swish"。
    activation: str = "leakyrelu"

    dropout: float = 0.0

    # 旧 SE 开关（仅 attention_type=='none' 生效）。
    use_se      : bool = False
    se_reduction: int = 16

    # 块内注意力："none" | "se" | "eca" | "cbam" | "coord"。
    attention_type: str = "none"

    # skip 连接上的 AttentionGate3D（Oktay 2018）。
    skip_attention: bool = False

    # 深度监督：多 decoder 级输出预测。
    deep_supervision: bool = False

    # 多 FOV 辅助分割监督（仅 2.5D + len(multi_res_scales)>1 生效）。主头预 view 0，辅助 view k 输出 (B, num_fg*D, H, W)。
    # 损失权重见 loss.aux_supervision_weights（空则默认 0.5^k）。单视图/3D 不生效。
    aux_seg_supervision: bool = False

    # 辅助头拓扑："linear" 单 Conv1×1（Plan A 推荐）；"conv" ConvNormAct(3×3)→Conv1×1（Plan C 推荐）。
    aux_head_mode: str = "linear"

    # Plan A 2.5D → 3D 提升（配合 block_type="r2plus1d"）。True 时 trainer 不折叠 D，模型输出 (B, num_fg, D, H, W)。
    # 与 data.aux_keep_native_d 互斥，仅在 2.5D 生效。
    lift_2_5d_to_3d: bool = False

    # Stem / patch-embed："conv3" | "conv7" | "dual" | "patch2" | "patch4"。patchN 降 N 倍分辨率（UNet3D 主输出加上采样）。
    stem_mode: str = "conv3"

    # 多 FOV 上下文融合（仅 2.5D + n_views>1）。示例："shared_stem"（全部过同一 stem）、"multi_stem_proj"（Plan A，逐视图 stem→cat→1×1）。
    # 还有 "hierarchical"（Plan C，aux k 注入 encoder 第 k 级）。3D 模式下忽略。
    context_fusion: str = "multi_stem_proj"

    # Decoder 拓扑："unet" 对称（默认）；"unetpp" 嵌套稠密；"unet3p" 全尺度 skip。
    decoder_type: str = "unet"

    # UNet3+ 各分支通道数（仅 decoder_type=="unet3p"）。
    unet3p_cat_channels: int = 64

    # 下采样："conv" | "maxpool" | "avgpool" | "blurpool" | "pixelunshuffle"。
    downsample_mode: str = "conv"

    # 上采样："transpose" | "trilinear" | "nearest" | "pixelshuffle" | "carafe" | "dysample"。
    upsample_mode: str = "transpose"

    # skip："cat" 或 "add"。
    skip_mode: str = "cat"

    # ConvNeXt: drop path / LayerScale / LN-first downsample。
    drop_path_rate: float = 0.0
    convnext_layer_scale_init: float = 1e-6  # <=0 禁用
    convnext_downsample_lnfirst: bool = True  # False 为通用 Downsample（消融用）

    # ---- ADM 专用（arch=="adm"） ----
    # 带多头自注意力的级索引（0=顶，L-1=bottleneck）。空=默认最深两级。
    adm_attention_levels: List[int] = field(default_factory=list)

    # 头数：仅在 adm_num_head_channels==-1 时使用。
    adm_num_heads: int = 4
    # !=-1 时 num_heads = channels // num_head_channels。
    adm_num_head_channels: int = -1

    # ---- LinearAttention（lucidrains 风格，可选） ----
    # 在指定级追加 Residual(PreNorm(LinearAttention))；O(N) 复杂度，可与 adm_attention_levels 叠加。
    adm_linear_attention_levels: List[int] = field(default_factory=list)
    adm_linear_attention_num_heads: int = 4
    adm_linear_attention_head_dim: int = 32

    # ---- EDM2 专用（arch=="edm2"） ----
    # 带自注意力的级索引。空=默认仅 bottleneck。
    edm2_attention_levels: List[int] = field(default_factory=list)

    # heads = out_ch // channels_per_head。
    edm2_channels_per_head: int = 64

    # MP 残差/注意力/skip-cat 平衡系数（论文 Eq. 88 / 103）。
    edm2_res_balance: float = 0.3
    edm2_attn_balance: float = 0.3
    edm2_concat_balance: float = 0.5

    # 输出激活裁剪（论文 6.4）；<=0 禁用。
    edm2_clip_act: float = 256.0


# ---------------------------------------------------------------------------
# Loss configuration
# ---------------------------------------------------------------------------
@dataclass
class LossConfig:
    """损失函数设置。输出为逐类独立 sigmoid，每个前景类产生 (B, 1, D, H, W) 二值输出。"""

    # 损失名：常用 "dice_bce" 或 "dice_focal"；其他选项见 validate() 白名单。
    name: str = "dice_bce"

    # 复合损失权重 [loss1_w, loss2_w]。
    compound_weights: List[float] = field(default_factory=lambda: [1.0, 1.0])

    # 逐类损失权重（空=均匀）；长度 = num_fg_classes。
    class_weights: List[float] = field(default_factory=list)

    # 逐区域空间权重：按 label 取值一个权重（含 bg）。例：[1.0, 2.0, 2.0, 1.0, 1.0] → label 1/2 位置损失×2。空=禁用。
    region_weights: List[float] = field(default_factory=list)

    # Dice 参数。
    dice_smooth: float = 1e-5
    dice_squared: bool = False

    # Focal 参数。
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # Tversky 参数（alpha=FP权重, beta=FN权重）。
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7

    # True：全 batch+空间上汇总 TP/分母后一次除（nnU-Net Dice 默认）。作用于 Dice/Tversky/FocalTversky/GDL。
    batch_dice: bool = False
    # 仅 per-sample：无 GT 的类从 dice 均值排除，避免空类≈1 掩盖错误。
    ignore_empty: bool = False

    # GDL 体积加权："square"（论文）| "simple"（w=1/Σt）| "uniform"（禁用）。
    gdl_weight_type: str = "square"
    gdl_w_max: float = 1.0e5  # 限住 1/volume。

    # Focal Tversky：(1-TI)^gamma，gamma≥1。
    focal_tversky_gamma: float = 4.0 / 3.0

    # Lovász-Hinge：True=逐 (B, C) 排序取均；False=批级排序（小 patch 更平滑）。
    lovasz_per_sample: bool = True

    # Soft clDice 骨架化迭代：2D 用 3，3D 取 3–10。
    cldice_iter: int = 3
    cldice_smooth: float = 1.0

    # 深度监督逐级权重。
    deep_supervision_weights: List[float] = field(
        default_factory=lambda: [1.0, 0.5, 0.25, 0.125])

    # 2.5D 损失聚合（仅 patch_mode=="2_5d"）："per_slice" 逐 slice 独立（空 slice Dice≈1 零梯度）；
    # "per_volume" 按整体在 (D,H,W) 上聚合（2.5D 推荐）。仅影响 Dice 系。
    slice_loss_reduction: str = "per_slice"

    # 2.5D 多 FOV 辅助头权重（仅 model.aux_seg_supervision=True）：长度 = n_views-1。空 = trainer 自填 0.5^k。
    aux_supervision_weights: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """训练循环设置。"""

    epochs: int = 200

    # 优化器："adam" | "adamw" | "sgd"。
    optimizer   : str = "adamw"
    lr          : float = 1e-3
    weight_decay: float = 1e-4
    momentum    : float = 0.99   # 仅 SGD
    nesterov    : bool = True    # 仅 SGD

    # 调度器："cosine" | "cosine_warm_restarts" | "poly" | "step" | "plateau" | "one_cycle"。
    scheduler    : str = "cosine"
    warmup_epochs: int = 5
    warmup_lr    : float = 1e-6
    cosine_min_lr: float = 1e-6
    # cosine_warm_restarts 重启周期 T_0 与倍率 T_mult。
    cosine_restart_period: int = 50
    cosine_restart_mult  : int = 2
    poly_power           : float = 0.9
    step_size            : int = 50
    step_gamma           : float = 0.1
    plateau_patience     : int = 10
    plateau_factor       : float = 0.5

    # 梯度累积（有效 batch = batch_size * accum_steps）。
    grad_accum_steps: int = 1

    # 梯度裁剪。
    grad_clip_norm: float = 12.0

    # AMP。amp_dtype 示例："float16"（需 GradScaler）、"bfloat16"（Ampere+，无需 scaler）。还有 "auto"（探测 BF16 否则回退 fp16）。
    use_amp  : bool = True
    amp_dtype: str = "float16"

    # torch.compile："none" | "default" | "reduce-overhead" | "max-autotune"。
    compile_mode: str = "none"

    # EMA。
    use_ema  : bool = True
    ema_decay: float = 0.999

    # Checkpoint 保存。
    output_dir      : str = "outputs"
    save_every      : int = 10
    # 选模标准: "loss" | "dice" | "dice+surface_dice"。覆盖下方 metric/mode（见 sync()）。
    save_best_criterion: str = "dice"
    # 内部解析字段（一般无需手动设）；sync() 会按 criterion 重写。
    save_best_metric: str = "mean_dice"
    save_best_mode  : str = "max"
    # Surface Dice 容差（像素，Chebyshev 邻域；0=严格表面 Dice）。
    surface_dice_tolerance: int = 1
    # 组合标准下 combined = (1-w)*dice + w*surface_dice。
    surface_dice_weight: float = 0.5

    # 提前停止（0=禁用）。
    early_stopping: int = 0

    # 日志。
    log_every: int = 10
    val_every: int = 1
    vis_every: int = 10

    seed         : int = 42
    deterministic: bool = False

    # Resume：从 checkpoint 完整恢复（model/EMA/optimizer/scheduler/scaler/epoch/RNG）。
    resume: str = ""

    # Pretrain：仅加载 model 权重作初始化。若同时设置了 resume 且存在则 pretrain 被忽略。
    pretrain: str = ""

    # strict 加载；默认 False 允许 head 形状不一致。
    pretrain_strict: bool = False

    # checkpoint 含 EMA shadow 时是否优先用 EMA 作初始。默认 False。
    pretrain_load_ema: bool = False


# ---------------------------------------------------------------------------
# Prediction / Inference configuration
# ---------------------------------------------------------------------------
@dataclass
class PredictConfig:
    """推理设置（z 轴滑动窗口）。"""

    # z 轴重叠比（0.0 = 不重叠，0.5 = 50%）。
    z_overlap: float = 0.5

    # 重叠区融合："gaussian" 或 "average"。
    blend_mode: str = "gaussian"

    # 推理 batch 大小。
    batch_size: int = 2

    # TTA flip。
    tta_flip: bool = False

    # sigmoid 二值化阈值。
    threshold: float = 0.5

    # 预测输出目录。
    output_dir: str = "predictions"

    # 是否保存概率图（在二值 mask 之外）。
    save_probabilities: bool = False

    # z 轴交错多流推理（仅 2.5D）：按 z 拆 k 个子体 (slices i,i+k,...)，独立推理后缝回原 z。
    # 动机：加宽 z 感受野。警告：子流表现为 k * z_spacing。
    z_interleave_enabled: bool = False

    # k 按物理 z 间距（mm）选择。thresholds 升序，factors 长度 = len(thresholds)+1（含 fallback）。
    # 默认：z<=1.0 → k=3；1.0<z<=1.5 → k=2；z>1.5 → k=1。
    z_interleave_thresholds: List[float] = field(
        default_factory=lambda: [1.0, 1.5])
    z_interleave_factors: List[int] = field(
        default_factory=lambda: [3, 2, 1])


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """顶层配置，聚合所有子配置。"""

    data   : DataConfig    = field(default_factory=DataConfig)
    augment: AugConfig     = field(default_factory=AugConfig)
    model  : ModelConfig   = field(default_factory=ModelConfig)
    loss   : LossConfig    = field(default_factory=LossConfig)
    train  : TrainConfig   = field(default_factory=TrainConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)

    def sync(self) -> None:
        """同步跨子配置的对应字段。

        R5：所有"模型几何派生量"（``in_channels`` / ``spatial_dims``）改由
        ``segtask_v1.models.topology.build_topology(self)`` 一次性算出再写回，
        以保持旧 yaml / 旧外部代码读 ``cfg.model.in_channels`` 不破坏。
        本方法仅保留"非派生"职责（``num_classes`` 推断、``z_boundary_mode``
        自动升级、resenc preset、best-criterion 映射）。
        """
        if self.data.label_values and self.data.num_classes == 0:
            self.data.num_classes = len(self.data.label_values)

        # z_boundary_mode 自动升级（lazy multi-res 隐式要求 edge_pad）—— 此为
        # *data 侧* 副作用，不属 ModelTopology 范畴。
        n_views = max(len(self.data.multi_res_scales), 1)
        if self.data.patch_mode == "2_5d":
            if (self.data.aux_keep_native_d and n_views > 1
                    and self.data.z_boundary_mode != "edge_pad"):
                logger.info(
                    "aux_keep_native_d=True implies z_boundary_mode='edge_pad'; "
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
        from .models.topology import build_topology
        topo = build_topology(self)
        self.model.in_channels = topo.in_channels
        self.model.spatial_dims = topo.spatial_dims

        self._apply_resenc_preset()
        self._resolve_save_best_criterion()

    def _resolve_save_best_criterion(self) -> None:
        """criterion → (save_best_metric, save_best_mode) 映射；显式覆盖低层字段。"""
        crit = str(self.train.save_best_criterion).lower().strip()
        mapping = {
            "loss": ("val_loss", "min"),
            "dice": ("mean_dice", "max"),
            "dice+surface_dice": ("mean_combined", "max"),
        }
        if crit in mapping:
            self.train.save_best_metric, self.train.save_best_mode = mapping[crit]

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
        """校验配置一致性。仅 arch=='unet' 时校验 backbone/block/norm 等旧字段。"""
        arch = str(getattr(self.model, "arch", "unet")).lower()
        assert arch in ("unet", "adm", "edm2"), (
            f"Invalid model.arch: {arch!r}. Valid: 'unet' | 'adm' | 'edm2'.")
        if arch == "unet":
            assert self.model.backbone in ("resnet", "convnext"), \
                f"Invalid backbone: {self.model.backbone}"
            assert self.model.norm_type in ("batch", "instance", "group"), \
                f"Invalid norm: {self.model.norm_type}"
            assert self.model.activation in (
                "relu", "leakyrelu", "gelu", "swish",
            ), f"Invalid activation: {self.model.activation}"
            assert self.model.downsample_mode in (
                "conv", "maxpool", "avgpool", "blurpool", "pixelunshuffle",
            ), f"Invalid downsample_mode: {self.model.downsample_mode}"
            assert self.model.upsample_mode in (
                "transpose", "trilinear", "nearest", "pixelshuffle",
                "carafe", "dysample",
            ), f"Invalid upsample_mode: {self.model.upsample_mode}"
            assert self.model.skip_mode in ("cat", "add"), \
                f"Invalid skip_mode: {self.model.skip_mode}"
        else:
            # ADM / EDM2 仅支持 2.5D + Plan A（shared_stem / multi_stem_proj）。
            assert self.data.patch_mode == "2_5d", (
                f"model.arch={arch!r} requires data.patch_mode='2_5d'; got {self.data.patch_mode!r}.")
            assert self.model.context_fusion in (
                "shared_stem", "multi_stem_proj",
            ), (
                f"model.arch={arch!r} only supports context_fusion in "
                f"('shared_stem','multi_stem_proj'); got {self.model.context_fusion!r}.")
        assert self.model.spatial_dims in (2, 3), \
            f"Invalid spatial_dims: {self.model.spatial_dims} (must be 2 or 3)"
        assert self.augment.wmap_interp_mode in ("nearest", "bilinear"), (
            f"Invalid augment.wmap_interp_mode: {self.augment.wmap_interp_mode!r} "
            "(expected 'nearest' or 'bilinear').")
        # 下面三项适用于所有 arch（ADM/EDM2 也读取）。
        assert self.model.stem_mode in (
            "conv3", "conv7", "dual", "patch2", "patch4",
        ), f"Invalid stem_mode: {self.model.stem_mode}"
        assert self.model.context_fusion in (
            "shared_stem", "multi_stem_proj", "hierarchical",
        ), f"Invalid context_fusion: {self.model.context_fusion!r}"
        assert getattr(self.model, "aux_head_mode", "linear") in (
            "linear", "conv",
        ), f"Invalid aux_head_mode: {self.model.aux_head_mode!r}"
        # 仅 arch=='unet' 使用以下 backbone/block/decoder/r2plus1d/ResEnc/注意力选项。
        if arch == "unet":
            assert self.model.attention_type in (
                "none", "se", "eca", "cbam", "coord",
            ), f"Invalid attention_type: {self.model.attention_type}"
            assert self.model.decoder_type in ("unet", "unetpp", "unet3p"), \
                f"Invalid decoder_type: {self.model.decoder_type}"
            assert self.model.unet3p_cat_channels > 0, \
                "unet3p_cat_channels must be > 0"
            assert self.model.block_type in (
                "basic", "preact", "bottleneck", "r2plus1d"), \
                f"Invalid block_type: {self.model.block_type}"
            # r2plus1d 需 D 为真空间轴；2.5D 下 D 在通道轴，拒绝。
            if self.model.block_type == "r2plus1d":
                assert self.model.spatial_dims == 3, (
                    "model.block_type='r2plus1d' requires spatial_dims=3; "
                    "incompatible with 2.5D (D folded into channel axis). "
                    "Use patch_mode='z_axis' for Plan A on z-slab data.")
            assert self.model.resenc_preset in ("none", "S", "M", "L", "XL"), \
                f"Invalid resenc_preset: {self.model.resenc_preset}"
        # 逐级 block 数长度需与 encoder 深度对齐。
        n_levels = len(self.model.encoder_channels)
        ebps = self.model.encoder_blocks_per_stage
        dbps = self.model.decoder_blocks_per_stage
        if ebps:
            assert len(ebps) == n_levels, (
                f"encoder_blocks_per_stage must have {n_levels} entries "
                f"(= len(encoder_channels)); got {len(ebps)}")
            assert all(b >= 1 for b in ebps), \
                "encoder_blocks_per_stage entries must all be >= 1"
        if dbps:
            assert len(dbps) == n_levels - 1, (
                f"decoder_blocks_per_stage must have {n_levels - 1} entries "
                f"(= len(encoder_channels) - 1); got {len(dbps)}")
            assert all(b >= 1 for b in dbps), \
                "decoder_blocks_per_stage entries must all be >= 1"
        assert self.loss.name in (
            # 单损失
            "dice", "bce", "focal", "tversky",
            "gdl", "focal_tversky", "lovasz", "cldice",
            # 复合损失
            "dice_bce", "dice_focal", "dice_tversky",
            "focal_plus_tversky", "dice_cldice", "dice_focal_tversky",
            "dice_lovasz", "bce_lovasz",
            "gdl_bce", "gdl_focal",
        ), f"Invalid loss: {self.loss.name}"
        assert self.loss.gdl_weight_type in ("square", "simple", "uniform"), (
            f"Invalid gdl_weight_type: {self.loss.gdl_weight_type}")
        assert self.loss.focal_tversky_gamma > 0, (
            f"focal_tversky_gamma must be > 0, got {self.loss.focal_tversky_gamma}")
        assert self.loss.cldice_iter >= 1, (
            f"cldice_iter must be >= 1, got {self.loss.cldice_iter}")
        assert self.loss.slice_loss_reduction in ("per_slice", "per_volume"), (
            f"Invalid slice_loss_reduction: {self.loss.slice_loss_reduction!r}; "
            "expected 'per_slice' or 'per_volume'.")
        assert self.train.optimizer in ("adam", "adamw", "sgd"), \
            f"Invalid optimizer: {self.train.optimizer}"
        assert self.train.scheduler in (
            "cosine", "cosine_warm_restarts", "poly", "step", "plateau", "one_cycle",
        ), f"Invalid scheduler: {self.train.scheduler}"
        assert len(self.data.patch_size) == 3, \
            "patch_size must be [D, H, W]"
        assert self.data.patch_mode in ("z_axis", "cubic", "whole", "2_5d"), \
            f"Invalid patch_mode: {self.data.patch_mode}"
        assert self.data.z_boundary_mode in ("stretch", "edge_pad"), (
            f"Invalid z_boundary_mode: {self.data.z_boundary_mode!r}; "
            "expected 'stretch' or 'edge_pad'.")
        if self.data.patch_mode == "whole":
            # whole 模式下多分辨率无物理意义。
            assert len(self.data.multi_res_scales) == 1 \
                and self.data.multi_res_scales[0] == 1.0, (
                f"whole-volume mode requires multi_res_scales=[1.0]; got {self.data.multi_res_scales}.")
        # aux_keep_native_d：仅 2.5D + 多视图有意义。
        if self.data.aux_keep_native_d:
            assert self.data.patch_mode == "2_5d", (
                f"data.aux_keep_native_d=True requires patch_mode='2_5d'; got {self.data.patch_mode!r}.")
            assert len(self.data.multi_res_scales) > 1, (
                "data.aux_keep_native_d=True requires len(multi_res_scales) > 1; "
                f"got {self.data.multi_res_scales}.")

        # keep_native_multi_res：aux_keep_native_d 的 3D 对应，dataset 发单 cube 后由 trainer 逐视图几何处理。
        if self.data.keep_native_multi_res:
            assert self.data.patch_mode in ("z_axis", "cubic"), (
                "data.keep_native_multi_res=True requires patch_mode in "
                f"('z_axis','cubic'); got {self.data.patch_mode!r}. Use "
                "data.aux_keep_native_d for the 2.5D analogue.")
            assert len(self.data.multi_res_scales) > 1, (
                "data.keep_native_multi_res=True requires len(multi_res_scales) > 1; "
                f"got {self.data.multi_res_scales}.")
            assert float(self.data.multi_res_scales[0]) == 1.0, (
                "data.keep_native_multi_res=True requires multi_res_scales[0]==1.0; "
                f"got {self.data.multi_res_scales}.")
            assert not self.data.aux_keep_native_d, (
                "keep_native_multi_res and aux_keep_native_d are mutually exclusive (3D vs 2.5D analogues).")
            if self.data.patch_mode == "z_axis":
                assert self.data.z_boundary_mode == "edge_pad", (
                    "keep_native_multi_res=True (z_axis) requires z_boundary_mode='edge_pad' "
                    f"(auto-set by sync()); got {self.data.z_boundary_mode!r}.")

        if self.data.patch_mode == "2_5d":
            # 2.5D 不变式重检（防手改后陈旧配置）。
            assert len(self.data.multi_res_scales) >= 1, (
                "2.5D mode requires at least one entry in multi_res_scales.")
            assert self.data.multi_res_scales[0] == 1.0, (
                "2.5D mode requires multi_res_scales[0]==1.0 (view 0 = prediction target); "
                f"got {self.data.multi_res_scales}.")
            n_views = len(self.data.multi_res_scales)
            lift = bool(getattr(self.model, "lift_2_5d_to_3d", False))
            if lift:
                # lift：D 保留为空间轴（真 3D UNet），与折叠-D 布局互斥。
                assert self.model.spatial_dims == 3, (
                    "lift_2_5d_to_3d=True requires model.spatial_dims=3 (auto-set by sync()).")
                assert self.model.in_channels == n_views, (
                    f"lift_2_5d_to_3d=True requires in_channels == n_views ({n_views}); "
                    f"got {self.model.in_channels}.")
                assert not self.data.aux_keep_native_d, (
                    "lift_2_5d_to_3d and aux_keep_native_d are mutually exclusive.")
                # 几何约束：D 需 % 2**(n_levels-1) == 0，且 >= 2**(n_levels-1)。
                n_levels = len(self.model.encoder_channels)
                D = int(self.data.patch_size[0])
                req = 1 << (n_levels - 1)
                if D < req or D % req != 0:
                    raise AssertionError(
                        f"lift_2_5d_to_3d=True with {n_levels} encoder stages requires "
                        f"patch_size[0] (D={D}) divisible by 2**(n_levels-1)={req}. "
                        f"Increase D to a multiple of {req}, or reduce len(encoder_channels).")
            else:
                assert self.model.spatial_dims == 2, (
                    "2.5D mode requires model.spatial_dims=2 (auto-set by sync()). "
                    "For Plan A 3D lift, set model.lift_2_5d_to_3d=True.")
            if (not lift) and self.data.aux_keep_native_d and n_views > 1:
                depths = self.aux_view_depths
                expected_in = int(sum(depths))
                assert self.model.in_channels == expected_in, (
                    f"2.5D + aux_keep_native_d=True requires in_channels == sum(D_k) = "
                    f"sum({depths}) = {expected_in}; got {self.model.in_channels}.")
                assert self.data.z_boundary_mode == "edge_pad", (
                    f"aux_keep_native_d=True requires z_boundary_mode='edge_pad'; "
                    f"got {self.data.z_boundary_mode!r}.")
                # 辅视图提供额外输入却无监督信号不合理。
                assert getattr(self.model, "aux_seg_supervision", False), (
                    "aux_keep_native_d=True requires model.aux_seg_supervision=True "
                    "(each native-depth view k drives an aux head).")
            elif not lift:
                expected_in = int(self.data.patch_size[0]) * n_views
                assert self.model.in_channels == expected_in, (
                    f"2.5D requires in_channels == patch_size[0] * n_views = "
                    f"{self.data.patch_size[0]} * {n_views} = {expected_in}; "
                    f"got {self.model.in_channels}.")
            # Plan C：aux view k 注入 encoder 第 k 级。
            if self.model.context_fusion == "hierarchical" and n_views > 1:
                n_stages = len(self.model.encoder_channels)
                assert n_views <= n_stages, (
                    f"context_fusion='hierarchical' requires n_views <= n_stages; "
                    f"got n_views={n_views}, n_stages={n_stages}.")
                stem_stride_map = {
                    "conv3": 1, "conv7": 1, "dual": 1,
                    "patch2": 2, "patch4": 4,
                }
                s0 = stem_stride_map[self.model.stem_mode]
                deepest = s0 * (2 ** (n_views - 1))
                pH, pW = int(self.data.patch_size[1]), int(self.data.patch_size[2])
                assert pH % deepest == 0 and pW % deepest == 0, (
                    f"hierarchical fusion with n_views={n_views}, stem_mode={self.model.stem_mode!r} "
                    f"requires patch H/W divisible by {deepest}; got ({pH}, {pW}).")
            # aux 监督：仅在 n_views > 1 时有意义。
            if getattr(self.model, "aux_seg_supervision", False):
                assert n_views > 1, (
                    "aux_seg_supervision=True requires n_views > 1; got 1.")
                aw = list(getattr(self.loss, "aux_supervision_weights", []))
                if aw:
                    assert len(aw) == n_views - 1, (
                        f"aux_supervision_weights length must = n_views-1 ({n_views-1}); got {aw}.")
                    assert all(w >= 0 for w in aw), (
                        f"aux_supervision_weights must be non-negative; got {aw}.")
                # Plan C 需 n_views < n_levels，使每 aux 头走不同 decoder 特征。
                if self.model.context_fusion == "hierarchical":
                    n_levels = len(self.model.encoder_channels)
                    assert n_views < n_levels, (
                        f"aux_seg_supervision + hierarchical requires n_views < n_levels; "
                        f"got n_views={n_views}, n_levels={n_levels}.")
        assert self.data.aug_oversample_ratio >= 1.0, \
            "aug_oversample_ratio must be >= 1.0"
        assert len(self.data.multi_res_scales) >= 1, \
            "multi_res_scales must have at least one scale (e.g. [1.0])"
        assert all(s >= 1.0 for s in self.data.multi_res_scales), \
            "All multi_res_scales must be >= 1.0"
        assert self.train.save_best_mode in ("max", "min"), \
            f"Invalid save_best_mode: {self.train.save_best_mode}"
        assert str(self.train.save_best_criterion).lower() in (
            "loss", "dice", "dice+surface_dice"), (
            f"Invalid save_best_criterion: {self.train.save_best_criterion!r}; "
            f"expected one of 'loss' | 'dice' | 'dice+surface_dice'.")
        assert int(self.train.surface_dice_tolerance) >= 0, \
            f"surface_dice_tolerance must be >= 0; got {self.train.surface_dice_tolerance}"
        assert 0.0 <= float(self.train.surface_dice_weight) <= 1.0, \
            f"surface_dice_weight must be in [0,1]; got {self.train.surface_dice_weight}"
        # z 轴交错推理检查（仅启用时）。
        if self.predict.z_interleave_enabled:
            assert self.data.patch_mode == "2_5d", (
                f"predict.z_interleave_enabled=True requires patch_mode='2_5d'; "
                f"got {self.data.patch_mode!r}.")
            thr = self.predict.z_interleave_thresholds
            fac = self.predict.z_interleave_factors
            assert len(fac) == len(thr) + 1, (
                f"z_interleave_factors length must = len(thresholds)+1; "
                f"got thresholds={thr}, factors={fac}.")
            assert all(t > 0 for t in thr), (
                f"z_interleave_thresholds must all > 0; got {thr}.")
            assert thr == sorted(thr), (
                f"z_interleave_thresholds must be ascending; got {thr}.")
            assert all(int(f) >= 1 for f in fac), (
                f"z_interleave_factors must all >= 1; got {fac}.")
            # stretch 会拉伸短子流、冲淡交错收益，仅警告。
            if self.data.z_boundary_mode != "edge_pad":
                logger.warning(
                    "z_interleave_enabled=True with z_boundary_mode=%r: "
                    "short sub-streams will be stretched along z. Prefer 'edge_pad'.",
                    self.data.z_boundary_mode)
        if self.data.num_classes < 2:
            logger.warning("num_classes=%d < 2, will auto-detect from data.",
                           self.data.num_classes)

    @property
    def num_fg_classes(self) -> int:
        """Number of foreground classes (excluding background)."""
        return max(self.data.num_classes - 1, 1)

    @property
    def aux_view_depths(self) -> List[int]:
        """2.5D 下每视图原生深度 D_k = round(D * s_k)，强制 D_0 = D。非 2.5D 返回空列表。

        R5：委托给 ``build_topology`` 以保持单一真相源；仅形状计算，不依赖
        ``data.aux_keep_native_d``，调用方自行根据该标志决定是否使用。
        """
        from .models.topology import build_topology
        return list(build_topology(self).aux_view_depths)


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------
_SUB_CONFIGS = {
    "data": DataConfig,
    "augment": AugConfig,
    "model": ModelConfig,
    "loss": LossConfig,
    "train": TrainConfig,
    "predict": PredictConfig,
}


def _dataclass_from_dict(cls, d: Dict[str, Any]):
    """Recursively construct a dataclass from a dict."""
    if not isinstance(d, dict):
        return d
    field_names = {f.name for f in fields(cls)}
    kwargs = {}
    for k, v in d.items():
        if k not in field_names:
            logger.warning("Unknown config key: %s", k)
            continue
        if k in _SUB_CONFIGS and isinstance(v, dict):
            v = _dataclass_from_dict(_SUB_CONFIGS[k], v)
        kwargs[k] = v
    return cls(**kwargs)


def load_config(path: Union[str, Path]) -> Config:
    """Load configuration from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = _dataclass_from_dict(Config, raw)
    cfg.sync()
    cfg.validate()
    return cfg


def save_config(cfg: Config, path: Union[str, Path]) -> None:
    """Save configuration to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False,
                  sort_keys=False, allow_unicode=True)
