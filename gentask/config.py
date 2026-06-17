"""Dataclass + YAML config. Each YAML file maps directly to nested dataclasses."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

logger = logging.getLogger(__name__)


class ConfigError(AssertionError, ValueError):
    """配置校验错误。

    同时继承 AssertionError（历史上 validate() 用裸 assert，调用方/测试捕
    AssertionError）与 ValueError（语义上是非法输入），保证向后兼容的同时
    不再依赖 assert 语句（``python -O`` 下不会被剥除）。新代码应捕
    ConfigError 或 ValueError。
    """


def _require(cond: bool, msg: str) -> None:
    """运行时配置校验：条件不成立时抛 ConfigError。"""
    if not cond:
        raise ConfigError(msg)


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
    keep_native_view_depth: bool = False

    # 3D 多 FOV 懒加载单 cube（z_axis/cubic）。True：dataset 发单 cube，trainer 逐视图裁剪/重采样。
    # 约束：scales[0]==1.0；与 keep_native_view_depth 互斥；z_axis 强制 edge_pad。
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

    # weight_map 插值模式："nearest" 保持离散权重（默认，含连续手标 wmap）；
    # "bilinear" 仅在确认权重为平滑连续场且可接受插值混合时使用。
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

    # 注：spatial_dims（2/3）与 in_channels 是由 patch_mode/multi_res_scales 等
    # 决定的"几何派生量"，不再作为可写字段/YAML 接口暴露（避免设了却被 sync 静默
    # 重写的困惑）。它们由 sync() 经 build_topology 算出，并以只读 property 暴露
    # （见类末尾），读 cfg.model.in_channels / spatial_dims 不变。

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
    # 与 data.keep_native_view_depth 互斥，仅在 2.5D 生效。
    lift_2_5d_to_3d: bool = False

    # Stem / patch-embed："conv3" | "conv7" | "dual" | "patch2" | "patch4"。patchN 降 N 倍分辨率（UNet3D 主输出加上采样）。
    stem_mode: str = "conv3"

    # 多 FOV 上下文融合（仅 2.5D + n_views>1）。示例："shared_stem"（全部过同一 stem）、"multi_stem_proj"（Plan A，逐视图 stem→cat→1×1）。
    # 还有 "hierarchical"（Plan C，aux k 注入 encoder 第 k 级）。3D 模式下忽略。
    stem_fusion_mode: str = "multi_stem_proj"

    # Decoder 拓扑："unet" 对称（默认）；"unetpp" 嵌套稠密；"unet3p" 全尺度 skip。
    decoder_type: str = "unet"

    # UNet3+ 各分支通道数（仅 decoder_type=="unet3p"）。
    unet3p_cat_channels: int = 64

    # 下采样："conv" | "maxpool" | "avgpool" | "blurpool" | "pixelunshuffle"。
    downsample_mode: str = "conv"

    # 上采样："transpose" | "trilinear" | "nearest" | "pixelshuffle" | "carafe" | "dysample"。
    upsample_mode: str = "transpose"

    # 各向异性下采样。True 时按 patch_size 自动推导逐级 per-axis stride：薄轴（如 z）
    # 分辨率落后才不降采样，避免深层被压成 1（nnU-Net 思路，保持各轴分辨率 2× 以内）。
    # 仅 decoder_type='unet' + downsample_mode∈{conv,maxpool,avgpool} +
    # upsample_mode∈{transpose,trilinear,nearest} + 非 ConvNeXt LN-first 下采样时支持。
    # 2.5D（spatial_dims=2）下 z 折进通道，仅作用于 H/W（通常各向同性，无实质变化）。
    anisotropic_pooling: bool = False

    # 显式逐级下采样 stride（非空时覆盖 anisotropic_pooling 自动推导）。
    # 长度 = len(encoder_channels)-1，每项长度 = spatial_dims，值 ∈ {1,2}。
    # 例（3D，5 级，保 z）：[[1,2,2],[1,2,2],[2,2,2],[2,2,2]]。
    downsample_strides: List[List[int]] = field(default_factory=list)

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

    # ---- 几何派生只读量（不暴露写接口；由 Config.sync() 经 build_topology 写入）----
    # 用私有 backing 字段承载，sync() 前读到的是安全默认值（3D / 单通道）。
    def __post_init__(self) -> None:
        self._spatial_dims: int = 3
        self._in_channels: int = 1

    @property
    def spatial_dims(self) -> int:
        """3 = 3D UNet（z_axis/cubic/whole）；2 = 2D UNet（2.5D 折叠 D）。"""
        return self._spatial_dims

    @property
    def in_channels(self) -> int:
        """模型输入通道数（2.5D 多 FOV 为 n_views*D 等，见 build_topology）。"""
        return self._in_channels


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
    # 选模标准（互斥）：
    #   * "loss"              → val_loss ↓
    #   * "dice"              → mean_dice ↑
    #   * "dice+surface_dice" → mean_combined = (1−w)·dice + w·sd ↑
    #   * "iou"               → mean_iou ↑                （Jaccard，对边界更严格）
    #   * "mcc"               → mean_mcc ↑                （类不平衡稳健，∈[−1,1]）
    #   * "min_dice"          → min_class_dice ↑          （短板：最差类 dice）
    #   * "balanced"          → mean_balanced ↑           （dice/sd/iou/mcc01 调和均值）
    # 单一真相源：选模标准只暴露 save_best_criterion 一个接口。底层的
    # (save_best_metric, save_best_mode) 由它派生（见类末尾的只读 property），
    # 故不再作为可写字段/YAML 接口暴露，避免"设了却被静默重写"的困惑。
    save_best_criterion: str = "dice"
    # Surface Dice 容差（像素，Chebyshev 邻域；0=严格表面 Dice）。
    surface_dice_tolerance: int = 1
    # 组合标准下 combined = (1-w)*dice + w*surface_dice。
    surface_dice_weight: float = 0.5
    # 任务化推荐预设：非空时由 sync() 覆盖上面三项 (criterion / tolerance / weight)。
    # 仅作"任务名 → 经验上推荐组合"的一键映射，便于复用与切任务。空串 = 不启用，
    # 完全沿用用户显式设置的三个字段。可选值见 ``_SAVE_BEST_PRESETS``：
    #   lung / vessel / airway / bone_multi / lymph_node / lesion_small /
    #   oar_multi / heart_chamber / bone_lung_combined
    save_best_preset: str = ""

    # 选模严格度（与 save_best_criterion 正交：criterion 决定"看哪个指标"，
    # 本项决定"指标在什么预测上算"）：
    #   * "medium" — 现状：在 val_loader 的随机 patch/切片上前向并算指标，快但
    #                非整卷，z 向上下文被切断，指标偏乐观/抖动。
    #   * "high"   — 严格：对每个 val 整卷做与部署一致的滑窗推理后再算指标，
    #                最可靠但更慢（每次验证多一遍整卷推理）。npz 无物理 z-spacing，
    #                故 high 不启用 predict.z_interleave，其余几何与推理一致。
    # 默认 "medium" 保持既有行为不变。
    val_metric_mode: str = "medium"

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

    # ---- 派生只读量（不暴露写接口；由 save_best_criterion 单一决定）----
    @property
    def save_best_metric(self) -> str:
        """被追踪的验证指标名（如 "mean_dice"）。由 save_best_criterion 派生。"""
        return _CRITERION_TO_METRIC.get(_norm_crit(self.save_best_criterion),
                                        _DEFAULT_CRIT_METRIC)[0]

    @property
    def save_best_mode(self) -> str:
        """选模方向 "max"/"min"。由 save_best_criterion 派生。"""
        return _CRITERION_TO_METRIC.get(_norm_crit(self.save_best_criterion),
                                        _DEFAULT_CRIT_METRIC)[1]


# ---------------------------------------------------------------------------
# 任务化推荐预设：``train.save_best_preset`` → (criterion, sd_tol, sd_w)。
# 设计来源详见 docs/选模标准建议（与 ``derive_overlap_metrics`` 引入的多维度
# pooled 指标配套）。每个 preset 仅覆盖三个底层字段，下游 trainer / validate
# 不感知 preset 本身的存在 —— 单一真相源仍是 (criterion, sd_tol, sd_w)。
#
# 用法（yaml）::
#     train:
#       save_best_preset: vessel    # 留空 / 删除 = 不启用，沿用显式三个字段
# ---------------------------------------------------------------------------
_SAVE_BEST_PRESETS: Dict[str, Dict[str, Any]] = {
    # 大实质器官：dice 易饱和，需用表面 dice 把边界质量拉进选模。
    "lung": {
        "save_best_criterion": "dice+surface_dice",
        "surface_dice_tolerance": 1,
        "surface_dice_weight": 0.5,
    },
    # 薄管状（血管）：dice 对断裂极不敏感；用 balanced 综合 dice/sd/iou/mcc，
    # tol=2 给重建几何留 1-2 体素余地。建议配合 dice_cldice 损失。
    "vessel": {
        "save_best_criterion": "balanced",
        "surface_dice_tolerance": 2,
        "surface_dice_weight": 0.5,
    },
    # 气道（拓扑敏感）：与血管同思路；建议配合 dice_cldice 损失。
    "airway": {
        "save_best_criterion": "balanced",
        "surface_dice_tolerance": 1,
        "surface_dice_weight": 0.5,
    },
    # 多块骨头：min_dice 守护最差类，防止漏一块被 mean_dice 掩盖。
    "bone_multi": {
        "save_best_criterion": "min_dice",
        "surface_dice_tolerance": 1,
        "surface_dice_weight": 0.5,
    },
    # 小淋巴结：极端类不平衡 + 小目标，MCC 最稳健（不可用 surface dice 作主指标）。
    "lymph_node": {
        "save_best_criterion": "mcc",
        "surface_dice_tolerance": 1,
        "surface_dice_weight": 0.5,
    },
    # 小病灶：同 lymph_node。
    "lesion_small": {
        "save_best_criterion": "mcc",
        "surface_dice_tolerance": 1,
        "surface_dice_weight": 0.5,
    },
    # 多器官 OAR（大小差异巨大）：min_dice 直接守底线；监控 balanced。
    "oar_multi": {
        "save_best_criterion": "min_dice",
        "surface_dice_tolerance": 1,
        "surface_dice_weight": 0.5,
    },
    # 心脏腔室 / 软组织块：表面质量与体积一致性都重要，sd 权重略低。
    "heart_chamber": {
        "save_best_criterion": "dice+surface_dice",
        "surface_dice_tolerance": 1,
        "surface_dice_weight": 0.4,
    },
    # 用户当前 lung+bone 多任务（带强 region weight）：balanced 是稳妥折中。
    "bone_lung_combined": {
        "save_best_criterion": "balanced",
        "surface_dice_tolerance": 1,
        "surface_dice_weight": 0.5,
    },
}


# ---------------------------------------------------------------------------
# 选模标准 → (被追踪指标名, 选模方向) 的唯一映射表。
# ``TrainConfig.save_best_metric / save_best_mode`` 是从此表派生的只读 property；
# ``Config.validate()`` 也以本表的键作为合法 criterion 白名单。单一真相源。
# ---------------------------------------------------------------------------
_CRITERION_TO_METRIC: Dict[str, Tuple[str, str]] = {
    "loss":              ("val_loss",        "min"),
    "dice":              ("mean_dice",       "max"),
    "dice+surface_dice": ("mean_combined",   "max"),
    "iou":               ("mean_iou",        "max"),
    "mcc":               ("mean_mcc",        "max"),
    "min_dice":          ("min_class_dice",  "max"),
    "balanced":          ("mean_balanced",   "max"),
}
# 未知 criterion 的兜底（validate() 会另行报错），保证 property 不抛异常。
_DEFAULT_CRIT_METRIC: Tuple[str, str] = ("mean_dice", "max")


def _norm_crit(crit: Any) -> str:
    """归一化 criterion 字符串（小写 + 去空白），供映射查表使用。"""
    return str(crit).lower().strip()


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

    # TTA flip 变体的批量化前向块大小（单卷推理提速；仅 tta_flip=True 时生效）。
    #   将多个 flip 变体沿 batch 轴 torch.cat 成一次前向，逐像素等价于串行，仅减少
    #   前向次数。3D 7 种 flip → ceil(7/tta_batch_size)+1 次前向；2.5D 3 种 → 同理。
    # None  — 退化为 predict.batch_size。
    # 显存提醒：单次前向样本数 ≈ batch_size * tta_batch_size，显存随之线性上升；
    #   显存吃紧时调小（如 2）。AdaBN per_volume 估计阶段会自动退回串行以保 BN 统计一致。
    tta_batch_size: Optional[int] = None

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

    # 测试时自适应 BatchNorm (AdaBN)：推理阶段用目标域前向重估 BN running stats，
    # 无需标签、不重训，针对跨数据集域漂移导致的假阳。仅当 model.norm_type=='batch'
    # 时有效（instance/group norm 为 no-op）。
    adabn_enabled: bool = False

    # 'global'    — 用 adabn_num_volumes 卷目标域整卷预热一次 BN 统计，全程复用。
    # 'per_volume'— 每卷推理前用该卷自身重估 BN，再冻结预测（transductive BN）。
    adabn_mode: str = "global"

    # global 模式预热用的目标域整卷数；per_volume 模式忽略。
    adabn_num_volumes: int = 8


# ---------------------------------------------------------------------------
# Task configuration（分割 vs 生成）
# ---------------------------------------------------------------------------
@dataclass
class TaskConfig:
    """任务类型与生成任务设置。

    ``type=='segmentation'``（默认）时本节其余字段全部忽略，旧配置零改动可跑。
    ``type=='generation'`` 时启用图像复原（当前仅超分 super-resolution），由
    ``algorithm`` 选择两类经典范式之一：

      * ``'regression'`` —— 前馈回归复原（DnCNN / SRCNN / U-Net regression）。
        模型一次前向把退化图映射回干净图；可选残差学习（预测 HR−LR）。
      * ``'diffusion'``  —— 条件扩散（DDPM / EDM），以退化图为条件迭代去噪采样
        （类 SR3 / Palette）。复用 ADM / EDM2 backbone（带 timestep/σ 条件）。

    设计为扁平字段，便于 YAML 直接映射、避免嵌套 dataclass 递归构造的歧义。
    """

    # "segmentation" | "generation"
    type: str = "segmentation"

    # 生成算法："regression" | "diffusion"。
    algorithm: str = "regression"

    # 复原子任务（退化算子）。当前仅 "superres"。
    degradation: str = "superres"

    # 生成输出通道数（CT 灰度=1）。决定生成头/模型输出通道。
    out_channels: int = 1

    # --- 超分退化（degradation=="superres"）---
    # 下采样倍率（各空间轴一致）。LR = down(HR, 1/scale) 再 up 回 HR 尺寸作输入。
    sr_scale: int = 2
    # 各向异性逐空间轴倍率（覆盖 sr_scale）；空=各向同性。
    # 顺序按空间轴：3D 为 (D,H,W)，2.5D 为 (H,W)。CT 厚层→薄层用 [2,1,1] 只超分 z 轴。
    sr_scale_per_axis: List[int] = field(default_factory=list)
    # 制作 LR 的下/上采样插值："trilinear" | "area" | "nearest"。area≈抗锯齿平均池化。
    sr_kernel: str = "area"
    # LR 上附加高斯噪声（模拟采集噪声）；0=禁用。
    sr_noise_std: float = 0.0

    # --- 回归算法（algorithm=="regression"）---
    # 残差学习：模型预测 (HR − LR_up)，最终输出 = pred + LR_up（DnCNN/VDSR 思路）。
    residual: bool = False
    # 重建损失："charbonnier" | "l1" | "mse"。
    recon_loss: str = "charbonnier"
    charbonnier_eps: float = 1e-3
    # SSIM 辅助损失权重（0=禁用）。总损失 = recon + ssim_weight*(1-SSIM) + grad_weight*grad。
    ssim_weight: float = 0.0
    ssim_window: int = 7
    # 梯度（边缘）L1 损失权重（0=禁用）。
    grad_weight: float = 0.0

    # --- 扩散算法（algorithm=="diffusion"）---
    # 参数化："edm"（Karras 2022 去噪预条件）| "ddpm_eps"（Ho 2020 ε-预测）。
    parameterization: str = "edm"
    # DDPM 训练步数与 β schedule（parameterization=="ddpm_eps"）。
    num_train_timesteps: int = 1000
    beta_schedule: str = "linear"  # "linear" | "cosine"
    # EDM σ 采样分布与数据尺度（parameterization=="edm"）。
    sigma_data: float = 0.5
    p_mean: float = -1.2
    p_std: float = 1.2
    # 采样器："edm_heun"（EDM 二阶）| "ddpm"（祖先采样）| "ddim"。
    sampler: str = "edm_heun"
    sample_steps: int = 18
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    ddim_eta: float = 0.0


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
    task   : TaskConfig    = field(default_factory=TaskConfig)

    def sync(self) -> None:
        """同步跨子配置的对应字段。

        所有"模型几何派生量"（``in_channels`` / ``spatial_dims``）由
        ``segtask_v1.models.topology.build_topology(self)`` 一次性算出，写入
        ``ModelConfig`` 的私有 backing 字段（对外是只读 property）。本方法仅保留
        "非派生"职责（``num_classes`` 推断、``z_boundary_mode`` 自动升级、resenc
        preset、save_best 预设）。
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
        from .models.topology import build_topology
        topo = build_topology(self)
        self.model._in_channels = topo.in_channels
        self.model._spatial_dims = topo.spatial_dims

        self._apply_resenc_preset()
        # 任务化预设（若启用）覆盖 (criterion, sd_tol, sd_w)；save_best_metric/mode
        # 是 criterion 的派生只读 property，无需在此写回。
        self._apply_save_best_preset()

    def _apply_save_best_preset(self) -> None:
        """``train.save_best_preset`` 非空时覆盖 (criterion, tolerance, weight)。

        预设由 ``_SAVE_BEST_PRESETS`` 定义；空串表示不启用、保留用户显式设置。
        非法预设名不在此处报错（避免 sync() 中产生 hard fail），由
        ``validate()`` 给出可读错误。本方法仅做"已知则覆盖"。
        """
        name = str(self.train.save_best_preset or "").strip().lower()
        if not name:
            return
        spec = _SAVE_BEST_PRESETS.get(name)
        if spec is None:
            return  # validate() 报错
        tc = self.train
        prev = (tc.save_best_criterion, tc.surface_dice_tolerance,
                tc.surface_dice_weight)
        tc.save_best_criterion    = str(spec["save_best_criterion"])
        tc.surface_dice_tolerance = int(spec["surface_dice_tolerance"])
        tc.surface_dice_weight    = float(spec["surface_dice_weight"])
        new = (tc.save_best_criterion, tc.surface_dice_tolerance,
               tc.surface_dice_weight)
        if prev != new:
            logger.info(
                "save_best_preset=%r overrode (criterion, sd_tol, sd_w): "
                "%s → %s.", name, prev, new)

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
        self._validate_augment()
        self._validate_loss()
        self._validate_data()
        self._validate_2_5d()
        self._validate_train()
        self._validate_predict()
        self._validate_task()
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
        if t.sr_scale_per_axis:
            sdims = 2 if str(self.data.patch_mode).lower() == "2_5d" else 3
            _require(
                len(t.sr_scale_per_axis) == sdims,
                f"task.sr_scale_per_axis length must equal spatial_dims ({sdims}) "
                f"for patch_mode={self.data.patch_mode!r}; got {t.sr_scale_per_axis}.")
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

    def _validate_augment(self) -> None:
        """augment.* 校验。"""
        _require(
            self.augment.wmap_interp_mode in ("nearest", "bilinear"),
            f"Invalid augment.wmap_interp_mode: {self.augment.wmap_interp_mode!r} "
            "(expected 'nearest' or 'bilinear').")

    def _validate_loss(self) -> None:
        """loss.* 名称与参数校验。"""
        _require(
            self.loss.name in (
            # 单损失
            "dice", "bce", "focal", "tversky",
            "gdl", "focal_tversky", "lovasz", "cldice",
            # 复合损失
            "dice_bce", "dice_focal", "dice_tversky",
            "focal_plus_tversky", "dice_cldice", "dice_focal_tversky",
            "dice_lovasz", "bce_lovasz",
            "gdl_bce", "gdl_focal",
        ),
            f"Invalid loss: {self.loss.name}")
        _require(
            self.loss.gdl_weight_type in ("square", "simple", "uniform"),
            f"Invalid gdl_weight_type: {self.loss.gdl_weight_type}")
        _require(
            self.loss.focal_tversky_gamma > 0,
            f"focal_tversky_gamma must be > 0, got {self.loss.focal_tversky_gamma}")
        _require(
            self.loss.cldice_iter >= 1,
            f"cldice_iter must be >= 1, got {self.loss.cldice_iter}")
        _require(
            self.loss.slice_loss_reduction in ("per_slice", "per_volume"),
            f"Invalid slice_loss_reduction: {self.loss.slice_loss_reduction!r}; "
            "expected 'per_slice' or 'per_volume'.")

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
            # whole 模式下多分辨率无物理意义。
            _require(
                len(self.data.multi_res_scales) == 1 \
                and self.data.multi_res_scales[0] == 1.0,
                f"whole-volume mode requires multi_res_scales=[1.0]; got {self.data.multi_res_scales}.")
        # keep_native_view_depth：仅 2.5D + 多视图有意义。
        if self.data.keep_native_view_depth:
            _require(
                self.data.patch_mode == "2_5d",
                f"data.keep_native_view_depth=True requires patch_mode='2_5d'; got {self.data.patch_mode!r}.")
            _require(
                len(self.data.multi_res_scales) > 1,
                "data.keep_native_view_depth=True requires len(multi_res_scales) > 1; "
                f"got {self.data.multi_res_scales}.")

        # keep_native_multi_res：keep_native_view_depth 的 3D 对应，dataset 发单 cube 后由 trainer 逐视图几何处理。
        if self.data.keep_native_multi_res:
            _require(
                self.data.patch_mode in ("z_axis", "cubic"),
                "data.keep_native_multi_res=True requires patch_mode in "
                f"('z_axis','cubic'); got {self.data.patch_mode!r}. Use "
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

    def _validate_2_5d(self) -> None:
        """2.5D 专属不变式（折叠通道 / lift / Plan A·C / aux 监督）。"""
        if self.data.patch_mode == "2_5d":
            # 2.5D 不变式重检（防手改后陈旧配置）。
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
                # lift：D 保留为空间轴（真 3D UNet），与折叠-D 布局互斥。
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
                # 几何约束：D 需 % 2**(n_levels-1) == 0，且 >= 2**(n_levels-1)。
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
                expected_in = int(sum(depths))
                _require(
                    self.model.in_channels == expected_in,
                    f"2.5D + keep_native_view_depth=True requires in_channels == sum(D_k) = "
                    f"sum({depths}) = {expected_in}; got {self.model.in_channels}.")
                _require(
                    self.data.z_boundary_mode == "edge_pad",
                    f"keep_native_view_depth=True requires z_boundary_mode='edge_pad'; "
                    f"got {self.data.z_boundary_mode!r}.")
                # 辅视图提供额外输入却无监督信号不合理。
                _require(
                    self.model.aux_seg_supervision,
                    "keep_native_view_depth=True requires model.aux_seg_supervision=True "
                    "(each native-depth view k drives an aux head).")
            elif not lift:
                expected_in = int(self.data.patch_size[0]) * n_views
                _require(
                    self.model.in_channels == expected_in,
                    f"2.5D requires in_channels == patch_size[0] * n_views = "
                    f"{self.data.patch_size[0]} * {n_views} = {expected_in}; "
                    f"got {self.model.in_channels}.")
            # Plan C：aux view k 注入 encoder 第 k 级。
            if self.model.stem_fusion_mode == "hierarchical" and n_views > 1:
                n_stages = len(self.model.encoder_channels)
                _require(
                    n_views <= n_stages,
                    f"stem_fusion_mode='hierarchical' requires n_views <= n_stages; "
                    f"got n_views={n_views}, n_stages={n_stages}.")
                stem_stride_map = {
                    "conv3": 1, "conv7": 1, "dual": 1,
                    "patch2": 2, "patch4": 4,
                }
                s0 = stem_stride_map[self.model.stem_mode]
                deepest = s0 * (2 ** (n_views - 1))
                pH, pW = int(self.data.patch_size[1]), int(self.data.patch_size[2])
                _require(
                    pH % deepest == 0 and pW % deepest == 0,
                    f"hierarchical fusion with n_views={n_views}, stem_mode={self.model.stem_mode!r} "
                    f"requires patch H/W divisible by {deepest}; got ({pH}, {pW}).")
            # aux 监督：仅在 n_views > 1 时有意义。
            if self.model.aux_seg_supervision:
                _require(
                    n_views > 1,
                    "aux_seg_supervision=True requires n_views > 1; got 1.")
                aw = list(self.loss.aux_supervision_weights)
                if aw:
                    _require(
                        len(aw) == n_views - 1,
                        f"aux_supervision_weights length must = n_views-1 ({n_views-1}); got {aw}.")
                    _require(
                        all(w >= 0 for w in aw),
                        f"aux_supervision_weights must be non-negative; got {aw}.")
                # Plan C 需 n_views < n_levels，使每 aux 头走不同 decoder 特征。
                if self.model.stem_fusion_mode == "hierarchical":
                    n_levels = len(self.model.encoder_channels)
                    _require(
                        n_views < n_levels,
                        f"aux_seg_supervision + hierarchical requires n_views < n_levels; "
                        f"got n_views={n_views}, n_levels={n_levels}.")

    def _validate_train(self) -> None:
        """train.* 优化器/调度器/选模标准校验。"""
        _require(
            self.train.optimizer in ("adam", "adamw", "sgd"),
            f"Invalid optimizer: {self.train.optimizer}")
        _require(
            self.train.scheduler in (
            "cosine", "cosine_warm_restarts", "poly", "step", "plateau", "one_cycle",
        ),
            f"Invalid scheduler: {self.train.scheduler}")
        # save_best_mode/metric 现为 criterion 的派生只读量（恒合法），无需单独校验。
        _require(
            _norm_crit(self.train.save_best_criterion) in _CRITERION_TO_METRIC,
            f"Invalid save_best_criterion: {self.train.save_best_criterion!r}; "
            f"expected one of: {' | '.join(repr(c) for c in _CRITERION_TO_METRIC)}.")
        _require(
            str(self.train.val_metric_mode).lower().strip() in ("medium", "high"),
            f"Invalid val_metric_mode: {self.train.val_metric_mode!r}; "
            "expected 'medium' (patch-level) or 'high' (full-volume).")
        # high 模式在整卷 blended 概率上算指标，无可逆 logits 故不产出 val_loss；
        # 因此 'loss' criterion 与 high 互斥（否则永远选不出 best）。改用重叠类指标。
        if (str(self.train.val_metric_mode).lower().strip() == "high"
                and _norm_crit(self.train.save_best_criterion) == "loss"):
            raise ConfigError(
                "train.save_best_criterion='loss' is incompatible with "
                "train.val_metric_mode='high' (full-volume inference produces "
                "blended probabilities, not invertible logits, so no val_loss "
                "is computed). Use an overlap-based criterion "
                "(dice / iou / mcc / min_dice / dice+surface_dice / balanced) "
                "or switch val_metric_mode to 'medium'.")
        # save_best_preset 空串 = 不启用；非空必须是已知预设名。
        preset = str(self.train.save_best_preset or "").strip().lower()
        if preset:
            _require(
                preset in _SAVE_BEST_PRESETS,
                f"Invalid save_best_preset: {self.train.save_best_preset!r}; "
                f"expected '' (disabled) or one of: "
                f"{' | '.join(repr(k) for k in _SAVE_BEST_PRESETS)}.")
        _require(
            int(self.train.surface_dice_tolerance) >= 0,
            f"surface_dice_tolerance must be >= 0; got {self.train.surface_dice_tolerance}")
        _require(
            0.0 <= float(self.train.surface_dice_weight) <= 1.0,
            f"surface_dice_weight must be in [0,1]; got {self.train.surface_dice_weight}")

    def _validate_predict(self) -> None:
        """predict.* z 交错与 AdaBN 校验。"""
        # z 轴交错推理检查（仅启用时）。
        if self.predict.z_interleave_enabled:
            _require(
                self.data.patch_mode == "2_5d",
                f"predict.z_interleave_enabled=True requires patch_mode='2_5d'; "
                f"got {self.data.patch_mode!r}.")
            thr = self.predict.z_interleave_thresholds
            fac = self.predict.z_interleave_factors
            _require(
                len(fac) == len(thr) + 1,
                f"z_interleave_factors length must = len(thresholds)+1; "
                f"got thresholds={thr}, factors={fac}.")
            _require(
                all(t > 0 for t in thr),
                f"z_interleave_thresholds must all > 0; got {thr}.")
            _require(
                thr == sorted(thr),
                f"z_interleave_thresholds must be ascending; got {thr}.")
            _require(
                all(int(f) >= 1 for f in fac),
                f"z_interleave_factors must all >= 1; got {fac}.")
            # stretch 会拉伸短子流、冲淡交错收益，仅警告。
            if self.data.z_boundary_mode != "edge_pad":
                logger.warning(
                    "z_interleave_enabled=True with z_boundary_mode=%r: "
                    "short sub-streams will be stretched along z. Prefer 'edge_pad'.",
                    self.data.z_boundary_mode)
        # 测试时自适应 BatchNorm 检查（仅启用时）。
        if self.predict.adabn_enabled:
            _require(
                self.predict.adabn_mode in ("global", "per_volume"),
                f"predict.adabn_mode must be 'global' or 'per_volume'; "
                f"got {self.predict.adabn_mode!r}.")
            _require(
                int(self.predict.adabn_num_volumes) >= 1,
                f"predict.adabn_num_volumes must be >= 1; "
                f"got {self.predict.adabn_num_volumes}.")
            # AdaBN 只对 BatchNorm 有意义；其余归一化层会使其成为 no-op，仅警告。
            if self.model.norm_type != "batch":
                logger.warning(
                    "predict.adabn_enabled=True but model.norm_type=%r != "
                    "'batch'; AdaBN will be a no-op (no BatchNorm to adapt).",
                    self.model.norm_type)


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

        生成任务下没有"前景类"概念，复用该接口返回 ``out_channels``，使依赖它
        推导输出通道数的下游（模型头 / 2.5D SliceChannel 重塑）保持统一口径。
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
        from .models.topology import build_topology
        return list(build_topology(self).per_view_depths)


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
    "task": TaskConfig,
}


# 旧 YAML 字段名 → 新字段名的向后兼容别名。读到旧名时自动改写并提示一次。
# 命名清晰化（TODO #4）：
#   data.aux_keep_native_d  → data.keep_native_view_depth（'aux' 误导：含主视图）
#   model.context_fusion    → model.stem_fusion_mode（与 num_stem_fusion_views 配对）
_FIELD_ALIASES: Dict[type, Dict[str, str]] = {
    DataConfig:  {"aux_keep_native_d": "keep_native_view_depth"},
    ModelConfig: {"context_fusion": "stem_fusion_mode"},
}


# 旧 YAML 中曾可手设、现已改为派生只读量的字段：读到时静默忽略（仅一次 info 提示），
# 而非按 "Unknown config key" 处理。TODO #4：派生量不再暴露可写接口。
#   train.save_best_metric / save_best_mode → 由 train.save_best_criterion 派生。
#   model.in_channels / spatial_dims        → 由 patch_mode/multi_res_scales 等派生
#                                             （sync() 经 build_topology 算出）。
_DEPRECATED_DERIVED_KEYS: Dict[type, Dict[str, str]] = {
    TrainConfig: {
        "save_best_metric": "save_best_criterion",
        "save_best_mode":   "save_best_criterion",
    },
    ModelConfig: {
        "in_channels":  "data.patch_mode / data.multi_res_scales",
        "spatial_dims": "data.patch_mode",
    },
}


def _dataclass_from_dict(cls, d: Dict[str, Any]):
    """Recursively construct a dataclass from a dict.

    支持向后兼容别名（``_FIELD_ALIASES``）：旧 YAML 字段名会被自动改写成新名，
    并打印一次弃用提示；若新旧名同时出现则报错。``_DEPRECATED_DERIVED_KEYS``
    列出的"曾可写、现派生只读"字段则直接忽略。
    """
    if not isinstance(d, dict):
        return d
    field_names = {f.name for f in fields(cls)}
    aliases = _FIELD_ALIASES.get(cls, {})
    derived = _DEPRECATED_DERIVED_KEYS.get(cls, {})
    kwargs = {}
    for k, v in d.items():
        if k in derived:
            logger.info(
                "Config key '%s' is now auto-derived from '%s' and no longer "
                "settable; ignoring the value in YAML.", k, derived[k])
            continue
        if k in aliases:
            new_key = aliases[k]
            if new_key in d:
                raise ValueError(
                    f"{cls.__name__}: both deprecated '{k}' and its "
                    f"replacement '{new_key}' are set; remove the deprecated one.")
            logger.warning(
                "Config key '%s' is deprecated; use '%s' instead "
                "(auto-remapped for backward compatibility).", k, new_key)
            k = new_key
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
