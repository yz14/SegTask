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

    # 可选逐样本体素重要性图目录（值 + 1）。预计算、与标签无关；生成任务可直接
    # 作为 batch["weight_map"] 使用。优先级：此目录 > data.region_weights。
    region_weight_dir   : str = ""
    region_weight_suffix: Union[str, List[str]] = ".nii.gz"

    # 静态按标签区域权重（值 + 1）；仅适用于基于标签的旧路径，生成任务不使用。
    # 空表示不启用。优先级低于 region_weight_dir。
    region_weights: List[float] = field(default_factory=list)

    # 可选逐样本条件卷目录；每个目录提供一个空间对齐的条件体（例如 mask / 预分割）。
    cond_dirs: List[str] = field(default_factory=list)
    cond_suffixes: Union[str, List[str]] = ".nii.gz"
    cond_intensity_min: float = -1024.0
    cond_intensity_max: float = 1024.0
    cond_normalize  : str = "minmax"
    cond_global_mean: float = 0.0
    cond_global_std : float = 1.0

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

    # 多 FOV 辅助重建监督（仅 2.5D + len(multi_res_scales)>1 生效）。主头预 view 0，
    # 辅助 view k 输出对应 view_k 的 HR slice；损失权重见 loss.aux_recon_weights（空则默认 0.5^k）。
    # 单视图/3D 不生效。
    # 生成任务多 FOV / 多视野辅助重建监督（仅 2.5D multi-view）。
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
    """Generation-side loss settings."""

    deep_supervision_weights: List[float] = field(
        default_factory=lambda: [1.0, 0.5, 0.25, 0.125])
    aux_recon_weights: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """训练循环设置。"""

    epochs: int = 200
    optimizer   : str = "adamw"
    lr          : float = 1e-3
    weight_decay: float = 1e-4
    momentum    : float = 0.99
    nesterov    : bool = True
    scheduler    : str = "cosine"
    warmup_epochs: int = 5
    warmup_lr    : float = 1e-6
    cosine_min_lr: float = 1e-6
    cosine_restart_period: int = 50
    cosine_restart_mult  : int = 2
    poly_power           : float = 0.9
    step_size            : int = 50
    step_gamma           : float = 0.1
    plateau_patience     : int = 10
    plateau_factor       : float = 0.5
    grad_accum_steps: int = 1
    grad_clip_norm: float = 12.0
    use_amp  : bool = True
    amp_dtype: str = "float16"
    compile_mode: str = "none"
    use_ema  : bool = True
    ema_decay: float = 0.999
    output_dir      : str = "outputs"
    save_every      : int = 10
    early_stopping: int = 0
    log_every: int = 10
    val_every: int = 1
    vis_every: int = 10
    seed         : int = 42
    deterministic: bool = False
    resume: str = ""
    pretrain: str = ""
    pretrain_strict: bool = False
    pretrain_load_ema: bool = False


# ---------------------------------------------------------------------------
# Prediction / Inference configuration
# ---------------------------------------------------------------------------
@dataclass
class PredictConfig:
    """Generation inference output settings."""

    output_dir: str = "predictions"


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
    # 退化采样方式："blur"（SISR，降采样模糊）| "decimate"（VFI 插帧，抽稀帧+线性插值填补）。
    sr_sampling: str = "blur"
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
