"""gentask 配置段：组合复用 ``taskcore.config`` 核心段 + 生成任务扩展。

设计（与 ssltask / clstask 同构）：核心段（data/model/loss/train/predict/aug）
直接子类化 ``taskcore.config`` 的同名 dataclass——公共字段/语义只有一份真相源；
生成任务专有字段（cond_* 条件卷、model.sisr 经典超分、val_full_volume 整卷验证、
滑窗复原 predict 字段等）在子类中追加；生成侧默认值不同的字段（保守增强
概率、predict.batch_size 等）在子类中显式覆盖，行为与迁移前逐位一致。

``TaskConfig``（分割 vs 生成的任务段）是 gentask 自有段，留在本包。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Union

from taskcore.config.core import (  # noqa: F401  (re-export: io.py/validation.py 使用)
    ConfigError,
    _require,
)
from taskcore.config import core as _core
from taskcore.config.model_migration import (
    SISR_FIELD_MAP,
    install_flat_model_compat,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------
@dataclass
class DataConfig(_core.DataConfig):
    """数据路径与预处理（核心段 + 生成任务扩展）。"""

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


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
@dataclass
class SISRConfig:
    """经典 SISR 专用（arch in ("edsr","rcan")，post-upsampling）。"""

    # 特征通道数（EDSR baseline 64）。
    channels: int = 64
    # EDSR 残差块数 / RCAN 每组 RCAB 数（EDSR baseline 16，RCAN 论文 20）。
    num_blocks: int = 16
    # RCAN 残差组数（EDSR 忽略；RCAN 论文 10）。
    num_groups: int = 10
    # EDSR 块残差缩放（大模型建议 0.1 稳定训练；RCAN 忽略）。
    res_scale: float = 1.0


@dataclass
class ModelConfig(_core.ModelConfig):
    """模型架构设置（核心段 + 生成任务扩展）。"""

    # ---- 经典 SISR 嵌套段（arch in ("edsr","rcan")） ----
    sisr: SISRConfig = field(default_factory=SISRConfig)


# @dataclass 重新生成了子类 __init__，须在子类上重装旧扁平接口兼容层
# （含 sisr_* → sisr.* 转发；嵌套段补齐含 sisr）。
install_flat_model_compat(
    ModelConfig,
    extra_flat_to_nested=SISR_FIELD_MAP,
    nested_sections=("unet", "adm", "edm2", "sisr"),
)


# ---------------------------------------------------------------------------
# Loss configuration
# ---------------------------------------------------------------------------
@dataclass
class LossConfig(_core.LossConfig):
    """损失设置（核心段 + 生成侧多视野辅助重建权重）。"""

    # 多 FOV 辅助重建监督权重（空则默认 0.5^k）。
    aux_recon_weights: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig(_core.TrainConfig):
    """训练循环设置（核心段 + 生成任务整卷验证）。"""

    # 整卷验证（M13，仅生成任务）：与部署同口径——在线退化整卷 → 推理器
    # 滑窗复原（复用 predict.overlap/blend 路径）→ 逐卷 PSNR/SSIM 平均；
    # 启用后选模/早停改用整卷 PSNR（patch 级指标仍一并报告）。
    val_full_volume: bool = False
    # 整卷验证最多评前 N 卷（控耗时）；0=全部 val 卷。
    val_full_volume_max: int = 0


# ---------------------------------------------------------------------------
# Prediction / Inference configuration
# ---------------------------------------------------------------------------
@dataclass
class PredictConfig(_core.PredictConfig):
    """生成推理输出设置（核心段 + 滑窗复原字段）。"""

    # 输入所在网格："hr"（输入已在 HR 网格上，例如已插值好的体 / 在线退化实验）
    # | "lr"（真实低分辨率输入，如原生厚层 CT）。"lr" 时先按 task.sr_scale
    # (_per_axis) 与训练一致的方式重采样到 HR 网格再入网：sr_sampling=='blur'
    # 用 sr_kernel_up 插值；'decimate' 用相位对齐线性插值（与训练退化对偶）。
    input_grid: str = "hr"

    # True：写出前把归一化域反变换回原强度（HU），并保留物理标定；
    # False：直接写归一化值（调试用）。
    denormalize: bool = True

    # 滑窗重叠比（每轴 stride = size*(1-overlap)）；0 = 不重叠。
    overlap: float = 0.5

    # 重叠区融合权重：'gaussian'（中心高权，消接缝）| 'uniform'（等权平均）。
    blend: str = "gaussian"

    # 每次前向拼接的滑窗数（>1 提升 GPU 利用率，显存占用同步增长）。
    # 回归模型数值等价；扩散模型固定 seed 下批内噪声流与逐窗不同
    # （仍可复现，但与 batch_size=1 结果不逐位一致）。
    batch_size: int = 1

    # spacing 感知 z 倍率（仅 input_grid=='lr'）：>0 时逐体读 NIfTI z spacing，
    # 以 round(z_spacing / target_z_spacing) 覆盖配置的 z 轴倍率（异质层厚数据
    # 下每体自适应）；0=禁用，用 task.sr_scale(_per_axis) 固定倍率。
    # 不支持 post-upsampling SISR（edsr/rcan 上采头倍率固定）。
    target_z_spacing: float = 0.0

    # 推理 autocast（仅 CUDA；dtype 口径同 train.amp_dtype）。
    use_amp: bool = True

    # 翻转 TTA：对称轴翻转多次复原取均值。仅在翻转不破坏退化相位的轴上
    # 生效（sr_sampling=='decimate' 时跳过被退化轴；2.5D 仅 H/W）。
    # 扩散采样下开启代价成倍增长，默认关。
    tta_flips: bool = False


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
    # 顺序按空间轴（长度=model.spatial_dims）：3D / 2.5D+lift 为 (D,H,W)，
    # 2.5D（未 lift）为 (H,W)。CT 厚层→薄层用 [2,1,1] 只超分 z 轴。
    sr_scale_per_axis: List[int] = field(default_factory=list)
    # 制作 LR 的下采样核："trilinear" | "area" | "nearest"（F.interpolate）
    # | "gauss" | "tri"（CT 层敏感度剖面 SSP 核，仅 sr_sampling=='blur'：逐退化轴
    # 1D 平滑后抽样，gauss FWHM=层厚、tri 半宽=层厚，比 box 均值更真实）。
    # area≈抗锯齿平均池化（对应理想 box 部分容积效应）。
    sr_kernel: str = "area"
    # LR 上采样回原尺寸的插值："trilinear" | "area" | "nearest"。默认 trilinear
    # （对应临床重建后线性插值到目标层厚；area 上采≈nearest，会产生块状 LR）。
    sr_kernel_up: str = "trilinear"
    # 退化采样方式："blur"（SISR，降采样模糊）| "decimate"（VFI 插帧，抽稀帧+线性插值填补）。
    sr_sampling: str = "blur"
    # LR 上附加高斯噪声（模拟采集噪声）；0=禁用。
    sr_noise_std: float = 0.0
    # 随机退化池（轻量 Real-ESRGAN 风格）：非空时训练每次前向从中随机抽一个
    # 下采核（取值同 sr_kernel）；验证/推理固定用 sr_kernel，指标可比。
    sr_kernel_pool: List[str] = field(default_factory=list)
    # 噪声 std 随机范围 [lo, hi]（训练每次均匀采样，覆盖 sr_noise_std）；
    # 空=固定用 sr_noise_std。验证/推理固定用 sr_noise_std。
    sr_noise_std_range: List[float] = field(default_factory=list)

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
    # 采样器：EDM 预条件用 "edm_heun"（二阶）| "edm_euler"（一阶确定性）；
    # DDPM 预条件用 "ddpm"（祖先采样）| "ddim"。
    sampler: str = "edm_heun"
    sample_steps: int = 18
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    ddim_eta: float = 0.0
    # DDPM 采样中 x0 预估的对称钳位半径：0 = 按 data.normalize 自动派生
    # （minmax→±1.5，zscore→±4.0）；>0 显式半径；<0 禁用钳位。
    x0_clip: float = 0.0


# ---------------------------------------------------------------------------
# Augmentation configuration
# ---------------------------------------------------------------------------
@dataclass
class AugConfig(_core.AugConfig):
    """GPU 数据增强（生成任务变体：无 label，作用于 image + cond + weight_map）。

    与核心段同名字段保持一致以便配置迁移；默认覆盖为"生成安全"的保守参数：
    空间变换保留（image/cond 同步 warp，重建目标 = 增强后的 image 自身，
    空间一致性不受影响）；破坏 HR 目标保真度的强度增强（noise / blur /
    lowres / dropout）默认关闭，仅保留温和的亮度 / 对比度 / gamma。
    """

    # --- 生成侧默认值覆盖（字段语义同核心段） ---
    random_brightness_prob: float = 0.15
    random_contrast_prob  : float = 0.15
    random_contrast_range : List[float] = field(default_factory=lambda: [0.9, 1.1])
    random_gamma_prob     : float = 0.1
    random_gamma_range    : List[float] = field(default_factory=lambda: [0.9, 1.1])
    gaussian_noise_prob   : float = 0.0
    gaussian_blur_prob    : float = 0.0
    simulate_lowres_prob  : float = 0.0
