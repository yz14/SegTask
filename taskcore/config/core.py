"""Dataclass + YAML config. Each YAML file maps directly to nested dataclasses."""

from __future__ import annotations

import logging
import re
from dataclasses import MISSING, dataclass, field, fields, asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml

from .model_migration import install_flat_model_compat, route_legacy_model_dict

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


def resolve_selfattn_stage(entry, default_type: str):
    """把单个 selfattn 逐 level 条目解析为注意力类型或 None（该层关）。

    取值：0/'none'/'off' → None；1/'on'/'default' → default_type；
    'softmax' → 'softmax'；'linear' → 'linear'；'window' → 'window'；
    'grid' → 'grid'。其它一律报错。
    """
    s = str(entry).strip().lower()
    if s in ("0", "none", "off", "false"):
        return None
    if s in ("1", "on", "default", "true"):
        return default_type
    if s in ("softmax", "soft", "qkv"):
        return "softmax"
    if s in ("linear", "lin", "linear_qkv"):
        return "linear"
    if s in ("window", "win", "local"):
        return "window"
    if s in ("grid", "grid2", "sparse"):
        return "grid"
    raise ConfigError(
        f"Invalid selfattn stage entry {entry!r}; expected one of "
        "0/'none', 1/'default', 'softmax', 'linear', 'window', 'grid'.")


# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    """Data paths and preprocessing.

    字段分组与 configs/seg*.yaml 一一对应：
    后缀 → 路径/数据源 → 标签 → patch 抽取 → 强度归一化 → 划分 → DataLoader → 采样 → 缓存。
    """

    # ---- 文件后缀 ----
    # 后缀：单值或候选列表（取首个存在）。例：".nii.gz" 或 [".nii.gz", "-seg.nii.gz"]。
    image_suffix: Union[str, List[str]] = ".nii.gz"
    label_suffix: Union[str, List[str]] = ".nii.gz"
    bbox_suffix: Union[str, List[str]] = ".nii.gz"
    region_weight_suffix: Union[str, List[str]] = ".nii.gz"
    npz_suffix: str = ".npz"

    # ---- 路径 / 数据源 ----
    image_dir: str = ""
    label_dir: str = ""
    # 可选 ROI bbox 掩码目录；设置后按 bbox 裁剪。空=禁用。
    bbox_dir: str = ""
    # 可选逐样本区域权重目录（值 +1）。优先级：此目录 > loss.region_weights。
    region_weight_dir: str = ""
    # 预生成 npz 包目录；设置后忽略上述 NIfTI 目录，避免多 worker gzip OOM。
    npz_dir: str = ""

    # 第二批（粗标）预生成 npz 包目录。空=单批金标准（现有行为）。
    # 非空时与第一批（金标准）按 mix_ratio 在每个 train batch 内混合；
    # 副源仅用于训练，验证集始终仅取金标准。
    npz_dir_secondary: str = ""
    # True=npz 缓存缺失时启动期自动调用 make_data 生成；False=要求手动预生成。
    npz_auto_build: bool = True
    # True=image 无配对 label 时 warning 后丢弃；False（默认）=缺配对即报错
    # （与 bbox/rw 的 fail-fast 口径对齐，防标注目录写错时静默少训）。
    allow_unpaired: bool = False
    # 每个 train batch 内 [金标准, 粗标] 的整数权重比；仅 npz_dir_secondary 非空时生效。
    # 要求 batch_size 能被 sum(mix_ratio) 整除、两元素均 >= 1（保证每 batch 同时含两源）。
    mix_ratio: List[int] = field(default_factory=lambda: [1, 1])

    # ---- 标签 ----
    # 标签取值集（0=背景）。空=从数据自动探测。
    label_values: List[int] = field(default_factory=list)
    num_classes: int = 0  # 由 label_values 自动设置

    # ---- Patch 抽取 ----
    # 3D patch 尺寸 [D, H, W]。
    patch_size: List[int] = field(default_factory=lambda: [64, 128, 128])

    # Patch 抽取模式。示例："z_axis"（仅 z 滑块，H/W 全尺寸）、"2_5d"（D 折叠为通道驱动 2D UNet）。
    # 其他："cubic" 3 轴中心抽取；"whole" 整体 resize。
    patch_mode: str = "z_axis"
    # CPU resize 下采样抗混叠预滤波；默认关闭，保持 scipy zoom 旧数值。
    resize_antialias: bool = False

    # 多分辨率 FOV：各 scale 同中心抽更宽 FOV，resize 后作额外输入通道。
    # 示例：[1.0] 单通道；[1.0, 1.5, 2.0] 3 通道。cubic 作用 3 轴，z_axis 仅 z 轴。
    multi_res_scales: List[float] = field(default_factory=lambda: [1.0])

    # 增强过采样比：先抽 round(patch_size*ratio)，增强后中心裁回。1.0=禁用；affine/elastic 建议 1.4–1.5。
    aug_oversample_ratio: float = 1.0

    # 2.5D 多视图保持原生深度。True 时 dataset 抽最大 FOV cube，trainer 按 D_k 中心裁；强制 edge_pad。
    # 仅在 patch_mode='2_5d' + len(scales)>1 + aux_seg_supervision=True 生效。
    keep_native_view_depth: bool = False

    # 3D 多 FOV 懒加载单 cube（z_axis/cubic）。True：dataset 发单 cube，trainer 逐视图裁剪/重采样。
    # 约束：scales[0]==1.0；与 keep_native_view_depth 互斥；z_axis 强制 edge_pad。
    keep_native_multi_res: bool = False

    # z 轴边界填充（z_axis/2.5D）：恒为 "edge_pad"（边缘复制）。"stretch" 已废弃，
    # sync() 会警告并自动升级为 edge_pad（训练侧从未实际使用 stretch 几何）。
    z_boundary_mode: str = "edge_pad"

    # ---- 强度归一化 / spacing ----
    # 强度窗（CT HU）。
    intensity_min: float = -1024.0
    intensity_max: float = 1024.0
    # 归一化："minmax"→[0,1]；"zscore"→零均值单位方差。
    normalize  : str = "minmax"
    global_mean: float = 0.0
    global_std : float = 1.0

    # 物理 spacing 归一化开关（B1）。False=现状（不做任何 target-spacing 重采样）；
    # True=make_data 烘焙阶段把每卷重采样到 target_spacing，Predictor 推理前镜像
    # 重采样、概率图再回采到原分辨率。改开关须重新烘焙 npz 才生效。
    spacing_normalization: bool = False
    # 目标 spacing [sz, sy, sx]（numpy 轴序 (D,H,W)，单位 mm）。仅 spacing_normalization=True 时用。
    # None=make_data 扫描全数据集头信息取逐轴中位数（nnU-Net 式指纹）后自动落定。
    target_spacing: Optional[List[float]] = None

    # 样本排除清单路径（每行一个 pid）。空=不过滤。
    exclude_list: str = ""

    # ---- 训/验划分 ----
    val_ratio : float = 0.2
    split_seed: int = 42
    # split 取整：legacy 保留各 splitter 原有取整；unified 统一采用
    # half-up，并可选将实际索引落盘供复核。默认 legacy。
    split_rounding_mode: str = "legacy"
    split_manifest_path: str = ""
    # 按首个前景类分层；样本太少时回退随机。
    stratified_split: bool = True
    # 患者/组级划分：对 npz 文件名（去 .npz 后缀的 stem）做 re.search，取首个
    # 捕获组（无捕获组则取整个匹配）作为 group id；同组样本整体进 train 或
    # val，杜绝同一患者的多序列/多时点跨集泄漏。例如文件名 'P0123_T1.npz'
    # 用 r'^(P\d+)' 归组为 'P0123'。启用时组级随机划分（stratified_split 被
    # 覆盖并告警）；空字符串 = 关闭（默认，保持"一文件一样本"划分）。
    group_id_regex: str = ""

    # ---- DataLoader ----
    batch_size        : int = 2
    num_workers       : int = 4
    pin_memory        : bool = True
    persistent_workers: bool = True
    prefetch_factor   : int = 4

    # ---- 采样 ----
    # 前景过采样：中心点落在前景上的概率。
    foreground_oversample_ratio: float = 0.5

    # 每体积每 epoch 采样次数。
    samples_per_volume: int = 8

    # medium 验证 patch 位置的确定性网格覆盖（opt-in，仅作用于 val split）：
    # False（默认）= 现行为，逐样本确定性 RNG 随机位置（epoch 间一致但空间
    # 覆盖随机）；True = 卷内第 j 个样本取均匀网格位置（z 轴等距；cubic 用
    # Halton 序列），epoch 间指标可比性更强、噪声更小。train split 不受影响。
    val_grid_coverage: bool = False

    # ---- 缓存 ----
    # 缓存："none" 或 "memory"（每 worker LRU）。cache_max_volumes=0 不限（OOM 风险）。
    # 本 Config 默认 1（每 worker 仅缓最近 1 卷）；Dataset 构造签名的默认 0 只是
    # 直接实例化时的后备，经 Config 路径始终以此处为准。
    cache_mode       : str = "memory"
    cache_max_volumes: int = 1
    # 缓存存储粒度："fp32"（默认，缓存预处理后卷，取用零开销）或
    # "int16"（缓存原始 int16 卷，内存减半，每次取用重跑强度窗+归一，
    # 以 CPU 换 RAM；大数据集/多 worker 更划算）。仅影响 image 缓存
    # （label/rw 本就按原始粒度缓存），两模式产出逐位一致。
    cache_dtype      : str = "fp32"


# ---------------------------------------------------------------------------
# Augmentation configuration
# ---------------------------------------------------------------------------
@dataclass
class AugConfig:
    """GPU 数据增强。所有空间变换逐样本独立。

    字段分组与 configs/seg*.yaml 一一对应：总控 → 空间变换 → 强度变换。
    """

    # ---- 总控 ----
    enabled: bool = True

    # 就地增强快路径：True 时跳过入口的 image/label/weight_map clone，直接在
    # 调用方张量上做增强，省一份 batch 体积的瞬时显存（aug_oversample_ratio>1
    # 时更明显）。契约：调用方须保证传入张量增强后不再以"原始值"被使用（训练
    # 循环的 H2D 私有拷贝满足）。默认 False 保持防御性 clone 现状。
    inplace: bool = False

    # 强度增强后按增强前逐样本逐通道 min/max 夹取（nnU-Net 惯例），避免
    # brightness/contrast/noise 叠加产生分布外越界值、污染 gamma 语义。
    intensity_clamp: bool = True

    # weight_map 插值模式："nearest" 保持离散权重（默认，含连续手标 wmap）；
    # "bilinear" 仅在确认权重为平滑连续场且可接受插值混合时使用。
    wmap_interp_mode: str = "nearest"

    # ---- 空间变换（image + label 同步） ----
    random_flip_prob: float = 0.2
    random_flip_axes: List[int] = field(default_factory=lambda: [2, 3, 4])

    # Affine：小角旋转 + 缩放 + 平移合成单一仿射；与 elastic 形变进一步融合为
    # 单次 grid_sample（同时选中的样本也只插值一遍）。
    random_affine_prob : float = 0.3
    random_rotate_range: List[float] = field(default_factory=lambda: [-15.0, 15.0])
    # 缩放作用在采样网格上（反向语义）：>1 采样窗外扩→物体在输出中变小，
    # <1 反之。范围对称时增强效果等价，仅语义方向与直觉相反。
    random_scale_range : List[float] = field(default_factory=lambda: [0.85, 1.15])
    # 逐轴旋转角范围（(x,y,z)=(W,H,D) 三对 [lo,hi]，度）。None=三轴共用
    # random_rotate_range。CT 惯例：面内(绕 D 轴，即 z)可大角、出面(绕 W/H)宜小角。
    random_rotate_range_per_axis: Optional[List[List[float]]] = None
    # 各向异性长宽比校正：在物理各向同性坐标里做旋转（R←A⁻¹RA，A=diag(W,H,D)），
    # 消除各向异性 patch 上旋转混入的剪切/非均匀缩放。
    random_affine_aspect_correct: bool = True
    # 随机平移范围（affine_grid 归一化坐标，[-1,1] 跨整轴）；[0,0]=禁用。
    # 平移/旋转以 padding_mode='border' 复制边缘填充；边缘伪影可由
    # data.aug_oversample_ratio 的中心裁剪余量裁掉（幅度超出余量时启动期警告）。
    # 固有限制：z_axis 模式 H/W 取全尺寸、无余量可裁，面内旋转/平移的边缘
    # 复制伪影会保留在训练样本内。
    random_translate_range: List[float] = field(default_factory=lambda: [-0.1, 0.1])

    # 弹性形变（B-spline 随机位移场）。
    elastic_deform_prob : float = 0.2
    elastic_deform_sigma: float = 5.0   # 位移平滑度
    # 位移幅度（voxel，近似标称）：位移场由粗网格 randn 上采平滑得到，
    # 方差衰减使实际典型位移小于该值；多分辨率时另除以 max_scale。
    elastic_deform_alpha: float = 7.0
    # 弹性场采样口径：legacy 保持粗网格 randn 上采样；gaussian 使用
    # 高斯核平滑位移场。默认 legacy，避免改变既有训练分布。
    elastic_field_mode: str = "legacy"
    # True 时按每样本位移 RMS 归一化后再乘 alpha；默认保留旧的绝对幅度口径。
    elastic_normalize_displacement: bool = False

    # Grid dropout：随机遮挡矩形子区域。
    grid_dropout_prob : float = 0.0
    grid_dropout_ratio: float = 0.3
    grid_dropout_holes: int = 4

    # ---- 强度变换（仅 image） ----
    # 注意：brightness/noise 的幅值为绝对量，隐含 image≈[0,1]（minmax 归一）。
    # zscore（std=1）下同一数值的扰动相对偏弱且量纲不同，需自行按幅度改配。
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


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
@dataclass
class MedNeXtConfig:
    """MedNeXt 模块（Roy et al., MICCAI 2023；仅 unet.backbone=='mednext'）。

    残差倒瓶颈块：dwconv(k³, groups=C) → 通道级 GroupNorm → 1×1 扩张(×R) → GELU →
    1×1 压缩 → +残差。块内 norm/act 固定为通道级 GroupNorm + GELU（类似 convnext 固定
    LN+GELU）；unet 段的 norm_type/activation/dropout 对 mednext 块内无效（仅作用于
    stem / 上下采样 / decoder skip 投影 / 头）。
    档位 A：重采样复用通用 Downsample/Upsample（downsample_mode/upsample_mode 仍生效，
    且与 anisotropic_pooling 兼容，区别于 ConvNeXt LN-first 下采样）；MedNeXt 原生重采样
    残差块 + UpKern 为后续档位 B。
    """

    # 扩张比 R（MedNeXt 用 2/3/4/8；ConvNeXt 固定 4）。
    expand_ratio: int = 4
    # 深度卷积核大小（MedNeXt 用 3 或 5；ConvNeXt 固定 7）。
    kernel_size: int = 3
    # 训练时大核 + 多个空洞分支并行，推理时折叠为单一大核 depthwise conv，
    # 零额外推理开销。
    dilated_reparam: bool = False
    # 可选显式分支核大小 / dilation 覆盖；空列表=使用默认分支集。
    dilated_reparam_kernel_sizes: List[int] = field(default_factory=list)
    dilated_reparam_dilations: List[int] = field(default_factory=list)


@dataclass
class MultiRFConfig:
    """多感受野（空洞卷积多分支融合）MultiRF（仅 unet.backbone=='resnet'）。

    把选定 stage 的标准 ResNet 块替换为「多膨胀率并行分支 → 融合」的残差块，
    在同一分辨率下同时获得多个感受野（类 ASPP/Res2Net）。默认全关，逐位兼容现状。
    """

    # 多感受野（空洞卷积多分支融合）总开关；仅 backbone=='resnet'。
    enabled: bool = False
    # 各并行分支的膨胀率，必须含 1（守门支路，抗网格效应/保细管）。建议 HDC 互质组如 [1,2,3]。
    dilations: List[int] = field(default_factory=lambda: [1, 2, 3])
    # 通道处理："split"（各分支均分 out_ch，≈等成本，推荐）| "parallel"（各分支全宽 out_ch，≈N×成本）。
    mode: str = "split"
    # 分支融合："concat_proj"（concat→1×1，推荐）| "sum"（逐元素相加，需 parallel）| "se"（concat→SE→1×1）。
    fusion: str = "concat_proj"
    # 膨胀作用轴（仅 3D 有区别）："all" D/H/W 都膨胀；"hw" 只在 H/W 膨胀、z 恒 dilation=1（各向异性数据推荐）。
    # 2.5D（spatial_dims=2）下 z 已折进通道，自动等价 "hw"。
    axes: str = "hw"
    # 编码器逐 stage 开关（0/1）。长度须 == len(encoder_channels)。空=该侧全关。
    encoder_stages: List[int] = field(default_factory=list)
    # 解码器逐 level 开关（0/1）。长度须 == len(encoder_channels)-1。空=该侧全关。仅 decoder_type=='unet' 支持。
    decoder_stages: List[int] = field(default_factory=list)
    # ASPP 风格 per-branch norm+act：每条膨胀卷积分支在 concat/相加融合「之前」各自接
    # norm+act，使各感受野分支成为独立的非线性特征提取器（默认关、向后兼容）。
    # 注：split 模式下分支通道=out_ch//n_branches，与 norm_type='group' 组合时 norm_groups
    # 须能整除分支通道，否则 MultiRFBlock 显式报错（不自适配，由用户改 ch 或换 norm）。
    branch_norm_act: bool = False


@dataclass
class SelfAttentionConfig:
    """内容寻址自注意力（仅 unet.backbone=='resnet' + arch=='unet'）。

    在选定 stage 末尾追加一个自注意力残差块（拍平空间轴 → 多头 QKV → 残差），
    2.5D/3D 通用。提供真正的全局 token 交互（区别于 SE/ECA/CBAM/Coord 的
    通道/轴向重标定）。默认全关。
    """

    # 内容寻址自注意力总开关；仅 backbone=='resnet' + arch=='unet'。
    enabled: bool = False
    # 全局默认类型（逐 level 写 1 时沿用此值）：'softmax' 标准多头自注意力 O(N²)（全保真，放最深/
    # 瓶颈层）；'linear' O(N) 线性注意力（放次深层）。
    type: str = "softmax"
    # 头数（head_dim==-1 时使用）。
    num_heads: int = 4
    # !=-1 时 num_heads = channels // head_dim（覆盖 num_heads）。
    head_dim: int = -1
    # 输出投影 zero-init：训练初始注意力分支输出≈0、整块为恒等残差，几乎不扰动已调好的基线（强烈建议 true）。
    zero_init: bool = True
    # RoPE（参数无关）仅作用于 softmax self-attn；默认关，开后按 2D/3D 位置做旋转编码。
    rope: bool = False
    # 额外 FFN：GEGLU + zero-init 输出投影；默认关，开后为注意力后再接一层残差 MLP。
    ffn: bool = False
    ffn_ratio: float = 4.0
    # Window/Grid 注意力块大小；默认 7 保持不启用时无影响，启用时用于浅层大分辨率 token 分块。
    window_size: int = 7
    grid_size: int = 7
    # 编码器逐 stage 开关（可逐层指定类型）。长度须 == len(encoder_channels)。空=该侧全关。
    # 每个元素：0/'none'=该层关；'softmax'=标准 QKV；'linear'=线性 QKV；1=沿用全局 type。
    encoder_stages: List = field(default_factory=list)
    # 解码器逐 level 开关（同上取值）。长度须 == len(encoder_channels)-1。空=该侧全关。仅 decoder_type=='unet' 支持。
    decoder_stages: List = field(default_factory=list)


@dataclass
class UNetConfig:
    """本项目 UNet 专属旋钮（仅 arch=='unet' 消费；ADM/EDM2 按论文固定结构）。

    字段分组：backbone/块 → 归一化/激活/正则 → 拓扑 → 注意力 → 拓扑辅助头 →
    2.5D lift → 显存 → ConvNeXt 专属 → mednext/multirf/selfattn 模块子段。
    """

    # Backbone："resnet" | "convnext" | "mednext"。
    backbone: str = "resnet"

    # 残差块变体（仅 resnet）。示例："basic" 标准 ResNet；"r2plus1d" (1,3,3)+(3,1,1) 分解卷积（需 spatial_dims=3）。
    # 还有 "preact" / "bottleneck"。
    block_type: str = "basic"

    # ---- 归一化 / 激活 / 正则 ----
    # 归一化："batch" | "instance" | "group"。
    norm_type  : str = "instance"
    norm_groups: int = 8

    # 激活："relu" | "leakyrelu" | "gelu" | "swish"。
    activation: str = "leakyrelu"

    # stochastic depth（ResNet / ConvNeXt / MedNeXt 共用；默认 0 = 恒等，无行为变化）。
    drop_path_rate: float = 0.0

    # ---- 拓扑：decoder / 上下采样 / skip ----
    # Decoder 拓扑："unet" 对称（默认）；"unetpp" 嵌套稠密；"unet3p" 全尺度 skip。
    decoder_type: str = "unet"

    # UNet3+ 各分支通道数（仅 decoder_type=="unet3p"）。
    unet3p_cat_channels: int = 64

    # 下采样："conv" | "maxpool" | "avgpool" | "blurpool" | "pixelunshuffle"。
    downsample_mode: str = "conv"

    # 上采样："transpose" | "trilinear" | "nearest" | "pixelshuffle" | "carafe" | "dysample"。
    upsample_mode: str = "transpose"
    # 仅插值模式（'trilinear'/'nearest'）：在上采样精修 conv 之后再接 norm+act，使插值
    # 分支成为真正的非线性特征变换（否则 interpolate→conv 两层连续线性，直到下游 stage
    # 才有非线性）。默认关、向后兼容；其余上采样模式忽略该选项。
    upsample_norm_act: bool = False

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

    # ---- 注意力（块内 / skip 门控） ----
    # 块内注意力："none" | "se" | "eca" | "cbam" | "coord" | "lka" | "msca"。
    # lka = 大核注意力（VAN，DW5+DW7@dil3+1×1，等效感受野≈21³）；
    # msca = 多尺度条形核注意力（SegNeXt，逐轴 7/11/21 条形 DW 核，适合
    # 各向异性体数据与细长结构）。两者均纯卷积、无归一化层。
    attention_type: str = "none"

    se_reduction: int = 16

    # skip 连接上的 AttentionGate3D（Oktay 2018）；attn_gate_norm 控制其归一化。
    # "auto"（默认）跟随全局 norm_type（3D 小 batch 下避免 BatchNorm 统计噪）；
    # 也可显式指定 batch/instance/group。
    skip_attention: bool = False
    attn_gate_norm: str = "auto"

    # ---- 拓扑辅助头 ----
    # 中心线/距离场辅助头（拓扑感知多任务监督，仅 arch=='unet'）。与多 FOV aux_seg_supervision 相互独立，
    # 可同时开启。该头与主分割头同形（out_channels 相同、读最高分辨率 decoder 特征），仅训练期前向，
    # eval 不输出（零推理开销）。target 由 label 即时派生：见 aux_topo_target。
    aux_topo_head: bool = False
    # 辅助目标："centerline" 软骨架（clDice 同款，保拓扑）；"distance" 形态学到边界距离/血管半径场。
    aux_topo_target: str = "centerline"
    # 辅助头拓扑："linear" 1×1；"conv" ConvNormAct(3×3)→1×1（细结构推荐）。
    aux_topo_head_mode: str = "conv"

    # ---- 2.5D 专属 ----
    # Plan A 2.5D → 3D 提升（配合 block_type="r2plus1d"）。True 时 trainer 不折叠 D，模型输出 (B, num_fg, D, H, W)。
    # 与 data.keep_native_view_depth 互斥，仅在 2.5D 生效。
    lift_2_5d_to_3d: bool = False

    # ---- 显存 ----
    # 逐 encoder stage 检查点掩码（0/1）；空=沿用 grad_checkpointing 对所有 stage 生效（现状）；
    # 非空长度须==len(encoder_channels)，仅对为 1 的 stage 做检查点；仅在 grad_checkpointing=True 时生效。
    # 深层低分辨率 stage 可置 0，省重算开销。
    grad_ckpt_encoder_stages: List[int] = field(default_factory=list)
    # ---- ConvNeXt / MedNeXt 专属 ----
    # ConvNeXt-V2 / MedNeXt 可选 GRN（Global Response Normalization）；gamma/beta 零初始化，
    # 默认关，开启后初始仍近似恒等。
    grn_enabled: bool = False
    convnext_layer_scale_init: float = 1e-6  # <=0 禁用
    convnext_downsample_lnfirst: bool = True  # False 为通用 Downsample（消融用）

    # ---- 模块子段 ----
    mednext : MedNeXtConfig       = field(default_factory=MedNeXtConfig)
    multirf : MultiRFConfig       = field(default_factory=MultiRFConfig)
    selfattn: SelfAttentionConfig = field(default_factory=SelfAttentionConfig)


@dataclass
class ADMLinearAttentionConfig:
    """ADM 可选 LinearAttention（lucidrains 风格）。

    在指定级追加 Residual(PreNorm(LinearAttention))；O(N) 复杂度，可与
    adm.attention_levels 叠加。
    """

    # 追加 LinearAttention 的级索引；空=不追加（默认）。
    levels: List[int] = field(default_factory=list)
    # 头数。
    num_heads: int = 4
    # 每头维度。
    head_dim: int = 32


@dataclass
class ADMConfig:
    """ADM U-Net 专属（arch=="adm"，仅 2.5D）。"""

    # 带多头自注意力的级索引（0=顶，L-1=bottleneck）。空列表 [] = 不加注意力（默认）；
    # 传 None 才会用"最深两级"默认（见 models.adm_unet._resolve_attention_levels）。
    attention_levels: List[int] = field(default_factory=list)

    # 头数：仅在 num_head_channels==-1 时使用。
    num_heads: int = 4
    # !=-1 时 num_heads = channels // num_head_channels。
    num_head_channels: int = -1

    # LinearAttention 子段。
    linear_attention: ADMLinearAttentionConfig = field(
        default_factory=ADMLinearAttentionConfig)


@dataclass
class EDM2Config:
    """EDM2 U-Net 专属（arch=="edm2"，仅 2.5D）。"""

    # 带自注意力的级索引。空=默认仅 bottleneck。
    attention_levels: List[int] = field(default_factory=list)

    # heads = out_ch // channels_per_head。
    channels_per_head: int = 64

    # MP 残差/注意力/skip-cat 平衡系数（论文 Eq. 88 / 103）。
    res_balance: float = 0.3
    attn_balance: float = 0.3
    concat_balance: float = 0.5

    # 输出激活裁剪（论文 6.4）；<=0 禁用。
    clip_act: float = 256.0


@dataclass
class ModelConfig:
    """模型架构设置（D2：公共字段 + 按 arch 嵌套的 unet/adm/edm2 子段）。

    顶层只保留三 arch 共同消费的字段；arch 专属旋钮位于 ``unet.*`` /
    ``adm.*`` / ``edm2.*``。旧扁平路径（YAML 键 / ``--override`` /
    Python 属性 ``cfg.model.backbone`` 等）由
    :mod:`taskcore.config.model_migration` 的兼容层继续支持（读写等价、
    落盘统一新嵌套格式）。
    """

    # ---- 架构与规模（三 arch 公共） ----
    # 架构族。示例："unet"（本项目 UNet，读 unet.* 子段）、"adm"（ADM U-Net，仅 2.5D，读 adm.*）。
    # 还有 "edm2"（EDM2 U-Net，仅 2.5D，读 edm2.*）。
    arch: str = "unet"

    # 注：spatial_dims（2/3）与 in_channels 是由 patch_mode/multi_res_scales 等
    # 决定的"几何派生量"，不再作为可写字段/YAML 接口暴露（避免设了却被 sync 静默
    # 重写的困惑）。它们由 sync() 经 build_topology 算出，并以只读 property 暴露
    # （见类末尾），读 cfg.model.in_channels / spatial_dims 不变。

    # 每级 encoder 通道数，决定深度。例：[32, 64, 128, 256, 512] = 5 级。
    encoder_channels: List[int] = field(
        default_factory=lambda: [32, 64, 128, 256, 512])

    # 每级 block 数默认值（仅在 encoder/decoder_blocks_per_stage 都为空时使用）。
    blocks_per_level: int = 2

    # 逐级 block 数（nnU-Net ResEncUNet 风格）。非空时长度须与网络深度匹配。
    encoder_blocks_per_stage: List[int] = field(default_factory=list)
    decoder_blocks_per_stage: List[int] = field(default_factory=list)

    # nnU-Net ResEnc 预设："none" | "S" | "M" | "L" | "XL"。非 none 且 *_blocks_per_stage 为空时 sync() 自填。
    resenc_preset: str = "none"

    dropout: float = 0.0

    # ---- stem（三 arch 公共） ----
    # Stem / patch-embed："conv3" | "conv7" | "dual" | "patch2" | "patch4"。patchN 降 N 倍分辨率（UNet3D 主输出加上采样）。
    stem_mode: str = "conv3"

    # 多 FOV 上下文融合（仅 2.5D + n_views>1）。示例："shared_stem"（全部过同一 stem）、"multi_stem_proj"（Plan A，逐视图 stem→cat→1×1）。
    # 还有 "hierarchical"（Plan C，aux k 注入 encoder 第 k 级）。3D 模式下忽略。
    stem_fusion_mode: str = "multi_stem_proj"

    # ---- 监督头（三 arch 公共） ----
    # 深度监督：多 decoder 级输出预测。
    deep_supervision: bool = False

    # 多 FOV 辅助分割监督（仅 2.5D + len(multi_res_scales)>1 生效）。主头预 view 0，辅助 view k 输出 (B, num_fg*D, H, W)。
    # 损失权重见 loss.aux_supervision_weights（空则默认 0.5^k）。单视图/3D 不生效。
    aux_seg_supervision: bool = False

    # 辅助头拓扑："linear" 单 Conv1×1（Plan A 推荐）；"conv" ConvNormAct(3×3)→Conv1×1（Plan C 推荐）。
    aux_head_mode: str = "linear"

    # ---- 显存（三 arch 公共） ----
    # 梯度检查点（gradient checkpointing）：以约 +20~33% 算力换取激活显存大幅下降，可放大
    # 3D patch/batch（利于气道/血管等需大上下文的细结构）或更深更宽的 backbone。覆盖 encoder
    # 各 stage 与 decoder（unet/unetpp/unet3p）各节点，以及 ADM/EDM2（含扩散 backbone）的
    # 逐块（ResBlock/Attention/Block）包装；stem/上下采样/头不包裹（开销小）。
    # 反向重算前向 → 数值/收敛与关闭时一致（use_reentrant=False + preserve_rng_state 保
    # DropPath 复现）；eval/验证（no_grad）下零开销直通。默认关、逐位兼容现状。
    # 注：与 torch.compile 同时开启偶有图重编译开销，建议二者组合先小规模验证。
    grad_checkpointing: bool = False
    # 模型统一初始化策略；legacy 不覆盖各 backbone 的既有初始化。
    init_strategy: str = "legacy"
    # 可选扩大范围；默认关闭以保持原有 checkpoint 覆盖逐位不变。
    grad_ckpt_stem_downsample: bool = False
    grad_ckpt_decoder_branches: bool = False

    # ---- arch 专属嵌套段 ----
    unet: UNetConfig  = field(default_factory=UNetConfig)
    adm : ADMConfig   = field(default_factory=ADMConfig)
    edm2: EDM2Config  = field(default_factory=EDM2Config)

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


# 旧扁平接口兼容层：转发 property（cfg.model.backbone ↔ cfg.model.unet.backbone
# 读写等价）、扁平 kwargs 构造、老 checkpoint pickle 状态迁移。
install_flat_model_compat(ModelConfig)


# ---------------------------------------------------------------------------
# Loss configuration
# ---------------------------------------------------------------------------
@dataclass
class LossConfig:
    """损失函数设置。输出为逐类独立 sigmoid，每个前景类产生 (B, 1, D, H, W) 二值输出。

    字段分组与 configs/seg*.yaml 一一对应：损失名与权重 → 监督权重 → 逐损失参数 → 聚合方式。
    """

    # ---- 损失名与权重 ----
    # 损失名：常用 "dice_bce" 或 "dice_focal"；其他选项见 validate() 白名单。
    name: str = "dice_bce"

    # 复合损失权重 [loss1_w, loss2_w]。
    compound_weights: List[float] = field(default_factory=lambda: [1.0, 1.0])

    # 逐类损失权重（空=均匀）；长度 = num_fg_classes。
    class_weights: List[float] = field(default_factory=list)

    # 逐区域空间权重：按 label 取值一个权重（含 bg）。与 data.region_weight_dir 文件
    # 同一语义：最终权重 = 配置值 + 1（配 0 → 权重 1，背景默认 1）。
    # 例：[0.0, 1.0, 1.0, 0.0, 0.0] → label 1/2 位置损失×2，其余 ×1。空=禁用。
    region_weights: List[float] = field(default_factory=list)

    # 2.5D 损失聚合（仅 patch_mode=="2_5d"）："per_slice" 逐 slice 独立（空 slice Dice≈1 零梯度）；
    # "per_volume" 按整体在 (D,H,W) 上聚合（2.5D 推荐）。仅影响 Dice 系。
    slice_loss_reduction: str = "per_slice"

    # ---- 监督权重 ----
    # 深度监督逐级权重。
    deep_supervision_weights: List[float] = field(
        default_factory=lambda: [1.0, 0.5, 0.25, 0.125])

    # 2.5D 多 FOV 辅助头权重（仅 model.aux_seg_supervision=True）：长度 = n_views-1。空 = trainer 自填 0.5^k。
    aux_supervision_weights: List[float] = field(default_factory=list)

    # 中心线/距离场辅助头损失（仅 model.aux_topo_head=True）。
    # 权重：总损失 += aux_topo_weight * topo_loss。
    aux_topo_weight: float = 0.3
    # soft-skeleton 迭代 / 距离场最大腐蚀步数：2D 用 3，3D 取 3–10。
    aux_topo_iter: int = 3
    # 损失类型："auto"（centerline→soft-dice，distance→smooth_l1）| "dice" | "bce" | "smooth_l1" | "mse"。
    aux_topo_loss: str = "auto"

    # ---- 逐损失参数 ----
    # Dice 参数。
    dice_smooth: float = 1e-5
    dice_squared: bool = False

    # Focal 参数。
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # Tversky 参数（alpha=FP权重, beta=FN权重）。
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7

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

    # ---- 聚合方式 ----
    # True：全 batch+空间上汇总 TP/分母后一次除（nnU-Net Dice 默认）。作用于 Dice/Tversky/FocalTversky/GDL。
    # 稀疏前景 patch 训练下 per-sample Dice 在空 GT 类上恒≈1（抬高基线、稀释梯度），
    # 故默认取 nnU-Net 的 batch_dice=True。
    batch_dice: bool = True
    # 仅 per-sample：无 GT 的类从 dice 均值排除，避免空类≈1 掩盖错误。
    # 边界：2.5D per_slice reduction 下每切片是独立样本，无 GT 切片的损失
    # 被计为 0 后仍占均值分母（稀释总损失），而非被剔除。
    ignore_empty: bool = False


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """训练循环设置。

    字段分组与 configs/seg*.yaml 一一对应：基本 → DDP 多卡 → 优化器 → 学习率调度 →
    梯度 → 混合精度/编译/显存 → EMA/SWA → 验证与选模 → checkpoint 保存 → 日志 →
    恢复/预训练。
    """

    # ---- 基本 ----
    epochs: int = 200

    seed         : int = 42
    deterministic: bool = False

    # 要使用的**物理 GPU 卡号列表**（如 [0, 2, 5, 7]）。
    #   * [] / 单元素                 → 单卡（或 CPU）路径，行为与历史完全一致；
    #                                   非空单元素时即用该物理卡（混用机选卡）。
    #   * 长度 >= 2 且 CUDA 可用        → 每卡 spawn 一个进程跑 DDP，训练样本经
    #                                   DistributedSampler 切分、整卷验证按 rank
    #                                   切分后 all-reduce（数值与单卡严格相等）。
    # 仅占用列出的卡，与他人共用机器互不干扰。
    gpus: List[int] = field(default_factory=list)

    # 输出根目录（checkpoint/日志/可视化/监控）。
    output_dir: str = "outputs"

    # ---- DDP 多卡（DistributedDataParallel） ----
    # DDP 是否启用 find_unused_parameters（深监督 / 多头若有未参与反传的参数则需开）。
    # 默认 True 以保正确性；确认所有参数每步都参与时可设 False 提一点速度。
    ddp_find_unused_parameters: bool = True
    # DDP static_graph：告知 PyTorch 计算图逐步固定（参与反传的参数集合不变），
    # 免除每步的 unused-parameter 全图遍历，并启用通信/计算重叠等图级优化；
    # 与激活检查点的组合也更稳。建议与 ddp_find_unused_parameters=False 搭配。
    # 若模型存在逐步变化的控制流（某些参数时而参与时而不参与反传）则不可开。
    # 默认关（保持现状）。
    ddp_static_graph: bool = False
    # DDP gradient_as_bucket_view：让 param.grad 直接是通信 bucket 的 view，免掉
    # bucket 与 grad 的双份存储，DDP 下省约 1× 梯度显存（fp32 参数量级）。PyTorch
    # 官方支持，与 no_sync/梯度累积兼容。默认关（保持现状）。
    ddp_gradient_as_bucket_view: bool = False
    # DDP rendezvous 端口。0 = 启动时自动挑选空闲端口（混用机避免端口冲突）。
    ddp_master_port: int = 0
    # DDP 下是否按 world_size 把每卡 DataLoader 的 num_workers 平摊到 1/world_size
    # （向下取整、至少 1）。每个 rank 是独立进程、各自 fork num_workers 个 worker 且各
    # 持一份逐 worker LRU 卷缓存——不分摊则 worker 进程数与缓存 RAM 都随卡数线性翻倍
    # （8 卡混用机上易触发 CPU 超额订阅 / 换页抖动 / 内核 soft-lockup）。分摊后**全机
    # 聚合** worker 数与缓存 RAM 与单卡基线一致。默认 True；置 False 保留旧行为（每卡满额）。
    ddp_scale_dataloader_per_rank: bool = True
    # NCCL collective 超时（分钟）。卡住的集合通信超过此时长会被 watchdog abort，
    # 避免某 rank 因 peer 已死而无限等待、永久挂起占显存。默认 30 分钟。
    ddp_timeout_minutes: int = 30
    # ZeRO-1 优化器状态分片（torch.distributed.optim.ZeroRedundancyOptimizer）：
    # DDP 下把 AdamW 的 2× 参数 fp32 状态均分到 world_size 张卡，每卡省
    # 2×参数×4B×(1−1/N)；step 后各 rank broadcast 自己分片的参数（数值与普通
    # DDP+AdamW 严格等价）。checkpoint 保存时自动 consolidate 到 rank0（全 rank
    # 集合操作）。单卡 / 非 DDP 下该开关被忽略并告警。默认关。
    zero_redundancy_optimizer: bool = False

    # ---- 优化器 ----
    # 优化器："adam" | "adamw" | "sgd"。
    optimizer   : str = "adamw"
    # CUDA 下用 fused AdamW，单 kernel 更新全部参数。
    adamw_fused : bool = True
    lr          : float = 1e-3
    weight_decay: float = 1e-4
    momentum    : float = 0.99   # 仅 SGD
    nesterov    : bool = True    # 仅 SGD

    # ---- 学习率调度 ----
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
    plateau_patience     : int = 10   # 单位为验证次数（非 epoch），见 early_stopping 注。
    plateau_factor       : float = 0.5

    # ---- 梯度 ----
    # 梯度累积（有效 batch = batch_size * accum_steps）。
    # 注意：比值型 batch 池化损失（batch_dice/GDL/Tversky 等）的统计窗口
    # 仍是单个 micro-batch（=batch_size），不随 accum 扩大（单卡亦然）。
    grad_accum_steps: int = 1

    # 梯度裁剪。
    grad_clip_norm: float = 12.0
    # 梯度裁剪范数的懒同步：clip_grad_norm_ 返回的范数是 GPU 标量，默认每个
    # 优化步 float(gn) 会强制一次 D2H 同步。开启后，仅在“fp16+GradScaler 且
    # 健康监控关闭”（该值无任何消费者：非有限守护由 scaler 自行完成）时跳过
    # 同步；裁剪本身照常执行，bf16/fp32 路径（范数参与非有限守护）与健康
    # 监控开启时行为完全不变。数值等价。默认关（保持现状）。
    grad_norm_lazy_sync: bool = False

    # ---- 混合精度 / 编译 / 显存 ----
    # AMP。amp_dtype 示例："float16"（需 GradScaler）、"bfloat16"（Ampere+，无需 scaler）。还有 "auto"（探测 BF16 否则回退 fp16）。
    use_amp  : bool = True
    amp_dtype: str = "float16"

    # torch.compile："none" | "default" | "reduce-overhead" | "max-autotune"。
    compile_mode: str = "none"
    # 可选 channels_last 内存格式；数值等价，Ampere+ 上 3D conv 可能提速但不保证正收益（需 benchmark）；默认关。
    channels_last: bool = False

    # 训练 batch 的 GPU 预取：用独立 CUDA copy stream 把下一个 batch 的 H2D
    # 拷贝与当前 batch 的前向/反向重叠，隐藏传输延迟。数值完全等价（仅改变
    # 拷贝与计算的重叠方式）。需 data.pin_memory=True 才有真实收益（pageable
    # 内存下异步拷贝退化为同步）。GPU 增强 / 大 patch 场景收益 ~5-15%。默认关。
    prefetch_to_gpu: bool = False

    # CUDA caching allocator 的 expandable segments（PyTorch 2.1+）：多分辨率视图 /
    # oversample 裁剪 / 滑窗尾窗等形状多变场景下显著缓解显存碎片（reserved >>
    # allocated 即碎片征兆，epoch 摘要有打印）。实现方式为进程启动早期注入
    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True（已有该环境变量时不覆盖）。
    # 默认关，行为与现状一致；个别旧驱动不支持时 PyTorch 自动回退并告警。
    cuda_expandable_segments: bool = False

    # 整卷验证前后各调一次 torch.cuda.empty_cache()：把训练激活占住的 cached
    # blocks 归还，避免滑窗大累加器因碎片 OOM。只影响 allocator、不影响数值；
    # 代价是验证前后各一次微小停顿。None（默认）= 自动：val_metric_mode='high'
    # 时开（整卷滑窗最需要连续显存），'medium' 时关；也可显式 True/False 覆盖。
    val_empty_cache: Optional[bool] = None

    # ---- EMA / SWA 权重平均 ----
    use_ema  : bool = True
    ema_decay: float = 0.999
    # EMA decay warmup（timm 式 ramp）：早期用 min(decay, (1+step)/(10+step))，
    # 避免随机初始权重长时间拖累 shadow、导致早期 best 判定失真。
    ema_warmup: bool = True
    # EMA shadow 存放位置：""（默认，跟随模型所在设备，行为不变）| "cpu"（shadow /
    # backup 常驻 pinned CPU，省 1× 参数量的 GPU 常驻显存）。"cpu" 模式每步 update
    # 多一次 GPU→CPU 参数拷贝（异步 + 一次流同步），数学与 "" 严格等价；验证换入 /
    # checkpoint 保存均自动跨设备拷贝。仅在显存吃紧时建议开启。
    ema_device: str = ""

    # SWA 尾段等权权重平均（Izmailov 2018，opt-in，与 EMA 正交可叠加）。
    # True 时从 swa_start_ratio*epochs 起每 epoch 将在线权重纳入等权平均
    # （shadow 常驻 CPU、fp32 累积，零显存开销，不影响训练/选模）；训练
    # 收尾时换入平均权重 → 重估 BN running stats（模型无 BN 则跳过）→
    # 跑一次验证并另存 swa_model.pth，best_model.pth 选模逻辑不变。
    swa_enabled: bool = False
    # 开始平均的训练进度比例（(0,1) 开区间）：0.75 = 最后 25% epoch 参与。
    swa_start_ratio: float = 0.75
    # 收尾 BN 统计重估用的 train batch 数（<=0 跳过重估；仅对含 BatchNorm
    # 的模型有效，instance/group norm 无 running stats 自动跳过）。
    swa_bn_update_steps: int = 50

    # ---- 验证与选模 ----
    val_every: int = 1

    # 选模严格度（与 save_best_criterion 正交：criterion 决定"看哪个指标"，
    # 本项决定"指标在什么预测上算"）：
    #   * "medium" — 现状：在 val_loader 的随机 patch/切片上前向并算指标，快但
    #                非整卷，z 向上下文被切断，指标偏乐观/抖动。
    #   * "high"   — 严格：对每个 val 整卷做与部署一致的滑窗推理后再算指标，
    #                最可靠但更慢（每次验证多一遍整卷推理）。npz 无物理 z-spacing，
    #                故 high 不启用 predict.z_interleave，其余几何与推理一致。
    # 默认 "medium" 保持既有行为不变。
    val_metric_mode: str = "medium"

    # 混合验证调度（仅 val_metric_mode='high' 时生效）：每 N 次验证才跑一次
    # 整卷滑窗 high 评估（及末 epoch 必跑），其余验证轮次跑 medium patch 评估
    # 作趋势监控。选模/早停/plateau 只看 high 轮次的指标（medium 与 high 口径
    # 不同，混用会污染 best 追踪）；早停/plateau 的"验证次数"单位相应变为
    # high 次数。高轮次判定按 epoch 推导（(epoch+1) % (val_every*N) == 0），
    # resume 后调度相位不变。1 = 每次验证都跑 high（既有行为）。
    val_high_interval: int = 1

    # 整卷验证（val_metric_mode='high'）指标计算前先按 pred∪GT 并集 bbox 裁剪：
    # dice/iou/recall/precision/vol_sim 只依赖 TP/FP/FN（全部落在并集 bbox 内），
    # 裁剪后严格等价；MCC 的 TN 通过按整卷形状回传总体素数保持严格等价；
    # surface_dice 按 tolerance+1 外扩边距后同样严格等价。前景占比小的整卷上
    # surface_dice 的 3D maxpool 可省一个量级计算与显存。默认关，行为与现状一致。
    val_metric_bbox_crop: bool = False

    # 整卷验证 RAM 缓存（仅 val_metric_mode='high' 时生效）：首次整卷验证把本
    # rank 分片的预处理 image（fp32）/原始 label（int16）/z_spacing 常驻 RAM，
    # 后续验证轮免磁盘读取与预处理。RAM 占用 ≈ 分片 val 卷总体素 × 6B/voxel
    # （逐 rank，不随 epoch 增长），需按数据规模评估。默认关，行为与现状一致。
    val_volume_cache: bool = False

    # 选模标准（互斥）：
    #   * "loss"              → val_base_loss ↓
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
    # >0 时启用物理空间 (mm) 各向异性欧氏 NSD（同 MONAI Surface Dice），
    # 覆盖上面的 voxel-Chebyshev 容差；需可解析的 spacing（data.spacing_normalization
    # + target_spacing / manifest 回读），否则告警并回退 voxel 容差。0=沿用像素容差。
    surface_dice_tolerance_mm: float = 0.0
    # 组合标准下 combined = (1-w)*dice + w*surface_dice。
    surface_dice_weight: float = 0.5
    # 任务化推荐预设：非空时由 sync() 覆盖上面三项 (criterion / tolerance / weight)。
    # 仅作"任务名 → 经验上推荐组合"的一键映射，便于复用与切任务。空串 = 不启用，
    # 完全沿用用户显式设置的三个字段。可选值见 ``_SAVE_BEST_PRESETS``：
    #   lung / vessel / airway / bone_multi / lymph_node / lesion_small /
    #   oar_multi / heart_chamber / bone_lung_combined
    save_best_preset: str = ""

    # 提前停止（0=禁用）。单位是“验证次数”而非 epoch：val_every>1 时
    # N 次无提升 ≈ N*val_every 个 epoch（plateau_patience 同理）。
    early_stopping: int = 0

    # ---- checkpoint 保存 ----
    save_every      : int = 10
    # 周期 checkpoint 保留策略：仅保留最近 k 个 checkpoint_epoch_*.pth，更早的
    # 自动删除（best_model.pth 不受影响）。<=0 = 不清理（保留全部）。
    save_keep_last  : int = 3
    # 异步 checkpoint 保存（仅 rank0）：权重先深拷到 CPU，后台线程 torch.save，
    # 训练主循环不被写盘阻塞（大模型/频繁 save 时可省数秒到数十秒/次）；
    # 代价是保存时刻额外一份 CPU 内存快照。默认关（保持同步写盘旧行为）。
    save_async      : bool = False

    # ---- 日志 ----
    log_every: int = 10
    vis_every: int = 10

    # ---- 恢复 / 预训练 ----
    # Resume：从 checkpoint 完整恢复（model/EMA/optimizer/scheduler/scaler/epoch/RNG）。
    # 路径不存在即报错（fail-fast，防静默从头训；全任务统一口径）。
    resume: str = ""

    # Pretrain：仅加载 model 权重作初始化。若同时设置了 resume 则 pretrain 被忽略；
    # 路径不存在即报错。
    pretrain: str = ""

    # strict 加载；默认 False 允许 head 形状不一致。
    pretrain_strict: bool = False

    # checkpoint 含 EMA shadow 时是否优先用 EMA 作初始。默认 False。
    pretrain_load_ema: bool = False

    # 仅对 backbone=='mednext' 有效：将预训练 checkpoint 的深度卷积权重按当前
    # mednext_kernel_size 做 UpKern 插值迁移（k=3→k=5 等）。
    pretrain_upkern: bool = False
    # UpKern 插值后按空间核总幅度归一化；默认关闭，保持旧迁移值。
    pretrain_upkern_normalize: bool = False
    pretrain_allow_geometry_mismatch: bool = False

    # 推理前是否将 MedNeXt 的可重参数化深度卷积折叠为 deploy 形态；默认关。
    # 开启后，io.run_inference 会在 load_state_dict 之后、device 转移之前先折叠。
    reparam_deploy: bool = False

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
    "loss":              ("val_base_loss",   "min"),
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
    """推理设置（z 轴滑动窗口）。

    字段分组与 configs/seg*.yaml 一一对应：滑窗核心 → TTA 与阈值 → 精度/显存 →
    提速开关 → AdaBN 域适应 → 2.5D 专属 z 交错 → 输出。
    """

    # ---- 滑窗核心 ----
    # z 轴重叠比（0.0 = 不重叠，0.5 = 50%）。
    z_overlap: float = 0.5

    # cubic 模式 H/W 轴重叠比；None（默认）= 沿用 z_overlap（三轴同值，
    # 现状不变）。各向异性数据（z 分辨率低、面内分辨率高）时可为面内轴
    # 单独设置（如 z_overlap=0.5, hw_overlap=0.25 减少窗口数提速）。
    # 仅 patch_mode='cubic' 生效；z_axis / 2_5d / whole 不受影响。
    hw_overlap: Optional[float] = None

    # 重叠区融合："gaussian" 或 "average"。
    blend_mode: str = "gaussian"

    # 推理 batch 大小。
    batch_size: int = 2

    # ---- TTA 与阈值 ----
    # TTA flip。
    tta_flip: bool = False

    # TTA flip 变体的批量化前向块大小（单卷推理提速；仅 tta_flip=True 时生效）。
    #   将多个 flip 变体沿 batch 轴 torch.cat 成一次前向，逐像素等价于串行，仅减少
    #   前向次数。3D 7 种 flip → ceil(7/tta_batch_size)+1 次前向；2.5D 3 种 → 同理。
    # None  — 退化为 predict.batch_size。
    # 显存提醒：单次前向样本数 ≈ batch_size * tta_batch_size，显存随之线性上升；
    #   显存吃紧时调小（如 2）。AdaBN per_volume 估计阶段会自动退回串行以保 BN 统计一致。
    tta_batch_size: Optional[int] = None

    # sigmoid 二值化阈值：标量（全类共享）或逐前景类列表（长度 = num_fg，与
    # label_values[1:] 一一对应）。one-vs-rest sigmoid 下不同类的最优操作点常差异
    # 很大（小结构类宜偏低阈值）。
    threshold: Union[float, List[float]] = 0.5

    # ---- 精度 / 显存 ----
    # 滑窗概率累加器 dtype："fp32"（默认）| "fp16"。大卷 × 多类时 fp16 使
    # acc_pred 显存减半；blend 权重归一后精度足够（nnU-Net 同款做法）。
    acc_dtype: str = "fp32"

    # GPU 常驻整卷张量（滑窗 GPU builder 路径的 vol_t）存储 dtype："fp32"（默认）
    # | "fp16"。fp16 使整卷常驻显存减半；取窗时 builder 会按窗 .float() 升回
    # fp32 再插值/前向，仅存储精度为 fp16（归一化后输入动态范围小，量化误差
    # ~1e-3 相对量级）。仅影响 GPU 全流 builder 路径；CPU 退化路径不受影响。
    vol_dtype: str = "fp32"

    # 累加器落 CPU 的逃生门：大卷 × 多类在消费级卡 OOM 时开启（每个 batch 多一次
    # GPU→CPU 拷贝，用速度换显存）。
    accumulate_on_cpu: bool = False

    # ---- 提速开关 ----
    # 独立 CLI 推理（predictor.io.run_inference）开启 cudnn.benchmark：滑窗窗口
    # 形状固定，让 cuDNN 首个 batch 自动选最优卷积算法（训练入口经
    # seed_everything 已默认开启，仅独立推理入口缺此设置）。默认关，行为与现状
    # 一致；开启后首卷首个 batch 有一次性 autotune 开销。
    cudnn_benchmark: bool = False

    # 推理前向用 torch.inference_mode() 替代 torch.no_grad()：在免记梯度之上
    # 进一步免除 autograd 的 version-counter/view 追踪簿记，纯速度收益、
    # 数值完全等价。AdaBN per_volume 的 BN 重估阶段自动回退 no_grad（避免
    # 对 BN buffer 原地更新的兼容性风险）。默认关（保持现状）。
    use_inference_mode: bool = False

    # 推理侧 channels_last 内存排布（与 TrainConfig.channels_last 同义，作用于
    # 推理模型与输入窗口）：数值等价，Ampere+ 上 conv 可能提速（需 benchmark
    # 验证正收益）。注：训练内整卷验证与训练共享同一模型对象，开启后排布
    # 转换会持续到训练侧（仍数值等价）；训练侧同样需求请用 train.channels_last。
    # 默认关（保持现状）。
    channels_last: bool = False

    # 滑窗跳过纯背景窗口（z_axis / 2_5d / cubic 路径）：取窗前先看该窗在
    # **归一化后**体素上的最大值，若 <= skip_empty_threshold 则不前向、不累加
    # （该区域概率保持 0 = 背景）。CT + minmax 归一化下，<= intensity_min 的
    # 空气体素被钳到 0，默认阈值 0.0 即“整窗全为钳底空气”才跳过，判据保守；
    # 其它归一化（z-score 等）需自行按归一化后的空气值调阈。注意与不跳过相比
    # 并非逐位等价：被跳过窗覆盖的重叠区不再被该窗的预测稀释（等效于空窗预测
    # 恒为背景 0）——对训练好的模型，空气窗预测本应 ≈0，实际差异可忽略。
    # whole 模式无滑窗，不受影响。默认关（保持现状）。
    skip_empty_windows: bool = False
    # 跳窗判据：窗内（归一化后）最大值 <= 此值才跳过。仅 skip_empty_windows
    # 开启时生效。
    skip_empty_threshold: float = 0.0

    # ---- AdaBN 域适应 ----
    # 测试时自适应 BatchNorm (AdaBN)：推理阶段用目标域前向重估 BN running stats，
    # 无需标签、不重训，针对跨数据集域漂移导致的假阳。仅当 model.norm_type=='batch'
    # 时有效（instance/group norm 为 no-op）。
    adabn_enabled: bool = False

    # 'global'    — 用 adabn_num_volumes 卷目标域整卷预热一次 BN 统计，全程复用。
    # 'per_volume'— 每卷推理前用该卷自身重估 BN，再冻结预测（transductive BN）。
    adabn_mode: str = "global"

    # global 模式预热用的目标域整卷数；per_volume 模式忽略。
    adabn_num_volumes: int = 8

    # BN 估计期的滑窗抽样比（(0, 1]）：<1 时估计前向只跑约 ratio 比例的
    # 窗口（确定性等步长抽样，首窗恒保留），估 BN 均值/方差通常 1/4
    # 窗口已足够稳定，可把 per_volume 的额外一遍推理成本从 2× 降到
    # ≈1.25×（global 预热同比例提速）。真实预测路径不受影响；whole 模式
    # 单前向无窗可抽，忽略此项。默认 1.0（全窗估计，现状不变）。
    adabn_sample_ratio: float = 1.0

    # ---- 2.5D 专属：z 交错 ----
    # z 轴交错多流推理（仅 2.5D）：按 z 拆 k 个子体 (slices i,i+k,...)，独立推理后缝回原 z。
    # 动机：加宽 z 感受野。警告：子流表现为 k * z_spacing。
    z_interleave_enabled: bool = False

    # k 按物理 z 间距（mm）选择。thresholds 升序，factors 长度 = len(thresholds)+1（含 fallback）。
    # 默认：z<=1.0 → k=3；1.0<z<=1.5 → k=2；z>1.5 → k=1。
    z_interleave_thresholds: List[float] = field(
        default_factory=lambda: [1.0, 1.5])
    z_interleave_factors: List[int] = field(
        default_factory=lambda: [3, 2, 1])

    # ---- 输出 ----
    # 预测输出目录。
    output_dir: str = "predictions"

    # 是否保存概率图（在二值 mask 之外）。
    save_probabilities: bool = False


# ---------------------------------------------------------------------------
# Visualization (TODO #2)
# ---------------------------------------------------------------------------
@dataclass
class VisConfig:
    """全流程可视化分析工具开关。``enabled=False`` 时零开销、零副作用。

    开启后，训练启动时自动导出一份自包含 HTML（数据流 / 模型流 / 预测流三视图），
    用于人工核对"数据流与模型架构是否符合 yaml、是否有优化空间"。生成过程仅用
    CPU dummy 张量、不读盘、不依赖 GPU 与真实数据。
    """

    # 总开关：关闭时 train.py 完全跳过可视化逻辑。
    enabled: bool = False
    # 输出目录；空串 → 落到 ``train.output_dir/visualization``。
    output_dir: str = ""
    # 输出文件名（单文件、三标签页）。
    filename: str = "pipeline_vis.html"
    # 要生成的视图子集；顺序即标签页顺序。
    flows: List[str] = field(
        default_factory=lambda: ["data", "model", "predict"])
    # 是否用 CPU dummy 前向 + hook 追踪模型流真实张量形状；False 退化为纯结构图。
    trace_shapes: bool = True
    # 详情面板每节点最多展示的参数条数（防爆）。
    max_detail_params: int = 200


# ---------------------------------------------------------------------------
# Training monitor (TODO #2)
# ---------------------------------------------------------------------------
@dataclass
class MonitorConfig:
    """训练过程监测仪表盘开关。``enabled=False`` 时零开销、零副作用。

    与 ``VisConfig``（静态架构 / 数据流图）正交：本配置控制**训练时序**监测
    —— 逐 epoch 落盘损失 / 指标 / 学习率 / 显存到 ``metrics.jsonl``，并周期性
    重渲染一份自包含 HTML 仪表盘（曲线 + best 模型指标卡片），支持训练中实时
    刷新与训练后复看 / 多 run 对比。落盘与渲染均封装在 ``taskcore.monitor``，
    失败被隔离、不会中断训练。
    """

    # 总开关：关闭时 Trainer 完全跳过监测逻辑。
    enabled: bool = False
    # 输出目录；空串 → 落到 ``train.output_dir/monitor``。
    output_dir: str = ""
    # 仪表盘 HTML 文件名。
    filename: str = "training_monitor.html"
    # 每多少个 epoch 重渲染一次 HTML（指标 jsonl 每 epoch 都写）；刷新 best 与
    # 训练收尾时强制重渲染，不受此节流影响。
    update_every: int = 1
    # 仪表盘内嵌 JS 自动重载间隔（秒）；<=0 关闭自动刷新（训练后静态复看）。
    auto_reload_seconds: int = 10
    # 本 run 名称（图例 / 标题用）；空串 → 取 output_dir 末级目录名。
    run_name: str = ""
    # 训练结束后额外渲染一份多 run 对比 HTML 的参照 run 目录列表；空则不生成。
    compare_runs: List[str] = field(default_factory=list)

    # 模型健康监测：逐 epoch 聚合梯度范数 / 非有限步计数 / 裁剪比例 / 权重范数 /
    # AMP scaler 标度等「训练是否正常」的轻量指标，并入 train 指标随仪表盘展示。
    # 成本极低（开 grad_clip 时复用其已算出的范数；仅 rank0 记录），失败被隔离。
    health_monitor: bool = True
    # 未开启 grad_clip_norm 时，是否仍在每个优化步边界手动算一次全局梯度范数。
    # 关闭后未裁剪场景将不记录 grad_norm（彻底零额外开销）。
    health_grad_norm_when_no_clip: bool = True
    # 全局 update/weight 比值（‖Δw‖/‖w‖，Karpathy 经典健康信号，健康区间约 1e-3）：
    # 每 epoch 仅在「第一个优化步边界」测一次——step 前后各算一次全参数范数。
    # 计算可忽略，但需对参数做一次瞬时 clone（峰值额外显存≈一份参数大小，用完即释放），
    # 故默认关闭，按需开启用于诊断 lr / 优化器是否合理。
    health_update_ratio: bool = False


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """顶层配置，聚合所有子配置。"""

    data   : DataConfig    = field(default_factory=DataConfig)
    augment: AugConfig     = field(default_factory=AugConfig)
    model  : ModelConfig   = field(default_factory=ModelConfig)
    train  : TrainConfig   = field(default_factory=TrainConfig)
    vis    : VisConfig     = field(default_factory=VisConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)

    def sync(self) -> None:
        """同步跨子配置的对应字段。

        所有"模型几何派生量"（``in_channels`` / ``spatial_dims``）由
        ``taskcore.models.topology.build_topology(self)`` 一次性算出，写入
        ``ModelConfig`` 的私有 backing 字段（对外是只读 property）。本方法仅保留
        "非派生"职责（``num_classes`` 推断、``z_boundary_mode`` 自动升级、resenc
        preset、save_best 预设）。
        """
        if self.data.label_values and self.data.num_classes == 0:
            self.data.num_classes = len(self.data.label_values)

        # z_boundary_mode='stretch' 已废弃：训练侧 dataset 恒走 edge-pad 几何
        # （stretch 在训练抽取中无分支），而推理侧会生效，薄卷（D <
        # patch 深度）时造成训练-推理几何不一致。统一自动升级为 edge_pad。
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
        from ..models.topology import build_topology
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
            if str(mc.unet.decoder_type).lower() == "unetpp":
                # UNet++ has triangular nested nodes; one value is an
                # explicit broadcast across all nodes.
                mc.decoder_blocks_per_stage = [1]
            else:
                mc.decoder_blocks_per_stage = [1] * (n_levels - 1)

    def validate(self, *, skip: "Optional[Set[str]]" = None) -> None:
        """校验配置一致性（按 section 拆分；非法配置抛 ConfigError）。

        * ``skip`` — 跳过指定 section 校验器名（``model`` / ``augment`` /
          ``data`` / ``2_5d`` / ``train`` / ``monitor``）。未知名（如组合式
          任务传入的 ``loss`` / ``predict``）静默忽略，由任务层自行处理。
        """
        skip = skip or set()
        validators = (
            ("model", self._validate_model),
            ("augment", self._validate_augment),
            ("data", self._validate_data),
            ("2_5d", self._validate_2_5d),
            ("train", self._validate_train),
            ("monitor", self._validate_monitor),
        )
        for name, fn in validators:
            if name not in skip:
                fn()
        if self.data.num_classes < 2:
            logger.warning("num_classes=%d < 2, will auto-detect from data.",
                           self.data.num_classes)

    def _validate_model(self) -> None:
        """model.* 架构选项与逐级拓扑长度校验。"""
        arch = str(self.model.arch).lower()
        _require(
            arch in ("unet", "adm", "edm2"),
            f"Invalid model.arch: {arch!r}. Valid: 'unet' | 'adm' | 'edm2'.")
        if arch == "unet":
            _require(
                self.model.unet.backbone in ("resnet", "convnext", "mednext"),
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
                self.model.unet.attn_gate_norm in ("auto", "batch", "instance", "group"),
                f"Invalid attn_gate_norm: {self.model.unet.attn_gate_norm}")
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
        from .section_validators import (
            validate_encoder_decoder_stage_lengths,
            validate_stem_modes,
        )
        validate_stem_modes(self)
        # 中心线/距离场辅助头校验。
        if self.model.unet.aux_topo_head:
            _require(
                arch == "unet",
                "aux_topo_head=True is only supported with model.arch=='unet'; "
                f"got arch={arch!r}.")
            _require(
                self.model.unet.aux_topo_target in ("centerline", "distance"),
                f"Invalid aux_topo_target: {self.model.unet.aux_topo_target!r}. "
                "Valid: 'centerline' | 'distance'.")
            _require(
                self.model.unet.aux_topo_head_mode in ("linear", "conv"),
                f"Invalid aux_topo_head_mode: {self.model.unet.aux_topo_head_mode!r}")
        # 仅 arch=='unet' 使用以下 backbone/block/decoder/r2plus1d/ResEnc/注意力选项。
        if arch == "unet":
            _require(
                self.model.unet.attention_type in (
                "none", "se", "eca", "cbam", "coord", "lka", "msca",
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
            # 与 sync()._apply_resenc_preset 的 .lower() 查表口径一致（大小写不敏感）。
            _require(
                str(self.model.resenc_preset or "none").lower()
                in ("none", "s", "m", "l", "xl"),
                f"Invalid resenc_preset: {self.model.resenc_preset}")
            # MedNeXt 块超参（仅 backbone=='mednext' 生效；默认值对其他 backbone 无害）。
            _require(
                self.model.unet.mednext.kernel_size in (3, 5, 7),
                f"Invalid mednext_kernel_size: {self.model.unet.mednext.kernel_size}; "
                "valid: 3 | 5 | 7.")
            _require(
                self.model.unet.mednext.expand_ratio >= 1,
                f"mednext_expand_ratio must be >= 1; got "
                f"{self.model.unet.mednext.expand_ratio}.")
            if self.model.unet.mednext.dilated_reparam:
                _require(
                    self.model.unet.backbone == "mednext",
                    "model.mednext_dilated_reparam=True requires "
                    "backbone='mednext'.")
                bk = list(self.model.unet.mednext.dilated_reparam_kernel_sizes)
                bd = list(self.model.unet.mednext.dilated_reparam_dilations)
                if bk or bd:
                    _require(
                        bk and bd and len(bk) == len(bd),
                        "mednext_dilated_reparam branch overrides require "
                        "both kernel_sizes and dilations with the same "
                        "non-zero length.")
                    for k, d in zip(bk, bd):
                        eff = (int(k) - 1) * int(d) + 1
                        _require(
                            int(k) % 2 == 1,
                            f"mednext_dilated_reparam branch kernel must be "
                            f"odd; got {k}.")
                        _require(
                            int(d) >= 1,
                            f"mednext_dilated_reparam branch dilation must "
                            f"be >= 1; got {d}.")
                        _require(
                            eff % 2 == 1,
                            f"mednext_dilated_reparam effective kernel must "
                            f"be odd; got kernel={k}, dilation={d}, "
                            f"effective={eff}.")
                        _require(
                            eff <= self.model.unet.mednext.kernel_size,
                            f"mednext_dilated_reparam effective kernel "
                            f"{eff} exceeds mednext_kernel_size="
                            f"{self.model.unet.mednext.kernel_size}.")
        n_levels = len(self.model.encoder_channels)
        validate_encoder_decoder_stage_lengths(self)
        from .section_validators import validate_patch_geometry
        validate_patch_geometry(self)
        ckpt_mask = list(self.model.unet.grad_ckpt_encoder_stages)
        if ckpt_mask:
            _require(
                len(ckpt_mask) == n_levels,
                f"grad_ckpt_encoder_stages must have {n_levels} entries "
                f"(= len(encoder_channels)); got {len(ckpt_mask)}")
            _require(
                all(int(v) in (0, 1) for v in ckpt_mask),
                f"grad_ckpt_encoder_stages values must be 0 or 1; got {ckpt_mask}.")
            if not self.model.grad_checkpointing:
                logger.warning(
                    "model.grad_ckpt_encoder_stages 已配置但 model.grad_checkpointing=False，"
                    "该掩码将被忽略。")
        # MultiRF（多感受野空洞分支）校验。默认关闭时完全跳过，逐位兼容现状。
        if self.model.unet.multirf.enabled:
            self._validate_multirf(n_levels)
        # SelfAttention（内容寻址自注意力）校验。默认关闭时完全跳过。
        if self.model.unet.selfattn.enabled:
            self._validate_selfattn(n_levels)

    #: softmax 自注意力 O(N²)，token 数超过此阈值时直接报错（防 3D 高分辨率层误 OOM）。
    _SELFATTN_MAX_SOFTMAX_TOKENS = 32768

    def _est_stage_tokens(self, stage_idx: int) -> int:
        """估算编码器 stage_idx 处特征图的 token（体素）数，best-effort（用于 softmax 护栏）。

        3D 用 patch [D,H,W]；2.5D（spatial_dims=2）用 [H,W]（D 已折进通道）。
        逐级下采样按 downsample_strides（若设）否则各轴 2；stem 下采样按 stem_mode 估。
        各向异性自动派生 stride 未显式给出时按 2 估，故为近似值。
        """
        mc = self.model
        sd = mc.spatial_dims
        ps = [int(v) for v in self.data.patch_size]
        axes = ps if sd == 3 else ps[1:]            # 2.5D 只算 H/W
        stem_stride_map = {"conv3": 1, "conv7": 1, "dual": 1,
                           "patch2": 2, "patch4": 4}
        s0 = stem_stride_map.get(mc.stem_mode, 1)
        factor = [s0] * len(axes)
        ds = list(mc.unet.downsample_strides) if mc.unet.downsample_strides else []
        for lvl in range(stage_idx):
            if lvl < len(ds):
                st = ds[lvl]
                st = [int(st)] * len(axes) if isinstance(st, int) else [int(v) for v in st]
            else:
                st = [2] * len(axes)
            for a in range(len(axes)):
                factor[a] *= st[a]
        n = 1
        for axis, f in zip(axes, factor):
            n *= max(1, axis // f)
        return n

    def _validate_selfattn(self, n_levels: int) -> None:
        """model.selfattn_* 校验（仅 selfattn_enabled=True 时调用）。"""
        mc = self.model
        _require(
            str(mc.arch).lower() == "unet",
            "model.selfattn_enabled=True is only supported for model.arch='unet'.")
        _require(
            mc.unet.backbone == "resnet",
            f"model.selfattn_enabled=True requires backbone='resnet'; "
            f"got {mc.unet.backbone!r}.")
        _require(
            mc.unet.selfattn.type in ("softmax", "linear", "window", "grid"),
            f"Invalid model.selfattn_type: {mc.unet.selfattn.type!r}; "
            "expected 'softmax', 'linear', 'window' or 'grid'.")
        _require(
            int(mc.unet.selfattn.num_heads) >= 1,
            f"model.selfattn_num_heads must be >= 1; got {mc.unet.selfattn.num_heads}.")
        hd = int(mc.unet.selfattn.head_dim)
        _require(
            hd == -1 or hd >= 1,
            f"model.selfattn_head_dim must be -1 or >= 1; got {hd}.")
        _require(
            float(mc.unet.selfattn.ffn_ratio) > 0.0,
            f"model.selfattn_ffn_ratio must be > 0; got {mc.unet.selfattn.ffn_ratio}.")
        _require(
            int(mc.unet.selfattn.window_size) >= 1,
            f"model.selfattn_window_size must be >= 1; got {mc.unet.selfattn.window_size}.")
        _require(
            int(mc.unet.selfattn.grid_size) >= 1,
            f"model.selfattn_grid_size must be >= 1; got {mc.unet.selfattn.grid_size}.")
        enc_st = list(mc.unet.selfattn.encoder_stages)
        dec_st = list(mc.unet.selfattn.decoder_stages)
        if enc_st:
            _require(
                len(enc_st) == n_levels,
                f"model.selfattn_encoder_stages must have {n_levels} entries "
                f"(= len(encoder_channels)); got {len(enc_st)}.")
        if dec_st:
            _require(
                len(dec_st) == n_levels - 1,
                f"model.selfattn_decoder_stages must have {n_levels - 1} entries "
                f"(= len(encoder_channels) - 1); got {len(dec_st)}.")
        # 逐 level 解析为类型（None=该层关）；非法取值在 resolve_selfattn_stage 内报错。
        enc_types = [resolve_selfattn_stage(v, mc.unet.selfattn.type) for v in enc_st]
        dec_types = [resolve_selfattn_stage(v, mc.unet.selfattn.type) for v in dec_st]
        # decoder 侧只有 unet 支持。
        if any(t is not None for t in dec_types):
            _require(
                mc.unet.decoder_type == "unet",
                f"model.selfattn_decoder_stages is only supported for "
                f"decoder_type='unet'; got {mc.unet.decoder_type!r}.")
        chans = [int(c) for c in mc.encoder_channels]
        # (索引, 类型, 通道) 三元组：编码器用 stage 索引；解码器 level j（深→浅）通道=encoder_channels[n-2-j]。
        active_enc = [(i, t, chans[i]) for i, t in enumerate(enc_types) if t]
        active_dec = [(j, t, chans[n_levels - 2 - j])
                      for j, t in enumerate(dec_types) if t]
        # 每个被选中层的通道须能被头数/head_dim 整除（建块时也会查，这里提前给清晰报错）。
        for _, _, ch in active_enc + active_dec:
            if hd != -1:
                _require(
                    ch % hd == 0,
                    f"model.selfattn_head_dim={hd} must divide every selected "
                    f"stage's channels; offending channels={ch}.")
            else:
                _require(
                    ch % int(mc.unet.selfattn.num_heads) == 0,
                    f"model.selfattn_num_heads={mc.unet.selfattn.num_heads} must divide "
                    f"every selected stage's channels; offending channels={ch}.")
        # softmax O(N²) 护栏：仅对解析为 'softmax' 的层生效；linear 层豁免。
        cap = self._SELFATTN_MAX_SOFTMAX_TOKENS
        for i, t, _ in active_enc:
            if t == "softmax":
                n_tok = self._est_stage_tokens(i)
                _require(
                    n_tok <= cap,
                    f"selfattn 'softmax' at encoder stage {i} would attend over "
                    f"~{n_tok} tokens (> {cap}); O(N^2) risks OOM. Use 'linear' "
                    f"at this stage or place attention only at deeper stages.")
        for j, t, _ in active_dec:
            if t == "softmax":
                ci = n_levels - 2 - j
                n_tok = self._est_stage_tokens(ci)
                _require(
                    n_tok <= cap,
                    f"selfattn 'softmax' at decoder level {j} (resolution of "
                    f"encoder stage {ci}) would attend over ~{n_tok} tokens "
                    f"(> {cap}); O(N^2) risks OOM. Use 'linear' here or place "
                    f"attention only deeper.")
        if not (active_enc or active_dec):
            logger.warning(
                "model.selfattn_enabled=True but neither selfattn_encoder_stages "
                "nor selfattn_decoder_stages has an active entry; "
                "SelfAttention is effectively a no-op.")

    def _validate_multirf(self, n_levels: int) -> None:
        """model.multirf_* 校验（仅 multirf_enabled=True 时调用）。"""
        mc = self.model
        _require(
            str(self.model.arch).lower() == "unet",
            "model.multirf_enabled=True is only supported for model.arch='unet'.")
        _require(
            mc.unet.backbone == "resnet",
            f"model.multirf_enabled=True requires backbone='resnet'; "
            f"got {mc.unet.backbone!r}.")
        dils = list(mc.unet.multirf.dilations)
        _require(
            len(dils) >= 1 and all(int(d) >= 1 for d in dils),
            f"model.multirf_dilations must be non-empty positive ints; got {dils}.")
        _require(
            1 in [int(d) for d in dils],
            f"model.multirf_dilations must contain 1 (the anti-gridding "
            f"identity branch); got {dils}.")
        _require(
            mc.unet.multirf.mode in ("split", "parallel"),
            f"Invalid model.multirf_mode: {mc.unet.multirf.mode!r}; "
            "expected 'split' or 'parallel'.")
        _require(
            mc.unet.multirf.fusion in ("concat_proj", "sum", "se"),
            f"Invalid model.multirf_fusion: {mc.unet.multirf.fusion!r}; "
            "expected 'concat_proj' | 'sum' | 'se'.")
        _require(
            mc.unet.multirf.axes in ("all", "hw"),
            f"Invalid model.multirf_axes: {mc.unet.multirf.axes!r}; "
            "expected 'all' or 'hw'.")
        # sum 融合要求各分支通道相同 → 仅 parallel 模式可用。
        if mc.unet.multirf.fusion == "sum":
            _require(
                mc.unet.multirf.mode == "parallel",
                "model.multirf_fusion='sum' requires multirf_mode='parallel' "
                "(branches must share channel count to sum).")
        # split 模式下每分支至少 1 通道：最小 stage 通道数须 >= 分支数。
        if mc.unet.multirf.mode == "split":
            min_ch = min(int(c) for c in mc.encoder_channels)
            _require(
                min_ch >= len(dils),
                f"model.multirf_mode='split' needs every stage channel >= "
                f"number of branches ({len(dils)}); smallest encoder_channels="
                f"{min_ch}. Reduce dilations or use mode='parallel'.")
        enc_st = list(mc.unet.multirf.encoder_stages)
        dec_st = list(mc.unet.multirf.decoder_stages)
        if enc_st:
            _require(
                len(enc_st) == n_levels,
                f"model.multirf_encoder_stages must have {n_levels} entries "
                f"(= len(encoder_channels)); got {len(enc_st)}.")
            _require(
                all(int(v) in (0, 1) for v in enc_st),
                f"model.multirf_encoder_stages values must be 0 or 1; got {enc_st}.")
        if dec_st:
            _require(
                len(dec_st) == n_levels - 1,
                f"model.multirf_decoder_stages must have {n_levels - 1} entries "
                f"(= len(encoder_channels) - 1); got {len(dec_st)}.")
            _require(
                all(int(v) in (0, 1) for v in dec_st),
                f"model.multirf_decoder_stages values must be 0 or 1; got {dec_st}.")
            if any(int(v) == 1 for v in dec_st):
                _require(
                    mc.unet.decoder_type == "unet",
                    f"model.multirf_decoder_stages is only supported for "
                    f"decoder_type='unet'; got {mc.unet.decoder_type!r}.")
        if not (any(int(v) == 1 for v in enc_st)
                or any(int(v) == 1 for v in dec_st)):
            logger.warning(
                "model.multirf_enabled=True but neither multirf_encoder_stages "
                "nor multirf_decoder_stages has an active (1) entry; MultiRF is "
                "effectively a no-op.")

    def _validate_augment(self) -> None:
        """augment.* 校验。"""
        _require(
            self.augment.wmap_interp_mode in ("nearest", "bilinear"),
            f"Invalid augment.wmap_interp_mode: {self.augment.wmap_interp_mode!r} "
            "(expected 'nearest' or 'bilinear').")
        per_axis = self.augment.random_rotate_range_per_axis
        if per_axis is not None:
            _require(
                len(per_axis) == 3
                and all(len(r) == 2 for r in per_axis),
                "augment.random_rotate_range_per_axis must be 3 [lo,hi] pairs "
                f"for axes (x,y,z)=(W,H,D); got {per_axis!r}.")
        _require(
            len(self.augment.random_translate_range) == 2,
            "augment.random_translate_range must be [lo, hi]; got "
            f"{self.augment.random_translate_range!r}.")
        # 平移把 padding_mode='border' 的复制边缘卷进 patch；oversample 余量
        # （增强后中心裁回）可把伪影裁掉。translate 幅度超出余量时仅警告。
        tr = self.augment.random_translate_range
        max_tr = max(abs(float(tr[0])), abs(float(tr[1])))
        if max_tr > 0.0:
            # 归一化坐标 [-1,1] 跨整轴：平移 t 卷入约 t/2 轴长的边缘复制带；
            # oversample 余量约 (ratio-1)/(2*ratio) 轴长。
            margin = ((self.data.aug_oversample_ratio - 1.0)
                      / (2.0 * self.data.aug_oversample_ratio))
            if max_tr / 2.0 > margin:
                logger.warning(
                    "augment.random_translate_range=%s 引入的边缘复制带（约 "
                    "%.1f%% 轴长）超过 data.aug_oversample_ratio=%.2f 的中心裁剪"
                    "余量（约 %.1f%% 轴长），border 复制伪影会留在训练 patch 内。"
                    "建议增大 aug_oversample_ratio 或减小平移幅度。",
                    tr, max_tr / 2.0 * 100.0,
                    self.data.aug_oversample_ratio, margin * 100.0)
        # 数值正性/区间校验：非法配置在 augment 运行期会产生 div-by-zero /
        # NaN 核或空尺寸，这里 fail-fast。
        blur = self.augment.gaussian_blur_sigma
        _require(
            len(blur) == 2 and 0.0 < float(blur[0]) <= float(blur[1]),
            "augment.gaussian_blur_sigma must be [lo, hi] with 0 < lo <= hi; "
            f"got {blur!r}.")
        _require(
            float(self.augment.elastic_deform_sigma) > 0.0,
            "augment.elastic_deform_sigma must be > 0; "
            f"got {self.augment.elastic_deform_sigma}.")
        _require(
            float(self.augment.elastic_deform_alpha) >= 0.0,
            "augment.elastic_deform_alpha must be >= 0; "
            f"got {self.augment.elastic_deform_alpha}.")
        _require(
            str(self.augment.elastic_field_mode).lower() in ("legacy", "gaussian"),
            "augment.elastic_field_mode must be 'legacy' or 'gaussian'; "
            f"got {self.augment.elastic_field_mode!r}.")
        _require(
            str(self.data.split_rounding_mode).lower() in ("legacy", "unified"),
            "data.split_rounding_mode must be 'legacy' or 'unified'; "
            f"got {self.data.split_rounding_mode!r}.")
        _require(
            str(self.model.init_strategy).lower()
            in ("legacy", "kaiming", "trunc_normal"),
            "model.init_strategy must be 'legacy', 'kaiming' or "
            f"'trunc_normal'; got {self.model.init_strategy!r}.")
        _require(
            float(self.augment.gaussian_noise_std) >= 0.0,
            "augment.gaussian_noise_std must be >= 0; "
            f"got {self.augment.gaussian_noise_std}.")
        zoom_r = self.augment.simulate_lowres_zoom
        _require(
            len(zoom_r) == 2
            and 0.0 < float(zoom_r[0]) <= float(zoom_r[1]) <= 1.0,
            "augment.simulate_lowres_zoom must be [lo, hi] with "
            f"0 < lo <= hi <= 1; got {zoom_r!r}.")
        gamma_r = self.augment.random_gamma_range
        _require(
            len(gamma_r) == 2 and 0.0 < float(gamma_r[0]) <= float(gamma_r[1]),
            "augment.random_gamma_range must be [lo, hi] with 0 < lo <= hi; "
            f"got {gamma_r!r}.")
        # brightness/noise 幅值为绝对量、隐含 image≈[0,1]（minmax）。zscore
        # （std≈1）下沿用 minmax 默认幅值时扰动相对偏弱且量纲不符，提示改配
        # （以 σ 为单位，典型 brightness±0.5σ、noise 0.1σ 量级）。
        if self.data.normalize == "zscore" and self.augment.enabled:
            b_r = [float(v) for v in self.augment.random_brightness_range]
            n_std = float(self.augment.gaussian_noise_std)
            defaults = AugConfig()
            hints = []
            if (self.augment.random_brightness_prob > 0
                    and b_r == list(defaults.random_brightness_range)):
                hints.append(
                    f"random_brightness_range={b_r}")
            if (self.augment.gaussian_noise_prob > 0
                    and n_std == float(defaults.gaussian_noise_std)):
                hints.append(f"gaussian_noise_std={n_std}")
            if hints:
                logger.warning(
                    "data.normalize='zscore' 下 %s 仍为 minmax 量纲的默认绝对"
                    "幅值：zscore（std≈1）上同数值扰动约弱一个量级。建议按 σ "
                    "为单位改配（如 brightness ±0.3~0.5、noise std 0.1）。",
                    "、".join(hints))

    def _validate_data(self) -> None:
        """data.* patch/multi-res/keep_native 校验。"""
        _require(
            len(self.data.patch_size) == 3,
            "patch_size must be [D, H, W]")
        if self.data.group_id_regex:
            try:
                re.compile(self.data.group_id_regex)
            except re.error as e:
                _require(False,
                         f"data.group_id_regex is not a valid regex: {e}")
        _require(
            self.data.patch_mode in ("z_axis", "cubic", "whole", "2_5d"),
            f"Invalid patch_mode: {self.data.patch_mode}")
        _require(
            self.data.z_boundary_mode in ("stretch", "edge_pad"),
            f"Invalid z_boundary_mode: {self.data.z_boundary_mode!r}; "
            "expected 'stretch' or 'edge_pad'.")
        _require(
            self.data.cache_dtype in ("fp32", "int16"),
            f"Invalid data.cache_dtype: {self.data.cache_dtype!r}; "
            "expected 'fp32' or 'int16'.")
        # 枚举/区间兜底（fail-fast）：normalize 非法会让下游静默走错归一化
        # 分支（直接违反训推一致性契约 C8）；cache_mode 非法会被消费点
        # `== "memory"` 静默当作 none。
        _require(
            self.data.normalize in ("minmax", "zscore"),
            f"Invalid data.normalize: {self.data.normalize!r}; "
            "expected 'minmax' or 'zscore'.")
        _require(
            self.data.cache_mode in ("none", "memory"),
            f"Invalid data.cache_mode: {self.data.cache_mode!r}; "
            "expected 'none' or 'memory'.")
        # val_ratio=0 并不产生"无验证集"（split 侧钳到至少 1 个 val 样本），
        # 语义上无效，直接拒绝。
        _require(
            0.0 < float(self.data.val_ratio) < 1.0,
            f"data.val_ratio must be in (0, 1); got {self.data.val_ratio}.")
        _require(
            0.0 <= float(self.data.foreground_oversample_ratio) <= 1.0,
            f"data.foreground_oversample_ratio must be in [0, 1]; "
            f"got {self.data.foreground_oversample_ratio}.")
        _require(
            int(self.data.samples_per_volume) >= 1,
            f"data.samples_per_volume must be >= 1; "
            f"got {self.data.samples_per_volume}.")
        _require(
            int(self.data.batch_size) >= 1,
            f"data.batch_size must be >= 1; got {self.data.batch_size}.")
        _require(
            int(self.data.num_workers) >= 0,
            f"data.num_workers must be >= 0; got {self.data.num_workers}.")
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

        # spacing 归一化：target_spacing 若显式给出须为 3 个正数（(D,H,W) mm）。
        if self.data.spacing_normalization and self.data.target_spacing is not None:
            ts = self.data.target_spacing
            _require(
                len(ts) == 3 and all(float(s) > 0.0 for s in ts),
                "data.target_spacing must be 3 positive floats [sz, sy, sx] (mm); "
                f"got {ts}.")

        _require(
            self.data.aug_oversample_ratio >= 1.0,
            "aug_oversample_ratio must be >= 1.0")
        _require(
            len(self.data.multi_res_scales) >= 1,
            "multi_res_scales must have at least one scale (e.g. [1.0])")
        _require(
            all(s >= 1.0 for s in self.data.multi_res_scales),
            "All multi_res_scales must be >= 1.0")

        # 双批混合：仅 npz_dir_secondary 非空时校验 mix_ratio。
        if self.data.npz_dir_secondary:
            mr = self.data.mix_ratio
            _require(
                len(mr) == 2,
                f"mix_ratio must be [gold, coarse] (length 2); got {mr}.")
            _require(
                all(isinstance(x, int) and x >= 1 for x in mr),
                f"mix_ratio elements must be integers >= 1 (each batch must "
                f"contain both sources); got {mr}.")
            _require(
                self.data.batch_size % sum(mr) == 0,
                f"batch_size ({self.data.batch_size}) must be divisible by "
                f"sum(mix_ratio) ({sum(mr)}) for integer per-batch counts.")

    def _validate_2_5d(self, *, check_channel_layout: bool = True) -> None:
        """2.5D 专属不变式（折叠通道 / lift / Plan A·C / aux 监督）。

        ``check_channel_layout``：分割任务校验 ``in_channels == D*n_views``
        （或 keep_native 的 sum(D_k)）及 ``aux_seg_supervision`` 强制项。
        生成任务通道含 cond 等扩展，应传 ``False`` 后由任务侧自管。
        """
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
            lift = bool(self.model.unet.lift_2_5d_to_3d)
            if lift:
                # ADM/EDM2 硬编码为折叠-D 的 2D 布局，与 lift 的真 3D 布局互斥。
                _require(
                    self.model.arch not in ("adm", "edm2"),
                    f"lift_2_5d_to_3d=True is not supported by model.arch="
                    f"{self.model.arch!r} (ADM/EDM2 are wired for the folded-D "
                    "2D layout only). Use arch='unet' or disable lift.")
                # lift：D 保留为空间轴（真 3D UNet），与折叠-D 布局互斥。
                _require(
                    self.model.spatial_dims == 3,
                    "lift_2_5d_to_3d=True requires model.spatial_dims=3 (auto-set by sync()).")
                if check_channel_layout:
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
            if check_channel_layout:
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
            elif (not lift) and self.data.keep_native_view_depth and n_views > 1:
                # 生成任务：仅校验 native-depth 几何契约，不强制 seg aux 通道布局。
                depths = self.per_view_depths
                _require(
                    len(depths) == n_views,
                    f"per_view_depths length must equal n_views ({n_views}); got {len(depths)}.")
                _require(
                    depths[0] == self.data.patch_size[0],
                    f"per_view_depths[0] must equal patch_size[0]={self.data.patch_size[0]}; "
                    f"got {depths[0]}.")
                _require(
                    self.data.z_boundary_mode == "edge_pad",
                    f"keep_native_view_depth=True requires z_boundary_mode='edge_pad'; "
                    f"got {self.data.z_boundary_mode!r}.")
                from ..models.topology import build_topology
                _require(
                    build_topology(self).in_ch_per_view_list is not None,
                    "keep_native_view_depth=True requires in_ch_per_view_list "
                    "(derived by build_topology).")
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
        _require(
            int(self.train.val_high_interval) >= 1,
            f"train.val_high_interval must be >= 1, "
            f"got {self.train.val_high_interval}")
        # high 模式在整卷 blended 概率上算指标，无可逆 logits 故不产出 val_base_loss；
        # 因此 'loss' criterion 与 high 互斥（否则永远选不出 best）。改用重叠类指标。
        if (str(self.train.val_metric_mode).lower().strip() == "high"
                and _norm_crit(self.train.save_best_criterion) == "loss"):
            raise ConfigError(
                "train.save_best_criterion='loss' is incompatible with "
                "train.val_metric_mode='high' (full-volume inference produces "
                "blended probabilities, not invertible logits, so no "
                "val_base_loss "
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
        _require(
            float(self.train.surface_dice_tolerance_mm) >= 0.0,
            f"surface_dice_tolerance_mm must be >= 0; got "
            f"{self.train.surface_dice_tolerance_mm}")
        # prefetch_to_gpu 依赖 pinned host 内存才能真正异步（否则正确但无收益）。
        if self.train.prefetch_to_gpu and not self.data.pin_memory:
            logger.warning(
                "train.prefetch_to_gpu=True but data.pin_memory=False: "
                "H2D copies from pageable memory are synchronous, so the "
                "prefetch overlap has no effect. Enable data.pin_memory.")
        if self.train.pretrain_upkern and self.model.unet.backbone != "mednext":
            logger.warning(
                "train.pretrain_upkern=True only affects backbone='mednext'; "
                "current backbone=%r, so UpKern remap will be ignored.",
                self.model.unet.backbone)
        # 多卡 DDP 选卡列表：物理卡号、非负、互不重复。
        gpus = list(self.train.gpus)
        _require(
            all(isinstance(g, int) and g >= 0 for g in gpus),
            f"train.gpus must be a list of non-negative ints (physical GPU "
            f"indices); got {self.train.gpus!r}.")
        _require(
            len(gpus) == len(set(gpus)),
            f"train.gpus must not contain duplicate GPU indices; got {gpus}.")
        # EMA / SWA / ZeRO：由 BaseTrainer 对所有任务生效。
        _require(
            str(self.train.ema_device) in ("", "cpu"),
            f"train.ema_device must be '' (follow model) or 'cpu'; "
            f"got {self.train.ema_device!r}.")
        if self.train.swa_enabled:
            _require(
                0.0 < float(self.train.swa_start_ratio) < 1.0,
                f"train.swa_start_ratio must be in (0, 1); "
                f"got {self.train.swa_start_ratio}")
        if self.train.zero_redundancy_optimizer and len(gpus) < 2:
            logger.warning(
                "train.zero_redundancy_optimizer=True 但未启用多卡 DDP（需 "
                "len(train.gpus) >= 2）；单卡下无分片收益，将回退普通优化器。")

    def _validate_monitor(self) -> None:
        """monitor.* 训练监测仪表盘校验（仅 monitor.enabled 时生效）。"""
        if not self.monitor.enabled:
            return
        m = self.monitor
        _require(
            int(m.update_every) >= 1,
            f"monitor.update_every must be >= 1; got {m.update_every}.")
        _require(
            isinstance(m.compare_runs, list),
            f"monitor.compare_runs must be a list of run dirs; got {m.compare_runs!r}.")

    @property
    def num_fg_classes(self) -> int:
        """Number of foreground classes (excluding background)."""
        return max(self.data.num_classes - 1, 1)

    @property
    def per_view_depths(self) -> List[int]:
        """2.5D 下每视图原生深度 D_k = round(D * s_k)，强制 D_0 = D。非 2.5D 返回空列表。

        R5：委托给 ``build_topology`` 以保持单一真相源；仅形状计算，不依赖
        ``data.keep_native_view_depth``，调用方自行根据该标志决定是否使用。
        """
        from ..models.topology import build_topology
        return list(build_topology(self).per_view_depths)


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------
_SUB_CONFIGS = {
    "data": DataConfig,
    "augment": AugConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "vis": VisConfig,
    "monitor": MonitorConfig,
}


# 旧 YAML 字段名 → 新字段名。此处仅用于报错信息里的迁移提示；旧名现已硬拒绝。
# 命名清晰化（TODO #4）：
#   data.aux_keep_native_d  → data.keep_native_view_depth（'aux' 误导：含主视图）
#   model.context_fusion    → model.stem_fusion_mode（与 num_stem_fusion_views 配对）
_FIELD_ALIASES: Dict[type, Dict[str, str]] = {
    DataConfig:  {"aux_keep_native_d": "keep_native_view_depth"},
    ModelConfig: {"context_fusion": "stem_fusion_mode"},
}


# 旧 YAML 中曾可手设、现已改为派生只读量的字段。此处仅用于报错信息里的迁移提示；
# 旧名现已硬拒绝。TODO #4：派生量不再暴露可写接口。
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


# 旧 YAML 中已移除、但需要更具体迁移提示的字段。旧名现已硬拒绝。
#   model.use_se → 改用 model.attention_type: "se"。
_REMOVED_KEYS: Dict[type, Dict[str, str]] = {
    ModelConfig: {"use_se": 'attention_type: "se"'},
}


def nested_dataclass_type(f) -> "Optional[type]":
    """字段声明为嵌套 dataclass 时返回其类型，否则 None。

    项目约定嵌套段一律写作 ``field(default_factory=SubConfig)``，因此从
    default_factory 判定（避免解析 ``from __future__ import annotations``
    下的字符串注解）。
    """
    factory = f.default_factory
    if factory is not MISSING and isinstance(factory, type) \
            and is_dataclass(factory):
        return factory
    return None


_nested_dataclass_type = nested_dataclass_type  # 旧名别名，兼容存量引用


@dataclass(frozen=True)
class DataclassLoadContext:
    """``dataclass_from_dict`` 的可选加载上下文（供 gen 等 fork 消重）。"""

    sub_configs: Optional[Dict[str, type]] = None
    field_aliases: Optional[Dict[type, Dict[str, str]]] = None
    deprecated_derived_keys: Optional[Dict[type, Dict[str, str]]] = None
    removed_keys: Optional[Dict[type, Dict[str, str]]] = None
    model_route_extra_flat_to_nested: Optional[Dict[str, str]] = None
    model_config_cls: type = ModelConfig
    error_cls: type = ConfigError


def _lookup_type_map(type_map: Dict[type, Dict[str, str]], cls) -> Dict[str, str]:
    """沿 MRO 查找类型映射（gen 子类 dataclass 复用 core 父类条目）。"""
    for c in getattr(cls, "__mro__", (cls,)):
        if c in type_map:
            return type_map[c]
    return {}


def dataclass_from_dict(
    cls,
    d: Dict[str, Any],
    ctx: Optional[DataclassLoadContext] = None,
):
    """Recursively construct a dataclass from a dict（任意深度嵌套）。

    * ``model`` 段的旧扁平键先经 :func:`route_legacy_model_dict` 路由；
    * 旧别名 / 派生只读字段 / 未知字段抛 ``ConfigError``（或 ``ctx.error_cls``）；
    * ``ctx.sub_configs`` 在 ``nested_dataclass_type`` 无法解析时作段名兜底
      （gen 的 ``task`` 段等）。
    """
    if not isinstance(d, dict):
        return d

    err = ctx.error_cls if ctx is not None else ConfigError
    model_cls = ctx.model_config_cls if ctx is not None else ModelConfig
    sub_configs = (
        ctx.sub_configs if ctx is not None and ctx.sub_configs is not None
        else _SUB_CONFIGS)
    aliases_map = (
        ctx.field_aliases if ctx is not None and ctx.field_aliases is not None
        else _FIELD_ALIASES)
    derived_map = (
        ctx.deprecated_derived_keys
        if ctx is not None and ctx.deprecated_derived_keys is not None
        else _DEPRECATED_DERIVED_KEYS)
    removed_map = (
        ctx.removed_keys if ctx is not None and ctx.removed_keys is not None
        else _REMOVED_KEYS)
    extra_flat = (
        ctx.model_route_extra_flat_to_nested if ctx is not None else None)

    if isinstance(cls, type) and issubclass(cls, model_cls):
        d, moved = route_legacy_model_dict(
            d, error_cls=err, extra_flat_to_nested=extra_flat)
        if moved:
            logger.info(
                "model 段旧扁平键已自动迁移到嵌套路径（建议更新 YAML）：%s",
                ", ".join(f"{k} -> {p}" for k, p in sorted(moved.items())))

    dc_fields = {f.name: f for f in fields(cls)}
    aliases = _lookup_type_map(aliases_map, cls)
    derived = _lookup_type_map(derived_map, cls)
    removed = _lookup_type_map(removed_map, cls)
    kwargs = {}
    for k, v in d.items():
        if k in removed:
            raise err(
                f"Config key '{k}' is removed from {cls.__name__}; use "
                f"{removed[k]} instead.")
        if k in derived:
            raise err(
                f"Config key '{k}' is removed from {cls.__name__}; it is now "
                f"auto-derived from '{derived[k]}' and must not be set in YAML.")
        if k in aliases:
            new_key = aliases[k]
            if new_key in d:
                raise err(
                    f"{cls.__name__}: both deprecated '{k}' and its "
                    f"replacement '{new_key}' are set; remove '{k}' and keep "
                    f"'{new_key}'.")
            raise err(
                f"Config key '{k}' is removed from {cls.__name__}; use "
                f"'{new_key}' instead.")
        if k not in dc_fields:
            raise err(
                f"Unknown config key '{k}' in {cls.__name__}.")
        sub_cls = nested_dataclass_type(dc_fields[k])
        if sub_cls is None and k in sub_configs:
            sub_cls = sub_configs[k]
        if sub_cls is not None and isinstance(v, dict):
            v = dataclass_from_dict(sub_cls, v, ctx)
        kwargs[k] = v
    return cls(**kwargs)


def _dataclass_from_dict(cls, d: Dict[str, Any]):
    """Seg 默认上下文加载（``load_config`` / ``task_io`` 使用）。"""
    return dataclass_from_dict(cls, d)


def load_config(path: Union[str, Path]) -> Config:
    """Load core configuration from a YAML file (不含 seg 专属 loss/predict 段)。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    from .seg_task import hoist_legacy_seg_sections
    hoist_legacy_seg_sections(raw)
    if "seg" in raw:
        logger.warning(
            "core.load_config(%s) discards top-level 'seg' (loss/predict); "
            "use segtask_v1.seg_config.load_config for segmentation configs.",
            path)
        raw.pop("seg", None)
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


# ---------------------------------------------------------------------------
# Backward-compat: legacy checkpoint unpickling
# ---------------------------------------------------------------------------
@dataclass
class _LegacySSLConfig:
    """占位类，仅用于反序列化历史 checkpoint。

    早期版本（提交 e0a7f26 一带）``SSLConfig`` 定义在本模块且是 ``Config.ssl``
    字段；trainer 会把整个 ``Config`` 对象 pickle 进 checkpoint（见
    ``trainer._build_state_dict`` 的 ``"config": self.cfg``）。之后 ``SSLConfig``
    被移到 :mod:`ssltask.config`，该符号在本模块消失，导致老 checkpoint 走
    ``torch.load(..., weights_only=False)`` 时 ``find_class`` 抛
    ``AttributeError: Can't get attribute 'SSLConfig'``。

    pickle 还原对象走 ``cls.__new__(cls)`` + 恢复 ``__dict__``，不调用 ``__init__``，
    因此这里只需存在一个同名类即可让老 checkpoint 正常反序列化；历史字段全为基础
    类型，会被原样挂到实例上。本占位不参与任何训练/推理逻辑——推理用的是命令行
    ``--config`` 的 YAML，checkpoint 内嵌的 ``config`` 对象并不被消费。
    """


_LEGACY_MODULE_ATTRS = {"SSLConfig": _LegacySSLConfig}


def __getattr__(name: str):
    """PEP 562 模块级钩子：把已迁走的历史符号解析为兼容占位。

    仅覆盖 :data:`_LEGACY_MODULE_ATTRS` 中登记的名字（当前只有 ``SSLConfig``），
    其余未知属性照常抛 ``AttributeError``，不会掩盖真正的拼写/导入错误。
    """
    target = _LEGACY_MODULE_ATTRS.get(name)
    if target is not None:
        logger.warning(
            "taskcore.config.core.%s 已迁至 ssltask.config；返回向后兼容占位以反序列化"
            "历史 checkpoint。请重新保存 checkpoint 以移除该历史引用。", name)
        return target
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}")
