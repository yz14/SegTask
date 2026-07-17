"""ssltask 配置系统。

设计：**复用 ``segtask_v1.config.Config``** 作为 data/model/train/... 的载体（即与下游
分割/分类共用同一份骨干配置真相源，满足 ``SSL.md`` §0.1“所有方案严格同一骨干”），
仅新增 SSL 专有的 :class:`SSLConfig`。``load_config`` 返回 ``(cfg, ssl)`` 二元组：

* ``cfg``  —— ``segtask_v1.config.Config``，喂给 ``build_model`` / 数据管线 / 优化器；
* ``ssl``  —— 本模块 :class:`SSLConfig`，承载方法选择与各方法超参。

这样既不重复 ``Config.sync/validate`` 的复杂派生/校验逻辑，也让 ssltask 与下游天然
共享骨干几何。YAML 中 ``ssl:`` 段被本模块单独解析，其余段交给 segtask 构造。
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import yaml

from taskcore.config.core import (
    Config as SegConfig,
    ConfigError,
    _dataclass_from_dict,
    _require,
)

logger = logging.getLogger(__name__)

#: 重建/回归损失。
RECON_LOSSES = ("l1", "smooth_l1", "mse")
#: 已实现的 SSL 方法注册键（须与 ``ssltask.methods`` 注册表保持同步；随 P2+ 扩展）。
METHODS = ("genesis", "prior", "simmim", "dino", "spark", "dino_gram", "jepa", "ibot", "sparkdino", "byol", "moco", "vicregl")


# ---------------------------------------------------------------------------
# SSL-specific config
# ---------------------------------------------------------------------------
@dataclass
class SSLConfig:
    """自监督预训练专有配置。

    骨干 / 优化器 / 调度器 / AMP / EMA / 输出目录 / epochs / lr / 数据预处理全部读
    ``segtask_v1.config.Config``（``cfg.model`` / ``cfg.train`` / ``cfg.data``）；此处只放
    SSL 目标本身相关的项。
    """

    # 预训练方法（见 ``ssltask.methods`` 注册表）：
    #   "genesis" — Models Genesis 式破坏→重建原图（SSL.md 方案③）。
    #   "prior"   — 回归经典 Frangi vesselness（免标注管状先验）。
    #   "simmim"  — mask token 稠密掩码图像建模（SSL.md 方案②）。
    #   "dino"    — 多裁剪 + EMA 教师自蒸馏（SSL.md 方案④）。
    #   "spark"   — 掩码-稠密等价 + 层次解码器像素重建（SSL.md 方案①）。
    #   "dino_gram" — DINO + Gram anchoring（SSL.md 方案⑤；④ 基础上加密集特征 Gram 约束）。
    #   "jepa"    — 隐空间掩码预测（上下文/EMA 目标编码器 + 预测器，SSL.md 方案⑦）。
    #   "ibot"    — DINO 全局蒸馏 + iBOT 掩码密集特征预测（SSL.md 方案⑥；④ 基础上加密集头）。
    #   "sparkdino" — SparK 像素重建 + DINO 全局蒸馏的朴素多任务组合（SSL.md 方案⑧；①+④ 共享 encoder）。
    method: str = "genesis"

    # 重建/回归损失："l1" | "smooth_l1" | "mse"。
    recon_loss: str = "l1"

    # --- Genesis 破坏（image-only，GPU 逐样本即时施加）---
    nonlinear_prob: float = 0.9
    local_shuffle_prob: float = 0.5
    local_shuffle_blocks: int = 100
    local_shuffle_max_block: List[int] = field(default_factory=lambda: [4, 8, 8])
    paint_prob: float = 0.9
    inner_paint_prob: float = 0.5
    paint_count: int = 5
    paint_block_range: List[float] = field(default_factory=lambda: [0.1, 0.25])

    # --- method='prior'：Frangi vesselness 回归目标（label-free）---
    prior_scales: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    prior_alpha: float = 0.5
    prior_beta: float = 0.5
    prior_black_vessels: bool = False
    prior_corrupt_input: bool = False
    # 物理体素间距 (sz,sy,sx) mm；给定时 prior_scales 解释为物理尺度(mm)，Frangi
    # 按各向异性 spacing 计算高斯/Hessian（各向异性数据下管径尺度物理一致）。空=
    # 体素单位(旧行为)。数据已重采样到统一 target_spacing 时填该值即可。
    prior_spacing: List[float] = field(default_factory=list)
    # 2.5D（spatial_dims==2）下 vesselness 目标的计算维度：
    #   True （默认，推荐）：在 trainer 折叠深度**之前**的 (B,1,D,H,W) 体上按 3D
    #     Frangi 计算，再折回 (B,D,H,W) 作目标——沿 z 走行的血管在轴位切片里表现为
    #     亮斑、被 2D Frangi 的 blobness 项系统性压制，3D 计算可正确检出；且归一在
    #     整卷上进行，层间目标尺度一致。
    #   False：对折叠后的每张切片独立算 2D（旧行为）——丢失穿面血管、层间尺度不一。
    # 给定 prior_spacing 时，target_3d 下其长度须为 3 (sz,sy,sx)。spatial_dims==3
    # （3D cubic）本就按 3D 计算，此项不生效。
    prior_target_3d: bool = True

    # --- image-only 采样：内容偏置拒绝采样（可选；默认关=沿 z 均匀随机中心）---
    # SSL 无标注，默认沿 z 均匀随机抽 patch；背景切片多的数据集会产出大量近空
    # patch（vesselness 目标近乎全零 / 掩码重建信息量低，监督退化）。开启后：以
    # ``sample_fg_ratio`` 概率对该样本做“内容偏置”——抽 ``sample_max_tries`` 个候选
    # z 中心，取前景占比（归一强度 > ``sample_fg_thresh`` 的体素比例）最高者；其余
    # 样本仍均匀随机，保留背景/解剖多样性，避免过拟合强度启发式。cubic 模式同理
    # （三轴中心）。阈值以**预处理归一后**的强度为准（minmax→[0,1]）。
    sample_content_bias: bool = False
    sample_fg_ratio: float = 0.5
    sample_fg_thresh: float = 0.1
    sample_max_tries: int = 4

    # --- method='simmim'：mask token 稠密掩码图像建模（SSL.md 方案②）---
    # 掩码比例（SimMIM 经验 0.5–0.6）。
    mim_mask_ratio: float = 0.6
    # 掩码单元边长（体素），patch-wise 掩码粒度（SSL.md 取 16，可更大）。
    mim_mask_unit: int = 16
    # 轻量预测头隐藏通道；<=0 自动取 max(encoder_channels[-1]//2, 32)。
    mim_head_dim: int = 0

    # --- method='spark'：掩码-稠密等价 MIM + 轻量层次解码器（SSL.md 方案①）---
    # 掩码比例（SparK 经验 0.6，低于 MAE 的 0.75）。
    spark_mask_ratio: float = 0.6
    # 掩码单元边长（体素），应与骨干总下采样步长对齐（SSL.md 取 16）。
    spark_mask_unit: int = 16
    # 轻量解码器宽度除数：解码各级通道 = max(encoder_channels[level]//div, min_dim)。
    # div 越大解码器越窄（SSL.md：参数约 encoder 的 1/5–1/10，用完即弃）。
    spark_decoder_dim_div: int = 4
    # 解码器各级最小通道数。
    spark_decoder_min_dim: int = 16
    # 是否对重建目标做 per-unit 归一化（减单元均值除单元标准差，MAE 经验）。
    spark_norm_pix: bool = True
    # 掩码前向时把 encoder 的 InstanceNorm 换为仅在可见位点上统计的 Masked 版本，
    # 消除置零位点对归一化统计的污染（训练/下游稠密前向分布一致）；参数名/形状
    # 不变，稠密前向（探针/下游/DINO 分支）走原生路径。False = 旧行为（有偏统计）。
    spark_masked_norm: bool = True
    # 解码器模式（仅 method='spark'）：
    #   "light" — SparK 原方窄解码器（dim_div 控宽），用完即弃，下游仅迁移 encoder.*；
    #   "seg"   — 重建经过下游真解码器（decoder.* 同名同形），SSL ckpt 同时
    #             warm-start encoder.*+decoder.*，下游仅 seg_head.* 随机（分割友好）。
    spark_decoder_mode: str = "light"

    # --- method='dino'：多裁剪 + EMA 教师自蒸馏（SSL.md 方案④）---
    # 投影头：原型数 out_dim、MLP 隐藏/瓶颈维、层数、是否 BN（小 batch 慎用）。
    dino_out_dim: int = 4096
    dino_hidden_dim: int = 2048
    dino_bottleneck_dim: int = 256
    dino_head_layers: int = 3
    dino_head_use_bn: bool = False
    # 多裁剪：global/local 数量、输出尺寸（空=由 patch_size 推导）、裁剪尺度区间。
    dino_global_crops: int = 2
    dino_local_crops: int = 6
    dino_global_size: List[int] = field(default_factory=list)
    dino_local_size: List[int] = field(default_factory=list)
    dino_global_scale: List[float] = field(default_factory=lambda: [0.5, 1.0])
    dino_local_scale: List[float] = field(default_factory=lambda: [0.15, 0.5])
    # 视图增广：随机翻转概率 + 强度缩放/平移幅度。
    dino_flip_prob: float = 0.5
    dino_intensity_scale: float = 0.1
    dino_intensity_shift: float = 0.1
    # 温度：学生固定、教师从 warmup 起点升到 final（前 warmup_frac 比例步内）。
    dino_student_temp: float = 0.1
    dino_teacher_temp: float = 0.07
    dino_teacher_temp_warmup: float = 0.04
    dino_warmup_teacher_temp_frac: float = 0.3
    # centering 动量 + 教师 EMA 动量（cosine base→final）。
    dino_center_momentum: float = 0.9
    dino_momentum_base: float = 0.996
    dino_momentum_final: float = 1.0
    # 前此比例的优化步内冻结学生投影头末层（原型层）梯度（DINO 稳定化技巧，
    # 避免训练初期原型剧烈重排引发崩塌；0=不冻结）。
    dino_freeze_last_layer_frac: float = 0.01

    # --- method='byol'：BYOL-3D（online/query + EMA target；无负样本）---
    byol_proj_dim: int = 128
    byol_pred_hidden_dim: int = 128
    byol_momentum_base: float = 0.996
    byol_momentum_final: float = 1.0

    # --- method='moco'：MoCo-3D（query/key + EMA queue）---
    moco_proj_dim: int = 128
    moco_queue_size: int = 4096
    moco_temperature: float = 0.2
    moco_momentum_base: float = 0.996
    moco_momentum_final: float = 1.0

    # --- method='vicregl'：VICRegL-3D（全局 VIC + 位置匹配稠密 VIC；孪生无 EMA）---
    # 全局投影头：输出维 / 隐藏维；稠密投影头输出维（1×1 conv MLP，保分辨率）。
    vicregl_proj_dim: int = 256
    vicregl_hidden_dim: int = 1024
    vicregl_dense_proj_dim: int = 64
    # 稠密嵌入取自哪一级 encoder 特征（feats 索引，-1=瓶颈）。
    vicregl_feature_level: int = -1
    # VIC 三项权重（invariance/variance/covariance，原论文 25/25/1）。
    vicregl_sim_coeff: float = 25.0
    vicregl_var_coeff: float = 25.0
    vicregl_cov_coeff: float = 1.0
    # 全局/局部加权 α：L = α·L_global + (1-α)·L_local（原论文 0.75）。
    vicregl_alpha: float = 0.75
    # 每样本每方向取多少对位置最近的位点匹配（top-γ，双向）。
    vicregl_num_matches: int = 20
    # 每样本每方向的特征空间最近邻匹配对数（VICRegL feature-based；0=关）。
    vicregl_feature_matches: int = 20
    # 位置匹配的最大距离半径（以目标视图位点间距为单位；超出则不配对，
    # 避免两裁剪框不重叠时强造正样本；<=0 关闭过滤）。
    vicregl_match_radius: float = 1.0
    # 两视图裁剪尺度区间（高重叠保证可匹配；翻转/强度增广复用 dino_*）。
    vicregl_crop_scale: List[float] = field(default_factory=lambda: [0.6, 1.0])

    # --- method='jepa'：隐空间掩码预测（SSL.md 方案⑦）---
    # 目标块掩码：单元尺寸（体素）与覆盖比例（被遮单元占比，作为预测目标）。
    jepa_mask_unit: int = 16
    jepa_mask_ratio: float = 0.6
    # 在哪一级 encoder 特征上预测（feats 索引，-1=瓶颈）。
    jepa_feature_level: int = -1
    # 轻量预测器：卷积块数 + 隐藏通道（0=取特征通道数）。
    jepa_predictor_depth: int = 2
    jepa_predictor_hidden: int = 0
    # 目标编码器 EMA 动量（cosine base→final）。
    jepa_momentum_base: float = 0.996
    jepa_momentum_final: float = 1.0
    # VICReg 抗坍缩正则权重（方差项 / 协方差项；0=关闭）。默认开启：本实现为
    # CNN 适配（密集 mask-token 上下文 + 单尺度随机块掩码），坍缩风险高于
    # 原版 I-JEPA（ViT 丢 token + 多目标块采样）。
    jepa_var_weight: float = 1.0
    jepa_cov_weight: float = 0.04

    # --- method='dino_gram'：DINO + Gram anchoring（SSL.md 方案⑤）---
    # 复用上面全部 dino_* 项；以下仅 Gram 分支专用。
    # Gram 损失权重 λ（L = L_DINO + λ·L_gram）。
    dino_gram_weight: float = 1.0
    # 训练进度 < 此比例时 λ=0（只跑纯 DINO，待密集特征成形后再锚定）。
    dino_gram_start_frac: float = 0.3
    # Gram 教师刷新间隔（优化步，以 λ 首次生效的锚定时刻为原点）：每多少步从当前
    # EMA 教师整份拷贝一次快照。须远大于 1——快照与 EMA 教师保持足够"时差"才有
    # 锚定意义（=1 时 Gram 教师≈当前教师，anchoring 退化为空操作）。
    dino_gram_refresh_steps: int = 1000
    # 计算 Gram 的 encoder 特征级（feats 索引，-1=瓶颈）。
    dino_gram_feature_level: int = -1

    # --- method='ibot'：DINO 全局蒸馏 + iBOT 掩码密集特征预测（SSL.md 方案⑥）---
    # 复用上面全部 dino_* 项（多裁剪/温度/center/EMA 动量/投影头维度）；以下仅 iBOT 分支专用。
    # iBOT 掩码密集特征损失权重（L = L_DINO + λ·L_iBOT）。
    ibot_weight: float = 1.0
    # 学生输入掩码比例（iBOT 经验 0.3–0.5）与掩码单元边长（体素）。
    ibot_mask_ratio: float = 0.4
    ibot_mask_unit: int = 16
    # 计算密集特征的 encoder 特征级（feats 索引，-1=瓶颈）。
    ibot_feature_level: int = -1
    # iBOT 密集头原型数（0=复用 dino_out_dim）。
    ibot_out_dim: int = 0
    # True=与全局 DINO 头共享原型；False=独立 iBOT 头（DINOv2 默认）。
    ibot_share_head: bool = False

    # --- method='sparkdino'：SparK 像素重建 + DINO 全局蒸馏（SSL.md 方案⑧）---
    # 双分支共享同一 encoder；SparK 侧复用全部 spark_* 超参、DINO 侧复用全部 dino_*。
    # L = L_SparK(像素重建) + μ·L_DINO(全局蒸馏)；μ 为 DINO 损失权重。
    sparkdino_dino_weight: float = 1.0

    # --- 在线分割线性探针（SSL.md §0.5；驱动 best 选择，避免按 SSL loss 选模）---
    # 启用后每 probe_every 个 epoch 在 probe_data_dir 的标注 npz 上冻结 encoder 跑线性探针。
    probe_enabled: bool = False
    probe_data_dir: str = ""            # 标注 npz 目录（含 image+label），enabled 时必填
    probe_every: int = 10               # 探针评估的 epoch 间隔
    probe_iters: int = 100              # 每次评估训练线性头的步数
    probe_lr: float = 1.0e-2            # 线性头学习率
    probe_finetune: bool = False        # True: encoder 也参与训练（encoder-finetune 读数）
    probe_finetune_lr: float = 1.0e-3   # encoder-finetune 时 encoder 学习率
    probe_val_ratio: float = 0.3        # 探针 train/val 划分比例
    probe_samples_per_volume: int = 4   # 探针每卷抽样数
    probe_seed: int = 0                 # 线性头重置 + 划分种子（保证跨 epoch 可比）
    # 组级划分：npz 文件名 stem 即 pid；同一患者多序列用 group_regex 的第 1 个捕获
    # 组归并，避免同患者跨 train/val 泄漏（空=按文件名 stem 分组）。
    probe_group_regex: str = ""
    # 只有 1 个组时是否允许 train==val（默认 False：抛错，因读数无效不能选模）。
    probe_allow_single_group: bool = False
    probe_select_best: bool = True      # True: 以 probe_dice 选 best；False: 退回 train loss
    # 探针头结构："linear" — 逐尺度 1×1 线性头（上采样求和，严格线性探针）；
    # "unet" — 轻量 UNet 式自顶向下融合头（lateral 1×1 + 逐级上采样 3×3 融合），
    # 读数更接近下游真实分割能力（仍冻结/微调 encoder，仅头容量不同）。
    probe_head: str = "linear"
    probe_head_width: int = 16          # "unet" 头的统一通道宽度

    # --- 在线分类探针（SSL.md §0.4；encoder + GAP + MLP 头；frozen/finetune）---
    cls_probe_iters: int = 100
    cls_probe_lr: float = 1.0e-2
    cls_probe_hidden_dim: int = 128
    cls_probe_finetune: bool = False
    cls_probe_finetune_lr: float = 1.0e-3
    cls_label_key: str = ""

    # --- P6 离线评测 / few-shot 曲线（§0.4）---
    eval_data_dir: str = ""             # 留出评测用标注 npz 目录；空=复用 probe_data_dir
    eval_shots: List[int] = field(default_factory=lambda: [10, 30, 50, 100])
    eval_readouts: List[str] = field(default_factory=lambda: ["linear", "finetune"])
    eval_tasks: List[str] = field(default_factory=lambda: ["seg", "cls"])
    eval_out_dir: str = ""              # 输出目录；空=自动落到 train.output_dir/eval
    eval_holdout_ratio: float = 0.2
    eval_seed: int = 0
    eval_group_regex: str = ""           # 同 probe_group_regex（离线评测组级留出）
    eval_allow_single_group: bool = False
    eval_entries: List[str] = field(default_factory=list)

    # 周期性保存 SSL ckpt 的 epoch 间隔（best-by-train-recon 始终单独保存）。
    save_every: int = 10


def _warn_mask_unit_alignment(name: str, unit: int, cfg: SegConfig) -> None:
    """掩码单元与模型空间尺寸的整除性检查（仅告警，不阻断）。

    单元网格以 ``ceil(spatial/unit)`` 构造并用最近邻重采样映射到输入/特征分辨率：
    若 patch 尺寸不被 unit 整除，末尾单元偏小且掩码边界与特征位点会产生半单元
    错位（部分"可见"位点混入被遮上下文），默默削弱 MIM/JEPA/iBOT 目标。
    """
    patch = [int(s) for s in cfg.data.patch_size]
    model_spatial = patch if int(cfg.model.spatial_dims) == 3 else patch[1:]
    bad = [s for s in model_spatial if s % max(int(unit), 1) != 0]
    if bad:
        logger.warning(
            "ssl.%s=%d does not evenly divide model patch dims %s: the "
            "trailing mask unit is smaller and mask/feature alignment is "
            "only approximate (nearest resample). Recommend patch dims "
            "divisible by the mask unit.", name, int(unit), model_spatial)


def validate_ssl(ssl: SSLConfig, cfg: SegConfig) -> None:
    """校验 SSL 配置与骨干配置的一致性；非法时抛 ``ConfigError``。"""
    _require(
        ssl.method in METHODS,
        f"Invalid ssl.method: {ssl.method!r}. Valid: {METHODS}.")
    if ssl.method == "prior":
        _require(
            len(ssl.prior_scales) >= 1 and all(float(v) > 0 for v in ssl.prior_scales),
            f"ssl.prior_scales must be a non-empty list of positive sigmas; "
            f"got {ssl.prior_scales}.")
        _require(
            float(ssl.prior_alpha) > 0 and float(ssl.prior_beta) > 0,
            f"ssl.prior_alpha/prior_beta must be > 0; "
            f"got {ssl.prior_alpha}, {ssl.prior_beta}.")
        if ssl.prior_spacing:
            # target_3d 且 2.5D 时，目标在 (B,1,D,H,W) 体上按 3D 计算 → 需 3 轴
            # 间距 (sz,sy,sx)；否则按 spatial_dims 轴数。
            spacing_ndim = (3 if (bool(ssl.prior_target_3d)
                                  and int(cfg.model.spatial_dims) == 2)
                            else int(cfg.model.spatial_dims))
            _require(
                len(ssl.prior_spacing) == spacing_ndim
                and all(float(v) > 0 for v in ssl.prior_spacing),
                f"ssl.prior_spacing (if set) must have length {spacing_ndim} "
                f"(=3 for 2.5D prior_target_3d else model.spatial_dims "
                f"{cfg.model.spatial_dims}) and be all-positive; "
                f"got {ssl.prior_spacing}.")
    _require(
        0.0 <= float(ssl.sample_fg_ratio) <= 1.0,
        f"ssl.sample_fg_ratio must be in [0,1]; got {ssl.sample_fg_ratio}.")
    _require(
        int(ssl.sample_max_tries) >= 1,
        f"ssl.sample_max_tries must be >= 1; got {ssl.sample_max_tries}.")
    if ssl.method == "simmim":
        _require(
            0.0 < float(ssl.mim_mask_ratio) < 1.0,
            f"ssl.mim_mask_ratio must be in (0,1); got {ssl.mim_mask_ratio}.")
        _require(
            int(ssl.mim_mask_unit) >= 1,
            f"ssl.mim_mask_unit must be >= 1; got {ssl.mim_mask_unit}.")
        _require(
            int(ssl.mim_head_dim) >= 0,
            f"ssl.mim_head_dim must be >= 0 (0=auto); got {ssl.mim_head_dim}.")
        _warn_mask_unit_alignment("mim_mask_unit", int(ssl.mim_mask_unit), cfg)
    if ssl.method in ("spark", "sparkdino"):
        _require(
            0.0 < float(ssl.spark_mask_ratio) < 1.0,
            f"ssl.spark_mask_ratio must be in (0,1); got {ssl.spark_mask_ratio}.")
        _require(
            int(ssl.spark_mask_unit) >= 1,
            f"ssl.spark_mask_unit must be >= 1; got {ssl.spark_mask_unit}.")
        _warn_mask_unit_alignment(
            "spark_mask_unit", int(ssl.spark_mask_unit), cfg)
        _require(
            int(ssl.spark_decoder_dim_div) >= 1,
            f"ssl.spark_decoder_dim_div must be >= 1; got "
            f"{ssl.spark_decoder_dim_div}.")
        _require(
            int(ssl.spark_decoder_min_dim) >= 1,
            f"ssl.spark_decoder_min_dim must be >= 1; got "
            f"{ssl.spark_decoder_min_dim}.")
        _require(
            str(ssl.spark_decoder_mode) in ("light", "seg"),
            f"ssl.spark_decoder_mode must be 'light' or 'seg'; got "
            f"{ssl.spark_decoder_mode!r}.")
    if ssl.method in ("dino", "dino_gram", "ibot", "sparkdino"):
        _require(
            int(ssl.dino_out_dim) >= 1,
            f"ssl.dino_out_dim must be >= 1; got {ssl.dino_out_dim}.")
        _require(
            int(ssl.dino_hidden_dim) >= 1 and int(ssl.dino_bottleneck_dim) >= 1,
            f"ssl.dino_hidden_dim/bottleneck_dim must be >= 1; got "
            f"{ssl.dino_hidden_dim}, {ssl.dino_bottleneck_dim}.")
        _require(
            int(ssl.dino_head_layers) >= 1,
            f"ssl.dino_head_layers must be >= 1; got {ssl.dino_head_layers}.")
        _require(
            int(ssl.dino_global_crops) >= 2,
            f"ssl.dino_global_crops must be >= 2 (DINO needs >=2 global views); "
            f"got {ssl.dino_global_crops}.")
        _require(
            int(ssl.dino_local_crops) >= 0,
            f"ssl.dino_local_crops must be >= 0; got {ssl.dino_local_crops}.")
        _require(
            float(ssl.dino_student_temp) > 0 and float(ssl.dino_teacher_temp) > 0
            and float(ssl.dino_teacher_temp_warmup) > 0,
            f"ssl.dino_student_temp/teacher_temp/teacher_temp_warmup must be > 0; "
            f"got {ssl.dino_student_temp}, {ssl.dino_teacher_temp}, "
            f"{ssl.dino_teacher_temp_warmup}.")
        _require(
            0.0 < float(ssl.dino_warmup_teacher_temp_frac) <= 1.0,
            f"ssl.dino_warmup_teacher_temp_frac must be in (0,1]; got "
            f"{ssl.dino_warmup_teacher_temp_frac}.")
        _require(
            0.0 <= float(ssl.dino_freeze_last_layer_frac) <= 1.0,
            f"ssl.dino_freeze_last_layer_frac must be in [0,1]; got "
            f"{ssl.dino_freeze_last_layer_frac}.")
        _require(
            0.0 <= float(ssl.dino_center_momentum) < 1.0,
            f"ssl.dino_center_momentum must be in [0,1); got "
            f"{ssl.dino_center_momentum}.")
        _require(
            0.0 <= float(ssl.dino_momentum_base) <= float(ssl.dino_momentum_final) <= 1.0,
            f"ssl.dino_momentum_base/final must satisfy 0<=base<=final<=1; got "
            f"{ssl.dino_momentum_base}, {ssl.dino_momentum_final}.")
        for nm, rng in (("dino_global_scale", ssl.dino_global_scale),
                        ("dino_local_scale", ssl.dino_local_scale)):
            _require(
                len(rng) == 2 and 0.0 < float(rng[0]) <= float(rng[1]) <= 1.0,
                f"ssl.{nm} must be [lo,hi] with 0<lo<=hi<=1; got {rng}.")
        for nm, sz in (("dino_global_size", ssl.dino_global_size),
                       ("dino_local_size", ssl.dino_local_size)):
            if sz:
                _require(
                    len(sz) == int(cfg.model.spatial_dims)
                    and all(int(s) >= 1 for s in sz),
                    f"ssl.{nm} (if set) must have length == model.spatial_dims "
                    f"({cfg.model.spatial_dims}) and all >= 1; got {sz}.")
        _require(
            0.0 <= float(ssl.dino_flip_prob) <= 1.0,
            f"ssl.dino_flip_prob must be in [0,1]; got {ssl.dino_flip_prob}.")
        _require(
            float(ssl.dino_intensity_scale) >= 0 and float(ssl.dino_intensity_shift) >= 0,
            f"ssl.dino_intensity_scale/shift must be >= 0; got "
            f"{ssl.dino_intensity_scale}, {ssl.dino_intensity_shift}.")
        if bool(cfg.train.use_ema):
            logger.warning(
                "ssl.method=%r with train.use_ema=True: the DINO teacher "
                "is already an EMA of the student; the trainer-level EMA is "
                "redundant. Recommend train.use_ema=false.", ssl.method)
    if ssl.method == "jepa":
        _require(
            0.0 < float(ssl.jepa_mask_ratio) < 1.0,
            f"ssl.jepa_mask_ratio must be in (0,1); got {ssl.jepa_mask_ratio}.")
        _require(
            int(ssl.jepa_mask_unit) >= 1,
            f"ssl.jepa_mask_unit must be >= 1; got {ssl.jepa_mask_unit}.")
        _warn_mask_unit_alignment("jepa_mask_unit", int(ssl.jepa_mask_unit), cfg)
        _require(
            int(ssl.jepa_predictor_depth) >= 1,
            f"ssl.jepa_predictor_depth must be >= 1; got "
            f"{ssl.jepa_predictor_depth}.")
        _require(
            int(ssl.jepa_predictor_hidden) >= 0,
            f"ssl.jepa_predictor_hidden must be >= 0 (0=feature channels); got "
            f"{ssl.jepa_predictor_hidden}.")
        _require(
            0.0 < float(ssl.jepa_momentum_base) <= float(ssl.jepa_momentum_final) <= 1.0,
            f"ssl.jepa_momentum_base/final must satisfy 0<base<=final<=1; got "
            f"{ssl.jepa_momentum_base}, {ssl.jepa_momentum_final}.")
        _require(
            float(ssl.jepa_var_weight) >= 0.0 and float(ssl.jepa_cov_weight) >= 0.0,
            f"ssl.jepa_var_weight/cov_weight must be >= 0; got "
            f"{ssl.jepa_var_weight}, {ssl.jepa_cov_weight}.")
        n_levels = len(cfg.model.encoder_channels)
        _require(
            -n_levels <= int(ssl.jepa_feature_level) < n_levels,
            f"ssl.jepa_feature_level must index encoder_channels (len "
            f"{n_levels}); got {ssl.jepa_feature_level}.")
        if bool(cfg.train.use_ema):
            logger.warning(
                "ssl.method='jepa' with train.use_ema=True: the JEPA target "
                "encoder is already an EMA of the context encoder; the "
                "trainer-level EMA is redundant. Recommend train.use_ema=false.")
    if ssl.method == "dino_gram":
        _require(
            float(ssl.dino_gram_weight) >= 0.0,
            f"ssl.dino_gram_weight must be >= 0; got {ssl.dino_gram_weight}.")
        _require(
            0.0 <= float(ssl.dino_gram_start_frac) <= 1.0,
            f"ssl.dino_gram_start_frac must be in [0,1]; got "
            f"{ssl.dino_gram_start_frac}.")
        _require(
            int(ssl.dino_gram_refresh_steps) >= 1,
            f"ssl.dino_gram_refresh_steps must be >= 1; got "
            f"{ssl.dino_gram_refresh_steps}.")
        n_levels = len(cfg.model.encoder_channels)
        _require(
            -n_levels <= int(ssl.dino_gram_feature_level) < n_levels,
            f"ssl.dino_gram_feature_level must index encoder_channels "
            f"(len {n_levels}); got {ssl.dino_gram_feature_level}.")
    if ssl.method == "ibot":
        _require(
            float(ssl.ibot_weight) >= 0.0,
            f"ssl.ibot_weight must be >= 0; got {ssl.ibot_weight}.")
        _require(
            0.0 < float(ssl.ibot_mask_ratio) < 1.0,
            f"ssl.ibot_mask_ratio must be in (0,1); got {ssl.ibot_mask_ratio}.")
        _require(
            int(ssl.ibot_mask_unit) >= 1,
            f"ssl.ibot_mask_unit must be >= 1; got {ssl.ibot_mask_unit}.")
        _warn_mask_unit_alignment("ibot_mask_unit", int(ssl.ibot_mask_unit), cfg)
        _require(
            int(ssl.ibot_out_dim) >= 0,
            f"ssl.ibot_out_dim must be >= 0 (0=use dino_out_dim); got "
            f"{ssl.ibot_out_dim}.")
        n_levels = len(cfg.model.encoder_channels)
        _require(
            -n_levels <= int(ssl.ibot_feature_level) < n_levels,
            f"ssl.ibot_feature_level must index encoder_channels (len "
            f"{n_levels}); got {ssl.ibot_feature_level}.")
    if ssl.method == "byol":
        _require(
            int(ssl.dino_hidden_dim) >= 1,
            f"ssl.dino_hidden_dim must be >= 1; got {ssl.dino_hidden_dim}.")
        _require(
            int(ssl.byol_proj_dim) >= 1,
            f"ssl.byol_proj_dim must be >= 1; got {ssl.byol_proj_dim}.")
        _require(
            int(ssl.byol_pred_hidden_dim) >= 1,
            f"ssl.byol_pred_hidden_dim must be >= 1; got {ssl.byol_pred_hidden_dim}.")
        for nm, rng in (("dino_global_scale", ssl.dino_global_scale),
                        ("dino_local_scale", ssl.dino_local_scale)):
            _require(
                len(rng) == 2 and 0.0 < float(rng[0]) <= float(rng[1]) <= 1.0,
                f"ssl.{nm} must be [lo,hi] with 0<lo<=hi<=1; got {rng}.")
        _require(
            0.0 <= float(ssl.dino_flip_prob) <= 1.0,
            f"ssl.dino_flip_prob must be in [0,1]; got {ssl.dino_flip_prob}.")
        _require(
            float(ssl.dino_intensity_scale) >= 0 and float(ssl.dino_intensity_shift) >= 0,
            f"ssl.dino_intensity_scale/shift must be >= 0; got "
            f"{ssl.dino_intensity_scale}, {ssl.dino_intensity_shift}.")
        _require(
            0.0 < float(ssl.byol_momentum_base) <= float(ssl.byol_momentum_final) <= 1.0,
            f"ssl.byol_momentum_base/final must satisfy 0<base<=final<=1; got "
            f"{ssl.byol_momentum_base}, {ssl.byol_momentum_final}.")
        if bool(cfg.train.use_ema):
            logger.warning(
                "ssl.method='byol' with train.use_ema=True: the BYOL target "
                "encoder is already an EMA of the online encoder; the "
                "trainer-level EMA is redundant. Recommend train.use_ema=false.")
    if ssl.method == "moco":
        _require(
            int(ssl.dino_hidden_dim) >= 1,
            f"ssl.dino_hidden_dim must be >= 1; got {ssl.dino_hidden_dim}.")
        _require(
            int(ssl.moco_proj_dim) >= 1,
            f"ssl.moco_proj_dim must be >= 1; got {ssl.moco_proj_dim}.")
        _require(
            int(ssl.moco_queue_size) >= 2,
            f"ssl.moco_queue_size must be >= 2; got {ssl.moco_queue_size}.")
        _require(
            float(ssl.moco_temperature) > 0,
            f"ssl.moco_temperature must be > 0; got {ssl.moco_temperature}.")
        for nm, rng in (("dino_global_scale", ssl.dino_global_scale),
                        ("dino_local_scale", ssl.dino_local_scale)):
            _require(
                len(rng) == 2 and 0.0 < float(rng[0]) <= float(rng[1]) <= 1.0,
                f"ssl.{nm} must be [lo,hi] with 0<lo<=hi<=1; got {rng}.")
        _require(
            0.0 <= float(ssl.dino_flip_prob) <= 1.0,
            f"ssl.dino_flip_prob must be in [0,1]; got {ssl.dino_flip_prob}.")
        _require(
            float(ssl.dino_intensity_scale) >= 0 and float(ssl.dino_intensity_shift) >= 0,
            f"ssl.dino_intensity_scale/shift must be >= 0; got "
            f"{ssl.dino_intensity_scale}, {ssl.dino_intensity_shift}.")
        _require(
            0.0 < float(ssl.moco_momentum_base) <= float(ssl.moco_momentum_final) <= 1.0,
            f"ssl.moco_momentum_base/final must satisfy 0<base<=final<=1; got "
            f"{ssl.moco_momentum_base}, {ssl.moco_momentum_final}.")
        if bool(cfg.train.use_ema):
            logger.warning(
                "ssl.method='moco' with train.use_ema=True: the MoCo key "
                "encoder is already an EMA of the query encoder; the "
                "trainer-level EMA is redundant. Recommend train.use_ema=false.")
    if ssl.method == "vicregl":
        for name in ("vicregl_proj_dim", "vicregl_hidden_dim",
                     "vicregl_dense_proj_dim", "vicregl_num_matches"):
            _require(
                int(getattr(ssl, name)) >= 1,
                f"ssl.{name} must be >= 1; got {getattr(ssl, name)}.")
        _require(
            int(ssl.vicregl_feature_matches) >= 0,
            f"ssl.vicregl_feature_matches must be >= 0; "
            f"got {ssl.vicregl_feature_matches}.")
        for name in ("vicregl_sim_coeff", "vicregl_var_coeff",
                     "vicregl_cov_coeff"):
            _require(
                float(getattr(ssl, name)) >= 0.0,
                f"ssl.{name} must be >= 0; got {getattr(ssl, name)}.")
        _require(
            0.0 <= float(ssl.vicregl_alpha) <= 1.0,
            f"ssl.vicregl_alpha must be in [0,1]; got {ssl.vicregl_alpha}.")
        n_levels = len(cfg.model.encoder_channels)
        _require(
            -n_levels <= int(ssl.vicregl_feature_level) < n_levels,
            f"ssl.vicregl_feature_level must index encoder_channels "
            f"(len {n_levels}); got {ssl.vicregl_feature_level}.")
        sc = [float(v) for v in ssl.vicregl_crop_scale]
        _require(
            len(sc) == 2 and 0.0 < sc[0] <= sc[1] <= 1.0,
            f"ssl.vicregl_crop_scale must be [lo, hi] with 0 < lo <= hi <= 1; "
            f"got {ssl.vicregl_crop_scale}.")
    if ssl.method == "sparkdino":
        _require(
            float(ssl.sparkdino_dino_weight) >= 0.0,
            f"ssl.sparkdino_dino_weight must be >= 0; got "
            f"{ssl.sparkdino_dino_weight}.")
    _require(
        ssl.recon_loss in RECON_LOSSES,
        f"Invalid ssl.recon_loss: {ssl.recon_loss!r}. Valid: {RECON_LOSSES}.")
    # SSL 重建路径走单视图（与分割同构 UNet，只换重建头）。
    _require(
        str(cfg.model.arch).lower() == "unet",
        f"ssl requires model.arch=='unet'; got {cfg.model.arch!r}.")
    _require(
        len(cfg.data.multi_res_scales) == 1,
        f"ssl requires a single view (len(data.multi_res_scales)==1); "
        f"got {cfg.data.multi_res_scales}.")
    for name in ("nonlinear_prob", "local_shuffle_prob",
                 "paint_prob", "inner_paint_prob"):
        v = float(getattr(ssl, name))
        _require(0.0 <= v <= 1.0, f"ssl.{name} must be in [0,1]; got {v}.")
    _require(
        int(ssl.local_shuffle_blocks) >= 0,
        f"ssl.local_shuffle_blocks must be >= 0; got {ssl.local_shuffle_blocks}.")
    _require(
        int(ssl.paint_count) >= 0,
        f"ssl.paint_count must be >= 0; got {ssl.paint_count}.")
    _require(
        all(int(b) >= 1 for b in ssl.local_shuffle_max_block),
        f"ssl.local_shuffle_max_block must all be >= 1; "
        f"got {ssl.local_shuffle_max_block}.")
    pr = ssl.paint_block_range
    _require(
        len(pr) == 2 and 0.0 < float(pr[0]) <= float(pr[1]) < 1.0,
        f"ssl.paint_block_range must be [lo,hi] with 0<lo<=hi<1; got {pr}.")
    if ssl.probe_enabled:
        _require(
            bool(ssl.probe_data_dir),
            "ssl.probe_enabled=True requires ssl.probe_data_dir "
            "(a directory of labelled image+label npz).")
        _require(
            int(cfg.model.spatial_dims) in (2, 3),
            f"ssl.probe_enabled supports 3D (spatial_dims==3) and 2.5D "
            f"(spatial_dims==2) backbones; got {cfg.model.spatial_dims}.")
        _require(
            int(ssl.probe_every) >= 1,
            f"ssl.probe_every must be >= 1; got {ssl.probe_every}.")
        _require(
            int(ssl.probe_iters) >= 1,
            f"ssl.probe_iters must be >= 1; got {ssl.probe_iters}.")
        _require(
            0.0 < float(ssl.probe_val_ratio) < 1.0,
            f"ssl.probe_val_ratio must be in (0,1); got {ssl.probe_val_ratio}.")
        _require(
            int(ssl.probe_samples_per_volume) >= 1,
            f"ssl.probe_samples_per_volume must be >= 1; "
            f"got {ssl.probe_samples_per_volume}.")
    for _name in ("probe_group_regex", "eval_group_regex"):
        _rx = str(getattr(ssl, _name))
        if _rx:
            try:
                _compiled = re.compile(_rx)
            except re.error as e:
                raise ConfigError(f"ssl.{_name} is not a valid regex: {e}")
            _require(
                _compiled.groups >= 1,
                f"ssl.{_name} must contain at least one capture group "
                f"(the group key); got {_rx!r}.")
    _require(
        str(ssl.probe_head) in ("linear", "unet"),
        f"ssl.probe_head must be 'linear' or 'unet'; got {ssl.probe_head!r}.")
    _require(
        int(ssl.probe_head_width) >= 1,
        f"ssl.probe_head_width must be >= 1; got {ssl.probe_head_width}.")
    _require(
        int(ssl.cls_probe_iters) >= 1,
        f"ssl.cls_probe_iters must be >= 1; got {ssl.cls_probe_iters}.")
    _require(
        float(ssl.cls_probe_lr) > 0 and float(ssl.cls_probe_finetune_lr) > 0,
        f"ssl.cls_probe_lr/cls_probe_finetune_lr must be > 0; got "
        f"{ssl.cls_probe_lr}, {ssl.cls_probe_finetune_lr}.")
    _require(
        int(ssl.cls_probe_hidden_dim) >= 1,
        f"ssl.cls_probe_hidden_dim must be >= 1; got "
        f"{ssl.cls_probe_hidden_dim}.")
    _require(
        0.0 < float(ssl.eval_holdout_ratio) < 1.0,
        f"ssl.eval_holdout_ratio must be in (0,1); got {ssl.eval_holdout_ratio}.")
    _require(
        all(int(s) >= 1 for s in ssl.eval_shots),
        f"ssl.eval_shots must contain positive integers; got {ssl.eval_shots}.")
    _require(
        all(str(r) in ("linear", "finetune") for r in ssl.eval_readouts),
        f"ssl.eval_readouts must be subset of ['linear', 'finetune']; got "
        f"{ssl.eval_readouts}.")
    _require(
        all(str(t) in ("seg", "cls") for t in ssl.eval_tasks),
        f"ssl.eval_tasks must be subset of ['seg', 'cls']; got {ssl.eval_tasks}.")
    _require(
        int(ssl.save_every) >= 1,
        f"ssl.save_every must be >= 1; got {ssl.save_every}.")


# ---------------------------------------------------------------------------
# YAML I/O + overrides
# ---------------------------------------------------------------------------
def _ssl_from_dict(d: Dict[str, Any]) -> SSLConfig:
    """从 ``ssl:`` YAML 段构造 :class:`SSLConfig`（未知键仅告警并忽略）。"""
    return _dataclass_from_dict(SSLConfig, dict(d or {}))


def _seg_from_dict(raw: Dict[str, Any]) -> SegConfig:
    """从 YAML（已剔除 ``ssl`` 段）构造并校验 segtask ``Config``。"""
    cfg = _dataclass_from_dict(SegConfig, raw)
    cfg.sync()
    cfg.validate()
    return cfg


def load_config(path: Union[str, Path]) -> Tuple[SegConfig, SSLConfig]:
    """加载 ssltask YAML，返回 ``(cfg, ssl)``。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    ssl_raw = dict(raw.pop("ssl", {}) or {})
    cfg = _seg_from_dict(raw)
    ssl = _ssl_from_dict(ssl_raw)
    validate_ssl(ssl, cfg)
    return cfg, ssl


def save_config(cfg: SegConfig, ssl: SSLConfig, path: Union[str, Path]) -> None:
    """把 ``(cfg, ssl)`` 落盘为单个 YAML（``ssl`` 段覆盖 seg 端可能残留的同名段）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = asdict(cfg)
    blob["ssl"] = asdict(ssl)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(blob, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)


def _coerce(old: Any, val: str) -> Any:
    """按既有字段类型把字符串 override 值转回原类型（与 segtask 行为一致）。"""
    if isinstance(old, bool):
        return val.lower() in ("true", "1", "yes")
    if isinstance(old, int):
        return int(val)
    if isinstance(old, float):
        return float(val)
    if isinstance(old, list):
        import json
        return json.loads(val)
    return val


def _set_dotted(obj: Any, dotted: str, val: str) -> None:
    parts = dotted.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    attr = parts[-1]
    old = getattr(obj, attr)
    new = _coerce(old, val)
    setattr(obj, attr, new)
    logger.info("Override: %s = %s -> %s", dotted, old, new)


def apply_overrides(cfg: SegConfig, ssl: SSLConfig, overrides: List[str]) -> None:
    """应用点记法 override；``ssl.*`` 路由到 ``ssl``，其余路由到 ``cfg``。

    示例：``--override train.epochs=50 ssl.method=prior ssl.recon_loss=mse``。
    调用方应在其后自行 ``cfg.sync(); cfg.validate(); validate_ssl(ssl, cfg)``。
    """
    for ov in overrides:
        if "=" not in ov:
            continue
        key, val = ov.split("=", 1)
        if key.startswith("ssl."):
            _set_dotted(ssl, key[len("ssl."):], val)
        else:
            _set_dotted(cfg, key, val)


__all__ = [
    "SSLConfig", "SegConfig", "ConfigError",
    "validate_ssl", "load_config", "save_config", "apply_overrides",
    "RECON_LOSSES", "METHODS",
]
