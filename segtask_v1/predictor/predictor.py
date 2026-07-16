"""3D 分割滑窗推理。

推理模式与训练 patch 模式一致：z_axis / cubic / whole / 2_5d。
支持 overlap 融合（gaussian/uniform）、flip TTA、AMP、多分辨率输入、
ckpt 加载（兼容 torch.compile / EMA / best-model EMA-primary）。
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import torch

from ..config import Config
from ..data.dataset import (
    load_nifti, load_nifti_with_spacing, preprocess_image,
    compute_bbox_from_volume, read_nifti_spacing, resample_to_spacing,
    resize_3d)
from ..models.topology import ModelTopology, build_topology
from ..trainer.amp import resolve_auto_amp_dtype
from . import blending as _blending
from . import sliding as _sliding

logger = logging.getLogger(__name__)


_AMP_DTYPES = {
    "float16": torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}


def _manifest_target_spacing(npz_dir: str) -> Optional[List[float]]:
    """从 make_data 写入的 ``npz_dir/_manifest.json`` 回读解析后的
    target_spacing（(D,H,W) mm）；目录/文件/字段缺失或非法时返 None。"""
    if not npz_dir:
        return None
    p = Path(npz_dir) / "_manifest.json"
    if not p.is_file():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            ts = json.load(f).get("target_spacing")
    except (OSError, ValueError) as exc:
        logger.warning("Failed to read %s: %s", p, exc)
        return None
    if (isinstance(ts, (list, tuple)) and len(ts) == 3
            and all(isinstance(s, (int, float)) and s > 0 for s in ts)):
        return [float(s) for s in ts]
    return None


class Predictor:
    """3D 分割滑窗推理器。假定多标签 sigmoid 训练，前 num_fg 个输出通道
    1∶1 对应 cfg.data.label_values[1:]。"""

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: Config,
        device: torch.device,
    ):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.model.eval()

        pc = cfg.predict
        self.overlap = pc.z_overlap            # z 轴（cubic 下 H/W 见 hw_overlap）
        # cubic H/W 轴重叠比：None 时沿用 z_overlap（三轴同值，现状不变）。
        self.hw_overlap = (
            float(pc.hw_overlap) if pc.hw_overlap is not None
            else float(pc.z_overlap))
        self.blend_mode = pc.blend_mode
        self.batch_size = pc.batch_size
        self.tta_flip = pc.tta_flip
        # TTA flip 变体批量化块大小（None → 退化为 batch_size）；见 forwards._tta_chunk_size。
        self.tta_batch_size: Optional[int] = pc.tta_batch_size
        # 标量（全类共享）或逐前景类列表；长度校验在 num_fg 确定后（见下方）。
        self.threshold = (
            [float(t) for t in pc.threshold]
            if isinstance(pc.threshold, (list, tuple)) else float(pc.threshold))
        # 诊断日志用的标量下界（逐类阈值时取最小值）。
        self.threshold_min = float(np.min(self.threshold))
        self.save_probs = pc.save_probabilities
        # 滑窗概率累加器 dtype / 落点（大卷 × 多类的显存逃生门）。
        self.acc_dtype = (torch.float16 if str(pc.acc_dtype) == "fp16"
                          else torch.float32)
        self.acc_device = (torch.device("cpu") if pc.accumulate_on_cpu
                           else device)
        # 推理前向用 inference_mode 替代 no_grad（数值等价，免 version-counter
        # 簿记）；AdaBN per_volume 估计期自动回退 no_grad，见 _forward_grad_ctx。
        self.use_inference_mode = bool(pc.use_inference_mode)
        # 滑窗跳过纯背景窗（归一化后窗内最大值 <= 阈值 → 不前向、概率保持 0）。
        self.skip_empty_windows = bool(pc.skip_empty_windows)
        self.skip_empty_threshold = float(pc.skip_empty_threshold)
        # 推理侧 channels_last：数值等价的内存排布转换（仅 CUDA 有意义）。
        # 模型权重此处转换；输入窗口在 forwards._to_channels_last 逐 batch 转换。
        self.channels_last = bool(pc.channels_last) and device.type == "cuda"
        if self.channels_last:
            fmt = (torch.channels_last_3d
                   if int(cfg.model.spatial_dims) == 3
                   else torch.channels_last)
            self.model = self.model.to(memory_format=fmt)
            logger.info(
                "Predictor channels_last=True: model converted to %s "
                "(numerically equivalent memory-format change).", fmt)
        # GPU 常驻整卷张量的存储 dtype（fp16 = 整卷显存减半；builder 取窗时按窗
        # 升回 fp32，见 config.PredictConfig.vol_dtype）。
        self.vol_dtype = (torch.float16 if str(pc.vol_dtype) == "fp16"
                          else torch.float32)
        # 逐卷滑窗进度日志开关（运行期内部量，不暴露到配置）：CLI 推理默认 True；
        # 训练内整卷验证（VolumeValEvaluator）会置 False 以免每 epoch 刷屏 81 卷。
        self.log_progress = True

        # 2.5D z 交错推理：按 stride k 将体积拆为 k 个子流，各自走 sliding.sliding_window_z，
        # 后以 out[:, i::k]=stream_i 缝回。k 由 z spacing 挑选（见 sliding.choose_interleave_factor）。
        # 仅 patch_mode=='2_5d' 生效；Config 验证。
        self.z_interleave_enabled = bool(
            pc.z_interleave_enabled
            and cfg.data.patch_mode == "2_5d")
        self.z_interleave_thresholds: List[float] = list(
            pc.z_interleave_thresholds)
        self.z_interleave_factors: List[int] = [
            int(f) for f in pc.z_interleave_factors]
        if self.z_interleave_enabled:
            logger.info(
                "Predictor z_interleave_enabled=True (2.5D only): "
                "thresholds=%s mm, factors=%s",
                self.z_interleave_thresholds, self.z_interleave_factors)

        # 测试时自适应 BatchNorm — per_volume 模式：每卷推理前用该卷自身重估 BN，
        # 再冻结预测（transductive BN）。global 模式在 run_inference 中处理，与此无关。
        self.adabn_enabled = bool(pc.adabn_enabled)
        self.adabn_mode = pc.adabn_mode
        # BN 估计期滑窗抽样比：<1 时估计前向只跑部分窗口（见 _adabn_keep_window）。
        self.adabn_sample_ratio = float(pc.adabn_sample_ratio)
        self._adabn_bn_modules: List[torch.nn.Module] = []
        if self.adabn_enabled and self.adabn_mode == "per_volume":
            from .adabn import collect_bn_modules
            self._adabn_bn_modules = collect_bn_modules(self.model)
            if not self._adabn_bn_modules:
                logger.warning(
                    "[AdaBN] per_volume enabled but model has no BatchNorm "
                    "layers (norm_type != 'batch'); per-volume adaptation "
                    "will be a no-op.")
            else:
                logger.info(
                    "[AdaBN] per_volume enabled: %d BatchNorm layer(s) will "
                    "be re-estimated from each volume before its prediction.",
                    len(self._adabn_bn_modules))

        # 物理 spacing 归一化（B1）：推理前把体积镜像重采样到 target_spacing，
        # 概率图再回采到原分辨率。须与烘焙用的 target_spacing 一致：显式配置优先，
        # 否则回读 make_data 写入 npz_dir/_manifest.json 的解析值（自动中位数）。
        self.spacing_norm = bool(cfg.data.spacing_normalization)
        self.target_spacing: Optional[Tuple[float, float, float]] = None
        if self.spacing_norm:
            ts = cfg.data.target_spacing
            if ts is None:
                ts = _manifest_target_spacing(cfg.data.npz_dir)
                if ts is not None:
                    logger.info(
                        "data.target_spacing not set; using target_spacing=%s "
                        "mm recorded by make_data in %s/_manifest.json.",
                        ts, cfg.data.npz_dir)
            if ts is None:
                raise ValueError(
                    "data.spacing_normalization=True requires "
                    "data.target_spacing [sz, sy, sx] (mm) for inference so it "
                    "matches the spacing used when baking npz. Set it "
                    "explicitly, or point data.npz_dir at the baked dataset "
                    "whose _manifest.json records the resolved value "
                    "(make_data >= 1.5).")
            self.target_spacing = tuple(float(s) for s in ts)
            logger.info(
                "Predictor spacing_normalization=True: resample inputs to "
                "target_spacing=%s mm (D,H,W), resample probabilities back.",
                self.target_spacing)

        self.patch_mode = cfg.data.patch_mode
        self.patch_D, self.patch_H, self.patch_W = cfg.data.patch_size
        self.label_values = cfg.data.label_values
        self.num_fg = cfg.num_fg_classes
        if (isinstance(self.threshold, list)
                and len(self.threshold) != self.num_fg):
            raise ValueError(
                f"predict.threshold list length {len(self.threshold)} != "
                f"num_fg {self.num_fg} (must map 1:1 to label_values[1:]).")
        # 默认单分辨率，避免下游 np.stack 报错。
        self.multi_res_scales = cfg.data.multi_res_scales or [1.0]
        # 与 DataConfig.z_boundary_mode 同步，使训/用边界处理几何一致。
        self.z_boundary_mode = cfg.data.z_boundary_mode
        if self.z_boundary_mode not in ("stretch", "edge_pad"):
            raise ValueError(
                f"Unknown z_boundary_mode {self.z_boundary_mode!r}; "
                "expected 'stretch' or 'edge_pad'.")

        # ---- R6：所有 mode 派生量来自 ModelTopology ----------------------
        # 重构前本 ``__init__`` 自行重算 ``lift_2_5d_to_3d`` / ``keep_native_view_depth``
        # / ``keep_native_multi_res`` / per-view depth / max-FOV target 等 ~80 行，
        # 与 Config.sync + models.factory.build_model 三处重复。R5 引入 ModelTopology
        # 之后这些应当只来自一处真相源。
        topo: ModelTopology = build_topology(cfg)
        self.topo = topo
        self.lift_2_5d_to_3d = topo.lift_2_5d_to_3d
        self.keep_native_view_depth = topo.keep_native_view_depth
        self.keep_native_multi_res = topo.keep_native_multi_res

        if self.lift_2_5d_to_3d:
            logger.info(
                "Predictor lift_2_5d_to_3d=True: 2.5D windows fed straight "
                "to a true-3D UNet (n_views=%d, in_channels=%d, output "
                "shape (B, num_fg=%d, pD=%d, pH=%d, pW=%d)).",
                topo.n_views, topo.in_channels,
                int(self.num_fg), int(self.patch_D),
                int(self.patch_H), int(self.patch_W))

        # 2.5D native_d 推理路径：(B, sum_k D_k, pH, pW)，D_k = round(pD*s_k)
        if self.keep_native_view_depth:
            self.per_view_depths: List[int] = list(topo.per_view_depths)
            self._eD_max: int = int(round(
                self.patch_D * float(max(self.multi_res_scales))))
            # 与模型实际 in_channels 一致性检查（topology 已保证，此处仅防御 stale-cfg）。
            expect_in = sum(self.per_view_depths)
            actual_in = int(cfg.model.in_channels)
            if actual_in != expect_in:
                raise ValueError(
                    f"keep_native_view_depth=True: model.in_channels={actual_in} "
                    f"!= sum(per_view_depths)={expect_in}. The model was "
                    "likely built with a stale Config — re-sync and rebuild.")
            logger.info(
                "Predictor keep_native_view_depth=True: per-view depths=%s, "
                "max-FOV cube depth=%d, in_channels=%d.",
                self.per_view_depths, self._eD_max, actual_in)
        else:
            self.per_view_depths = []
            self._eD_max = int(self.patch_D)

        # 3D 懒 max-FOV cube 推理路径：(B, C_res, pD, pH, pW)；
        # 几何尺寸（per-view native (D_k, H_k, W_k) + max-FOV target）依赖 patch_mode
        # ∈ {z_axis, cubic} 的几何语义，topology 不涉及空间布局，仅模式判定，故下面
        # 仍由 predictor 计算（与 dataset / trainer 的同名几何一致）。
        if self.keep_native_multi_res:
            sizes: List[Tuple[int, int, int]] = []
            for s in self.multi_res_scales:
                D_k = int(round(self.patch_D * float(s)))
                if self.patch_mode == "z_axis":
                    H_k, W_k = int(self.patch_H), int(self.patch_W)
                else:  # cubic
                    H_k = int(round(self.patch_H * float(s)))
                    W_k = int(round(self.patch_W * float(s)))
                sizes.append((D_k, H_k, W_k))
            sizes[0] = (int(self.patch_D), int(self.patch_H), int(self.patch_W))
            self._mr_native_sizes: List[Tuple[int, int, int]] = sizes
            ms = float(max(self.multi_res_scales))
            if self.patch_mode == "z_axis":
                self._mr_target_shape: Tuple[int, int, int] = (
                    int(round(self.patch_D * ms)),
                    int(self.patch_H),
                    int(self.patch_W))
            else:
                self._mr_target_shape = (
                    int(round(self.patch_D * ms)),
                    int(round(self.patch_H * ms)),
                    int(round(self.patch_W * ms)))
            logger.info(
                "Predictor keep_native_multi_res=True (%s): per-view "
                "native sizes=%s, max-FOV target=%s, n_views=%d.",
                self.patch_mode, sizes, self._mr_target_shape, topo.n_views)
        else:
            self._mr_native_sizes = []
            self._mr_target_shape = (
                int(self.patch_D), int(self.patch_H), int(self.patch_W))

        # AMP：与训练同 dtype（'auto' 按与 trainer 相同规则按设备解析）。
        # 未知值退 bf16 避免静默切换。
        amp_name = cfg.train.amp_dtype
        if amp_name == "auto":
            amp_name = resolve_auto_amp_dtype(device)
        if amp_name not in _AMP_DTYPES:
            logger.warning("Unknown amp_dtype=%r, falling back to bfloat16.",
                           amp_name)
            amp_name = "bfloat16"
        self.amp_dtype = _AMP_DTYPES[amp_name]
        self.use_amp = (
            cfg.train.use_amp and device.type == "cuda")

        # 推理 dtype：run_inference 可能将 model.half()，下面依据 model_dtype
        # 调节输入转型与是否启用 autocast（fp16/bf16 下不启 autocast）。
        try:
            self.model_dtype = next(model.parameters()).dtype
        except StopIteration:
            self.model_dtype = torch.float32
        if self.model_dtype in (torch.float16, torch.bfloat16):
            if self.use_amp:
                logger.info(
                    "Predictor: model weights are in %s — disabling autocast "
                    "(inputs will be cast to %s before each forward).",
                    self.model_dtype, self.model_dtype)
            self.use_amp = False

        # 边界填充值：归一化后 0 不一定是“空气”（z-score 下空气≈0 附近）。未设时走 mode='edge'。
        self.pad_value: Optional[float] = getattr(
            cfg.data, "pad_value", None)

        # 逐卷“首 batch 已记录”护孔。predict_volume 顶部重置，forwards.forward_batch* 消费。
        self._diag_first_batch_logged: bool = True

        # AdaBN per_volume 估计期标志：置 True 时 TTA 强制退回串行前向，避免把多个 flip
        # 变体拼成一个大 batch 改变 BN 统计构成（BN 估计期处于 train 模式 + 累积平均，
        # batch 构成会影响 running stats）。真实 eval 预测路径不受影响、仍走批量化。
        self._adabn_estimating: bool = False

        # 约定：输出通道 ↔ label_values[1:]
        if len(self.label_values) - 1 != self.num_fg:
            raise ValueError(
                f"num_fg_classes={self.num_fg} inconsistent with "
                f"label_values={self.label_values} (expected "
                f"{len(self.label_values) - 1} foreground labels).")

    # ==================================================================
    # Public API
    # ==================================================================
    def _log_normalized_input_stats(self, vol: np.ndarray) -> None:
        """诊断：归一化后输入统计（与训练不一致时 range/分位数会明显偏差）。"""
        dc = self.cfg.data
        try:
            vmin = float(vol.min()); vmax = float(vol.max())
            vmean = float(vol.mean()); vstd = float(vol.std())
            q = np.quantile(vol, [0.01, 0.5, 0.99])
            logger.info(
                "[diag] normalized input: shape=%s, min=%.4f, max=%.4f, "
                "mean=%.4f, std=%.4f, q1=%.4f, q50=%.4f, q99=%.4f "
                "(normalize=%s, intensity=[%.1f,%.1f])",
                tuple(vol.shape), vmin, vmax, vmean, vstd,
                float(q[0]), float(q[1]), float(q[2]),
                dc.normalize, float(dc.intensity_min), float(dc.intensity_max))
        except Exception as e:
            logger.warning("[diag] normalized-input stat failed: %s", e)

    def _log_inroi_prob_stats(self, prob_volume: np.ndarray) -> None:
        """诊断：blend 后概率统计（拼回原画布前计算，避免 ROI 外 0 偏移统计）。

        frac_gt_thr ≈1.0 提示是模型输出本身饱和（训练侧问题），非后处理。
        """
        try:
            max_per_vox = prob_volume.max(axis=0)
            frac_gt_thr = float((max_per_vox >= self.threshold_min).mean())
            q = np.quantile(prob_volume, [0.5, 0.9, 0.99, 0.999])
            logger.info(
                "[diag] in-ROI prob volume: shape=%s, min=%.4f, max=%.4f, "
                "mean=%.4f, q50=%.4f, q90=%.4f, q99=%.4f, q999=%.4f, "
                "frac(max_prob>=%.2f)=%.4f",
                tuple(prob_volume.shape), float(prob_volume.min()),
                float(prob_volume.max()), float(prob_volume.mean()),
                float(q[0]), float(q[1]), float(q[2]), float(q[3]),
                self.threshold_min, frac_gt_thr)
            if frac_gt_thr > 0.95:
                logger.warning(
                    "[diag] %.1f%% of in-ROI voxels exceed threshold — "
                    "the model itself is outputting near-saturated "
                    "foreground; this is a TRAINING-side issue (most "
                    "likely: training bbox/label semantics differ from "
                    "this run, OR region weights drove the model to a "
                    "trivial 'all-fg' minimum). Re-check cfg.data.bbox_dir, "
                    "label_dir and region_weight_dir USED ON THE SERVER.",
                    100.0 * frac_gt_thr)
        except Exception as e:
            logger.warning("[diag] prob-volume stat failed: %s", e)

    @torch.no_grad()
    def predict_volume(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        bbox_path: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """对单卷 NIfTI 推理。返回 {label_map (D,H,W) int, probabilities (num_fg,D,H,W) fp32}。

        patch_mode 调度：whole 全卷/cubic 3 轴滑窗/z_axis z 轴滑窗/2_5d 同 z_axis 几何但 forward 是 2D。
        bbox_path 可选：在 bbox 内推理后拼回原尺寸画布，外部为背景。
        """
        dc = self.cfg.data
        # 仅 z-interleave 需物理 z spacing；其余走 load_nifti 以保旧数值。
        if self.z_interleave_enabled:
            raw_vol, z_spacing = load_nifti_with_spacing(image_path)
        else:
            raw_vol = load_nifti(image_path)
            z_spacing = None
        D_orig, H_orig, W_orig = raw_vol.shape
        if z_spacing is not None:
            logger.info(
                "Loaded %s: shape=(%d, %d, %d), z_spacing=%.4f mm",
                image_path, D_orig, H_orig, W_orig, z_spacing)
        else:
            logger.info("Loaded %s: shape=(%d, %d, %d)",
                        image_path, D_orig, H_orig, W_orig)

        # 可选 ROI 裁剪：保留 (offsets, full_shape) 以便推理后拼回原坐标系。
        bbox = None
        if bbox_path is not None:
            bbox_vol = load_nifti(bbox_path)
            if bbox_vol.shape != raw_vol.shape:
                raise ValueError(
                    f"BBox shape {bbox_vol.shape} != image shape "
                    f"{raw_vol.shape} for {image_path} (bbox={bbox_path})")
            bbox = compute_bbox_from_volume(bbox_vol)
            if bbox is None:
                logger.warning(
                    "BBox %s is empty; falling back to full-volume "
                    "inference for %s.", bbox_path, image_path)
            else:
                (d0, d1), (h0, h1), (w0, w1) = bbox
                logger.info(
                    "BBox crop: D[%d:%d] H[%d:%d] W[%d:%d] "
                    "(orig=(%d,%d,%d) → crop=(%d,%d,%d))",
                    d0, d1, h0, h1, w0, w1,
                    D_orig, H_orig, W_orig,
                    d1 - d0, h1 - h0, w1 - w0)
                raw_vol = raw_vol[d0:d1, h0:h1, w0:w1]

        # 物理 spacing 归一化：把（可能已 bbox 裁剪的）体积从原生 spacing 镜像重采样
        # 到 target_spacing，推理后再把概率图回采到此处记录的 pre-resample 形状。
        pre_resample_shape: Optional[Tuple[int, int, int]] = None
        if self.spacing_norm and self.target_spacing is not None:
            src_spacing = read_nifti_spacing(image_path)
            pre_resample_shape = raw_vol.shape
            raw_vol = resample_to_spacing(
                raw_vol, src_spacing, self.target_spacing, is_label=False)
            logger.info(
                "spacing_normalization: %s mm → %s mm, shape %s → %s.",
                src_spacing, self.target_spacing,
                pre_resample_shape, raw_vol.shape)
            # 归一化后 z 已是 target_spacing[0]，供 z-interleave 一致选因子。
            if z_spacing is not None:
                z_spacing = float(self.target_spacing[0])

        vol = preprocess_image(
            raw_vol, dc.intensity_min, dc.intensity_max,
            dc.normalize, dc.global_mean, dc.global_std)

        self._log_normalized_input_stats(vol)
        # AdaBN per_volume：用该卷自身先跑一遍前向重估 BN running stats，再冻结预测
        # （每卷推理成本 2×）。估计期 TTA 仅降为串行、flip 变体仍前向，故 BN
        # 统计含原图+各 flip 的混合分布——与真实预测（也含 TTA）自洽，属有意取舍；
        # 若期望 BN 只反映原图分布，需在估计期跳过 TTA。
        # 估计期暂时抑制 forward 诊断（置护孔为已记录），随后再重置以让真实预测发一次。
        if (self.adabn_enabled and self.adabn_mode == "per_volume"
                and self._adabn_bn_modules):
            from . import adabn as _adabn
            logger.info(
                "[AdaBN] per_volume: re-estimating BN stats from this "
                "volume before prediction.")
            self._diag_first_batch_logged = True
            # 估计期强制 TTA 串行（见 self._adabn_estimating 注释）。
            self._adabn_estimating = True
            try:
                _adabn.estimate_bn_stats(
                    self._adabn_bn_modules,
                    lambda: self.predict_preprocessed_array(
                        vol, z_spacing=z_spacing))
            finally:
                self._adabn_estimating = False

        # 重置诊断护孔使 forward 路径发一次 logits/prob 统计块。
        self._diag_first_batch_logged = False

        prob_volume = self.predict_preprocessed_array(vol, z_spacing=z_spacing)

        # 备忘：此诊断在 spacing 回采之前，spacing_normalization=True 时统计口径
        # 是 target-spacing 分辨率而非原生分辨率（仅展示层，数值影响可忽）。
        self._log_inroi_prob_stats(prob_volume)

        # spacing 归一化：把概率图从 target-spacing 分辨率回采到 pre-resample 形状
        # （bbox 裁剪后的原生分辨率），随后再走原有 bbox 拼回逻辑。
        if pre_resample_shape is not None and prob_volume.shape[1:] != pre_resample_shape:
            prob_volume = resize_3d(
                prob_volume, pre_resample_shape[0], pre_resample_shape[1],
                pre_resample_shape[2], is_label=False)

        # 拼回原尺寸画布以保 NIfTI affine 一致；bbox 外体素保留 0 概率。
        if bbox is not None:
            (d0, d1), (h0, h1), (w0, w1) = bbox
            full_prob = np.zeros(
                (self.num_fg, D_orig, H_orig, W_orig), dtype=np.float32)
            full_prob[:, d0:d1, h0:h1, w0:w1] = prob_volume
            prob_volume = full_prob

        label_map = _blending.prob_to_label(
            prob_volume,
            label_values=self.label_values,
            num_fg=self.num_fg,
            threshold=self.threshold)
        result = {"label_map": label_map, "probabilities": prob_volume}

        if output_dir:
            self._save_predictions(image_path, label_map, prob_volume,
                                   output_dir)
        return result

    @torch.no_grad()
    def predict_preprocessed_array(
        self,
        vol: np.ndarray,
        z_spacing: Optional[float] = None,
    ) -> np.ndarray:
        """对**已预处理**（强度窗 + 归一化）的整卷数组做 patch_mode 滑窗推理。

        纯 mode 派发核心，从 ``predict_volume`` 抽出以便复用（如训练时的整卷
        验证 / 选模），不涉及 NIfTI I/O、bbox、保存与诊断重置。输入 ``vol`` 为
        ``(D, H, W)`` fp32；返回概率体 ``(num_fg, D, H, W)`` fp32。

        ``z_spacing`` 仅 2.5D z-交错需要；为 ``None`` 时即便 ``z_interleave_enabled``
        也回退到标准 z 轴滑窗（npz 缓存无物理 spacing 的场景）。
        """
        with self._forward_grad_ctx():
            if self.patch_mode == "whole":
                return _sliding.whole_volume_forward(self, vol)
            if self.patch_mode == "cubic":
                return _sliding.sliding_window_cubic(self, vol)
            if self.z_interleave_enabled and z_spacing is not None:
                # 2.5D z-交错；k≤1 时会退化为标准 sliding_window_z。
                return _sliding.sliding_window_z_interleaved(
                    self, vol, float(z_spacing))
            # z_axis / 2_5d 几何同，forward 侧区别。
            return _sliding.sliding_window_z(self, vol)

    def _forward_grad_ctx(self):
        """推理前向的免梯度上下文：``use_inference_mode=True`` 时用
        ``torch.inference_mode()``（在外层 ``no_grad`` 基础上再免除 autograd
        簿记，数值等价）；否则为空上下文（沿用装饰器的 ``no_grad``，行为不变）。

        AdaBN per_volume 估计期（``self._adabn_estimating``）固定回退空上下文：
        该阶段 BN 处于 train 模式、需原地更新 running-stats buffer，避开
        inference_mode 对原地更新的限制。
        """
        if self.use_inference_mode and not self._adabn_estimating:
            return torch.inference_mode()
        return contextlib.nullcontext()

    def _adabn_keep_window(self, idx: int) -> bool:
        """AdaBN 估计期窗口抽样判据：非估计期或 ``adabn_sample_ratio>=1``
        恒 True（真实预测路径不受影响）；否则按 ``round(1/ratio)`` 步长
        确定性保留（``idx==0`` 恒留，保证至少一窗驱动 BN 更新）。"""
        if not self._adabn_estimating or self.adabn_sample_ratio >= 1.0:
            return True
        step = max(1, int(round(1.0 / self.adabn_sample_ratio)))
        return idx % step == 0

    # ==================================================================
    # NIfTI I/O
    # ==================================================================
    def _save_predictions(
        self,
        image_path: str,
        label_map: np.ndarray,
        prob_volume: np.ndarray,
        output_dir: str,
    ) -> None:
        """以 SimpleITK 写 NIfTI：origin/spacing/direction 从源图复制；
        输入数组顺序 (D,H,W) == (Z,Y,X)，无需转置；输出 gzip 压缩。。"""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).name.replace(".nii.gz", "").replace(".nii", "")

        # ReadImage 读完整图获元数据（像素解码成本远小于推理）。
        ref_img = sitk.ReadImage(str(image_path))

        lbl_img = sitk.GetImageFromArray(label_map)
        lbl_img.CopyInformation(ref_img)
        lbl_path = out_dir / f"{stem}_pred.nii.gz"
        sitk.WriteImage(lbl_img, str(lbl_path), useCompression=True)
        logger.info("Saved label map: %s", lbl_path)

        if self.save_probs:
            for c in range(prob_volume.shape[0]):
                prob_arr = prob_volume[c].astype(np.float32, copy=False)
                prob_img = sitk.GetImageFromArray(prob_arr)
                prob_img.CopyInformation(ref_img)
                prob_path = out_dir / f"{stem}_prob_class{c}.nii.gz"
                sitk.WriteImage(prob_img, str(prob_path), useCompression=True)
            logger.info("Saved probability maps: %d classes",
                        prob_volume.shape[0])


# ==================================================================
# Checkpoint loading + run_inference moved to ``predictor.io`` (R6).
# Re-exported via the package ``__init__`` for backward compatibility.
# ==================================================================