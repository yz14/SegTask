"""3D 分割滑窗推理。

推理模式与训练 patch 模式一致：z_axis / cubic / whole / 2_5d。
支持 overlap 融合（gaussian/uniform）、flip TTA、AMP、多分辨率输入、
ckpt 加载（兼容 torch.compile / EMA / best-model EMA-primary）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import torch

from ..config import Config
from ..data.dataset import (
    load_nifti, load_nifti_with_spacing, preprocess_image,
    compute_bbox_from_volume)
from ..models.topology import ModelTopology, build_topology
from . import blending as _blending
from . import forwards as _forwards
from . import inputs as _inputs
from . import sliding as _sliding

logger = logging.getLogger(__name__)


_AMP_DTYPES = {
    "float16": torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}


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
        self.overlap = pc.z_overlap            # cubic 下三轴复用
        self.blend_mode = pc.blend_mode
        self.batch_size = pc.batch_size
        self.tta_flip = pc.tta_flip
        # TTA flip 变体批量化块大小（None → 退化为 batch_size）；见 forwards._tta_chunk_size。
        self.tta_batch_size: Optional[int] = getattr(pc, "tta_batch_size", None)
        self.threshold = pc.threshold
        self.save_probs = pc.save_probabilities
        # 逐卷滑窗进度日志开关（运行期内部量，不暴露到配置）：CLI 推理默认 True；
        # 训练内整卷验证（VolumeValEvaluator）会置 False 以免每 epoch 刷屏 81 卷。
        self.log_progress = True

        # 2.5D z 交错推理：按 stride k 将体积拆为 k 个子流，各自走 _sliding_window_z，
        # 后以 out[:, i::k]=stream_i 缝回。k 由 z spacing 挑选（见 _choose_interleave_factor）。
        # 仅 patch_mode=='2_5d' 生效；Config 验证。
        self.z_interleave_enabled = bool(
            getattr(pc, "z_interleave_enabled", False)
            and cfg.data.patch_mode == "2_5d")
        self.z_interleave_thresholds: List[float] = list(
            getattr(pc, "z_interleave_thresholds", [1.0, 1.5]))
        self.z_interleave_factors: List[int] = [
            int(f) for f in getattr(pc, "z_interleave_factors", [3, 2, 1])]
        if self.z_interleave_enabled:
            logger.info(
                "Predictor z_interleave_enabled=True (2.5D only): "
                "thresholds=%s mm, factors=%s",
                self.z_interleave_thresholds, self.z_interleave_factors)

        # 测试时自适应 BatchNorm — per_volume 模式：每卷推理前用该卷自身重估 BN，
        # 再冻结预测（transductive BN）。global 模式在 run_inference 中处理，与此无关。
        self.adabn_enabled = bool(getattr(pc, "adabn_enabled", False))
        self.adabn_mode = getattr(pc, "adabn_mode", "global")
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

        self.patch_mode = cfg.data.patch_mode
        self.patch_D, self.patch_H, self.patch_W = cfg.data.patch_size
        self.label_values = cfg.data.label_values
        self.num_fg = cfg.num_fg_classes
        # 默认单分辨率，避免下游 np.stack 报错。
        self.multi_res_scales = cfg.data.multi_res_scales or [1.0]
        # 与 DataConfig.z_boundary_mode 同步，使训/用几何一致；老配置默认 stretch。
        self.z_boundary_mode = getattr(
            cfg.data, "z_boundary_mode", "stretch")
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
            actual_in = int(getattr(cfg.model, "in_channels", expect_in))
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

        # AMP：与训练同 dtype。未知值退 bf16 避免静默切换。
        amp_name = getattr(cfg.train, "amp_dtype", "bfloat16")
        if amp_name not in _AMP_DTYPES:
            logger.warning("Unknown amp_dtype=%r, falling back to bfloat16.",
                           amp_name)
            amp_name = "bfloat16"
        self.amp_dtype = _AMP_DTYPES[amp_name]
        self.use_amp = (
            getattr(cfg.train, "use_amp", True) and device.type == "cuda")

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

        # 逐卷“首 batch 已记录”护孔。predict_volume 顶部重置，_forward_batch* 消费。
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

        vol = preprocess_image(
            raw_vol, dc.intensity_min, dc.intensity_max,
            dc.normalize, dc.global_mean, dc.global_std)

        # 诊断：归一化后输入统计（与训练不一致时 range/分位数会明显偏差）。
        try:
            _v = vol
            _vmin = float(_v.min()); _vmax = float(_v.max())
            _vmean = float(_v.mean()); _vstd = float(_v.std())
            _q = np.quantile(_v, [0.01, 0.5, 0.99])
            logger.info(
                "[diag] normalized input: shape=%s, min=%.4f, max=%.4f, "
                "mean=%.4f, std=%.4f, q1=%.4f, q50=%.4f, q99=%.4f "
                "(normalize=%s, intensity=[%.1f,%.1f])",
                tuple(_v.shape), _vmin, _vmax, _vmean, _vstd,
                float(_q[0]), float(_q[1]), float(_q[2]),
                dc.normalize, float(dc.intensity_min), float(dc.intensity_max))
        except Exception as _e:
            logger.warning("[diag] normalized-input stat failed: %s", _e)
        # AdaBN per_volume：用该卷自身先跑一遍前向重估 BN running stats，再冻结预测。
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

        # 诊断：blend 后概率统计（拼回原画布前计算，避免 ROI 外 0 偏移统计）。
        # frac_gt_thr ≈1.0 提示是模型输出本身饱和（训练侧问题），非后处理。
        try:
            _pv = prob_volume  # (num_fg, D, H, W) inside-ROI only
            _max_per_vox = _pv.max(axis=0)
            _frac_gt_thr = float((_max_per_vox >= self.threshold).mean())
            _q = np.quantile(_pv, [0.5, 0.9, 0.99, 0.999])
            logger.info(
                "[diag] in-ROI prob volume: shape=%s, min=%.4f, max=%.4f, "
                "mean=%.4f, q50=%.4f, q90=%.4f, q99=%.4f, q999=%.4f, "
                "frac(max_prob>=%.2f)=%.4f",
                tuple(_pv.shape), float(_pv.min()), float(_pv.max()),
                float(_pv.mean()), float(_q[0]), float(_q[1]),
                float(_q[2]), float(_q[3]), self.threshold, _frac_gt_thr)
            if _frac_gt_thr > 0.95:
                logger.warning(
                    "[diag] %.1f%% of in-ROI voxels exceed threshold — "
                    "the model itself is outputting near-saturated "
                    "foreground; this is a TRAINING-side issue (most "
                    "likely: training bbox/label semantics differ from "
                    "this run, OR region weights drove the model to a "
                    "trivial 'all-fg' minimum). Re-check cfg.data.bbox_dir, "
                    "label_dir and region_weight_dir USED ON THE SERVER.",
                    100.0 * _frac_gt_thr)
        except Exception as _e:
            logger.warning("[diag] prob-volume stat failed: %s", _e)

        # 拼回原尺寸画布以保 NIfTI affine 一致；bbox 外体素保留 0 概率。
        if bbox is not None:
            (d0, d1), (h0, h1), (w0, w1) = bbox
            full_prob = np.zeros(
                (self.num_fg, D_orig, H_orig, W_orig), dtype=np.float32)
            full_prob[:, d0:d1, h0:h1, w0:w1] = prob_volume
            prob_volume = full_prob

        label_map = self._prob_to_label(prob_volume)
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
        if self.patch_mode == "whole":
            return self._whole_volume_forward(vol)
        if self.patch_mode == "cubic":
            return self._sliding_window_cubic(vol)
        if self.z_interleave_enabled and z_spacing is not None:
            # 2.5D z-交错；k≤1 时会退化为标准 _sliding_window_z。
            return self._sliding_window_z_interleaved(vol, float(z_spacing))
        # z_axis / 2_5d 几何同，forward 侧区别。
        return self._sliding_window_z(vol)

    # ==================================================================
    # Sliding loops (R6: thin shims delegating to predictor.sliding)
    # ==================================================================
    def _choose_interleave_factor(self, z_spacing: float) -> int:
        """z-interleave factor 选择；R6 委托至 ``predictor.sliding.choose_interleave_factor``。"""
        return _sliding.choose_interleave_factor(self, z_spacing)

    def _sliding_window_z_interleaved(
        self, vol: np.ndarray, z_spacing: float,
    ) -> np.ndarray:
        """2.5D z-交错推理；R6 委托至 ``predictor.sliding.sliding_window_z_interleaved``。"""
        return _sliding.sliding_window_z_interleaved(self, vol, z_spacing)

    def _sliding_window_z(self, vol: np.ndarray) -> np.ndarray:
        """z 轴滑窗推理；R6 委托至 ``predictor.sliding.sliding_window_z``。"""
        return _sliding.sliding_window_z(self, vol)

    # ==================================================================
    # Z-axis window builders (R6: thin shims delegating to predictor.inputs)
    # ==================================================================
    def _build_z_window_input_gpu(
        self, vol_t: torch.Tensor, z0: int, z1: int) -> torch.Tensor:
        """单分辨率 GPU 窗口建造；R6 委托至 ``predictor.inputs.build_z_window_single_res_gpu``。"""
        return _inputs.build_z_window_single_res_gpu(
            vol_t, z0, z1,
            pD=self.patch_D, pH=self.patch_H, pW=self.patch_W,
            z_boundary_mode=self.z_boundary_mode)

    def _build_z_window_input_native_multi_res_gpu(
        self, vol_t: torch.Tensor, z0: int, z1: int) -> torch.Tensor:
        """3D ``z_axis`` ON 窗口建造；R6 委托至 ``predictor.inputs.build_z_window_native_multi_res_gpu``。"""
        return _inputs.build_z_window_native_multi_res_gpu(
            vol_t, z0, z1,
            pD=self.patch_D, pH=self.patch_H, pW=self.patch_W,
            target_shape=self._mr_target_shape,
            native_sizes=self._mr_native_sizes)

    def _build_z_window_input_native_d_gpu(
        self, vol_t: torch.Tensor, z0: int, z1: int) -> torch.Tensor:
        """2.5D ON 窗口建造；R6 委托至 ``predictor.inputs.build_z_window_native_d_gpu``。"""
        return _inputs.build_z_window_native_d_gpu(
            vol_t, z0, z1,
            pH=self.patch_H, pW=self.patch_W,
            eD_max=self._eD_max, view_depths=self.per_view_depths)

    # ==================================================================
    # Forward + TTA + diag (R6: thin shims delegating to predictor.forwards)
    # ==================================================================
    @torch.no_grad()
    def _forward_batch_gpu(self, x: torch.Tensor) -> torch.Tensor:
        """Mode-aware GPU forward；R6 委托至 ``predictor.forwards.forward_batch_gpu``。"""
        return _forwards.forward_batch_gpu(self, x)

    @torch.no_grad()
    def _diag_log_first_batch(
        self, tag: str,
        x: torch.Tensor,
        logits: torch.Tensor,
        prob: torch.Tensor,
    ) -> None:
        """首 batch 诊断；R6 委托至 ``predictor.forwards.diag_log_first_batch``。"""
        _forwards.diag_log_first_batch(self, tag, x, logits, prob)

    def _build_z_window_input(
        self, vol: np.ndarray, z0: int, z1: int) -> np.ndarray:
        """多分辨率 z 窗口堆（CPU 退化路径）；R6 委托至 ``predictor.inputs.build_z_window_cpu_multi_res``。"""
        return _inputs.build_z_window_cpu_multi_res(
            vol, z0, z1,
            pD=self.patch_D, pH=self.patch_H, pW=self.patch_W,
            multi_res_scales=self.multi_res_scales,
            z_boundary_mode=self.z_boundary_mode)

    def _whole_volume_forward(self, vol: np.ndarray) -> np.ndarray:
        """全卷 resize 单次 forward；R6 委托至 ``predictor.sliding.whole_volume_forward``。"""
        return _sliding.whole_volume_forward(self, vol)

    def _sliding_window_cubic(self, vol: np.ndarray) -> np.ndarray:
        """3 轴 cubic 滑窗推理；R6 委托至 ``predictor.sliding.sliding_window_cubic``。"""
        return _sliding.sliding_window_cubic(self, vol)

    # ==================================================================
    # Cubic batch builders (R6: thin shims delegating to predictor.inputs)
    # ==================================================================
    def _build_batch_native_multi_res_cubic_gpu(
        self,
        centers: List[Tuple[int, int, int]],
        vol_t: torch.Tensor,
    ) -> torch.Tensor:
        """3D ``cubic`` ON 批建造；R6 委托至 ``predictor.inputs.build_cubic_batch_native_multi_res``。"""
        return _inputs.build_cubic_batch_native_multi_res(
            centers, vol_t,
            pD=self.patch_D, pH=self.patch_H, pW=self.patch_W,
            target_shape=self._mr_target_shape,
            native_sizes=self._mr_native_sizes)

    def _build_batch_multi_res(
        self,
        patches: List[np.ndarray],
        centers: List[Tuple[int, int, int]],
        vol: np.ndarray,
    ) -> torch.Tensor:
        """``cubic`` CPU 多分辨率批；R6 委托至 ``predictor.inputs.build_cubic_batch_cpu_multi_res``。"""
        return _inputs.build_cubic_batch_cpu_multi_res(
            patches, centers, vol,
            pD=self.patch_D, pH=self.patch_H, pW=self.patch_W,
            multi_res_scales=self.multi_res_scales,
            device=self.device)

    # ==================================================================
    # Forward + TTA (R6: thin shims delegating to predictor.forwards)
    # ==================================================================
    def _forward_batch(self, x: torch.Tensor) -> np.ndarray:
        """numpy-返版 forward；R6 委托至 ``predictor.forwards.forward_batch_numpy``。"""
        return _forwards.forward_batch_numpy(self, x)

    def _forward_batch_2_5d(self, x: torch.Tensor) -> np.ndarray:
        """2.5D forward (numpy 返)；R6 委托至 ``predictor.forwards.forward_batch_2_5d_numpy``。"""
        return _forwards.forward_batch_2_5d_numpy(self, x)

    def _reshape_2_5d_input(self, x: torch.Tensor) -> torch.Tensor:
        """2.5D rank-5 → rank-4；R6 委托至 ``predictor.forwards.reshape_2_5d_input``。"""
        return _forwards.reshape_2_5d_input(self, x)

    def _tta_flip_ensemble(
        self, x: torch.Tensor, base_prob: torch.Tensor,
    ) -> torch.Tensor:
        """3D TTA；R6 委托至 ``predictor.forwards.tta_flip_ensemble``。"""
        return _forwards.tta_flip_ensemble(self, x, base_prob)

    def _tta_flip_ensemble_2_5d(
        self, x_2d: torch.Tensor, base_prob: torch.Tensor,
    ) -> torch.Tensor:
        """2.5D TTA；R6 委托至 ``predictor.forwards.tta_flip_ensemble_2_5d``。"""
        return _forwards.tta_flip_ensemble_2_5d(self, x_2d, base_prob)

    # ==================================================================
    # Geometry helpers (R6: thin shims delegating to predictor.blending)
    # ==================================================================
    @staticmethod
    def _compute_1d_positions(
        length: int, patch: int, stride: int,
    ) -> List[Tuple[int, int]]:
        """逐轴返 ``(start, end)`` 窗口；R6 委托至 ``predictor.blending.compute_1d_positions``。"""
        return _blending.compute_1d_positions(length, patch, stride)

    @staticmethod
    def _build_1d_weight(n: int, mode: str = "gaussian") -> np.ndarray:
        """对称 1D blending 窗；R6 委托至 ``predictor.blending.build_1d_weight``。"""
        return _blending.build_1d_weight(n, mode)

    @staticmethod
    def _build_3d_weight(pD: int, pH: int, pW: int, mode: str) -> np.ndarray:
        """可分离 3D blending 权重；R6 委托至 ``predictor.blending.build_3d_weight``。"""
        return _blending.build_3d_weight(pD, pH, pW, mode)

    # ==================================================================
    # Probability → label map (R6: thin shim delegating to predictor.blending)
    # ==================================================================
    def _prob_to_label(self, prob_volume: np.ndarray) -> np.ndarray:
        """概率体 → 整数 label map；R6 委托至 ``predictor.blending.prob_to_label``。"""
        return _blending.prob_to_label(
            prob_volume,
            label_values=self.label_values,
            num_fg=self.num_fg,
            threshold=self.threshold)

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