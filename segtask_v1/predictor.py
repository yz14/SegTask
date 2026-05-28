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
import torch.nn.functional as F
from einops import rearrange
from torch.amp import autocast

from .config import Config
from .data.dataset import (
    load_nifti, load_nifti_with_spacing, preprocess_image, resize_3d,
    _extract_cubic_patch, extract_z_patch_padded, compute_bbox_from_volume)

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
        self.threshold = pc.threshold
        self.save_probs = pc.save_probabilities

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

        # Plan A lift：2.5D pipeline 走真 3D 模型。启用后不折 C_res*D，rank-5 输入直送
        # 三维 UNet，输出 (B, num_fg, pD, pH, pW)。Config 仅允许 patch_mode=='2_5d'，与
        # aux_keep_native_d 互斥。
        self.lift_2_5d_to_3d = bool(
            getattr(cfg.model, "lift_2_5d_to_3d", False)
            and self.patch_mode == "2_5d")
        if self.lift_2_5d_to_3d:
            logger.info(
                "Predictor lift_2_5d_to_3d=True: 2.5D windows fed straight "
                "to a true-3D UNet (n_views=%d, in_channels=%d, output "
                "shape (B, num_fg=%d, pD=%d, pH=%d, pW=%d)).",
                len(self.multi_res_scales),
                int(getattr(cfg.model, "in_channels", -1)),
                int(self.num_fg), int(self.patch_D),
                int(self.patch_H), int(self.patch_W))

        # 2.5D 原生深度多 FOV 推理路径：与 Trainer.aux_keep_native_d 同步，输入布局
        # (B, sum_k D_k, pH, pW)，D_k=round(pD*s_k)。
        self.aux_keep_native_d = bool(
            getattr(cfg.data, "aux_keep_native_d", False)
            and self.patch_mode == "2_5d"
            and len(self.multi_res_scales) > 1
            and not self.lift_2_5d_to_3d)  # 防御；validate 也会拒
        if self.aux_keep_native_d:
            depths = [int(round(self.patch_D * float(s)))
                      for s in self.multi_res_scales]
            depths[0] = int(self.patch_D)  # s_0 == 1.0 不变式
            self.aux_view_depths: List[int] = depths
            self._eD_max: int = int(round(
                self.patch_D * float(max(self.multi_res_scales))))
            # 与模型实际 in_channels 一致性检查。
            expect_in = sum(depths)
            actual_in = int(getattr(cfg.model, "in_channels", expect_in))
            if actual_in != expect_in:
                raise ValueError(
                    f"aux_keep_native_d=True: model.in_channels={actual_in} "
                    f"!= sum(aux_view_depths)={expect_in}. The model was "
                    "likely built with a stale Config — re-sync and rebuild.")
            logger.info(
                "Predictor aux_keep_native_d=True: per-view depths=%s, "
                "max-FOV cube depth=%d, in_channels=%d.",
                depths, self._eD_max, actual_in)
        else:
            self.aux_view_depths = []
            self._eD_max = int(self.patch_D)

        # 3D 懒 max-FOV cube 推理路径：与 Trainer.keep_native_multi_res 同步，输入布局
        # (B, C_res, pD, pH, pW)；在 GPU 上以一次 max-FOV 抽取 + 逐视图裁代替 K 次 CPU zoom。
        # 仅 3D 模式启用；2.5D 走 aux_keep_native_d。
        self.keep_native_multi_res = bool(
            getattr(cfg.data, "keep_native_multi_res", False)
            and self.patch_mode in ("z_axis", "cubic")
            and len(self.multi_res_scales) > 1)
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
                self.patch_mode, sizes, self._mr_target_shape,
                len(self.multi_res_scales))
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
        # 重置诊断护孔使 forward 路径发一次 logits/prob 统计块。
        self._diag_first_batch_logged = False

        if self.patch_mode == "whole":
            prob_volume = self._whole_volume_forward(vol)
        elif self.patch_mode == "cubic":
            prob_volume = self._sliding_window_cubic(vol)
        elif self.z_interleave_enabled:
            # 2.5D z-交错；k≤1 时会退化为标准 _sliding_window_z。
            prob_volume = self._sliding_window_z_interleaved(
                vol, float(z_spacing))
        else:
            # z_axis / 2_5d 几何同，forward 侧区别。
            prob_volume = self._sliding_window_z(vol)

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

    # ==================================================================
    # Z-axis sliding window — interleaved (TODO 1) wrapper
    # ==================================================================
    def _choose_interleave_factor(self, z_spacing: float) -> int:
        """根据物理 z spacing(mm) 选 k：首个 thresholds[j]>=z_spacing 返 factors[j]，否则 fallback。。"""
        thresholds = self.z_interleave_thresholds
        factors = self.z_interleave_factors
        for t, f in zip(thresholds, factors):
            if z_spacing <= float(t):
                return max(1, int(f))
        return max(1, int(factors[-1]))

    def _sliding_window_z_interleaved(
        self, vol: np.ndarray, z_spacing: float,
    ) -> np.ndarray:
        """2.5D z-交错推理：按 stride-k 拆为 k 个互斥子体独立推理后以 out[:, i::k]=stream_i 缝回。
        k≤1 时退化为标准 _sliding_window_z；覆盖全划分，缝接无需跨流加权。。"""
        k = self._choose_interleave_factor(z_spacing)
        if k <= 1:
            logger.info(
                "z-interleave: z_spacing=%.4f mm → k=1 (no split); "
                "falling through to standard 2.5D z-sliding window.",
                z_spacing)
            return self._sliding_window_z(vol)

        D, H, W = vol.shape
        logger.info(
            "z-interleave: z_spacing=%.4f mm → k=%d. Splitting volume "
            "(D=%d) into %d disjoint stride-%d sub-streams; per-stream "
            "depths=%s.",
            z_spacing, k, D, k, k,
            [int(np.ceil((D - i) / k)) for i in range(k)])

        out = np.zeros((self.num_fg, D, H, W), dtype=np.float32)
        for i in range(k):
            # vol[i::k] 为 view；copy 以免下游 in-place 错误。
            sub_vol = np.ascontiguousarray(vol[i::k])
            sub_D = sub_vol.shape[0]
            logger.info(
                "  z-interleave stream %d/%d: indices=%d::%d, sub_D=%d",
                i + 1, k, i, k, sub_D)
            sub_prob = self._sliding_window_z(sub_vol)
            # 防御：_sliding_window_z 须保证输出深度 == 输入深度。
            if sub_prob.shape != (self.num_fg, sub_D, H, W):
                raise RuntimeError(
                    f"z-interleave stream {i}: expected sub-prob shape "
                    f"({self.num_fg}, {sub_D}, {H}, {W}), got "
                    f"{tuple(sub_prob.shape)}")
            out[:, i::k, :, :] = sub_prob
        return out

    # ==================================================================
    # Z-axis sliding window
    # ==================================================================
    def _sliding_window_z(self, vol: np.ndarray) -> np.ndarray:
        """z 轴滑窗推理。H/W 总 resize 到模型输入尺寸；z 轴按 stride 滑动并 blend。
        多分辨率：逐 scale 抽 round(pD*s) 切片 resize 回后堆属 (B,C_res,pD,pH,pW)（与 SegDataset3D 一致）。。"""
        D_orig, H_orig, W_orig = vol.shape
        pD, pH, pW = self.patch_D, self.patch_H, self.patch_W

        stride = max(1, int(pD * (1 - self.overlap)))
        z_positions = self._compute_1d_positions(D_orig, pD, stride)

        logger.info(
            "Z-axis sliding window: D_patch=%d, stride=%d, num_windows=%d, "
            "scales=%s, blend=%s",
            pD, stride, len(z_positions), self.multi_res_scales, self.blend_mode)

        # GPU 常驻：体积 + 累加器都在 GPU，F.interpolate 替代 scipy.ndimage.zoom；
        # 仅最后 blend 后的概率体返 host。
        vol_t = torch.from_numpy(vol).to(self.device, non_blocking=True)

        z_weight_t = torch.from_numpy(
            self._build_1d_weight(pD)).to(self.device)  # (pD,)

        acc_pred = torch.zeros(
            (self.num_fg, D_orig, H_orig, W_orig),
            dtype=torch.float32, device=self.device)
        acc_weight = torch.zeros(
            (1, D_orig, 1, 1), dtype=torch.float32, device=self.device)

        # 多分辨率 z 轴实践上少见（2.5D 强制 [1.0]）；单分辨率走 GPU 抽取路径。
        single_res = (len(self.multi_res_scales) == 1
                      and self.multi_res_scales[0] == 1.0)
        # 3D ON 路径：GPU builder 返 (C_res,pD,pH,pW)，与旧多分辨率 z_axis 布局一致。
        keep_native_3d = bool(self.keep_native_multi_res
                              and self.patch_mode == "z_axis")

        # 2.5D ON 路径：builder 返 rank-3 (sum(D_k),pH,pW)，stack 后直接到模型入参，无需 reshape。
        window_inputs: List[torch.Tensor] = []
        patch_metas: List[Tuple[int, int, int]] = []  # (z0, z1, actual_d)

        n_windows = len(z_positions)
        for idx, (z0, z1) in enumerate(z_positions):
            actual_d = z1 - z0
            if self.aux_keep_native_d:
                # 2.5D ON: rank-3 (sum(D_k), pH, pW)。
                window_inputs.append(
                    self._build_z_window_input_native_d_gpu(vol_t, z0, z1))
            elif keep_native_3d:
                # 3D ON: rank-4 (C_res, pD, pH, pW)。
                window_inputs.append(
                    self._build_z_window_input_native_multi_res_gpu(
                        vol_t, z0, z1))
            elif single_res:
                window_inputs.append(
                    self._build_z_window_input_gpu(vol_t, z0, z1))
            else:
                # 多分辨率退化：CPU builder 后一次上 GPU。
                wi_np = self._build_z_window_input(vol, z0, z1)
                window_inputs.append(
                    torch.from_numpy(wi_np).to(
                        self.device, non_blocking=True))
            patch_metas.append((z0, z1, actual_d))

            is_last = idx == n_windows - 1
            if len(window_inputs) >= self.batch_size or is_last:
                # (B, C_res, pD, pH, pW)
                batch = torch.stack(window_inputs, dim=0).float()
                # (B, num_fg, pD, pH, pW) on GPU
                probs = self._forward_batch_gpu(batch)

                # 按 actual_d 分组 → 合并为一次 F.interpolate。常见场景 ad==pD，仅一次上采样。
                groups: Dict[int, List[int]] = {}
                for i, (_, _, ad) in enumerate(patch_metas):
                    groups.setdefault(ad, []).append(i)

                for ad, idxs in groups.items():
                    sub = probs[idxs]  # (b, num_fg, pD, pH, pW)

                    # 倒 resize 回原几何：edge_pad+ad<pD 时仅取中心 ad 切片不插值 z（H/W 可 resize）；
                    # 其余走一次性 trilinear resize 到 (ad, H_orig, W_orig)。
                    if self.z_boundary_mode == "edge_pad" and ad < pD:
                        pad_before = (pD - ad) // 2
                        sub = sub[:, :, pad_before:pad_before + ad, :, :]
                        if (H_orig != pH) or (W_orig != pW):
                            sub = F.interpolate(
                                sub, size=(ad, H_orig, W_orig),
                                mode="trilinear", align_corners=False)
                    elif (ad != pD) or (H_orig != pH) or (W_orig != pW):
                        sub = F.interpolate(
                            sub, size=(ad, H_orig, W_orig),
                            mode="trilinear", align_corners=False)
                    # 逐 ad 对称 blending 权重。
                    if ad == pD:
                        w = z_weight_t
                    else:
                        w = torch.from_numpy(
                            self._build_1d_weight(ad)).to(self.device)
                    w_4d = rearrange(w, 'c -> 1 c 1 1')

                    for j, i in enumerate(idxs):
                        zs, ze, _ = patch_metas[i]
                        # in-place fused mul-add。
                        acc_pred[:, zs:ze, :, :].addcmul_(
                            sub[j], w_4d, value=1.0)
                        acc_weight[:, zs:ze, :, :].add_(w_4d)

                window_inputs.clear()
                patch_metas.clear()

                if (idx + 1) % max(1, 10 * self.batch_size) == 0 or is_last:
                    logger.info("  z-window %d/%d", idx + 1, n_windows)

        acc_weight.clamp_(min=1e-8)
        return (acc_pred / acc_weight).cpu().numpy()

    def _build_z_window_input_gpu(
        self, vol_t: torch.Tensor, z0: int, z1: int) -> torch.Tensor:
        """单分辨率 GPU 版 _build_z_window_input。z_boundary_mode：
        stretch 取 vol[z0:z1] 后三线性 resize 到 (pD,pH,pW)；
        edge_pad 在 ad<pD 时对称复制填充到 pD 后再 resize（与训练 multi-res 一致）。返 (1,pD,pH,pW)。。"""
        pD, pH, pW = self.patch_D, self.patch_H, self.patch_W
        patch = vol_t[z0:z1]  # (actual_d, H_orig, W_orig)
        ad, H, W = patch.shape

        if self.z_boundary_mode == "edge_pad" and ad < pD:
            # 对称复制填充（与 extract_z_patch_padded 对齐）。
            pad_before = (pD - ad) // 2
            pad_after = pD - ad - pad_before
            chunks = []
            if pad_before > 0:
                chunks.append(patch[0:1].expand(pad_before, -1, -1))
            chunks.append(patch)
            if pad_after > 0:
                chunks.append(patch[-1:].expand(pad_after, -1, -1))
            patch = torch.cat(chunks, dim=0)  # (pD, H, W)
            ad = pD  # depth resolved; only H/W remains for resize.

        # F.interpolate 需 (N,C,D,H,W)；补上 batch + channel。
        patch = patch.unsqueeze(0).unsqueeze(0).float()
        if (ad != pD) or (H != pH) or (W != pW):
            patch = F.interpolate(
                patch, size=(pD, pH, pW),
                mode="trilinear", align_corners=False)
        return patch.squeeze(0)  # (1, pD, pH, pW)

    def _build_z_window_input_native_multi_res_gpu(
        self, vol_t: torch.Tensor, z0: int, z1: int) -> torch.Tensor:
        """3D z_axis ON 模式窗口建造：抽单 max-FOV cube → 面内 resize 到 (pH,pW) →
        逐视图中心裁 D_k 后 D 轴 trilinear 回 pD → 拼 C_res。返 (C_res,pD,pH,pW)。与训练几何一致。。"""
        pD, pH, pW = self.patch_D, self.patch_H, self.patch_W
        eD_max, eH_max, eW_max = self._mr_target_shape   # eH_max=pH, eW_max=pW
        z_center = (z0 + z1) // 2
        D_vol = vol_t.shape[0]

        # 抽 eD_max 深的 slab，越界双侧复制边界。
        zlo = z_center - eD_max // 2
        zhi = zlo + eD_max
        zlo_in = max(zlo, 0)
        zhi_in = min(zhi, D_vol)
        slab = vol_t[zlo_in:zhi_in]  # (ad, H_orig, W_orig)
        pad_before = max(0, -zlo)
        pad_after = max(0, zhi - D_vol)
        if pad_before > 0 or pad_after > 0:
            chunks: List[torch.Tensor] = []
            if pad_before > 0:
                chunks.append(slab[0:1].expand(pad_before, -1, -1))
            chunks.append(slab)
            if pad_after > 0:
                chunks.append(slab[-1:].expand(pad_after, -1, -1))
            slab = torch.cat(chunks, dim=0)
        if slab.shape[0] != eD_max:
            raise RuntimeError(
                f"native-multi-res z builder: expected slab depth "
                f"{eD_max}, got {slab.shape[0]} (z0={z0}, z1={z1}, "
                f"D_vol={D_vol}).")

        # 面内 resize（D 轴保持 eD_max）。
        H_orig, W_orig = slab.shape[1], slab.shape[2]
        slab = slab.unsqueeze(0).unsqueeze(0).float()  # (1,1,eD_max,H,W)
        if H_orig != pH or W_orig != pW:
            slab = F.interpolate(
                slab, size=(eD_max, pH, pW),
                mode="trilinear", align_corners=False)

        # 逐视图裁 + D 轴 resize。cat 代替 stack：按通道轴连接为 C_res。
        view_chunks: List[torch.Tensor] = []
        for D_k, _, _ in self._mr_native_sizes:
            d0 = (eD_max - D_k) // 2
            crop = slab[:, :, d0:d0 + D_k, :, :]  # (1, 1, D_k, pH, pW)
            if D_k != pD:
                crop = F.interpolate(
                    crop, size=(pD, pH, pW),
                    mode="trilinear", align_corners=False)
            view_chunks.append(crop[0])  # (1, pD, pH, pW)
        return torch.cat(view_chunks, dim=0).contiguous()  # (C_res, pD, pH, pW)

    def _build_z_window_input_native_d_gpu(
        self, vol_t: torch.Tensor, z0: int, z1: int) -> torch.Tensor:
        """2.5D ON 模式（aux_keep_native_d=True）窗口建造：抽 eD_max max-FOV cube、
        面内 resize 到 (pH,pW)、逐视图中心抽 D_k 切片后拼通道 → (sum(D_k),pH,pW)。。"""
        pD, pH, pW = self.patch_D, self.patch_H, self.patch_W
        eD_max = self._eD_max
        z_center = (z0 + z1) // 2
        D_vol = vol_t.shape[0]

        # ---- Edge-padded extraction of an eD_max-deep slab ---------------
        zlo = z_center - eD_max // 2
        zhi = zlo + eD_max
        # In-bounds slab; edge-replicate above / below if needed.
        zlo_in = max(zlo, 0)
        zhi_in = min(zhi, D_vol)
        slab = vol_t[zlo_in:zhi_in]  # (ad, H_orig, W_orig)
        pad_before = max(0, -zlo)
        pad_after = max(0, zhi - D_vol)
        if pad_before > 0 or pad_after > 0:
            chunks: List[torch.Tensor] = []
            if pad_before > 0:
                chunks.append(slab[0:1].expand(pad_before, -1, -1))
            chunks.append(slab)
            if pad_after > 0:
                chunks.append(slab[-1:].expand(pad_after, -1, -1))
            slab = torch.cat(chunks, dim=0)
        if slab.shape[0] != eD_max:
            raise RuntimeError(
                f"native-d builder: expected slab depth {eD_max}, got "
                f"{slab.shape[0]} (z0={z0}, z1={z1}, eD_max={eD_max}, "
                f"D_vol={D_vol}). This indicates a window-position "
                "computation error.")

        # ---- In-plane resize (D axis preserved at eD_max) ----------------
        H_orig, W_orig = slab.shape[1], slab.shape[2]
        slab = slab.unsqueeze(0).unsqueeze(0).float()  # (1,1,eD_max,H,W)
        if H_orig != pH or W_orig != pW:
            slab = F.interpolate(
                slab, size=(eD_max, pH, pW),
                mode="trilinear", align_corners=False)
        slab = slab[0, 0]  # (eD_max, pH, pW)

        # ---- Per-view center crop + concat along channel axis ------------
        view_chunks: List[torch.Tensor] = []
        for D_k in self.aux_view_depths:
            d0 = (eD_max - D_k) // 2
            view_chunks.append(slab[d0:d0 + D_k])  # (D_k, pH, pW)
        return torch.cat(view_chunks, dim=0).contiguous()  # (sum(D_k), pH, pW)

    @torch.no_grad()
    def _forward_batch_gpu(self, x: torch.Tensor) -> torch.Tensor:
        """GPU 返回版 _forward_batch（推理中概率不过 host）。行为与 _forward_batch 一致。"""
        if self.patch_mode == "2_5d":
            # Plan A lift：rank-5 (B,n_views,pD,pH,pW) 直送三维 UNet，输出 (B,num_fg,pD,pH,pW)。
            # TTA 复用 3D ensemble（D 是真空间轴）。
            if self.lift_2_5d_to_3d:
                if x.ndim != 5:
                    raise ValueError(
                        "lift_2_5d_to_3d=True expects rank-5 input "
                        f"(B, n_views, D, H, W); got x.shape={tuple(x.shape)}")
                with autocast(device_type="cuda", enabled=self.use_amp,
                              dtype=self.amp_dtype):
                    pred = self.model(x.to(self.model_dtype))
                    if isinstance(pred, list):
                        pred = pred[0]
                    if pred.shape[1] < self.num_fg:
                        raise ValueError(
                            f"Lift-mode model output has {pred.shape[1]} "
                            f"channels at dim 1; expected at least "
                            f"num_fg={self.num_fg}.")
                    prob = torch.sigmoid(pred.float())[:, :self.num_fg]
                    self._diag_log_first_batch(
                        "2.5D lift", x, pred[:, :self.num_fg], prob)
                    if self.tta_flip:
                        prob = self._tta_flip_ensemble(x, prob)
                return prob

            # 两种输入：OFF rank-5 (B,C_res,pD,pH,pW) 需折 C_res*D；
            # ON rank-4 (B,sum(D_k),H,W) 已在入参布局，直接透传。
            if x.ndim == 5:
                x_2d = self._reshape_2_5d_input(x)  # (B, C_res*D, H, W)
            elif x.ndim == 4:
                x_2d = x
            else:
                raise ValueError(
                    f"2.5D forward expects rank-4 or rank-5 input; "
                    f"got x.shape={tuple(x.shape)}")
            D = self.patch_D
            B, _, H, W = x_2d.shape
            with autocast(device_type="cuda", enabled=self.use_amp,
                          dtype=self.amp_dtype):
                pred = self.model(x_2d.to(self.model_dtype))
                if isinstance(pred, list):
                    pred = pred[0]
                expected_c = self.num_fg * D
                if pred.shape[1] != expected_c:
                    raise ValueError(
                        f"2.5D model output channels {pred.shape[1]} != "
                        f"num_fg*D = {self.num_fg}*{D} = {expected_c}")
                pred_5d = rearrange(
                    pred, 'b (c d) h w -> b c d h w', c=self.num_fg, d=D)
                prob = torch.sigmoid(pred_5d.float())
                self._diag_log_first_batch(
                    "2.5D folded", x_2d, pred_5d, prob)
                if self.tta_flip:
                    prob = self._tta_flip_ensemble_2_5d(x_2d, prob)
            return prob

        with autocast(device_type="cuda", enabled=self.use_amp,
                      dtype=self.amp_dtype):
            pred = self.model(x.to(self.model_dtype))
            if isinstance(pred, list):
                pred = pred[0]
            assert pred.shape[1] >= self.num_fg, (
                f"Model output has {pred.shape[1]} channels; "
                f"expected at least num_fg={self.num_fg} at 1x resolution.")
            prob = torch.sigmoid(pred.float())[:, :self.num_fg]
            self._diag_log_first_batch(
                "3D", x, pred[:, :self.num_fg], prob)
            if self.tta_flip:
                prob = self._tta_flip_ensemble(x, prob)
        return prob

    @torch.no_grad()
    def _diag_log_first_batch(
        self, tag: str,
        x: torch.Tensor,
        logits: torch.Tensor,
        prob: torch.Tensor,
    ) -> None:
        """逐卷一次性诊断：记录输入 / logits / sigmoid 的 stats 与阈上比例，
        区分“模型本身饱和”（logits≫5, sigmoid≈1, frac≈1）与 “blend/后处理坍塌”。"""
        if self._diag_first_batch_logged:
            return
        self._diag_first_batch_logged = True

        def _q3(t: torch.Tensor, qs):
            """采样式分位数：torch.quantile 在 CUDA 上对 >~1.6e7 元素会拒；
            用整数 stride 切片避免 float32 linspace 越界。"""
            flat = t.detach().float().flatten()
            n = flat.numel()
            cap = 1_000_000
            if n > cap:
                stride = max(1, n // cap)
                flat = flat[::stride]
            qs_t = torch.tensor(qs, device=flat.device, dtype=flat.dtype)
            return torch.quantile(flat, qs_t).cpu().tolist()

        try:
            xs = x.detach().float()
            ls = logits.detach().float()
            ps = prob.detach().float()
            xq = _q3(xs, [0.01, 0.5, 0.99])
            lq = _q3(ls, [0.01, 0.5, 0.99])
            pq = _q3(ps, [0.5, 0.9, 0.99])
            n_nan_logits = int(torch.isnan(ls).sum().item())
            n_nan_prob = int(torch.isnan(ps).sum().item())
            frac_thr = float((ps >= self.threshold).float().mean().item())
            logger.info(
                "[diag/forward %s] input: shape=%s, min=%.4f, max=%.4f, "
                "mean=%.4f, q1=%.4f, q50=%.4f, q99=%.4f",
                tag, tuple(xs.shape),
                float(xs.min()), float(xs.max()), float(xs.mean()),
                float(xq[0]), float(xq[1]), float(xq[2]))
            logger.info(
                "[diag/forward %s] logits: shape=%s, min=%.4f, max=%.4f, "
                "mean=%.4f, q1=%.4f, q50=%.4f, q99=%.4f, n_nan=%d",
                tag, tuple(ls.shape),
                float(ls.min()), float(ls.max()), float(ls.mean()),
                float(lq[0]), float(lq[1]), float(lq[2]), n_nan_logits)
            logger.info(
                "[diag/forward %s] sigmoid: shape=%s, min=%.4f, max=%.4f, "
                "mean=%.4f, q50=%.4f, q90=%.4f, q99=%.4f, "
                "frac(prob>=%.2f)=%.4f, n_nan=%d",
                tag, tuple(ps.shape),
                float(ps.min()), float(ps.max()), float(ps.mean()),
                float(pq[0]), float(pq[1]), float(pq[2]),
                self.threshold, frac_thr, n_nan_prob)
            if n_nan_logits > 0 or n_nan_prob > 0:
                logger.error(
                    "[diag/forward %s] NaN detected (logits=%d, prob=%d). "
                    "This is the root cause of the 'all-foreground' "
                    "predictions — re-run with '--precision bf16'.",
                    tag, n_nan_logits, n_nan_prob)
        except Exception as _e:
            logger.warning("[diag/forward %s] stat failed: %s", tag, _e)

    def _build_z_window_input(
        self, vol: np.ndarray, z0: int, z1: int) -> np.ndarray:
        """多分辨率 z 窗口堆。s>1 总走 edge-padded 抽 round(pD*s) 切片；
        s=1 按 z_boundary_mode（stretch / edge_pad）。返 (C_res,pD,pH,pW) fp32。"""
        pD, pH, pW = self.patch_D, self.patch_H, self.patch_W
        z_center = (z0 + z1) // 2
        channels: List[np.ndarray] = []
        for scale in self.multi_res_scales:
            if scale == 1.0:
                if self.z_boundary_mode == "edge_pad":
                    patch = extract_z_patch_padded(vol, z_center, pD)
                else:
                    # 旧尾窗行为：取实际切片，后面 resize。
                    patch = vol[z0:z1]
            else:
                D_s = int(round(pD * scale))
                patch = extract_z_patch_padded(vol, z_center, D_s)
            patch = resize_3d(patch, pD, pH, pW, is_label=False)
            channels.append(patch)
        return np.stack(channels, axis=0).astype(np.float32)

    # ==================================================================
    # Whole-volume inference (no sliding window)
    # ==================================================================
    def _whole_volume_forward(self, vol: np.ndarray) -> np.ndarray:
        """全卷 resize 到输入尺寸单次 forward，后 resize 概率回原尺寸。仅 patch_mode=='whole' 用。"""
        D_orig, H_orig, W_orig = vol.shape
        pD, pH, pW = self.patch_D, self.patch_H, self.patch_W

        logger.info(
            "Whole-volume inference: orig=(%d,%d,%d) → model=(%d,%d,%d)",
            D_orig, H_orig, W_orig, pD, pH, pW)

        vol_resized = resize_3d(vol, pD, pH, pW, is_label=False)
        batch = torch.from_numpy(vol_resized[np.newaxis, np.newaxis]) \
            .float().to(self.device, non_blocking=True)
        probs = self._forward_batch(batch)       # (1, num_fg, pD, pH, pW)
        prob_small = probs[0]                    # (num_fg, pD, pH, pW)

        # resize_3d 原生支持领头通道轴（ndim==4）。
        return resize_3d(
            prob_small, D_orig, H_orig, W_orig, is_label=False)

    # ==================================================================
    # Cubic sliding window
    # ==================================================================
    def _sliding_window_cubic(self, vol: np.ndarray) -> np.ndarray:
        """3 轴 cubic 滑窗推理（带 overlap blending）。"""
        D_orig, H_orig, W_orig = vol.shape
        pD, pH, pW = self.patch_D, self.patch_H, self.patch_W

        stride_d = max(1, int(pD * (1 - self.overlap)))
        stride_h = max(1, int(pH * (1 - self.overlap)))
        stride_w = max(1, int(pW * (1 - self.overlap)))
        pos_d = self._compute_1d_positions(D_orig, pD, stride_d)
        pos_h = self._compute_1d_positions(H_orig, pH, stride_h)
        pos_w = self._compute_1d_positions(W_orig, pW, stride_w)

        total_windows = len(pos_d) * len(pos_h) * len(pos_w)
        logger.info(
            "Cubic sliding window: patch=(%d,%d,%d), strides=(%d,%d,%d), "
            "windows=%d×%d×%d=%d, blend=%s",
            pD, pH, pW, stride_d, stride_h, stride_w,
            len(pos_d), len(pos_h), len(pos_w), total_windows, self.blend_mode)

        weight_3d = self._build_3d_weight(pD, pH, pW, self.blend_mode)

        acc_pred = np.zeros((self.num_fg, D_orig, H_orig, W_orig),
                            dtype=np.float32)
        acc_weight = np.zeros((1, D_orig, H_orig, W_orig), dtype=np.float32)

        # 3D cubic ON 路径：体积一次上 GPU，builder 全程 on-device（逐视图一次 F.interpolate，
        # 零 scipy.ndimage.zoom）。OFF 路径保旧 CPU pipeline。
        keep_native_3d = bool(self.keep_native_multi_res
                              and self.patch_mode == "cubic")
        vol_t: Optional[torch.Tensor] = (
            torch.from_numpy(vol).float().to(self.device, non_blocking=True)
            if keep_native_3d else None)

        patches: List[np.ndarray] = []
        coords: List[Tuple[int, int, int, int, int, int, int, int, int]] = []
        centers: List[Tuple[int, int, int]] = []
        processed = 0

        def _flush():
            nonlocal processed
            if not patches:
                return
            if keep_native_3d:
                batch = self._build_batch_native_multi_res_cubic_gpu(
                    centers, vol_t)
            else:
                batch = self._build_batch_multi_res(patches, centers, vol)
            probs = self._forward_batch(batch)   # (B, num_fg, pD, pH, pW)
            for pred, (d0, d1, h0, h1, w0, w1, ad, ah, aw) in zip(probs, coords):
                # Trim prediction to actual (non-padded) size in each axis
                pred_trim = pred[:, :ad, :ah, :aw]
                w_trim = weight_3d[:ad, :ah, :aw]
                acc_pred[:, d0:d0 + ad, h0:h0 + ah, w0:w0 + aw] += (
                    pred_trim * w_trim[np.newaxis])
                acc_weight[:, d0:d0 + ad, h0:h0 + ah, w0:w0 + aw] += (
                    w_trim[np.newaxis])
            processed += len(patches)
            if processed % max(1, 10 * self.batch_size) == 0 \
                    or processed == total_windows:
                logger.info("  cubic window %d/%d", processed, total_windows)
            patches.clear()
            coords.clear()
            centers.clear()

        for d0, d1 in pos_d:
            for h0, h1 in pos_h:
                for w0, w1 in pos_w:
                    patch = vol[d0:d1, h0:h1, w0:w1]
                    ad, ah, aw = patch.shape

                    # 填短尾窗口到 (pD,pH,pW)。默认 mode='edge' 复制边界（归一化后 0 不是空气）。
                    if ad < pD or ah < pH or aw < pW:
                        pad_width = ((0, pD - ad), (0, pH - ah), (0, pW - aw))
                        if self.pad_value is None:
                            patch = np.pad(patch, pad_width, mode="edge")
                        else:
                            patch = np.pad(
                                patch, pad_width, mode="constant",
                                constant_values=self.pad_value)

                    patches.append(patch)
                    coords.append((d0, d1, h0, h1, w0, w1, ad, ah, aw))
                    centers.append(
                        ((d0 + d1) // 2, (h0 + h1) // 2, (w0 + w1) // 2))

                    if len(patches) >= self.batch_size:
                        _flush()

        _flush()

        np.maximum(acc_weight, 1e-8, out=acc_weight)
        return acc_pred / acc_weight

    # ==================================================================
    # Batch construction
    # ==================================================================
    def _build_batch_native_multi_res_cubic_gpu(
        self,
        centers: List[Tuple[int, int, int]],
        vol_t: torch.Tensor,
    ) -> torch.Tensor:
        """3D cubic ON 模式批建造：逐中心抽单 max-FOV cube → 逐视图中心裁 (D_k,H_k,W_k)
        后 trilinear 回 (pD,pH,pW) → 拼 C_res。返 (B,C_res,pD,pH,pW)。与训练几何一致。。"""
        pD, pH, pW = self.patch_D, self.patch_H, self.patch_W
        tD, tH, tW = self._mr_target_shape
        D_vol, H_vol, W_vol = vol_t.shape

        def _edge_pad_axis(t: torch.Tensor, axis: int,
                            pad_before: int, pad_after: int
                            ) -> torch.Tensor:
            """沿 axis 复制边界填充（expand+cat 零拷贝，cat 为唯一分配）。"""
            if pad_before == 0 and pad_after == 0:
                return t
            chunks: List[torch.Tensor] = []
            base_shape = list(t.shape)
            if pad_before > 0:
                first = t.narrow(axis, 0, 1)
                shape = list(base_shape)
                shape[axis] = pad_before
                chunks.append(first.expand(shape))
            chunks.append(t)
            if pad_after > 0:
                last = t.narrow(axis, t.shape[axis] - 1, 1)
                shape = list(base_shape)
                shape[axis] = pad_after
                chunks.append(last.expand(shape))
            return torch.cat(chunks, dim=axis)

        cubes: List[torch.Tensor] = []
        for (cd, ch, cw) in centers:
            # 逐轴复制填充以避免材料化完整填后 slab。
            d_lo = cd - tD // 2
            d_hi = d_lo + tD
            h_lo = ch - tH // 2
            h_hi = h_lo + tH
            w_lo = cw - tW // 2
            w_hi = w_lo + tW

            d_lo_in, d_hi_in = max(d_lo, 0), min(d_hi, D_vol)
            h_lo_in, h_hi_in = max(h_lo, 0), min(h_hi, H_vol)
            w_lo_in, w_hi_in = max(w_lo, 0), min(w_hi, W_vol)
            slab = vol_t[d_lo_in:d_hi_in, h_lo_in:h_hi_in,
                          w_lo_in:w_hi_in]
            slab = _edge_pad_axis(
                slab, 0, max(0, -d_lo), max(0, d_hi - D_vol))
            slab = _edge_pad_axis(
                slab, 1, max(0, -h_lo), max(0, h_hi - H_vol))
            slab = _edge_pad_axis(
                slab, 2, max(0, -w_lo), max(0, w_hi - W_vol))
            if slab.shape != (tD, tH, tW):
                raise RuntimeError(
                    f"native-multi-res cubic builder: slab shape "
                    f"{tuple(slab.shape)} != target {self._mr_target_shape}")

            # 逐视图裁 + resize，拼为 C_res。
            cube = slab.unsqueeze(0).unsqueeze(0).float()  # (1,1,tD,tH,tW)
            view_chunks: List[torch.Tensor] = []
            for (D_k, H_k, W_k) in self._mr_native_sizes:
                d0 = (tD - D_k) // 2
                h0 = (tH - H_k) // 2
                w0 = (tW - W_k) // 2
                crop = cube[:, :, d0:d0 + D_k, h0:h0 + H_k, w0:w0 + W_k]
                if (D_k, H_k, W_k) != (pD, pH, pW):
                    crop = F.interpolate(
                        crop, size=(pD, pH, pW),
                        mode="trilinear", align_corners=False)
                view_chunks.append(crop[0])  # (1, pD, pH, pW)
            # cat along channel axis → (C_res, pD, pH, pW)
            cubes.append(torch.cat(view_chunks, dim=0))

        return torch.stack(cubes, dim=0).contiguous()  # (B, C_res, pD, pH, pW)

    def _build_batch_multi_res(
        self,
        patches: List[np.ndarray],
        centers: List[Tuple[int, int, int]],
        vol: np.ndarray,
    ) -> torch.Tensor:
        """cubic 模式 (B,C_res,D,H,W) 批。s≠1 时重抽并 resize；沿用 _extract_cubic_patch 边界填充。。"""
        pD, pH, pW = self.patch_D, self.patch_H, self.patch_W
        batch_list: List[np.ndarray] = []
        for patch_1x, center in zip(patches, centers):
            channels: List[np.ndarray] = []
            for scale in self.multi_res_scales:
                if scale == 1.0:
                    channels.append(patch_1x)
                    continue
                sD = int(round(pD * scale))
                sH = int(round(pH * scale))
                sW = int(round(pW * scale))
                patch_s = _extract_cubic_patch(vol, center, (sD, sH, sW))
                patch_s = resize_3d(patch_s, pD, pH, pW, is_label=False)
                channels.append(patch_s)
            batch_list.append(np.stack(channels, axis=0))
        batch = np.stack(batch_list, axis=0)  # (B, C_res, D, H, W)
        return torch.from_numpy(batch).float().to(
            self.device, non_blocking=True)

    # ==================================================================
    # Forward + TTA
    # ==================================================================
    def _forward_batch(self, x: torch.Tensor) -> np.ndarray:
        """预装载批 forward 返 (B,num_fg,D,H,W) fp32 numpy。兼 DS list / flip TTA / AMP；
        2.5D 侧需折 C_res、2D forward 后 reshape 回 5D 以与 3D 贯通。。"""
        if self.patch_mode == "2_5d":
            return self._forward_batch_2_5d(x)

        autocast_ctx = autocast(
            device_type="cuda", enabled=self.use_amp, dtype=self.amp_dtype)
        with autocast_ctx:
            pred = self.model(x.to(self.model_dtype))
            if isinstance(pred, list):
                pred = pred[0]
            assert pred.shape[1] >= self.num_fg, (
                f"Model output has {pred.shape[1]} channels; "
                f"expected at least num_fg={self.num_fg} at 1x resolution.")
            prob = torch.sigmoid(pred.float())[:, :self.num_fg]

            if self.tta_flip:
                prob = self._tta_flip_ensemble(x, prob)

        return prob.float().cpu().numpy()

    def _forward_batch_2_5d(self, x: torch.Tensor) -> np.ndarray:
        """2.5D forward：折 C_res 为输入通道走 2D 模型后 reshape 为 (B,num_fg,D,H,W)。
        x: (B,C_res,D,H,W)，view 0 = 1× FOV 为监督目标；输出与 3D 合同供下游 blending。"""
        # Plan A lift：同 GPU 分支，不折 C_res*D，rank-5 直送 3D 模型。
        if self.lift_2_5d_to_3d:
            if x.ndim != 5:
                raise ValueError(
                    "lift_2_5d_to_3d=True expects rank-5 input "
                    f"(B, n_views, D, H, W); got x.shape={tuple(x.shape)}")
            autocast_ctx = autocast(
                device_type="cuda", enabled=self.use_amp,
                dtype=self.amp_dtype)
            with autocast_ctx:
                pred = self.model(x.to(self.model_dtype))
                if isinstance(pred, list):
                    pred = pred[0]
                if pred.shape[1] < self.num_fg:
                    raise ValueError(
                        f"Lift-mode model output has {pred.shape[1]} "
                        f"channels at dim 1; expected at least "
                        f"num_fg={self.num_fg}.")
                prob = torch.sigmoid(pred.float())[:, :self.num_fg]
                if self.tta_flip:
                    prob = self._tta_flip_ensemble(x, prob)
            return prob.float().cpu().numpy()

        x_2d = self._reshape_2_5d_input(x)  # (B, C_res*D, H, W)
        D = self.patch_D
        B, _, H, W = x_2d.shape
        autocast_ctx = autocast(
            device_type="cuda", enabled=self.use_amp, dtype=self.amp_dtype)
        with autocast_ctx:
            pred = self.model(x_2d.to(self.model_dtype))
            if isinstance(pred, list):
                pred = pred[0]
            expected_c = self.num_fg * D
            if pred.shape[1] != expected_c:
                raise ValueError(
                    f"2.5D model output channels {pred.shape[1]} != "
                    f"num_fg*D = {self.num_fg}*{D} = {expected_c}")
            # (B, num_fg*D, H, W) → (B, num_fg, D, H, W)
            pred_5d = rearrange(
                pred, 'b (c d) h w -> b c d h w', c=self.num_fg, d=D)
            prob = torch.sigmoid(pred_5d.float())

            if self.tta_flip:
                prob = self._tta_flip_ensemble_2_5d(x_2d, prob)

        return prob.float().cpu().numpy()

    def _reshape_2_5d_input(self, x: torch.Tensor) -> torch.Tensor:
        """折 2.5D 输入的 C_res 轴 → (B,C_res*D,H,W)，与 Trainer._squeeze_2_5d 一致。。"""
        if x.ndim != 5:
            raise ValueError(
                "2.5D inference expects rank-5 input "
                f"(B, C_res, D, H, W); got x.shape={tuple(x.shape)}")
        B, C_res, D, H, W = x.shape
        if D != self.patch_D:
            raise ValueError(
                f"2.5D input D-axis ({D}) != patch_D ({self.patch_D}). "
                "Window builder produced an unexpected slice count.")
        return rearrange(x, 'b c d h w -> b (c d) h w').contiguous()

    def _tta_flip_ensemble(
        self, x: torch.Tensor, base_prob: torch.Tensor,
    ) -> torch.Tensor:
        """3D TTA：原始 + 7 种轴 flip 组合取均；每次反 flip 后在原几何上累计。。"""
        total = base_prob.clone()
        count = 1.0
        for flip_dims in ([2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]):
            x_flip = torch.flip(x, flip_dims)
            pred_flip = self.model(x_flip.to(self.model_dtype))
            if isinstance(pred_flip, list):
                pred_flip = pred_flip[0]
            prob_flip = torch.sigmoid(pred_flip.float())[:, :self.num_fg]
            prob_flip = torch.flip(prob_flip, flip_dims)
            total = total + prob_flip
            count += 1.0
        return total / count

    def _tta_flip_ensemble_2_5d(
        self, x_2d: torch.Tensor, base_prob: torch.Tensor,
    ) -> torch.Tensor:
        """2.5D TTA：仅 H/W flip（D 是输入通道轴，flip 会反转物理切片顺序 → 分布偏移）。。"""
        B, _, H, W = x_2d.shape
        D = self.patch_D
        total = base_prob.clone()
        count = 1.0
        # x_2d 轴：2=H,3=W；prob_5d 轴：3=H,4=W（D 插在 2）。
        for flip_x_dims, flip_prob_dims in (
            ([2], [3]),       # H
            ([3], [4]),       # W
            ([2, 3], [3, 4])  # H + W
        ):
            x_flip = torch.flip(x_2d, flip_x_dims)
            pred_flip = self.model(x_flip.to(self.model_dtype))
            if isinstance(pred_flip, list):
                pred_flip = pred_flip[0]
            # (B, num_fg*D, H, W) → (B, num_fg, D, H, W)
            pred_flip_5d = rearrange(
                pred_flip, 'b (c d) h w -> b c d h w', c=self.num_fg, d=D)
            prob_flip = torch.sigmoid(pred_flip_5d.float())
            prob_flip = torch.flip(prob_flip, flip_prob_dims)
            total = total + prob_flip
            count += 1.0
        return total / count

    # ==================================================================
    # Geometry helpers
    # ==================================================================
    @staticmethod
    def _compute_1d_positions(
        length: int, patch: int, stride: int,
    ) -> List[Tuple[int, int]]:
        """逐轴返 (start,end) 窗口，尾窗反推使长度恢复为 patch（全覆盖）。。"""
        if length <= patch:
            return [(0, length)]
        positions: List[Tuple[int, int]] = []
        pos = 0
        while pos + patch <= length:
            positions.append((pos, pos + patch))
            pos += stride
        if positions[-1][1] < length:
            positions.append((length - patch, length))
        return positions

    @staticmethod
    def _build_1d_weight(n: int, mode: str = "gaussian") -> np.ndarray:
        """对称 1D blending 窗（长 n），fp32。"""
        if mode == "gaussian" and n > 1:
            center = (n - 1) / 2.0
            sigma = max(n / 4.0, 1e-6)
            z = np.arange(n, dtype=np.float32)
            return np.exp(-0.5 * ((z - center) / sigma) ** 2).astype(np.float32)
        return np.ones(n, dtype=np.float32)

    @staticmethod
    def _build_3d_weight(pD: int, pH: int, pW: int, mode: str) -> np.ndarray:
        """可分离 3D blending 权重（三轴独立 1D 外积），fp32。"""
        if mode == "gaussian":
            wd = Predictor._build_1d_weight(pD, "gaussian")
            wh = Predictor._build_1d_weight(pH, "gaussian")
            ww = Predictor._build_1d_weight(pW, "gaussian")
            return (wd[:, None, None] * wh[None, :, None]
                    * ww[None, None, :]).astype(np.float32)
        return np.ones((pD, pH, pW), dtype=np.float32)

    # ==================================================================
    # Probability → label map
    # ==================================================================
    def _prob_to_label(self, prob_volume: np.ndarray) -> np.ndarray:
        """概率体 → 整数 label map。逐体素：max fg 概率>threshold 取对应类别，否则 bg。
        NaN 体素强制为背景并报 ERROR（避免 fp16 LayerNorm 溢出造成“全前景”错误）。。"""
        bg_val = self.label_values[0]
        fg_values = np.array(self.label_values[1:], dtype=np.int64)
        assert len(fg_values) == self.num_fg

        nan_mask = np.isnan(prob_volume).any(axis=0)  # (D, H, W)
        n_nan = int(nan_mask.sum())
        if n_nan > 0:
            total = int(nan_mask.size)
            logger.error(
                "_prob_to_label: %d/%d voxels (%.2f%%) contain NaN "
                "probabilities — forcing to background. Root cause is "
                "almost always fp16 forward overflow; rerun inference "
                "with '--precision bf16' (or 'fp32').",
                n_nan, total, 100.0 * n_nan / max(1, total))

        # NaN → -inf 使 argmax/max 忽略；nan_mask 后面强制为 bg。
        if n_nan > 0:
            prob_volume = np.where(np.isnan(prob_volume),
                                   np.float32(-np.inf), prob_volume)
        max_prob = prob_volume.max(axis=0)            # (D, H, W)
        max_class = prob_volume.argmax(axis=0)        # (D, H, W)
        label_map = fg_values[max_class]
        label_map[max_prob < self.threshold] = bg_val
        if n_nan > 0:
            label_map[nan_mask] = bg_val

        # 选能装下所有 label 的最小有符号整型。
        max_abs = int(max(abs(v) for v in self.label_values))
        if max_abs <= np.iinfo(np.int8).max:
            out_dtype = np.int8
        elif max_abs <= np.iinfo(np.int16).max:
            out_dtype = np.int16
        else:
            out_dtype = np.int32
        return label_map.astype(out_dtype)

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
# Checkpoint loading
# ==================================================================
def _strip_compile_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """剥去 torch.compile 添加的 _orig_mod. 前缀。"""
    prefix = "_orig_mod."
    if any(k.startswith(prefix) for k in sd):
        return {(k[len(prefix):] if k.startswith(prefix) else k): v
                for k, v in sd.items()}
    return sd


def _unwrap_ema_state(ema_sd: Dict) -> Dict[str, torch.Tensor]:
    """将 {shadow, decay} 拆为普通 state_dict；已是拆过的旧格式原返。"""
    if isinstance(ema_sd, dict) and "shadow" in ema_sd and isinstance(
            ema_sd["shadow"], dict):
        return ema_sd["shadow"]
    return ema_sd


def _select_state_dict(
    ckpt: Dict, variant: str,
) -> Tuple[Dict[str, torch.Tensor], str]:
    """从 ckpt 选权重。variant: 'auto'（优 EMA）/'ema'/'online'。返 (sd, label) 供日志。"""
    has_online = "model_online_state_dict" in ckpt
    has_ema = "ema_state_dict" in ckpt
    primary = ckpt["model_state_dict"]

    if variant == "online":
        return (ckpt["model_online_state_dict"] if has_online else primary,
                "online")
    if variant == "ema":
        if has_ema:
            return _unwrap_ema_state(ckpt["ema_state_dict"]), "ema"
        logger.warning("EMA requested but not found in checkpoint; "
                       "using online weights.")
        return (ckpt["model_online_state_dict"] if has_online else primary,
                "online")
    # auto
    if has_ema:
        return _unwrap_ema_state(ckpt["ema_state_dict"]), "ema"
    return primary, "online"


_PRECISION_CHOICES = ("auto", "fp32", "bf16", "fp16")


def _resolve_inference_precision(precision: str, cfg: Config) -> str:
    """选推理 dtype。auto 跟随 cfg.train.amp_dtype：{bf16}→bf16, {fp16}→fp16, 其余退 bf16。
    fp16 下 ConvNeXt LayerNorm 可能 NaN，仅 opt-in。"""
    p = precision.lower()
    if p not in _PRECISION_CHOICES:
        raise ValueError(
            f"precision={precision!r} not in {_PRECISION_CHOICES}")
    if p != "auto":
        return p
    amp = (getattr(cfg.train, "amp_dtype", "bfloat16") or "bfloat16").lower()
    if amp in ("float16", "fp16"):
        return "fp16"
    return "bf16"


def run_inference(
    cfg: Config,
    checkpoint_path: str,
    image_paths: List[str],
    weight_variant: str = "auto",
    bbox_paths: Optional[List[str]] = None,
    precision: str = "auto",
) -> None:
    """对一组图像运行推理。

    weight_variant：'auto'(优 EMA) | 'ema' | 'online'。
    bbox_paths：与 image_paths 1∶1 的 ROI 掩膜；None 走全卷。
    precision：auto/fp32/bf16/fp16（fp16 仅兼容，ConvNeXt LayerNorm 可能 NaN）。
    """
    if bbox_paths is not None and len(bbox_paths) != len(image_paths):
        raise ValueError(
            f"bbox_paths length {len(bbox_paths)} != image_paths "
            f"length {len(image_paths)}")
    from .models.factory import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg)
    # weights_only=False：本 trainer ckpt 含 Config/numpy RNG，PyTorch 2.6+ 默认安全模式会拒。
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd, label = _select_state_dict(ckpt, weight_variant)
    sd = _strip_compile_prefix(sd)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning("Missing keys when loading checkpoint: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading checkpoint: %s", unexpected)
    # 加载 <半参数时硬错：避免随机初始权重静默推理。
    n_total = len(model.state_dict())
    n_loaded = n_total - len(missing)
    if n_total > 0 and n_loaded < max(1, n_total // 2):
        raise RuntimeError(
            f"Only {n_loaded}/{n_total} parameters loaded from "
            f"{checkpoint_path} (variant={label}). The checkpoint key "
            f"layout does not match the model — refusing to predict with "
            f"random weights. Unexpected keys: {unexpected[:8]}")

    model = model.to(device).eval()
    # 推理精度：auto 跟随 train.amp_dtype（默认 bf16 autocast，fp32 权重）；
    # fp16 仅 opt-in（LayerNorm 会 NaN，见 _resolve_inference_precision）。
    resolved_precision = _resolve_inference_precision(precision, cfg)
    if device.type == "cuda" and resolved_precision == "fp16":
        model = model.half()
        logger.info(
            "Inference precision: fp16 (model cast to float16; device=%s). "
            "WARNING: fp16 + LayerNorm backbones can overflow on near-"
            "constant patches and produce NaN — prefer 'bf16' or 'fp32' "
            "if you see NaN-driven 'all foreground' predictions.",
            device)
    else:
        logger.info(
            "Inference precision: %s (model in fp32; autocast=%s; device=%s).",
            resolved_precision,
            resolved_precision if resolved_precision in ("bf16", "fp16") else "off",
            device)
    logger.info("Model loaded from %s (variant=%s)", checkpoint_path, label)

    predictor = Predictor(model, cfg, device)

    n = len(image_paths)
    for i, path in enumerate(image_paths, 1):
        bbox_path = bbox_paths[i - 1] if bbox_paths is not None else None
        logger.info("[%d/%d] Processing: %s%s", i, n, path,
                    f" (bbox={bbox_path})" if bbox_path else "")
        try:
            result = predictor.predict_volume(
                path, output_dir=cfg.predict.output_dir,
                bbox_path=bbox_path)
            logger.info("  Label map shape: %s, unique labels: %s",
                        result["label_map"].shape,
                        np.unique(result["label_map"]).tolist())
        except Exception as e:
            logger.exception("Failed to process %s: %s", path, e)
            continue