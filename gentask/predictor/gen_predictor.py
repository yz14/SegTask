"""Generation inference for super-resolution.

The predictor reuses the shared NIfTI I/O and model topology layer, but it
operates on restored image volumes instead of segmentation logits.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..config import Config
from ..data.dataset import denormalize_image, load_nifti, preprocess_image
from ..data.degradation import _interp_mode, _phase_aligned_linear_upsample
from ..data.loader import match_condition_paths
from ..models.topology import build_topology
from ..trainer.checkpoint import (
    _select_state_dict,
    _strip_compile_prefix,
    unwrap_compile,
)

logger = logging.getLogger(__name__)

try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover - 仅写出阶段需要
    sitk = None


class GenerationPredictor:
    """超分复原推理器（回归 / 扩散通用，经 ``model.restore``）。"""

    def __init__(self, model: torch.nn.Module, cfg: Config, device: torch.device):
        if int(cfg.task.out_channels) != 1:
            raise NotImplementedError(
                "GenerationPredictor 目前仅支持 task.out_channels==1；"
                f"得到 {cfg.task.out_channels}。")
        self.cfg = cfg
        self.device = device
        self.model = model.to(device).eval()
        self.bare = unwrap_compile(self.model)
        self.is_2_5d = cfg.data.patch_mode == "2_5d"
        self.slab_depth = int(cfg.data.patch_size[0])
        self.cond_dirs = list(cfg.data.cond_dirs)
        self.cond_suffixes = cfg.data.cond_suffixes
        self.input_is_lr = str(cfg.predict.input_grid).lower() == "lr"
        self.vol_axis_scales = self._volume_axis_scales(cfg)
        # 经典 SISR（post-upsampling）：网络自身把真 LR 放大回 HR 网格，
        # 推理不做前置重采样；restore 输出尺寸 = 输入 × out_scales。
        self.net_upsamples = str(cfg.model.arch).lower() in ("edsr", "rcan")
        if self.net_upsamples and not self.input_is_lr:
            raise ValueError(
                "model.arch in ('edsr','rcan') consumes the true LR grid; "
                "set predict.input_grid='lr' (input volume is real LR).")
        self.out_scales = (self.vol_axis_scales if self.net_upsamples
                           else (1, 1, 1))
        # 多视图推理几何（与 trainer/pipelines 对称，H10）。
        topo = build_topology(cfg)
        self.topo = topo
        self.n_views = int(topo.n_views)
        self.scales = [float(s) for s in (cfg.data.multi_res_scales or [1.0])]
        self.lift = bool(topo.lift_2_5d_to_3d)
        self.native_d = bool(topo.keep_native_view_depth)
        if self.is_2_5d:
            self.view_depths = (list(topo.per_view_depths)
                                or [self.slab_depth] * self.n_views)
        else:
            self.view_depths = []
        if (str(cfg.task.algorithm).lower() == "diffusion"
                and self.n_views > 1):
            raise NotImplementedError(
                "DiffusionModel does not support multi-view inference "
                f"(multi_res_scales={cfg.data.multi_res_scales}).")

    @staticmethod
    def _volume_axis_scales(cfg: Config) -> Tuple[int, int, int]:
        """退化倍率映射到体轴 (D, H, W)。

        模型空间轴顺序：3D / 2.5D+lift 为 (D,H,W)；2.5D（未 lift）为 (H,W)，
        此时 z 轴折进通道、不参与退化（D 倍率恒为 1）。
        """
        task = cfg.task
        sdims = int(cfg.model.spatial_dims)
        per_axis = [int(s) for s in task.sr_scale_per_axis]
        if not per_axis:
            per_axis = [int(task.sr_scale)] * sdims
        if sdims == 3:
            return (per_axis[0], per_axis[1], per_axis[2])
        return (1, per_axis[0], per_axis[1])

    def _volume_scales_for(self, image_path: str) -> Tuple[int, int, int]:
        """该体的轴倍率 (D,H,W)。``predict.target_z_spacing>0`` 时逐体读 NIfTI
        z spacing，以 ``round(z_spacing/target)`` 覆盖配置 z 倍率（M7，spacing 感知）。"""
        scales = self.vol_axis_scales
        tz = float(self.cfg.predict.target_z_spacing)
        if tz <= 0.0:
            return scales
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(image_path))
        reader.ReadImageInformation()
        sp = reader.GetSpacing()  # (sx, sy, sz)
        z_sp = float(sp[2]) if len(sp) >= 3 else 0.0
        if not np.isfinite(z_sp) or z_sp <= 0.0:
            logger.warning(
                "%s: invalid z spacing in header; falling back to configured "
                "z scale %d.", image_path, scales[0])
            return scales
        z_scale = max(1, int(round(z_sp / tz)))
        if z_scale != scales[0]:
            logger.info(
                "%s: spacing-aware z scale = %d (z_spacing=%.3f, target=%.3f).",
                image_path, z_scale, z_sp, tz)
        return (z_scale, scales[1], scales[2])

    def _upsample_to_hr_grid(
        self, t: torch.Tensor,
        scales: Optional[Tuple[int, int, int]] = None) -> torch.Tensor:
        """真实 LR 体 (C,D,H,W) → HR 网格，与训练退化对偶（H9）。

        sr_sampling=='blur'：用 sr_kernel_up 插值（训练时 LR = down+up，推理输入
        相当于 down 后的体，补上 up 步骤）；'decimate'：相位对齐线性插值
        （保留帧逐体素保留，与训练 _decimate_interp 一致）。
        """
        if scales is None:
            scales = self.vol_axis_scales
        if all(s == 1 for s in scales):
            return t
        x = t[None]  # (1,C,D,H,W)
        out_size = tuple(int(n * s) for n, s in zip(x.shape[-3:], scales))
        if str(self.cfg.task.sr_sampling).lower() == "decimate":
            for i, sc in enumerate(scales):
                if sc > 1:
                    x = _phase_aligned_linear_upsample(
                        x, dim=2 + i, scale=int(sc), out_size=out_size[i])
        else:
            mode = _interp_mode(str(self.cfg.task.sr_kernel_up).lower(), 3)
            align = None if mode in ("area", "nearest") else False
            x = F.interpolate(x, size=out_size, mode=mode, align_corners=align)
        return x[0]

    def _load_cond_volume(self, cond_paths: Optional[List[str]]) -> Optional[np.ndarray]:
        if not cond_paths:
            return None
        dc = self.cfg.data
        cond_vols = [
            preprocess_image(
                load_nifti(path), dc.cond_intensity_min, dc.cond_intensity_max,
                dc.cond_normalize, dc.cond_global_mean, dc.cond_global_std)
            for path in cond_paths]
        return np.stack(cond_vols, axis=0)

    # ------------------------------------------------------------------
    # 多视图窗口抽取（与 trainer/pipelines 同几何；越界 edge 复制）
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_window(
        t: torch.Tensor, starts: List[int], sizes: List[int]) -> torch.Tensor:
        """对 ``t`` 的最后 ``len(sizes)`` 个维度抽取窗口；越界索引钳位到边界
        （等价 edge-pad，与训练 z_boundary_mode='edge_pad' 一致）。"""
        nd = len(sizes)
        first = t.ndim - nd
        for ax, (st, sz) in enumerate(zip(starts, sizes)):
            dim = first + ax
            idx = torch.arange(
                int(st), int(st) + int(sz), device=t.device
            ).clamp_(0, t.shape[dim] - 1)
            t = t.index_select(dim, idx)
        return t

    def _slab_views_2_5d(self, t: torch.Tensor, s: int) -> torch.Tensor:
        """slab 起点 ``s`` 处构造 2.5D 多视图模型输入（同中心多 FOV）。

        与 ``StackedMultiResPipeline`` / ``NativeDPipeline`` 的 center_crop
        几何一致：view k 相对主视图的 z 偏移 = (Dm-D_k)//2 - (Dm-D)//2。
        native_d 保持原生深度拼通道；uniform 各视图 resize 回 D 后拼通道；
        lift 堆到 rank-5 视图轴。``t`` 形如 (D,H,W)。
        """
        d = self.slab_depth
        depths = self.view_depths
        dm = max(depths)
        hw = list(t.shape[-2:])
        views = []
        for dk in depths:
            off = (dm - dk) // 2 - (dm - d) // 2
            slab = self._extract_window(
                t, [s + off, 0, 0], [dk] + hw)  # (dk,H,W)
            if not self.native_d and dk != d:
                slab = F.interpolate(
                    slab[None, None], size=tuple([d] + hw),
                    mode="trilinear", align_corners=False)[0, 0]
            views.append(slab)
        if self.lift:
            return torch.stack(views, dim=0)[None]  # (1, n_views, D, H, W)
        return torch.cat(views, dim=0)[None]        # (1, ΣD_k | n*D, H, W)

    def _patch_views_3d(
        self, t: torch.Tensor, starts: List[int], patch: List[int]) -> torch.Tensor:
        """3D patch 多视图：以 patch 中心为公共中心逐视图抽 FOV 并 resize 回 patch。

        z_axis 仅 z 轴放大 FOV，cubic 三轴同步（与 view_sizes_z/cubic 一致）；
        越界部分 edge 复制。输出 ``(1, n_views, pD, pH, pW)``。``t`` (D,H,W)。
        """
        cubic = str(self.cfg.data.patch_mode).lower() == "cubic"
        all_sizes = []
        for sc in self.scales:
            if cubic:
                all_sizes.append([int(round(p * sc)) for p in patch])
            else:  # z_axis：仅 z 轴放大
                all_sizes.append([int(round(patch[0] * sc)), patch[1], patch[2]])
        maxes = [max(s[ax] for s in all_sizes) for ax in range(3)]
        views = []
        for sizes in all_sizes:
            offs = [(m - sz) // 2 - (m - p) // 2
                    for m, sz, p in zip(maxes, sizes, patch)]
            win = self._extract_window(
                t, [s + o for s, o in zip(starts, offs)], sizes)
            if list(win.shape) != list(patch):
                win = F.interpolate(
                    win[None, None], size=tuple(patch),
                    mode="trilinear", align_corners=False)[0, 0]
            views.append(win)
        return torch.stack(views, dim=0)[None]

    # ------------------------------------------------------------------
    # 滑窗聚合（M15+M16）：重叠步长 + 高斯 / 等权融合
    # ------------------------------------------------------------------
    @staticmethod
    def _window_starts(n: int, size: int, stride: int) -> List[int]:
        """轴向滑窗起点：末窗对齐尾部；``n <= size`` 时单窗（edge 复制补足）。"""
        if n <= size:
            return [0]
        starts = list(range(0, n - size + 1, max(1, stride)))
        if starts[-1] != n - size:
            starts.append(n - size)
        return starts

    def _blend_weight(self, sizes: List[int]) -> torch.Tensor:
        """滑窗融合权重（可分离）：'gaussian' 中心高权（σ = size/8，下限 1），
        'uniform' 全 1。返回形状 ``sizes`` 的张量。"""
        if str(self.cfg.predict.blend).lower() == "uniform":
            return torch.ones(*sizes, device=self.device)
        axes = []
        for n in sizes:
            x = torch.arange(n, device=self.device, dtype=torch.float32)
            x = x - (n - 1) / 2.0
            sigma = max(n / 8.0, 1.0)
            axes.append(torch.exp(-0.5 * (x / sigma) ** 2))
        w = axes[0]
        for a in axes[1:]:
            w = w[..., None] * a
        # 角部极小权重截断，避免 fp32 累加下溢（同 MONAI importance map 处理）
        return w.clamp(min=1e-3)

    def _restore_3d_sliding(
        self, t: torch.Tensor, cond_t: Optional[torch.Tensor]) -> torch.Tensor:
        """3D 滑窗推理：patch 尺度与训练一致，重叠区加权融合。``t`` (D,H,W)。"""
        vol = list(t.shape)
        patch = [int(p) for p in self.cfg.data.patch_size]
        strides = [max(1, int(round(p * (1.0 - float(self.cfg.predict.overlap)))))
                   for p in patch]
        axis_starts = [self._window_starts(n, p, st)
                       for n, p, st in zip(vol, patch, strides)]
        osc = self.out_scales  # SISR 网络自身上采：输出网格 = 输入 × osc
        out_shape = [n * s for n, s in zip(vol, osc)]
        out = torch.zeros(*out_shape, device=t.device)
        weight = torch.zeros(*out_shape, device=t.device)
        w = self._blend_weight([p * s for p, s in zip(patch, osc)])
        for sz in axis_starts[0]:
            for sy in axis_starts[1]:
                for sx in axis_starts[2]:
                    starts = [sz, sy, sx]
                    x = self._patch_views_3d(t, starts, patch)
                    cond_win = None
                    if cond_t is not None:
                        cond_win = self._extract_window(
                            cond_t, starts, patch)[None]
                    rec = self.bare.restore(x, cond=cond_win)[0, 0]
                    valid = [min(p, n - s) * sc
                             for p, n, s, sc in zip(patch, vol, starts, osc)]
                    vd, vh, vw = valid
                    wv = w[:vd, :vh, :vw]
                    region = tuple(
                        slice(s * sc, s * sc + v)
                        for s, sc, v in zip(starts, osc, valid))
                    out[region] += rec[:vd, :vh, :vw] * wv
                    weight[region] += wv
        return out / weight.clamp(min=1e-8)

    @torch.no_grad()
    def restore_volume(
        self, vol: np.ndarray, cond_vol: Optional[np.ndarray] = None) -> np.ndarray:
        """复原归一化体数据 ``vol`` (D,H,W) → HR 体数据 (D,H,W)。"""
        t = torch.from_numpy(np.ascontiguousarray(vol)).float().to(self.device)
        cond_t = None if cond_vol is None else torch.from_numpy(
            np.ascontiguousarray(cond_vol)).float().to(self.device)
        if not self.is_2_5d:  # 3D
            if str(self.cfg.data.patch_mode).lower() == "whole":
                # whole：训练即整卷，单次前向
                rec = self.bare.restore(t[None, None], cond=(
                    None if cond_t is None else cond_t[None]))[0, 0]
            else:  # z_axis / cubic：滑窗（patch 尺度与训练一致）
                rec = self._restore_3d_sliding(t, cond_t)
            return rec.float().cpu().numpy()

        dz = t.shape[0]
        d = self.slab_depth
        # 2.5D 下 z 折进通道不参与退化（out_scales[0]==1），仅 H/W 可能放大。
        out_hw = [n * s for n, s in zip(t.shape[-2:], self.out_scales[1:])]
        out = torch.zeros(dz, *out_hw, device=t.device)
        count = torch.zeros(dz, device=self.device)
        stride = max(1, int(round(d * (1.0 - float(self.cfg.predict.overlap)))))
        starts = self._window_starts(dz, d, stride)
        wz = self._blend_weight([d])  # (d,)
        for s in starts:
            x = self._slab_views_2_5d(t, s)
            cond_slab = None
            if cond_t is not None:
                cond_slab = self._extract_window(
                    cond_t, [s, 0, 0], [d] + list(cond_t.shape[-2:]))[None]
            rec = self.bare.restore(x, cond=cond_slab)
            rec = rec[0, 0] if rec.ndim == 5 else rec[0]  # (d,H,W)
            valid = min(d, dz - s)
            wv = wz[:valid]
            out[s:s + valid] += rec[:valid] * wv[:, None, None]
            count[s:s + valid] += wv
        out = out / count[:, None, None].clamp(min=1e-8)
        return out.float().cpu().numpy()

    def predict_volume(
        self, image_path: str, output_dir: str,
        cond_paths: Optional[List[str]] = None) -> np.ndarray:
        """读取 NIfTI → 归一化 → 复原 → 写出 ``*_sr.nii.gz``。"""
        if sitk is None:
            raise ImportError("SimpleITK 未安装，无法写出 NIfTI。")
        dc = self.cfg.data
        raw = load_nifti(image_path)
        vol = preprocess_image(
            raw, dc.intensity_min, dc.intensity_max,
            dc.normalize, dc.global_mean, dc.global_std)
        cond_vol = self._load_cond_volume(cond_paths)
        scales = (self._volume_scales_for(image_path) if self.input_is_lr
                  else self.vol_axis_scales)
        if self.input_is_lr and not self.net_upsamples:
            # 真实 LR：先重采样到 HR 网格（与训练退化对偶）；SISR 网络
            # （net_upsamples）直接吃真 LR，由上采头放大，不做前置重采样。
            vol_t = torch.from_numpy(np.ascontiguousarray(vol)).float()
            vol = self._upsample_to_hr_grid(vol_t[None], scales)[0].numpy()
            if cond_vol is not None:
                cond_t = torch.from_numpy(np.ascontiguousarray(cond_vol)).float()
                cond_vol = self._upsample_to_hr_grid(cond_t, scales).numpy()
        rec = self.restore_volume(vol, cond_vol=cond_vol)
        if bool(self.cfg.predict.denormalize):
            rec = denormalize_image(
                rec, dc.intensity_min, dc.intensity_max,
                dc.normalize, dc.global_mean, dc.global_std)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).name.replace(".nii.gz", "").replace(".nii", "")
        ref = sitk.ReadImage(str(image_path))
        img = sitk.GetImageFromArray(rec.astype(np.float32, copy=False))
        if rec.shape == tuple(raw.shape):
            img.CopyInformation(ref)
        else:  # LR→HR 网格：spacing 按倍率细分，origin 按插值相位修正
            self._set_hr_geometry(img, ref, scales)
        out_path = out_dir / f"{stem}_sr.nii.gz"
        sitk.WriteImage(img, str(out_path), useCompression=True)
        logger.info("Saved super-resolved volume: %s", out_path)
        return rec

    def _set_hr_geometry(
        self, img: "sitk.Image", ref: "sitk.Image",
        scales: Optional[Tuple[int, int, int]] = None) -> None:
        """按体轴倍率写 HR 几何：spacing/scale；origin 保持体素中心对应关系。

        sitk 轴序为 (x,y,z)=(W,H,D)，与体轴 (D,H,W) 相反。'decimate' 相位对齐
        （输出 j 对应输入 j/scale，索引 0 重合）origin 不变；'blur' 插值
        （align_corners=False，半像素相位）origin 沿各轴平移 (sp_new-sp_old)/2。
        """
        if scales is None:
            scales = self.vol_axis_scales
        scales_xyz = tuple(float(s) for s in reversed(scales))
        old_sp = ref.GetSpacing()
        new_sp = tuple(sp / sc for sp, sc in zip(old_sp, scales_xyz))
        img.SetSpacing(new_sp)
        img.SetDirection(ref.GetDirection())
        origin = np.asarray(ref.GetOrigin(), dtype=np.float64)
        if str(self.cfg.task.sr_sampling).lower() != "decimate":
            direction = np.asarray(ref.GetDirection(), dtype=np.float64).reshape(3, 3)
            local = 0.5 * (np.asarray(new_sp) - np.asarray(old_sp))
            origin = origin + direction @ local
        img.SetOrigin(tuple(origin))


def run_generation_inference(
    cfg: Config,
    checkpoint_path: str,
    image_paths: List[str],
    weight_variant: str = "auto",
    output_dir: Optional[str] = None) -> None:
    """生成推理顶层入口：建模型、载权重、逐卷复原写出。"""
    from ..models.factory import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd, label = _select_state_dict(ckpt, weight_variant)
    sd = _strip_compile_prefix(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    n_total = len(model.state_dict())
    if n_total > 0 and (n_total - len(missing)) < max(1, n_total // 2):
        raise RuntimeError(
            f"Only {n_total - len(missing)}/{n_total} params loaded from "
            f"{checkpoint_path} (variant={label}); refusing random-weight "
            f"inference. Unexpected: {unexpected[:8]}")
    logger.info("Generation model loaded from %s (variant=%s)",
                checkpoint_path, label)

    predictor = GenerationPredictor(model, cfg, device)
    out_dir = output_dir or cfg.predict.output_dir
    cond_path_sets: Optional[List[List[str]]] = None
    if cfg.data.cond_dirs:
        cond_path_sets = []
        for cond_dir in cfg.data.cond_dirs:
            cond_path_sets.append(match_condition_paths(
                image_paths, cond_dir, cfg.data.image_suffix, cfg.data.cond_suffixes))
    for idx, path in enumerate(image_paths):
        cond_paths = None
        if cond_path_sets is not None:
            cond_paths = [paths[idx] for paths in cond_path_sets]
        predictor.predict_volume(path, out_dir, cond_paths=cond_paths)


__all__ = ["GenerationPredictor", "run_generation_inference"]
