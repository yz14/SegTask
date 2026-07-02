"""Generation inference for super-resolution.

The predictor reuses the shared NIfTI I/O and model topology layer, but it
operates on restored image volumes instead of segmentation logits.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from ..config import Config
from ..data.dataset import load_nifti, preprocess_image
from ..data.loader import match_condition_paths
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

    @torch.no_grad()
    def restore_volume(
        self, vol: np.ndarray, cond_vol: Optional[np.ndarray] = None) -> np.ndarray:
        """复原归一化体数据 ``vol`` (D,H,W) → HR 体数据 (D,H,W)。"""
        t = torch.from_numpy(np.ascontiguousarray(vol)).float().to(self.device)
        if not self.is_2_5d:  # 3D：整卷 (1,1,D,H,W)
            cond_t = None if cond_vol is None else torch.from_numpy(
                np.ascontiguousarray(cond_vol)).float().to(self.device)
            rec = self.bare.restore(t[None, None], cond=(
                None if cond_t is None else cond_t[None]))[0, 0]
            return rec.float().cpu().numpy()

        dz = t.shape[0]
        d = self.slab_depth
        out = torch.zeros_like(t)
        count = torch.zeros(dz, device=self.device)
        cond_t = None if cond_vol is None else torch.from_numpy(
            np.ascontiguousarray(cond_vol)).float().to(self.device)
        if dz <= d:  # 体深不足一个 slab：零填充到 d 后裁回
            pad = t.new_zeros(d, t.shape[1], t.shape[2])
            pad[:dz] = t
            cond_pad = None
            if cond_t is not None:
                cond_pad = cond_t.new_zeros(cond_t.shape[0], d, t.shape[1], t.shape[2])
                cond_pad[:, :dz] = cond_t
            rec = self.bare.restore(pad[None], cond=(
                None if cond_pad is None else cond_pad[None]))[0]  # (d,H,W)
            return rec[:dz].float().cpu().numpy()

        starts = list(range(0, dz - d + 1, d))
        if starts[-1] != dz - d:
            starts.append(dz - d)  # 末窗对齐尾部，重叠处平均
        for s in starts:
            slab = t[s:s + d]                  # (d,H,W) 折叠为通道
            cond_slab = None
            if cond_t is not None:
                cond_slab = cond_t[:, s:s + d]
            rec = self.bare.restore(slab[None], cond=(
                None if cond_slab is None else cond_slab[None]))[0]  # (d,H,W)
            out[s:s + d] += rec
            count[s:s + d] += 1
        out = out / count[:, None, None].clamp(min=1.0)
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
        rec = self.restore_volume(vol, cond_vol=cond_vol)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).name.replace(".nii.gz", "").replace(".nii", "")
        ref = sitk.ReadImage(str(image_path))
        img = sitk.GetImageFromArray(rec.astype(np.float32, copy=False))
        img.CopyInformation(ref)
        out_path = out_dir / f"{stem}_sr.nii.gz"
        sitk.WriteImage(img, str(out_path), useCompression=True)
        logger.info("Saved super-resolved volume: %s", out_path)
        return rec


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
