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

    @torch.no_grad()
    def restore_volume(self, vol: np.ndarray) -> np.ndarray:
        """复原归一化体数据 ``vol`` (D,H,W) → HR 体数据 (D,H,W)。"""
        t = torch.from_numpy(np.ascontiguousarray(vol)).float().to(self.device)
        if not self.is_2_5d:  # 3D：整卷 (1,1,D,H,W)
            rec = self.bare.restore(t[None, None])[0, 0]
            return rec.float().cpu().numpy()

        dz = t.shape[0]
        d = self.slab_depth
        out = torch.zeros_like(t)
        count = torch.zeros(dz, device=self.device)
        if dz <= d:  # 体深不足一个 slab：零填充到 d 后裁回
            pad = t.new_zeros(d, t.shape[1], t.shape[2])
            pad[:dz] = t
            rec = self.bare.restore(pad[None])[0]  # (d,H,W)
            return rec[:dz].float().cpu().numpy()

        starts = list(range(0, dz - d + 1, d))
        if starts[-1] != dz - d:
            starts.append(dz - d)  # 末窗对齐尾部，重叠处平均
        for s in starts:
            slab = t[s:s + d]                  # (d,H,W) 折叠为通道
            rec = self.bare.restore(slab[None])[0]  # (d,H,W)
            out[s:s + d] += rec
            count[s:s + d] += 1
        out = out / count[:, None, None].clamp(min=1.0)
        return out.float().cpu().numpy()

    def predict_volume(self, image_path: str, output_dir: str) -> np.ndarray:
        """读取 NIfTI → 归一化 → 复原 → 写出 ``*_sr.nii.gz``。"""
        if sitk is None:
            raise ImportError("SimpleITK 未安装，无法写出 NIfTI。")
        dc = self.cfg.data
        raw = load_nifti(image_path)
        vol = preprocess_image(
            raw, dc.intensity_min, dc.intensity_max,
            dc.normalize, dc.global_mean, dc.global_std)
        rec = self.restore_volume(vol)

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
    for path in image_paths:
        predictor.predict_volume(path, out_dir)


__all__ = ["GenerationPredictor", "run_generation_inference"]
