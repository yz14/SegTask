"""Prediction / inference pipeline for segmentation.

Supports:
- 2D slice-by-slice inference
- 2.5D sliding window inference
- 3D sliding window with Gaussian blending
- Test-Time Augmentation (TTA) via flipping
- Post-processing (connected components, hole filling)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from segtask.config import Config
from segtask.data.dataset import load_nifti, preprocess_image
from segtask.models.unet import UNet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gaussian importance map for blending
# ---------------------------------------------------------------------------
def _gaussian_importance_map(
    patch_size: Tuple[int, ...], sigma_scale: float = 0.125
) -> np.ndarray:
    """Create a Gaussian importance map for weighted blending."""
    center = [(s - 1) / 2.0 for s in patch_size]
    sigma = [s * sigma_scale for s in patch_size]

    grids = np.meshgrid(
        *[np.arange(s) for s in patch_size], indexing="ij"
    )
    gaussian = np.ones(patch_size, dtype=np.float32)
    for g, c, s in zip(grids, center, sigma):
        gaussian *= np.exp(-0.5 * ((g - c) / max(s, 1e-8)) ** 2)

    # Normalize to [0, 1] range with minimum floor
    gaussian = gaussian / gaussian.max()
    gaussian = np.clip(gaussian, 1e-4, None)
    return gaussian


# ---------------------------------------------------------------------------
# Sliding window inference
# ---------------------------------------------------------------------------
def sliding_window_inference_3d(
    model: nn.Module,
    volume: np.ndarray,
    patch_size: Tuple[int, int, int],
    num_classes: int,
    overlap: float = 0.5,
    batch_size: int = 4,
    blend_mode: str = "gaussian",
    device: torch.device = torch.device("cpu"),
    use_amp: bool = True,
    output_mode: str = "softmax",
) -> np.ndarray:
    """3D sliding window inference with overlapping patches.

    Training resizes H,W to patch_size[1:] while preserving D. To match this
    at inference, the volume is first resized to (D, ph, pw), then sliding
    window runs along D only. The output is resized back to original resolution.

    Args:
        model: Trained segmentation model (in eval mode).
        volume: Input volume (D, H, W), already preprocessed.
        patch_size: (pd, ph, pw).
        num_classes: Number of output channels (num_classes for softmax, num_fg for per_class).
        overlap: Fraction of overlap between patches (along D axis).
        batch_size: Number of patches per forward pass.
        blend_mode: "constant" or "gaussian".
        device: Computation device.
        use_amp: Use mixed precision.
        output_mode: "softmax" or "per_class".

    Returns:
        Predicted probability map (num_classes, D, H, W) at ORIGINAL resolution.
    """
    from segtask.data.dataset import resize_2d

    D_orig, H_orig, W_orig = volume.shape
    pd, ph, pw = patch_size

    # Resize H,W to match training resolution (training uses resize_2d to ph,pw)
    if H_orig != ph or W_orig != pw:
        volume_resized = resize_2d(volume, ph, pw, is_label=False)
    else:
        volume_resized = volume
    D = volume_resized.shape[0]

    # Pad D if needed
    pad_d = max(0, pd - D)
    if pad_d > 0:
        volume_resized = np.pad(volume_resized, ((0, pad_d), (0, 0), (0, 0)), mode="constant")
    D_pad = volume_resized.shape[0]

    # Sliding window along D axis only (H,W already match training size)
    step_d = max(1, int(pd * (1 - overlap)))

    # Importance map for D-axis blending
    if blend_mode == "gaussian":
        importance = _gaussian_importance_map(patch_size)
    else:
        importance = np.ones(patch_size, dtype=np.float32)

    output_sum = np.zeros((num_classes, D_pad, ph, pw), dtype=np.float32)
    count_map = np.zeros((D_pad, ph, pw), dtype=np.float32)

    max_d0 = max(1, D_pad - pd + 1)
    origins = list(range(0, max_d0, step_d))
    # Ensure last patch is included
    if origins[-1] + pd < D_pad:
        origins.append(max(0, D_pad - pd))
    logger.info("Sliding window: %d D-patches (%dx%dx%d, overlap=%.1f)", len(origins), pd, ph, pw, overlap)

    model.eval()
    for batch_start in range(0, len(origins), batch_size):
        batch_d0s = origins[batch_start : batch_start + batch_size]
        patches = []
        for d0 in batch_d0s:
            patches.append(volume_resized[d0:d0 + pd])

        batch_tensor = torch.from_numpy(np.stack(patches)[:, np.newaxis]).float().to(device)

        with torch.no_grad():
            with autocast(enabled=use_amp):
                pred = model(batch_tensor)
                if isinstance(pred, list):
                    pred = pred[0]
                if output_mode == "per_class":
                    prob = torch.sigmoid(pred).cpu().numpy()
                else:
                    prob = torch.softmax(pred, dim=1).cpu().numpy()

        for i, d0 in enumerate(batch_d0s):
            output_sum[:, d0:d0 + pd] += prob[i] * importance
            count_map[d0:d0 + pd] += importance

    count_map = np.maximum(count_map, 1e-8)
    output_sum /= count_map[np.newaxis]

    # Crop D back to original depth
    output_sum = output_sum[:, :D]

    # Resize output back to original H,W resolution
    if H_orig != ph or W_orig != pw:
        output_sum = resize_2d(output_sum, H_orig, W_orig, is_label=False)

    return output_sum


# ---------------------------------------------------------------------------
# 2D / 2.5D slice inference
# ---------------------------------------------------------------------------
def slice_inference(
    model: nn.Module,
    volume: np.ndarray,
    num_classes: int,
    mode: str = "2d",
    num_slices_per_side: int = 1,
    target_size: tuple = (0, 0),
    batch_size: int = 16,
    device: torch.device = torch.device("cpu"),
    use_amp: bool = True,
) -> np.ndarray:
    """Slice-by-slice inference for 2D or 2.5D models.

    Slices are RESIZED to target_size before model input, then output
    probabilities are resized back to original spatial dimensions.
    This works without labels — correct for real-world inference.

    For 2.5D: model outputs S*C channels. We apply per-slice softmax
    and keep only the center slice's predictions.

    Args:
        model: Trained model.
        volume: Input volume (D, H, W), preprocessed.
        num_classes: Number of semantic classes.
        mode: "2d" or "2.5d".
        num_slices_per_side: For 2.5D, number of context slices per side.
        target_size: (H, W) spatial size the model was trained on.
        batch_size: Batch size for inference.
        device: Device.
        use_amp: Use mixed precision.

    Returns:
        Predicted probability map (num_classes, D, H, W) at original resolution.

    2D mode: model outputs (B, num_classes, H, W) — standard softmax.
    2.5D mode: model outputs (B, num_classes * C, H, W) → reshape → (B, num_classes, C, H, W)
      softmax over C, take center slice → (B, num_classes, H, W).
    """
    from segtask.data.dataset import resize_2d

    D, H, W = volume.shape
    output = np.zeros((num_classes, D, H, W), dtype=np.float32)

    th, tw = target_size if (target_size[0] > 0 and target_size[1] > 0) else (H, W)
    need_resize = (H != th or W != tw)

    total_slices = getattr(model, 'total_slices', 1)
    semantic_classes = getattr(model, 'semantic_classes', num_classes)

    model.eval()
    for batch_start in range(0, D, batch_size):
        batch_indices = list(range(batch_start, min(batch_start + batch_size, D)))
        batch_tensors = []

        for center in batch_indices:
            if mode == "2d":
                img = volume[center][np.newaxis]  # (1, H, W)
            else:
                # 2.5D: stack context slices
                slices = []
                for offset in range(-num_slices_per_side, num_slices_per_side + 1):
                    s = center + offset
                    slices.append(volume[s] if 0 <= s < D else np.zeros((H, W), dtype=np.float32))
                img = np.stack(slices, axis=0)  # (C, H, W)

            # Resize to model's target_size
            if need_resize:
                img = resize_2d(img, th, tw, is_label=False)

            batch_tensors.append(img)

        batch_tensor = torch.from_numpy(np.stack(batch_tensors)).float().to(device)

        with torch.no_grad():
            with autocast(enabled=use_amp):
                pred = model(batch_tensor)
                if isinstance(pred, list):
                    pred = pred[0]

                if total_slices > 1:
                    # 2.5D: (B, num_classes*C, H, W) → (B, num_classes, C, H, W)
                    # softmax over C → take center slice
                    B_p = pred.shape[0]
                    N = semantic_classes
                    pred = pred.view(B_p, N, total_slices, th, tw)
                    pred = torch.softmax(pred, dim=2)  # (B, N, C, H, W)
                    center_idx = num_slices_per_side
                    prob = pred[:, :, center_idx].cpu().numpy()  # (B, N, H, W)
                else:
                    prob = torch.softmax(pred, dim=1).cpu().numpy()  # (B, num_classes, H, W)

        # Resize output back to original spatial dimensions
        for i, center in enumerate(batch_indices):
            if need_resize:
                output[:, center] = resize_2d(prob[i], H, W, is_label=False)
            else:
                output[:, center] = prob[i]

    return output


# ---------------------------------------------------------------------------
# Test-Time Augmentation
# ---------------------------------------------------------------------------
def tta_inference(
    model: nn.Module,
    volume: np.ndarray,
    num_classes: int,
    cfg: Config,
    device: torch.device,
) -> np.ndarray:
    """Inference with Test-Time Augmentation (flip).

    Averages predictions from original + flipped versions.
    """
    pc = cfg.predict
    dc = cfg.data

    output_mode = cfg.loss.output_mode

    # Base prediction
    def _single_pass(vol):
        if dc.mode == "3d":
            return sliding_window_inference_3d(
                model, vol,
                patch_size=tuple(dc.patch_size),
                num_classes=num_classes,
                overlap=pc.patch_overlap,
                batch_size=pc.batch_size,
                blend_mode=pc.blend_mode,
                device=device,
                use_amp=cfg.train.use_amp,
                output_mode=output_mode,
            )
        else:
            return slice_inference(
                model, vol,
                num_classes=num_classes,
                mode=dc.mode,
                num_slices_per_side=dc.num_slices_per_side,
                target_size=tuple(dc.target_size),
                batch_size=pc.batch_size,
                device=device,
                use_amp=cfg.train.use_amp,
            )

    output = _single_pass(volume)
    count = 1

    if pc.tta_enabled and pc.tta_flips:
        # Flip along each spatial axis
        for axis in range(volume.ndim):
            flipped_vol = np.flip(volume, axis=axis).copy()
            flipped_pred = _single_pass(flipped_vol)
            # Flip prediction back
            flipped_pred = np.flip(flipped_pred, axis=axis + 1).copy()  # +1 for class dim
            output += flipped_pred
            count += 1

    output /= count
    return output


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def postprocess(
    pred_prob: np.ndarray,
    threshold: float = 0.5,
    min_component_size: int = 0,
    fill_holes: bool = False,
) -> np.ndarray:
    """Post-process prediction probabilities to final segmentation.

    Args:
        pred_prob: (num_classes, *spatial) probability map.
        threshold: Probability threshold for binary segmentation.
        min_component_size: Minimum connected component size.
        fill_holes: Whether to fill holes in each class mask.

    Returns:
        Integer label map (*spatial) with class indices.
    """
    # Argmax to get class predictions
    pred_labels = np.argmax(pred_prob, axis=0)  # (*spatial)

    if min_component_size > 0 or fill_holes:
        try:
            from scipy import ndimage
        except ImportError:
            logger.warning("scipy not available for post-processing")
            return pred_labels

        num_classes = pred_prob.shape[0]
        for c in range(1, num_classes):  # skip background
            mask = (pred_labels == c)
            if not mask.any():
                continue

            if min_component_size > 0:
                labeled, n_components = ndimage.label(mask)
                for comp_id in range(1, n_components + 1):
                    comp_mask = labeled == comp_id
                    if comp_mask.sum() < min_component_size:
                        pred_labels[comp_mask] = 0  # set to background

            if fill_holes:
                mask = (pred_labels == c)
                filled = ndimage.binary_fill_holes(mask)
                pred_labels[filled & ~mask] = c

    return pred_labels


# ---------------------------------------------------------------------------
# Full prediction pipeline
# ---------------------------------------------------------------------------
class Predictor:
    """Full prediction pipeline for a single volume."""

    def __init__(
        self,
        model: UNet,
        cfg: Config,
        device: torch.device,
        checkpoint_path: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device

        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        self.model.eval()

    def _load_checkpoint(self, path: str) -> None:
        """Load model weights from checkpoint."""
        logger.info("Loading model from: %s", path)
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Model loaded (epoch %d, best_metric=%.4f)",
                     ckpt.get("epoch", -1), ckpt.get("best_metric", 0))

    def predict_volume(
        self,
        image_path: str,
        label_values: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
        """Predict segmentation for a single NIfTI volume.

        Args:
            image_path: Path to input NIfTI file.
            label_values: Label values to map back to.

        Returns:
            (pred_prob, pred_labels, nii_reference)
            pred_prob: (num_classes, D, H, W) probabilities
            pred_labels: (D, H, W) integer labels
            nii_reference: original NIfTI image for header/affine
        """
        dc = self.cfg.data
        pc = self.cfg.predict

        # Load and preprocess
        nii = nib.load(image_path)
        volume = load_nifti(image_path, dtype=np.float32)
        volume = preprocess_image(
            volume, dc.intensity_min, dc.intensity_max,
            dc.normalize, dc.global_mean, dc.global_std,
        )

        output_mode = self.cfg.loss.output_mode
        if output_mode == "per_class":
            num_out_ch = dc.num_classes - 1  # foreground channels only
        else:
            num_out_ch = dc.num_classes

        # Inference
        pred_prob = tta_inference(
            self.model, volume, num_out_ch, self.cfg, self.device,
        )

        # Post-process: convert probabilities to integer label map
        if output_mode == "per_class":
            # pred_prob: (num_fg, D, H, W), each channel is sigmoid probability
            # Assign voxel to fg class with highest prob > threshold, else background
            fg_max = pred_prob.max(axis=0)  # (D, H, W)
            fg_argmax = pred_prob.argmax(axis=0)  # (D, H, W) — index into fg classes
            pred_labels = np.where(fg_max > pc.threshold, fg_argmax + 1, 0).astype(np.int32)
        else:
            pred_labels = postprocess(
                pred_prob,
                threshold=pc.threshold,
                min_component_size=pc.min_component_size,
                fill_holes=pc.fill_holes,
            )

        # Connected component / hole filling for per_class too
        if output_mode == "per_class" and (pc.min_component_size > 0 or pc.fill_holes):
            pred_labels = postprocess(
                pred_prob, threshold=pc.threshold,
                min_component_size=pc.min_component_size,
                fill_holes=pc.fill_holes,
            )

        # Map back to original label values if needed
        if label_values:
            num_total = dc.num_classes
            if len(label_values) == num_total:
                mapped = np.zeros_like(pred_labels)
                for i, lv in enumerate(label_values):
                    mapped[pred_labels == i] = lv
                pred_labels = mapped

        return pred_prob, pred_labels, nii

    def predict_and_save(
        self,
        image_path: str,
        output_dir: str,
        label_values: Optional[List[int]] = None,
        save_probabilities: bool = False,
    ) -> str:
        """Predict and save results as NIfTI files.

        Returns:
            Path to the saved segmentation file.
        """
        pred_prob, pred_labels, nii_ref = self.predict_volume(image_path, label_values)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(image_path).name.replace(".nii.gz", "").replace(".nii", "")

        # Need to transpose back: (D,H,W) -> (W,H,D) for NIfTI (X,Y,Z)
        seg_data = pred_labels.transpose(2, 1, 0).astype(np.int16)
        seg_nii = nib.Nifti1Image(seg_data, nii_ref.affine, nii_ref.header)
        seg_path = out_dir / f"{stem}_pred.nii.gz"
        nib.save(seg_nii, str(seg_path))
        logger.info("Saved prediction: %s", seg_path)

        if save_probabilities:
            # Save each class probability
            for c in range(pred_prob.shape[0]):
                prob_data = pred_prob[c].transpose(2, 1, 0).astype(np.float32)
                prob_nii = nib.Nifti1Image(prob_data, nii_ref.affine, nii_ref.header)
                prob_path = out_dir / f"{stem}_prob_class{c}.nii.gz"
                nib.save(prob_nii, str(prob_path))

        return str(seg_path)
