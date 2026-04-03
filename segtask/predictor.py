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
import torch.nn.functional as F
from torch.cuda.amp import autocast

from .config import Config
from .data.dataset import load_nifti, preprocess_image
from .models.unet import UNet

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
) -> np.ndarray:
    """3D sliding window inference with overlapping patches.

    Args:
        model: Trained segmentation model (in eval mode).
        volume: Input volume (D, H, W), already preprocessed.
        patch_size: (pd, ph, pw).
        num_classes: Number of output classes.
        overlap: Fraction of overlap between patches.
        batch_size: Number of patches per forward pass.
        blend_mode: "constant" or "gaussian".
        device: Computation device.
        use_amp: Use mixed precision.

    Returns:
        Predicted probability map (num_classes, D, H, W).
    """
    D, H, W = volume.shape
    pd, ph, pw = patch_size

    # Pad if needed
    pad_d = max(0, pd - D)
    pad_h = max(0, ph - H)
    pad_w = max(0, pw - W)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        volume = np.pad(volume, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant")
    D_pad, H_pad, W_pad = volume.shape

    # Compute step size
    step_d = max(1, int(pd * (1 - overlap)))
    step_h = max(1, int(ph * (1 - overlap)))
    step_w = max(1, int(pw * (1 - overlap)))

    # Importance map
    if blend_mode == "gaussian":
        importance = _gaussian_importance_map(patch_size)
    else:
        importance = np.ones(patch_size, dtype=np.float32)

    # Output accumulator
    output_sum = np.zeros((num_classes, D_pad, H_pad, W_pad), dtype=np.float32)
    count_map = np.zeros((D_pad, H_pad, W_pad), dtype=np.float32)

    # Generate all patch origins
    origins = []
    for d0 in range(0, max(1, D_pad - pd + 1), step_d):
        for h0 in range(0, max(1, H_pad - ph + 1), step_h):
            for w0 in range(0, max(1, W_pad - pw + 1), step_w):
                d0 = min(d0, D_pad - pd)
                h0 = min(h0, H_pad - ph)
                w0 = min(w0, W_pad - pw)
                origins.append((d0, h0, w0))

    # Remove duplicates
    origins = list(set(origins))
    logger.info("Sliding window: %d patches (%dx%dx%d, overlap=%.1f)", len(origins), pd, ph, pw, overlap)

    # Process in batches
    model.eval()
    for batch_start in range(0, len(origins), batch_size):
        batch_origins = origins[batch_start : batch_start + batch_size]
        patches = []
        for d0, h0, w0 in batch_origins:
            patch = volume[d0:d0+pd, h0:h0+ph, w0:w0+pw]
            patches.append(patch)

        # Stack: (B, 1, D, H, W)
        batch_tensor = torch.from_numpy(np.stack(patches)[:, np.newaxis]).float().to(device)

        with torch.no_grad():
            with autocast(enabled=use_amp):
                pred = model(batch_tensor)
                if isinstance(pred, list):
                    pred = pred[0]
                prob = torch.softmax(pred, dim=1).cpu().numpy()

        for i, (d0, h0, w0) in enumerate(batch_origins):
            output_sum[:, d0:d0+pd, h0:h0+ph, w0:w0+pw] += prob[i] * importance
            count_map[d0:d0+pd, h0:h0+ph, w0:w0+pw] += importance

    # Normalize
    count_map = np.maximum(count_map, 1e-8)
    output_sum /= count_map[np.newaxis]

    # Crop back to original size
    output_sum = output_sum[:, :D, :H, :W]

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
    crop_size: tuple = (0, 0),
    batch_size: int = 16,
    device: torch.device = torch.device("cpu"),
    use_amp: bool = True,
) -> np.ndarray:
    """Slice-by-slice inference for 2D or 2.5D models.

    For 2.5D: the model outputs S*C channels (S slices × C semantic classes).
    We reshape to (B, S, C, H, W), apply per-slice softmax, and keep only
    the center slice's predictions — as required by the 2.5D spec.

    Args:
        model: Trained model.
        volume: Input volume (D, H, W), preprocessed.
        num_classes: Number of semantic classes.
        mode: "2d" or "2.5d".
        num_slices_per_side: For 2.5D, number of context slices per side.
        crop_size: (H, W) spatial size the model was trained on.
            If (0,0), use original volume size (no padding/cropping).
        batch_size: Batch size for inference.
        device: Device.
        use_amp: Use mixed precision.

    Returns:
        Predicted probability map (num_classes, D, H, W).
    """
    D, H, W = volume.shape
    output = np.zeros((num_classes, D, H, W), dtype=np.float32)

    # Determine if we need to pad slices to crop_size
    ch, cw = crop_size if (crop_size[0] > 0 and crop_size[1] > 0) else (H, W)
    need_pad = (H != ch or W != cw)
    # Compute center-crop origin for extracting results back
    pad_h = max(0, ch - H)
    pad_w = max(0, cw - W)
    # After padding, center-crop origin
    padded_H = H + pad_h
    padded_W = W + pad_w
    h0 = max(0, (padded_H - ch) // 2)
    w0 = max(0, (padded_W - cw) // 2)
    # Origin within padded volume for extracting original region
    orig_h0 = max(0, (ch - H) // 2) if pad_h > 0 else h0
    orig_w0 = max(0, (cw - W) // 2) if pad_w > 0 else w0

    model.eval()

    slices_to_process = list(range(D))
    for batch_start in range(0, D, batch_size):
        batch_indices = slices_to_process[batch_start : batch_start + batch_size]
        batch_tensors = []

        for center in batch_indices:
            if mode == "2d":
                img = volume[center][np.newaxis]  # (1, H, W)
            else:
                # 2.5D: stack context slices
                channels = []
                pad = num_slices_per_side
                for offset in range(-pad, pad + 1):
                    s = center + offset
                    if 0 <= s < D:
                        channels.append(volume[s])
                    else:
                        channels.append(np.zeros((H, W), dtype=np.float32))
                img = np.stack(channels, axis=0)  # (C, H, W)

            # Pad/crop to crop_size if needed
            if need_pad:
                img_padded = np.zeros((*img.shape[:-2], ch, cw), dtype=np.float32)
                # Center the original slice in the padded array
                src_h = min(H, ch)
                src_w = min(W, cw)
                dst_h0 = (ch - src_h) // 2
                dst_w0 = (cw - src_w) // 2
                src_h0 = (H - src_h) // 2
                src_w0 = (W - src_w) // 2
                img_padded[..., dst_h0:dst_h0+src_h, dst_w0:dst_w0+src_w] = \
                    img[..., src_h0:src_h0+src_h, src_w0:src_w0+src_w]
                img = img_padded

            batch_tensors.append(img)

        batch_tensor = torch.from_numpy(np.stack(batch_tensors)).float().to(device)

        with torch.no_grad():
            with autocast(enabled=use_amp):
                pred = model(batch_tensor)
                if isinstance(pred, list):
                    pred = pred[0]

                # 2.5D: reshape (B, S*C, H, W) → per-slice softmax → center slice
                total_slices = getattr(model, 'total_slices', 1)
                semantic_classes = getattr(model, 'semantic_classes', num_classes)

                if total_slices > 1:
                    B_p = pred.shape[0]
                    pH, pW = pred.shape[2], pred.shape[3]
                    # (B, S*C, H, W) → (B*S, C, H, W) → softmax → (B, S, C, H, W)
                    pred_r = pred.reshape(B_p, total_slices, semantic_classes, pH, pW)
                    pred_r = pred_r.reshape(B_p * total_slices, semantic_classes, pH, pW)
                    prob_r = torch.softmax(pred_r, dim=1)
                    prob_r = prob_r.reshape(B_p, total_slices, semantic_classes, pH, pW)
                    # Keep only center slice
                    center_idx = num_slices_per_side  # center of the stack
                    prob = prob_r[:, center_idx].cpu().numpy()  # (B, C, H, W)
                else:
                    prob = torch.softmax(pred, dim=1).cpu().numpy()

        for i, center in enumerate(batch_indices):
            if need_pad:
                # Extract the region corresponding to the original volume
                src_h = min(H, ch)
                src_w = min(W, cw)
                dst_h0_e = (ch - src_h) // 2
                dst_w0_e = (cw - src_w) // 2
                src_h0_e = (H - src_h) // 2
                src_w0_e = (W - src_w) // 2
                output[:, center, src_h0_e:src_h0_e+src_h, src_w0_e:src_w0_e+src_w] = \
                    prob[i, :, dst_h0_e:dst_h0_e+src_h, dst_w0_e:dst_w0_e+src_w]
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
            )
        else:
            return slice_inference(
                model, vol,
                num_classes=num_classes,
                mode=dc.mode,
                num_slices_per_side=dc.num_slices_per_side,
                crop_size=tuple(dc.crop_size),
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

        num_classes = dc.num_classes

        # Inference
        pred_prob = tta_inference(
            self.model, volume, num_classes, self.cfg, self.device,
        )

        # Post-process
        pred_labels = postprocess(
            pred_prob,
            threshold=pc.threshold,
            min_component_size=pc.min_component_size,
            fill_holes=pc.fill_holes,
        )

        # Map back to original label values if needed
        if label_values and len(label_values) == num_classes:
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
