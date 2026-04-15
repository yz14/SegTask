"""Z-axis sliding window inference for 3D segmentation.

Inference strategy (matching the training z-axis patching approach):
1. Load full volume (D_orig, H_orig, W_orig)
2. Preprocess (intensity windowing + normalization)
3. Slide along z-axis with configurable overlap:
   - For each window: extract D slices, resize H,W to model input
   - Run model forward pass (sigmoid output)
   - Accumulate predictions with Gaussian or uniform blending weights
4. Resize predictions back to original H,W resolution
5. Threshold + argmax → final label map
6. Save as NIfTI

Blending strategies for overlapping z-windows:
  - "average": uniform weight in overlap regions
  - "gaussian": Gaussian weighting along z-axis (higher weight at center)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .data.dataset import load_nifti, preprocess_image, resize_3d

logger = logging.getLogger(__name__)


class Predictor:
    """Z-axis sliding window predictor for 3D segmentation.

    Handles full-volume inference by:
    1. Sliding along z-axis with overlap
    2. Resizing each patch to model input size
    3. Blending overlapping predictions
    4. Restoring original resolution
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: Config,
        device: torch.device):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.model.eval()

        pc = cfg.predict
        self.z_overlap = pc.z_overlap
        self.blend_mode = pc.blend_mode
        self.batch_size = pc.batch_size
        self.tta_flip = pc.tta_flip
        self.threshold = pc.threshold
        self.save_probs = pc.save_probabilities

        self.patch_D, self.patch_H, self.patch_W = cfg.data.patch_size
        self.label_values = cfg.data.label_values
        self.num_fg = cfg.num_fg_classes

    @torch.no_grad()
    def predict_volume(
        self,
        image_path: str,
        output_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Run inference on a single NIfTI volume.

        Args:
            image_path: Path to input NIfTI file.
            output_dir: If provided, save predictions as NIfTI files.

        Returns:
            Dict with keys:
              "label_map": (D, H, W) int32 — predicted label map
              "probabilities": (num_fg, D, H, W) float32 — sigmoid probabilities
        """
        # Load and preprocess
        dc = self.cfg.data
        raw_vol = load_nifti(image_path)  # (D_orig, H_orig, W_orig)
        D_orig, H_orig, W_orig = raw_vol.shape
        logger.info("Loaded %s: shape=(%d, %d, %d)", image_path, D_orig, H_orig, W_orig)

        vol = preprocess_image(  # 归一化
            raw_vol, dc.intensity_min, dc.intensity_max,
            dc.normalize, dc.global_mean, dc.global_std)

        # Run sliding window inference
        prob_volume = self._sliding_window_z(vol)  # (num_fg, D_orig, H_orig, W_orig)

        # Threshold → label map
        label_map = self._prob_to_label(prob_volume)  # (D_orig, H_orig, W_orig)

        result = {
            "label_map": label_map,
            "probabilities": prob_volume,
        }

        # Save outputs
        if output_dir:
            self._save_predictions(
                image_path, label_map, prob_volume, output_dir)

        return result

    def _sliding_window_z(self, vol: np.ndarray) -> np.ndarray:
        """Sliding window along z-axis with overlap and blending.

        Args:
            vol: Preprocessed volume (D_orig, H_orig, W_orig) float32.

        Returns:
            Probability volume (num_fg, D_orig, H_orig, W_orig) float32.
        """
        D_orig, H_orig, W_orig = vol.shape
        D_patch = self.patch_D

        # Compute z-axis window positions
        stride = max(1, int(D_patch * (1 - self.z_overlap)))
        z_positions = self._compute_z_positions(D_orig, D_patch, stride)

        logger.info("Sliding window: D_patch=%d, stride=%d, num_windows=%d, blend=%s",
                     D_patch, stride, len(z_positions), self.blend_mode)

        # Blending weight along z-axis (per patch)
        z_weight = self._build_z_weight(D_patch)  # (D_patch,)

        # Accumulator for weighted predictions and weight sum
        acc_pred = np.zeros((self.num_fg, D_orig, H_orig, W_orig), dtype=np.float64)
        acc_weight = np.zeros((1, D_orig, 1, 1), dtype=np.float64)

        # Collect patches in batches
        patches = []
        patch_metas = []  # (z_start, z_end, actual_d)

        for z_start, z_end in z_positions:
            # Extract z-window
            patch = vol[z_start:z_end]  # (actual_d, H_orig, W_orig)
            actual_d = patch.shape[0]

            # Resize to model input: (D_patch, H_patch, W_patch)
            patch_resized = resize_3d(
                patch, D_patch, self.patch_H, self.patch_W, is_label=False)

            patches.append(patch_resized)
            patch_metas.append((z_start, z_end, actual_d))

            # Process batch when full or at end
            if len(patches) >= self.batch_size or (z_start, z_end) == z_positions[-1]:
                batch_pred = self._infer_batch(patches)  # list of (num_fg, D_patch, H_patch, W_patch)

                for pred, (zs, ze, ad) in zip(batch_pred, patch_metas):
                    # Resize prediction back: (num_fg, D_patch, H_patch, W_patch) → (num_fg, actual_d, H_orig, W_orig)
                    pred_orig = resize_3d(pred, ad, H_orig, W_orig, is_label=False)

                    # Compute weight for this window
                    w = z_weight[:ad]  # trim to actual depth

                    # Accumulate with blending weights
                    w_4d = w.reshape(1, -1, 1, 1)  # (1, actual_d, 1, 1)
                    acc_pred[:, zs:ze, :, :] += pred_orig * w_4d
                    acc_weight[:, zs:ze, :, :] += w_4d

                patches = []
                patch_metas = []

        # Normalize by accumulated weights
        acc_weight = np.maximum(acc_weight, 1e-8)
        prob_volume = (acc_pred / acc_weight).astype(np.float32)

        return prob_volume

    def _compute_z_positions(
        self, D_orig: int, D_patch: int, stride: int) -> List[Tuple[int, int]]:
        """Compute (z_start, z_end) for each sliding window position.

        Ensures full coverage: the last window is shifted back to cover the tail.
        """
        positions = []
        z = 0
        while z + D_patch <= D_orig:
            positions.append((z, z + D_patch))
            z += stride

        # If volume not fully covered, add a final window at the tail
        if not positions or positions[-1][1] < D_orig:
            z_start = max(0, D_orig - D_patch)
            positions.append((z_start, D_orig))

        return positions

    def _build_z_weight(self, D_patch: int) -> np.ndarray:
        """Build blending weight vector along z-axis.

        Args:
            D_patch: Number of z-slices in a patch.

        Returns:
            Weight array of shape (D_patch,) float64.
        """
        if self.blend_mode == "gaussian":
            # Gaussian centered at middle, sigma = D/4
            center = (D_patch - 1) / 2.0
            sigma = D_patch / 4.0
            z = np.arange(D_patch, dtype=np.float64)
            w = np.exp(-0.5 * ((z - center) / sigma) ** 2)
            return w
        else:
            # Uniform weight
            return np.ones(D_patch, dtype=np.float64)

    def _infer_batch(self, patches: List[np.ndarray]) -> List[np.ndarray]:
        """Run model inference on a batch of patches.

        Args:
            patches: List of (D_patch, H_patch, W_patch) numpy arrays.

        Returns:
            List of (num_fg, D_patch, H_patch, W_patch) probability arrays.
        """
        # Stack into batch tensor: (B, 1, D, H, W)
        batch = np.stack([p[np.newaxis] for p in patches], axis=0)
        x = torch.from_numpy(batch).float().to(self.device)

        # Forward pass
        pred = self.model(x)
        if isinstance(pred, list):
            pred = pred[0]
        prob = torch.sigmoid(pred)  # (B, num_fg, D, H, W)

        # Optional TTA: flip augmentation
        if self.tta_flip:
            prob = self._tta_flip_ensemble(x, prob)

        prob_np = prob.cpu().numpy()
        return [prob_np[i] for i in range(prob_np.shape[0])]

    def _tta_flip_ensemble(
        self, x: torch.Tensor, base_prob: torch.Tensor,
    ) -> torch.Tensor:
        """Test-time augmentation via flipping along D, H, W axes.

        Averages predictions from original + flipped versions.
        """
        count = 1.0
        total = base_prob.clone()

        for flip_dims in [[2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]:
            x_flip = torch.flip(x, flip_dims)
            pred_flip = self.model(x_flip)
            if isinstance(pred_flip, list):
                pred_flip = pred_flip[0]
            prob_flip = torch.sigmoid(pred_flip)
            # Flip back to original orientation
            prob_flip = torch.flip(prob_flip, flip_dims)
            total = total + prob_flip
            count += 1.0

        return total / count

    def _prob_to_label(self, prob_volume: np.ndarray) -> np.ndarray:
        """Convert probability volume to integer label map.

        For each voxel:
        1. If max probability across fg classes > threshold → assign that fg label
        2. Otherwise → assign background (label_values[0])

        Args:
            prob_volume: (num_fg, D, H, W) float32.

        Returns:
            (D, H, W) int32 label map with original label values.
        """
        bg_val = self.label_values[0]
        fg_values = self.label_values[1:]

        # Find the class with max probability at each voxel
        max_prob = prob_volume.max(axis=0)  # (D, H, W)
        max_class = prob_volume.argmax(axis=0)  # (D, H, W) — index into fg_values

        # Map index → label value
        fg_arr = np.array(fg_values, dtype=np.int32)
        label_map = fg_arr[max_class]  # (D, H, W)

        # Background where max prob < threshold
        label_map[max_prob < self.threshold] = bg_val

        return label_map.astype(np.int32)

    def _save_predictions(
        self,
        image_path: str,
        label_map: np.ndarray,
        prob_volume: np.ndarray,
        output_dir: str,
    ) -> None:
        """Save prediction results as NIfTI files."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).name.replace(".nii.gz", "").replace(".nii", "")

        # Load original NIfTI for header/affine
        ref_nii = nib.load(image_path)
        affine = ref_nii.affine

        # Label map: transpose back to NIfTI convention (D,H,W) → (W,H,D) = (X,Y,Z)
        lbl_nifti = label_map.transpose(2, 1, 0).astype(np.int16)
        nib.save(nib.Nifti1Image(lbl_nifti, affine), str(out_dir / f"{stem}_pred.nii.gz"))
        logger.info("Saved label map: %s", out_dir / f"{stem}_pred.nii.gz")

        # Probability maps (optional)
        if self.save_probs:
            for c in range(prob_volume.shape[0]):
                prob_nifti = prob_volume[c].transpose(2, 1, 0).astype(np.float32)
                fname = f"{stem}_prob_class{c}.nii.gz"
                nib.save(nib.Nifti1Image(prob_nifti, affine), str(out_dir / fname))
            logger.info("Saved probability maps: %d classes", prob_volume.shape[0])


def run_inference(cfg: Config, checkpoint_path: str, image_paths: List[str]) -> None:
    """Run inference on a list of images using a trained model.

    Args:
        cfg: Full configuration.
        checkpoint_path: Path to model checkpoint.
        image_paths: List of NIfTI file paths.
    """
    import torch
    from .models.factory import build_model
    from .utils import ModelEMA

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    model = build_model(cfg)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info("Model loaded from %s", checkpoint_path)

    # Create predictor
    predictor = Predictor(model, cfg, device)

    # Run inference on each image
    for path in image_paths:
        logger.info("Processing: %s", path)
        result = predictor.predict_volume(
            path, output_dir=cfg.predict.output_dir)
        logger.info("  Label map shape: %s, unique labels: %s",
                     result["label_map"].shape,
                     np.unique(result["label_map"]).tolist())
