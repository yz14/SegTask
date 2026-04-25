"""Sliding window inference for 3D segmentation.

Two inference modes matching training patch modes:
  - "z_axis": slide along z-axis, resize H,W to model input
  - "cubic":  slide a 3D cube along all axes (D, H, W)

Both modes support:
  - Configurable overlap ratio
  - Gaussian or uniform blending for overlap regions
  - Test-time augmentation (flip)
  - AMP-consistent forward (matches training dtype)
  - Multi-resolution input construction (cubic mode only; a single-res
    model just uses scales=[1.0])
  - Checkpoint loading that handles torch.compile prefix, EMA weights,
    and best-model's EMA-primary convention
  - NIfTI output with affine preserved from source
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.amp import autocast

from .config import Config
from .data.dataset import (
    load_nifti, preprocess_image, resize_3d, _extract_cubic_patch,
    extract_z_patch_padded)

logger = logging.getLogger(__name__)


_AMP_DTYPES = {
    "float16": torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}


class Predictor:
    """Sliding window predictor for 3D segmentation.

    This class assumes the model was trained as multi-label sigmoid, with
    output channels ordered so that the first `num_fg` channels at the 1x
    resolution correspond 1-to-1 with `cfg.data.label_values[1:]`. That
    contract is asserted at construction time and re-checked per batch.
    """

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
        self.overlap = pc.z_overlap            # reused for all axes in cubic
        self.blend_mode = pc.blend_mode
        self.batch_size = pc.batch_size
        self.tta_flip = pc.tta_flip
        self.threshold = pc.threshold
        self.save_probs = pc.save_probabilities

        self.patch_mode = cfg.data.patch_mode
        self.patch_D, self.patch_H, self.patch_W = cfg.data.patch_size
        self.label_values = cfg.data.label_values
        self.num_fg = cfg.num_fg_classes
        # Default to single-resolution so empty config doesn't break the
        # downstream np.stack.
        self.multi_res_scales = cfg.data.multi_res_scales or [1.0]

        # AMP: match the training dtype so conv accumulation precision is
        # consistent between trainer and predictor. Any unknown value falls
        # back to bf16 to avoid silent dtype flip.
        amp_name = getattr(cfg.train, "amp_dtype", "bfloat16")
        if amp_name not in _AMP_DTYPES:
            logger.warning("Unknown amp_dtype=%r, falling back to bfloat16.",
                           amp_name)
            amp_name = "bfloat16"
        self.amp_dtype = _AMP_DTYPES[amp_name]
        self.use_amp = (
            getattr(cfg.train, "use_amp", True) and device.type == "cuda")

        # Pad value for volume-edge patches. Zeros after normalization is
        # *not* a safe default (for z-score CT, "air" sits near -mean/std,
        # not 0). If the config doesn't specify, fall back to the volume's
        # per-patch edge (handled via `np.pad(mode="edge")` below).
        self.pad_value: Optional[float] = getattr(
            cfg.data, "pad_value", None)

        # Contract: channels ↔ label_values[1:]
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
    ) -> Dict[str, np.ndarray]:
        """Run inference on a single NIfTI volume.

        Returns a dict with:
          "label_map":     (D, H, W) int — predicted label map
          "probabilities": (num_fg, D, H, W) float32 — sigmoid probabilities

        Patch-mode dispatch:
          - "whole"        — single forward on the full resized volume.
          - "cubic"        — 3D cubic sliding window with overlap.
          - "z_axis"       — z-axis sliding window with overlap.
          - "2_5d"         — z-axis sliding window with the SAME geometry
                             as ``z_axis``; the per-window forward squeezes
                             ``C_res=1`` to feed a 2D model whose output
                             is reshaped back to ``(num_fg, D, H, W)`` so
                             the existing accumulation/blending code works
                             unchanged. See ``_forward_batch``.
        """
        dc = self.cfg.data
        raw_vol = load_nifti(image_path)  # (D_orig, H_orig, W_orig)
        D_orig, H_orig, W_orig = raw_vol.shape
        logger.info("Loaded %s: shape=(%d, %d, %d)",
                    image_path, D_orig, H_orig, W_orig)

        vol = preprocess_image(
            raw_vol, dc.intensity_min, dc.intensity_max,
            dc.normalize, dc.global_mean, dc.global_std)

        if self.patch_mode == "whole":
            prob_volume = self._whole_volume_forward(vol)
        elif self.patch_mode == "cubic":
            prob_volume = self._sliding_window_cubic(vol)
        else:
            # "z_axis" or "2_5d" — same window geometry; see _forward_batch
            # for the 2.5D-specific squeeze + reshape.
            prob_volume = self._sliding_window_z(vol)

        label_map = self._prob_to_label(prob_volume)
        result = {"label_map": label_map, "probabilities": prob_volume}

        if output_dir:
            self._save_predictions(image_path, label_map, prob_volume,
                                   output_dir)
        return result

    # ==================================================================
    # Z-axis sliding window
    # ==================================================================
    def _sliding_window_z(self, vol: np.ndarray) -> np.ndarray:
        """Sliding window along z-axis with overlap and blending.

        H and W are always resized to the model's input size (no spatial
        windowing on those axes).

        Multi-resolution support (z-axis only):
            For each scale s in ``multi_res_scales``, extract
            ``round(pD * s)`` slices centered on the window's z-center
            (edge-replicated at volume bounds), resize to ``(pD, pH, pW)``,
            and stack as channel s. The resulting batch has shape
            ``(B, C_res, pD, pH, pW)``, matching the training contract of
            ``SegDataset3D`` with ``multi_res_scales``.

            For a single-scale ``[1.0]`` config, the per-window tensor is
            built identically to the legacy single-res z-axis path (tail
            windows with ``actual_d < pD`` are still resized-stretched,
            preserving previous behaviour).
        """
        D_orig, H_orig, W_orig = vol.shape
        pD, pH, pW = self.patch_D, self.patch_H, self.patch_W

        stride = max(1, int(pD * (1 - self.overlap)))
        z_positions = self._compute_1d_positions(D_orig, pD, stride)

        logger.info(
            "Z-axis sliding window: D_patch=%d, stride=%d, num_windows=%d, "
            "scales=%s, blend=%s",
            pD, stride, len(z_positions), self.multi_res_scales, self.blend_mode)

        z_weight = self._build_1d_weight(pD)  # (pD,) float32

        # float32 accumulators: for segmentation-scale volumes float64 more
        # than doubles memory without measurable accuracy gain (Gaussian
        # sums stay in [0, O(n_overlap)] range).
        acc_pred = np.zeros((self.num_fg, D_orig, H_orig, W_orig),
                            dtype=np.float32)
        acc_weight = np.zeros((1, D_orig, 1, 1), dtype=np.float32)

        # Per-window input tensors (already stacked across scales:
        # (C_res, pD, pH, pW)) and their source metadata for accumulation.
        window_inputs: List[np.ndarray] = []
        patch_metas: List[Tuple[int, int, int]] = []  # (z0, z1, actual_d)

        n_windows = len(z_positions)
        for idx, (z0, z1) in enumerate(z_positions):
            actual_d = z1 - z0
            window_inputs.append(
                self._build_z_window_input(vol, z0, z1))
            patch_metas.append((z0, z1, actual_d))

            is_last = idx == n_windows - 1
            if len(window_inputs) >= self.batch_size or is_last:
                # (B, C_res, pD, pH, pW)
                batch = torch.from_numpy(
                    np.stack(window_inputs, axis=0).astype(np.float32)
                ).to(self.device, non_blocking=True)
                probs = self._forward_batch(batch)   # (B, num_fg, pD, pH, pW)

                for pred, (zs, ze, ad) in zip(probs, patch_metas):
                    # Resize prediction back: (num_fg, pD, pH, pW)
                    #   → (num_fg, ad, H_orig, W_orig)
                    pred_orig = resize_3d(
                        pred, ad, H_orig, W_orig, is_label=False)

                    # Build a weight that is symmetric on the actual depth.
                    # For short tail windows, trimming z_weight[:ad] would
                    # be asymmetric; rebuild a length-ad window instead.
                    w = (z_weight if ad == pD
                         else self._build_1d_weight(ad))
                    w_4d = w.reshape(1, -1, 1, 1).astype(np.float32)
                    acc_pred[:, zs:ze, :, :] += pred_orig * w_4d
                    acc_weight[:, zs:ze, :, :] += w_4d

                window_inputs.clear()
                patch_metas.clear()

                if (idx + 1) % max(1, 10 * self.batch_size) == 0 or is_last:
                    logger.info("  z-window %d/%d", idx + 1, n_windows)

        np.maximum(acc_weight, 1e-8, out=acc_weight)
        return acc_pred / acc_weight

    def _build_z_window_input(
        self, vol: np.ndarray, z0: int, z1: int) -> np.ndarray:
        """Build the multi-scale input stack for one z-sliding window.

        For scale == 1.0 the legacy path is preserved exactly:
            take ``vol[z0:z1]`` (possibly shorter than pD at the tail)
            and resize to ``(pD, pH, pW)``.

        For scale > 1.0 we extract ``round(pD * scale)`` slices centered
        on the window's z-center with edge-replicate padding, so the
        physical z-FOV stays proportional to ``scale`` even when the
        window touches the volume boundary.

        Returns:
            ``(C_res, pD, pH, pW)`` float32 — one channel per scale, in
            the same order as ``self.multi_res_scales``.
        """
        pD, pH, pW = self.patch_D, self.patch_H, self.patch_W
        z_center = (z0 + z1) // 2
        channels: List[np.ndarray] = []
        for scale in self.multi_res_scales:
            if scale == 1.0:
                # Legacy tail-window behaviour: take actual slice, resize.
                patch = vol[z0:z1]
            else:
                D_s = int(round(pD * scale))
                patch = extract_z_patch_padded(vol, z_center, D_s)
            # (d, H_orig, W_orig) → (pD, pH, pW)
            patch = resize_3d(patch, pD, pH, pW, is_label=False)
            channels.append(patch)
        return np.stack(channels, axis=0).astype(np.float32)

    # ==================================================================
    # Whole-volume inference (no sliding window)
    # ==================================================================
    def _whole_volume_forward(self, vol: np.ndarray) -> np.ndarray:
        """Run a single forward pass on the ENTIRE volume resized to
        model input size, then resize the probabilities back to original.

        No sliding window, no blending. Mirrors the training-time
        ``SegDataset3DWhole`` data contract: 1-channel input of shape
        ``(1, pD, pH, pW)`` (TTA still stacks per-flip variants in the
        forward helper). Used only when ``patch_mode == "whole"``.
        """
        D_orig, H_orig, W_orig = vol.shape
        pD, pH, pW = self.patch_D, self.patch_H, self.patch_W

        logger.info(
            "Whole-volume inference: orig=(%d,%d,%d) → model=(%d,%d,%d)",
            D_orig, H_orig, W_orig, pD, pH, pW)

        vol_resized = resize_3d(vol, pD, pH, pW, is_label=False)
        # (1, 1, pD, pH, pW) — batch and channel dims.
        batch = torch.from_numpy(vol_resized[np.newaxis, np.newaxis]) \
            .float().to(self.device, non_blocking=True)
        probs = self._forward_batch(batch)       # (1, num_fg, pD, pH, pW)
        prob_small = probs[0]                    # (num_fg, pD, pH, pW)

        # Resize each class channel back to (D_orig, H_orig, W_orig).
        # `resize_3d` handles the leading channel axis (ndim==4) natively.
        return resize_3d(
            prob_small, D_orig, H_orig, W_orig, is_label=False)

    # ==================================================================
    # Cubic sliding window
    # ==================================================================
    def _sliding_window_cubic(self, vol: np.ndarray) -> np.ndarray:
        """3D cubic sliding window with overlap and blending."""
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

        patches: List[np.ndarray] = []
        coords: List[Tuple[int, int, int, int, int, int, int, int, int]] = []
        centers: List[Tuple[int, int, int]] = []
        processed = 0

        def _flush():
            nonlocal processed
            if not patches:
                return
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

                    # Pad short tail windows to (pD, pH, pW). Using a
                    # config-supplied pad_value (or "edge" replication) is
                    # much safer than constant 0: after normalization,
                    # zero is a valid tissue intensity, not air.
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
    def _build_batch_multi_res(
        self,
        patches: List[np.ndarray],
        centers: List[Tuple[int, int, int]],
        vol: np.ndarray,
    ) -> torch.Tensor:
        """(B, C_res, D, H, W) tensor for cubic mode.

        For each `scale != 1.0`, a larger/smaller cube is extracted around
        the patch center and resized to (pD, pH, pW). `_extract_cubic_patch`
        is expected to pad on out-of-bounds access using the same pad
        convention the dataset uses during training — if the project's
        dataset uses constant 0 here, the cubic inference will inherit
        that behaviour for consistency with training.
        """
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
        """Run the model on a pre-assembled batch tensor and return
        (B, num_fg, D, H, W) float32 probabilities as a numpy array.

        Mode dispatch:
          - 3D path  : x is (B, C_res, D, H, W); model produces
                       (B, num_fg*C_res, D, H, W); we slice to num_fg.
          - 2.5D path: x is still (B, 1, D, H, W) on the call boundary,
                       but the model is planar 2D with in_channels=D.
                       Squeeze C_res, forward, then reshape
                       (B, num_fg*D, H, W) → (B, num_fg, D, H, W) so the
                       accumulation code is shape-identical with 3D.

        Handles: deep-supervision list output, optional flip-TTA, and
        AMP dtype matching training.
        """
        if self.patch_mode == "2_5d":
            return self._forward_batch_2_5d(x)

        autocast_ctx = autocast(
            device_type="cuda", enabled=self.use_amp, dtype=self.amp_dtype)
        with autocast_ctx:
            pred = self.model(x)
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
        """2.5D forward: squeeze C_res=1, run 2D model, reshape pred.

        Args:
            x: (B, 1, D, H, W) tensor produced by ``_build_z_window_input``
               under the single-resolution z-axis contract that 2.5D
               training shares.

        Returns:
            (B, num_fg, D, H, W) sigmoid probabilities, matching the 3D
            path's contract so all downstream blending stays unchanged.
        """
        if x.shape[1] != 1:
            raise ValueError(
                "2.5D inference expects single-resolution input "
                f"(C_res=1); got x.shape={tuple(x.shape)}")
        # (B, 1, D, H, W) → (B, D, H, W) — D becomes the input-channel axis.
        x_2d = x.squeeze(1)
        B, D, H, W = x_2d.shape
        autocast_ctx = autocast(
            device_type="cuda", enabled=self.use_amp, dtype=self.amp_dtype)
        with autocast_ctx:
            pred = self.model(x_2d)
            if isinstance(pred, list):
                pred = pred[0]
            expected_c = self.num_fg * D
            if pred.shape[1] != expected_c:
                raise ValueError(
                    f"2.5D model output channels {pred.shape[1]} != "
                    f"num_fg*D = {self.num_fg}*{D} = {expected_c}")
            # (B, num_fg*D, H, W) → (B, num_fg, D, H, W)
            pred_5d = pred.reshape(B, self.num_fg, D, H, W)
            prob = torch.sigmoid(pred_5d.float())

            if self.tta_flip:
                prob = self._tta_flip_ensemble_2_5d(x_2d, prob)

        return prob.float().cpu().numpy()

    def _tta_flip_ensemble(
        self, x: torch.Tensor, base_prob: torch.Tensor,
    ) -> torch.Tensor:
        """Average predictions across the 7 non-identity axis-flip
        combinations plus the original. Each flipped forward slices to
        `num_fg` and is un-flipped before accumulation, matching the
        spatial convention of `base_prob`.
        """
        total = base_prob.clone()
        count = 1.0
        for flip_dims in ([2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]):
            x_flip = torch.flip(x, flip_dims)
            pred_flip = self.model(x_flip)
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
        """2.5D TTA: flip only along H/W (the model's spatial axes).

        D is the model's input-channel axis (one channel per slice) and
        is geometrically meaningful, but flipping it would reverse the
        physical slice ordering — a distribution shift the model has
        not seen at training time. Flipping only H/W stays within the
        2D model's spatial symmetry group.

        Args:
            x_2d:      (B, D, H, W) input fed to the 2D model.
            base_prob: (B, num_fg, D, H, W) un-flipped reference output.

        Returns:
            (B, num_fg, D, H, W) average over identity + 3 flip variants.
        """
        B, D, H, W = x_2d.shape
        total = base_prob.clone()
        count = 1.0
        # x_2d axes: 2=H, 3=W; prob_5d axes: 3=H, 4=W (D inserted at 2).
        for flip_x_dims, flip_prob_dims in (
            ([2], [3]),       # H
            ([3], [4]),       # W
            ([2, 3], [3, 4])  # H + W
        ):
            x_flip = torch.flip(x_2d, flip_x_dims)
            pred_flip = self.model(x_flip)
            if isinstance(pred_flip, list):
                pred_flip = pred_flip[0]
            # (B, num_fg*D, H, W) → (B, num_fg, D, H, W)
            pred_flip_5d = pred_flip.reshape(B, self.num_fg, D, H, W)
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
        """(start, end) windows along one axis with guaranteed full coverage.
        The tail window is shifted back so it still has exactly `patch`
        voxels whenever the axis is at least `patch` long.
        """
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
        """Symmetric 1D blending window of length n, float32."""
        if mode == "gaussian" and n > 1:
            center = (n - 1) / 2.0
            sigma = max(n / 4.0, 1e-6)
            z = np.arange(n, dtype=np.float32)
            return np.exp(-0.5 * ((z - center) / sigma) ** 2).astype(np.float32)
        return np.ones(n, dtype=np.float32)

    @staticmethod
    def _build_3d_weight(pD: int, pH: int, pW: int, mode: str) -> np.ndarray:
        """Separable 3D blending weight, float32."""
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
        """Convert probability volume to integer label map.

        Channel `c` of `prob_volume` corresponds to `label_values[c + 1]`.
        For each voxel: if max fg-probability > threshold, assign the
        winning class's label value; otherwise, background.
        """
        bg_val = self.label_values[0]
        fg_values = np.array(self.label_values[1:], dtype=np.int64)
        assert len(fg_values) == self.num_fg

        max_prob = prob_volume.max(axis=0)            # (D, H, W)
        max_class = prob_volume.argmax(axis=0)        # (D, H, W)
        label_map = fg_values[max_class]
        label_map[max_prob < self.threshold] = bg_val

        # Pick the smallest signed int dtype that fits every label.
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
        """Save prediction results as NIfTI files.

        The transpose convention (2, 1, 0) mirrors `load_nifti`, which
        reorients the source volume from (X, Y, Z) to (Z, Y, X) — keep
        this inverse-consistent with the loader.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).name.replace(".nii.gz", "").replace(".nii", "")

        ref_nii = nib.load(image_path)
        affine = ref_nii.affine

        lbl_nifti = label_map.transpose(2, 1, 0)
        nib.save(nib.Nifti1Image(lbl_nifti, affine),
                 str(out_dir / f"{stem}_pred.nii.gz"))
        logger.info("Saved label map: %s", out_dir / f"{stem}_pred.nii.gz")

        if self.save_probs:
            for c in range(prob_volume.shape[0]):
                prob_nifti = prob_volume[c].transpose(2, 1, 0).astype(np.float32)
                fname = f"{stem}_prob_class{c}.nii.gz"
                nib.save(nib.Nifti1Image(prob_nifti, affine),
                         str(out_dir / fname))
            logger.info("Saved probability maps: %d classes",
                        prob_volume.shape[0])


# ==================================================================
# Checkpoint loading
# ==================================================================
def _strip_compile_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove the `_orig_mod.` prefix that `torch.compile` adds to
    state_dict keys, so a checkpoint saved from a compiled model can be
    loaded into an uncompiled model without surgery."""
    prefix = "_orig_mod."
    if any(k.startswith(prefix) for k in sd):
        return {(k[len(prefix):] if k.startswith(prefix) else k): v
                for k, v in sd.items()}
    return sd


def _select_state_dict(
    ckpt: Dict, variant: str,
) -> Tuple[Dict[str, torch.Tensor], str]:
    """Pick the right weights from a checkpoint given a variant request.

    variant:
      - "auto":   prefer EMA if present, else online.
      - "ema":    require EMA (falls back with warning if missing).
      - "online": use the online weights. For best-model checkpoints the
                  trainer writes online weights to `model_online_state_dict`
                  (because `model_state_dict` there holds EMA as primary).

    The `(sd, label)` tuple makes the choice visible in logs.
    """
    has_online = "model_online_state_dict" in ckpt
    has_ema = "ema_state_dict" in ckpt
    primary = ckpt["model_state_dict"]

    if variant == "online":
        return (ckpt["model_online_state_dict"] if has_online else primary,
                "online")
    if variant == "ema":
        if has_ema:
            return ckpt["ema_state_dict"], "ema"
        logger.warning("EMA requested but not found in checkpoint; "
                       "using online weights.")
        return (ckpt["model_online_state_dict"] if has_online else primary,
                "online")
    # auto
    if has_ema:
        return ckpt["ema_state_dict"], "ema"
    return primary, "online"


def run_inference(
    cfg: Config,
    checkpoint_path: str,
    image_paths: List[str],
    weight_variant: str = "auto",
) -> None:
    """Run inference on a list of images using a trained model.

    Args:
        cfg: Full configuration.
        checkpoint_path: Path to model checkpoint.
        image_paths: List of NIfTI file paths.
        weight_variant: "auto" | "ema" | "online". "auto" prefers EMA.
    """
    from .models.factory import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg)
    ckpt = torch.load(checkpoint_path, map_location=device)
    sd, label = _select_state_dict(ckpt, weight_variant)
    sd = _strip_compile_prefix(sd)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning("Missing keys when loading checkpoint: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading checkpoint: %s", unexpected)

    model = model.to(device).eval()
    logger.info("Model loaded from %s (variant=%s)", checkpoint_path, label)

    predictor = Predictor(model, cfg, device)

    n = len(image_paths)
    for i, path in enumerate(image_paths, 1):
        logger.info("[%d/%d] Processing: %s", i, n, path)
        try:
            result = predictor.predict_volume(
                path, output_dir=cfg.predict.output_dir)
            logger.info("  Label map shape: %s, unique labels: %s",
                        result["label_map"].shape,
                        np.unique(result["label_map"]).tolist())
        except Exception as e:
            logger.exception("Failed to process %s: %s", path, e)
            continue