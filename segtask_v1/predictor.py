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

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
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

        # ---- Z-axis interleaved 2.5D inference (TODO 1) ----------------
        # Wraps the standard 2.5D z-sliding-window path: split the volume
        # along z into ``k`` interleaved sub-volumes (stride k starting at
        # offsets 0..k-1), run the existing ``_sliding_window_z`` on each
        # independently, then weave the per-stream probabilities back into
        # the original z indices. ``k`` is selected per volume from the
        # physical z spacing (see ``_choose_interleave_factor``).
        #
        # Only activates for ``patch_mode == "2_5d"``; validated in
        # ``Config.validate``. Defaults to a no-op for legacy configs.
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
        # Default to single-resolution so empty config doesn't break the
        # downstream np.stack.
        self.multi_res_scales = cfg.data.multi_res_scales or [1.0]
        # Mirror DataConfig.z_boundary_mode so train/inference geometries
        # stay strictly consistent across the toggle. Falls back to
        # "stretch" (legacy) on stale configs that lack the field.
        self.z_boundary_mode = getattr(
            cfg.data, "z_boundary_mode", "stretch")
        if self.z_boundary_mode not in ("stretch", "edge_pad"):
            raise ValueError(
                f"Unknown z_boundary_mode {self.z_boundary_mode!r}; "
                "expected 'stretch' or 'edge_pad'.")

        # ---- Native-depth multi-FOV inference path (ON mode) ------------
        # Mirrors ``Trainer.aux_keep_native_d`` so the inference window
        # builder produces the SAME channel layout the model was trained on:
        #   (B, sum_k D_k, pH, pW) where D_k = round(pD * s_k)
        # rather than the legacy ``(B, n_views * pD, pH, pW)``. View 0
        # occupies the leading ``D_0 == pD`` channels (centered slices of
        # the max-FOV cube); aux views k=1..K-1 follow at native depth.
        # Plan A "lift" — 2.5D pipeline routed through a true-3D model.
        # When True the per-window forward SKIPS the C_res * D channel
        # collapse: the rank-5 ``(B, n_views, pD, pH, pW)`` window is fed
        # straight into the 3D UNet, whose output is ``(B, num_fg, pD,
        # pH, pW)`` — already in the contract shared with the 3D path.
        # Validated upstream by Config.validate(): only legal for
        # patch_mode=="2_5d" and mutually exclusive with aux_keep_native_d.
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

        self.aux_keep_native_d = bool(
            getattr(cfg.data, "aux_keep_native_d", False)
            and self.patch_mode == "2_5d"
            and len(self.multi_res_scales) > 1
            and not self.lift_2_5d_to_3d)  # defensive; validate also rejects
        if self.aux_keep_native_d:
            depths = [int(round(self.patch_D * float(s)))
                      for s in self.multi_res_scales]
            depths[0] = int(self.patch_D)  # s_0 == 1.0 invariant
            self.aux_view_depths: List[int] = depths
            self._eD_max: int = int(round(
                self.patch_D * float(max(self.multi_res_scales))))
            # Sanity-check vs. model's actual in_channels (set by sync()).
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

        # ---- 3D lazy single-max-FOV-cube path (R3, ON mode) ------------
        # Mirrors ``Trainer.keep_native_multi_res`` so the inference
        # window builder produces the SAME ``(B, C_res, pD, pH, pW)``
        # input contract the model was trained on, but extracts ONE
        # max-FOV cube around the window centre and crop+resizes per
        # view on the GPU instead of running K independent CPU zooms.
        # Strict gating (3D modes only with n_views > 1); 2.5D uses
        # ``aux_keep_native_d`` instead (built above).
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
        bbox_path: Optional[str] = None,
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

        Optional ROI (``bbox_path``):
            When supplied, ``bbox_path`` must point to a NIfTI mask whose
            spatial shape matches the input image. We compute the axis-
            aligned bbox of its nonzero voxels, crop the image to that
            bbox, run sliding-window / whole-volume inference inside the
            cropped sub-volume, and finally splice the prediction back
            into a full-size (D_orig, H_orig, W_orig) canvas — voxels
            outside the bbox stay at probability 0, which the standard
            ``_prob_to_label`` thresholding maps to background. This both
            preserves the source image's NIfTI affine on save and avoids
            spending compute on uninteresting regions of large CT scans.
        """
        dc = self.cfg.data
        # Only the z-interleave path needs physical z spacing; everywhere
        # else we keep the long-standing ``load_nifti`` call so the legacy
        # numerics are bit-identical.
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

        # Optional ROI cropping. We keep the (offsets, full_shape) so the
        # cropped prediction can be spliced back into the original volume
        # coordinate system before saving.
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

        if self.patch_mode == "whole":
            prob_volume = self._whole_volume_forward(vol)
        elif self.patch_mode == "cubic":
            prob_volume = self._sliding_window_cubic(vol)
        elif self.z_interleave_enabled:
            # 2.5D z-interleaved path (TODO 1). ``z_spacing`` is set
            # above for this branch; falls through to a normal k=1
            # ``_sliding_window_z`` when the spacing rule selects k<=1.
            prob_volume = self._sliding_window_z_interleaved(
                vol, float(z_spacing))
        else:
            # "z_axis" or "2_5d" — same window geometry; see _forward_batch
            # for the 2.5D-specific squeeze + reshape.
            prob_volume = self._sliding_window_z(vol)

        # Splice the cropped prediction back to the original volume's
        # spatial extent so the saved NIfTI shares the source image's
        # shape and affine. Outside-bbox voxels remain probability 0.
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
        """Pick interleave factor ``k`` from physical z spacing (mm).

        The rule, parameterised by config:
            ``z_interleave_thresholds = [t_1, ..., t_n]`` (ascending) and
            ``z_interleave_factors = [f_1, ..., f_n, f_fallback]``
            (length n+1) yields ``f_j`` for the first ``j`` with
            ``z_spacing <= t_j``, else ``f_fallback``.

        Returns ``k >= 1``. ``k == 1`` is the no-op fallback (caller is
        expected to short-circuit straight into ``_sliding_window_z``).
        """
        thresholds = self.z_interleave_thresholds
        factors = self.z_interleave_factors
        for t, f in zip(thresholds, factors):
            if z_spacing <= float(t):
                return max(1, int(f))
        return max(1, int(factors[-1]))

    def _sliding_window_z_interleaved(
        self, vol: np.ndarray, z_spacing: float,
    ) -> np.ndarray:
        """2.5D z-interleaved inference (TODO 1).

        Splits ``vol`` (shape ``(D, H, W)``) into ``k`` disjoint
        sub-volumes by stride-k slicing — ``vol[i::k]`` for
        ``i = 0..k-1`` — runs the standard 2.5D z-sliding-window
        inference on each, then weaves the per-stream
        ``(num_fg, D_i, H, W)`` probabilities back into a single
        ``(num_fg, D, H, W)`` output via ``out[:, i::k] = stream_i``.

        Streams cover the full slice index set partition (∪_i
        {i, i+k, i+2k, ...} = {0..D-1}, pairwise disjoint), so the
        recombination is exact — no cross-stream weighting needed.

        When the spacing-driven ``k == 1`` the call short-circuits to
        the legacy single-stream path. This keeps a single dispatch
        point in ``predict_volume`` without polluting hot paths with
        the ``k == 1`` no-op overhead.
        """
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
            # ``vol[i::k]`` is a view; copy to keep the rest of the
            # pipeline (which expects a contiguous (D_i, H, W) array)
            # safe against downstream in-place ops.
            sub_vol = np.ascontiguousarray(vol[i::k])
            sub_D = sub_vol.shape[0]
            logger.info(
                "  z-interleave stream %d/%d: indices=%d::%d, sub_D=%d",
                i + 1, k, i, k, sub_D)
            sub_prob = self._sliding_window_z(sub_vol)
            # Strict shape check: defensive, helps catch any future
            # change in ``_sliding_window_z`` that breaks the
            # "output depth == input depth" contract.
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

        # ----- GPU-resident pipeline ----------------------------------
        # Hot-path bottleneck before this rewrite: each window paid for
        #   (a) scipy.ndimage.zoom upsample of (num_fg, pD, pH, pW) →
        #       (num_fg, ad, H_orig, W_orig) on CPU,
        #   (b) `.cpu().numpy()` sync + numpy float32 accumulation into a
        #       (num_fg, D_orig, H_orig, W_orig) buffer.
        # For 512² volumes both are massively faster on CUDA. We keep the
        # volume + accumulators on the GPU and only move the final
        # blended probability volume back at the end.
        vol_t = torch.from_numpy(vol).to(self.device, non_blocking=True)

        z_weight_t = torch.from_numpy(
            self._build_1d_weight(pD)).to(self.device)  # (pD,)

        acc_pred = torch.zeros(
            (self.num_fg, D_orig, H_orig, W_orig),
            dtype=torch.float32, device=self.device)
        acc_weight = torch.zeros(
            (1, D_orig, 1, 1), dtype=torch.float32, device=self.device)

        # Multi-resolution z-axis is rare in practice (2.5D forces [1.0]).
        # Keep the legacy CPU build path only when scales > 1.0 are present;
        # otherwise extract directly on GPU to skip the host round-trip.
        single_res = (len(self.multi_res_scales) == 1
                      and self.multi_res_scales[0] == 1.0)
        # 3D ON path: GPU builder returns (C_res, pD, pH, pW) ready to
        # stack into (B, C_res, pD, pH, pW) — same layout as legacy
        # multi-res z_axis, so downstream forward / blending paths are
        # bit-identical. Mutually exclusive with ``aux_keep_native_d``
        # (2.5D analogue).
        keep_native_3d = bool(self.keep_native_multi_res
                              and self.patch_mode == "z_axis")

        # ON-mode native-depth path: builder returns rank-3
        # ``(in_channels=sum(D_k), pH, pW)`` so ``torch.stack`` produces
        # ``(B, in_channels, pH, pW)`` directly — exactly the layout the
        # 2D model was trained on (no reshape needed in _forward_batch_gpu).
        # Legacy paths keep the rank-4 ``(C_res, pD, pH, pW)`` window shape.
        window_inputs: List[torch.Tensor] = []
        patch_metas: List[Tuple[int, int, int]] = []  # (z0, z1, actual_d)

        n_windows = len(z_positions)
        for idx, (z0, z1) in enumerate(z_positions):
            actual_d = z1 - z0
            if self.aux_keep_native_d:
                # 2.5D ON mode: rank-3 (sum(D_k), pH, pW).
                window_inputs.append(
                    self._build_z_window_input_native_d_gpu(vol_t, z0, z1))
            elif keep_native_3d:
                # 3D ON mode: rank-4 (C_res, pD, pH, pW), same layout
                # as the legacy multi-res builder — drop into the same
                # ``torch.stack`` collate at the bottom of the loop.
                window_inputs.append(
                    self._build_z_window_input_native_multi_res_gpu(
                        vol_t, z0, z1))
            elif single_res:
                window_inputs.append(
                    self._build_z_window_input_gpu(vol_t, z0, z1))
            else:
                # Fallback: legacy multi-res builder runs on CPU then
                # ships once to GPU. Keeps multi-FOV semantics identical.
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

                # Group by actual_d so windows that share a target depth
                # can share one F.interpolate launch. In the common case
                # all but possibly the last window have ad == pD, which
                # collapses this to a single batched upsample.
                groups: Dict[int, List[int]] = {}
                for i, (_, _, ad) in enumerate(patch_metas):
                    groups.setdefault(ad, []).append(i)

                for ad, idxs in groups.items():
                    sub = probs[idxs]  # (b, num_fg, pD, pH, pW)

                    # Reverse-resize to the original spatial geometry.
                    # Two paths:
                    #   - "edge_pad" + ad < pD: forward saw an exactly-pD
                    #     replicate-padded input, so the prediction is also
                    #     pD slices deep. The valid central ``ad`` slices
                    #     (matching the original ``vol[z0:z1]`` extent)
                    #     are sliced out — NO depth interpolation. H/W
                    #     are still resized back to (H_orig, W_orig).
                    #   - "stretch" (or ad == pD): legacy single-shot
                    #     trilinear resize to (ad, H_orig, W_orig).
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
                    # Per-ad blending weight (symmetric on actual depth).
                    if ad == pD:
                        w = z_weight_t
                    else:
                        w = torch.from_numpy(
                            self._build_1d_weight(ad)).to(self.device)
                    w_4d = w.view(1, -1, 1, 1)

                    for j, i in enumerate(idxs):
                        zs, ze, _ = patch_metas[i]
                        # In-place fused mul-add (one GPU kernel each).
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
        """GPU equivalent of ``_build_z_window_input`` for the single-res
        path (the only path 2.5D mode ever uses).

        Window geometry depends on ``self.z_boundary_mode``:

        ``stretch`` (legacy, backward compatible)
            Take ``vol[z0:z1]`` as-is (possibly fewer than pD slices when
            ``D_orig < pD``) and trilinear-resize the whole tensor to
            ``(pD, pH, pW)``. Boundary windows are stretched along z.

        ``edge_pad``
            When ``z1 - z0 < pD``, edge-replicate-pad along z symmetrically
            up to ``pD`` slices BEFORE any resize, so every slice in the
            output corresponds to a physical 1-slice spacing. This mirrors
            ``extract_z_patch_padded(vol, z_center, pD)`` used by the
            multi-resolution / training paths under the same toggle.

        Returns ``(C_res=1, pD, pH, pW)`` float32 on the model's device.
        """
        pD, pH, pW = self.patch_D, self.patch_H, self.patch_W
        patch = vol_t[z0:z1]  # (actual_d, H_orig, W_orig)
        ad, H, W = patch.shape

        if self.z_boundary_mode == "edge_pad" and ad < pD:
            # Centred replicate-pad along the z axis. The split mirrors
            # ``extract_z_patch_padded`` so train/inference share the same
            # symmetry contract: the centre of the output corresponds to
            # the centre of the in-bounds slab.
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

        # F.interpolate expects (N, C, D, H, W); add batch+channel.
        patch = patch.unsqueeze(0).unsqueeze(0).float()
        if (ad != pD) or (H != pH) or (W != pW):
            patch = F.interpolate(
                patch, size=(pD, pH, pW),
                mode="trilinear", align_corners=False)
        return patch.squeeze(0)  # (1, pD, pH, pW)

    def _build_z_window_input_native_multi_res_gpu(
        self, vol_t: torch.Tensor, z0: int, z1: int) -> torch.Tensor:
        """3D z_axis ON-mode (``keep_native_multi_res=True``) window builder.

        Mirrors :meth:`SegDataset3D._getitem_native_multi_res_z` and the
        trainer's :meth:`_split_views_native_3d` so inference and training
        share one geometry contract end-to-end:

          1. Extract a SINGLE max-FOV cube of depth ``self._mr_target_shape[0]``
             centred on ``(z0+z1)//2`` (edge-replicate padded along z).
          2. In-plane resize to (pH, pW); D axis stays at the max-FOV depth.
          3. For each view k=0..K-1: center-crop ``D_k = round(pD*s_k)``
             slices and ``F.interpolate`` (trilinear) the D axis back to
             ``pD``. View 0 takes the centred ``pD`` slices with no resize.
          4. Stack views along the channel axis →
             ``(C_res, pD, pH, pW)`` — exactly the legacy 3D z_axis
             multi-res contract that the model expects.

        Returns rank-4 ``(C_res, pD, pH, pW)`` float32 on the model's
        device. Stacking across windows yields the model's input batch
        directly, no further reshape required.
        """
        pD, pH, pW = self.patch_D, self.patch_H, self.patch_W
        eD_max, eH_max, eW_max = self._mr_target_shape   # eH_max=pH, eW_max=pW
        z_center = (z0 + z1) // 2
        D_vol = vol_t.shape[0]

        # ---- Edge-padded extraction of an eD_max-deep slab --------------
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

        # ---- In-plane resize (D axis preserved at eD_max) ---------------
        H_orig, W_orig = slab.shape[1], slab.shape[2]
        slab = slab.unsqueeze(0).unsqueeze(0).float()  # (1,1,eD_max,H,W)
        if H_orig != pH or W_orig != pW:
            slab = F.interpolate(
                slab, size=(eD_max, pH, pW),
                mode="trilinear", align_corners=False)
        # slab now (1, 1, eD_max, pH, pW)

        # ---- Per-view crop + D-axis resize, stack as C_res channels ----
        view_chunks: List[torch.Tensor] = []
        for D_k, _, _ in self._mr_native_sizes:
            d0 = (eD_max - D_k) // 2
            crop = slab[:, :, d0:d0 + D_k, :, :]  # (1, 1, D_k, pH, pW)
            if D_k != pD:
                crop = F.interpolate(
                    crop, size=(pD, pH, pW),
                    mode="trilinear", align_corners=False)
            view_chunks.append(crop[0])  # (1, pD, pH, pW)
        # stack(dim=0) of K (1, pD, pH, pW) → (K, 1, pD, pH, pW)? No —
        # we want one channel per view, so squeeze the singleton C
        # before stacking on dim=0.
        return torch.cat(view_chunks, dim=0).contiguous()  # (C_res, pD, pH, pW)

    def _build_z_window_input_native_d_gpu(
        self, vol_t: torch.Tensor, z0: int, z1: int) -> torch.Tensor:
        """ON-mode (aux_keep_native_d=True) window builder.

        Mirrors :meth:`SegDataset3D._getitem_native_d` and the trainer's
        :meth:`_split_views_native_d` so inference and training share one
        channel layout end-to-end:

          1. Extract a SINGLE max-FOV cube of depth ``self._eD_max``
             centred on ``(z0+z1)//2``. ``edge_pad`` is unconditional —
             aux_keep_native_d implies edge_pad (Config validates this).
          2. In-plane resize to (pH, pW); D axis stays at ``eD_max``
             (i.e. the native physical span of the widest FOV).
          3. Per-view center crop along the D axis: view k takes the
             centred ``D_k = round(pD * s_k)`` slices. View 0 takes the
             centred ``D_0 == pD`` slices (== the legacy single-FOV view).
          4. Concatenate views along the channel axis →
             ``(sum_k D_k, pH, pW)`` — exactly the layout the model
             expects after the trainer's ``_split_views_native_d`` step.

        Returns rank-3 ``(in_channels, pH, pW)`` float32 on the model's
        device. Stacking across windows yields the model's input batch
        directly, no reshape required.
        """
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
        """GPU-tensor-returning variant of ``_forward_batch``.

        Used by the GPU-resident z-axis path so prediction probabilities
        never round-trip through host memory between forward and the
        final blend / argmax. Behaviour is identical to ``_forward_batch``
        otherwise (AMP, TTA, deep-supervision unwrap).
        """
        if self.patch_mode == "2_5d":
            # Plan A lift: feed the rank-5 ``(B, n_views, pD, pH, pW)``
            # window straight into the 3D UNet. Output is already
            # ``(B, num_fg, pD, pH, pW)`` — shape contract identical to
            # the 3D path below, so reshape-on-collapse is unnecessary.
            # TTA reuses the 3D ensemble (D is now a real spatial axis
            # so flipping it is geometrically valid; the model has seen
            # D-flips at training time via random_flip_axes=[2,3,4]).
            if self.lift_2_5d_to_3d:
                if x.ndim != 5:
                    raise ValueError(
                        "lift_2_5d_to_3d=True expects rank-5 input "
                        f"(B, n_views, D, H, W); got x.shape={tuple(x.shape)}")
                with autocast(device_type="cuda", enabled=self.use_amp,
                              dtype=self.amp_dtype):
                    pred = self.model(x)
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
                return prob

            # Two input layouts arriving here:
            #   * Legacy / OFF path: rank-5 (B, C_res, pD, pH, pW) — needs
            #     the C_res-collapse reshape.
            #   * ON path (aux_keep_native_d=True): rank-4 (B, sum(D_k), H, W)
            #     produced by ``_build_z_window_input_native_d_gpu``;
            #     already in the model's input channel layout — pass
            #     through.
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
                pred = self.model(x_2d)
                if isinstance(pred, list):
                    pred = pred[0]
                expected_c = self.num_fg * D
                if pred.shape[1] != expected_c:
                    raise ValueError(
                        f"2.5D model output channels {pred.shape[1]} != "
                        f"num_fg*D = {self.num_fg}*{D} = {expected_c}")
                pred_5d = pred.reshape(B, self.num_fg, D, H, W)
                prob = torch.sigmoid(pred_5d.float())
                if self.tta_flip:
                    prob = self._tta_flip_ensemble_2_5d(x_2d, prob)
            return prob

        with autocast(device_type="cuda", enabled=self.use_amp,
                      dtype=self.amp_dtype):
            pred = self.model(x)
            if isinstance(pred, list):
                pred = pred[0]
            assert pred.shape[1] >= self.num_fg, (
                f"Model output has {pred.shape[1]} channels; "
                f"expected at least num_fg={self.num_fg} at 1x resolution.")
            prob = torch.sigmoid(pred.float())[:, :self.num_fg]
            if self.tta_flip:
                prob = self._tta_flip_ensemble(x, prob)
        return prob

    def _build_z_window_input(
        self, vol: np.ndarray, z0: int, z1: int) -> np.ndarray:
        """Build the multi-scale input stack for one z-sliding window.

        For ``scale > 1.0`` we ALWAYS extract ``round(pD * scale)`` slices
        centred on the window's z-center with edge-replicate padding, so
        the physical z-FOV stays proportional to ``scale`` even when the
        window touches the volume boundary.

        For ``scale == 1.0`` the path depends on ``self.z_boundary_mode``:
          * "stretch"  — legacy: take ``vol[z0:z1]`` (possibly shorter
            than pD at the tail) and trilinear-resize to ``(pD, pH, pW)``.
            Boundary windows are stretched along z.
          * "edge_pad" — centred replicate-pad to exactly pD slices via
            ``extract_z_patch_padded(vol, z_center, pD)``, matching the
            scale > 1.0 contract.

        Returns:
            ``(C_res, pD, pH, pW)`` float32 — one channel per scale, in
            the same order as ``self.multi_res_scales``.
        """
        pD, pH, pW = self.patch_D, self.patch_H, self.patch_W
        z_center = (z0 + z1) // 2
        channels: List[np.ndarray] = []
        for scale in self.multi_res_scales:
            if scale == 1.0:
                if self.z_boundary_mode == "edge_pad":
                    # Same edge-replicate semantics as scale > 1.0.
                    patch = extract_z_patch_padded(vol, z_center, pD)
                else:
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

        # 3D cubic ON path: ship the volume to GPU once and let the
        # builder do max-FOV extraction + per-view crop+resize entirely
        # on-device (one F.interpolate per view, zero scipy.ndimage.zoom
        # calls). OFF path keeps the legacy CPU pipeline.
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
    def _build_batch_native_multi_res_cubic_gpu(
        self,
        centers: List[Tuple[int, int, int]],
        vol_t: torch.Tensor,
    ) -> torch.Tensor:
        """3D cubic ON-mode (``keep_native_multi_res=True``) batch builder.

        Mirrors :meth:`SegDataset3DCubic._getitem_native_multi_res_cubic`
        and the trainer's :meth:`_split_views_native_3d` so inference and
        training share one geometry contract end-to-end:

          1. For each window centre, extract ONE max-FOV cube of size
             ``self._mr_target_shape`` (= round(patch_size * max_scale))
             with edge-replicate padding on every axis.
          2. Per view k=0..K-1: center-crop the native size
             ``(D_k, H_k, W_k) = round(patch_size * s_k)`` and trilinear-
             ``F.interpolate`` the crop back to ``(pD, pH, pW)``. View 0
             takes the centred ``patch_size`` crop with no resize.
          3. Stack views along the channel axis →
             ``(B, C_res, pD, pH, pW)`` — exactly the legacy 3D cubic
             multi-res contract that the model expects.

        Differences from the OFF-path :meth:`_build_batch_multi_res`:
          * Single max-FOV cube extraction per centre instead of K
            independent ``_extract_cubic_patch`` calls.
          * Resampling done once on GPU via ``F.interpolate`` instead of
            K ``scipy.ndimage.zoom`` calls on CPU.
          * Edge replication everywhere — uniform with the training-side
            extractor under the same flag.

        Returns ``(B, C_res, pD, pH, pW)`` float32 on the model's device.
        """
        pD, pH, pW = self.patch_D, self.patch_H, self.patch_W
        tD, tH, tW = self._mr_target_shape
        D_vol, H_vol, W_vol = vol_t.shape

        def _edge_pad_axis(t: torch.Tensor, axis: int,
                            pad_before: int, pad_after: int
                            ) -> torch.Tensor:
            """Replicate-pad ``t`` along ``axis`` (zero-copy via expand).

            ``narrow(axis, 0, 1)`` keeps the size-1 boundary slice; we
            then ``expand`` along that axis to the requested width and
            ``cat`` everything together. ``cat`` is the only allocation.
            """
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
            # Edge-padded extraction, axis-by-axis to avoid materialising
            # the full padded slab when only one axis needs padding.
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

            # ---- Per-view crop + resize, stack as C_res channels --------
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
        """2.5D forward: collapse C_res, run 2D model, reshape pred.

        Args:
            x: (B, C_res, D, H, W) tensor produced by
               ``_build_z_window_input``. ``C_res >= 1`` — view 0 is the
               1× FOV (true geometry; supervision target during training)
               and views 1..K are wider z-FOVs each resampled back to D
               channels (see SegDataset3D z-axis multi-res contract).

        Returns:
            (B, num_fg, D, H, W) sigmoid probabilities at the 1× geometry,
            matching the 3D path's contract so all downstream blending
            stays unchanged.
        """
        # Plan A lift (CPU-returning variant): same logic as the GPU
        # branch in ``_forward_batch_gpu`` — skip the C_res*D collapse
        # and feed the rank-5 input straight to the 3D model. Output
        # shape ``(B, num_fg, D, H, W)`` already matches the contract
        # required by the caller.
        if self.lift_2_5d_to_3d:
            if x.ndim != 5:
                raise ValueError(
                    "lift_2_5d_to_3d=True expects rank-5 input "
                    f"(B, n_views, D, H, W); got x.shape={tuple(x.shape)}")
            autocast_ctx = autocast(
                device_type="cuda", enabled=self.use_amp,
                dtype=self.amp_dtype)
            with autocast_ctx:
                pred = self.model(x)
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

    def _reshape_2_5d_input(self, x: torch.Tensor) -> torch.Tensor:
        """Collapse the C_res (multi-FOV) axis for the 2D model input.

        Mirrors ``Trainer._squeeze_2_5d`` exactly so train and inference
        share one channel-layout contract:

          (B, C_res, D, H, W) → (B, C_res * D, H, W)

        With ``C_res == 1`` this collapses to the legacy ``squeeze(1)``
        and is bit-identical to single-FOV inference. The downstream
        2D model's stem (``Encoder.stem``) is responsible for splitting
        the C_res*D channels back into per-view chunks of size D when
        ``MultiStemProj`` fusion is in use.
        """
        if x.ndim != 5:
            raise ValueError(
                "2.5D inference expects rank-5 input "
                f"(B, C_res, D, H, W); got x.shape={tuple(x.shape)}")
        B, C_res, D, H, W = x.shape
        if D != self.patch_D:
            raise ValueError(
                f"2.5D input D-axis ({D}) != patch_D ({self.patch_D}). "
                "Window builder produced an unexpected slice count.")
        return x.reshape(B, C_res * D, H, W).contiguous()

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
            x_2d:      (B, C_res*D, H, W) input fed to the 2D model.
                       For multi-FOV (C_res > 1) all view slabs are
                       flipped together — they share the same H/W spatial
                       grid by construction.
            base_prob: (B, num_fg, D, H, W) un-flipped reference output
                       at the 1× geometry.

        Returns:
            (B, num_fg, D, H, W) average over identity + 3 flip variants.
        """
        B, _, H, W = x_2d.shape
        D = self.patch_D
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
        """Save prediction results as NIfTI files via SimpleITK.

        Spatial metadata (origin / spacing / direction) is copied from
        the source image so the saved volumes overlay the input
        perfectly. SimpleITK's ``GetImageFromArray`` accepts arrays in
        ``(Z, Y, X) == (D, H, W)`` order — i.e. the same layout
        ``load_nifti`` returns — so no transpose is needed (mirroring
        the no-transpose-on-read contract of the loader). All outputs
        are gzip-compressed via ``useCompression=True``.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).name.replace(".nii.gz", "").replace(".nii", "")

        # Read just the header (no pixel decode) by reading native and
        # discarding the array — SimpleITK still loads pixels here, but
        # that cost is dwarfed by the inference itself. Keeping it
        # simple via ReadImage ensures origin/spacing/direction stay
        # exactly in sync with the loader contract.
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
    """Remove the `_orig_mod.` prefix that `torch.compile` adds to
    state_dict keys, so a checkpoint saved from a compiled model can be
    loaded into an uncompiled model without surgery."""
    prefix = "_orig_mod."
    if any(k.startswith(prefix) for k in sd):
        return {(k[len(prefix):] if k.startswith(prefix) else k): v
                for k, v in sd.items()}
    return sd


def _unwrap_ema_state(ema_sd: Dict) -> Dict[str, torch.Tensor]:
    """Unwrap ``ModelEMA.state_dict()`` into a plain model state_dict.

    ``ModelEMA.state_dict`` returns ``{"shadow": {...weights...},
    "decay": float}``. Feeding that directly to ``model.load_state_dict``
    silently leaves every parameter at its random init (every key is
    "missing", `shadow`/`decay` are "unexpected") — which manifests as
    perfect train Dice but garbage predictions.

    Best-model checkpoints additionally store EMA-as-primary in
    ``model_state_dict`` (already unwrapped); this helper is only invoked
    on ``ckpt["ema_state_dict"]`` and tolerates the unwrapped legacy form
    too.
    """
    if isinstance(ema_sd, dict) and "shadow" in ema_sd and isinstance(
            ema_sd["shadow"], dict):
        return ema_sd["shadow"]
    return ema_sd  # already a plain state_dict (legacy format)


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
            return _unwrap_ema_state(ckpt["ema_state_dict"]), "ema"
        logger.warning("EMA requested but not found in checkpoint; "
                       "using online weights.")
        return (ckpt["model_online_state_dict"] if has_online else primary,
                "online")
    # auto
    if has_ema:
        return _unwrap_ema_state(ckpt["ema_state_dict"]), "ema"
    return primary, "online"


def run_inference(
    cfg: Config,
    checkpoint_path: str,
    image_paths: List[str],
    weight_variant: str = "auto",
    bbox_paths: Optional[List[str]] = None,
) -> None:
    """Run inference on a list of images using a trained model.

    Args:
        cfg: Full configuration.
        checkpoint_path: Path to model checkpoint.
        image_paths: List of NIfTI file paths.
        weight_variant: "auto" | "ema" | "online". "auto" prefers EMA.
        bbox_paths: Optional ROI bbox NIfTI paths aligned 1:1 with
            ``image_paths``. When supplied, each prediction is computed
            inside the bbox and the output is written back into the full
            volume's coordinate system. Pass ``None`` (default) to keep
            full-volume inference.
    """
    if bbox_paths is not None and len(bbox_paths) != len(image_paths):
        raise ValueError(
            f"bbox_paths length {len(bbox_paths)} != image_paths "
            f"length {len(image_paths)}")
    from .models.factory import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg)
    # weights_only=False: checkpoint is written by our trainer (trusted)
    # and contains non-tensor payloads (Config, numpy RNG state) that
    # PyTorch 2.6+'s default weights_only=True refuses.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd, label = _select_state_dict(ckpt, weight_variant)
    sd = _strip_compile_prefix(sd)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning("Missing keys when loading checkpoint: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading checkpoint: %s", unexpected)
    # Hard-fail when the checkpoint contributed essentially nothing to the
    # model: a near-empty load means the model is still at random init and
    # every prediction would be garbage. This is the failure mode caused by
    # treating ModelEMA's wrapped {"shadow", "decay"} dict as a state_dict.
    n_total = len(model.state_dict())
    n_loaded = n_total - len(missing)
    if n_total > 0 and n_loaded < max(1, n_total // 2):
        raise RuntimeError(
            f"Only {n_loaded}/{n_total} parameters loaded from "
            f"{checkpoint_path} (variant={label}). The checkpoint key "
            f"layout does not match the model — refusing to predict with "
            f"random weights. Unexpected keys: {unexpected[:8]}")

    model = model.to(device).eval()
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