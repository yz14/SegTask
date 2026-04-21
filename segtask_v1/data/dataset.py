"""3D segmentation datasets.

Two patch extraction modes:
  - SegDataset3D ("z_axis"): slide along z, extract D slices, resize H,W
  - SegDataset3DCubic ("cubic"): sample center (x,y,z), extract 3D cube

Both share common I/O, preprocessing, and caching via module-level functions.

Each foreground class gets its own binary channel:
  label_values = [0, 1, 2] → output has 2 channels (class 1, class 2)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Volume I/O
# ---------------------------------------------------------------------------
def load_nifti(path: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """Load NIfTI file → (D, H, W) numpy array.

    NIfTI convention is (X, Y, Z); we transpose to (Z, Y, X) = (D, H, W)
    so that D (axial slices) is the first axis.
    """
    data = nib.load(path).get_fdata().astype(dtype)
    if data.ndim == 3:
        data = data.transpose(2, 1, 0)  # (X,Y,Z) → (D,H,W)
    return data


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_image(
    volume: np.ndarray,
    intensity_min: float,
    intensity_max: float,
    normalize: str,
    global_mean: float = 0.0,
    global_std: float = 1.0) -> np.ndarray:
    """Intensity windowing + normalization → float32."""
    vol = np.clip(volume, intensity_min, intensity_max)
    if normalize == "minmax":
        denom = intensity_max - intensity_min
        vol = (vol - intensity_min) / denom if denom > 0 else vol * 0
    elif normalize == "zscore":
        vol = (vol - global_mean) / global_std if global_std > 0 else vol * 0
    else:
        raise ValueError(f"Unknown normalize: {normalize}")
    return vol.astype(np.float32)


def compute_region_weight_map(
    volume: np.ndarray,
    label_values: List[int],
    region_weights: List[float]) -> np.ndarray:
    """Generate per-voxel spatial weight map from raw label and region weights.

    Args:
        volume: Integer label volume (D, H, W).
        label_values: [bg, fg1, fg2, ...] — all label values in the mask.
        region_weights: One weight per label value, same length as label_values.
            E.g. label_values=[0,1,2], region_weights=[1.0, 2.0, 1.5]

    Returns:
        Weight map (1, D, H, W) float32. Voxels not matching any label get weight 1.0.
    """
    vol = np.round(volume).astype(np.int32)
    wmap = np.ones_like(vol, dtype=np.float32)
    for lv, w in zip(label_values, region_weights):
        wmap[vol == lv] = w
    return wmap[np.newaxis]  # (1, D, H, W)


def preprocess_label(volume: np.ndarray, label_values: List[int]) -> np.ndarray:
    """Convert integer label → per-foreground-class binary masks.

    Args:
        volume: Integer label volume of shape (D, H, W).
        label_values: [bg, fg1, fg2, ...]. Background (index 0) is excluded.

    Returns:
        Binary masks (num_fg, D, H, W) — one channel per foreground class.
    """
    vol = np.round(volume).astype(np.int32)
    fg_values = label_values[1:]  # exclude background
    # Vectorized: (C, 1, 1, 1) == (D, H, W) → (C, D, H, W)
    lv = np.array(fg_values, dtype=np.int32).reshape(-1, *([1] * vol.ndim))
    return (vol[np.newaxis] == lv).astype(np.float32)


# ---------------------------------------------------------------------------
# Resize helpers
# ---------------------------------------------------------------------------
def resize_3d(arr: np.ndarray, target_d: int, target_h: int, target_w: int, is_label: bool = False) -> np.ndarray:
    """Resize (D, H, W) or (C, D, H, W) to target shape.

    Uses order=1 (linear) for images, order=0 (nearest) for labels.
    """
    if arr.ndim == 3:
        D, H, W = arr.shape
        if D == target_d and H == target_h and W == target_w:
            return arr
        factors = [target_d / D, target_h / H, target_w / W]
    elif arr.ndim == 4:
        _, D, H, W = arr.shape
        if D == target_d and H == target_h and W == target_w:
            return arr
        factors = [1.0, target_d / D, target_h / H, target_w / W]
    else:
        raise ValueError(f"Expected 3D or 4D array, got {arr.ndim}D")
    order = 0 if is_label else 1
    return zoom(arr, factors, order=order).astype(arr.dtype)


# ---------------------------------------------------------------------------
# Volume cache
# ---------------------------------------------------------------------------
class VolumeCache:
    """Simple in-memory cache for loaded volumes."""

    def __init__(self, enabled: bool = False):
        self._enabled = enabled
        self._store: Dict[str, np.ndarray] = {}

    def get(self, path: str) -> Optional[np.ndarray]:
        return self._store.get(path) if self._enabled else None

    def put(self, path: str, data: np.ndarray) -> None:
        if self._enabled:
            self._store[path] = data

    @property
    def size(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# 3D Segmentation Dataset
# ---------------------------------------------------------------------------
class SegDataset3D(Dataset):
    """3D patch-based segmentation dataset.

    Z-axis patching strategy:
      - Select a center z-position (with foreground oversampling)
      - Extract D slices centered at z (shift window at boundaries)
      - If volume depth < D: extract all slices, pad to D with zeros
      - Resample the patch to (D, H, W) via 3D zoom

    Each sample returns:
      image: (1, D, H, W) float32
      label: (num_fg, D, H, W) float32 binary masks
    """

    def __init__(
        self,
        image_paths: List[str],
        label_paths: List[str],
        label_values: List[int],
        patch_size: Tuple[int, int, int] = (64, 128, 128),
        intensity_min: float = -1024.0,
        intensity_max: float = 3071.0,
        normalize: str = "minmax",
        global_mean: float = 0.0,
        global_std: float = 1.0,
        foreground_oversample_ratio: float = 0.5,
        samples_per_volume: int = 8,
        is_train: bool = True,
        cache_enabled: bool = True,
        region_weights: Optional[List[float]] = None):
        super().__init__()
        assert len(image_paths) == len(label_paths)
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.label_values = label_values
        self.patch_size = tuple(patch_size)
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max
        self.normalize = normalize
        self.global_mean = global_mean
        self.global_std = global_std
        self.fg_ratio = foreground_oversample_ratio
        self.samples_per_volume = samples_per_volume
        self.is_train = is_train
        self.region_weights = region_weights

        self._img_cache = VolumeCache(cache_enabled)
        self._lbl_cache = VolumeCache(cache_enabled)

        # Build per-slice index for foreground oversampling
        self._vol_fg_slices: List[np.ndarray] = []  # fg slice indices per volume
        self._vol_all_slices: List[int] = []        # total depth per volume
        self._build_index()                         # D维度前景坐标

    def _build_index(self) -> None:
        """Scan all volumes and record which slices have foreground."""
        logger.info("Building dataset index for %d volumes...", len(self.image_paths))
        total_fg = 0
        total_slices = 0
        for i in range(len(self.image_paths)):
            lbl = self._load_label(i)
            D = lbl.shape[0]
            self._vol_all_slices.append(D)
            # A slice has foreground if any non-background label present
            bg_val = self.label_values[0]
            lbl_int = np.round(lbl).astype(np.int32)
            # Per-slice foreground check: vectorized over H,W
            fg_mask = np.any(lbl_int != bg_val, axis=(1, 2))  # (D,)
            fg_indices = np.where(fg_mask)[0]
            self._vol_fg_slices.append(fg_indices)
            total_fg += len(fg_indices)
            total_slices += D
        logger.info("Index built: %d volumes, %d/%d foreground slices",
                     len(self.image_paths), total_fg, total_slices)

    def _load_image(self, vol_idx: int) -> np.ndarray:
        """Load and preprocess image volume with caching."""
        path   = self.image_paths[vol_idx]
        cached = self._img_cache.get(path)
        if cached is not None:
            return cached
        img = load_nifti(path)
        img = preprocess_image(  # 归一化
            img, self.intensity_min, self.intensity_max,
            self.normalize, self.global_mean, self.global_std)
        self._img_cache.put(path, img)
        return img

    def _load_label(self, vol_idx: int) -> np.ndarray:
        """Load raw label volume with caching."""
        path   = self.label_paths[vol_idx]
        cached = self._lbl_cache.get(path)
        if cached is not None:
            return cached
        lbl = load_nifti(path)
        self._lbl_cache.put(path, lbl)
        return lbl

    def __len__(self) -> int:
        return len(self.image_paths) * self.samples_per_volume

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        vol_idx  = idx % len(self.image_paths)
        img, lbl = self._load_image(vol_idx), self._load_label(vol_idx)
        D_vol    = img.shape[0]
        D_patch, H_patch, W_patch = self.patch_size

        # Select center z-position, z轴中心坐标
        z = self._sample_z(vol_idx, D_vol)

        # Extract D_patch slices centered at z, TODO 这里可以改为2*D_patch，然后在训练中aug后取中间D_patch
        img_patch, lbl_patch = self._extract_z_patch(img, lbl, z, D_patch)

        # 3D resample to target (D_patch, H_patch, W_patch)
        img_patch = resize_3d(img_patch, D_patch, H_patch, W_patch, is_label=False)
        lbl_patch = resize_3d(lbl_patch, D_patch, H_patch, W_patch, is_label=True)

        # Return RAW integer label as a single channel — matches the cubic
        # dataset contract (C_res, D, H, W). Binarization is performed by
        # MultiResolutionLoss._label_to_binary at loss time, which keeps the
        # label pipeline identical across z_axis / cubic modes and avoids
        # the previous shape mismatch that silently zeroed-out class≥2 targets.
        result = {
            "image": torch.from_numpy(img_patch[np.newaxis]).float(),  # (1, D, H, W)
            "label": torch.from_numpy(lbl_patch[np.newaxis]).float()}  # (1, D, H, W)

        # Spatial region weight map (optional)
        if self.region_weights:
            wmap = compute_region_weight_map(lbl_patch, self.label_values, self.region_weights)
            result["weight_map"] = torch.from_numpy(wmap).float()  # (1, D, H, W)

        return result

    def _sample_z(self, vol_idx: int, D_vol: int) -> int:
        """Sample a center z-position with optional foreground oversampling."""
        fg_slices = self._vol_fg_slices[vol_idx]
        if (self.is_train
            and self.fg_ratio > 0
            and len(fg_slices) > 0
            and np.random.random() < self.fg_ratio):
            return int(np.random.choice(fg_slices))
        return np.random.randint(0, D_vol)

    def _extract_z_patch(
        self, img: np.ndarray, lbl: np.ndarray, z_center: int, D_patch: int) -> Tuple[np.ndarray, np.ndarray]:
        """Extract D_patch slices from z-axis, centered at z_center.

        If volume depth < D_patch: take all slices, pad with zeros.
        If volume depth >= D_patch: sliding window clamped to bounds.
        """
        D_vol = img.shape[0]
        half  = D_patch // 2
        # Clamp to volume bounds
        d_start = max(0, z_center - half)
        d_end   = min(D_vol, d_start + D_patch)

        img_patch = img[d_start:d_end]
        lbl_patch = lbl[d_start:d_end]

        return img_patch.copy(), lbl_patch.copy()


# ---------------------------------------------------------------------------
# 3D Cubic Patch Dataset
# ---------------------------------------------------------------------------
def _extract_cubic_patch(
    vol: np.ndarray, center: Tuple[int, int, int], size: Tuple[int, int, int]) -> np.ndarray:
    """Extract a cubic patch centered at (d, h, w), with zero-padding if needed.

    Args:
        vol: (D, H, W) volume.
        center: (d, h, w) center coordinates.
        size: (pD, pH, pW) patch size to extract.

    Returns:
        Patch of exactly (pD, pH, pW), zero-padded where out of bounds.
    """
    D, H, W    = vol.shape
    pD, pH, pW = size
    cd, ch, cw = center

    # Compute start/end for each axis
    starts, ends, pad_before, pad_after = [], [], [], []
    for c, p, s in [(cd, pD, D), (ch, pH, H), (cw, pW, W)]:
        half = p // 2
        lo = c - half
        hi = lo + p
        # Clamp to volume bounds and compute padding
        src_lo = max(lo, 0)
        src_hi = min(hi, s)
        starts.append(src_lo)
        ends.append(src_hi)
        pad_before.append(max(-lo, 0))
        pad_after.append(max(hi - s, 0))

    patch = vol[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]

    # Pad to the exact requested `size` whenever the cube extended beyond
    # volume bounds. Without this, an off-boundary center returns a smaller
    # cube that downstream `resize_3d` stretches non-uniformly, producing
    # anisotropic distortion (severely biased proportions for fg voxels
    # located near the volume edges).
    #
    # `mode="edge"` replicates the nearest boundary voxel — consistent with
    # the inference-time padding used in `predictor._sliding_window_cubic`
    # (when `pad_value` is not configured) and avoids introducing "air"
    # artefacts for non-zero-normalized intensities.
    if any(pb > 0 or pa > 0 for pb, pa in zip(pad_before, pad_after)):
        patch = np.pad(
            patch,
            list(zip(pad_before, pad_after)),
            mode="edge")

    return patch


class SegDataset3DCubic(Dataset):
    """3D cubic patch dataset.

    Samples a center point (d, h, w) and extracts a full 3D cube.

    Features:
      - Augmentation oversample: extract larger cube, trainer crops after aug.
      - Multi-resolution input: extract multiple scales at same center,
        resize to same size, stack as channels.

    Output format depends on multi_res_scales:
      - Disabled (empty): image (1, eD, eH, eW), label (num_fg, eD, eH, eW)
      - Enabled:          image (C_res, eD, eH, eW), label (C_res, eD, eH, eW)
        where label channels are RAW integer labels (preprocess_label at loss time).
    """

    def __init__(
        self,
        image_paths: List[str],
        label_paths: List[str],
        label_values: List[int],
        patch_size: Tuple[int, int, int] = (64, 128, 128),
        aug_oversample_ratio: float = 1.0,
        multi_res_scales: Optional[List[float]] = None,
        intensity_min: float = -1024.0,
        intensity_max: float = 3071.0,
        normalize: str = "minmax",
        global_mean: float = 0.0,
        global_std: float = 1.0,
        foreground_oversample_ratio: float = 0.5,
        samples_per_volume: int = 8,
        is_train: bool = True,
        cache_enabled: bool = True,
        region_weights: Optional[List[float]] = None):
        super().__init__()
        assert len(image_paths) == len(label_paths)
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.label_values = label_values
        self.patch_size = tuple(patch_size)
        self.oversample = aug_oversample_ratio
        # Effective extraction size (may be larger than patch_size for oversample)
        self.extract_size = tuple(
            int(round(p * aug_oversample_ratio)) for p in patch_size)
        self.multi_res_scales = multi_res_scales or []
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max
        self.normalize = normalize
        self.global_mean = global_mean
        self.global_std = global_std
        self.fg_ratio = foreground_oversample_ratio
        self.samples_per_volume = samples_per_volume
        self.is_train = is_train
        self.region_weights = region_weights

        self._img_cache = VolumeCache(cache_enabled)
        self._lbl_cache = VolumeCache(cache_enabled)

        # Build 3D foreground voxel index for oversampling
        self._vol_shapes: List[Tuple[int, int, int]] = []
        self._vol_fg_coords: List[np.ndarray] = []  # (N, 3) fg voxel coords per volume
        self._build_index()

    def _build_index(self) -> None:
        """Scan volumes and record foreground voxel coordinates."""
        logger.info("Building cubic dataset index for %d volumes...", len(self.image_paths))
        total_fg = 0
        for i in range(len(self.image_paths)):
            lbl = self._load_label(i)
            self._vol_shapes.append(lbl.shape)
            bg_val  = self.label_values[0]
            lbl_int = np.round(lbl).astype(np.int32)
            fg_mask = lbl_int != bg_val
            # Store sparse fg coordinates: (N, 3) array of (d, h, w)
            coords = np.argwhere(fg_mask)  # (N, 3)
            # Subsample if too many (memory efficiency)
            if len(coords) > 50000:
                rng = np.random.RandomState(42)
                coords = coords[rng.choice(len(coords), 50000, replace=False)]
            self._vol_fg_coords.append(coords)
            total_fg += len(coords)
        logger.info("Cubic index: %d volumes, %d fg voxels sampled",
                     len(self.image_paths), total_fg)

    def _load_image(self, vol_idx: int) -> np.ndarray:
        path = self.image_paths[vol_idx]
        cached = self._img_cache.get(path)
        if cached is not None:
            return cached
        img = load_nifti(path)
        img = preprocess_image(
            img, self.intensity_min, self.intensity_max,
            self.normalize, self.global_mean, self.global_std)
        self._img_cache.put(path, img)
        return img

    def _load_label(self, vol_idx: int) -> np.ndarray:
        path = self.label_paths[vol_idx]
        cached = self._lbl_cache.get(path)
        if cached is not None:
            return cached
        lbl = load_nifti(path)
        self._lbl_cache.put(path, lbl)
        return lbl

    def __len__(self) -> int:
        return len(self.image_paths) * self.samples_per_volume

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Unified multi-resolution path.

        multi_res_scales is always >= 1 element ([1.0] for single-res).
        For each scale: extract (scale * extract_size) cube → resize to extract_size.
        Output:
          image: (C_res, eD, eH, eW) — C_res channels, one per scale
          label: (C_res, eD, eH, eW) — raw integer labels per scale
          weight_map: (C_res, eD, eH, eW) — optional, per-scale region weights
        """
        vol_idx = idx % len(self.image_paths)
        img = self._load_image(vol_idx)
        lbl = self._load_label(vol_idx)
        D, H, W = img.shape

        center = self._sample_center(vol_idx, D, H, W)
        eD, eH, eW = self.extract_size

        img_channels, lbl_channels, wmap_channels = [], [], []
        for scale in self.multi_res_scales:
            sD = int(round(eD * scale))
            sH = int(round(eH * scale))
            sW = int(round(eW * scale))

            img_s = _extract_cubic_patch(img, center, (sD, sH, sW))
            lbl_s = _extract_cubic_patch(lbl, center, (sD, sH, sW))

            img_s = resize_3d(img_s, eD, eH, eW, is_label=False)
            lbl_s = resize_3d(lbl_s, eD, eH, eW, is_label=True)

            img_channels.append(img_s)
            lbl_channels.append(lbl_s)

            if self.region_weights:
                wmap_s = compute_region_weight_map(lbl_s, self.label_values, self.region_weights)
                wmap_channels.append(wmap_s[0])  # (D, H, W), squeeze the leading 1

        result = {
            "image": torch.from_numpy(np.stack(img_channels, axis=0).astype(np.float32)),
            "label": torch.from_numpy(np.stack(lbl_channels, axis=0).astype(np.float32))}
        if wmap_channels:
            result["weight_map"] = torch.from_numpy(
                np.stack(wmap_channels, axis=0).astype(np.float32))  # (C_res, eD, eH, eW)
        return result

    def _sample_center(self, vol_idx: int, D: int, H: int, W: int) -> Tuple[int, int, int]:
        """Sample a center (d, h, w) with optional foreground oversampling."""
        fg_coords = self._vol_fg_coords[vol_idx]
        if (self.is_train and self.fg_ratio > 0
                and len(fg_coords) > 0
                and np.random.random() < self.fg_ratio):
            idx = np.random.randint(len(fg_coords))
            return tuple(fg_coords[idx])
        return (np.random.randint(0, D), np.random.randint(0, H), np.random.randint(0, W))