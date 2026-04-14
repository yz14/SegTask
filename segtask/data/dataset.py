"""Dataset classes for 2D and 3D segmentation.

- SegDataset2D: individual 2D slices (B, 1, H, W)
- SegDataset3D: 3D patches (B, 1, D, H, W), also used for 2.5D mode

2.5D mode directly reuses SegDataset3D with patch_size=(total_slices, H, W).
After 3D augmentation, the trainer squeezes to (B, D, H, W) for the 2D model.

Preprocessing (intensity windowing, normalization) is done at load time.
Augmentation is applied on GPU in the training loop, not here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from .matching import SampleRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Volume loading and caching
# ---------------------------------------------------------------------------
class VolumeCache:
    """Simple in-memory cache for loaded NIfTI volumes."""

    def __init__(self, enabled: bool = False):
        self._enabled = enabled
        self._cache: Dict[str, np.ndarray] = {}

    def get(self, path: str) -> Optional[np.ndarray]:
        if self._enabled and path in self._cache:
            return self._cache[path]
        return None

    def put(self, path: str, data: np.ndarray) -> None:
        if self._enabled:
            self._cache[path] = data

    def clear(self) -> None:
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


def load_nifti(path: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """Load a NIfTI file and return numpy array.

    Returns array in (D, H, W) or (H, W) order, with the last axis
    of the NIfTI (typically axial slices) as dimension 0.
    """
    nii = nib.load(path)
    data = nii.get_fdata().astype(dtype)
    # NIfTI convention: (X, Y, Z) where Z is typically axial
    # We want (Z, Y, X) = (D, H, W) for 3D, where D=slices
    if data.ndim == 3:
        data = data.transpose(2, 1, 0)  # (X,Y,Z) -> (Z,Y,X) = (D,H,W)
    return data


def preprocess_image(
    volume: np.ndarray,
    intensity_min: float = -1024.0, intensity_max: float = 3071.0,
    normalize: str = "minmax", global_mean: float = 0.0, global_std: float = 1.0) -> np.ndarray:
    """Apply intensity windowing and normalization.

    Args:
        volume: Raw image volume.
        intensity_min/max: HU window range.
        normalize: "minmax" → [0,1], "zscore" → zero-mean unit-var.

    Returns:
        Preprocessed float32 volume.
    """
    vol = volume.copy()
    vol = np.clip(vol, intensity_min, intensity_max)

    if   normalize == "minmax":
        denom = intensity_max - intensity_min
        if denom > 0:
            vol = (vol - intensity_min) / denom
    elif normalize == "zscore":
        if global_std > 0:
            vol = (vol - global_mean) / global_std
    else:
        raise ValueError(f"Unknown normalize mode: {normalize}")

    return vol.astype(np.float32)


def preprocess_label(
    volume: np.ndarray,
    label_values: List[int],
    exclude_background: bool = False,
) -> np.ndarray:
    """Convert multi-value label to per-class binary masks.

    Args:
        volume: Integer label volume of shape (...).
        label_values: List of integer label values to include.
            label_values[0] is background.
        exclude_background: If True, omit the background (index 0) channel.
            Used for per_class output mode where each foreground class gets
            its own independent binary output.

    Returns:
        Binary masks of shape (C, ...) where:
        - C = len(label_values) if exclude_background=False (softmax mode)
        - C = len(label_values)-1 if exclude_background=True (per_class mode)
    """
    vol = np.round(volume).astype(np.int32)
    values = label_values[1:] if exclude_background else label_values
    # Vectorized: broadcast (C, 1, ...) == (1, ...) → (C, ...)
    lv_arr = np.array(values, dtype=np.int32).reshape(-1, *([1] * vol.ndim))
    return (vol[np.newaxis] == lv_arr).astype(np.float32)


def load_and_preprocess(
    rec: SampleRecord,
    img_cache: VolumeCache, lbl_cache: VolumeCache,
    intensity_min: float, intensity_max: float,
    normalize: str, global_mean: float, global_std: float) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess an image-label pair with caching. Shared by all datasets."""
    img = img_cache.get(rec.image_path)
    if img is None:
        img = load_nifti(rec.image_path, dtype=np.float32)
        img = preprocess_image(img, intensity_min, intensity_max, normalize, global_mean, global_std)
        img_cache.put(rec.image_path, img)

    lbl = lbl_cache.get(rec.label_path)
    if lbl is None:
        lbl = load_nifti(rec.label_path, dtype=np.float32)
        lbl_cache.put(rec.label_path, lbl)

    return img, lbl


def has_foreground(label_slice: np.ndarray, label_values: List[int]) -> bool:
    """Check if a slice contains any foreground (non-background) label. Vectorized."""
    lbl_int = np.round(label_slice).astype(np.int32)
    bg_val = label_values[0]
    return bool(np.any(lbl_int != bg_val))


class SegInferenceDataset(Dataset):
    """Inference-only dataset for 3D volumes without labels.

    Loads and preprocesses raw image volumes for sliding-window inference.
    This is the test-time counterpart of SegDataset3D — it does NOT crop
    or pad to patch_size, because the Predictor uses sliding window inference.

    Works correctly for any volume size (smaller or larger than patch_size),
    matching how real-world data is handled at inference time.
    """

    def __init__(
        self,
        image_paths: List[str],
        intensity_min: float = -1024.0,
        intensity_max: float = 3071.0,
        normalize: str = "minmax",
        global_mean: float = 0.0,
        global_std: float = 1.0,
        cache_enabled: bool = False,
    ):
        super().__init__()
        self.image_paths = image_paths
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max
        self.normalize = normalize
        self.global_mean = global_mean
        self.global_std = global_std
        self._img_cache = VolumeCache(cache_enabled)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.image_paths[idx]
        img = self._load_image(path)
        return {
            "image": torch.from_numpy(img[np.newaxis]).float(),  # (1, D, H, W)
            "subject_id": Path(path).stem,
            "original_shape": torch.tensor(list(img.shape), dtype=torch.long),
        }

    def _load_image(self, path: str) -> np.ndarray:
        cached = self._img_cache.get(path)
        if cached is not None:
            return cached
        img = load_nifti(path, dtype=np.float32)
        img = preprocess_image(
            img, self.intensity_min, self.intensity_max,
            self.normalize, self.global_mean, self.global_std,
        )
        self._img_cache.put(path, img)
        return img


def resize_2d(arr: np.ndarray, target_h: int, target_w: int, is_label: bool = False) -> np.ndarray:
    """Resize the last two dimensions (H, W) of an array to (target_h, target_w).

    Works for any leading dimensions: (H, W), (C, H, W), (D, H, W), etc.
    Uses bilinear interpolation for images, nearest-neighbor for labels.
    Works identically at train and test time — no labels needed.
    """
    from scipy.ndimage import zoom

    H, W = arr.shape[-2], arr.shape[-1]
    if H == target_h and W == target_w:
        return arr

    scale_h = target_h / H
    scale_w = target_w / W
    # Build zoom factors: 1.0 for all leading dims, scale for last two
    factors = [1.0] * (arr.ndim - 2) + [scale_h, scale_w]
    order = 0 if is_label else 1  # nearest for labels, bilinear for images
    return zoom(arr, factors, order=order).astype(arr.dtype)


# ---------------------------------------------------------------------------
# 2D Dataset
# ---------------------------------------------------------------------------
class SegDataset2D(Dataset):
    """2D slice-based dataset.

    Each sample is a single axial slice extracted from a 3D volume.
    All slices are RESIZED to target_size for uniform batching.
    This works identically at train and test time (no labels needed).
    """

    def __init__(
        self,
        records: List[SampleRecord], label_values: List[int],
        target_size: Tuple[int, int] = (256, 256),
        intensity_min: float = -1024.0, intensity_max: float = 3071.0,
        normalize: str = "minmax",
        global_mean: float = 0.0, global_std: float = 1.0,
        cache_enabled: bool = False,
        foreground_oversample_ratio: float = 0.0,
        is_train: bool = True,
        exclude_background: bool = False):
        super().__init__()
        self.records = records
        self.label_values = label_values
        self.target_size = tuple(target_size)
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max
        self.normalize = normalize
        self.global_mean = global_mean
        self.global_std = global_std
        self.foreground_ratio = foreground_oversample_ratio
        self.is_train = is_train
        self.exclude_background = exclude_background

        self._img_cache = VolumeCache(cache_enabled)
        self._lbl_cache = VolumeCache(cache_enabled)

        self._index: List[Tuple[int, int]] = []
        self._fg_indices: List[int] = []
        self._bg_indices: List[int] = []
        self._build_index()

    def _load_volume(self, rec: SampleRecord) -> Tuple[np.ndarray, np.ndarray]:
        return load_and_preprocess(
            rec, self._img_cache, self._lbl_cache,
            self.intensity_min, self.intensity_max,
            self.normalize, self.global_mean, self.global_std)

    def _build_index(self) -> None:
        logger.info("Building 2D slice index for %d volumes...", len(self.records))
        for vol_idx, rec in enumerate(self.records):
            _, lbl = self._load_volume(rec)
            for s in range(lbl.shape[0]):
                flat_idx = len(self._index)
                self._index.append((vol_idx, s))
                if has_foreground(lbl[s], self.label_values):
                    self._fg_indices.append(flat_idx)
                else:
                    self._bg_indices.append(flat_idx)
        logger.info(
            "2D index: %d slices (%d fg, %d bg) from %d volumes",
            len(self._index), len(self._fg_indices),
            len(self._bg_indices), len(self.records))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if (
            self.is_train
            and self.foreground_ratio > 0
            and self._fg_indices
            and np.random.random() < self.foreground_ratio
        ):
            idx = self._fg_indices[np.random.randint(len(self._fg_indices))]

        vol_idx, slice_idx = self._index[idx]
        rec = self.records[vol_idx]
        img, lbl = self._load_volume(rec)
        img_slice = img[slice_idx]  # (H, W)
        lbl_slice = lbl[slice_idx]  # (H, W)

        # Resize to uniform target_size — works at both train and test time
        th, tw = self.target_size
        img_slice = resize_2d(img_slice, th, tw, is_label=False)
        lbl_slice = resize_2d(lbl_slice, th, tw, is_label=True)

        lbl_mc = preprocess_label(lbl_slice, self.label_values, self.exclude_background)

        return {
            "image": torch.from_numpy(img_slice[np.newaxis]).float(),  # (1, H, W)
            "label": torch.from_numpy(lbl_mc).float(),
            "subject_id": rec.subject_id,
            "slice_idx": slice_idx}


def resize_3d(arr: np.ndarray, target_d: int, target_h: int, target_w: int, is_label: bool = False) -> np.ndarray:
    """Resize the last three dimensions (D, H, W) of an array.

    Works for any leading dimensions: (D, H, W), (C, D, H, W), etc.
    Uses bilinear interpolation for images, nearest-neighbor for labels.
    Works identically at train and test time — no labels needed.
    """
    from scipy.ndimage import zoom

    D, H, W = arr.shape[-3:]
    if D == target_d and H == target_h and W == target_w:
        return arr

    scale_d, scale_h, scale_w = target_d / D, target_h / H, target_w / W
    factors = [scale_d, scale_h, scale_w]
    if arr.ndim > 3:
        factors = [1.0] * (arr.ndim - 3) + factors
    order = 0 if is_label else 1
    return zoom(arr, factors, order=order).astype(arr.dtype)


# ---------------------------------------------------------------------------
# 3D Dataset (also used for 2.5D — see loader.py)
# ---------------------------------------------------------------------------
class SegDataset3D(Dataset):
    """3D patch-based dataset.

    Resizes 3D volumes to patch_size for uniform batching.
    Uses foreground oversampling to ensure patches contain anatomical structures.
    Works identically at train and test time.
    """

    def __init__(
        self,
        records: List[SampleRecord], label_values: List[int],
        patch_size: Tuple[int, int, int] = (96, 96, 96),
        intensity_min: float = -1024.0, intensity_max: float = 3071.0,
        normalize: str = "minmax",
        global_mean: float = 0.0, global_std: float = 1.0,
        cache_enabled: bool = False,
        foreground_oversample_ratio: float = 0.5,
        samples_per_volume: int = 4,
        is_train: bool = True,
        exclude_background: bool = False):
        super().__init__()
        self.records = records
        self.label_values = label_values
        self.patch_size = tuple(patch_size)
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max
        self.normalize = normalize
        self.global_mean = global_mean
        self.global_std = global_std
        self.foreground_ratio = foreground_oversample_ratio
        self.samples_per_volume = samples_per_volume
        self.is_train = is_train
        self.exclude_background = exclude_background

        self._img_cache = VolumeCache(cache_enabled)
        self._lbl_cache = VolumeCache(cache_enabled)

        self._index: List[Tuple[int, int]] = []
        self._fg_indices: List[int] = []
        self._bg_indices: List[int] = []
        self._build_index()

    def _load_volume(self, rec: SampleRecord) -> Tuple[np.ndarray, np.ndarray]:
        return load_and_preprocess(
            rec, self._img_cache, self._lbl_cache,
            self.intensity_min, self.intensity_max,
            self.normalize, self.global_mean, self.global_std)

    def _build_index(self) -> None:
        """Build per-slice index for foreground oversampling."""
        logger.info("Building 3D slice index for %d volumes...", len(self.records))
        for vol_idx, rec in enumerate(self.records):
            _, lbl = self._load_volume(rec)
            D = lbl.shape[0]
            for s in range(D):
                flat_idx = len(self._index)
                self._index.append((vol_idx, s))
                if has_foreground(lbl[s], self.label_values):
                    self._fg_indices.append(flat_idx)
                else:
                    self._bg_indices.append(flat_idx)
        logger.info(
            "3D index: %d slices (%d fg, %d bg) from %d volumes",
            len(self._index), len(self._fg_indices),
            len(self._bg_indices), len(self.records))

    def __len__(self) -> int:
        return len(self._index) * self.samples_per_volume

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Sample a slice index, with foreground oversampling
        slice_idx = idx % len(self._index)
        if (self.is_train
            and self.foreground_ratio > 0
            and self._fg_indices
            and np.random.random() < self.foreground_ratio):
            slice_idx = self._fg_indices[np.random.randint(len(self._fg_indices))]

        vol_idx, z = self._index[slice_idx]
        rec = self.records[vol_idx]

        img, lbl   = self._load_volume(rec)  # (D, H, W)
        pd, ph, pw = self.patch_size

        # Extract 3D patch centered around the selected slice
        # Key: cut exactly pd slices to preserve D-axis information.
        # If at boundary or volume < pd, PAD with zeros (never resize D).
        half_d  = pd // 2
        d_start = max(0, z - half_d)
        d_end   = d_start + pd
        # Clamp to volume bounds, then shift window if possible
        if d_end > img.shape[0]:
            d_end = img.shape[0]
            d_start = max(0, d_end - pd)

        img_patch = img[d_start:d_end]  # (actual_d, H_orig, W_orig)
        lbl_patch = lbl[d_start:d_end]

        # Pad D if volume has fewer slices than pd (zero-pad, not resize)
        actual_d = img_patch.shape[0]
        if actual_d < pd:
            pad_d = pd - actual_d
            img_patch = np.pad(img_patch, ((0, pad_d), (0, 0), (0, 0)), mode='constant')
            lbl_patch = np.pad(lbl_patch, ((0, pad_d), (0, 0), (0, 0)), mode='constant')

        # Only resize H, W — preserve D resolution (core design principle)
        img_patch = resize_2d(img_patch, ph, pw, is_label=False)
        lbl_patch = resize_2d(lbl_patch, ph, pw, is_label=True)

        lbl_mc = preprocess_label(lbl_patch, self.label_values, self.exclude_background)

        return {
            "image": torch.from_numpy(img_patch[np.newaxis]).float(),  # (1, D, H, W)
            "label": torch.from_numpy(lbl_mc).float(),  # (num_classes, D, H, W)
            "subject_id": rec.subject_id,
            "slice_idx": z}