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
    intensity_min: float = -1024.0,
    intensity_max: float = 3071.0,
    normalize: str = "minmax",
    global_mean: float = 0.0,
    global_std: float = 1.0,
) -> np.ndarray:
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

    if normalize == "minmax":
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
) -> np.ndarray:
    """Convert multi-value label to per-class binary masks.

    Args:
        volume: Integer label volume of shape (...).
        label_values: List of integer label values to include.
            label_values[0] is background.

    Returns:
        Binary masks of shape (num_classes, ...) where num_classes = len(label_values).
        Each channel is 1 where the volume equals the corresponding label value.
    """
    vol = np.round(volume).astype(np.int32)
    # Vectorized: broadcast (num_classes, 1, ...) == (1, ...) → (num_classes, ...)
    lv_arr = np.array(label_values, dtype=np.int32).reshape(-1, *([1] * vol.ndim))
    return (vol[np.newaxis] == lv_arr).astype(np.float32)


def load_and_preprocess(
    rec: SampleRecord,
    img_cache: VolumeCache,
    lbl_cache: VolumeCache,
    intensity_min: float,
    intensity_max: float,
    normalize: str,
    global_mean: float,
    global_std: float,
) -> Tuple[np.ndarray, np.ndarray]:
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


def _compute_crop_origin(
    H: int,
    W: int,
    crop_h: int,
    crop_w: int,
    is_train: bool,
    fg_mask: Optional[np.ndarray] = None,
    fg_ratio: float = 0.0,
) -> Tuple[int, int]:
    """Compute the (h0, w0) crop origin for a (H, W) spatial field.

    This is computed ONCE and shared between image and label to ensure
    they receive the same spatial crop.
    """
    if H <= crop_h and W <= crop_w:
        return 0, 0

    if is_train:
        if (
            fg_ratio > 0
            and fg_mask is not None
            and np.random.random() < fg_ratio
            and fg_mask.any()
        ):
            fy, fx = np.where(fg_mask)
            idx = np.random.randint(len(fy))
            cy, cx = fy[idx], fx[idx]
            h0 = int(np.clip(cy - crop_h // 2, 0, max(0, H - crop_h)))
            w0 = int(np.clip(cx - crop_w // 2, 0, max(0, W - crop_w)))
        else:
            h0 = np.random.randint(0, max(1, H - crop_h + 1))
            w0 = np.random.randint(0, max(1, W - crop_w + 1))
    else:
        h0 = max(0, (H - crop_h) // 2)
        w0 = max(0, (W - crop_w) // 2)

    return h0, w0


def _apply_crop_pad(
    arr: np.ndarray,
    crop_h: int,
    crop_w: int,
    h0: int,
    w0: int,
) -> np.ndarray:
    """Pad array if needed, then crop at (h0, w0) to (crop_h, crop_w).

    Args:
        arr: Array with last two dims being (H, W).
        crop_h, crop_w: Target spatial size.
        h0, w0: Crop origin (from _compute_crop_origin).
    """
    H, W = arr.shape[-2], arr.shape[-1]
    pad_h = max(0, crop_h - H)
    pad_w = max(0, crop_w - W)
    if pad_h > 0 or pad_w > 0:
        pad_width = [(0, 0)] * (arr.ndim - 2) + [(0, pad_h), (0, pad_w)]
        arr = np.pad(arr, pad_width, mode="constant", constant_values=0)

    return arr[..., h0:h0 + crop_h, w0:w0 + crop_w]


# ---------------------------------------------------------------------------
# 2D Dataset
# ---------------------------------------------------------------------------
class SegDataset2D(Dataset):
    """2D slice-based dataset.

    Each sample is a single axial slice extracted from a 3D volume.
    Returns (image, label) tensors of shape (1, H, W) and (num_classes, H, W).
    Slices are cropped/padded to crop_size for uniform batching.
    """

    def __init__(
        self,
        records: List[SampleRecord],
        label_values: List[int],
        crop_size: Tuple[int, int] = (256, 256),
        intensity_min: float = -1024.0,
        intensity_max: float = 3071.0,
        normalize: str = "minmax",
        global_mean: float = 0.0,
        global_std: float = 1.0,
        cache_enabled: bool = False,
        foreground_oversample_ratio: float = 0.0,
        is_train: bool = True,
    ):
        super().__init__()
        self.records = records
        self.label_values = label_values
        self.crop_size = tuple(crop_size)
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max
        self.normalize = normalize
        self.global_mean = global_mean
        self.global_std = global_std
        self.foreground_ratio = foreground_oversample_ratio
        self.is_train = is_train

        self._img_cache = VolumeCache(cache_enabled)
        self._lbl_cache = VolumeCache(cache_enabled)

        # Build slice index: (volume_idx, slice_idx)
        self._index: List[Tuple[int, int]] = []
        # Track which slices have foreground
        self._fg_indices: List[int] = []
        self._bg_indices: List[int] = []

        self._build_index()

    def _load_volume(self, rec: SampleRecord) -> Tuple[np.ndarray, np.ndarray]:
        return load_and_preprocess(
            rec, self._img_cache, self._lbl_cache,
            self.intensity_min, self.intensity_max,
            self.normalize, self.global_mean, self.global_std,
        )

    def _build_index(self) -> None:
        """Build flat index of (volume_idx, slice_idx) pairs."""
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
            len(self._bg_indices), len(self.records),
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Foreground oversampling: with some probability, redirect to foreground
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

        # Build foreground mask for crop biasing
        fg_mask = None
        if self.is_train and self.foreground_ratio > 0:
            lbl_int = np.round(lbl_slice).astype(np.int32)
            fg_mask = np.zeros_like(lbl_int, dtype=bool)
            for lv in self.label_values[1:]:
                fg_mask |= (lbl_int == lv)

        # Compute crop origin ONCE, apply to both image and label
        ch, cw = self.crop_size
        H, W = img_slice.shape[-2], img_slice.shape[-1]
        padded_H = max(H, ch)
        padded_W = max(W, cw)
        # Pad fg_mask for origin computation
        pad_fg = fg_mask
        if fg_mask is not None and (padded_H > H or padded_W > W):
            pad_fg = np.pad(fg_mask, ((0, padded_H - H), (0, padded_W - W)), mode="constant")
        h0, w0 = _compute_crop_origin(padded_H, padded_W, ch, cw, self.is_train, pad_fg, self.foreground_ratio)
        img_slice = _apply_crop_pad(img_slice, ch, cw, h0, w0)
        lbl_slice = _apply_crop_pad(lbl_slice, ch, cw, h0, w0)

        # Convert label to multi-channel
        lbl_mc = preprocess_label(lbl_slice, self.label_values)  # (C, H, W)

        return {
            "image": torch.from_numpy(img_slice[np.newaxis]).float(),  # (1, H, W)
            "label": torch.from_numpy(lbl_mc).float(),  # (num_classes, H, W)
            "subject_id": rec.subject_id,
            "slice_idx": slice_idx,
        }


# ---------------------------------------------------------------------------
# 3D Dataset (also used for 2.5D — see loader.py)
# ---------------------------------------------------------------------------
class SegDataset3D(Dataset):
    """3D patch-based dataset.

    Randomly crops 3D patches from volumes. Uses foreground oversampling
    to ensure patches contain anatomical structures.
    """

    def __init__(
        self,
        records: List[SampleRecord],
        label_values: List[int],
        patch_size: Tuple[int, int, int] = (96, 96, 96),
        intensity_min: float = -1024.0,
        intensity_max: float = 3071.0,
        normalize: str = "minmax",
        global_mean: float = 0.0,
        global_std: float = 1.0,
        cache_enabled: bool = False,
        foreground_oversample_ratio: float = 0.5,
        samples_per_volume: int = 4,
        is_train: bool = True,
    ):
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

        self._img_cache = VolumeCache(cache_enabled)
        self._lbl_cache = VolumeCache(cache_enabled)

        # Precompute foreground voxel locations per volume
        self._fg_coords: List[Optional[np.ndarray]] = []
        self._build_fg_coords()

    def _load_volume(self, rec: SampleRecord) -> Tuple[np.ndarray, np.ndarray]:
        return load_and_preprocess(
            rec, self._img_cache, self._lbl_cache,
            self.intensity_min, self.intensity_max,
            self.normalize, self.global_mean, self.global_std,
        )

    def _build_fg_coords(self) -> None:
        """Precompute foreground voxel coordinates for each volume."""
        logger.info("Building 3D foreground coords for %d volumes...", len(self.records))
        bg_val = self.label_values[0]
        for rec in self.records:
            _, lbl = self._load_volume(rec)
            # Foreground = anything not background (vectorized, no loop over classes)
            fg_mask = np.round(lbl).astype(np.int32) != bg_val

            if fg_mask.any():
                coords = np.argwhere(fg_mask)  # (N, 3)
                # Subsample to keep memory bounded
                if len(coords) > 10000:
                    rng = np.random.RandomState(42)
                    coords = coords[rng.choice(len(coords), 10000, replace=False)]
                self._fg_coords.append(coords)
            else:
                self._fg_coords.append(None)

        n_with_fg = sum(1 for c in self._fg_coords if c is not None)
        logger.info("3D fg coords: %d/%d volumes have foreground", n_with_fg, len(self.records))

    def _random_crop_origin(
        self, vol_shape: Tuple[int, ...], fg_coords: Optional[np.ndarray]
    ) -> Tuple[int, int, int]:
        """Compute random crop origin, optionally centered on foreground."""
        D, H, W = vol_shape
        pd, ph, pw = self.patch_size

        if (
            fg_coords is not None
            and len(fg_coords) > 0
            and np.random.random() < self.foreground_ratio
        ):
            # Pick a random foreground voxel and center patch there
            idx = np.random.randint(len(fg_coords))
            cd, ch, cw = fg_coords[idx]
            d0 = int(np.clip(cd - pd // 2, 0, max(0, D - pd)))
            h0 = int(np.clip(ch - ph // 2, 0, max(0, H - ph)))
            w0 = int(np.clip(cw - pw // 2, 0, max(0, W - pw)))
        else:
            d0 = np.random.randint(0, max(1, D - pd + 1))
            h0 = np.random.randint(0, max(1, H - ph + 1))
            w0 = np.random.randint(0, max(1, W - pw + 1))

        return d0, h0, w0

    def _pad_if_needed(self, volume: np.ndarray) -> np.ndarray:
        """Pad volume if smaller than patch_size."""
        pd, ph, pw = self.patch_size
        D, H, W = volume.shape[-3:]
        pad_d = max(0, pd - D)
        pad_h = max(0, ph - H)
        pad_w = max(0, pw - W)
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            if volume.ndim == 3:
                volume = np.pad(volume, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant")
            else:
                # For multi-channel
                volume = np.pad(
                    volume,
                    ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
                    mode="constant",
                )
        return volume

    def __len__(self) -> int:
        return len(self.records) * self.samples_per_volume

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        vol_idx = idx // self.samples_per_volume
        rec = self.records[vol_idx]

        img, lbl = self._load_volume(rec)
        img = self._pad_if_needed(img)
        lbl = self._pad_if_needed(lbl)

        if self.is_train:
            d0, h0, w0 = self._random_crop_origin(img.shape, self._fg_coords[vol_idx])
        else:
            # Center crop for validation
            D, H, W = img.shape
            pd, ph, pw = self.patch_size
            d0 = max(0, (D - pd) // 2)
            h0 = max(0, (H - ph) // 2)
            w0 = max(0, (W - pw) // 2)

        pd, ph, pw = self.patch_size
        img_patch = img[d0:d0+pd, h0:h0+ph, w0:w0+pw]  # (D, H, W)
        lbl_patch = lbl[d0:d0+pd, h0:h0+ph, w0:w0+pw]  # (D, H, W)

        # Convert label
        lbl_mc = preprocess_label(lbl_patch, self.label_values)  # (C, D, H, W)

        return {
            "image": torch.from_numpy(img_patch[np.newaxis]).float(),  # (1, D, H, W)
            "label": torch.from_numpy(lbl_mc).float(),  # (num_classes, D, H, W)
            "subject_id": rec.subject_id,
            "crop_origin": torch.tensor([d0, h0, w0]),
        }
