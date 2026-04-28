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
from collections import OrderedDict
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
    volume: np.ndarray, label_values: List[int],
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
    """In-memory LRU cache for loaded volumes.

    When `max_volumes > 0`, entries are evicted in least-recently-used
    order once the cache reaches capacity. `max_volumes = 0` keeps the
    legacy unbounded behaviour (useful when the dataset is known to fit
    fully in RAM; risky otherwise).

    `enabled=False` disables caching entirely (no store, no eviction).
    """

    def __init__(self, enabled: bool = False, max_volumes: int = 0):
        self._enabled = enabled
        self._max = max(int(max_volumes), 0)
        self._store: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def get(self, path: str) -> Optional[np.ndarray]:
        if not self._enabled:
            return None
        data = self._store.get(path)
        if data is not None:
            # Mark as most-recently-used.
            self._store.move_to_end(path)
        return data

    def put(self, path: str, data: np.ndarray) -> None:
        if not self._enabled:
            return
        if path in self._store:
            self._store.move_to_end(path)
            self._store[path] = data
            return
        self._store[path] = data
        if self._max > 0:
            while len(self._store) > self._max:
                # popitem(last=False) pops the LEAST-recently-used entry.
                self._store.popitem(last=False)

    @property
    def size(self) -> int:
        return len(self._store)

    # ------------------------------------------------------------------
    # Pickling: drop the cache contents when the Dataset is shipped to a
    # DataLoader worker. On Windows (spawn start method) the entire
    # Dataset object is pickled through an OS pipe for every worker on
    # every epoch; a fully populated label cache (built eagerly in
    # ``_build_index``) easily inflates the payload past the pipe write
    # limit, surfacing as ``OSError: [Errno 22] Invalid argument`` on the
    # writer side and ``_pickle.UnpicklingError: pickle data was
    # truncated`` on the reader side.
    #
    # Each worker process must populate its own cache anyway (no shared
    # memory between spawned workers), so transferring the parent's
    # cached arrays is pure overhead. We strip ``_store`` on pickle and
    # restore an empty ``OrderedDict`` on unpickle; the LRU behaviour is
    # preserved per-process.
    # ------------------------------------------------------------------
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_store"] = OrderedDict()
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if not isinstance(self._store, OrderedDict):
            self._store = OrderedDict()


# ---------------------------------------------------------------------------
# 3D Segmentation Dataset
# ---------------------------------------------------------------------------
class SegDataset3D(Dataset):
    """3D z-axis sliding-window segmentation dataset.

    Z-axis patching semantics — the window slides ALONG Z ONLY; the in-plane
    (H, W) extent is always the full volume resolution, not a sub-crop.

    Pipeline per sample (per scale s in ``multi_res_scales``)::

        (D_vol, H_vol, W_vol)               e.g. (300, 512, 512)
            │  sample center z, take round(eD*s) slices along z axis
            │  (edge-replicate outside volume bounds when s > 1)
            ▼
        (round(eD*s), H_vol, W_vol)         full-resolution in-plane
            │  resize_3d to (eD, pH, pW)
            ▼
        (eD, pH, pW)                        stacked as channel s → (C_res, …)

    The trainer later center-crops eD → pD along the depth axis after
    GPU augmentation (when ``aug_oversample_ratio > 1``).

    ``aug_oversample_ratio`` applies to the Z axis ONLY. H, W already
    collapse to ``patch_size`` directly and therefore need no extra
    margin — consistent with ``predictor._sliding_window_z``.

    ``multi_res_scales`` (default ``[1.0]``) controls z-axis multi-FOV
    inputs. Each scale ``s`` gives the network a physically wider z-range
    (same center z, ``round(eD*s)`` slices) compressed back to ``eD``.
    ``s == 1.0`` preserves bit-identical legacy behaviour; ``s > 1.0``
    always uses edge-replicate padding at volume bounds so the physical
    z-FOV is preserved without stretch artefacts.

    Output shape::
      image: (C_res, eD, pH, pW) float32
      label: (C_res, eD, pH, pW) float32 raw integer labels (binarized
             at loss time by MultiResolutionLoss)
      weight_map (optional): (C_res, eD, pH, pW) float32
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
        cache_max_volumes: int = 0,
        region_weights: Optional[List[float]] = None):
        super().__init__()
        assert len(image_paths) == len(label_paths)
        assert aug_oversample_ratio >= 1.0, (
            f"aug_oversample_ratio must be >= 1.0, got {aug_oversample_ratio}")
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.label_values = label_values
        self.patch_size = tuple(patch_size)
        self.oversample = float(aug_oversample_ratio)
        # Z-axis mode: ONLY the z (depth) extent is oversampled so the trainer
        # can center-crop rotation / elastic margin along z after GPU aug.
        # H, W are taken at full volume resolution during extraction and
        # resized straight to patch_size (pH, pW) — no in-plane sub-crop
        # exists, so no oversample margin is needed or meaningful there.
        # This also matches `predictor._sliding_window_z`, which feeds the
        # model H_vol → pH, pW in a single resize step.
        pD, pH, pW = self.patch_size
        self.extract_size = (int(round(pD * self.oversample)), pH, pW)
        # Multi-resolution input: z-axis only. `[1.0]` = single-channel
        # (legacy). Scales > 1 extract proportionally wider z-FOVs around
        # the same center z and resize back, giving the network multi-FOV
        # context as extra input channels.
        self.multi_res_scales = list(multi_res_scales) if multi_res_scales else [1.0]
        assert all(s >= 1.0 for s in self.multi_res_scales), (
            f"All multi_res_scales must be >= 1.0, got {self.multi_res_scales}")
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max
        self.normalize = normalize
        self.global_mean = global_mean
        self.global_std = global_std
        self.fg_ratio = foreground_oversample_ratio
        self.samples_per_volume = samples_per_volume
        self.is_train = is_train
        self.region_weights = region_weights

        self._img_cache = VolumeCache(cache_enabled, cache_max_volumes)
        self._lbl_cache = VolumeCache(cache_enabled, cache_max_volumes)

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
        # `extract_size == (eD, pH, pW)` — only z is oversampled (the trainer
        # later center-crops eD → pD after GPU augmentation); H, W go straight
        # to patch_size via the resize below. When oversample == 1.0 then
        # eD == pD and the trainer skips the z crop as well.
        eD, eH, eW = self.extract_size

        # Select center z-position (shared across all scales so the
        # multi-FOV views are physically nested around the same anchor).
        z = self._sample_z(vol_idx, D_vol)

        # Build per-scale channel stack. For scale=1.0 we keep the legacy
        # clamp-then-stretch extraction (`_extract_z_patch`) for bit-exact
        # backward compatibility with prior single-res training. For
        # scale>1.0 we use edge-replicate padding so the physical z-FOV
        # (= scale * eD slices around z) is preserved without stretch
        # artefacts when z is near the volume boundary.
        img_channels: List[np.ndarray] = []
        lbl_channels: List[np.ndarray] = []
        wmap_channels: List[np.ndarray] = []
        for scale in self.multi_res_scales:
            D_s = int(round(eD * scale))
            if scale == 1.0:
                img_s, lbl_s = self._extract_z_patch(img, lbl, z, D_s)
            else:
                img_s, lbl_s = self._extract_z_patch_padded(img, lbl, z, D_s)

            # Resize in a single 3D zoom:
            #   (actual_d, H_vol, W_vol) → (eD, pH, pW)
            # H_vol, W_vol collapse directly to patch_size (pH, pW) —
            # matching `predictor._sliding_window_z`.
            img_s = resize_3d(img_s, eD, eH, eW, is_label=False)
            lbl_s = resize_3d(lbl_s, eD, eH, eW, is_label=True)
            img_channels.append(img_s)
            lbl_channels.append(lbl_s)

            if self.region_weights:
                wmap_s = compute_region_weight_map(
                    lbl_s, self.label_values, self.region_weights)
                wmap_channels.append(wmap_s[0])  # drop the leading 1

        # Stack scales as channel 0 → (C_res, eD, pH, pW). For the legacy
        # single-res default (`multi_res_scales=[1.0]`), C_res == 1 and
        # the shape is identical to the pre-multires z-axis output.
        result = {
            "image": torch.from_numpy(
                np.stack(img_channels, axis=0).astype(np.float32)),
            "label": torch.from_numpy(
                np.stack(lbl_channels, axis=0).astype(np.float32))}
        if wmap_channels:
            result["weight_map"] = torch.from_numpy(
                np.stack(wmap_channels, axis=0).astype(np.float32))
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

    def _extract_z_patch_padded(
        self, img: np.ndarray, lbl: np.ndarray, z_center: int,
        D_patch: int) -> Tuple[np.ndarray, np.ndarray]:
        """Paired image+label edge-padded extraction (see module-level
        `extract_z_patch_padded` for semantics). Kept as a method for
        API continuity with `_extract_z_patch`.
        """
        return (
            extract_z_patch_padded(img, z_center, D_patch),
            extract_z_patch_padded(lbl, z_center, D_patch),
        )


# ---------------------------------------------------------------------------
# Module-level z-axis patch extractor (shared with Predictor)
# ---------------------------------------------------------------------------
def extract_z_patch_padded(
    vol: np.ndarray, z_center: int, D_patch: int) -> np.ndarray:
    """Extract EXACTLY ``D_patch`` consecutive slices from ``vol`` along
    the z axis, centered at ``z_center`` and edge-replicate-padded when
    the window exceeds volume bounds.

    Unlike a plain slice, this preserves the physical z-FOV: the output
    always has depth ``D_patch`` regardless of volume size / boundary
    conditions. Required for z-axis multi-resolution (scale > 1) inputs
    so different scales are directly comparable — without padding,
    ``resize_3d`` would stretch a short boundary window to D_patch and
    undo the multi-FOV effect.

    In-plane (H, W) axes are left untouched (matches z-axis mode
    semantics). Labels are safe under ``mode="edge"`` because the
    replication is of an existing boundary slice's discrete values.
    """
    D_vol = vol.shape[0]
    half  = D_patch // 2
    lo = z_center - half
    hi = lo + D_patch
    src_lo = max(lo, 0)
    src_hi = min(hi, D_vol)
    pad_before = max(-lo, 0)
    pad_after  = max(hi - D_vol, 0)

    patch = vol[src_lo:src_hi]
    if pad_before > 0 or pad_after > 0:
        pad_width = [(pad_before, pad_after)] + [(0, 0)] * (vol.ndim - 1)
        patch = np.pad(patch, pad_width, mode="edge")
    return patch.copy()


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
        cache_max_volumes: int = 0,
        region_weights: Optional[List[float]] = None):
        super().__init__()
        assert len(image_paths) == len(label_paths)
        assert aug_oversample_ratio >= 1.0, (
            f"aug_oversample_ratio must be >= 1.0, got {aug_oversample_ratio}")
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.label_values = label_values
        self.patch_size = tuple(patch_size)
        self.oversample = aug_oversample_ratio
        # Effective extraction size (may be larger than patch_size for oversample)
        self.extract_size = tuple(
            int(round(p * aug_oversample_ratio)) for p in patch_size)
        self.multi_res_scales = multi_res_scales or []
        # Largest multi-res scale determines the biggest physical cube that
        # must stay in-bounds to avoid excessive edge-replicate padding.
        self._max_scale = max(self.multi_res_scales) if self.multi_res_scales else 1.0
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max
        self.normalize = normalize
        self.global_mean = global_mean
        self.global_std = global_std
        self.fg_ratio = foreground_oversample_ratio
        self.samples_per_volume = samples_per_volume
        self.is_train = is_train
        self.region_weights = region_weights

        self._img_cache = VolumeCache(cache_enabled, cache_max_volumes)
        self._lbl_cache = VolumeCache(cache_enabled, cache_max_volumes)

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

    def _safe_center_range(
        self, D: int, H: int, W: int) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """Return `(lo, hi)` center-coordinate bounds on each axis that keep
        the entire largest multi-res cube inside the volume.

        When the volume is smaller than the cube on an axis, we relax that
        axis to `[half, half + 1)` (== volume centre), which reproduces the
        legacy edge-replicate behaviour exactly where it is unavoidable.
        Using `hi` as an *exclusive* upper bound lets callers plug straight
        into `np.random.randint(lo, hi)` / `np.clip(..., lo, hi - 1)`.
        """
        eD, eH, eW = self.extract_size
        # Physical cube size for the largest scale (rounded like the dataset).
        sD = int(round(eD * self._max_scale))
        sH = int(round(eH * self._max_scale))
        sW = int(round(eW * self._max_scale))

        def _axis(size: int, patch: int) -> Tuple[int, int]:
            half = patch // 2
            lo = half
            # `_extract_cubic_patch` takes [c - patch//2, c - patch//2 + patch),
            # so the exclusive upper centre bound that keeps the top slice
            # in-bounds is `size - (patch - half)`.
            hi = size - (patch - half)
            if hi <= lo:
                # Volume too small on this axis — centre it and accept padding.
                mid = size // 2
                return mid, mid + 1
            return lo, hi

        return _axis(D, sD), _axis(H, sH), _axis(W, sW)

    def _sample_center(self, vol_idx: int, D: int, H: int, W: int) -> Tuple[int, int, int]:
        """Sample a center (d, h, w) with optional foreground oversampling.

        The sampled centre is clamped into the `_safe_center_range` box so
        that the largest multi-res cube extracted around it sits entirely
        within the volume. Without this clamp, sampling an fg voxel right
        at the volume corner produced patches where >50 % of voxels came
        from `np.pad(mode='edge')` — massively skewing training toward
        synthetic replicated borders (BUG-D in the audit report).
        """
        (dlo, dhi), (hlo, hhi), (wlo, whi) = self._safe_center_range(D, H, W)
        fg_coords = self._vol_fg_coords[vol_idx]
        if (self.is_train and self.fg_ratio > 0
                and len(fg_coords) > 0
                and np.random.random() < self.fg_ratio):
            idx = np.random.randint(len(fg_coords))
            d, h, w = fg_coords[idx]
            # `dhi - 1` because np.clip upper bound is INCLUSIVE.
            d = int(np.clip(int(d), dlo, dhi - 1))
            h = int(np.clip(int(h), hlo, hhi - 1))
            w = int(np.clip(int(w), wlo, whi - 1))
            return (d, h, w)
        return (int(np.random.randint(dlo, dhi)),
                int(np.random.randint(hlo, hhi)),
                int(np.random.randint(wlo, whi)))


# ---------------------------------------------------------------------------
# 3D Whole-Volume Dataset (no sliding window, no sub-cropping)
# ---------------------------------------------------------------------------
class SegDataset3DWhole(Dataset):
    """Whole-volume 3D segmentation dataset — each sample is the entire
    volume resized to ``extract_size`` (oversampled patch_size).

    Semantics:
      - No patching, no center sampling. The full volume is loaded,
        resized via `resize_3d` to ``(eD, eH, eW)`` = round(patch_size *
        oversample), and returned.
      - The trainer center-crops to ``patch_size`` after augmentation,
        identical to the other modes — this both removes rotation/elastic
        zero-padded corners AND finalises the model-facing input size.
      - ``samples_per_volume`` controls how many augmentation variants per
        epoch (no patch-location diversity to draw from).
      - ``foreground_oversample_ratio`` is ignored (no center sampling).
      - ``multi_res_scales`` must be ``[1.0]`` (validated in Config) —
        scaling a whole-volume resize has no physical meaning.

    Output (matches other modes for interoperability with the loss stack):
      image: (1, eD, eH, eW) float32
      label: (1, eD, eH, eW) float32 raw integer labels
      weight_map (optional): (1, eD, eH, eW) float32
    """

    def __init__(
        self,
        image_paths: List[str],
        label_paths: List[str],
        label_values: List[int],
        patch_size: Tuple[int, int, int] = (64, 128, 128),
        aug_oversample_ratio: float = 1.0,
        intensity_min: float = -1024.0,
        intensity_max: float = 3071.0,
        normalize: str = "minmax",
        global_mean: float = 0.0,
        global_std: float = 1.0,
        samples_per_volume: int = 1,
        is_train: bool = True,
        cache_enabled: bool = True,
        cache_max_volumes: int = 0,
        region_weights: Optional[List[float]] = None):
        super().__init__()
        assert len(image_paths) == len(label_paths)
        assert aug_oversample_ratio >= 1.0, (
            f"aug_oversample_ratio must be >= 1.0, got {aug_oversample_ratio}")
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.label_values = label_values
        self.patch_size = tuple(patch_size)
        self.oversample = float(aug_oversample_ratio)
        # 3-axis oversample matches cubic mode: provides augmentation
        # margin so rotation / elastic black corners get center-cropped
        # away by the trainer.
        self.extract_size = tuple(
            int(round(p * self.oversample)) for p in self.patch_size)
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max
        self.normalize = normalize
        self.global_mean = global_mean
        self.global_std = global_std
        self.samples_per_volume = samples_per_volume
        self.is_train = is_train
        self.region_weights = region_weights

        self._img_cache = VolumeCache(cache_enabled, cache_max_volumes)
        self._lbl_cache = VolumeCache(cache_enabled, cache_max_volumes)

        logger.info(
            "Whole-volume dataset: %d volumes, extract_size=%s, "
            "samples_per_volume=%d",
            len(self.image_paths), self.extract_size, self.samples_per_volume)

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
        vol_idx = idx % len(self.image_paths)
        img = self._load_image(vol_idx)
        lbl = self._load_label(vol_idx)
        eD, eH, eW = self.extract_size

        # Resize the entire volume in a single 3D zoom.
        img_r = resize_3d(img, eD, eH, eW, is_label=False)
        lbl_r = resize_3d(lbl, eD, eH, eW, is_label=True)

        result = {
            "image": torch.from_numpy(img_r[np.newaxis]).float(),  # (1, eD, eH, eW)
            "label": torch.from_numpy(lbl_r[np.newaxis]).float()}

        if self.region_weights:
            wmap = compute_region_weight_map(
                lbl_r, self.label_values, self.region_weights)
            result["weight_map"] = torch.from_numpy(wmap).float()  # (1, eD, eH, eW)
        return result