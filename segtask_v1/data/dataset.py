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
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Volume I/O
# ---------------------------------------------------------------------------
# Transient NIfTI read failures (esp. on network / virtual-mount drives such
# as 百度网盘 cached volumes) manifest as ``RuntimeError: nifti_image_load
# failed for file: ...`` raised from ``sitk.ReadImage``. The file itself is
# fine — re-reading the same path a moment later succeeds. We therefore wrap
# every ReadImage with a bounded exponential-backoff retry so a single
# DataLoader worker hiccup does not crash the whole training run after
# many successful epochs.
#
# Tunables via env vars (no config-file churn):
#   SEGTASK_NIFTI_READ_RETRIES     - max attempts (default 4, => 3 retries)
#   SEGTASK_NIFTI_READ_BACKOFF_S   - initial backoff seconds (default 0.5)
_NIFTI_READ_RETRIES = max(1, int(os.environ.get("SEGTASK_NIFTI_READ_RETRIES", "4")))
_NIFTI_READ_BACKOFF_S = max(0.0, float(os.environ.get("SEGTASK_NIFTI_READ_BACKOFF_S", "0.5")))


# Substrings in a ``RuntimeError`` message that indicate the failure is NOT
# a transient I/O hiccup but an out-of-memory / allocation failure. Retrying
# these is counterproductive: it wastes time, keeps partially allocated
# buffers alive across attempts, and hides the real root cause (e.g. an
# oversized per-worker volume cache). We surface them immediately as
# ``MemoryError`` so DataLoader / the user can react appropriately
# (reduce ``cache_max_volumes`` / ``num_workers`` / ``batch_size``, etc.).
_ALLOC_ERROR_MARKERS = (
    "bad allocation",
    "failed to allocate memory",
    "std::bad_alloc",
    "cannot allocate memory",
)


def _is_alloc_error(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    msg = str(exc).lower()
    return any(m in msg for m in _ALLOC_ERROR_MARKERS)


def _sitk_read_with_retry(read_callable, path: str) -> "sitk.Image":
    """Invoke ``read_callable()`` (a zero-arg sitk read closure) with
    bounded retries.

    The callable is supplied by the caller so we can wrap arbitrary sitk
    read paths — ``sitk.ReadImage(...)`` for the legacy whole-volume
    reader, or an ``sitk.ImageFileReader`` configured with
    ``SetExtractIndex/SetExtractSize`` for the streamed sub-region path
    used by ``load_nifti_cropped``. The retry / OOM-detection contract is
    identical regardless of which closure is passed.

    Retries ONLY genuine I/O transients. Allocation failures (host OOM)
    are re-raised immediately as ``MemoryError`` — retrying them wastes
    time and masks the real problem.

    Logs a WARNING on every transient failure (including the file path so
    the offending volume is identifiable in multi-worker logs) and raises
    a descriptive ``RuntimeError`` if all I/O attempts fail.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(1, _NIFTI_READ_RETRIES + 1):
        try:
            return read_callable()
        except RuntimeError as exc:  # SimpleITK wraps low-level errors here
            if _is_alloc_error(exc):
                # Host OOM — do NOT retry. Re-raise as MemoryError so
                # Python / DataLoader treat it as the resource failure it
                # is instead of a recoverable I/O glitch.
                raise MemoryError(
                    f"NIfTI read aborted (host OOM) for {path}: {exc}") from exc
            last_exc = exc
            if attempt >= _NIFTI_READ_RETRIES:
                break
            wait = _NIFTI_READ_BACKOFF_S * (2 ** (attempt - 1))
            logger.warning(
                "NIfTI read failed (attempt %d/%d) for %s: %s — retrying in %.2fs",
                attempt, _NIFTI_READ_RETRIES, path, exc, wait)
            if wait > 0:
                time.sleep(wait)
    raise RuntimeError(
        f"NIfTI read permanently failed after {_NIFTI_READ_RETRIES} attempts "
        f"for {path}: {last_exc}") from last_exc


def load_nifti(path: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """Load NIfTI file → (D, H, W) numpy array via SimpleITK.

    Implementation notes
    --------------------
    Switched from ``nibabel.load(...).get_fdata()`` to SimpleITK because
    nibabel's ``_get_scaled`` promotes the decode buffer to
    ``np.promote_types(scl_slope.dtype, dtype)``. NIfTI headers almost
    always store ``scl_slope`` as float64 (even when its value is 1.0),
    so requesting float32 decode still allocates a float64 transient
    buffer — for a 512×512×349 CT volume that is 698 MiB per file, and
    multiplied across DataLoader workers it OOMs ~16 GB-RAM machines.

    SimpleITK reads pixels in the stored dtype (typically int16 for CT,
    uint8 for masks), applies ``scl_slope`` / ``scl_inter`` natively
    when a float pixel type is requested, and ``GetArrayFromImage``
    already returns the array in ``(Z, Y, X) == (D, H, W)`` order — so
    we save both memory AND the transpose pass.

    Args:
        path: NIfTI file path (.nii / .nii.gz).
        dtype: Output numpy dtype. Default ``np.float32``. Floating
            requests ask SimpleITK to decode directly to that float
            type (slope/intercept applied). Integer requests read the
            stored dtype natively and cast (used by label-pre-scan
            helpers that round to int32 immediately afterwards).

    Returns:
        ``(D, H, W)`` numpy array of the requested dtype.
    """
    np_dtype = np.dtype(dtype)
    if np.issubdtype(np_dtype, np.floating):
        # Decode directly into the requested float precision; SimpleITK
        # applies any scl_slope / scl_inter during the cast.
        sitk_pixel = (sitk.sitkFloat32 if np_dtype == np.float32
                      else sitk.sitkFloat64)
        read_args = (str(path), sitk_pixel)
    else:
        # Read native stored dtype (no float promotion); cast after.
        read_args = (str(path),)

    img = _sitk_read_with_retry(lambda: sitk.ReadImage(*read_args), path)
    arr = sitk.GetArrayFromImage(img)  # (Z, Y, X) = (D, H, W)
    if arr.dtype != np_dtype:
        arr = arr.astype(np_dtype, copy=False)
    return arr


def load_nifti_with_spacing(
    path: str, dtype: np.dtype = np.float32,
) -> "Tuple[np.ndarray, float]":
    """Load NIfTI → ``(volume, z_spacing_mm)``.

    The returned ``volume`` follows the same ``(D, H, W)`` layout as
    :func:`load_nifti` (SimpleITK already emits ``(Z, Y, X)`` order).
    ``z_spacing`` is the physical voxel size along that depth axis in
    millimetres, read from ``sitk.Image.GetSpacing()[2]`` (SimpleITK
    reports spacing in (X, Y, Z) order). Falls back to ``1.0`` if the
    file is missing valid Z-spacing metadata.

    Kept separate from :func:`load_nifti` so the (heavily reused)
    training data path stays untouched — only the inference-time
    z-interleave wrapper (``Predictor._sliding_window_z_interleaved``)
    needs physical spacing.
    """
    np_dtype = np.dtype(dtype)
    if np.issubdtype(np_dtype, np.floating):
        sitk_pixel = (sitk.sitkFloat32 if np_dtype == np.float32
                      else sitk.sitkFloat64)
        read_args = (str(path), sitk_pixel)
    else:
        read_args = (str(path),)
    img = _sitk_read_with_retry(lambda: sitk.ReadImage(*read_args), path)
    arr = sitk.GetArrayFromImage(img)
    if arr.dtype != np_dtype:
        arr = arr.astype(np_dtype, copy=False)
    spacing = img.GetSpacing()  # (sx, sy, sz)
    z_spacing = float(spacing[2]) if len(spacing) >= 3 else 1.0
    if not np.isfinite(z_spacing) or z_spacing <= 0.0:
        z_spacing = 1.0
    return arr, z_spacing


def load_nifti_cropped(
    path: str,
    bbox: "Optional[BBox]" = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Load NIfTI and (optionally) bbox-crop in a single pass.

    Memory profile (the whole point of this helper)
    ------------------------------------------------
    Two earlier iterations existed:

      1. ``load_nifti(...) + apply_bbox(...).copy()`` materialised the
         FULL volume TWICE at peak — once inside the SimpleITK image
         buffer, and once more in the numpy copy produced by
         ``GetArrayFromImage``.
      2. ``sitk.ReadImage(path, sitkFloat32) → GetArrayViewFromImage →
         slice → np.array(copy=True)`` halved that to ``1x full_volume +
         1x cropped`` by deferring the owned copy to the cropped view.

    Both still kept the FULL volume in fp32 alive in our process heap
    until the function returned. For ``num_workers=4`` validating with
    cache misses on every new volume — image + label + region_weight
    all forced to fp32 by the loader contract — the per-worker
    transient peaks into multi-GiB territory and trivially OOMs a
    16-GiB host. The validation log shows exactly this:
    ``RuntimeError: bad allocation`` on the third (region_weight)
    concurrent load.

    The current implementation drops the full-volume retained buffer
    entirely: it uses ``sitk.ImageFileReader.SetExtractIndex /
    SetExtractSize`` so SimpleITK returns ONLY the bbox ROI as a
    sitk.Image, with ``SetOutputPixelType`` taking care of
    ``scl_slope`` / ``scl_inter`` exactly as the whole-volume
    ``ReadImage(path, sitkFloat32)`` path did. Peak per-load drops
    from ``1x full_volume_bytes`` (in our heap, retained until
    function return) to ``1x cropped_bytes`` (≈ 14x smaller for a
    typical thoracic-CT-with-tight-ROI workload).

    Note on ``.nii.gz`` files: the gzip stream is not seekable, so
    ITK's NiftiImageIO must still decompress the full file inside
    ``Execute()``. That decompression buffer lives entirely inside
    ITK and is freed the instant ``Execute()`` returns — it never
    enters our ``sitk.Image`` Python object. Net effect: the
    decompression is a brief peak (one buffer in native dtype, ~2x
    smaller than fp32), not a sustained one.

    Args:
        path:  NIfTI file path (.nii / .nii.gz).
        bbox:  Optional ROI bbox (see ``BBox`` typedef). ``None``
               returns the full volume (still as an owned buffer).
        dtype: Output numpy dtype. Floating requests ask SimpleITK
               to decode directly to that float type (``scl_slope`` /
               ``scl_inter`` applied natively); integer requests read
               the stored dtype and cast after.

    Returns:
        ``(D', H', W')`` C-contiguous numpy array of ``dtype``, where
        the primed dims are the bbox extents (or full volume dims
        when ``bbox is None``). The returned buffer is independent
        of any sitk storage — safe to mutate, cache, or hand off to
        DataLoader workers.
    """
    np_dtype = np.dtype(dtype)
    floating = np.issubdtype(np_dtype, np.floating)
    sitk_pixel = (
        sitk.sitkFloat32 if (floating and np_dtype == np.float32)
        else sitk.sitkFloat64 if floating
        else None)

    def _read() -> "sitk.Image":
        # Build a fresh reader inside the closure so that retried
        # attempts don't reuse stale internal IORegion state from
        # a partially-failed Execute() call.
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        # Header-only pass: cheap, gives us GetSize() so we can clamp
        # the bbox defensively. NiftiImageIO reads ~352 bytes here.
        reader.ReadImageInformation()
        if bbox is not None:
            # SimpleITK uses (X, Y, Z) ordering — the OPPOSITE of numpy
            # / our BBox tuple. Translate carefully.
            full_w, full_h, full_d = reader.GetSize()
            (d0, d1), (h0, h1), (w0, w1) = bbox
            # Defensive clamping: a bbox that pokes past the file
            # bounds (e.g. mismatched ROI mask spacing) would otherwise
            # raise a cryptic ITK ``RequestedRegion is outside the
            # LargestPossibleRegion`` error inside Execute().
            d0c = max(0, min(d0, full_d))
            d1c = max(d0c, min(d1, full_d))
            h0c = max(0, min(h0, full_h))
            h1c = max(h0c, min(h1, full_h))
            w0c = max(0, min(w0, full_w))
            w1c = max(w0c, min(w1, full_w))
            if d1c > d0c and h1c > h0c and w1c > w0c:
                reader.SetExtractIndex([w0c, h0c, d0c])
                reader.SetExtractSize([w1c - w0c, h1c - h0c, d1c - d0c])
            # Else: empty bbox — fall through to a full-volume read.
            # The caller already handled None-bbox semantics; this
            # branch only fires on degenerate ROI inputs and is left
            # explicit so the decode still produces SOMETHING for
            # downstream code to error on with a clear message.
        if sitk_pixel is not None:
            # Force fp32/fp64 output; sitk applies scl_slope/scl_inter
            # during the cast — same semantics as
            # ``sitk.ReadImage(path, sitkFloat32)`` but on the
            # extracted sub-region only.
            reader.SetOutputPixelType(sitk_pixel)
        return reader.Execute()

    img = _sitk_read_with_retry(_read, path)
    # ``GetArrayViewFromImage`` returns a numpy array sharing memory
    # with the sitk image buffer — no copy. We MUST copy (into an
    # owned buffer) before dropping ``img`` or the view becomes
    # dangling memory.
    view = sitk.GetArrayViewFromImage(img)  # (Z, Y, X) = (D', H', W')
    # Force an explicit owned copy in the requested dtype / C order.
    # ``np.array(..., copy=True)`` materialises a fresh buffer every
    # call; a subsequent ``astype(copy=False)`` is free when dtypes
    # already match, so the cost is a single owned allocation.
    arr = np.array(view, copy=True, order="C")
    if arr.dtype != np_dtype:
        arr = arr.astype(np_dtype, copy=False)
    # Drop the sitk image buffer as early as possible.
    del view
    del img
    return arr


# ---------------------------------------------------------------------------
# NPZ pre-computed package I/O
# ---------------------------------------------------------------------------
# When ``DataConfig.npz_dir`` is configured, the dataset reads bbox-
# cropped image / label / region_weight plus pre-computed foreground
# indices directly from ``<pid>.npz`` (produced by
# ``segtask_v1.data.make_data``). The arrays inside are stored as raw
# uncompressed ``.npy`` payloads inside the zip (``ZIP_STORED``), so
# ``np.load(...)`` reads each array verbatim with NO gzip decompress
# step — the gzip peak that triggers ``bad allocation`` on the legacy
# NIfTI path on multi-worker concurrent decode is eliminated. Combined
# with bbox pre-cropping (~14× smaller working set on the lung CT
# example) and pre-computed foreground indices (no startup label
# scan), this makes the runtime data path dramatically more memory-
# stable. Note: numpy ignores ``mmap_mode`` for ``.npz`` archives
# (only ``.npy`` files support memmapped views) — each worker still
# materialises its own owned ndarray copy. The OS page cache shares
# the raw zip bytes across workers (only relevant after the first
# read of each volume).
#
# Contract (set by ``make_data``):
#   image       int16    (D', H', W')   raw HU, bbox-cropped
#   label       int16    (D', H', W')   raw labels, bbox-cropped
#   rw          float32  (D', H', W')   +1 shifted (optional key)
#   fg_slices   int32    (M,)
#   fg_coords   int32    (N, 3)         seed=42, capped at 50000
#   meta        object 0-d dict         provenance


def _open_npz(path: str) -> "np.lib.npyio.NpzFile":
    """Open ``path`` as an ``NpzFile`` (zip directory parsed; arrays
    NOT yet read).

    ``allow_pickle=True`` is required to deserialise the ``meta``
    dict; the rest of the arrays are plain numeric tensors. Note
    that numpy silently ignores ``mmap_mode`` on ``.npz``, so
    ``f['key']`` always returns an owned ndarray (the zip is
    ``ZIP_STORED`` here, so the read is a verbatim byte copy from
    the disk-resident zip into a fresh ndarray buffer — no gzip
    decompress, no transient peak).
    """
    return np.load(path, allow_pickle=True)


def load_npz_image(
    path: str,
    intensity_min: float,
    intensity_max: float,
    normalize: str,
    global_mean: float = 0.0,
    global_std: float = 1.0) -> np.ndarray:
    """Read ``image`` from a make_data npz, then run the regular
    ``preprocess_image`` pipeline.

    The npz stores the image as raw int16 HU (= uncalibrated NIfTI
    pixel values) so windowing parameters remain a runtime hyper-
    parameter. ``preprocess_image`` allocates a fresh fp32 buffer
    when fed an int16 input (the dtype mismatch path in
    ``np.asarray``), so the int16 buffer is held only briefly during
    the cast and is dropped on return — cache footprint matches the
    legacy NIfTI path (one owned fp32 ROI buffer per cached volume).
    """
    f = _open_npz(path)
    img_int16 = f["image"]   # owned int16 (zip-stored, no decompress)
    img = preprocess_image(
        img_int16, intensity_min, intensity_max,
        normalize, global_mean, global_std,
        inplace=False)       # returns owned fp32
    return img


def load_npz_label(path: str) -> np.ndarray:
    """Return the ``label`` array as an owned int16 ndarray.

    The npz read is a verbatim byte copy from the zip's
    ``ZIP_STORED`` entry into a fresh int16 buffer (no gzip
    decompress, no float promotion). Cache footprint matches the
    legacy NIfTI int16 cache exactly.
    """
    f = _open_npz(path)
    return f["label"]


def load_npz_region_weight(path: str) -> Optional[np.ndarray]:
    """Return the ``rw`` array as an owned float32 ndarray, or
    ``None`` if the npz does not include a region-weight payload.

    The +1 shift is already applied at make_data time, so the
    returned array is value-identical to
    ``load_region_weight_volume(rw_nifti_path, bbox=bbox)``.

    Storage dtype dispatch:
      * Newer npz packages store ``rw`` as ``int16`` (hand-
        annotated integer weights — 4× smaller on disk).
      * Older / fallback packages store ``rw`` as ``float32``
        (non-integer or out-of-int16-range sources).
    Both paths cast to ``float32`` here so the cache and the
    downstream loss multiplier stay uniformly fp32 — bit-equivalent
    to the legacy NIfTI ``load_region_weight_volume`` contract.
    """
    f = _open_npz(path)
    if "rw" not in f.files:
        return None
    rw = f["rw"]
    if rw.dtype != np.float32:
        rw = rw.astype(np.float32, copy=False)
    return rw


def npz_has_rw(path: str) -> bool:
    """Cheap presence test for the ``rw`` key (no array payload
    decoded). Used by the dataset's ``_has_region_weight_file``
    override in npz mode to honour the per-sample-file > static-
    mapping precedence rule."""
    f = _open_npz(path)
    return "rw" in f.files


def load_npz_fg_slices(path: str) -> np.ndarray:
    """Pre-computed per-z foreground index list (cropped frame)."""
    f = _open_npz(path)
    return np.asarray(f["fg_slices"], dtype=np.int32)


def load_npz_fg_coords(path: str) -> np.ndarray:
    """Pre-computed (N, 3) foreground voxel coords (cropped frame)."""
    f = _open_npz(path)
    return np.asarray(f["fg_coords"], dtype=np.int32)


def load_npz_label_for_split(path: str) -> np.ndarray:
    """Owned int16 copy of ``label`` — used by stratified-split /
    label-value detection helpers in ``loader.py``. We force an
    owned copy here because the helpers run BEFORE workers fork
    and we don't want the parent process to keep an open mmap
    handle per sample for the entire pre-flight scan.
    """
    f = _open_npz(path)
    return np.array(f["label"])


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_image(
    volume: np.ndarray,
    intensity_min: float,
    intensity_max: float,
    normalize: str,
    global_mean: float = 0.0,
    global_std: float = 1.0,
    inplace: bool = False) -> np.ndarray:
    """Intensity windowing + normalization → float32.

    Memory-conscious implementation: we allocate **one** float32 output
    buffer (= size of the input volume) and perform clip + affine
    normalization strictly in-place on it. The previous version chained
    ``np.clip`` → ``(vol - min) / denom`` → ``.astype(float32)`` and
    transiently held ~3× the volume size as temporaries, which — combined
    with the per-worker LRU volume cache — drove OOMs on large CT scans.

    Args:
        inplace: When True AND the input is already ``float32``, reuse the
            input buffer (saves one full-volume copy). Callers that own the
            input array — e.g. ``SegDataset3D._load_image``, which always
            passes a freshly-loaded buffer from ``load_nifti`` — should
            enable this. Default False preserves the legacy defensive-copy
            behaviour for callers that share arrays (tests, ``predictor``).
    """
    # Single allocation of the final float32 output buffer.
    vol = np.asarray(volume, dtype=np.float32)
    if vol is volume and not inplace:
        # Input was already float32 and caller hasn't explicitly opted
        # into in-place mutation; protect their array.
        vol = volume.copy()
    np.clip(vol, intensity_min, intensity_max, out=vol)

    if normalize == "minmax":
        denom = float(intensity_max - intensity_min)
        if denom > 0:
            vol -= float(intensity_min)
            vol /= denom
        else:
            vol.fill(0.0)
    elif normalize == "zscore":
        if global_std > 0:
            vol -= float(global_mean)
            vol /= float(global_std)
        else:
            vol.fill(0.0)
    else:
        raise ValueError(f"Unknown normalize: {normalize}")
    return vol


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


def load_region_weight_volume(
    path: str, bbox: "Optional[BBox]" = None) -> np.ndarray:
    """Load a per-sample region-weight NIfTI and apply the +1 shift.

    File convention (hand-annotated): background = 0, non-background
    voxels carry the desired weight value. We add +1 uniformly on load
    so background voxels become weight 1.0 and annotated voxels become
    ``w + 1`` — matching the semantics of ``compute_region_weight_map``
    (bg=1, fg>=1) so the loss stack's voxel-wise multiplicative weight
    behaves identically regardless of weight source.

    When ``bbox`` is supplied, the crop is materialised BEFORE the +1
    shift (via ``load_nifti_cropped``) so the arithmetic runs on the
    small ROI buffer rather than a full-volume float32 temporary —
    crucial for Host-OOM reduction when per-worker cache misses
    coincide with large CT volumes.

    Returns a float32 (D', H', W') array where the primed dims are the
    bbox extents (or the full volume dims when ``bbox is None``).
    """
    # Cropped decode avoids the legacy ``full-volume float32 decode +
    # full-volume ascontiguousarray crop copy'' sequence which peaked
    # at ``2x + crop''. See ``load_nifti_cropped'' docstring.
    rw = load_nifti_cropped(path, bbox=bbox, dtype=np.float32)
    rw += 1.0
    return rw


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
    return (vol[np.newaxis] == lv).astype(np.float32, copy=False)


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
    # ``scipy.ndimage.zoom`` already returns the output in the input
    # dtype; the trailing cast is a defensive pre-condition but
    # ``copy=False`` skips the redundant buffer allocation when dtypes
    # already match (the common case).
    return zoom(arr, factors, order=order).astype(arr.dtype, copy=False)


# ---------------------------------------------------------------------------
# Bounding-box helpers (optional ROI cropping of image / label volumes)
# ---------------------------------------------------------------------------
# Per-sample bbox tuple convention (also used by `apply_bbox` below):
#   ((d0, d1), (h0, h1), (w0, w1)) — half-open like Python slices.
BBox = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]


def compute_bbox_from_volume(vol: np.ndarray) -> Optional[BBox]:
    """Compute the axis-aligned bounding box of the nonzero region of a
    (D, H, W) ROI mask.

    Returns ``None`` when the mask is entirely empty (caller should
    fall back to using the full volume).

    Implementation note: we collapse the mask along two axes at a time
    via ``np.any`` instead of ``np.argwhere``. For large CT volumes this
    is dramatically faster (O(D*H*W) bool reductions vs. materialising
    a (N, 3) coord array) and uses negligible memory.
    """
    if vol.ndim != 3:
        raise ValueError(f"BBox volume must be 3D (D,H,W), got {vol.ndim}D")
    mask = np.round(vol).astype(np.int16) != 0
    if not mask.any():
        return None
    d_any = np.any(mask, axis=(1, 2))
    h_any = np.any(mask, axis=(0, 2))
    w_any = np.any(mask, axis=(0, 1))

    def _span(flat: np.ndarray) -> Tuple[int, int]:
        idx = np.where(flat)[0]
        return int(idx[0]), int(idx[-1]) + 1  # half-open

    return (_span(d_any), _span(h_any), _span(w_any))


def apply_bbox(vol: np.ndarray, bbox: Optional[BBox]) -> np.ndarray:
    """Crop a (D, H, W) volume to ``bbox``. ``None`` returns ``vol``
    unchanged (used for samples whose ROI mask was empty)."""
    if bbox is None:
        return vol
    (d0, d1), (h0, h1), (w0, w1) = bbox
    return vol[d0:d1, h0:h1, w0:w1]


def precompute_bboxes(bbox_paths: List[str]) -> List[Optional[BBox]]:
    """Load every ROI mask once, compute its bbox, log the mean bbox
    size across the dataset, and return the per-sample bbox list.

    Empty / missing-foreground masks are kept as ``None`` (the dataset
    will then fall back to the uncropped volume for that sample) and
    counted separately in the log line so silent ROI failures surface
    immediately.
    """
    bboxes: List[Optional[BBox]] = []
    sizes: List[Tuple[int, int, int]] = []
    n_empty = 0
    for p in bbox_paths:
        # BBox masks are binary/small-int — int16 decode is 4× cheaper
        # than the default float32 and ``compute_bbox_from_volume`` does
        # ``np.round().astype(int32) != 0`` which is a no-op on ints.
        bb = compute_bbox_from_volume(load_nifti(p, dtype=np.int16))
        bboxes.append(bb)
        if bb is None:
            n_empty += 1
            continue
        (d0, d1), (h0, h1), (w0, w1) = bb
        sizes.append((d1 - d0, h1 - h0, w1 - w0))

    if sizes:
        arr = np.asarray(sizes, dtype=np.float64)
        mean = arr.mean(axis=0)
        mn = arr.min(axis=0)
        mx = arr.max(axis=0)
        logger.info(
            "BBox precomputed: %d/%d masks have foreground; mean (D,H,W)="
            "(%.1f, %.1f, %.1f), min=(%d, %d, %d), max=(%d, %d, %d)",
            len(sizes), len(bbox_paths),
            mean[0], mean[1], mean[2],
            int(mn[0]), int(mn[1]), int(mn[2]),
            int(mx[0]), int(mx[1]), int(mx[2]))
    if n_empty:
        logger.warning(
            "BBox: %d/%d masks were entirely empty; falling back to the "
            "full volume for those samples.", n_empty, len(bbox_paths))
    return bboxes


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
        image_paths                : List[str],
        label_paths                : List[str],
        label_values               : List[int],
        patch_size                 : Tuple[int, int, int] = (64, 128, 128),
        aug_oversample_ratio       : float = 1.0,
        multi_res_scales           : Optional[List[float]] = None,
        intensity_min              : float = -1024.0,
        intensity_max              : float = 3071.0,
        normalize                  : str = "minmax",
        global_mean                : float = 0.0,
        global_std                 : float = 1.0,
        foreground_oversample_ratio: float = 0.5,
        samples_per_volume         : int = 8,
        is_train                   : bool = True,
        cache_enabled              : bool = True,
        cache_max_volumes          : int = 0,
        region_weights             : Optional[List[float]] = None,
        bbox_paths                 : Optional[List[str]] = None,
        region_weight_paths        : Optional[List[str]] = None,
        z_boundary_mode            : str = "stretch",
        aux_keep_native_d          : bool = False,
        keep_native_multi_res      : bool = False,
        npz_paths                  : Optional[List[str]] = None):

        super().__init__()
        assert len(image_paths) == len(label_paths)
        assert aug_oversample_ratio >= 1.0, (
            f"aug_oversample_ratio must be >= 1.0, got {aug_oversample_ratio}")
        if z_boundary_mode not in ("stretch", "edge_pad"):
            raise ValueError(
                f"z_boundary_mode must be 'stretch' or 'edge_pad', "
                f"got {z_boundary_mode!r}")
        self.image_paths  = image_paths
        self.label_paths  = label_paths
        self.label_values = label_values
        self.patch_size   = tuple(patch_size)
        self.oversample   = float(aug_oversample_ratio)
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
        self.intensity_min      = intensity_min
        self.intensity_max      = intensity_max
        self.normalize          = normalize
        self.global_mean        = global_mean
        self.global_std         = global_std
        self.fg_ratio           = foreground_oversample_ratio
        self.samples_per_volume = samples_per_volume
        self.is_train           = is_train
        self.region_weights     = region_weights
        # Boundary handling for the scale=1.0 z-window. ``stretch`` keeps the
        # legacy "clamp + resize-stretch" behaviour; ``edge_pad`` switches
        # to ``extract_z_patch_padded`` so every window has exactly eD
        # physical-1-slice-spacing slices, matching the ``scale > 1.0``
        # contract and the inference predictor under the same toggle.
        self.z_boundary_mode    = z_boundary_mode

        # ---- Native-depth multi-FOV path (2.5D + aux_seg_supervision) ----
        # When True (validated upstream in Config.validate to be only with
        # 2_5d + n_views > 1), __getitem__ takes a SIMPLIFIED single-cube
        # path instead of stacking per-view resampled views:
        #
        #   1. Extract ONE cube of depth ``round(eD * max_scale)`` around
        #      the sampled z-center, edge-padded — this is the largest
        #      physical FOV needed by any view. (View 0 is the centered
        #      ``D`` slices; view k is the centered ``D_k`` slices.)
        #   2. Resize H, W to (eH, eW) — D axis is left at native depth.
        #   3. Output shape ``(1, round(eD * max_scale), eH, eW)`` so the
        #      existing GPU augmentor — which expects a (B, 1, D, H, W)
        #      cube — runs without modification on the largest physical
        #      FOV. The trainer center-crops per view *after* augment.
        #
        # Geometry: because all views share the same z-center and unit
        # slice spacing, center-cropping ``D_k`` slices from the max-FOV
        # cube produces THE SAME physical slice set as the per-view
        # independent extraction in the False-path — voxel-equivalent up
        # to numerical noise. The benefits over the False-path:
        #   * single shared augmentation field (cross-view geometric
        #     consistency by construction);
        #   * no z-axis resampling for aux views (full information);
        #   * lower memory (one max-FOV cube instead of K stacked copies).
        self.aux_keep_native_d = bool(aux_keep_native_d)
        if self.aux_keep_native_d:
            assert len(self.multi_res_scales) > 1, (
                "aux_keep_native_d=True requires len(multi_res_scales) > 1; "
                f"got {self.multi_res_scales}")
            assert self.multi_res_scales[0] == 1.0, (
                "aux_keep_native_d=True requires multi_res_scales[0] == 1.0 "
                "(view 0 is the supervision target); got "
                f"{self.multi_res_scales}")
            assert self.z_boundary_mode == "edge_pad", (
                "aux_keep_native_d=True requires z_boundary_mode='edge_pad'; "
                f"got {self.z_boundary_mode!r}.")
            self._max_scale = float(max(self.multi_res_scales))
        else:
            self._max_scale = 1.0

        # ---- 3D z_axis lazy single-max-FOV-cube path ------------------
        # When True (validated upstream), ``__getitem__`` emits ONE
        # max-FOV cube of depth ``round(eD * max_scale)`` (edge-padded)
        # at full canonical H/W, shape ``(1, eD_max, eH, eW)``. The
        # trainer (R2) center-crops per view at native physical depth
        # ``D_k = round(eD * s_k)`` and resizes each view back to ``eD``
        # immediately before forward, finally producing the standard
        # ``(B, C_res, eD, eH, eW)`` 3D model input.
        #
        # Mutually exclusive with ``aux_keep_native_d`` (2.5D analogue).
        # Both flags share the same ``_max_scale`` book-keeping; we
        # therefore consolidate the OR of the two when setting it.
        self.keep_native_multi_res = bool(keep_native_multi_res)
        if self.keep_native_multi_res:
            assert not self.aux_keep_native_d, (
                "keep_native_multi_res and aux_keep_native_d are mutually "
                "exclusive (3D vs 2.5D analogues).")
            assert len(self.multi_res_scales) > 1, (
                "keep_native_multi_res=True requires len(multi_res_scales) > 1; "
                f"got {self.multi_res_scales}")
            assert self.multi_res_scales[0] == 1.0, (
                "keep_native_multi_res=True requires multi_res_scales[0] == 1.0 "
                f"(canonical view); got {self.multi_res_scales}")
            assert self.z_boundary_mode == "edge_pad", (
                "keep_native_multi_res=True (z_axis) requires "
                f"z_boundary_mode='edge_pad'; got {self.z_boundary_mode!r}.")
            self._max_scale = float(max(self.multi_res_scales))

        self._img_cache = VolumeCache(cache_enabled, cache_max_volumes)
        self._lbl_cache = VolumeCache(cache_enabled, cache_max_volumes)

        # Optional ROI bbox cropping. When `bbox_paths` is supplied, we
        # precompute one bbox per sample upfront (logging the dataset
        # mean / min / max bbox size) and apply it inside `_load_image`
        # / `_load_label` BEFORE caching so the cached volumes are
        # already cropped — saving both RAM and downstream compute.
        if bbox_paths is not None:
            assert len(bbox_paths) == len(image_paths), (
                f"bbox_paths length {len(bbox_paths)} != image_paths "
                f"length {len(image_paths)}")
        self.bbox_paths = bbox_paths
        self._bboxes: Optional[List[Optional[BBox]]] = (
            precompute_bboxes(bbox_paths) if bbox_paths else None)

        # Optional per-sample region-weight NIfTI. When supplied, takes
        # precedence over ``region_weights`` at __getitem__ time — we load
        # the file, apply the +1 shift (bg=1, fg=w+1), bbox-crop to match
        # image/label, and cache. Extraction / resize paths mirror the
        # IMAGE pipeline (continuous values, not labels) so interpolation
        # is linear instead of nearest — preserving hand-annotated weight
        # gradients that would otherwise be quantised.
        if region_weight_paths is not None:
            assert len(region_weight_paths) == len(image_paths), (
                f"region_weight_paths length {len(region_weight_paths)} != "
                f"image_paths length {len(image_paths)}")
        self.region_weight_paths = region_weight_paths
        self._rw_cache = VolumeCache(cache_enabled, cache_max_volumes)

        # ---- NPZ pre-computed package mode ---------------------------
        # When ``npz_paths`` is supplied (1:1 with image_paths), every
        # ``_load_*`` and ``_build_index`` path dispatches to the npz
        # readers. Bbox is ALREADY applied inside the npz; any
        # ``bbox_paths`` argument is ignored with a warning. The
        # per-sample rw key-presence is cached here lazily so
        # ``_has_region_weight_file`` (which is hit per-__getitem__)
        # does not reopen the zip on every call.
        self._npz_paths: Optional[List[str]] = (
            list(npz_paths) if npz_paths is not None else None)
        if self._npz_paths is not None:
            assert len(self._npz_paths) == len(image_paths), (
                f"npz_paths length {len(self._npz_paths)} != image_paths "
                f"length {len(image_paths)}")
            if self._bboxes is not None:
                logger.warning(
                    "npz mode: ignoring supplied bbox_paths (bbox is "
                    "already pre-applied inside the npz packages).")
                self._bboxes = None
                self.bbox_paths = None
        self._npz_has_rw_cache: Dict[int, bool] = {}

        # Build per-slice index for foreground oversampling
        self._vol_fg_slices: List[np.ndarray] = []  # fg slice indices per volume
        self._vol_all_slices: List[int] = []        # total depth per volume
        self._build_index()                         # D维度前景坐标

    def _build_index(self) -> None:
        """Scan all volumes and record which slices have foreground."""
        if self._npz_paths is not None:
            self._build_index_from_npz()
            return
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

    def _build_index_from_npz(self) -> None:
        """NPZ-mode index: read ``fg_slices`` directly from each
        package and use the stored ``image`` array shape for the
        per-volume depth — no label scan, no bbox decode.
        """
        logger.info(
            "Loading pre-computed fg indices from %d npz packages...",
            len(self._npz_paths))
        total_fg = 0
        total_slices = 0
        for path in self._npz_paths:
            f = _open_npz(path)
            fg = np.asarray(f["fg_slices"], dtype=np.int32)
            D = int(f["image"].shape[0])
            self._vol_fg_slices.append(fg)
            self._vol_all_slices.append(D)
            total_fg += len(fg)
            total_slices += D
        logger.info(
            "NPZ index built: %d volumes, %d/%d foreground slices",
            len(self._npz_paths), total_fg, total_slices)

    def _bbox_for(self, vol_idx: int) -> Optional[BBox]:
        return self._bboxes[vol_idx] if self._bboxes is not None else None

    def _load_image(self, vol_idx: int) -> np.ndarray:
        """Load and preprocess image volume with caching."""
        if self._npz_paths is not None:
            path = self._npz_paths[vol_idx]
            cached = self._img_cache.get(path)
            if cached is not None:
                return cached
            img = load_npz_image(
                path, self.intensity_min, self.intensity_max,
                self.normalize, self.global_mean, self.global_std)
            self._img_cache.put(path, img)
            return img
        path   = self.image_paths[vol_idx]
        cached = self._img_cache.get(path)
        if cached is not None:
            return cached
        bb = self._bbox_for(vol_idx)
        # ``load_nifti_cropped`` fuses decode + bbox crop into a single
        # owned-buffer allocation, halving peak RAM per load compared
        # with the legacy ``load_nifti → apply_bbox → ascontiguousarray``
        # path (which transiently held 2x full-volume float32 buffers
        # — the main driver of the ``SimpleITK bad allocation'' host
        # OOM on large CT scans).
        img = load_nifti_cropped(path, bbox=bb, dtype=np.float32)
        img = preprocess_image(  # 归一化 (in-place on the owned buffer)
            img, self.intensity_min, self.intensity_max,
            self.normalize, self.global_mean, self.global_std,
            inplace=True)
        self._img_cache.put(path, img)
        return img

    def _load_label(self, vol_idx: int) -> np.ndarray:
        """Load raw label volume with caching.

        Labels are decoded as ``int16`` (not ``float32``) — they are small
        integer class indices, storing them as float32 wasted 4× RAM in
        the per-worker volume cache and 4× CPU→GPU bandwidth for every
        batch. Every downstream op is dtype-agnostic (``np.round`` is a
        no-op on int arrays; ``resize_3d(..., is_label=True)`` uses
        nearest interpolation which preserves dtype; the final per-sample
        stack casts to float32 exactly once at tensor-emission time).
        """
        if self._npz_paths is not None:
            path = self._npz_paths[vol_idx]
            cached = self._lbl_cache.get(path)
            if cached is not None:
                return cached
            lbl = load_npz_label(path)
            self._lbl_cache.put(path, lbl)
            return lbl
        path   = self.label_paths[vol_idx]
        cached = self._lbl_cache.get(path)
        if cached is not None:
            return cached
        # Fused decode + bbox crop. The previous ``load_nifti + apply_bbox``
        # stored a VIEW (``arr.base`` pointing to the full int16 buffer)
        # in the cache, silently retaining the full volume's RAM for the
        # lifetime of the cache entry. ``load_nifti_cropped`` always
        # returns an owned, C-contiguous buffer — the cached footprint
        # now matches the logged estimate.
        lbl = load_nifti_cropped(
            path, bbox=self._bbox_for(vol_idx), dtype=np.int16)
        self._lbl_cache.put(path, lbl)
        return lbl

    def _has_region_weight_file(self, vol_idx: int) -> bool:
        if self._npz_paths is not None:
            cached = self._npz_has_rw_cache.get(vol_idx)
            if cached is None:
                cached = npz_has_rw(self._npz_paths[vol_idx])
                self._npz_has_rw_cache[vol_idx] = cached
            return cached
        return (self.region_weight_paths is not None
                and self.region_weight_paths[vol_idx] is not None
                and self.region_weight_paths[vol_idx] != "")

    def _load_region_weight(self, vol_idx: int) -> np.ndarray:
        """Load per-sample region-weight volume (bbox-cropped, +1 shifted)."""
        if self._npz_paths is not None:
            path = self._npz_paths[vol_idx]
            cached = self._rw_cache.get(path)
            if cached is not None:
                return cached
            rw = load_npz_region_weight(path)
            # Caller guards with ``_has_region_weight_file``; rw
            # should always be non-None here, but stay defensive.
            if rw is not None:
                self._rw_cache.put(path, rw)
            return rw
        path   = self.region_weight_paths[vol_idx]
        cached = self._rw_cache.get(path)
        if cached is not None:
            return cached
        # Bbox-aware load: the +1 shift is applied on the ROI buffer
        # (not the full volume) — avoids a transient full-volume
        # float32 arithmetic temporary.
        rw = load_region_weight_volume(
            path, bbox=self._bbox_for(vol_idx))
        self._rw_cache.put(path, rw)
        return rw

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

        # Per-sample region-weight file takes precedence over the static
        # ``region_weights`` mapping. We load the (already bbox-cropped,
        # +1-shifted) volume once per __getitem__ call; the same z-window
        # / resize is then applied per scale, so the resulting weight map
        # stays spatially aligned with image and label channels.
        rw_vol = (self._load_region_weight(vol_idx)
                  if self._has_region_weight_file(vol_idx) else None)

        # ---- Native-depth multi-FOV simplified path ---------------------
        # Single max-FOV cube extraction; the trainer center-crops per view
        # after GPU augmentation. Output shape ``(1, eD_max, eH, eW)`` so
        # the existing rank-5 (B, 1, D, H, W) augmentor pipeline applies
        # unchanged. ``eD_max = round(eD * max_scale)`` covers the largest
        # required physical z-FOV; smaller views get center crops at native
        # depth ``D_k = round(eD * s_k)`` which by construction (shared z
        # center, unit slice spacing) matches the per-view independent
        # extraction of the legacy False-path voxel-for-voxel.
        if self.aux_keep_native_d:
            return self._getitem_native_d(vol_idx, img, lbl, rw_vol, z, eD, eH, eW)

        # ---- 3D lazy single-max-FOV-cube path (z_axis) -----------------
        # Same emission shape contract as the legacy multi-res output
        # except the leading ``C_res`` axis is collapsed to 1 and the
        # depth axis carries ``eD_max = round(eD * max_scale)`` slices
        # at native physical resolution. The trainer's per-view crop+
        # resize step (R2) reproduces the legacy ``(B, C_res, eD, eH,
        # eW)`` model input exactly — center-cropping ``D_k`` slices
        # from ``eD_max`` then resizing each view back to ``eD`` is
        # voxel-equivalent to the per-view independent extraction
        # done by the False-path (modulo ONE interpolation pass instead
        # of two: the dataset zoom is gone, the augment grid_sample is
        # the only resampling stage).
        if self.keep_native_multi_res:
            return self._getitem_native_multi_res_z(vol_idx, img, lbl, rw_vol, z, eD, eH, eW)

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
            # Dispatch the z-extraction:
            #   - scale > 1.0 ALWAYS uses edge-replicate padding (preserves
            #     the requested physical z-FOV exactly).
            #   - scale == 1.0 honours ``z_boundary_mode``:
            #       * "stretch"  -> legacy clamp path (may return < D_s
            #                       slices that the resize below stretches
            #                       back to eD, producing the train/test
            #                       physical-spacing mismatch documented
            #                       on ``DataConfig.z_boundary_mode``);
            #       * "edge_pad" -> same padded path as scale > 1.0 so the
            #                       window has EXACTLY D_s physical-1-slice
            #                       slices regardless of where ``z`` sits.
            use_padded = (scale != 1.0) or (self.z_boundary_mode == "edge_pad")
            if use_padded:
                img_s, lbl_s = self._extract_z_patch_padded(img, lbl, z, D_s)
            else:
                img_s, lbl_s = self._extract_z_patch(img, lbl, z, D_s)
            rw_s = (self._extract_z_single(rw_vol, z, D_s, use_padded)
                    if rw_vol is not None else None)

            # Resize in a single 3D zoom:
            #   (actual_d, H_vol, W_vol) → (eD, pH, pW)
            # H_vol, W_vol collapse directly to patch_size (pH, pW) —
            # matching `predictor._sliding_window_z`.
            img_s = resize_3d(img_s, eD, eH, eW, is_label=False)
            lbl_s = resize_3d(lbl_s, eD, eH, eW, is_label=True)
            img_channels.append(img_s)
            lbl_channels.append(lbl_s)

            # Region-weight precedence: per-sample file > static mapping.
            if rw_s is not None:
                # Linear resize (continuous weights) — preserves hand-
                # annotated gradients that nearest would quantise.
                wmap_s = resize_3d(rw_s, eD, eH, eW, is_label=False)
                wmap_channels.append(wmap_s)
            elif self.region_weights:
                wmap_s = compute_region_weight_map(
                    lbl_s, self.label_values, self.region_weights)
                wmap_channels.append(wmap_s[0])  # drop the leading 1

        # Stack scales as channel 0 → (C_res, eD, pH, pW). For the legacy
        # single-res default (`multi_res_scales=[1.0]`), C_res == 1 and
        # the shape is identical to the pre-multires z-axis output.
        # P8: label stays int16 through CPU→GPU transfer (half the PCIe
        # bandwidth vs float32). The trainer casts to float on the GPU
        # right after ``.to(device)``. Image / weight_map remain float32
        # because (a) autocast needs a float input and (b) they carry
        # continuous values where int quantisation is lossy.
        result = {
            "image": torch.from_numpy(
                np.stack(img_channels, axis=0).astype(np.float32, copy=False)),
            "label": torch.from_numpy(
                np.ascontiguousarray(np.stack(lbl_channels, axis=0)))}
        if wmap_channels:
            result["weight_map"] = torch.from_numpy(
                np.stack(wmap_channels, axis=0).astype(np.float32, copy=False))
        return result

    def _getitem_native_d(
        self,
        vol_idx: int,
        img: np.ndarray,
        lbl: np.ndarray,
        rw_vol: Optional[np.ndarray],
        z: int,
        eD: int,
        eH: int,
        eW: int,
    ) -> Dict[str, torch.Tensor]:
        """Single max-FOV cube emission for the native-depth multi-FOV path.

        Output is shape-equivalent to the single-resolution legacy path —
        ``(1, eD_max, eH, eW)`` for image/label/(weight_map) — except eD
        is replaced by ``eD_max = round(eD * max_scale)``. This keeps the
        downstream collate / augmentor / center-crop / 2.5D-squeeze
        contract IDENTICAL to the single-resolution case; the only
        bespoke step happens in the trainer right before the model
        forward, where ``_split_views_native_d`` center-crops the cube
        per view at native depth ``D_k``.

        Why a single cube instead of a per-view stack:
          * Augmentation runs ONCE on one shared cube → views are by
            construction warp-consistent (same affine, same elastic).
          * No z-axis resampling for aux views (the legacy False-path
            compresses ``round(eD * s_k) → eD`` and loses information).
          * Lower memory than emitting K stacked copies.

        Region-weight semantics: the per-sample weight volume (when
        provided) is extracted with the SAME z-padded path and uses
        linear resize for H, W (preserves continuous weight gradients),
        mirroring the False-path's per-scale region-weight rule.
        """
        eD_max = int(round(eD * self._max_scale))
        # Edge-padded extraction guarantees exactly eD_max physical-1-slice
        # spacing slices regardless of where ``z`` sits relative to the
        # volume boundary — required for ``aux_keep_native_d`` since the
        # trainer's per-view center crop assumes uniform unit z spacing.
        img_s, lbl_s = self._extract_z_patch_padded(img, lbl, z, eD_max)
        rw_s = (self._extract_z_single(rw_vol, z, eD_max, use_padded=True)
                if rw_vol is not None else None)

        # In-plane resize to (eH, eW); D axis preserved at eD_max.
        img_s = resize_3d(img_s, eD_max, eH, eW, is_label=False)
        lbl_s = resize_3d(lbl_s, eD_max, eH, eW, is_label=True)
        result = {
            # ``(1, eD_max, eH, eW)`` — leading axis kept for parity with
            # the legacy ``(C_res, eD, eH, eW)`` output so the collate /
            # augmentor / squeeze pipeline does NOT need to special-case
            # this path. The "1" is the conventional C_res dim with
            # n_views collapsed to a single physical cube.
            "image": torch.from_numpy(img_s[None].astype(np.float32, copy=False)),
            # P8: int16 label (see __getitem__ for rationale).
            "label": torch.from_numpy(np.ascontiguousarray(lbl_s[None])),
        }
        # Region-weight precedence: per-sample file > static mapping.
        if rw_s is not None:
            wmap_s = resize_3d(rw_s, eD_max, eH, eW, is_label=False)
            result["weight_map"] = torch.from_numpy(
                wmap_s[None].astype(np.float32, copy=False))
        elif self.region_weights:
            wmap_s = compute_region_weight_map(
                lbl_s, self.label_values, self.region_weights)
            # ``compute_region_weight_map`` returns (1, eD_max, eH, eW).
            result["weight_map"] = torch.from_numpy(
                wmap_s.astype(np.float32, copy=False))
        return result

    def _getitem_native_multi_res_z(
        self,
        vol_idx: int,
        img: np.ndarray,
        lbl: np.ndarray,
        rw_vol: Optional[np.ndarray],
        z: int,
        eD: int,
        eH: int,
        eW: int,
    ) -> Dict[str, torch.Tensor]:
        """Single max-FOV cube emission for the 3D z_axis lazy path.

        Output shape: ``(1, eD_max, eH, eW)`` for image/label/(weight_map)
        with ``eD_max = round(eD * max_scale)``. The leading "1" stands in
        for the legacy ``C_res`` axis (collapsed: views are reconstructed
        downstream by the trainer instead of stacked here).

        Geometric contract
        ------------------
        * z-axis: ``extract_z_patch_padded`` always (regardless of
          boundary), so the cube has EXACTLY ``eD_max`` physical-1-slice
          spacing slices centered on the sampled z. Required so the
          trainer can center-crop arbitrary ``D_k <= eD_max`` slices
          and trust uniform z spacing across views.
        * H, W: full canonical resize to (eH, eW) — same as the
          False-path, so post-augment the in-plane geometry is
          identical (the False-path also resizes H,W per view to (eH,
          eW); since z_axis multi-res only scales z, the per-view
          in-plane targets are all (eH, eW), making the lazy path
          numerically equivalent on H, W).

        Equivalence with the False-path
        -------------------------------
        For view k with native depth ``D_k = round(eD * s_k) <= eD_max``,
        center-cropping ``D_k`` slices from this cube and z-resizing
        them to ``eD`` reproduces the False-path's per-view extraction
        ``extract_z_patch_padded(img, z, D_k) → resize_3d(..., eD, eH,
        eW)`` voxel-for-voxel — both paths share the same z-center and
        the same edge-padding rule. The R2 trainer step does exactly
        this crop+resize on the GPU (one ``F.interpolate`` per view).

        Region weights are extracted with the same z-padded path and
        linearly resized in-plane to (eH, eW), mirroring the
        False-path's per-scale region-weight rule.
        """
        eD_max = int(round(eD * self._max_scale))
        # Always edge-padded — uniform 1-slice z spacing is a hard
        # requirement for the trainer's per-view center-crop.
        img_s, lbl_s = self._extract_z_patch_padded(img, lbl, z, eD_max)
        rw_s = (self._extract_z_single(rw_vol, z, eD_max, use_padded=True)
                if rw_vol is not None else None)

        # In-plane resize to (eH, eW); D axis preserved at eD_max.
        img_s = resize_3d(img_s, eD_max, eH, eW, is_label=False)
        lbl_s = resize_3d(lbl_s, eD_max, eH, eW, is_label=True)

        result = {
            # Leading "1" = collapsed C_res. Trainer (R2) splits along
            # depth and resizes back to ``(B, C_res, eD, eH, eW)``.
            "image": torch.from_numpy(img_s[None].astype(np.float32, copy=False)),
            # P8: int16 label (see __getitem__ for rationale).
            "label": torch.from_numpy(np.ascontiguousarray(lbl_s[None])),
        }
        # Region-weight precedence: per-sample file > static mapping.
        if rw_s is not None:
            wmap_s = resize_3d(rw_s, eD_max, eH, eW, is_label=False)
            result["weight_map"] = torch.from_numpy(
                wmap_s[None].astype(np.float32, copy=False))
        elif self.region_weights:
            wmap_s = compute_region_weight_map(
                lbl_s, self.label_values, self.region_weights)
            # ``compute_region_weight_map`` returns shape (1, eD_max, eH, eW).
            result["weight_map"] = torch.from_numpy(
                wmap_s.astype(np.float32, copy=False))
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

    def _extract_z_single(
        self, vol: np.ndarray, z_center: int, D_patch: int,
        use_padded: bool) -> np.ndarray:
        """Single-volume z-patch extraction mirroring the paired helpers.

        Used for the region-weight volume so it stays spatially aligned
        with the image / label extraction at the same ``z_center`` and
        ``D_patch`` under both z-boundary modes.
        """
        if use_padded:
            return extract_z_patch_padded(vol, z_center, D_patch)
        D_vol   = vol.shape[0]
        half    = D_patch // 2
        d_start = max(0, z_center - half)
        d_end   = min(D_vol, d_start + D_patch)
        return vol[d_start:d_end].copy()


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
        region_weights: Optional[List[float]] = None,
        bbox_paths: Optional[List[str]] = None,
        region_weight_paths: Optional[List[str]] = None,
        keep_native_multi_res: bool = False,
        npz_paths: Optional[List[str]] = None):
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

        # ---- 3D cubic lazy single-max-FOV-cube path -------------------
        # When True (validated upstream in Config.validate), __getitem__
        # emits ONE max-FOV cube of size ``round(extract_size *
        # max_scale)`` around the sampled centre, shape ``(1, eD_max,
        # eH_max, eW_max)``. The trainer (R2) center-crops per view at
        # native physical size ``round(extract_size * s_k)`` and resizes
        # each view back to ``extract_size`` immediately before forward,
        # producing the standard ``(B, C_res, eD, eH, eW)`` 3D model
        # input.
        #
        # Compared to the legacy multi-res path, this saves K-1 per-view
        # ``scipy.ndimage.zoom`` calls in the CPU dataset workers and
        # gives augmentation a single shared grid_sample over the full
        # max-FOV cube (cross-view warp consistency by construction).
        self.keep_native_multi_res = bool(keep_native_multi_res)
        if self.keep_native_multi_res:
            assert len(self.multi_res_scales) > 1, (
                "keep_native_multi_res=True requires len(multi_res_scales) > 1; "
                f"got {self.multi_res_scales}")
            assert self.multi_res_scales[0] == 1.0, (
                "keep_native_multi_res=True requires multi_res_scales[0] == 1.0 "
                f"(canonical view); got {self.multi_res_scales}")
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

        # Optional ROI bbox cropping. When `bbox_paths` is supplied, we
        # precompute one bbox per sample upfront (logging the dataset
        # mean / min / max bbox size) and apply it inside `_load_image`
        # / `_load_label` BEFORE caching so the cached volumes are
        # already cropped — saving both RAM and downstream compute.
        if bbox_paths is not None:
            assert len(bbox_paths) == len(image_paths), (
                f"bbox_paths length {len(bbox_paths)} != image_paths "
                f"length {len(image_paths)}")
        self.bbox_paths = bbox_paths
        self._bboxes: Optional[List[Optional[BBox]]] = (
            precompute_bboxes(bbox_paths) if bbox_paths else None)

        # Optional per-sample region-weight NIfTI (see SegDataset3D for
        # the full contract). File precedence over ``region_weights``.
        if region_weight_paths is not None:
            assert len(region_weight_paths) == len(image_paths), (
                f"region_weight_paths length {len(region_weight_paths)} != "
                f"image_paths length {len(image_paths)}")
        self.region_weight_paths = region_weight_paths
        self._rw_cache = VolumeCache(cache_enabled, cache_max_volumes)

        # ---- NPZ pre-computed package mode (see SegDataset3D) -------
        self._npz_paths: Optional[List[str]] = (
            list(npz_paths) if npz_paths is not None else None)
        if self._npz_paths is not None:
            assert len(self._npz_paths) == len(image_paths), (
                f"npz_paths length {len(self._npz_paths)} != image_paths "
                f"length {len(image_paths)}")
            if self._bboxes is not None:
                logger.warning(
                    "npz mode: ignoring supplied bbox_paths (bbox is "
                    "already pre-applied inside the npz packages).")
                self._bboxes = None
                self.bbox_paths = None
        self._npz_has_rw_cache: Dict[int, bool] = {}

        # Build 3D foreground voxel index for oversampling
        self._vol_shapes: List[Tuple[int, int, int]] = []
        self._vol_fg_coords: List[np.ndarray] = []  # (N, 3) fg voxel coords per volume
        self._build_index()

    def _build_index(self) -> None:
        """Scan volumes and record foreground voxel coordinates."""
        if self._npz_paths is not None:
            self._build_index_from_npz()
            return
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

    def _build_index_from_npz(self) -> None:
        """NPZ-mode index: read ``fg_coords`` and stored shape from
        each npz package directly. Sub-sampling has already been
        applied by ``make_data`` (seed=42, cap=50000) so the cubic
        center sampler is bit-equivalent to the legacy on-the-fly
        path.
        """
        logger.info(
            "Loading pre-computed fg coords from %d npz packages...",
            len(self._npz_paths))
        total_fg = 0
        for path in self._npz_paths:
            f = _open_npz(path)
            coords = np.asarray(f["fg_coords"], dtype=np.int32)
            shape = tuple(int(s) for s in f["image"].shape)
            self._vol_shapes.append(shape)
            self._vol_fg_coords.append(coords)
            total_fg += len(coords)
        logger.info(
            "NPZ cubic index: %d volumes, %d fg voxels sampled",
            len(self._npz_paths), total_fg)

    def _bbox_for(self, vol_idx: int) -> Optional[BBox]:
        return self._bboxes[vol_idx] if self._bboxes is not None else None

    def _load_image(self, vol_idx: int) -> np.ndarray:
        if self._npz_paths is not None:
            path = self._npz_paths[vol_idx]
            cached = self._img_cache.get(path)
            if cached is not None:
                return cached
            img = load_npz_image(
                path, self.intensity_min, self.intensity_max,
                self.normalize, self.global_mean, self.global_std)
            self._img_cache.put(path, img)
            return img
        path = self.image_paths[vol_idx]
        cached = self._img_cache.get(path)
        if cached is not None:
            return cached
        # Fused decode + bbox crop — see ``SegDataset3D._load_image''
        # for the host-OOM rationale.
        img = load_nifti_cropped(
            path, bbox=self._bbox_for(vol_idx), dtype=np.float32)
        img = preprocess_image(
            img, self.intensity_min, self.intensity_max,
            self.normalize, self.global_mean, self.global_std,
            inplace=True)
        self._img_cache.put(path, img)
        return img

    def _load_label(self, vol_idx: int) -> np.ndarray:
        """Load raw label volume with caching (int16 for RAM savings).

        See ``SegDataset3D._load_label`` for the dtype rationale.
        """
        if self._npz_paths is not None:
            path = self._npz_paths[vol_idx]
            cached = self._lbl_cache.get(path)
            if cached is not None:
                return cached
            lbl = load_npz_label(path)
            self._lbl_cache.put(path, lbl)
            return lbl
        path = self.label_paths[vol_idx]
        cached = self._lbl_cache.get(path)
        if cached is not None:
            return cached
        lbl = load_nifti_cropped(
            path, bbox=self._bbox_for(vol_idx), dtype=np.int16)
        self._lbl_cache.put(path, lbl)
        return lbl

    def _has_region_weight_file(self, vol_idx: int) -> bool:
        if self._npz_paths is not None:
            cached = self._npz_has_rw_cache.get(vol_idx)
            if cached is None:
                cached = npz_has_rw(self._npz_paths[vol_idx])
                self._npz_has_rw_cache[vol_idx] = cached
            return cached
        return (self.region_weight_paths is not None
                and self.region_weight_paths[vol_idx] is not None
                and self.region_weight_paths[vol_idx] != "")

    def _load_region_weight(self, vol_idx: int) -> np.ndarray:
        """Load per-sample region-weight volume (bbox-cropped, +1 shifted)."""
        if self._npz_paths is not None:
            path = self._npz_paths[vol_idx]
            cached = self._rw_cache.get(path)
            if cached is not None:
                return cached
            rw = load_npz_region_weight(path)
            if rw is not None:
                self._rw_cache.put(path, rw)
            return rw
        path   = self.region_weight_paths[vol_idx]
        cached = self._rw_cache.get(path)
        if cached is not None:
            return cached
        rw = load_region_weight_volume(
            path, bbox=self._bbox_for(vol_idx))
        self._rw_cache.put(path, rw)
        return rw

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

        # Per-sample region-weight file takes precedence over the static
        # ``region_weights`` mapping; load once and re-extract per scale.
        rw_vol = (self._load_region_weight(vol_idx)
                  if self._has_region_weight_file(vol_idx) else None)

        # ---- 3D cubic lazy single-max-FOV-cube path -------------------
        # Emit ONE cube of size ``round(extract_size * max_scale)``
        # (edge-padded) at native resolution. The trainer's per-view
        # crop+resize step (R2) reproduces the legacy
        # ``(B, C_res, eD, eH, eW)`` model input — center-cropping to
        # ``round(extract_size * s_k)`` and resizing each view back to
        # ``extract_size`` is voxel-equivalent to the False-path's
        # per-view ``_extract_cubic_patch + resize_3d`` (modulo a
        # single shared interpolation pass instead of K independent
        # ones).
        if self.keep_native_multi_res:
            return self._getitem_native_multi_res_cubic(
                center, img, lbl, rw_vol, eD, eH, eW)

        img_channels, lbl_channels, wmap_channels = [], [], []
        for scale in self.multi_res_scales:
            sD = int(round(eD * scale))
            sH = int(round(eH * scale))
            sW = int(round(eW * scale))

            img_s = _extract_cubic_patch(img, center, (sD, sH, sW))
            lbl_s = _extract_cubic_patch(lbl, center, (sD, sH, sW))
            rw_s = (_extract_cubic_patch(rw_vol, center, (sD, sH, sW))
                    if rw_vol is not None else None)

            img_s = resize_3d(img_s, eD, eH, eW, is_label=False)
            lbl_s = resize_3d(lbl_s, eD, eH, eW, is_label=True)

            img_channels.append(img_s)
            lbl_channels.append(lbl_s)

            # Region-weight precedence: per-sample file > static mapping.
            if rw_s is not None:
                wmap_s = resize_3d(rw_s, eD, eH, eW, is_label=False)
                wmap_channels.append(wmap_s)
            elif self.region_weights:
                wmap_s = compute_region_weight_map(lbl_s, self.label_values, self.region_weights)
                wmap_channels.append(wmap_s[0])  # (D, H, W), squeeze the leading 1

        # P8: label stays int16 (see SegDataset3D.__getitem__ for rationale).
        result = {
            "image": torch.from_numpy(np.stack(img_channels, axis=0).astype(np.float32, copy=False)),
            "label": torch.from_numpy(np.ascontiguousarray(np.stack(lbl_channels, axis=0)))}
        if wmap_channels:
            result["weight_map"] = torch.from_numpy(
                np.stack(wmap_channels, axis=0).astype(np.float32, copy=False))  # (C_res, eD, eH, eW)
        return result

    def _getitem_native_multi_res_cubic(
        self,
        center: Tuple[int, int, int],
        img: np.ndarray,
        lbl: np.ndarray,
        rw_vol: Optional[np.ndarray],
        eD: int,
        eH: int,
        eW: int,
    ) -> Dict[str, torch.Tensor]:
        """Single max-FOV cube emission for the 3D cubic lazy path.

        Output shape: ``(1, eD_max, eH_max, eW_max)`` for image / label /
        (weight_map) with ``(eD_max, eH_max, eW_max) = round(extract_size
        * max_scale)``. The leading "1" stands in for the legacy
        ``C_res`` axis (collapsed: views are reconstructed by the
        trainer instead of stacked here).

        Geometric contract
        ------------------
        ``_safe_center_range`` already accounts for the max-scale cube
        when sampling ``center``, so the largest cube fits in-bounds (or
        the volume is degenerately small along an axis, in which case
        ``_extract_cubic_patch`` edge-pads — same fallback as the
        False-path).

        Equivalence with the False-path
        -------------------------------
        For view k with native size ``S_k = round(extract_size * s_k)``,
        center-cropping ``S_k`` voxels around the cube centre and
        resizing the crop back to ``extract_size`` reproduces the
        False-path's per-view extraction
        ``_extract_cubic_patch(...,(S_k,...)) → resize_3d(..., eD, eH,
        eW)`` voxel-for-voxel — both paths share the same ``center``
        and the same edge-pad rule. The R2 trainer step does this
        crop+resize on the GPU (one ``F.interpolate`` per view).

        Region weights are extracted with the same cubic edge-padded
        path; downstream linear resize preserves continuous gradients.
        """
        eD_max = int(round(eD * self._max_scale))
        eH_max = int(round(eH * self._max_scale))
        eW_max = int(round(eW * self._max_scale))
        size_max = (eD_max, eH_max, eW_max)

        img_s = _extract_cubic_patch(img, center, size_max)
        lbl_s = _extract_cubic_patch(lbl, center, size_max)
        rw_s = (_extract_cubic_patch(rw_vol, center, size_max)
                if rw_vol is not None else None)

        result = {
            # Leading "1" = collapsed C_res. Trainer (R2) splits along
            # all 3 spatial axes and resizes back to ``(B, C_res, eD,
            # eH, eW)``.
            "image": torch.from_numpy(img_s[None].astype(np.float32, copy=False)),
            # P8: int16 label (see SegDataset3D.__getitem__ for rationale).
            "label": torch.from_numpy(np.ascontiguousarray(lbl_s[None])),
        }
        # Region-weight precedence: per-sample file > static mapping.
        if rw_s is not None:
            result["weight_map"] = torch.from_numpy(
                rw_s[None].astype(np.float32, copy=False))
        elif self.region_weights:
            wmap_s = compute_region_weight_map(
                lbl_s, self.label_values, self.region_weights)
            # ``compute_region_weight_map`` returns shape (1, eD_max,
            # eH_max, eW_max).
            result["weight_map"] = torch.from_numpy(
                wmap_s.astype(np.float32, copy=False))
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
        region_weights: Optional[List[float]] = None,
        bbox_paths: Optional[List[str]] = None,
        region_weight_paths: Optional[List[str]] = None,
        npz_paths: Optional[List[str]] = None):
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

        # Optional ROI bbox cropping. When `bbox_paths` is supplied, we
        # precompute one bbox per sample upfront (logging the dataset
        # mean / min / max bbox size) and apply it inside `_load_image`
        # / `_load_label` BEFORE caching so the cached volumes are
        # already cropped — saving both RAM and downstream compute.
        if bbox_paths is not None:
            assert len(bbox_paths) == len(image_paths), (
                f"bbox_paths length {len(bbox_paths)} != image_paths "
                f"length {len(image_paths)}")
        self.bbox_paths = bbox_paths
        self._bboxes: Optional[List[Optional[BBox]]] = (
            precompute_bboxes(bbox_paths) if bbox_paths else None)

        # Optional per-sample region-weight NIfTI (see SegDataset3D for
        # the full contract). File precedence over ``region_weights``.
        if region_weight_paths is not None:
            assert len(region_weight_paths) == len(image_paths), (
                f"region_weight_paths length {len(region_weight_paths)} != "
                f"image_paths length {len(image_paths)}")
        self.region_weight_paths = region_weight_paths
        self._rw_cache = VolumeCache(cache_enabled, cache_max_volumes)

        # ---- NPZ pre-computed package mode (see SegDataset3D) -------
        self._npz_paths: Optional[List[str]] = (
            list(npz_paths) if npz_paths is not None else None)
        if self._npz_paths is not None:
            assert len(self._npz_paths) == len(image_paths), (
                f"npz_paths length {len(self._npz_paths)} != image_paths "
                f"length {len(image_paths)}")
            if self._bboxes is not None:
                logger.warning(
                    "npz mode: ignoring supplied bbox_paths (bbox is "
                    "already pre-applied inside the npz packages).")
                self._bboxes = None
                self.bbox_paths = None
        self._npz_has_rw_cache: Dict[int, bool] = {}

        logger.info(
            "Whole-volume dataset: %d volumes, extract_size=%s, "
            "samples_per_volume=%d%s",
            len(self.image_paths), self.extract_size, self.samples_per_volume,
            " [npz mode]" if self._npz_paths is not None else "")

    def _bbox_for(self, vol_idx: int) -> Optional[BBox]:
        return self._bboxes[vol_idx] if self._bboxes is not None else None

    def _load_image(self, vol_idx: int) -> np.ndarray:
        if self._npz_paths is not None:
            path = self._npz_paths[vol_idx]
            cached = self._img_cache.get(path)
            if cached is not None:
                return cached
            img = load_npz_image(
                path, self.intensity_min, self.intensity_max,
                self.normalize, self.global_mean, self.global_std)
            self._img_cache.put(path, img)
            return img
        path = self.image_paths[vol_idx]
        cached = self._img_cache.get(path)
        if cached is not None:
            return cached
        # Fused decode + bbox crop — see ``SegDataset3D._load_image''
        # for the host-OOM rationale.
        img = load_nifti_cropped(
            path, bbox=self._bbox_for(vol_idx), dtype=np.float32)
        img = preprocess_image(
            img, self.intensity_min, self.intensity_max,
            self.normalize, self.global_mean, self.global_std,
            inplace=True)
        self._img_cache.put(path, img)
        return img

    def _load_label(self, vol_idx: int) -> np.ndarray:
        """Load raw label volume with caching (int16 for RAM savings).

        See ``SegDataset3D._load_label`` for the dtype rationale.
        """
        if self._npz_paths is not None:
            path = self._npz_paths[vol_idx]
            cached = self._lbl_cache.get(path)
            if cached is not None:
                return cached
            lbl = load_npz_label(path)
            self._lbl_cache.put(path, lbl)
            return lbl
        path = self.label_paths[vol_idx]
        cached = self._lbl_cache.get(path)
        if cached is not None:
            return cached
        lbl = load_nifti_cropped(
            path, bbox=self._bbox_for(vol_idx), dtype=np.int16)
        self._lbl_cache.put(path, lbl)
        return lbl

    def _has_region_weight_file(self, vol_idx: int) -> bool:
        if self._npz_paths is not None:
            cached = self._npz_has_rw_cache.get(vol_idx)
            if cached is None:
                cached = npz_has_rw(self._npz_paths[vol_idx])
                self._npz_has_rw_cache[vol_idx] = cached
            return cached
        return (self.region_weight_paths is not None
                and self.region_weight_paths[vol_idx] is not None
                and self.region_weight_paths[vol_idx] != "")

    def _load_region_weight(self, vol_idx: int) -> np.ndarray:
        """Load per-sample region-weight volume (bbox-cropped, +1 shifted)."""
        if self._npz_paths is not None:
            path = self._npz_paths[vol_idx]
            cached = self._rw_cache.get(path)
            if cached is not None:
                return cached
            rw = load_npz_region_weight(path)
            if rw is not None:
                self._rw_cache.put(path, rw)
            return rw
        path   = self.region_weight_paths[vol_idx]
        cached = self._rw_cache.get(path)
        if cached is not None:
            return cached
        rw = load_region_weight_volume(
            path, bbox=self._bbox_for(vol_idx))
        self._rw_cache.put(path, rw)
        return rw

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

        # P8: int16 label emitted as-is. Image forced to float32 for the
        # autocast forward path.
        result = {
            "image": torch.from_numpy(img_r[np.newaxis]).float(),  # (1, eD, eH, eW)
            "label": torch.from_numpy(np.ascontiguousarray(lbl_r[np.newaxis]))}

        # Region-weight precedence: per-sample file > static mapping.
        if self._has_region_weight_file(vol_idx):
            rw_vol = self._load_region_weight(vol_idx)
            wmap = resize_3d(rw_vol, eD, eH, eW, is_label=False)  # (eD, eH, eW)
            result["weight_map"] = torch.from_numpy(
                wmap[np.newaxis]).float()  # (1, eD, eH, eW)
        elif self.region_weights:
            wmap = compute_region_weight_map(
                lbl_r, self.label_values, self.region_weights)
            result["weight_map"] = torch.from_numpy(wmap).float()  # (1, eD, eH, eW)
        return result