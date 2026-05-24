"""Configuration system using dataclasses + YAML.

All tunable parameters are centralized here. The YAML config file maps
directly to nested dataclasses for type safety and IDE autocompletion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    """Data paths and preprocessing settings."""

    image_dir: str = ""
    label_dir: str = ""
    # Suffix fields below all accept either a single string (legacy
    # contract) or a YAML list of candidate suffixes. With a list,
    # pairing under :mod:`segtask_v1.data.loader` strips the suffix to
    # compute a *base* name and tries each candidate in order, taking
    # the first existing file under the corresponding directory. This
    # lets images / labels / bboxes / region-weights live side-by-side
    # with mixed naming conventions (e.g. ``case01.nii.gz`` paired with
    # ``case01-seg.nii.gz`` or ``case01_pred.nii.gz``). Order in the
    # list expresses precedence (first match wins per sample).
    image_suffix: Union[str, List[str]] = ".nii.gz"
    label_suffix: Union[str, List[str]] = ".nii.gz"

    # Optional ROI bounding-box directory. When non-empty, each sample is
    # expected to have a matching NIfTI mask under ``bbox_dir`` (filename
    # base-matched against the image, then any suffix in ``bbox_suffix``
    # tried in order). At dataset init
    # time we load every bbox mask once, compute its axis-aligned
    # bounding box of nonzero voxels, log the mean (D, H, W) bbox size
    # across the dataset, and cache the per-sample bbox tuple. Each time
    # an image / label volume is loaded it is cropped to that bbox before
    # any downstream preprocessing or patch extraction. This both shrinks
    # the working volume for large CT scans where only a sub-region is
    # of interest, and keeps the rest of the pipeline (patching, multi-
    # resolution, augmentation, prediction) untouched. Empty string =
    # disabled (legacy full-volume behaviour).
    bbox_dir: str = ""
    bbox_suffix: Union[str, List[str]] = ".nii.gz"

    # Optional per-sample region-weight NIfTI directory. When non-empty,
    # each sample must have a matching NIfTI file under ``region_weight_dir``
    # (filename match against the image, suffix ``region_weight_suffix``).
    # The file is expected to be a hand-annotated continuous weight volume
    # with background = 0 and non-background voxels set to the desired
    # weight value; the dataset adds +1 on load so background → 1 and the
    # annotated regions become (w + 1). Precedence:
    #   per-sample file (if dir set)  >  ``loss.region_weights``  >  disabled.
    # When ``region_weight_dir`` is set, every sample must have a matching
    # file (FileNotFoundError otherwise) — mirrors the ``bbox_dir`` strong-
    # match contract to avoid silent per-sample weight regressions.
    region_weight_dir: str = ""
    region_weight_suffix: Union[str, List[str]] = ".nii.gz"

    # ---- Pre-computed npz pipeline ---------------------------------
    # Optional pre-computed npz package directory (output of
    # ``segtask_v1.data.make_data``). When non-empty, the trainer's
    # data loader reads bbox-cropped image / label / region_weight
    # plus pre-computed foreground indices straight from
    # ``<npz_dir>/<pid>.npz`` and IGNORES ``image_dir`` /
    # ``label_dir`` / ``bbox_dir`` / ``region_weight_dir`` at runtime.
    # The npz files store:
    #   image      int16  (D', H', W')  raw HU, bbox-cropped
    #   label      int16  (D', H', W')  raw labels, bbox-cropped
    #   rw         float32 (D', H', W')  +1 shifted, optional
    #   fg_slices  int32  (M,)          per-z fg index in cropped frame
    #   fg_coords  int32  (N, 3)        sub-sampled (seed=42, N<=50000)
    #   meta       object dict          provenance
    # Benefits: (1) eliminates the SimpleITK gzip-decompress peak that
    # OOMs ``num_workers >= 4`` with concurrent image+label+rw reads;
    # (2) lets DataLoader workers ``mmap`` arrays out of the OS page
    # cache (shared across workers); (3) skips the dataset's startup
    # ``precompute_bboxes`` + ``_build_index`` scans (fg indices are
    # baked in). Strong contract: every sample expected from
    # discovery (or every pid in ``<npz_dir>/_manifest.json``) must
    # have a matching npz file. Empty string = legacy NIfTI pipeline.
    #
    # Companion field ``npz_suffix`` lets users co-locate npz files
    # with non-standard extensions (e.g. when sharded across folders);
    # default ``.npz`` matches make_data's output verbatim.
    npz_dir: str = ""
    npz_suffix: str = ".npz"

    # When True (default) and ``npz_dir`` points at a missing or
    # empty directory at trainer startup, ``build_dataloaders``
    # calls ``segtask_v1.data.make_data.prepare_dataset`` inline
    # before constructing the loaders — turning the npz pipeline
    # into a one-shot "just set npz_dir and run train" UX. Set to
    # False to require an explicit ``python -m segtask_v1.data.
    # make_data ...`` invocation (recommended for cluster runs
    # where the build step should be a separate scheduled job).
    # This switch only fires when the directory is empty; a
    # partially-populated directory (e.g. resume after a crashed
    # build) is treated as authoritative — re-run make_data
    # manually with ``--overwrite`` or wipe the directory to force
    # a rebuild.
    npz_auto_build: bool = True

    # 样本排除清单（文本文件路径，每行一个 pid / stem）。pid 定义为
    # `image_path.name`（不含后缀，即 `<name>{image_suffix}` 去掉后缀的部分）。
    # 命中的 pid 会在 `discover_samples` 之后立刻从 image/label/bbox 配对中剔除。
    # 典型用途：跳过 SimpleITK 无法读取的非正交 direction cosines 文件
    # （参见 `tools/scan_bad_nifti.py` 扫描脚本）。留空 = 不过滤。
    exclude_list: str = ""

    # Label mapping: integer label values in the mask (0=background).
    # e.g. [0, 1, 2] for 3-class. Empty = auto-detect from data.
    label_values: List[int] = field(default_factory=list)
    num_classes: int = 0  # auto-set from label_values

    # 3D patch size: [D, H, W] — model input resolution
    patch_size: List[int] = field(default_factory=lambda: [64, 128, 128])

    # Patch extraction mode:
    #   "z_axis" — slide along z-axis, extract D slices, resize H,W to target.
    #              Supports `multi_res_scales` (z-axis-only scaling).
    #   "cubic"  — sample center (x,y,z), extract full 3D cube of patch_size.
    #              Supports `multi_res_scales` (all 3 axes scale).
    #   "whole"  — resize the ENTIRE volume to `patch_size` (no sliding
    #              window, no sub-cropping). Simplest mode; useful when the
    #              object of interest spans most of every volume and memory
    #              / compute budget allows feeding the full downsampled
    #              volume each step. `multi_res_scales` must be [1.0] here
    #              (scaling has no physical meaning beyond the volume).
    #   "2_5d"   — 2.5D mode: reuses the z_axis dataset path with
    #              `multi_res_scales=[1.0]` (forced). The trainer squeezes
    #              the C_res=1 axis after augmentation, treating the D
    #              slices as input channels for a planar 2D UNet. Model
    #              ``spatial_dims`` is auto-set to 2 and ``in_channels``
    #              auto-set to ``patch_size[0] = D``. The model output
    #              ``(B, num_fg*D, H, W)`` is split per fg class into
    #              D-channel binary maps by ``SliceChannelLoss``.
    patch_mode: str = "z_axis"

    # Augmentation oversample ratio (applies to BOTH z_axis and cubic modes).
    # Dataset extracts a patch of size `round(patch_size * ratio)` on every
    # axis, the augmentor applies spatial transforms (rotate/elastic with
    # `zeros` padding), and the trainer center-crops back to patch_size.
    # This removes the black-corner artefacts that grid_sample introduces
    # at rotated edges. 1.0 = disabled (legacy behaviour), 1.4~1.5 recommended
    # whenever `random_affine_prob` or `elastic_deform_prob` > 0.
    aug_oversample_ratio: float = 1.0

    # Multi-resolution input — supported in BOTH z_axis and cubic modes,
    # with axis semantics matching each mode:
    #   cubic  — scale applies on ALL three axes (D, H, W). Each scale
    #            extracts a physically larger cube around the same center
    #            and resizes back to extract_size.
    #   z_axis — scale applies ON Z ONLY. Each scale extracts a wider z
    #            range (round(eD * scale) slices) around the same z center
    #            and resizes back to extract_size. H, W are always full
    #            volume resolution in z_axis mode — no in-plane scaling
    #            makes sense there.
    # Each scale's output is stacked as an input channel: [1.0] = 1-channel
    # (legacy), [1.0, 1.5, 2.0] = 3-channel input.
    multi_res_scales: List[float] = field(default_factory=lambda: [1.0])

    # Intensity windowing (HU for CT)
    intensity_min: float = -1024.0
    intensity_max: float = 3071.0
    # Normalization: "minmax" -> [0,1], "zscore" -> zero-mean unit-var
    normalize: str = "minmax"
    global_mean: float = 0.0
    global_std: float = 1.0

    # Train/val split
    val_ratio: float = 0.2
    split_seed: int = 42
    # Stratified split by each volume's primary foreground class.
    # Strongly recommended when class distribution is imbalanced (typical
    # medical imaging case). Falls back to random split if the dataset is
    # too small to stratify cleanly.
    stratified_split: bool = True

    # DataLoader
    batch_size: int = 2
    num_workers: int = 4
    pin_memory: bool = True
    # Keep worker processes alive across epochs. On Windows (spawn start
    # method) and macOS this avoids re-pickling the whole Dataset and
    # re-warming every per-worker volume cache at the start of every
    # epoch — usually the single biggest source of "epoch start stall"
    # on non-Linux hosts. Only consulted when ``num_workers > 0``.
    persistent_workers: bool = True
    # How many batches each worker prefetches ahead of consumption. The
    # PyTorch default is 2; raising to 4 hides sitk decode + preprocessing
    # latency behind GPU compute more effectively. Only consulted when
    # ``num_workers > 0``.
    prefetch_factor: int = 4

    # Foreground oversampling: probability of centering patch on foreground
    foreground_oversample_ratio: float = 0.5

    # Samples per volume per epoch (controls epoch length)
    samples_per_volume: int = 8

    # Caching: "none" or "memory".
    # `memory` keeps decoded volumes (image+label) in an LRU-bounded in-RAM
    # cache. `cache_max_volumes` caps the number of cached volumes per
    # worker — set to 0 for unbounded (matches the legacy behaviour, but
    # risks OOM on large datasets). The recommended setting is a few times
    # the effective prefetch horizon (= num_workers * samples_per_volume).
    cache_mode: str = "memory"
    cache_max_volumes: int = 0  # 0 = unbounded

    # ---- Z-axis boundary-window handling (z_axis / 2.5D modes) ----
    # Controls how the ``scale=1.0`` channel of a z-window is built when
    # the candidate window has fewer than ``extract_size[0]`` real slices
    # — i.e. (a) the volume is shorter than the patch in z, or (b) the
    # sampled center sits close enough to a volume boundary that
    # ``[z_center - eD/2, z_center + eD/2)`` partially falls outside the
    # volume.
    #
    # "stretch"  (default, backward compatible): take the in-bounds
    #     slices verbatim and let ``resize_3d`` / ``F.interpolate``
    #     stretch them to ``eD`` slices along z. Pitfall: in 2.5D mode
    #     the model's input-channel index implicitly encodes a
    #     "channel k = z_center + (k - eD/2)" physical mapping; a
    #     stretched boundary window remaps that mapping non-linearly,
    #     producing a train-test slice-spacing mismatch and
    #     systematically lower quality near volume edges.
    #
    # "edge_pad" (recommended for 2.5D): edge-replicate-pad the
    #     window symmetrically along z to exactly ``eD`` slices BEFORE
    #     any resize, so every channel keeps its physical 1-slice
    #     spacing regardless of where the window sits relative to the
    #     volume boundary. ``scale > 1.0`` channels already use this
    #     contract (via ``extract_z_patch_padded``) — turning the
    #     toggle on simply makes the ``scale=1.0`` channel match.
    #
    # The toggle covers BOTH the training dataset (``SegDataset3D``)
    # and the inference predictor (``Predictor._build_z_window_input``
    # CPU/GPU paths + ``_sliding_window_z`` reverse-resize) so that
    # train and inference geometries stay strictly consistent. Modes
    # other than z_axis / 2.5D are unaffected.
    z_boundary_mode: str = "stretch"

    # ---- 2.5D multi-FOV: keep auxiliary views at NATIVE depth ----
    # When False (default, fully backward compatible):
    #   Each scale s_k extracts ``round(eD * s_k)`` slices around the
    #   sampled z-center and is **z-resampled back to eD slices** so that
    #   all views share the same D channel count, producing a stacked
    #   input ``(B, n_views * D, H, W)``. Auxiliary FOVs therefore lose
    #   information along z (compression).
    #
    # When True (only valid for ``patch_mode == "2_5d"`` with
    # ``len(multi_res_scales) > 1`` and ``model.aux_seg_supervision = True``):
    #   The dataset extracts a SINGLE max-FOV cube of depth
    #   ``round(eD * max_scale)`` (edge-padded at volume boundaries),
    #   runs all 3D augmentations on that single cube — **once** — and the
    #   trainer center-crops per view at native depth ``D_k = round(eD *
    #   s_k)`` immediately before the model forward. Each aux head therefore
    #   predicts ``(B, num_fg * D_k, H, W)`` against view k's native-depth
    #   label, with no z-axis information loss for wider FOVs.
    #
    # Geometric equivalence
    # ---------------------
    # All views share the same z-center (``_sample_z`` is computed once).
    # Center-cropping ``D_k`` slices from the max-FOV cube yields exactly
    # the same physical slice set as the per-view independent extraction
    # used by the False-path (slice spacing == 1 along z) — but with a
    # SINGLE shared augmentation field, eliminating the cross-view
    # geometric drift that ``False`` introduces by running grid_sample
    # independently per view.
    #
    # Side constraints (enforced in ``validate()``)
    # ---------------------------------------------
    # * ``z_boundary_mode`` is forced to ``"edge_pad"`` (the max-scale path
    #   inherently uses ``extract_z_patch_padded``; ``stretch`` would have
    #   no consumer and would silently mislead).
    # * Inactive in 3D modes (multi_res_scales is a channel-stack there,
    #   D-resampling is not part of that semantics).
    aux_keep_native_d: bool = False

    # ---- 3D multi-FOV: lazy single-cube extraction (z_axis / cubic) ----
    # When False (default, fully backward compatible):
    #   For each scale s_k in ``multi_res_scales`` the dataset extracts a
    #   physical cube of size ``round(extract_size * s_k)`` around the
    #   sampled centre (z-only for ``z_axis``; all three axes for
    #   ``cubic``), resizes it to ``extract_size`` (one ``scipy.ndimage.
    #   zoom`` call per view), and stacks all views as the leading
    #   ``C_res`` axis → ``(C_res, eD, eH, eW)``. Augmentation then runs
    #   ONCE on the canonical-resolution stack with a SHARED grid_sample,
    #   which means: (a) each view's high-frequency content is already
    #   attenuated by the per-view zoom BEFORE augment, and (b) the K
    #   per-view zooms run on every CPU worker.
    #
    # When True (only valid for ``patch_mode in {"z_axis", "cubic"}``,
    # ``len(multi_res_scales) > 1`` and ``multi_res_scales[0] == 1.0``):
    #   The dataset extracts a SINGLE max-FOV cube at the largest
    #   physical resolution and emits it as ``(1, eD_max, eH_max, eW_max)``
    #   (raw integer labels, continuous weights). All 3D augmentations
    #   then run on this single cube with one shared grid_sample call.
    #   The trainer (R2 — see ``_split_views_native_3d``) center-crops
    #   per view at native physical size ``round(extract_size * s_k)`` and
    #   resizes each view back to ``extract_size`` immediately before the
    #   3D forward, finally producing the standard ``(B, C_res, eD, eH,
    #   eW)`` model input.
    #
    # Geometric equivalence
    # ---------------------
    # All views share the same centre by construction (centre sampling
    # runs once on the max-FOV cube). Center-cropping the per-view sub-
    # cube and resizing to canonical size produces the SAME physical
    # voxel set as the per-view independent extraction in the False-path
    # (modulo a single linear/nearest interpolation pass instead of two:
    # one in dataset + one inside grid_sample). The benefits over False
    # are: (a) one shared aug field (cross-view warp consistency by
    # construction); (b) aux views are not pre-downsampled before aug,
    # preserving high-frequency detail; (c) no per-view scipy zoom in
    # the CPU dataset workers.
    #
    # Side constraints (enforced in ``validate()``)
    # ---------------------------------------------
    # * ``z_boundary_mode`` is forced to ``"edge_pad"`` in z_axis mode
    #   (the max-scale path always uses ``extract_z_patch_padded``;
    #   ``stretch`` would have no consumer and silently mislead).
    # * Mutually exclusive with ``aux_keep_native_d`` (which is the
    #   2.5D analogue; that flag has fundamentally different semantics
    #   — no per-view resize at all, since the 2D model consumes the
    #   per-view depth as input channels directly).
    # * Inactive in ``2_5d`` (use ``aux_keep_native_d``) and ``whole``
    #   (multi-res has no physical meaning there).
    # * ``multi_res_scales[0]`` must be ``1.0`` (view 0 = canonical
    #   geometry; same invariant as the False-path and as 2.5D).
    #
    # Predictor / inference
    # ---------------------
    # Inference path (predict.py / Predictor) is NOT wired in this
    # release — train-only switch. Enable only for training experiments
    # until the inference codepath ships. Setting True together with
    # ``train.resume`` of a False-path checkpoint is fine — only the
    # data emission contract changes; model weights / shapes are
    # identical.
    keep_native_multi_res: bool = False


# ---------------------------------------------------------------------------
# Augmentation configuration
# ---------------------------------------------------------------------------
@dataclass
class AugConfig:
    """GPU data augmentation settings.

    All spatial transforms are per-sample independent (not batch-level).
    """

    enabled: bool = True

    # --- Spatial (applied to image + label jointly) ---
    random_flip_prob: float = 0.5
    random_flip_axes: List[int] = field(default_factory=lambda: [2, 3, 4])

    # Affine: rotation (small angles, degrees) + scale, composed into one grid_sample
    random_affine_prob: float = 0.3
    random_rotate_range: List[float] = field(default_factory=lambda: [-15.0, 15.0])
    random_scale_range: List[float] = field(default_factory=lambda: [0.85, 1.15])

    # Elastic deformation (B-spline random displacement field)
    elastic_deform_prob: float = 0.2
    elastic_deform_sigma: float = 5.0   # Smoothness of displacement (coarse grid spacing)
    elastic_deform_alpha: float = 7.0   # Displacement magnitude in voxels (std)

    # Grid dropout (mask out rectangular sub-regions)
    grid_dropout_prob: float = 0.0
    grid_dropout_ratio: float = 0.3  # fraction of spatial area to drop
    grid_dropout_holes: int = 4      # number of rectangular holes

    # --- Intensity (image only) ---
    random_brightness_prob: float = 0.3
    random_brightness_range: List[float] = field(default_factory=lambda: [-0.1, 0.1])

    random_contrast_prob: float = 0.3
    random_contrast_range: List[float] = field(default_factory=lambda: [0.8, 1.2])

    random_gamma_prob: float = 0.2
    random_gamma_range: List[float] = field(default_factory=lambda: [0.8, 1.2])

    gaussian_noise_prob: float = 0.15
    gaussian_noise_std: float = 0.05

    gaussian_blur_prob: float = 0.1
    gaussian_blur_sigma: List[float] = field(default_factory=lambda: [0.5, 1.5])

    # Simulate low resolution (downsample then upsample)
    simulate_lowres_prob: float = 0.1
    simulate_lowres_zoom: List[float] = field(default_factory=lambda: [0.5, 1.0])

    # ---- Weight-map spatial-resampling interpolation ------------------
    # Controls how ``weight_map`` is resampled by ``_random_affine`` and
    # ``_elastic_deform`` (the only two transforms that touch wmap with
    # an interpolation kernel — flip uses ``torch.flip``, dropout/intensity
    # transforms never touch wmap).
    #
    #   "nearest"  — preserves the EXACT discrete weight values produced
    #                by ``compute_region_weight_map`` (e.g. bg=1, fg=4).
    #                Required for the common case where region weights are
    #                derived from a label map. SAFE DEFAULT.
    #   "bilinear" — keeps continuous gradients intact at the cost of
    #                quantising fg/bg integer weights into a noisy mixture
    #                (the symptom: bg voxels become "0.x ~ 1.x", fg voxels
    #                become "2.x ~ 4" near boundaries). Use this ONLY when
    #                the per-sample wmap NIfTI is intentionally continuous
    #                (e.g. distance-to-boundary maps in
    #                ``region_weight_dir``).
    #
    # If you provide BOTH discrete fg/bg weights via ``compute_region_weight_map``
    # AND you don't have a per-sample continuous rw NIfTI, leave this at
    # "nearest". If your ``region_weight_dir`` ships continuous weights,
    # set this to "bilinear" in your YAML.
    wmap_interp_mode: str = "nearest"


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """UNet model architecture settings."""

    # Backbone: "resnet" or "convnext"
    backbone: str = "resnet"

    # Spatial dimensionality of the network. 3 = volumetric 3D UNet
    # (default, used by z_axis / cubic / whole patch modes). 2 = planar
    # 2D UNet (used by the 2.5D patch mode where D slices are stacked
    # as input channels). All blocks/stages/decoders honour this value.
    spatial_dims: int = 3

    # Input channels (always 1 for single-modality 3D)
    in_channels: int = 1

    # Channel progression per encoder level (determines network depth)
    # e.g. [32, 64, 128, 256, 512] = 5 levels
    encoder_channels: List[int] = field(
        default_factory=lambda: [32, 64, 128, 256, 512]
    )

    # Blocks per encoder/decoder level (used when encoder_blocks_per_stage
    # and decoder_blocks_per_stage are both empty — kept for back-compat).
    blocks_per_level: int = 2

    # Residual block variant (see models.resnet):
    #   "basic"      — classic post-act ResNet (default).
    #   "preact"     — pre-activation ResNet (deep encoders).
    #   "bottleneck" — 1×1×1/3×3×3/1×1×1 expansion (nnU-Net ResEnc-XL).
    #   "r2plus1d"   — factorised (2+1)D residual block (Plan A: inject
    #                  z-axis context via a (1,3,3) spatial conv + (3,1,1)
    #                  temporal conv with mid non-linearity). REQUIRES
    #                  ``spatial_dims=3`` (i.e. patch_mode in z_axis /
    #                  cubic / whole). Rejected at validate() time when
    #                  used with 2.5D mode — see validate() for details.
    # ConvNeXt backbone ignores this field.
    block_type: str = "basic"

    # Asymmetric per-stage block counts (nnU-Net ResEncUNet style).
    # Length must equal len(encoder_channels) when non-empty. Decoder length
    # must equal len(encoder_channels) - 1 when non-empty.
    encoder_blocks_per_stage: List[int] = field(default_factory=list)
    decoder_blocks_per_stage: List[int] = field(default_factory=list)

    # nnU-Net ResEnc preset (Isensee et al., MICCAI 2024).
    # One of: "none" | "S" | "M" | "L" | "XL". When != "none" AND the user
    # has not supplied explicit per-stage counts, ``sync()`` auto-populates
    # encoder_blocks_per_stage (trimmed/extended to len(encoder_channels))
    # and sets decoder_blocks_per_stage = [1, 1, ...].
    resenc_preset: str = "none"

    # Normalization: "batch", "instance", "group"
    norm_type: str = "instance"
    norm_groups: int = 8

    # Activation: "relu", "leakyrelu", "gelu", "swish"
    activation: str = "leakyrelu"

    # Dropout in blocks
    dropout: float = 0.0

    # Squeeze-and-Excitation attention (legacy flag; prefer attention_type).
    # When attention_type == "none" and use_se == True, SE is enabled.
    use_se: bool = False
    se_reduction: int = 16

    # In-block channel/spatial attention applied inside each ResNet/ConvNeXt
    # block. One of: "none" | "se" | "eca" | "cbam" | "coord".
    attention_type: str = "none"

    # AttentionGate3D on skip connections (Oktay et al., MIDL 2018).
    skip_attention: bool = False

    # Deep supervision: output predictions at multiple decoder levels
    deep_supervision: bool = False

    # ---- Multi-FOV auxiliary segmentation supervision (2.5D mode only) ----
    # When True AND ``data.patch_mode == "2_5d"`` AND
    # ``len(data.multi_res_scales) > 1``, the seg head is mirrored across
    # views: in addition to the main view-0 prediction, the model emits
    # one auxiliary prediction per aux view k=1..n_views-1 (each shaped
    # ``(B, num_fg * D, H, W)`` — same contract as the main head).
    #
    # Geometric symmetry with the stem fusion choice (auto-detected from
    # ``context_fusion``):
    #   - "shared_stem"  / "multi_stem_proj" (Plan A): aux heads are
    #       mounted in PARALLEL on the highest-resolution decoder feature
    #       (``dec_features[-1]``), each with its own 1×1 conv. Mirrors the
    #       early-fusion stem layout — every view shares the full encoder/
    #       decoder pyramid and only differs in its final classifier.
    #   - "hierarchical" (Plan C): aux head k reads
    #       ``dec_features[-1-k]`` (the decoder feature at the same
    #       semantic depth as the encoder stage where view k was injected)
    #       then 1×1-conv + interpolate back to (H, W). Mirrors the
    #       hierarchical injection point — coarse-FOV supervision lands
    #       at the matching low-resolution decoder feature.
    #
    # Loss-side wiring lives in ``loss.aux_supervision_weights``: per-aux-
    # view scalar weights (length ``n_views-1``). Empty → defaults to
    # geometric decay ``0.5^k``. Set ``aux_seg_supervision=False`` to
    # disable entirely (bit-identical to legacy path; no extra heads built).
    #
    # Always inactive when ``len(multi_res_scales) == 1`` (no aux views to
    # supervise) or in 3D modes (multi-FOV is fed as scale channels there,
    # not as views). The forward call keeps emitting a single tensor at
    # eval time — predictor.py is unchanged.
    aux_seg_supervision: bool = False

    # ---- Aux seg head topology (only when aux_seg_supervision==True) ----
    # Controls the per-aux-view classifier shape:
    #   "linear" — single ``Conv1×1(out_ch=num_fg*D)`` (default; minimal
    #              cost, equal capacity to the main head). Recommended for
    #              Plan A (multi_stem_proj / shared_stem) where every aux
    #              head shares the highest-resolution decoder feature with
    #              the main head — extra capacity is unlikely to help.
    #   "conv"   — ``ConvNormAct(3×3) → Conv1×1`` (≈2 layers). Recommended
    #              for Plan C (hierarchical) because aux head ``k`` reads
    #              the LOW-RESOLUTION decoder feature at level ``k`` (i.e.
    #              ``input/(stem_stride * 2^k)``), where the spatial
    #              context aggregation per output cell is closer to the
    #              decoder's stage block than to the main head's full-res
    #              feature; a 3×3 conv lets the head re-aggregate before
    #              the linear classifier (closer to "main head + a stage
    #              block" capacity). Adds <1% params at typical sizes.
    # The mode is applied uniformly to all aux heads of a build; no per-k
    # override is exposed (we observed no benefit in preliminary checks
    # and the extra config surface would obscure intent).
    aux_head_mode: str = "linear"

    # ---- Plan A 2.5D-to-3D lift (used together with block_type="r2plus1d") ----
    # When True AND ``data.patch_mode == "2_5d"`` the trainer SKIPS the
    # ``(B, C_res, D, H, W) → (B, C_res*D, H, W)`` squeeze that folds D
    # into the channel axis. Instead D is preserved as a real spatial axis
    # and the model is rebuilt as a true 3D UNet:
    #
    #   * ``spatial_dims`` is auto-set to 3 (overrides the 2.5D-mode default).
    #   * ``in_channels`` is auto-set to ``len(data.multi_res_scales)``
    #     (one channel per FOV view) instead of ``D * n_views``.
    #   * Model output is ``(B, num_fg, D, H, W)`` (single-resolution true
    #     3D segmentation), not the folded ``(B, num_fg * D, H, W)``.
    #   * The trainer routes the loss through ``MultiResolutionLoss``
    #     ``(num_res=1)`` (using only view 0 = 1× FOV as supervision target),
    #     bypassing ``SliceChannelLoss`` entirely. ``loss.slice_loss_reduction``
    #     is therefore ignored in lift mode.
    #
    # Why this exists: Plan A's R(2+1)D block (``block_type="r2plus1d"``)
    # decomposes a 3D conv into a (1,3,3) spatial conv + a (3,1,1) temporal
    # conv. The temporal sub-conv is meaningful ONLY when D is a real
    # spatial axis. Lifting the 2.5D pipeline to 3D gives R(2+1)D direct
    # access to inter-slice context while keeping the 2.5D dataset /
    # augmentation / oversampling defaults — switching this flag is a
    # single-line A/B test against the folded baseline.
    #
    # Restrictions (enforced in validate()):
    #   * Only valid when ``data.patch_mode == "2_5d"``.
    #   * Mutually exclusive with ``data.aux_keep_native_d`` (which packs
    #     view-specific ``D_k`` slabs into the channel axis — fundamentally
    #     a folded-D layout).
    #
    # Composes with (orthogonal):
    #   * ``aux_seg_supervision`` — each aux head emits
    #     ``(B, num_fg, D, H, W)`` (3D 1×1×1 conv, since spatial_dims=3)
    #     and the per-view aux loss runs through MultiResolutionLoss
    #     (num_res=1) on view k's z-resampled D-deep label. View 0 (1×
    #     FOV) drives the main head; view k (k=1..n_views-1) drives
    #     aux head k.
    #   * ``deep_supervision`` — main path produces a list of decoder-
    #     resolution outputs as before; aux path stays single-resolution
    #     per view (DS structure is reserved for the main path).
    lift_2_5d_to_3d: bool = False

    # Stem / patch-embed (see models.stem.build_stem):
    # "conv3" | "conv7" | "dual" | "patch2" | "patch4".
    # patchN stems reduce input resolution by N; UNet3D adds a matching
    # trilinear upsample on the main output to restore original resolution.
    stem_mode: str = "conv3"

    # ---- Multi-FOV context fusion (2.5D mode only) ----
    # When ``data.patch_mode == "2_5d"`` AND ``len(data.multi_res_scales) > 1``,
    # the model input is laid out as ``(B, n_views * D, H, W)`` with view 0 =
    # the 1× FOV (real D slices) and views 1..K = wider z-FOVs each resampled
    # back to D channels (see SegDataset3D z-axis multi-res semantics).
    #
    # "shared_stem"     — feed all ``n_views * D`` channels through ONE stem.
    #                     Cheapest, but mixes physically heterogeneous
    #                     channels (raw vs. resampled "virtual" slices)
    #                     through a single filter bank.
    # "multi_stem_proj" — Plan A. ``n_views`` independent stems
    #                     (each on D channels) → cat → 1×1 ConvNormAct
    #                     fusion back to ``encoder_channels[0]``. Strictly
    #                     more expressive at negligible param cost; encoder
    #                     downstream is contract-identical (early fusion
    #                     at full resolution).
    # "hierarchical"    — Plan C. View 0 drives the main stem; aux view
    #                     ``k`` (k=1..n_views-1) goes through a stride-
    #                     ``main_stem_stride * 2^k`` patchify stem and is
    #                     cat-fused into the main path at the entrance of
    #                     encoder stage ``k`` (post-Downsample-k). A 1×1
    #                     ConvNormAct compresses back to the stage's
    #                     expected channel count, so decoder/skip
    #                     contracts are bit-identical. Coarse-FOV context
    #                     thus enters at semantically-matched depth
    #                     instead of being squashed at the input layer.
    #                     Requires ``len(multi_res_scales) <= len(encoder_channels)``.
    # When ``len(multi_res_scales) == 1`` this field is a no-op (the path
    # collapses to the single-stem legacy behaviour and is bit-identical
    # to pre-multi-FOV training). Ignored entirely in 3D modes.
    context_fusion: str = "multi_stem_proj"

    # Decoder topology:
    #   "unet"   — classical symmetric UNet decoder (default).
    #   "unetpp" — UNet++ nested dense decoder (Zhou et al., DLMIA 2018).
    #   "unet3p" — Full-scale skip decoder (Huang et al., ICASSP 2020).
    decoder_type: str = "unet"

    # UNet3+ per-branch channel count (only used when decoder_type=="unet3p").
    unet3p_cat_channels: int = 64

    # Downsampling mode (see models.blocks.Downsample):
    # "conv" | "maxpool" | "avgpool" | "blurpool" | "pixelunshuffle"
    downsample_mode: str = "conv"

    # Upsampling mode (see models.blocks.Upsample):
    # "transpose" | "trilinear" | "nearest" | "pixelshuffle"
    #   | "carafe" | "dysample"
    upsample_mode: str = "transpose"

    # Skip connection mode: "cat" (concatenate) or "add"
    skip_mode: str = "cat"

    # Stochastic depth (drop path) rate — ConvNext only
    drop_path_rate: float = 0.0

    # ConvNeXt LayerScale (Touvron et al.) initial value — only used when
    # ``backbone == "convnext"``. Initialises a learnable per-channel scale
    # ``gamma = layer_scale_init * ones(C)`` applied to the block branch
    # before residual addition, making each block start near-identity. This
    # matches official ConvNeXt and is essential for stable training of
    # deep networks combined with stochastic depth. Set to ``0.0`` (or
    # negative) to DISABLE LayerScale and recover the legacy behaviour.
    convnext_layer_scale_init: float = 1e-6

    # ConvNeXt paper-faithful downsample topology. When True (default) and
    # ``backbone == "convnext"``, inter-stage downsamples use
    # ``LayerNorm → Conv(k=2, s=2)`` (norm-first, LN specifically) instead
    # of the generic ``Downsample`` (which would otherwise pick up
    # ``downsample_mode`` + ``norm_type``). Set to False to fall back to the
    # generic Downsample path (legacy behaviour) for ablation purposes.
    convnext_downsample_lnfirst: bool = True


# ---------------------------------------------------------------------------
# Loss configuration
# ---------------------------------------------------------------------------
@dataclass
class LossConfig:
    """Loss function settings.

    Output is always per-class independent sigmoid:
    each foreground class gets its own binary output (B, 1, D, H, W).
    """

    # Loss: "dice", "bce", "dice_bce", "focal", "dice_focal", "tversky"
    name: str = "dice_bce"

    # Weights for compound losses [loss1_w, loss2_w]
    compound_weights: List[float] = field(default_factory=lambda: [1.0, 1.0])

    # Per-class loss weights (empty = uniform). Length = num_fg_classes.
    class_weights: List[float] = field(default_factory=list)

    # Per-region spatial weights: one weight per label value (including bg).
    # e.g. label_values=[0,1,2,3,4], region_weights=[1.0, 2.0, 2.0, 1.0, 1.0]
    # means voxels with label 1 or 2 get 2x loss weight at that spatial position.
    # Empty = disabled (uniform spatial weight).
    region_weights: List[float] = field(default_factory=list)

    # Dice settings
    dice_smooth: float = 1e-5
    dice_squared: bool = False

    # Focal loss settings
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # Tversky loss settings
    tversky_alpha: float = 0.3  # FP weight
    tversky_beta: float = 0.7   # FN weight

    # Dice / Tversky aggregation mode: batch_dice sums TP / denom across
    # the whole batch+spatial before dividing (nnU-Net default for Dice).
    # Affects BinaryDiceLoss, BinaryTverskyLoss, BinaryFocalTverskyLoss,
    # GeneralizedDiceLoss (for GDL the default here is overridden to True
    # in _build_gdl — see paper).
    batch_dice: bool = False
    # Per-sample mode only: exclude classes with no GT voxels in the
    # current sample from the dice mean (prevents empty-class Dice≈1 from
    # masking errors on other classes).
    ignore_empty: bool = False

    # ---- Generalized Dice Loss (Sudre et al., DLMIA 2017) ----
    # Volume-based class re-weighting scheme.
    # "square" (paper) | "simple" (w=1/Σt) | "uniform" (disabled).
    gdl_weight_type: str = "square"
    gdl_w_max: float = 1.0e5    # clamp 1/volume to avoid explosion on empty classes

    # ---- Focal Tversky Loss (Abraham & Khan, ISBI 2019) ----
    # Our convention: (1 - TI)^gamma with gamma ≥ 1 → focus on hard classes.
    # Default 4/3 matches the authors' γ_paper = 0.75 recommendation.
    focal_tversky_gamma: float = 4.0 / 3.0

    # ---- Lovász-Hinge (Berman et al., CVPR 2018) ----
    # per_sample=True → average loss over (B, C) independent sorts (default);
    # per_sample=False → batch-level Lovász (one sort over all B samples per
    #                    channel), smoother on tiny patches.
    lovasz_per_sample: bool = True

    # ---- Soft clDice (Shit et al., CVPR 2021) ----
    # Skeletonisation iterations. Paper: 3 for 2D, 3–10 for 3D depending on
    # structure thickness.
    cldice_iter: int = 3
    cldice_smooth: float = 1.0

    # Deep supervision weight decay
    deep_supervision_weights: List[float] = field(
        default_factory=lambda: [1.0, 0.5, 0.25, 0.125]
    )

    # ---- 2.5D loss reduction (only used when data.patch_mode == "2_5d") ----
    # Controls how ``SliceChannelLoss`` aggregates the per-class binary loss
    # across the D slice axis (which the 2.5D model exposes as input
    # channels and a (num_fg * D)-channel output).
    #
    # "per_slice"  (default, backward compatible): the loss is computed
    #     INDEPENDENTLY on every 2D slice. Internally pred / target are
    #     reshaped to ``(B*D, 1, H, W)`` so the base loss treats each
    #     slice as a standalone 2D binary segmentation problem. Dice /
    #     Tversky reduce only over (H, W).
    #
    #     Pitfall: a slice with no foreground gives Dice ≈
    #     ``(0+smooth)/(0+smooth) ≈ 1`` → loss ≈ 0. With D=12 and FG
    #     concentrated in a few slices, most slices contribute zero
    #     gradient and dilute the useful signal. There is also no
    #     mechanism enforcing across-slice structural coherence, which
    #     can produce "stairstep" artefacts after Gaussian z-blending.
    #
    # "per_volume" (recommended for 2.5D): the loss is computed on the
    #     full per-window volume. Internally pred is reshaped to
    #     ``(B, num_fg, D, H, W)`` and split by class into
    #     ``(B, 1, D, H, W)`` so Dice / Tversky reduce over (D, H, W) as
    #     a single volumetric Dice. Empty slices no longer game the loss
    #     because the whole-window denominator stays large; the network
    #     is also implicitly regularised toward 3D-consistent predictions.
    #
    #     BCE / Focal / Lovász-style losses are mathematically equivalent
    #     under both reductions (per-voxel mean over the same voxels);
    #     only Dice-family aggregation is affected.
    slice_loss_reduction: str = "per_slice"

    # ---- Multi-FOV aux segmentation supervision weights (2.5D mode) ----
    # Used only when ``model.aux_seg_supervision == True``. One weight per
    # aux view (k = 1..n_views-1, where n_views = len(data.multi_res_scales)).
    # The total training loss is::
    #
    #     L_total = L_main(view_0) + Σ_{k=1..n_views-1} w_k * L_aux(view_k)
    #
    # ``L_main`` runs through the full DS+SliceChannel pipeline as before;
    # each ``L_aux`` is a SliceChannelLoss on view k's resampled label at
    # the model's native (H, W) resolution (no DS for aux paths).
    #
    # Empty list → trainer auto-fills with geometric decay ``0.5 ** k``
    # (e.g. n_views=3 → weights=[0.5, 0.25]). Length must equal
    # n_views - 1 when explicitly provided.
    aux_supervision_weights: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Training loop settings."""

    epochs: int = 200

    # Optimizer: "adam", "adamw", "sgd"
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.99   # SGD only
    nesterov: bool = True    # SGD only

    # Scheduler: "cosine", "cosine_warm_restarts", "poly", "step", "plateau", "one_cycle"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    warmup_lr: float = 1e-6
    cosine_min_lr: float = 1e-6
    # Cosine warm restarts: restart period in epochs (T_0), multiplier (T_mult)
    cosine_restart_period: int = 50
    cosine_restart_mult: int = 2
    poly_power: float = 0.9
    step_size: int = 50
    step_gamma: float = 0.1
    plateau_patience: int = 10
    plateau_factor: float = 0.5

    # Gradient accumulation (effective batch = batch_size * accum_steps)
    grad_accum_steps: int = 1

    # Gradient clipping
    grad_clip_norm: float = 12.0

    # Mixed precision (AMP)
    #   amp_dtype:
    #     - "float16" / "fp16": legacy default; requires GradScaler; tends
    #        to overflow on large dice/BCE reductions (handled in-trainer
    #        via fp32 loss cast + logit clamp).
    #     - "bfloat16" / "bf16": Ampere+ only (RTX 30/40, A100, H100...).
    #        Same fp32 dynamic range (no ±inf / NaN from overflow), no
    #        loss scaler needed — skips the unscale pass on every step.
    #     - "auto": resolved at Trainer-build time to "bfloat16" iff the
    #        current CUDA device reports bf16 support (``cuda_capability
    #        >= (8, 0)`` or ``torch.cuda.is_bf16_supported()``), else
    #        falls back to "float16". Recommended default for mixed
    #        fleets; bit-identical to "float16" on pre-Ampere GPUs.
    use_amp: bool = True
    amp_dtype: str = "float16"

    # torch.compile (PyTorch 2.0+, "none", "default", "reduce-overhead", "max-autotune")
    compile_mode: str = "none"

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999

    # Checkpointing
    output_dir: str = "outputs"
    save_every: int = 10
    save_best_metric: str = "mean_dice"
    save_best_mode: str = "max"

    # Early stopping (0 = disabled)
    early_stopping: int = 0

    # Logging
    log_every: int = 10
    val_every: int = 1
    vis_every: int = 10

    # Reproducibility
    seed: int = 42
    deterministic: bool = False

    # Resume: 从训练 checkpoint 完整恢复（model/EMA/optimizer/scheduler/scaler/epoch/RNG）。
    resume: str = ""

    # Pretrain: 仅加载 model 权重作为初始化（迁移学习用）。不恢复任何训练状态：
    #   - epoch 从 0 开始，optimizer/scheduler/scaler/best_metric/patience/RNG 全部不动。
    #   - 若启用 EMA，EMA shadow 会被加载后的 model 权重重新对齐。
    #   - 若同时设置了 `resume` 且 resume 文件存在，则 pretrain 被忽略（resume 优先）。
    pretrain: str = ""

    # 是否对 pretrain 权重 strict 加载。默认 False 以允许 head 形状不一致（不同任务 num_classes）。
    # 加载完成后会日志输出 missing / unexpected keys 数量与示例。
    pretrain_strict: bool = False

    # 当 pretrain checkpoint 含 EMA shadow 时，是否优先用 EMA shadow 作为初始权重
    # （EMA 权重通常更稳定，更适合做迁移起点）。默认 False（用 online 权重）。
    pretrain_load_ema: bool = False


# ---------------------------------------------------------------------------
# Prediction / Inference configuration
# ---------------------------------------------------------------------------
@dataclass
class PredictConfig:
    """Inference settings for z-axis sliding window prediction."""

    # Sliding window overlap ratio along z-axis (0.0 = no overlap, 0.5 = 50%)
    z_overlap: float = 0.5

    # Blending mode for overlapping regions: "gaussian" or "average"
    blend_mode: str = "gaussian"

    # Batch size for inference patches
    batch_size: int = 2

    # Test-time augmentation: flip along axes
    tta_flip: bool = False

    # Binarization threshold for sigmoid output
    threshold: float = 0.5

    # Output directory for predictions
    output_dir: str = "predictions"

    # Save probability maps (in addition to binary masks)
    save_probabilities: bool = False

    # ---- Z-axis interleaved multi-stream prediction (2.5D only) ----
    # Splits the input volume into ``k`` interleaved sub-volumes along z
    # (slices ``i, i+k, i+2k, ...`` for ``i = 0..k-1``), runs the
    # standard 2.5D z-sliding-window inference on each sub-volume
    # independently, then weaves the per-stream probabilities back into
    # the original z indices (``out[:, i::k] = stream_i_prob``). Streams
    # cover disjoint slice sets so the recombination is exact — no
    # cross-stream blending required.
    #
    # Rationale: the 2.5D model sees ``patch_D`` "pseudo-adjacent"
    # channel-slices. Sampling every k-th slice makes each window span
    # ``k * patch_D * z_spacing`` mm physically, widening the effective
    # z receptive field without retraining. Most useful on thin-slice
    # scans where adjacent slices are highly redundant.
    #
    # Distribution-shift caveat: inputs become an apparent-spacing of
    # ``k * z_spacing``. Recommend an A/B vs. k=1 on held-out data
    # before relying on this in production.
    #
    # Disabled by default to preserve legacy behaviour bit-exactly.
    z_interleave_enabled: bool = False

    # Per-volume k is chosen by physical z spacing (mm). With sorted
    # ``z_interleave_thresholds = [t_1, t_2, ..., t_n]`` (ascending) and
    # ``z_interleave_factors = [f_1, f_2, ..., f_n, f_fallback]``
    # (length n+1), the rule is:
    #   z_spacing <= t_1 → k = f_1
    #   t_1 < z_spacing <= t_2 → k = f_2
    #   ...
    #   z_spacing > t_n → k = f_fallback
    # Defaults follow TODO 1: ≤1.0 mm → k=3; (1.0, 1.5] → k=2; >1.5 → k=1.
    z_interleave_thresholds: List[float] = field(
        default_factory=lambda: [1.0, 1.5])
    z_interleave_factors: List[int] = field(
        default_factory=lambda: [3, 2, 1])


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Top-level configuration combining all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    augment: AugConfig = field(default_factory=AugConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)

    def sync(self) -> None:
        """Synchronize dependent fields across sub-configs."""
        if self.data.label_values and self.data.num_classes == 0:
            self.data.num_classes = len(self.data.label_values)

        if self.data.patch_mode == "2_5d":
            # 2.5D mode: planar 2D backbone consuming D slices as channels.
            # Multi-FOV (multi_res_scales) extension: each extra z-FOV view
            # contributes additional input channels.
            #
            # Two channel-layouts are supported, gated by ``data.aux_keep_native_d``:
            #
            #   False (legacy):
            #     Each view k is z-resampled to D channels — total
            #     ``in_channels = D * n_views``.
            #
            #   True (native depth):
            #     View k keeps its native depth ``D_k = round(D * s_k)``;
            #     concatenation along the channel axis gives
            #     ``in_channels = sum_k D_k``. View-0 always uses
            #     ``s_0 == 1`` so ``D_0 == D`` (the supervision target).
            n_views = max(len(self.data.multi_res_scales), 1)
            D = int(self.data.patch_size[0])
            lift = bool(getattr(self.model, "lift_2_5d_to_3d", False))
            if lift:
                # Lift mode: D stays a real spatial axis. The 2D-folded
                # layout is bypassed end-to-end (trainer skips squeeze,
                # loss bypasses SliceChannelLoss). Each FOV view becomes
                # ONE input channel; the stem consumes ``n_views`` input
                # channels and the model output is single-resolution
                # ``(B, num_fg, D, H, W)``. Mutually exclusive with
                # aux_keep_native_d (validated below).
                self.model.spatial_dims = 3
                self.model.in_channels = n_views
            else:
                self.model.spatial_dims = 2
            if lift:
                pass  # in_channels already set above; skip folded layouts
            elif self.data.aux_keep_native_d and n_views > 1:
                # Native-depth layout. Force edge_pad — the single-cube
                # extraction always uses ``extract_z_patch_padded`` so a
                # ``stretch`` value would dangle without effect (and worse,
                # mislead readers). Quietly upgrade and log.
                if self.data.z_boundary_mode != "edge_pad":
                    logger.info(
                        "aux_keep_native_d=True implies z_boundary_mode='edge_pad'; "
                        "auto-upgraded from %r.", self.data.z_boundary_mode)
                    self.data.z_boundary_mode = "edge_pad"
                depths = [int(round(D * float(s))) for s in self.data.multi_res_scales]
                # View 0 must equal D exactly (s_0 == 1.0 invariant).
                depths[0] = D
                self.model.in_channels = int(sum(depths))
            else:
                self.model.in_channels = D * n_views
        else:
            # 3D modes: in_channels follows multi_res_scales (legacy).
            # Both z_axis and cubic stack per-scale views as input channels;
            # a single scale ([1.0]) gives the legacy 1-channel input.
            #
            # ``data.keep_native_multi_res`` does NOT change the channel
            # count: even though the dataset emits a ``(1, eD_max, ...)``
            # single-cube tensor, the trainer (R2) splits it back to
            # ``(B, C_res, eD, eH, eW)`` before forward — model contract
            # is bit-identical. The flag here only affects the
            # dataset/trainer geometry path; the model construction
            # below is untouched.
            self.model.in_channels = len(self.data.multi_res_scales)
            # Auto-upgrade z_boundary_mode for the lazy-extraction path
            # (z_axis only — cubic doesn't use this knob). Mirrors the
            # 2.5D ``aux_keep_native_d`` behaviour: ``stretch`` would
            # have no consumer in the single-max-FOV-cube path.
            if (self.data.keep_native_multi_res
                    and self.data.patch_mode == "z_axis"
                    and len(self.data.multi_res_scales) > 1
                    and self.data.z_boundary_mode != "edge_pad"):
                logger.info(
                    "keep_native_multi_res=True implies z_boundary_mode="
                    "'edge_pad'; auto-upgraded from %r.",
                    self.data.z_boundary_mode)
                self.data.z_boundary_mode = "edge_pad"

        # nnU-Net ResEnc preset: populate per-stage block counts when the
        # user has not supplied explicit lists.
        self._apply_resenc_preset()

    def _apply_resenc_preset(self) -> None:
        """Expand ``model.resenc_preset`` into per-stage block counts."""
        mc = self.model
        preset = (mc.resenc_preset or "none").lower()
        if preset == "none":
            return
        if mc.encoder_blocks_per_stage and mc.decoder_blocks_per_stage:
            # User-supplied lists win over preset.
            return

        n_levels = len(mc.encoder_channels)
        templates = {
            "s":  [1, 2, 2, 2, 2, 2],
            "m":  [1, 3, 4, 6, 6, 6],
            "l":  [1, 3, 4, 6, 6, 6, 6],
            "xl": [1, 4, 6, 8, 8, 10, 10, 10],
        }
        if preset not in templates:
            return  # validate() will flag the error.

        tpl = templates[preset]
        # Trim or extend (repeating the deepest-stage count) to match n_levels.
        if n_levels <= len(tpl):
            enc_blocks = tpl[:n_levels]
        else:
            enc_blocks = tpl + [tpl[-1]] * (n_levels - len(tpl))

        if not mc.encoder_blocks_per_stage:
            mc.encoder_blocks_per_stage = enc_blocks
        if not mc.decoder_blocks_per_stage:
            # Lightweight decoder = 1 block / stage (ResEnc recipe).
            mc.decoder_blocks_per_stage = [1] * (n_levels - 1)

    def validate(self) -> None:
        """Validate configuration for consistency."""
        assert self.model.backbone in ("resnet", "convnext"), \
            f"Invalid backbone: {self.model.backbone}"
        assert self.model.spatial_dims in (2, 3), \
            f"Invalid spatial_dims: {self.model.spatial_dims} (must be 2 or 3)"
        assert self.model.norm_type in ("batch", "instance", "group"), \
            f"Invalid norm: {self.model.norm_type}"
        assert self.model.activation in ("relu", "leakyrelu", "gelu", "swish"), \
            f"Invalid activation: {self.model.activation}"
        assert self.model.downsample_mode in (
            "conv", "maxpool", "avgpool", "blurpool", "pixelunshuffle",
        ), f"Invalid downsample_mode: {self.model.downsample_mode}"
        assert self.model.upsample_mode in (
            "transpose", "trilinear", "nearest", "pixelshuffle",
            "carafe", "dysample",
        ), f"Invalid upsample_mode: {self.model.upsample_mode}"
        assert self.model.skip_mode in ("cat", "add"), \
            f"Invalid skip_mode: {self.model.skip_mode}"
        assert self.augment.wmap_interp_mode in ("nearest", "bilinear"), (
            f"Invalid augment.wmap_interp_mode: "
            f"{self.augment.wmap_interp_mode!r}; expected "
            f"'nearest' (discrete fg/bg weights, default) or "
            f"'bilinear' (continuous hand-annotated weights).")
        assert self.model.attention_type in (
            "none", "se", "eca", "cbam", "coord",
        ), f"Invalid attention_type: {self.model.attention_type}"
        assert self.model.stem_mode in (
            "conv3", "conv7", "dual", "patch2", "patch4",
        ), f"Invalid stem_mode: {self.model.stem_mode}"
        assert self.model.context_fusion in (
            "shared_stem", "multi_stem_proj", "hierarchical",
        ), f"Invalid context_fusion: {self.model.context_fusion!r}"
        assert getattr(self.model, "aux_head_mode", "linear") in (
            "linear", "conv",
        ), f"Invalid aux_head_mode: {self.model.aux_head_mode!r}"
        assert self.model.decoder_type in ("unet", "unetpp", "unet3p"), \
            f"Invalid decoder_type: {self.model.decoder_type}"
        assert self.model.unet3p_cat_channels > 0, \
            "unet3p_cat_channels must be > 0"
        assert self.model.block_type in (
            "basic", "preact", "bottleneck", "r2plus1d"), \
            f"Invalid block_type: {self.model.block_type}"
        # ``r2plus1d`` factorises 3D conv into spatial (1,3,3) + temporal
        # (3,1,1) — the temporal sub-conv only reaches neighbouring slices
        # when D is a real spatial axis, i.e. spatial_dims=3. In 2.5D mode
        # D is folded into channels and the temporal kernel becomes a
        # no-op cross-channel mixer; reject it up-front with a precise
        # diagnostic instead of silently degrading to that pathological
        # behaviour at forward-time.
        if self.model.block_type == "r2plus1d":
            assert self.model.spatial_dims == 3, (
                "model.block_type='r2plus1d' requires spatial_dims=3 "
                "(D must be a real spatial axis). It is incompatible "
                "with the 2.5D patch_mode where D is folded into the "
                "channel axis. To use Plan A on z-slab data, switch "
                "your config to patch_mode='z_axis' (3D thin-slab) and "
                "keep block_type='r2plus1d'.")
        assert self.model.resenc_preset in ("none", "S", "M", "L", "XL"), \
            f"Invalid resenc_preset: {self.model.resenc_preset}"
        # Per-stage block-count lengths must align with encoder depth.
        n_levels = len(self.model.encoder_channels)
        ebps = self.model.encoder_blocks_per_stage
        dbps = self.model.decoder_blocks_per_stage
        if ebps:
            assert len(ebps) == n_levels, (
                f"encoder_blocks_per_stage must have {n_levels} entries "
                f"(= len(encoder_channels)); got {len(ebps)}")
            assert all(b >= 1 for b in ebps), \
                "encoder_blocks_per_stage entries must all be >= 1"
        if dbps:
            assert len(dbps) == n_levels - 1, (
                f"decoder_blocks_per_stage must have {n_levels - 1} entries "
                f"(= len(encoder_channels) - 1); got {len(dbps)}")
            assert all(b >= 1 for b in dbps), \
                "decoder_blocks_per_stage entries must all be >= 1"
        assert self.loss.name in (
            # Classical single losses.
            "dice", "bce", "focal", "tversky",
            # High-quality single losses (Round "new losses").
            "gdl", "focal_tversky", "lovasz", "cldice",
            # Compounds.
            "dice_bce", "dice_focal", "dice_tversky",
            "focal_plus_tversky",   # legacy (Focal + Tversky summed)
            "dice_cldice",          # Shit et al. 2021 recipe
            "dice_focal_tversky",   # Dice + Abraham 2019 FTL
            "dice_lovasz", "bce_lovasz",
            "gdl_bce", "gdl_focal",
        ), f"Invalid loss: {self.loss.name}"
        assert self.loss.gdl_weight_type in ("square", "simple", "uniform"), (
            f"Invalid gdl_weight_type: {self.loss.gdl_weight_type}")
        assert self.loss.focal_tversky_gamma > 0, (
            f"focal_tversky_gamma must be > 0, got {self.loss.focal_tversky_gamma}")
        assert self.loss.cldice_iter >= 1, (
            f"cldice_iter must be >= 1, got {self.loss.cldice_iter}")
        assert self.loss.slice_loss_reduction in ("per_slice", "per_volume"), (
            f"Invalid slice_loss_reduction: {self.loss.slice_loss_reduction!r}; "
            "expected 'per_slice' or 'per_volume'.")
        assert self.train.optimizer in ("adam", "adamw", "sgd"), \
            f"Invalid optimizer: {self.train.optimizer}"
        assert self.train.scheduler in (
            "cosine", "cosine_warm_restarts", "poly", "step", "plateau", "one_cycle",
        ), f"Invalid scheduler: {self.train.scheduler}"
        assert len(self.data.patch_size) == 3, \
            "patch_size must be [D, H, W]"
        assert self.data.patch_mode in ("z_axis", "cubic", "whole", "2_5d"), \
            f"Invalid patch_mode: {self.data.patch_mode}"
        assert self.data.z_boundary_mode in ("stretch", "edge_pad"), (
            f"Invalid z_boundary_mode: {self.data.z_boundary_mode!r}; "
            "expected 'stretch' or 'edge_pad'.")
        if self.data.patch_mode == "whole":
            # Multi-resolution has no physical meaning in whole-volume mode:
            # the input already spans the entire volume, there is nothing
            # outside to extract a "wider FOV" view from.
            assert len(self.data.multi_res_scales) == 1 \
                and self.data.multi_res_scales[0] == 1.0, (
                "whole-volume mode requires multi_res_scales=[1.0]; got "
                f"{self.data.multi_res_scales}.")
        # ``aux_keep_native_d`` is meaningful only in 2.5D + multi-view +
        # aux-supervision. Reject misuse early with a precise diagnostic
        # rather than silently letting downstream surgery fire.
        if self.data.aux_keep_native_d:
            assert self.data.patch_mode == "2_5d", (
                "data.aux_keep_native_d=True is only valid in patch_mode="
                f"'2_5d'; got patch_mode={self.data.patch_mode!r}.")
            assert len(self.data.multi_res_scales) > 1, (
                "data.aux_keep_native_d=True requires at least one auxiliary "
                "view (len(multi_res_scales) > 1); got "
                f"multi_res_scales={self.data.multi_res_scales}.")

        # ``keep_native_multi_res`` is the 3D analogue of
        # ``aux_keep_native_d``: lazy single-max-FOV-cube extraction
        # in the dataset, with per-view crop+resize deferred to the
        # trainer (R2). Strict gating early so a misuse never silently
        # downgrades the data emission contract.
        if self.data.keep_native_multi_res:
            assert self.data.patch_mode in ("z_axis", "cubic"), (
                "data.keep_native_multi_res=True is only valid in 3D "
                "patch_mode in {'z_axis', 'cubic'}; got patch_mode="
                f"{self.data.patch_mode!r}. Use data.aux_keep_native_d "
                "for the 2.5D analogue.")
            assert len(self.data.multi_res_scales) > 1, (
                "data.keep_native_multi_res=True requires at least one "
                "auxiliary view (len(multi_res_scales) > 1); got "
                f"multi_res_scales={self.data.multi_res_scales}. "
                "With a single scale the lazy path has nothing to defer.")
            assert float(self.data.multi_res_scales[0]) == 1.0, (
                "data.keep_native_multi_res=True requires "
                "multi_res_scales[0] == 1.0 (view 0 = canonical "
                f"geometry); got multi_res_scales={self.data.multi_res_scales}.")
            assert not self.data.aux_keep_native_d, (
                "data.keep_native_multi_res and data.aux_keep_native_d "
                "are mutually exclusive (3D vs 2.5D analogues). Pick one.")
            if self.data.patch_mode == "z_axis":
                assert self.data.z_boundary_mode == "edge_pad", (
                    "keep_native_multi_res=True (z_axis) requires "
                    "z_boundary_mode='edge_pad' (set automatically by "
                    f"sync()); got {self.data.z_boundary_mode!r}.")

        if self.data.patch_mode == "2_5d":
            # 2.5D mode invariants enforced by sync(); re-check here so a
            # stale config caught after manual edit fails fast.
            assert len(self.data.multi_res_scales) >= 1, (
                "2.5D mode requires at least one entry in multi_res_scales.")
            assert self.data.multi_res_scales[0] == 1.0, (
                "2.5D mode requires multi_res_scales[0] == 1.0 — view 0 is "
                "the true-geometry FOV used as the prediction target. "
                f"Got multi_res_scales={self.data.multi_res_scales}.")
            n_views = len(self.data.multi_res_scales)
            lift = bool(getattr(self.model, "lift_2_5d_to_3d", False))
            if lift:
                # Lift mode: D preserved as a real spatial axis, model is
                # a true 3D UNet over (B, n_views, D, H, W). Mutually
                # exclusive with the folded-D channel-packing layouts.
                assert self.model.spatial_dims == 3, (
                    "lift_2_5d_to_3d=True requires model.spatial_dims=3 "
                    "(set automatically by sync()).")
                assert self.model.in_channels == n_views, (
                    f"lift_2_5d_to_3d=True requires model.in_channels == "
                    f"len(multi_res_scales) = {n_views}; got "
                    f"in_channels={self.model.in_channels}. Set automatically "
                    f"by sync(); a mismatch means in_channels was hand-edited.")
                assert not self.data.aux_keep_native_d, (
                    "lift_2_5d_to_3d=True is mutually exclusive with "
                    "data.aux_keep_native_d (folded-D channel slabs vs. "
                    "real-D spatial axis are incompatible). Disable one.")
                # lift + aux_seg_supervision IS now supported. The aux
                # heads emit ``(B, num_fg, D, H, W)`` (3D 1×1×1 conv,
                # gated by ``spatial_dims=3``) and the trainer routes
                # the per-view aux loss through MultiResolutionLoss
                # (num_res=1) instead of SliceChannelLoss. The only
                # remaining mutex is with ``aux_keep_native_d`` (folded-D
                # channel slabs cannot coexist with the real-D spatial
                # axis), already enforced above.
                # In lift mode the loss bypasses SliceChannelLoss entirely,
                # so slice_loss_reduction has no effect. Surface the dead
                # knob now rather than silently letting it look meaningful.
                if getattr(self.loss, "slice_loss_reduction", "per_slice") not in ("per_slice", "per_volume"):
                    pass  # type validation handled elsewhere
                # No need to validate in_channels arithmetic further; sync()
                # owns the formula and lift skips the folded layouts above.

                # --- Geometric constraint: thin-slab D must survive every
                # encoder downsample. The shared ``Downsample`` block uses
                # an isotropic factor-2 stride (``conv``/``maxpool``/...) with
                # kernel_size=2; given ``n_levels`` encoder stages there are
                # ``n_down = n_levels - 1`` halvings, so D must be divisible
                # by ``2**n_down`` (and >= it). The 2.5D folded path was
                # immune because D was on the channel axis; lift mode hits
                # this constraint head-on. Surface it now with a precise
                # diagnostic — otherwise the user gets an opaque
                # ``Conv3d kernel size > input size`` error deep in forward.
                n_levels = len(self.model.encoder_channels)
                n_down = n_levels - 1
                D = int(self.data.patch_size[0])
                req = 1 << n_down
                if D < req or D % req != 0:
                    raise AssertionError(
                        f"lift_2_5d_to_3d=True with len(encoder_channels)="
                        f"{n_levels} requires patch_size[0] (D={D}) to be "
                        f"divisible by 2**(n_levels-1)={req}. The shared "
                        f"Downsample block halves every spatial axis (D "
                        f"included) at each stage, and an isotropic "
                        f"kernel-2/stride-2 conv on D<2 fails. Fixes:\n"
                        f"  * Increase patch_size[0] to >= {req} (and a "
                        f"multiple of {req}).\n"
                        f"  * Or reduce len(encoder_channels) so 2**(n-1) "
                        f"<= D (e.g. 4 stages need D>=8, 3 stages need D>=4).\n"
                        f"  * Anisotropic per-axis strides (keep D unchanged "
                        f"at deep stages) is not yet implemented and would "
                        f"be a separate feature.")
            else:
                assert self.model.spatial_dims == 2, (
                    "2.5D mode requires model.spatial_dims=2 (set "
                    "automatically by sync()). To run a 3D model on the "
                    "same 2.5D pipeline (Plan A), set "
                    "model.lift_2_5d_to_3d=True (and typically "
                    "model.block_type='r2plus1d').")
            if (not lift) and self.data.aux_keep_native_d and n_views > 1:
                # Native-depth layout: in_channels = sum_k round(D * s_k),
                # with s_0 == 1.0 fixing D_0 == D. This is the channel
                # count after the trainer's per-view center-crop and
                # cat-along-channel step before the 2D forward.
                depths = self.aux_view_depths
                expected_in = int(sum(depths))
                assert self.model.in_channels == expected_in, (
                    f"2.5D + aux_keep_native_d=True requires "
                    f"model.in_channels == sum(round(D * s_k)) = "
                    f"sum({depths}) = {expected_in}; got "
                    f"in_channels={self.model.in_channels}. "
                    f"This is normally set automatically by sync(); a "
                    f"mismatch here means in_channels was hand-edited "
                    f"after sync() ran.")
                assert self.data.z_boundary_mode == "edge_pad", (
                    "aux_keep_native_d=True requires z_boundary_mode="
                    "'edge_pad' (set automatically by sync()); got "
                    f"{self.data.z_boundary_mode!r}.")
                # ON-mode is meaningful only when aux supervision is on —
                # otherwise the wider FOVs would contribute extra channels
                # to the model trunk but never receive a target signal,
                # which is almost certainly a configuration error.
                assert getattr(self.model, "aux_seg_supervision", False), (
                    "aux_keep_native_d=True is only meaningful with "
                    "model.aux_seg_supervision=True (each native-depth "
                    "view k must drive an aux head predicting "
                    "(B, num_fg * D_k, H, W)). Either enable "
                    "aux_seg_supervision or set aux_keep_native_d=False.")
            elif not lift:
                expected_in = int(self.data.patch_size[0]) * n_views
                assert self.model.in_channels == expected_in, (
                    f"2.5D mode requires model.in_channels == "
                    f"patch_size[0] * len(multi_res_scales) = "
                    f"{self.data.patch_size[0]} * {n_views} = {expected_in}; "
                    f"got in_channels={self.model.in_channels}.")
            # Plan C constraints: aux view k injects at encoder stage k,
            # so we need at least one stage per aux view + the main one.
            if self.model.context_fusion == "hierarchical" and n_views > 1:
                n_stages = len(self.model.encoder_channels)
                assert n_views <= n_stages, (
                    f"context_fusion='hierarchical' requires "
                    f"len(multi_res_scales) <= len(encoder_channels) so each "
                    f"aux view k=1..n_views-1 has a matching stage k to "
                    f"inject into; got n_views={n_views}, "
                    f"n_stages={n_stages}.")
                # The deepest aux stem has stride main_stem_stride * 2^(n_views-1).
                # Validate H and W are divisible by that stride to avoid a
                # silent spatial mismatch at fusion time.
                stem_stride_map = {
                    "conv3": 1, "conv7": 1, "dual": 1,
                    "patch2": 2, "patch4": 4,
                }
                s0 = stem_stride_map[self.model.stem_mode]
                deepest = s0 * (2 ** (n_views - 1))
                pH, pW = int(self.data.patch_size[1]), int(self.data.patch_size[2])
                assert pH % deepest == 0 and pW % deepest == 0, (
                    f"context_fusion='hierarchical' with n_views={n_views} "
                    f"and stem_mode={self.model.stem_mode!r} requires "
                    f"patch_size[1] and patch_size[2] divisible by "
                    f"{deepest}; got patch_size=({pH}, {pW}).")
            # Aux seg supervision constraints — only meaningful when
            # there is at least one aux view (n_views > 1).
            if getattr(self.model, "aux_seg_supervision", False):
                assert n_views > 1, (
                    "model.aux_seg_supervision=True requires "
                    "len(multi_res_scales) > 1 (at least one aux FOV "
                    "to supervise); got n_views=1.")
                aw = list(getattr(self.loss, "aux_supervision_weights", []))
                if aw:
                    assert len(aw) == n_views - 1, (
                        f"loss.aux_supervision_weights length ({len(aw)}) "
                        f"must equal n_views-1 ({n_views - 1}); got {aw}.")
                    assert all(w >= 0 for w in aw), (
                        f"loss.aux_supervision_weights must be non-negative; "
                        f"got {aw}.")
                # Plan C requires len(decoder)>=n_views to give each aux
                # view a unique decoder feature index. ``unet`` decoder
                # produces n_levels-1 features and we mount aux head k on
                # dec_features[-1-k] for k=1..n_views-1 → need n_views-1
                # < n_levels-1, i.e. n_views < n_levels. Plan A (parallel)
                # has no such constraint.
                if self.model.context_fusion == "hierarchical":
                    n_levels = len(self.model.encoder_channels)
                    assert n_views < n_levels, (
                        f"aux_seg_supervision with context_fusion="
                        f"'hierarchical' requires n_views < "
                        f"len(encoder_channels) (one decoder feature per "
                        f"aux view + the main one); got n_views={n_views}, "
                        f"n_levels={n_levels}.")
        assert self.data.aug_oversample_ratio >= 1.0, \
            "aug_oversample_ratio must be >= 1.0"
        assert len(self.data.multi_res_scales) >= 1, \
            "multi_res_scales must have at least one scale (e.g. [1.0])"
        assert all(s >= 1.0 for s in self.data.multi_res_scales), \
            "All multi_res_scales must be >= 1.0"
        # Multi-resolution is now supported in both z_axis and cubic modes.
        # In z_axis mode the scale factor applies to the z-axis only
        # (see DataConfig.multi_res_scales docstring); `sync()` auto-sets
        # `model.in_channels = len(multi_res_scales)` in both modes so the
        # network input/output channel count matches the stacked views.
        assert self.train.save_best_mode in ("max", "min"), \
            f"Invalid save_best_mode: {self.train.save_best_mode}"
        # ---- z-interleaved 2.5D inference: shape & monotonicity checks ----
        # Off by default; only validated when the flag is on, so legacy
        # configs without these fields remain bit-exactly accepted.
        if self.predict.z_interleave_enabled:
            assert self.data.patch_mode == "2_5d", (
                "predict.z_interleave_enabled=True is only valid for "
                f"patch_mode='2_5d'; got {self.data.patch_mode!r}. The "
                "interleaved scheme widens the z receptive field of the "
                "2D-folded D-channel input — it has no effect on true-3D "
                "patch modes (z_axis/cubic/whole).")
            thr = self.predict.z_interleave_thresholds
            fac = self.predict.z_interleave_factors
            assert len(fac) == len(thr) + 1, (
                "predict.z_interleave_factors must have exactly "
                "len(z_interleave_thresholds)+1 entries (one per spacing "
                f"bucket + a fallback for >max-threshold); got "
                f"thresholds={thr}, factors={fac}.")
            assert all(t > 0 for t in thr), (
                f"predict.z_interleave_thresholds must all be > 0; got {thr}.")
            assert thr == sorted(thr), (
                f"predict.z_interleave_thresholds must be ascending; got {thr}.")
            assert all(int(f) >= 1 for f in fac), (
                f"predict.z_interleave_factors must all be >= 1; got {fac}.")
            # edge_pad keeps short sub-streams geometrically faithful; the
            # 'stretch' legacy behaviour would rescale a (D//k)-slice
            # tail-stream up to patch_D and partially defeat the
            # interleaving's whole point. Warn rather than hard-fail so
            # an existing 'stretch' config can still opt-in for a probe.
            if self.data.z_boundary_mode != "edge_pad":
                logger.warning(
                    "predict.z_interleave_enabled=True with "
                    "z_boundary_mode=%r: short sub-streams will be "
                    "stretched along z when their length < patch_D, "
                    "which dilutes the interleave effect. Prefer "
                    "'edge_pad'.", self.data.z_boundary_mode)
        if self.data.num_classes < 2:
            logger.warning("num_classes=%d < 2, will auto-detect from data.",
                           self.data.num_classes)

    @property
    def num_fg_classes(self) -> int:
        """Number of foreground classes (excluding background)."""
        return max(self.data.num_classes - 1, 1)

    @property
    def aux_view_depths(self) -> List[int]:
        """Per-view native depths ``D_k = round(D * s_k)`` for 2.5D mode.

        Always returns a list of length ``len(data.multi_res_scales)``,
        with element 0 fixed to ``D`` (view 0 invariant). For modes other
        than ``2_5d`` returns an empty list.

        This helper is shape-only — it does NOT depend on
        ``data.aux_keep_native_d``. Consumers gate on the flag explicitly:
          - flag OFF: ignore this list and use the legacy ``D * n_views``
            channel layout.
          - flag ON: use the list to (a) center-crop each view at its
            native depth, (b) size per-view stems & aux heads.
        """
        if self.data.patch_mode != "2_5d":
            return []
        D = int(self.data.patch_size[0])
        depths = [int(round(D * float(s))) for s in self.data.multi_res_scales]
        if depths:
            depths[0] = D  # enforce s_0 == 1.0 invariant
        return depths


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------
_SUB_CONFIGS = {
    "data": DataConfig,
    "augment": AugConfig,
    "model": ModelConfig,
    "loss": LossConfig,
    "train": TrainConfig,
    "predict": PredictConfig,
}


def _dataclass_from_dict(cls, d: Dict[str, Any]):
    """Recursively construct a dataclass from a dict."""
    if not isinstance(d, dict):
        return d
    field_names = {f.name for f in fields(cls)}
    kwargs = {}
    for k, v in d.items():
        if k not in field_names:
            logger.warning("Unknown config key: %s", k)
            continue
        if k in _SUB_CONFIGS and isinstance(v, dict):
            v = _dataclass_from_dict(_SUB_CONFIGS[k], v)
        kwargs[k] = v
    return cls(**kwargs)


def load_config(path: Union[str, Path]) -> Config:
    """Load configuration from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = _dataclass_from_dict(Config, raw)
    cfg.sync()
    cfg.validate()
    return cfg


def save_config(cfg: Config, path: Union[str, Path]) -> None:
    """Save configuration to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False,
                  sort_keys=False, allow_unicode=True)
