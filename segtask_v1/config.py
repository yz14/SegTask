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
    image_suffix: str = ".nii.gz"
    label_suffix: str = ".nii.gz"

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


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """UNet model architecture settings."""

    # Backbone: "resnet" or "convnext"
    backbone: str = "resnet"

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

    # Residual block variant (see models.resnet): "basic" | "preact" | "bottleneck".
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

    # Stem / patch-embed (see models.stem.build_stem):
    # "conv3" | "conv7" | "dual" | "patch2" | "patch4".
    # patchN stems reduce input resolution by N; UNet3D adds a matching
    # trilinear upsample on the main output to restore original resolution.
    stem_mode: str = "conv3"

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

    # Resume
    resume: str = ""


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

        # Auto-set in_channels from multi_res_scales (always >= 1). Both
        # patch modes now stack per-scale views as input channels; a single
        # scale ([1.0]) gives the legacy 1-channel input.
        self.model.in_channels = len(self.data.multi_res_scales)

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
        assert self.model.attention_type in (
            "none", "se", "eca", "cbam", "coord",
        ), f"Invalid attention_type: {self.model.attention_type}"
        assert self.model.stem_mode in (
            "conv3", "conv7", "dual", "patch2", "patch4",
        ), f"Invalid stem_mode: {self.model.stem_mode}"
        assert self.model.decoder_type in ("unet", "unetpp", "unet3p"), \
            f"Invalid decoder_type: {self.model.decoder_type}"
        assert self.model.unet3p_cat_channels > 0, \
            "unet3p_cat_channels must be > 0"
        assert self.model.block_type in ("basic", "preact", "bottleneck"), \
            f"Invalid block_type: {self.model.block_type}"
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
        assert self.train.optimizer in ("adam", "adamw", "sgd"), \
            f"Invalid optimizer: {self.train.optimizer}"
        assert self.train.scheduler in (
            "cosine", "cosine_warm_restarts", "poly", "step", "plateau", "one_cycle",
        ), f"Invalid scheduler: {self.train.scheduler}"
        assert len(self.data.patch_size) == 3, \
            "patch_size must be [D, H, W]"
        assert self.data.patch_mode in ("z_axis", "cubic", "whole"), \
            f"Invalid patch_mode: {self.data.patch_mode}"
        if self.data.patch_mode == "whole":
            # Multi-resolution has no physical meaning in whole-volume mode:
            # the input already spans the entire volume, there is nothing
            # outside to extract a "wider FOV" view from.
            assert len(self.data.multi_res_scales) == 1 \
                and self.data.multi_res_scales[0] == 1.0, (
                "whole-volume mode requires multi_res_scales=[1.0]; got "
                f"{self.data.multi_res_scales}.")
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
        if self.data.num_classes < 2:
            logger.warning("num_classes=%d < 2, will auto-detect from data.",
                           self.data.num_classes)

    @property
    def num_fg_classes(self) -> int:
        """Number of foreground classes (excluding background)."""
        return max(self.data.num_classes - 1, 1)


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
