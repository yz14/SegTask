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
    #   "z_axis" — slide along z-axis, extract D slices, resize H,W to target
    #   "cubic"  — sample center (x,y,z), extract full 3D cube of patch_size
    patch_mode: str = "z_axis"

    # Augmentation oversample ratio (cubic mode only).
    # Extract patch_size * ratio, augment on GPU, then center-crop to patch_size.
    # This avoids blank edges from spatial transforms. 1.0 = disabled, 1.4~1.5 recommended.
    aug_oversample_ratio: float = 1.0

    # Multi-resolution input (cubic mode only).
    # Scales at which to extract patches centered at the same point.
    # Each scale's patch is resized to extract_size and stacked as a channel.
    # [1.0] = single-resolution (1-channel), [1.0, 1.5, 2.0] = 3-channel.
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

    # DataLoader
    batch_size: int = 2
    num_workers: int = 4
    pin_memory: bool = True

    # Foreground oversampling: probability of centering patch on foreground
    foreground_oversample_ratio: float = 0.5

    # Samples per volume per epoch (controls epoch length)
    samples_per_volume: int = 8

    # Caching: "none" or "memory"
    cache_mode: str = "memory"


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

    # Blocks per encoder/decoder level
    blocks_per_level: int = 2

    # Normalization: "batch", "instance", "group"
    norm_type: str = "instance"
    norm_groups: int = 8

    # Activation: "relu", "leakyrelu", "gelu", "swish"
    activation: str = "leakyrelu"

    # Dropout in blocks
    dropout: float = 0.0

    # Squeeze-and-Excitation attention (channel attention)
    use_se: bool = False
    se_reduction: int = 16

    # Deep supervision: output predictions at multiple decoder levels
    deep_supervision: bool = False

    # Upsampling mode: "transpose" or "trilinear"
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

        # Auto-set in_channels from multi_res_scales (always >= 1)
        if self.data.patch_mode == "cubic":
            self.model.in_channels = len(self.data.multi_res_scales)

    def validate(self) -> None:
        """Validate configuration for consistency."""
        assert self.model.backbone in ("resnet", "convnext"), \
            f"Invalid backbone: {self.model.backbone}"
        assert self.model.norm_type in ("batch", "instance", "group"), \
            f"Invalid norm: {self.model.norm_type}"
        assert self.model.activation in ("relu", "leakyrelu", "gelu", "swish"), \
            f"Invalid activation: {self.model.activation}"
        assert self.loss.name in (
            "dice", "bce", "dice_bce", "focal", "dice_focal", "tversky",
        ), f"Invalid loss: {self.loss.name}"
        assert self.train.optimizer in ("adam", "adamw", "sgd"), \
            f"Invalid optimizer: {self.train.optimizer}"
        assert self.train.scheduler in (
            "cosine", "cosine_warm_restarts", "poly", "step", "plateau", "one_cycle",
        ), f"Invalid scheduler: {self.train.scheduler}"
        assert len(self.data.patch_size) == 3, \
            "patch_size must be [D, H, W]"
        assert self.data.patch_mode in ("z_axis", "cubic"), \
            f"Invalid patch_mode: {self.data.patch_mode}"
        assert self.data.aug_oversample_ratio >= 1.0, \
            "aug_oversample_ratio must be >= 1.0"
        assert len(self.data.multi_res_scales) >= 1, \
            "multi_res_scales must have at least one scale (e.g. [1.0])"
        assert all(s >= 1.0 for s in self.data.multi_res_scales), \
            "All multi_res_scales must be >= 1.0"
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
