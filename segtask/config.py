"""Configuration system using dataclasses + YAML.

All tunable parameters are centralized here. The YAML config file maps
directly to nested dataclasses for type safety and IDE autocompletion.
"""

from __future__ import annotations

import copy
import logging
import os
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

    # Paths
    image_dir: str = ""
    label_dir: str = ""
    meta_csv: str = ""
    image_suffix: str = ".nii.gz"
    label_suffix: str = ".nii.gz"
    label_prefix: str = ""
    label_postfix: str = ""

    # Label mapping: which integer labels in the mask to use
    # e.g. [0, 1] for binary, [0, 1, 2] for 3-class
    # empty = auto-detect from data
    label_values: List[int] = field(default_factory=list)
    num_classes: int = 0  # auto-set from label_values if 0

    # Data mode: "2d", "2.5d", "3d"
    mode: str = "2.5d"

    # 2.5D settings: sample C=num_slices_per_side*2+1 contiguous slices
    # For prediction, only the center slice(s) are kept
    num_slices_per_side: int = 1  # total input channels = 2*num_slices_per_side+1

    # 2.5D depth-aware resizing:
    # - "uniform": resize all volumes to target_depth slices (depth × spatial)
    # - "keep_depth": keep original depth, only resize spatial dimensions
    #   → REQUIRED for 2.5D: preserving head/tail slices is essential
    depth_resize_mode: str = "keep_depth"
    # target_depth: only used when depth_resize_mode == "uniform"
    # WARNING: uniform mode DESTROYS head/tail slices and degrades 3D context
    target_depth: int = 64

    # 2D/2.5D target spatial size [H, W] — all slices are RESIZED to this
    # size for uniform batching. Uses bilinear interpolation for images and
    # nearest-neighbor for labels. Works at both train and test time.
    target_size: List[int] = field(default_factory=lambda: [256, 256])

    # 3D patch settings
    patch_size: List[int] = field(default_factory=lambda: [96, 96, 96])

    # Intensity windowing (HU for CT)
    intensity_min: float = -1024.0
    intensity_max: float = 3071.0
    # Normalization: "minmax" → [0,1], "zscore" → zero-mean unit-var
    normalize: str = "minmax"
    # Global statistics for zscore (computed from training set if empty)
    global_mean: float = 0.0
    global_std: float = 1.0

    # Spatial resampling (0 = keep original spacing)
    target_spacing: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # Train/val/test split
    # "meta" = use meta.csv split column; "random" = random split
    split_method: str = "meta"
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_seed: int = 42

    # DataLoader
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

    # Sampling
    # foreground oversampling: probability of sampling a patch containing foreground
    foreground_oversample_ratio: float = 0.5
    # per-class sampling weights (empty = uniform)
    class_sample_weights: List[float] = field(default_factory=list)

    # Caching: "none", "memory", "disk"
    cache_mode: str = "none"
    cache_dir: str = ""


# ---------------------------------------------------------------------------
# Augmentation configuration
# ---------------------------------------------------------------------------
@dataclass
class AugmentConfig:
    """Data augmentation settings (applied on GPU when possible)."""

    enabled: bool = True

    # Spatial
    random_flip_axes: List[int] = field(default_factory=lambda: [0, 1])
    random_flip_prob: float = 0.5

    random_rotate_prob: float = 0.3
    random_rotate_range: float = 15.0  # degrees

    random_scale_prob: float = 0.2
    random_scale_range: List[float] = field(default_factory=lambda: [0.85, 1.15])

    elastic_deform_prob: float = 0.0
    elastic_deform_sigma: float = 5.0
    elastic_deform_alpha: float = 100.0

    # Intensity (image only)
    random_brightness_prob: float = 0.3
    random_brightness_range: List[float] = field(default_factory=lambda: [-0.1, 0.1])

    random_contrast_prob: float = 0.3
    random_contrast_range: List[float] = field(default_factory=lambda: [0.8, 1.2])

    random_gamma_prob: float = 0.2
    random_gamma_range: List[float] = field(default_factory=lambda: [0.8, 1.2])

    gaussian_noise_prob: float = 0.2
    gaussian_noise_std: float = 0.05

    gaussian_blur_prob: float = 0.1
    gaussian_blur_sigma: List[float] = field(default_factory=lambda: [0.5, 1.5])

    # Cutout / random erasing
    cutout_prob: float = 0.0
    cutout_num_holes: int = 1
    cutout_size_ratio: List[float] = field(default_factory=lambda: [0.05, 0.15])

    # Mixup (applied at batch level during training)
    mixup_prob: float = 0.0
    mixup_alpha: float = 0.2


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """UNet model architecture settings."""

    # Encoder backbone: "vgg", "resnet", "vit"
    encoder_name: str = "resnet"
    # Decoder backbone: "vgg", "resnet", "vit"
    decoder_name: str = "resnet"

    # Dimensionality: 2 or 3 (auto-set from data.mode)
    spatial_dims: int = 3

    # Input channels (auto-set from data.mode + num_slices)
    in_channels: int = 1

    # Channel progression for each encoder level
    # e.g. [32, 64, 128, 256, 512] means 5 levels
    encoder_channels: List[int] = field(
        default_factory=lambda: [32, 64, 128, 256, 512]
    )

    # Blocks per encoder level
    encoder_blocks_per_level: List[int] = field(
        default_factory=lambda: [2, 2, 2, 2, 2]
    )

    # Decoder channels (auto-set as reverse of encoder if empty)
    decoder_channels: List[int] = field(default_factory=list)

    # Blocks per decoder level
    decoder_blocks_per_level: List[int] = field(
        default_factory=lambda: [2, 2, 2, 2]
    )

    # Bottleneck channels (0 = same as last encoder channel)
    bottleneck_channels: int = 0

    # Normalization: "batch", "instance", "group"
    norm_type: str = "instance"
    # Group norm groups (only used if norm_type == "group")
    norm_groups: int = 8

    # Activation: "relu", "leakyrelu", "gelu", "swish"
    activation: str = "leakyrelu"

    # Dropout in encoder/decoder blocks
    dropout: float = 0.0

    # Deep supervision: output at multiple decoder levels
    deep_supervision: bool = False

    # ViT-specific settings
    vit_patch_size: int = 16
    vit_num_heads: int = 8
    vit_mlp_ratio: float = 4.0
    vit_qkv_bias: bool = True
    vit_drop_path_rate: float = 0.1

    # Upsampling in decoder: "transpose", "bilinear", "trilinear"
    upsample_mode: str = "transpose"

    # Skip connection mode: "cat" (concatenate) or "add"
    skip_mode: str = "cat"


# ---------------------------------------------------------------------------
# Loss configuration
# ---------------------------------------------------------------------------
@dataclass
class LossConfig:
    """Loss function settings."""

    # Output mode:
    #   "softmax"   — standard multi-class softmax (output includes background)
    #   "per_class" — per-class independent sigmoid (each fg class gets its own
    #                 binary output channel; background is implicit)
    output_mode: str = "softmax"

    # Primary loss: "dice", "ce", "dice_ce", "focal", "tversky", "dice_focal"
    # In per_class mode: "ce" becomes BCE, "focal" uses binary focal, etc.
    name: str = "dice_ce"

    # Weights for compound losses (e.g. dice_ce → [dice_w, ce_w])
    compound_weights: List[float] = field(default_factory=lambda: [1.0, 1.0])

    # Per-class loss weights (empty = uniform)
    class_weights: List[float] = field(default_factory=list)

    # Dice loss settings
    dice_smooth: float = 1e-5
    dice_squared: bool = False  # squared denominators

    # Focal loss settings
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # Tversky loss settings
    tversky_alpha: float = 0.3  # FP weight
    tversky_beta: float = 0.7  # FN weight

    # Label smoothing for CE
    label_smoothing: float = 0.0

    # Region-based weighting: per-pixel weight map from distance transform
    # "none", "border" (upweight boundaries), "distance" (distance transform)
    spatial_weight_mode: str = "none"
    border_weight_sigma: float = 5.0
    border_weight_w0: float = 10.0

    # Deep supervision loss weight decay (geometric)
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
    # SGD-specific
    momentum: float = 0.99
    nesterov: bool = True

    # Scheduler: "cosine", "poly", "step", "plateau", "one_cycle"
    scheduler: str = "cosine"
    # Warmup epochs
    warmup_epochs: int = 5
    warmup_lr: float = 1e-6
    # Cosine min LR
    cosine_min_lr: float = 1e-6
    # Poly power
    poly_power: float = 0.9
    # Step decay
    step_size: int = 50
    step_gamma: float = 0.1
    # ReduceOnPlateau
    plateau_patience: int = 10
    plateau_factor: float = 0.5

    # Gradient clipping
    grad_clip_norm: float = 12.0
    grad_clip_value: float = 0.0  # 0 = disabled

    # Mixed precision (AMP)
    use_amp: bool = True
    amp_dtype: str = "float16"  # "float16" or "bfloat16"

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999

    # Checkpointing
    output_dir: str = "outputs"
    save_every: int = 10  # save checkpoint every N epochs
    save_best_metric: str = "mean_dice"  # metric to track for best model
    save_best_mode: str = "max"  # "max" or "min"

    # Early stopping
    early_stopping: int = 0  # 0 = disabled, N = patience in epochs

    # Logging
    log_every: int = 10  # log every N steps
    val_every: int = 1  # validate every N epochs
    vis_every: int = 10  # save visualization every N epochs

    # Reproducibility
    seed: int = 42
    deterministic: bool = False

    # Resume
    resume: str = ""  # path to checkpoint

    # Multi-GPU (DDP)
    distributed: bool = False


# ---------------------------------------------------------------------------
# Prediction configuration
# ---------------------------------------------------------------------------
@dataclass
class PredictConfig:
    """Inference / prediction settings."""

    # Sliding window settings
    # patch_overlap: fraction of overlap between patches (0-1)
    patch_overlap: float = 0.5
    # Blending mode: "constant" (average), "gaussian" (weighted)
    blend_mode: str = "gaussian"
    # Batch size for inference patches
    batch_size: int = 4

    # Test-Time Augmentation
    tta_enabled: bool = False
    tta_flips: bool = True  # flip augmentation

    # Post-processing
    # Minimum connected component size (0 = disabled)
    min_component_size: int = 0
    # Fill holes
    fill_holes: bool = False
    # Threshold for binarization (for sigmoid output)
    threshold: float = 0.5

    # Output
    save_probabilities: bool = False
    output_dir: str = "predictions"


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Top-level configuration combining all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)

    def sync(self) -> None:
        """Synchronize dependent fields across sub-configs."""
        # Auto-set num_classes from label_values
        if self.data.label_values and self.data.num_classes == 0:
            self.data.num_classes = len(self.data.label_values)

        # Auto-set spatial_dims from data mode
        # Note: factory.py may override these for 2.5D mode
        if self.data.mode == "3d":
            self.model.spatial_dims = 3
        else:
            self.model.spatial_dims = 2

        # Auto-set in_channels from data mode
        if self.data.mode == "2d":
            self.model.in_channels = 1
        elif self.data.mode == "2.5d":
            self.model.in_channels = 2 * self.data.num_slices_per_side + 1
        elif self.data.mode == "3d":
            self.model.in_channels = 1

        # Auto-set decoder channels if not specified
        if not self.model.decoder_channels:
            enc = self.model.encoder_channels
            self.model.decoder_channels = list(reversed(enc[:-1]))

        # Auto-set bottleneck
        if self.model.bottleneck_channels == 0:
            self.model.bottleneck_channels = self.model.encoder_channels[-1]

    def validate(self) -> None:
        """Validate configuration for consistency."""
        assert self.data.mode in ("2d", "2.5d", "3d"), \
            f"Invalid data mode: {self.data.mode}"
        assert self.model.encoder_name in ("vgg", "resnet", "vit"), \
            f"Invalid encoder: {self.model.encoder_name}"
        assert self.model.decoder_name in ("vgg", "resnet", "vit"), \
            f"Invalid decoder: {self.model.decoder_name}"
        assert self.model.norm_type in ("batch", "instance", "group"), \
            f"Invalid norm: {self.model.norm_type}"
        assert self.model.activation in ("relu", "leakyrelu", "gelu", "swish"), \
            f"Invalid activation: {self.model.activation}"
        assert self.loss.output_mode in ("softmax", "per_class"), \
            f"Invalid output_mode: {self.loss.output_mode}"
        assert self.loss.name in (
            "dice", "ce", "dice_ce", "focal", "tversky", "dice_focal",
        ), f"Invalid loss: {self.loss.name}"
        assert self.train.optimizer in ("adam", "adamw", "sgd"), \
            f"Invalid optimizer: {self.train.optimizer}"
        assert self.train.scheduler in (
            "cosine", "poly", "step", "plateau", "one_cycle",
        ), f"Invalid scheduler: {self.train.scheduler}"

        n_enc = len(self.model.encoder_channels)
        assert len(self.model.encoder_blocks_per_level) == n_enc, \
            "encoder_blocks_per_level must match encoder_channels length"

        if self.data.num_classes < 2:
            logger.warning(
                "num_classes=%d < 2. Will be auto-detected from data.",
                self.data.num_classes,
            )


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------
def _dataclass_from_dict(cls, d: Dict[str, Any]):
    """Recursively construct a dataclass from a dict."""
    if not isinstance(d, dict):
        return d
    field_types = {f.name: f.type for f in fields(cls)}
    kwargs = {}
    for k, v in d.items():
        if k in field_types:
            ft = field_types[k]
            # Check if the field type is a dataclass
            for sub_cls in [
                DataConfig, AugmentConfig, ModelConfig,
                LossConfig, TrainConfig, PredictConfig,
            ]:
                if ft == sub_cls.__name__ or (
                    hasattr(ft, '__origin__') is False and
                    isinstance(ft, type) and issubclass(ft, sub_cls)
                ):
                    v = _dataclass_from_dict(sub_cls, v)
                    break
            kwargs[k] = v
        else:
            logger.warning("Unknown config key: %s", k)
    return cls(**kwargs)


def load_config(path: Union[str, Path]) -> Config:
    """Load configuration from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    cfg = _dataclass_from_dict(Config, raw)
    cfg.sync()
    cfg.validate()
    return cfg


def save_config(cfg: Config, path: Union[str, Path]) -> None:
    """Save configuration to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = asdict(cfg)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(d, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    logger.info("Config saved to %s", path)
