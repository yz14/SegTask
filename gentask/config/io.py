"""YAML load/save helpers for gentask.config."""

from __future__ import annotations

import logging
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, Union

import yaml

from .dataclasses import (
    AugConfig, ConfigError, DataConfig, LossConfig, ModelConfig, PredictConfig,
    TaskConfig, TrainConfig, _require,
)
from .validation import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------
_SUB_CONFIGS = {
    "data": DataConfig,
    "model": ModelConfig,
    "loss": LossConfig,
    "train": TrainConfig,
    "predict": PredictConfig,
    "task": TaskConfig,
    "augment": AugConfig,
}


# 旧 YAML 字段名 → 新字段名的向后兼容别名。读到旧名时自动改写并提示一次。
# 命名清晰化（TODO #4）：
#   data.aux_keep_native_d  → data.keep_native_view_depth（'aux' 误导：含主视图）
#   model.context_fusion    → model.stem_fusion_mode（与 num_stem_fusion_views 配对）
_FIELD_ALIASES: Dict[type, Dict[str, str]] = {
    DataConfig:  {"aux_keep_native_d": "keep_native_view_depth"},
    ModelConfig: {"context_fusion": "stem_fusion_mode"},
}


# 旧 YAML 中曾可手设、现已改为派生只读量的字段：读到时静默忽略（仅一次 info 提示），
# 而非按 "Unknown config key" 处理。TODO #4：派生量不再暴露可写接口。
# 旧 YAML 中曾可手设、现已改为派生只读量的字段：读到时静默忽略（仅一次 info 提示），
# 而非按 "Unknown config key" 处理。TODO #4：派生量不再暴露可写接口。
#   model.in_channels / spatial_dims        → 由 patch_mode/multi_res_scales 等派生
#                                             （sync() 经 build_topology 算出）。
_DEPRECATED_DERIVED_KEYS: Dict[type, Dict[str, str]] = {
    ModelConfig: {
        "in_channels":  "data.patch_mode / data.multi_res_scales",
        "spatial_dims": "data.patch_mode",
    },
}


def _dataclass_from_dict(cls, d: Dict[str, Any]):
    """Recursively construct a dataclass from a dict.

    支持向后兼容别名（``_FIELD_ALIASES``）：旧 YAML 字段名会被自动改写成新名，
    并打印一次弃用提示；若新旧名同时出现则报错。``_DEPRECATED_DERIVED_KEYS``
    列出的"曾可写、现派生只读"字段则直接忽略。
    """
    if not isinstance(d, dict):
        return d
    field_names = {f.name for f in fields(cls)}
    aliases = _FIELD_ALIASES.get(cls, {})
    derived = _DEPRECATED_DERIVED_KEYS.get(cls, {})
    kwargs = {}
    for k, v in d.items():
        if k in derived:
            logger.info(
                "Config key '%s' is now auto-derived from '%s' and no longer "
                "settable; ignoring the value in YAML.", k, derived[k])
            continue
        if k in aliases:
            new_key = aliases[k]
            if new_key in d:
                raise ValueError(
                    f"{cls.__name__}: both deprecated '{k}' and its "
                    f"replacement '{new_key}' are set; remove the deprecated one.")
            logger.warning(
                "Config key '%s' is deprecated; use '%s' instead "
                "(auto-remapped for backward compatibility).", k, new_key)
            k = new_key
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
