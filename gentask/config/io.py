"""YAML load/save helpers for gentask.config."""

from __future__ import annotations

import logging
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, Union

import yaml

from taskcore.config.core import MonitorConfig, nested_dataclass_type
from taskcore.config.model_migration import (
    SISR_FIELD_MAP,
    route_legacy_model_dict,
)

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
    "monitor": MonitorConfig,
}


# 旧 YAML 字段名 → 新字段名；读到旧名直接拒绝并给出迁移提示（与 taskcore.config.core 一致）。
# 命名清晰化（TODO #4）：
#   data.aux_keep_native_d  → data.keep_native_view_depth（'aux' 误导：含主视图）
#   model.context_fusion    → model.stem_fusion_mode（与 num_stem_fusion_views 配对）
_FIELD_ALIASES: Dict[type, Dict[str, str]] = {
    DataConfig:  {"aux_keep_native_d": "keep_native_view_depth"},
    ModelConfig: {"context_fusion": "stem_fusion_mode"},
}


# 旧 YAML 中曾可手设、现已改为派生只读量的字段：读到时直接拒绝并给出派生来源。
#   model.in_channels / spatial_dims        → 由 patch_mode/multi_res_scales 等派生
#                                             （sync() 经 build_topology 算出）。
_DEPRECATED_DERIVED_KEYS: Dict[type, Dict[str, str]] = {
    ModelConfig: {
        "in_channels":  "data.patch_mode / data.multi_res_scales",
        "spatial_dims": "data.patch_mode",
    },
}


# 旧 YAML 中已移除、但需要更具体迁移提示的字段（与 taskcore.config.core 同源）。
#   model.use_se → 改用 model.attention_type: "se"。
_REMOVED_KEYS: Dict[type, Dict[str, str]] = {
    ModelConfig: {"use_se": 'attention_type: "se"'},
}


def _dataclass_from_dict(cls, d: Dict[str, Any]):
    """Recursively construct a dataclass from a dict.

    旧别名（``_FIELD_ALIASES``）、曾经可写但现已派生的字段
    （``_DEPRECATED_DERIVED_KEYS``）以及未知字段都直接抛
    ``ConfigError``，并给出迁移提示（与 ``taskcore.config.core`` 一致）。
    """
    if not isinstance(d, dict):
        return d
    if isinstance(cls, type) and issubclass(cls, ModelConfig):
        # D2 兼容层：旧扁平 model 键路由进嵌套子段（与 taskcore 同口径）。
        d, moved = route_legacy_model_dict(
            d, error_cls=ConfigError, extra_flat_to_nested=SISR_FIELD_MAP)
        if moved:
            logger.info(
                "model 段旧扁平键已自动迁移到嵌套路径（建议更新 YAML）：%s",
                ", ".join(f"{k} -> {p}" for k, p in sorted(moved.items())))
    dc_fields = {f.name: f for f in fields(cls)}
    aliases = _FIELD_ALIASES.get(cls, {})
    derived = _DEPRECATED_DERIVED_KEYS.get(cls, {})
    removed = _REMOVED_KEYS.get(cls, {})
    kwargs = {}
    for k, v in d.items():
        if k in removed:
            raise ConfigError(
                f"Config key '{k}' is removed from {cls.__name__}; use "
                f"{removed[k]} instead.")
        if k in derived:
            raise ConfigError(
                f"Config key '{k}' is removed from {cls.__name__}; it is now "
                f"auto-derived from '{derived[k]}' and must not be set in YAML.")
        if k in aliases:
            new_key = aliases[k]
            if new_key in d:
                raise ConfigError(
                    f"{cls.__name__}: both deprecated '{k}' and its "
                    f"replacement '{new_key}' are set; remove '{k}' and keep "
                    f"'{new_key}'.")
            raise ConfigError(
                f"Config key '{k}' is removed from {cls.__name__}; use "
                f"'{new_key}' instead.")
        if k not in dc_fields:
            raise ConfigError(
                f"Unknown config key '{k}' in {cls.__name__}.")
        sub_cls = nested_dataclass_type(dc_fields[k])
        if sub_cls is None and k in _SUB_CONFIGS:
            sub_cls = _SUB_CONFIGS[k]
        if sub_cls is not None and isinstance(v, dict):
            v = _dataclass_from_dict(sub_cls, v)
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
