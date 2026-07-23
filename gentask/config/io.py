"""YAML load/save helpers for gentask.config."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Union

import yaml

from taskcore.config.core import (
    DataclassLoadContext,
    MonitorConfig,
    dataclass_from_dict,
)
from taskcore.config.model_migration import SISR_FIELD_MAP

from .dataclasses import (
    AugConfig,
    DataConfig,
    LossConfig,
    ModelConfig,
    PredictConfig,
    TaskConfig,
    TrainConfig,
)
from .validation import Config

logger = logging.getLogger(__name__)

# gen 顶层段名 → 类型（``nested_dataclass_type`` 解析失败时的兜底）。
_GEN_SUB_CONFIGS = {
    "data": DataConfig,
    "model": ModelConfig,
    "loss": LossConfig,
    "train": TrainConfig,
    "predict": PredictConfig,
    "task": TaskConfig,
    "augment": AugConfig,
    "monitor": MonitorConfig,
}

_GEN_LOAD_CTX = DataclassLoadContext(
    sub_configs=_GEN_SUB_CONFIGS,
    model_route_extra_flat_to_nested=SISR_FIELD_MAP,
    model_config_cls=ModelConfig,
)


def _dataclass_from_dict(cls, d: Dict[str, Any]):
    """与 ``taskcore.config.core.dataclass_from_dict`` 同契约；保留供测试导入。"""
    return dataclass_from_dict(cls, d, ctx=_GEN_LOAD_CTX)


def load_config(path: Union[str, Path]) -> Config:
    """Load configuration from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = dataclass_from_dict(Config, raw, ctx=_GEN_LOAD_CTX)
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
