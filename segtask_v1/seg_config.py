"""segtask 配置：core Config + seg 任务段（P2a loss/predict 下沉）。

``load_config`` 返回 :class:`~taskcore.config.seg_bundle.SegBundle`，对外仍可用
``cfg.loss`` / ``cfg.predict`` / ``cfg.data`` 等同址访问。
兼容旧式 YAML（顶层 ``loss``/``predict``）与新式 ``seg:`` 段。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Union

import yaml

from taskcore.config.core import Config, ConfigError
from taskcore.config.registry import (
    TaskSectionSpec,
    apply_task_overrides,
    load_task_config,
    register_task_section,
    save_task_config as save_task_config_registry,
    validate_core_config,
)
from taskcore.config.seg_bundle import SegBundle, merge_seg_bundle
from taskcore.config.seg_task import (
    SegTaskConfig,
    hoist_legacy_seg_sections,
    validate_seg_task,
)

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def _load_raw(path: PathLike) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    hoist_legacy_seg_sections(raw)
    return raw


def load_config(path: PathLike) -> SegBundle:
    """加载分割配置，返回统一 ``cfg`` 视图（SegBundle）。"""
    raw = _load_raw(path)
    core_raw = dict(raw)
    seg_raw = dict(core_raw.pop("seg", {}) or {})
    from taskcore.config.core import dataclass_from_dict
    core = dataclass_from_dict(Config, core_raw)
    seg = dataclass_from_dict(SegTaskConfig, seg_raw)
    bundle = merge_seg_bundle(core, seg)
    bundle.sync()
    bundle.validate()
    return bundle


def load_config_parts(path: PathLike) -> Tuple[Config, SegTaskConfig]:
    """加载并返回 ``(core, seg)`` 二元组（registry 契约 / 测试用）。"""
    raw = _load_raw(path)
    path = Path(path)
    # 写临时逻辑：复用 registry 需文件；直接内存加载
    core_raw = dict(raw)
    seg_raw = dict(core_raw.pop("seg", {}) or {})
    from taskcore.config.core import dataclass_from_dict
    core = dataclass_from_dict(Config, core_raw)
    seg = dataclass_from_dict(SegTaskConfig, seg_raw)
    core.sync()
    core.validate()
    validate_seg_task(seg, core)
    return core, seg


def save_config(cfg: SegBundle, path: PathLike) -> None:
    """落盘为 YAML（``seg:`` 段包含 loss/predict）。"""
    save_task_config_registry(cfg.core, cfg.seg, path, section="seg")


def apply_overrides(cfg: SegBundle, overrides) -> None:
    """点记法 override。

    * ``seg.*`` 路由到任务段；
    * 旧式顶层 ``loss.*`` / ``predict.*`` 自动改写为 ``seg.loss.*`` /
      ``seg.predict.*``（与 YAML hoist 对称），避免写到已下沉的 core Config。
    """
    rewritten = []
    for ov in overrides or []:
        if "=" not in ov:
            rewritten.append(ov)
            continue
        key, val = ov.split("=", 1)
        if key == "loss" or key.startswith("loss."):
            key = f"seg.{key}"
        elif key == "predict" or key.startswith("predict."):
            key = f"seg.{key}"
        rewritten.append(f"{key}={val}")
    apply_task_overrides(cfg.core, cfg.seg, rewritten, section="seg")


def validate_core(cfg: Config) -> None:
    validate_core_config(cfg, "seg")


register_task_section(TaskSectionSpec(
    name="seg",
    task_cls=SegTaskConfig,
    validate_task=validate_seg_task,
    skip_core_validators=(),
    preprocess_raw=hoist_legacy_seg_sections,
))


__all__ = [
    "Config",
    "ConfigError",
    "SegTaskConfig",
    "SegBundle",
    "load_config",
    "load_config_parts",
    "save_config",
    "apply_overrides",
    "validate_core",
]
