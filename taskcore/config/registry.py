"""任务段注册表（P2a）：组合式任务（cls / det / ssl）统一装配入口。

各任务在 ``config.py`` 模块 import 时注册 :class:`TaskSectionSpec`；
``load_task_config`` / ``save_task_config`` / ``apply_task_overrides``
按段名查表，避免各任务重复样板 I/O。

完整 P2a 后续：loss/predict 下沉 seg 任务段、gen 消 fork——本模块先
收敛「核心 Config + 顶层任务段」的注册与 I/O 契约。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence, Tuple, Type, TypeVar

from .core import Config
from .task_io import (
    apply_dotted_overrides,
    load_core_and_task_config,
    save_core_and_task_config,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
ValidateFn = Callable[[Any, Config], None]

# 组合式任务 core 不再含 seg 专属 loss/predict 段（P2a 已下沉 seg 任务段）。
_COMPOSITE_SKIP_CORE: Tuple[str, ...] = ()

_REGISTRY: Dict[str, "TaskSectionSpec"] = {}


@dataclass(frozen=True)
class TaskSectionSpec:
    """单个任务的 YAML 顶层段规格。"""

    name: str
    task_cls: Type[Any]
    validate_task: ValidateFn
    core_cls: Type[Config] = Config
    skip_core_validators: Tuple[str, ...] = _COMPOSITE_SKIP_CORE


def register_task_section(spec: TaskSectionSpec) -> None:
    """注册任务段；同名重复注册 fail-fast。"""
    if spec.name in _REGISTRY:
        raise ValueError(
            f"task section {spec.name!r} already registered "
            f"({_REGISTRY[spec.name].task_cls.__name__})")
    _REGISTRY[spec.name] = spec
    logger.debug("Registered task section %r -> %s", spec.name, spec.task_cls.__name__)


def get_task_section(name: str) -> TaskSectionSpec:
    """按段名取规格；未注册时给出已知段列表。"""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(
            f"unregistered task section {name!r}; known: {known}") from exc


def registered_task_sections() -> Tuple[str, ...]:
    """已注册段名（排序）。"""
    return tuple(sorted(_REGISTRY))


def load_task_config(path, section: str) -> Tuple[Config, Any]:
    """加载 ``(core_cfg, task_cfg)``。"""
    spec = get_task_section(section)
    return load_core_and_task_config(
        path,
        section=spec.name,
        task_cls=spec.task_cls,
        validate_task=spec.validate_task,
        core_cls=spec.core_cls,
        skip_core_validators=spec.skip_core_validators,
    )


def validate_core_config(cfg: Config, section: str) -> None:
    """按注册表规格校验 core Config（override 后重验用）。"""
    spec = get_task_section(section)
    cfg.validate(skip=set(spec.skip_core_validators))


def save_task_config(cfg: Config, task_cfg: Any, path, section: str) -> None:
    """落盘 ``(core_cfg, task_cfg)``。"""
    spec = get_task_section(section)
    save_core_and_task_config(cfg, task_cfg, path, section=spec.name)


def apply_task_overrides(
    cfg: Config,
    task_cfg: Any,
    overrides: Sequence[str],
    section: str,
) -> None:
    """点记法 override；``{section}.*`` 路由到任务段，其余到 core。"""
    spec = get_task_section(section)
    apply_dotted_overrides(cfg, overrides, sections={spec.name: task_cfg})


def clear_task_registry() -> None:
    """测试专用：清空注册表。"""
    _REGISTRY.clear()


__all__ = [
    "TaskSectionSpec",
    "register_task_section",
    "get_task_section",
    "registered_task_sections",
    "load_task_config",
    "save_task_config",
    "apply_task_overrides",
    "validate_core_config",
    "clear_task_registry",
]
