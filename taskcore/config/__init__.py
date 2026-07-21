"""taskcore.config —— 五任务共用配置层（核心段：data/aug/model/loss/train/predict/vis/monitor）。"""

from .core import *  # noqa: F401,F403
from . import core as _core
from .task_io import (  # noqa: F401
    apply_dotted_overrides,
    coerce_override_value,
    load_core_and_task_config,
    save_core_and_task_config,
    set_dotted_attr,
)
from .registry import (  # noqa: F401
    TaskSectionSpec,
    apply_task_overrides,
    clear_task_registry,
    get_task_section,
    load_task_config,
    register_task_section,
    registered_task_sections,
    save_task_config,
    validate_core_config,
)

__all__ = list(getattr(_core, "__all__", [])) or [
    n for n in dir(_core) if not n.startswith("_")]
__all__ = list(__all__) + [
    "apply_dotted_overrides",
    "coerce_override_value",
    "load_core_and_task_config",
    "save_core_and_task_config",
    "set_dotted_attr",
    "TaskSectionSpec",
    "apply_task_overrides",
    "clear_task_registry",
    "get_task_section",
    "load_task_config",
    "register_task_section",
    "registered_task_sections",
    "validate_core_config",
    "save_task_config",
]
