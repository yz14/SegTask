"""任务配置 YAML I/O 与点记法 override（五任务共用）。

组合式任务（cls / det / ssl）共用 ``Config`` 核心段 + 顶层任务段
（``cls:`` / ``det:`` / ``ssl:``）；本模块把三者近乎逐字重复的
``load_config`` / ``save_config`` / ``_coerce`` / ``apply_overrides``
收敛为一处。单段配置（seg / gen）也可直接用 :func:`apply_dotted_overrides`
与 :func:`coerce_override_value`。
"""

from __future__ import annotations

import logging
from dataclasses import asdict
import types
from dataclasses import fields, is_dataclass
from typing import get_args, get_origin, get_type_hints
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union

import yaml

from .core import Config, ConfigError, _dataclass_from_dict

logger = logging.getLogger(__name__)

T = TypeVar("T")
PathLike = Union[str, Path]

_BOOL_TRUE = frozenset({"true", "1", "yes"})
_BOOL_FALSE = frozenset({"false", "0", "no"})


def coerce_override_value(old: Any, val: str, declared_type: Any = None) -> Any:
    """按字段声明类型把字符串 override 值转回原类型。

    ``old is None``（Optional 字段默认值）时按 YAML 语义解析，使
    ``--override data.target_spacing=[1,1,1]`` 等可正确写入。
    """
    typ = declared_type
    if typ is None:
        typ = type(old) if old is not None else Any
    parsed = yaml.safe_load(val)
    origin = get_origin(typ)
    args = get_args(typ)
    if origin in (Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        if isinstance(parsed, list):
            list_types = [a for a in non_none if get_origin(a) in (list, List)]
            typ = list_types[0] if list_types else (non_none[0] if non_none else Any)
        else:
            typ = non_none[0] if non_none else Any
        origin, args = get_origin(typ), get_args(typ)
    if typ is Any or typ is object:
        return parsed
    if origin in (list, List):
        if not isinstance(parsed, list):
            raise ConfigError(f"Override value {val!r} must be a list")
        item_type = args[0] if args else Any
        return [_coerce_declared(item_type, item) for item in parsed]
    if typ is bool:
        low = val.lower()
        if low in _BOOL_TRUE:
            return True
        if low in _BOOL_FALSE:
            return False
        raise ValueError(
            f"Invalid bool override {val!r}; use true/false/1/0/yes/no")
    if typ is int:
        return int(parsed)
    if typ is float:
        return float(parsed)
    if typ is str:
        return str(parsed)
    return parsed


def _coerce_declared(typ: Any, value: Any) -> Any:
    if typ in (Any, object):
        return value
    origin, args = get_origin(typ), get_args(typ)
    if origin in (Union, types.UnionType):
        choices = [a for a in args if a is not type(None)]
        return _coerce_declared(choices[0], value) if choices else value
    if origin in (list, List):
        return [_coerce_declared(args[0], v) for v in value]
    if typ is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"Expected bool override value, got {value!r}")
    try:
        return typ(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"Cannot coerce override value {value!r} to {typ}") from exc


def set_dotted_attr(obj: Any, dotted: str, val: str) -> None:
    """沿点路径设置属性，值经 :func:`coerce_override_value` 转型。"""
    parts = dotted.split(".")
    try:
        for p in parts[:-1]:
            obj = getattr(obj, p)
        attr = parts[-1]
        old = getattr(obj, attr)
    except AttributeError as exc:
        raise ConfigError(f"Unknown override path: {dotted!r}") from exc
    try:
        declared = get_type_hints(type(obj)).get(attr)
    except (NameError, TypeError):
        declared = None
    try:
        new = coerce_override_value(old, val, declared)
    except ConfigError:
        raise
    except (TypeError, ValueError, yaml.YAMLError) as exc:
        raise ConfigError(
            f"Invalid override {dotted!r}={val!r}: cannot coerce to "
            f"{declared or type(old).__name__}") from exc
    setattr(obj, attr, new)
    logger.info("Override: %s = %s -> %s", dotted, old, new)


def apply_dotted_overrides(
    root: Any,
    overrides: Sequence[str],
    *,
    sections: "Optional[Mapping[str, Any]]" = None,
) -> None:
    """应用 ``key=value`` 点记法 override。

    * ``sections={"cls": cls_obj}``：``cls.*`` 路由到任务段，其余到 ``root``；
    * ``sections is None``：全部写入 ``root``（seg / gen 单段配置）。

    调用方应在其后自行 ``sync`` / ``validate``（及任务段校验）。
    """
    prefix_map: Dict[str, Any] = {}
    if sections:
        for name, obj in sections.items():
            prefix_map[f"{name}."] = obj

    for ov in overrides:
        if "=" not in ov:
            if ov.strip():
                logger.warning(
                    "Ignoring override without '=': %r "
                    "(expected dotted.path=value)", ov)
            continue
        key, val = ov.split("=", 1)
        routed = False
        for prefix, obj in prefix_map.items():
            if key.startswith(prefix):
                set_dotted_attr(obj, key[len(prefix):], val)
                routed = True
                break
        if not routed:
            set_dotted_attr(root, key, val)


def load_core_and_task_config(
    path: PathLike,
    *,
    section: str,
    task_cls: Type[T],
    validate_task: "Callable[[T, Config], None]",
    core_cls: Type[Config] = Config,
    skip_core_validators: Sequence[str] = (),
    preprocess_raw: Optional[Callable[[dict], dict]] = None,
) -> Tuple[Config, T]:
    """加载「核心 Config + 顶层任务段」YAML，返回 ``(cfg, task_cfg)``。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if preprocess_raw is not None:
        processed = preprocess_raw(raw)
        if processed is not None:
            raw = processed
    task_raw = dict(raw.pop(section, {}) or {})
    cfg = _dataclass_from_dict(core_cls, raw)
    cfg.sync()
    cfg.validate(skip=set(skip_core_validators))
    task_cfg = _dataclass_from_dict(task_cls, task_raw)
    validate_task(task_cfg, cfg)
    return cfg, task_cfg


def save_core_and_task_config(
    cfg: Config,
    task_cfg: Any,
    path: PathLike,
    *,
    section: str,
) -> None:
    """把 ``(cfg, task_cfg)`` 落盘为单个 YAML（任务段覆盖同名残留键）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = asdict(cfg)
    blob[section] = asdict(task_cfg)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(blob, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)


__all__ = [
    "coerce_override_value",
    "set_dotted_attr",
    "apply_dotted_overrides",
    "load_core_and_task_config",
    "save_core_and_task_config",
]
