"""把 ``schema``（字段元信息）与 ``manifest``（per-mode 有效参数）拼装成前端载荷，
并实现表单值 → 运行 YAML 的构建与校验（全程复用 ``segtask_v1.config``）。

对外函数：

* ``build_payload(mode)``   —— 渲染表单所需的完整 JSON（分组、控件、tooltip、
                              enum、depends_on、默认值、predict CLI 运行参数）。
* ``build_config(values)``  —— 由表单值字典构造 ``Config``（含 ``sync()``）。
* ``validate_values(...)``  —— 复用 ``Config.validate()`` 给出友好错误。
* ``values_to_yaml(...)``   —— 生成运行 YAML 文本（不落盘，供预览）。
* ``write_run_yaml(...)``   —— 落盘到 ``configs/_runs/<时间戳>_<task>.yaml``。
* ``list_base_configs()``   —— 列出 ``configs/*.yaml`` 供"载入模板"。
* ``load_base_values(path)``—— 读模板 YAML → 扁平 {section:{field:val}} 值。
"""

from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .. import config as _config
from . import manifest as _manifest
from .schema import build_field_schema

# 仓库根目录（…/repo-sldjiesl94）。launcher 在 segtask_v1/launcher 下。
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
RUN_DIR = CONFIG_DIR / "_runs"


# ---------------------------------------------------------------------------
# 前端载荷
# ---------------------------------------------------------------------------
def _enum_for(ref: str, fld: "_manifest.Field") -> Optional[List[str]]:
    """字段显式 enum 优先，否则回退 manifest.ENUMS（None 表示无枚举）。"""
    if fld.enum is not None:
        return fld.enum
    return _manifest.ENUMS.get(ref)


def build_payload(mode: str) -> Dict[str, Any]:
    """构造某模式（'2_5d' / '3d'）的完整前端渲染载荷。"""
    if mode not in ("2_5d", "3d"):
        raise ValueError(f"unknown mode: {mode!r}")
    schema = build_field_schema()
    groups_out: List[Dict[str, Any]] = []
    for g in _manifest.build_groups(mode):
        fields_out: List[Dict[str, Any]] = []
        for fld in g.fields:
            section, _, field_name = fld.ref.partition(".")
            meta = schema.get(section, {}).get(field_name)
            if meta is None:
                continue
            enum = _enum_for(fld.ref, fld)
            control = "enum" if enum else meta["control"]
            fields_out.append({
                "ref": fld.ref,
                "section": section,
                "field": field_name,
                "label": fld.label or field_name,
                "control": control,
                "default": meta["default"],
                "tooltip": meta["tooltip"],
                "enum": enum,
                "depends_on": fld.depends_on,
                "readonly": fld.readonly,
            })
        groups_out.append({
            "title": g.title,
            "section_tag": g.section_tag,
            "fields": fields_out,
        })

    # predict CLI-only 运行参数。
    run_args_out: List[Dict[str, Any]] = []
    for ra in _manifest.predict_run_args():
        run_args_out.append({
            "name": ra.name,
            "flag": ra.flag,
            "control": ra.control,
            "default": ra.default,
            "tooltip": ra.tooltip,
            "enum": _manifest.RUN_ARG_ENUMS.get(ra.name),
            "required": ra.required,
        })

    return {
        "mode": mode,
        "mode_label": "2.5D" if mode == "2_5d" else "3D",
        "patch_mode_default": "2_5d" if mode == "2_5d" else "z_axis",
        "groups": groups_out,
        "task_sections": _manifest.TASK_SECTIONS,
        "predict_run_args": run_args_out,
        "all_defaults": _all_defaults(schema),
    }


def _all_defaults(schema: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """{section: {field: default}}，供前端初始化完整值字典。"""
    out: Dict[str, Dict[str, Any]] = {}
    for section, fields_meta in schema.items():
        out[section] = {fn: m["default"] for fn, m in fields_meta.items()}
    return out


# ---------------------------------------------------------------------------
# 表单值 → Config / YAML
# ---------------------------------------------------------------------------
def build_config(values: Dict[str, Dict[str, Any]]) -> "_config.Config":
    """由 {section: {field: value}} 构造 Config（复用 config 的 from_dict + sync）。

    缺省 section/field 自动使用 dataclass 默认值；派生只读键被 from_dict 忽略。
    """
    cfg = _config._dataclass_from_dict(_config.Config, values)
    cfg.sync()
    return cfg


def validate_values(
    values: Dict[str, Dict[str, Any]],
) -> Tuple[bool, str]:
    """返回 (ok, message)。复用 Config.validate()，捕获 ConfigError 为友好文本。"""
    try:
        cfg = build_config(values)
        cfg.validate()
    except _config.ConfigError as e:
        return False, str(e)
    except Exception as e:  # noqa: BLE001  其它构造期异常也回传，避免 500。
        return False, f"{type(e).__name__}: {e}"
    return True, "配置校验通过。"


def values_to_yaml(values: Dict[str, Dict[str, Any]]) -> str:
    """生成运行 YAML 文本（经 sync()，含派生量回写后的快照）。"""
    cfg = build_config(values)
    buf = io.StringIO()
    yaml.dump(_dc.asdict(cfg), buf, default_flow_style=False,
              sort_keys=False, allow_unicode=True)
    return buf.getvalue()


def predict_cli_args(run_values: Dict[str, Any]) -> List[str]:
    """把 run 组的值映射为 predict.py 的命令行参数列表（不含 --config）。

    bool flag 仅在真值时加入；str/enum 留空则不传（由 predict.py 取默认/回退）。
    ``bbox`` 留空即回退 ``cfg.data.bbox_dir``；如需对本次运行禁用 bbox，请在基础
    配置里清空 ``data.bbox_dir``。
    """
    args: List[str] = []
    for ra in _manifest.predict_run_args():
        val = run_values.get(ra.name, ra.default)
        if ra.control == "bool":
            if bool(val):
                args.append(ra.flag)
            continue
        if val is None or str(val) == "":
            continue
        args += [ra.flag, str(val)]
    return args


def write_run_yaml(values: Dict[str, Dict[str, Any]], task: str) -> Path:
    """落盘运行 YAML 到 configs/_runs/<时间戳>_<task>.yaml，返回路径。"""
    cfg = build_config(values)
    cfg.validate()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RUN_DIR / f"{ts}_{task}.yaml"
    _config.save_config(cfg, path)
    return path


# ---------------------------------------------------------------------------
# 模板（已有 configs/*.yaml）载入
# ---------------------------------------------------------------------------
_MODE_PATCH_MODES = {
    "2_5d": {"2_5d"},
    "3d":   {"z_axis", "cubic", "whole"},
}


def list_base_configs(mode: Optional[str] = None) -> List[str]:
    """列出 configs/ 下与分割任务相关、且 patch_mode 匹配 ``mode`` 的模板。

    过滤规则（轻量读 raw YAML，不全量构造 Config）：
      * 排除含顶层 ``task`` 键的（gentask 超分配置，超出本启动器范围）；
      * 排除含顶层 ``ssl`` 键的（自监督预训练配置）；
      * 给定 ``mode`` 时仅保留 ``data.patch_mode`` 落在该模式集合内的。
    返回相对 REPO_ROOT 的 posix 路径列表。
    """
    if not CONFIG_DIR.is_dir():
        return []
    want = _MODE_PATCH_MODES.get(mode) if mode else None
    out: List[str] = []
    for p in sorted(CONFIG_DIR.glob("*.yaml")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(raw, dict):
            continue
        if "task" in raw or "ssl" in raw:
            continue
        if want is not None:
            pm = (raw.get("data") or {}).get("patch_mode", "z_axis")
            if pm not in want:
                continue
        out.append(str(p.relative_to(REPO_ROOT)).replace("\\", "/"))
    return out


def load_base_values(rel_path: str) -> Dict[str, Dict[str, Any]]:
    """读取模板 YAML，经 Config 规整后回传 {section: {field: value}} 全量值。

    经 load→sync 确保派生量与默认补齐，便于前端按 manifest 取所需字段。
    model 段展平回旧扁平字段名（与表单 schema 口径一致，见 launcher.schema）。
    """
    from taskcore.config.model_migration import flatten_model_dict

    path = (REPO_ROOT / rel_path).resolve()
    # 安全：限制只能读 configs/ 内文件。
    if CONFIG_DIR.resolve() not in path.parents:
        raise ValueError("base config must live under configs/.")
    cfg = _config.load_config(path)
    blob = _dc.asdict(cfg)
    blob["model"] = flatten_model_dict(blob["model"])
    return blob
