"""从 ``segtask_v1/config.py`` 自动抽取每个配置字段的元信息（单一真相源）。

抽取内容（用于前端表单渲染与 ``?`` 悬浮说明）：

* ``name``     —— 字段名；
* ``default``  —— 运行期真实默认值（取 ``dataclasses.fields`` 的 default /
                  default_factory，避免与源码字面量脱节）；
* ``py_type``  —— Python 类型名（``bool`` / ``int`` / ``float`` / ``str`` /
                  ``list`` / ``NoneType``）；
* ``control``  —— 前端控件类型（bool→开关、int/float→数字、list→JSON 文本、
                  str→文本，若 manifest 提供 enum 则升级为下拉）；
* ``tooltip``  —— 从源码注释（字段上方连续 ``#`` 行 + 行尾 ``#`` 注释）提取的
                  中文说明，是 ``?`` 悬浮气泡的内容。

只读"几何派生量"（``model.in_channels`` / ``model.spatial_dims`` 等）是
``@property`` 而非 dataclass field，``dataclasses.fields`` 天然不会列出它们，
因此不会出现在表单里——无需额外排除。
"""

from __future__ import annotations

import ast
import dataclasses as _dc
import inspect
from typing import Any, Dict, List, Optional

from .. import config as _config

# section 名 → dataclass。复用 config 的单一真相源映射。
_SUB_CONFIGS = _config._SUB_CONFIGS


def _field_default(f: _dc.Field) -> Any:
    """取字段的真实默认值（普通 default 或调用 default_factory）。"""
    if f.default is not _dc.MISSING:
        return f.default
    if f.default_factory is not _dc.MISSING:  # type: ignore[misc]
        return f.default_factory()
    return None


def _control_for(default: Any, py_type: str) -> str:
    """由默认值的 Python 类型推断前端控件类型（manifest 可再升级为 enum）。"""
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "float"
    if isinstance(default, list):
        return "list"
    if default is None:
        # 目前仅 predict.tta_batch_size 为 Optional[int]，按可空数字处理。
        return "int"
    return "str"


def _extract_comments(cls: type) -> Dict[str, str]:
    """解析 dataclass 源码，得到 {field_name: tooltip}。

    tooltip = 字段定义行**上方**连续的 ``#`` 注释块（跳过纯分隔线）
              + 该行**行尾**的 ``#`` 注释，按阅读顺序拼接。
    """
    try:
        src = inspect.getsource(cls)
    except (OSError, TypeError):
        return {}
    lines = src.splitlines()

    # 字段名 → 在 src 内的（0 基）行号，用 AST 精确定位 AnnAssign。
    field_line: Dict[str, int] = {}
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {}
    class_node = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)), None)
    if class_node is None:
        return {}
    for node in class_node.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            field_line.setdefault(node.target.id, node.lineno - 1)

    def _clean(comment: str) -> str:
        # 去掉前缀 '#' 与两端的分隔用 '-'/空白。
        c = comment.lstrip("#").strip()
        c = c.strip("-").strip()
        return c

    out: Dict[str, str] = {}
    for name, ln in field_line.items():
        parts: List[str] = []
        # 上方连续注释块。
        i = ln - 1
        block: List[str] = []
        while i >= 0:
            stripped = lines[i].strip()
            if stripped.startswith("#"):
                block.append(stripped)
                i -= 1
                continue
            break
        block.reverse()
        for b in block:
            txt = _clean(b)
            if txt:
                parts.append(txt)
        # 行尾注释。
        cur = lines[ln]
        hash_pos = _find_inline_hash(cur)
        if hash_pos != -1:
            txt = _clean(cur[hash_pos:])
            if txt:
                parts.append(txt)
        tip = " ".join(parts).strip()
        if tip:
            out[name] = tip
    return out


def _find_inline_hash(line: str) -> int:
    """返回行尾注释 ``#`` 的位置（忽略字符串内的 #）；无则 -1。"""
    in_str: Optional[str] = None
    esc = False
    for idx, ch in enumerate(line):
        if in_str is not None:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == in_str:
                in_str = None
            continue
        if ch in ("'", '"'):
            in_str = ch
        elif ch == "#":
            return idx
    return -1


def build_field_schema() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """返回 {section: {field: {name, default, py_type, control, tooltip}}}。

    覆盖 config 的全部 section/field；per-mode 过滤与 enum 由 manifest 负责。
    """
    schema: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for section, cls in _SUB_CONFIGS.items():
        comments = _extract_comments(cls)
        fields_meta: Dict[str, Dict[str, Any]] = {}
        for f in _dc.fields(cls):
            default = _field_default(f)
            py_type = type(default).__name__
            fields_meta[f.name] = {
                "name": f.name,
                "default": default,
                "py_type": py_type,
                "control": _control_for(default, py_type),
                "tooltip": comments.get(f.name, ""),
            }
        schema[section] = fields_meta
    return schema
