"""把训练历史渲染成自包含 HTML 仪表盘（单 run）/ 对比页（多 run）。

对外仅暴露三个函数：

* :func:`render_dashboard` —— 单 run 历史 → HTML 字符串；
* :func:`render_comparison` —— 多 run 历史 → 对比 HTML 字符串；
* :func:`write_dashboard` —— 渲染单 run 并原子写盘（训练期实时刷新用）。

渲染分两层：Python 端（``charts.py``）把历史整理成「渲染就绪」的 payload，
内嵌为 JSON；浏览器端（``assets.py`` 的通用 JS）据此绘制 SVG。整页零外部
依赖、可离线打开。
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Optional, Sequence, Union

from . import assets, charts
from .history import MetricsHistory


def _page(payload: dict, page_title: str) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False)
    # 防止 payload 里出现的 "</script>" 提前闭合脚本块。
    payload_json = payload_json.replace("</", "<\\/")
    js = assets.JS.replace("__PAYLOAD__", payload_json)
    esc_title = html.escape(page_title)
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc_title}</title>
<style>{assets.CSS}</style>
</head>
<body>
<header>
  <h1 id="title">{esc_title}</h1>
  <div class="meta" id="meta"></div>
</header>
<div class="wrap" id="wrap"></div>
<script>{js}</script>
</body>
</html>
"""


def render_dashboard(
    history: MetricsHistory,
    *,
    auto_reload_seconds: int = 0,
    run_name: Optional[str] = None,
) -> str:
    """单 run 历史 → 自包含 HTML 字符串。

    ``auto_reload_seconds > 0`` 时页面会定时自重载（训练中实时刷新用）。
    """
    payload = charts.build_single_payload(
        history, auto_reload_seconds=auto_reload_seconds, run_name=run_name)
    title = payload.get("title") or "Training Monitor"
    return _page(payload, f"训练监测 · {title}")


def render_comparison(
    histories: Sequence[MetricsHistory],
    *,
    run_names: Optional[Sequence[str]] = None,
    auto_reload_seconds: int = 0,
) -> str:
    """多 run 历史 → 对比 HTML 字符串（同名指标叠加 + best 对照表）。"""
    payload = charts.build_compare_payload(
        histories, run_names=run_names, auto_reload_seconds=auto_reload_seconds)
    return _page(payload, "训练监测 · 多 run 对比")


def _atomic_write_html(out_path: Union[str, Path], htmlstr: str) -> Path:
    """临时文件 + ``os.replace`` 原子写盘，避免实时刷新与浏览器读取竞争产生半截
    文件。"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(htmlstr)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out)
    return out


def write_dashboard(
    history: MetricsHistory,
    out_path: Union[str, Path],
    *,
    auto_reload_seconds: int = 0,
    run_name: Optional[str] = None,
) -> Path:
    """渲染单 run 仪表盘并**原子写盘**，返回写入路径。"""
    htmlstr = render_dashboard(
        history, auto_reload_seconds=auto_reload_seconds, run_name=run_name)
    return _atomic_write_html(out_path, htmlstr)


def write_comparison(
    histories: Sequence[MetricsHistory],
    out_path: Union[str, Path],
    *,
    run_names: Optional[Sequence[str]] = None,
    auto_reload_seconds: int = 0,
) -> Path:
    """渲染多 run 对比页并**原子写盘**，返回写入路径。"""
    htmlstr = render_comparison(
        histories, run_names=run_names, auto_reload_seconds=auto_reload_seconds)
    return _atomic_write_html(out_path, htmlstr)


__all__ = [
    "render_dashboard", "render_comparison",
    "write_dashboard", "write_comparison",
]
