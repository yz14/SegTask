"""组装单页 HTML（每个模式一个页面），内联 ``assets.CSS`` / ``assets.JS``。

页面骨架：左侧 = 任务切换 + 基础配置载入 + 分组参数表单；右侧 = YAML 预览 /
实时日志切换面板 + 操作按钮 + 状态栏。模式（``2_5d`` / ``3d``）通过注入的
``window.__MODE__`` 传给 JS，其余渲染数据由 ``/api/payload`` 拉取。

注：CSS/JS 内含大量 ``{}``，故用占位符 ``replace`` 而非 ``str.format`` 注入。
"""

from __future__ import annotations

from .assets import CSS, JS

_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Segtask 启动器 · @@MODE_LABEL@@</title>
<style>@@CSS@@</style>
</head>
<body>
<header>
  <h1>Segtask 启动器</h1>
  <span class="badge">@@MODE_LABEL@@</span>
  <span class="modes">
    <a href="/2_5d" class="@@A25@@">2.5D</a>
    <a href="/3d" class="@@A3D@@">3D</a>
  </span>
  <span class="seg" style="margin-left:8px">
    <button id="taskTrain" class="on">训练 Train</button>
    <button id="taskPredict">推理 Predict</button>
  </span>
  <span class="muted">本地服务 · 仅本机 127.0.0.1</span>
</header>
<div class="wrap">
  <div class="left">
    <div class="baserow" id="baseRow">
      <span class="muted">基础配置：</span>
      <select id="baseSel" style="max-width:360px"></select>
      <button id="applyBase">载入</button>
      <span class="muted">推理需先载入与训练一致的配置（模型/几何必须匹配权重）。</span>
    </div>
    <div class="toolbar">
      <button id="validateBtn">校验</button>
      <button id="previewBtn">预览 YAML</button>
      <button id="launchBtn" class="primary">开始训练</button>
      <button id="stopBtn" class="danger" disabled>停止</button>
    </div>
    <div class="note">仅展示当前模式下「可调且生效」的参数；带条件的参数会随依赖项自动显隐。
      鼠标悬停 <span class="help">?</span> 查看说明。</div>
    <div id="form"></div>
  </div>
  <div class="right">
    <div class="rhead">
      <span><span id="statusdot" class="dot"></span><span id="statustext">就绪</span></span>
      <span class="tabs">
        <button id="tabLogs" class="on">实时日志</button>
        <button id="tabYaml">YAML 预览</button>
      </span>
    </div>
    <div id="term" class="term">
      <div class="termbar">
        <span class="lights"><i class="r"></i><i class="y"></i><i class="g"></i></span>
        <span id="termtitle" class="ttitle">terminal · 就绪</span>
        <span class="ttools">
          <button id="copyBtn" class="tbtn">复制</button>
          <button id="clearBtn" class="tbtn">清空</button>
        </span>
      </div>
      <div id="logs"></div>
      <button id="toBottom" class="tobottom hidden">↓ 回到底部</button>
    </div>
    <pre id="yamlview" class="hidden"></pre>
    <div id="msg" class="msgbar muted">就绪。</div>
  </div>
</div>
<script>window.__MODE__ = "@@MODE@@";</script>
<script>@@JS@@</script>
</body>
</html>
"""


def render_page(mode: str) -> str:
    """返回某模式（'2_5d' / '3d'）的完整 HTML 文本。"""
    mode_label = "2.5D" if mode == "2_5d" else "3D"
    html = _TEMPLATE
    # 先注入小字段，再注入 CSS/JS（CSS/JS 内可能含 @@? 不会，安全）。
    repl = {
        "@@MODE_LABEL@@": mode_label,
        "@@MODE@@": mode,
        "@@A25@@": "active" if mode == "2_5d" else "",
        "@@A3D@@": "active" if mode == "3d" else "",
        "@@CSS@@": CSS,
        "@@JS@@": JS,
    }
    for k, v in repl.items():
        html = html.replace(k, v)
    return html
