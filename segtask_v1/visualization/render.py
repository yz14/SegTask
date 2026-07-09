"""IR → 自包含 HTML 渲染器（ELK.js 浏览器内实时布局）。

把若干 ``VisGraph`` 序列化为内嵌 JSON，配一段原生 HTML/CSS/JS：
* 顶部标签页切换 Data / Model / Predict；
* 布局由内嵌的 **ELK.js**（Eclipse Layout Kernel，分层 Sugiyama 算法，EPL-2.0）
  在浏览器内实时计算：嵌套容器、正交布线、跨层级跳连/残差均由通用算法处理，
  折叠/展开任意容器即重新布局——不再有任何手工车道路由与架构特判；
* 双击任意框 → 右侧详情抽屉展示完整参数；单击顶层框 → 聚焦高亮其直接连接。

ELK.js 打包源码（segtask_v1/visualization/static/elk.bundled.js）随包分发并
内联进生成的 HTML，故输出仍是**单文件、离线可打开、零运行环境依赖**。
"""

from __future__ import annotations

import html
import json
import os
from typing import Dict, List

from .graph import VisGraph

_ELK_PATH = os.path.join(os.path.dirname(__file__), "static", "elk.bundled.js")


def _elk_source() -> str:
    """读取随包分发的 ELK.js 打包源码（内联进 HTML）。"""
    try:
        with open(_ELK_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as e:  # pragma: no cover - 包损坏才会触发
        raise RuntimeError(
            f"缺少 ELK.js 布局引擎资源：{_ELK_PATH}；"
            "请确认 segtask_v1/visualization/static/elk.bundled.js 随包完整分发。"
        ) from e


# ---------------------------------------------------------------------------
# CSS —— 克制配色：浅底 + 中性灰 + 按 kind 区分的少量强调色。
# ---------------------------------------------------------------------------
_CSS = """
:root {
  --bg: #f6f7f9; --panel: #ffffff; --ink: #1f2430; --muted: #6b7280;
  --line: #c7ccd6; --edge: #94a3b8; --accent: #2563eb;
  /* 字体栈：无衬线优先现代 UI 字体，逐级回退到系统字体与 CJK；技术令牌
     （形状元组/类型/ops）走等宽字体并启用等宽数字，使数字纵向对齐更整齐。 */
  --font-sans: "Inter", "SF Pro Text", -apple-system, BlinkMacSystemFont,
    "Segoe UI", Roboto, "Helvetica Neue", Arial, "PingFang SC",
    "Microsoft YaHei", sans-serif;
  --font-mono: "JetBrains Mono", "SF Mono", SFMono-Regular, ui-monospace,
    "Cascadia Code", Menlo, Consolas, "Liberation Mono", monospace;
  --c-data: #0e7490; --c-data-bg: #ecfeff;
  --c-process: #b45309; --c-process-bg: #fffbeb;
  --c-input: #4338ca; --c-input-bg: #eef2ff;
  --c-stage: #334155; --c-stage-bg: #f1f5f9;
  --c-conv: #1d4ed8; --c-conv-bg: #eff6ff;
  --c-norm: #047857; --c-norm-bg: #ecfdf5;
  --c-act: #a21caf; --c-act-bg: #fdf4ff;
  --c-op: #475569; --c-op-bg: #f8fafc;
  --c-merge: #d97706; --c-merge-bg: #fff7ed;
  --c-head: #be185d; --c-head-bg: #fdf2f8;
  --c-output: #15803d; --c-output-bg: #f0fdf4;
  --c-loss: #b91c1c; --c-loss-bg: #fef2f2;
  --c-model: #6d28d9; --c-model-bg: #f5f3ff;
}
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; }
body {
  font-family: var(--font-sans);
  background: var(--bg); color: var(--ink); font-size: 13px;
  line-height: 1.45;
  -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
  text-rendering: optimizeLegibility;
  font-feature-settings: "kern" 1, "liga" 1, "calt" 1;
}
header {
  background: var(--panel); border-bottom: 1px solid var(--line);
  padding: 12px 20px; position: sticky; top: 0; z-index: 30;
}
header h1 { margin: 0 0 6px; font-size: 16px; font-weight: 680;
  letter-spacing: -.01em; }
.meta { display: flex; flex-wrap: wrap; gap: 6px 14px; color: var(--muted);
  font-size: 12px; font-variant-numeric: tabular-nums; }
.meta b { color: var(--ink); font-weight: 600; }
.tabs { display: flex; gap: 6px; padding: 10px 20px 0; background: var(--bg);
  position: sticky; top: 62px; z-index: 20; }
.tab {
  border: 1px solid var(--line); border-bottom: none; background: #eceef2;
  color: var(--muted); padding: 7px 16px; border-radius: 8px 8px 0 0;
  cursor: pointer; font-weight: 600; font-size: 12.5px;
}
.tab.active { background: var(--panel); color: var(--accent);
  box-shadow: 0 -2px 0 var(--accent) inset; }
.canvas-wrap { padding: 18px 20px 60px; }
.flow { display: none; }
.flow.active { display: block; }
/* 画布：ELK 输出绝对坐标，所有框/容器绝对定位其上；连线 SVG 覆盖整画布。
   连线由 ELK 正交路由绕开框体，故 SVG 层可安全置于框之上（z 20/30），
   聚焦上浮线与残差线始终可见、不再被容器背景遮挡。 */
.canvas { position: relative; }
svg.edges { position: absolute; left: 0; top: 0; pointer-events: none;
  z-index: 20; overflow: visible; }
svg.edges-top { z-index: 30; }
/* 隐藏量测箱：先把叶卡/容器标题渲染进来量出天然尺寸，再喂给 ELK 定坐标。 */
.meas { position: fixed; left: -2200px; top: 0; width: 1600px; height: 0;
  overflow: hidden; visibility: hidden; }

/* leaf node card */
.node {
  position: absolute;
  background: var(--panel); border: 1px solid var(--line); border-left-width: 4px;
  border-radius: 8px; padding: 8px 12px; min-width: 200px; max-width: 460px;
  box-shadow: 0 1px 2px rgba(15,23,42,.06); cursor: pointer;
}
.node:hover { box-shadow: 0 2px 10px rgba(37,99,235,.18); }
.node .title { font-weight: 650; font-size: 12.5px; display: flex;
  justify-content: space-between; gap: 10px; align-items: baseline;
  letter-spacing: -.005em; }
.node .badge { font-size: 10px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .06em; opacity: .8; font-family: var(--font-mono); }
.node .kv { margin-top: 4px; font-size: 11.5px; color: var(--muted);
  line-height: 1.55; }
.node .kv span { color: var(--ink); font-family: var(--font-mono);
  font-size: 11px; font-variant-numeric: tabular-nums; }
.node .hint { position: absolute; right: 8px; bottom: 4px; font-size: 9.5px;
  color: var(--muted); opacity: 0; }
.node:hover .hint { opacity: .7; }

/* —— 聚焦高亮（单击顶层模块）：仅被点模块+直接邻居+相连边醒目，其余淡出 —— */
/* foc-vis 标记「保持明亮」的框（被聚焦框/相关框本身 + 其全部后代 + 其祖先 stage）：
   淡化规则只作用于**不带 foc-vis** 的框,故选中框与相关框的内部子模块都完整可见。 */
.node, .stage { transition: opacity .15s, box-shadow .15s; }
.canvas.focusing .node:not(.foc-vis),
.canvas.focusing .stage:not(.foc-vis) {
  opacity: .12; filter: saturate(.5); }
.node.foc-on, .stage.foc-on {
  opacity: 1; border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent), 0 6px 20px rgba(37,99,235,.32); }
.node.foc-nb, .stage.foc-nb {
  opacity: 1;
  box-shadow: 0 0 0 2.5px #f59e0b, 0 4px 14px rgba(245,158,11,.30); }

/* stage container：展开时仅绘外框+标题（子框独立绝对定位于画布），
   折叠时缩为标题条。 */
.stage {
  position: absolute;
  background: var(--c-stage-bg); border: 1px dashed var(--c-stage);
  border-radius: 12px;
}
.stage > .stage-head {
  display: flex; align-items: center; gap: 8px; padding: 8px 12px;
  cursor: pointer; font-weight: 700; color: var(--c-stage); user-select: none;
  white-space: nowrap;
}
.stage > .stage-head .caret { transition: transform .15s; font-size: 11px;
  cursor: pointer; }
.stage > .stage-head.collapsed .caret { transform: rotate(-90deg); }
.stage > .stage-head .s-kv { color: var(--muted); font-weight: 500;
  font-size: 10.5px; margin-left: auto; font-family: var(--font-mono);
  font-variant-numeric: tabular-nums; letter-spacing: -.01em; }
.stage > .stage-head > span:not(.caret):not(.s-kv) { letter-spacing: -.005em; }

/* edge legend (线型说明) */
.legend .eline { display: inline-block; width: 22px; height: 0;
  border-top-width: 2px; border-top-style: solid; margin-right: 4px;
  vertical-align: middle; }

/* color by kind */
.k-data   { border-left-color: var(--c-data);   background: var(--c-data-bg); }
.k-process{ border-left-color: var(--c-process);background: var(--c-process-bg);}
.k-input  { border-left-color: var(--c-input);  background: var(--c-input-bg); }
.k-conv   { border-left-color: var(--c-conv);   background: var(--c-conv-bg); }
.k-norm   { border-left-color: var(--c-norm);   background: var(--c-norm-bg); }
.k-act    { border-left-color: var(--c-act);    background: var(--c-act-bg); }
.k-op     { border-left-color: var(--c-op);     background: var(--c-op-bg); }
/* merge：融合算子（cat / + / × …）渲染成紧凑胶囊，靠形状即可与普通叶子区分。 */
.k-merge  { border: 1.5px solid var(--c-merge); border-left-width: 1.5px;
  background: var(--c-merge-bg); min-width: 0; width: fit-content;
  border-radius: 999px; padding: 5px 16px; }
.k-merge .title { justify-content: center; gap: 6px; font-size: 13px;
  font-weight: 800; color: var(--c-merge); }
.k-merge .badge { display: none; }
.k-merge .kv { text-align: center; margin-top: 2px; }
.k-head   { border-left-color: var(--c-head);   background: var(--c-head-bg); }
.k-output { border-left-color: var(--c-output); background: var(--c-output-bg); }
.k-loss   { border-left-color: var(--c-loss);   background: var(--c-loss-bg); }
.k-model  { border-left-color: var(--c-model);  background: var(--c-model-bg);
  min-width: 300px; }
.badge.k-data{color:var(--c-data)} .badge.k-process{color:var(--c-process)}
.badge.k-input{color:var(--c-input)} .badge.k-conv{color:var(--c-conv)}
.badge.k-norm{color:var(--c-norm)} .badge.k-act{color:var(--c-act)}
.badge.k-op{color:var(--c-op)} .badge.k-merge{color:var(--c-merge)}
.badge.k-head{color:var(--c-head)}
.badge.k-output{color:var(--c-output)} .badge.k-loss{color:var(--c-loss)}
.badge.k-model{color:var(--c-model)}

/* detail drawer */
.drawer {
  position: fixed; top: 0; right: 0; height: 100%; width: 380px;
  background: var(--panel); border-left: 1px solid var(--line);
  box-shadow: -4px 0 24px rgba(15,23,42,.12); transform: translateX(100%);
  transition: transform .2s ease; z-index: 50; display: flex;
  flex-direction: column;
}
.drawer.open { transform: translateX(0); }
.drawer-head { padding: 14px 16px; border-bottom: 1px solid var(--line);
  display: flex; justify-content: space-between; align-items: center; }
.drawer-head h2 { margin: 0; font-size: 14px; }
.drawer-head .x { cursor: pointer; color: var(--muted); font-size: 20px;
  line-height: 1; border: none; background: none; }
.drawer-body { padding: 12px 16px; overflow-y: auto; }
.drawer-body .grp { margin-bottom: 16px; }
.drawer-body .grp h3 { margin: 0 0 6px; font-size: 11px; text-transform: uppercase;
  letter-spacing: .05em; color: var(--muted); }
.drawer-body table { width: 100%; border-collapse: collapse; font-size: 12px; }
.drawer-body td { padding: 3px 6px; vertical-align: top;
  border-bottom: 1px solid #eef0f3; }
.drawer-body td.k { color: var(--muted); white-space: nowrap; width: 42%; }
.drawer-body td.v { color: var(--ink); word-break: break-word;
  font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
.legend { display: flex; flex-wrap: wrap; gap: 10px; padding: 6px 20px 0;
  color: var(--muted); font-size: 11px; }
.legend .dot { display: inline-block; width: 10px; height: 10px;
  border-radius: 3px; margin-right: 4px; vertical-align: middle; }
.empty { color: var(--muted); padding: 40px; text-align: center; }
.layouting { color: var(--muted); padding: 24px; text-align: center;
  font-size: 12px; }
"""

# ---------------------------------------------------------------------------
# JS —— ELK.js 浏览器内布局 + 绝对定位 DOM + SVG 正交连线 + 折叠 + 聚焦 + 抽屉。
#
# 与旧实现的本质区别：不再有 CSS Grid 车道/侧缘同心弧/贪心两色划分等任何手工
# 路由——节点坐标与每条边的正交折点全部由 ELK 分层算法给出（嵌套容器经
# INCLUDE_CHILDREN 一体布局、edgeCoords=ROOT 使折点直接落在画布坐标系）。
# 折叠/展开任意容器 = 图变化 = 重新调 ELK 布局，对任意 decoder_type /
# 任意架构通用，无一处特判。
# ---------------------------------------------------------------------------
_JS = r"""
const DATA = __DATA__;
const elk = new ELK();
// 连边线型/配色：forward 主流（实线灰）、skip 跳连（实线蓝）、
// residual 残差（琥珀虚线）。几何全部来自 ELK 正交路由。
const EDGE_STYLE = {
  forward:  { color: "#94a3b8", width: "1.6", dash: null,  marker: "arrow" },
  skip:     { color: "#2563eb", width: "1.8", dash: null,  marker: "arrow-skip" },
  residual: { color: "#d97706", width: "1.8", dash: "3 3", marker: "arrow-res" },
};
const LAYOUT_OPTS = {
  "elk.algorithm": "layered",
  "elk.direction": "DOWN",
  "elk.hierarchyHandling": "INCLUDE_CHILDREN",
  "elk.edgeRouting": "ORTHOGONAL",
  "org.eclipse.elk.json.edgeCoords": "ROOT",
  "elk.layered.spacing.nodeNodeBetweenLayers": "32",
  "elk.spacing.nodeNode": "26",
  "elk.spacing.edgeNode": "14",
  "elk.spacing.edgeEdge": "10",
  "elk.layered.spacing.edgeNodeBetweenLayers": "14",
  "elk.layered.spacing.edgeEdgeBetweenLayers": "10",
  "elk.padding": "[top=8,left=8,bottom=8,right=8]",
};
const STATE = {};       // flowKey -> 每视图状态（折叠集合/聚焦/布局缓存）
let measBox = null;     // 隐藏量测箱

function el(tag, cls, txt) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (txt != null) e.textContent = txt;
  return e;
}
function svgNS(tag) {
  return document.createElementNS("http://www.w3.org/2000/svg", tag);
}

// ---------------- IR 索引 ----------------
function indexGraph(g) {
  const byId = {}, kids = {};
  (g.nodes || []).forEach(n => { byId[n.id] = n; });
  (g.nodes || []).forEach(n => {
    if (n.parent_id) (kids[n.parent_id] = kids[n.parent_id] || []).push(n);
  });
  return { byId, kids };
}

// ---------------- DOM 构建（卡片 / 容器标题） ----------------
function leafCard(node, st) {
  const card = el("div", "node k-" + node.kind);
  card.dataset.nodeId = node.id;
  const title = el("div", "title");
  title.appendChild(el("span", null, node.label));
  title.appendChild(el("span", "badge k-" + node.kind, node.kind));
  card.appendChild(title);
  const keys = Object.keys(node.key_info || {});
  if (keys.length) {
    const kv = el("div", "kv");
    keys.forEach(k => {
      const line = el("div");
      line.appendChild(el("b", null, k + ": "));
      line.appendChild(el("span", null, node.key_info[k]));
      kv.appendChild(line);
    });
    card.appendChild(kv);
  }
  const topLevel = !node.parent_id;
  card.appendChild(el("div", "hint",
    topLevel ? "\u5355\u51FB\u9AD8\u4EAE \u00B7 \u53CC\u51FB\u8BE6\u60C5"
             : "\u53CC\u51FB\u8BE6\u60C5"));
  if (st) attachNodeHandlers(card, node, st, topLevel);
  return card;
}

function stageBox(node, st, collapsed) {
  const box = el("div", "stage k-stage");
  box.dataset.nodeId = node.id;
  const head = el("div", "stage-head" + (collapsed ? " collapsed" : ""));
  const caret = el("span", "caret", "\u25BC");
  head.appendChild(caret);
  head.appendChild(el("span", null, node.label));
  const sk = Object.entries(node.key_info || {})
    .map(([k, v]) => k + " " + v).join("  ·  ");
  if (sk) head.appendChild(el("span", "s-kv", sk));
  box.appendChild(head);
  if (st) {
    // 折叠/展开仅由标题左侧三角触发；标题其余区域单击=聚焦（仅顶层）、双击=详情。
    caret.addEventListener("click", (ev) => {
      ev.stopPropagation();
      if (st.collapsed.has(node.id)) st.collapsed.delete(node.id);
      else st.collapsed.add(node.id);
      relayout(st);
    });
    attachNodeHandlers(head, node, st, !node.parent_id);
  }
  return box;
}

let _clickTimer = null;
function attachNodeHandlers(elm, node, st, topLevel) {
  if (topLevel) {
    elm.addEventListener("click", (ev) => {
      ev.stopPropagation();
      if (_clickTimer) return;           // 双击进行中：交由 dblclick 处理
      _clickTimer = setTimeout(() => {
        _clickTimer = null;
        st.focus = (st.focus === node.id) ? null : node.id;
        applyFocus(st);
      }, 220);
    });
    elm.addEventListener("dblclick", (ev) => {
      ev.stopPropagation();
      if (_clickTimer) { clearTimeout(_clickTimer); _clickTimer = null; }
      openDrawer(node);
    });
  } else {
    // 框内子模块不参与聚焦：单击不做任何事，仅双击看详情。
    elm.addEventListener("click", (ev) => ev.stopPropagation());
    elm.addEventListener("dblclick", (ev) => {
      ev.stopPropagation(); openDrawer(node);
    });
  }
}

// ---------------- 尺寸量测（喂给 ELK） ----------------
// 叶卡按内容自适应（min/max-width 由 CSS 约束）；容器只量标题条（展开时
// 作为 ELK padding-top 与最小宽，折叠时即整框尺寸）。结果按 node.id 缓存，
// 字体加载完成后统一失效重量。
function measure(node, variant, st) {
  const ck = variant + "|" + node.id;
  if (st.measCache[ck]) return st.measCache[ck];
  const e = (variant === "leaf") ? leafCard(node, null)
                                 : stageBox(node, null, true);
  measBox.appendChild(e);
  const r = e.getBoundingClientRect();
  const m = { w: Math.ceil(r.width), h: Math.ceil(r.height) };
  measBox.removeChild(e);
  st.measCache[ck] = m;
  return m;
}

// ---------------- IR → ELK 图 ----------------
function toElk(n, st) {
  const kids = st.ix.kids[n.id] || [];
  if (kids.length && !st.collapsed.has(n.id)) {
    const hm = measure(n, "head", st);
    return {
      id: n.id,
      layoutOptions: {
        "elk.padding": "[top=" + (hm.h + 6) + ",left=12,bottom=12,right=12]",
        "elk.nodeSize.constraints": "MINIMUM_SIZE",
        "elk.nodeSize.minimum": "(" + (hm.w + 4) + "," + (hm.h + 30) + ")",
      },
      children: kids.map(k => toElk(k, st)),
    };
  }
  const m = measure(n, kids.length ? "head" : "leaf", st);
  return { id: n.id, width: m.w, height: m.h };
}

// 折叠上卷：节点若藏在折叠容器内，其连边端点改挂到最外层折叠祖先。
function visAnchor(id, st) {
  const nd = st.ix.byId[id];
  if (!nd) return null;
  let anchor = id, cur = nd.parent_id;
  while (cur) {
    if (st.collapsed.has(cur)) anchor = cur;
    const p = st.ix.byId[cur];
    cur = p ? p.parent_id : null;
  }
  return anchor;
}
function isAncestor(a, b, st) {
  let cur = (st.ix.byId[b] || {}).parent_id;
  while (cur) {
    if (cur === a) return true;
    cur = (st.ix.byId[cur] || {}).parent_id;
  }
  return false;
}
function visEdges(st) {
  const seen = new Set(), out = [];
  (st.g.edges || []).forEach(ed => {
    const s = visAnchor(ed.src, st), t = visAnchor(ed.dst, st);
    if (!s || !t || s === t) return;
    if (isAncestor(s, t, st) || isAncestor(t, s, st)) return;
    const kind = ed.kind || "forward";
    const key = s + ">" + t + ">" + kind;
    if (seen.has(key)) return;
    seen.add(key);
    out.push({ id: "e" + out.length, vs: s, vt: t,
               kind, label: ed.label || "" });
  });
  return out;
}

// ---------------- 布局 + 绘制 ----------------
async function relayout(st) {
  const my = ++st.seq;                 // 丢弃过期的异步布局结果
  const tops = (st.g.nodes || []).filter(n => !n.parent_id);
  const canvas = st.flowEl.querySelector(".canvas");
  if (!tops.length) {
    canvas.innerHTML = "";
    canvas.appendChild(el("div", "empty", "（此视图无节点）"));
    st.laidOut = true;
    return;
  }
  const eds = visEdges(st);
  const root = {
    id: "$root",
    layoutOptions: LAYOUT_OPTS,
    children: tops.map(n => toElk(n, st)),
    edges: eds.map(e => ({ id: e.id, sources: [e.vs], targets: [e.vt] })),
  };
  let res;
  try {
    res = await elk.layout(root);
  } catch (err) {
    console.error("ELK layout failed:", err);
    canvas.innerHTML = "";
    canvas.appendChild(el("div", "empty", "布局失败：" + err));
    return;
  }
  if (my !== st.seq) return;
  paint(st, res, eds);
  st.laidOut = true;
}

function marker(id, color) {
  return '<marker id="' + id + '" viewBox="0 0 10 10" refX="9" refY="5" ' +
    'markerWidth="7" markerHeight="7" orient="auto-start-reverse">' +
    '<path d="M 0 1 L 9 5 L 0 9 z" fill="' + color + '"/></marker>';
}
function svgDefs() {
  return '<defs>' +
    marker("arrow", EDGE_STYLE.forward.color) +
    marker("arrow-skip", EDGE_STYLE.skip.color) +
    marker("arrow-res", EDGE_STYLE.residual.color) +
    '</defs>';
}

// 正交折线 + 圆角：给定折点序列，输出带圆角拐弯的 path。
function ortho(pts, r) {
  r = (r == null) ? 7 : r;
  if (pts.length < 2) return "";
  let d = "M " + pts[0][0] + " " + pts[0][1];
  for (let i = 1; i < pts.length - 1; i++) {
    const p0 = pts[i - 1], p1 = pts[i], p2 = pts[i + 1];
    const d1 = Math.hypot(p1[0] - p0[0], p1[1] - p0[1]) || 1;
    const d2 = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]) || 1;
    const rr = Math.min(r, d1 / 2, d2 / 2);
    const u1x = (p1[0] - p0[0]) / d1, u1y = (p1[1] - p0[1]) / d1;
    const u2x = (p2[0] - p1[0]) / d2, u2y = (p2[1] - p1[1]) / d2;
    d += " L " + (p1[0] - u1x * rr) + " " + (p1[1] - u1y * rr)
       + " Q " + p1[0] + " " + p1[1] + " "
       + (p1[0] + u2x * rr) + " " + (p1[1] + u2y * rr);
  }
  const last = pts.length - 1;
  d += " L " + pts[last][0] + " " + pts[last][1];
  return d;
}

// 折线中点（按弧长），用于放边标签。
function midOf(pts) {
  let total = 0;
  for (let i = 1; i < pts.length; i++)
    total += Math.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1]);
  let acc = 0;
  for (let i = 1; i < pts.length; i++) {
    const seg = Math.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1]);
    if (acc + seg >= total / 2) {
      const t = seg ? (total / 2 - acc) / seg : 0;
      return [pts[i-1][0] + t * (pts[i][0] - pts[i-1][0]),
              pts[i-1][1] + t * (pts[i][1] - pts[i-1][1])];
    }
    acc += seg;
  }
  return pts[pts.length - 1];
}

function paint(st, root, eds) {
  const canvas = st.flowEl.querySelector(".canvas");
  canvas.innerHTML = "";
  const svg = svgNS("svg");
  svg.setAttribute("class", "edges");
  svg.innerHTML = svgDefs();
  const svgTop = svgNS("svg");
  svgTop.setAttribute("class", "edges edges-top");
  svgTop.innerHTML = svgDefs();
  canvas.appendChild(svg);

  // 节点绝对定位：ELK 子坐标相对父容器，累加得画布绝对坐标。容器 z 按深度
  // 递增（外层最低），叶卡恒在容器之上（z10），连线层再往上（z20/30）。
  const nodeEls = {};
  const walk = (n, ox, oy, depth) => {
    (n.children || []).forEach(c => {
      const x = ox + (c.x || 0), y = oy + (c.y || 0);
      const nd = st.ix.byId[c.id];
      const kids = st.ix.kids[c.id] || [];
      const isBox = kids.length > 0;
      const e = isBox ? stageBox(nd, st, st.collapsed.has(c.id))
                      : leafCard(nd, st);
      e.style.left = x + "px";
      e.style.top = y + "px";
      e.style.width = c.width + "px";
      e.style.height = c.height + "px";
      e.style.zIndex = isBox ? String(1 + depth) : "10";
      canvas.appendChild(e);
      nodeEls[c.id] = e;
      walk(c, x, y, depth + 1);
    });
  };
  walk(root, 0, 0, 0);
  canvas.appendChild(svgTop);
  const W = Math.ceil(root.width || 0), H = Math.ceil(root.height || 0);
  canvas.style.width = W + "px";
  canvas.style.height = H + "px";
  [svg, svgTop].forEach(s => {
    s.setAttribute("width", W); s.setAttribute("height", H);
    s.style.width = W + "px"; s.style.height = H + "px";
  });

  // 连线：直接采用 ELK 的正交 sections（edgeCoords=ROOT，画布坐标系）。
  const byEid = {};
  eds.forEach(e => { byEid[e.id] = e; });
  const edgeEls = [];
  (root.edges || []).forEach(le => {
    const meta = byEid[le.id];
    if (!meta || !le.sections || !le.sections.length) return;
    const pts = [];
    le.sections.forEach(s => {
      pts.push([s.startPoint.x, s.startPoint.y]);
      (s.bendPoints || []).forEach(p => pts.push([p.x, p.y]));
      pts.push([s.endPoint.x, s.endPoint.y]);
    });
    const style = EDGE_STYLE[meta.kind] || EDGE_STYLE.forward;
    const path = svgNS("path");
    path.setAttribute("d", ortho(pts, 7));
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", style.color);
    path.setAttribute("stroke-width", style.width);
    if (style.dash) path.setAttribute("stroke-dasharray", style.dash);
    path.setAttribute("marker-end", "url(#" + style.marker + ")");
    const home = (meta.kind === "residual") ? svgTop : svg;
    home.appendChild(path);
    let txt = null;
    if (meta.label) {
      const [mx, my2] = midOf(pts);
      txt = svgNS("text");
      txt.setAttribute("x", mx + 5);
      txt.setAttribute("y", my2 - 3);
      txt.setAttribute("fill", style.color);
      txt.setAttribute("font-size", "10.5");
      txt.setAttribute("font-weight", meta.kind !== "forward" ? "700" : "400");
      txt.textContent = meta.label;
      home.appendChild(txt);
    }
    edgeEls.push({ meta, path, txt, home });
  });
  st.rc = { nodeEls, edgeEls, svg, svgTop, canvas };
  applyFocus(st);
}

// —— 聚焦高亮：单击顶层模块后，仅其本身+直接邻居+相连边醒目，其余淡出 ——
// 通用、数据驱动（基于折叠上卷后的可见邻接）：相连边几何不变（仍是 ELK 路由），
// 只上浮到顶层 SVG 并保持满不透明；无关边淡出；相关框及其后代/祖先保持明亮。
function applyFocus(st) {
  const rc = st.rc;
  if (!rc) return;
  const f = st.focus;
  Object.values(rc.nodeEls).forEach(e =>
    e.classList.remove("foc-on", "foc-nb", "foc-vis"));
  rc.canvas.classList.toggle("focusing", f != null);
  rc.edgeEls.forEach(it => {
    it.path.style.opacity = "";
    if (it.txt) it.txt.style.opacity = "";
    if (it.path.parentNode !== it.home) it.home.appendChild(it.path);
    if (it.txt && it.txt.parentNode !== it.home) it.home.appendChild(it.txt);
  });
  if (f == null) return;
  // 「与聚焦框相连」= 边跨越聚焦框边界（一端在框内/即框本身，另一端在框外）。
  // 对展开的容器同样成立：其后代的对外连边视作容器的连边（滚动上卷）。
  const inF = (id) => id === f || isAncestor(f, id, st);
  const focusSet = new Set([f]);
  const focEdges = new Set();
  rc.edgeEls.forEach(it => {
    const a = inF(it.meta.vs), b = inF(it.meta.vt);
    if (a === b) return;
    focEdges.add(it);
    focusSet.add(a ? it.meta.vt : it.meta.vs);
  });
  const markVis = (nid) => {
    const e2 = rc.nodeEls[nid];
    if (e2) e2.classList.add("foc-vis");
    (st.ix.kids[nid] || []).forEach(k => markVis(k.id));
  };
  const mark = (id, cls) => {
    const e = rc.nodeEls[id];
    if (!e) return;
    if (!e.classList.contains("foc-on")) e.classList.add(cls);
    markVis(id);
    let p = (st.ix.byId[id] || {}).parent_id;
    while (p) {
      if (rc.nodeEls[p]) rc.nodeEls[p].classList.add("foc-vis");
      p = (st.ix.byId[p] || {}).parent_id;
    }
  };
  mark(f, "foc-on");
  focusSet.forEach(id => { if (id !== f) mark(id, "foc-nb"); });
  rc.edgeEls.forEach(it => {
    const isFoc = focEdges.has(it);
    if (isFoc) {
      rc.svgTop.appendChild(it.path);
      if (it.txt) rc.svgTop.appendChild(it.txt);
      return;
    }
    const a = rc.nodeEls[it.meta.vs], b = rc.nodeEls[it.meta.vt];
    const internal = a && b && a.classList.contains("foc-vis")
      && b.classList.contains("foc-vis");
    if (!internal) {
      it.path.style.opacity = "0.08";
      if (it.txt) it.txt.style.opacity = "0.08";
    }
  });
}

function clearFocus(st) {
  if (st.focus !== null) { st.focus = null; applyFocus(st); }
}
function activeState() {
  const flow = document.querySelector(".flow.active");
  return flow ? STATE[flow.dataset.flow] : null;
}

// ---------------- 详情抽屉 ----------------
function openDrawer(node) {
  const d = document.getElementById("drawer");
  document.getElementById("drawer-title").textContent = node.label;
  const body = document.getElementById("drawer-body");
  body.innerHTML = "";
  body.appendChild(grpTable("Type", { kind: node.kind }));
  if (Object.keys(node.key_info || {}).length)
    body.appendChild(grpTable("Key info", node.key_info));
  if (Object.keys(node.detail || {}).length)
    body.appendChild(grpTable("Details", node.detail));
  d.classList.add("open");
}
function grpTable(title, obj) {
  const g = el("div", "grp");
  g.appendChild(el("h3", null, title));
  const t = el("table");
  Object.entries(obj).forEach(([k, v]) => {
    const tr = el("tr");
    tr.appendChild(el("td", "k", k));
    tr.appendChild(el("td", "v", String(v)));
    t.appendChild(tr);
  });
  g.appendChild(t);
  return g;
}
function closeDrawer() {
  document.getElementById("drawer").classList.remove("open");
}

// ---------------- 初始化 / 标签页 ----------------
function init() {
  measBox = el("div", "meas");
  document.body.appendChild(measBox);
  const order = DATA.order.filter(k => DATA.flows[k]);
  const tabsEl = document.getElementById("tabs");
  const wrap = document.getElementById("canvas-wrap");
  order.forEach((k, idx) => {
    const g = DATA.flows[k];
    const ix = indexGraph(g);
    const st = { key: k, g, ix, collapsed: new Set(), focus: null,
                 laidOut: false, rc: null, measCache: {}, seq: 0 };
    (g.nodes || []).forEach(n => {
      if (n.collapsed && (ix.kids[n.id] || []).length) st.collapsed.add(n.id);
    });
    STATE[k] = st;
    const tab = el("div", "tab" + (idx === 0 ? " active" : ""),
                   DATA.titles[k] || k);
    tab.dataset.flow = k;
    tab.addEventListener("click", () => activate(k));
    tabsEl.appendChild(tab);
    const flow = el("div", "flow" + (idx === 0 ? " active" : ""));
    flow.dataset.flow = k;
    const canvas = el("div", "canvas");
    canvas.appendChild(el("div", "layouting", "布局计算中…"));
    // 点击画布空白处 → 清除聚焦（节点/标题的 click 均 stopPropagation）。
    canvas.addEventListener("click", () => clearFocus(st));
    flow.appendChild(canvas);
    st.flowEl = flow;
    wrap.appendChild(flow);
  });
  renderMeta(order[0]);
  relayout(STATE[order[0]]);   // 其余标签页首次激活时才布局（懒加载）
  // 字体异步换字会改变卡片实测尺寸：加载完成后失效量测缓存并重排。
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(() => {
      Object.values(STATE).forEach(st => {
        st.measCache = {};
        if (st.laidOut) relayout(st);
      });
    });
  }
}
function activate(k) {
  document.querySelectorAll(".tab").forEach(t =>
    t.classList.toggle("active", t.dataset.flow === k));
  document.querySelectorAll(".flow").forEach(f =>
    f.classList.toggle("active", f.dataset.flow === k));
  renderMeta(k);
  const st = STATE[k];
  if (!st.laidOut) relayout(st);
}
function renderMeta(k) {
  const m = (DATA.flows[k] && DATA.flows[k].meta) || {};
  const box = document.getElementById("meta");
  box.innerHTML = "";
  Object.entries(m).forEach(([kk, vv]) => {
    const s = el("span");
    s.appendChild(el("b", null, kk + ": "));
    s.appendChild(document.createTextNode(vv));
    box.appendChild(s);
  });
}

window.addEventListener("keydown", e => {
  if (e.key === "Escape") {
    closeDrawer();
    const st = activeState();
    if (st) clearFocus(st);
  }
});
document.addEventListener("DOMContentLoaded", init);
"""


def _legend_html() -> str:
    items = [
        ("data", "数据"), ("process", "处理"), ("input", "输入"),
        ("stage", "阶段"), ("conv", "卷积"), ("norm", "归一化"),
        ("act", "激活"), ("op", "算子"), ("merge", "融合"), ("head", "头"),
        ("output", "输出"), ("loss", "损失"), ("model", "模型"),
    ]
    spans = []
    for kind, name in items:
        spans.append(
            f'<span><i class="dot" style="background:var(--c-{kind})"></i>'
            f'{html.escape(name)}</span>')
    # 连边线型说明（与 EDGE_STYLE 对应）。
    edges = [
        ("#94a3b8", "solid", "前向 forward"),
        ("#2563eb", "solid", "跳连 skip"),
        ("#d97706", "dashed", "残差 residual"),
    ]
    spans.append('<span style="width:1px;height:14px;background:var(--line)">'
                 '</span>')
    for color, style, name in edges:
        spans.append(
            f'<span><i class="eline" style="border-top-color:{color};'
            f'border-top-style:{style}"></i>{html.escape(name)}</span>')
    # 交互提示：单击模块高亮其连接，空白/Esc 还原。
    spans.append(
        '<span style="margin-left:auto;color:var(--accent);font-weight:600">'
        '单击模块：高亮其连接 · 空白处/Esc 还原</span>')
    return '<div class="legend">' + "".join(spans) + "</div>"


def render_html(
    graphs: Dict[str, VisGraph],
    order: List[str],
    page_title: str = "SegTask Pipeline Visualization",
) -> str:
    """把 ``{flow_key: VisGraph}`` 渲染成单文件 HTML 字符串。

    ``order`` 决定标签页顺序；缺失的 flow 会被 JS 自动跳过。
    """
    titles = {
        "data": "数据流 Data Flow",
        "model": "模型流 Model Flow",
        "predict": "预测流 Prediction Flow",
    }
    payload = {
        "flows": {k: g.to_dict() for k, g in graphs.items()},
        "order": list(order),
        "titles": {k: titles.get(k, k) for k in order},
    }
    data_json = json.dumps(payload, ensure_ascii=False)
    js = _JS.replace("__DATA__", data_json)
    esc_title = html.escape(page_title)
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc_title}</title>
<style>{_CSS}</style>
</head>
<body>
<header>
  <h1>{esc_title}</h1>
  <div class="meta" id="meta"></div>
</header>
{_legend_html()}
<div class="tabs" id="tabs"></div>
<div class="canvas-wrap" id="canvas-wrap"></div>
<div class="drawer" id="drawer">
  <div class="drawer-head">
    <h2 id="drawer-title">Details</h2>
    <button class="x" onclick="closeDrawer()">&times;</button>
  </div>
  <div class="drawer-body" id="drawer-body"></div>
</div>
<script>/* ELK.js (Eclipse Layout Kernel, EPL-2.0) — 内联打包，供浏览器内实时布局 */
{_elk_source()}</script>
<script>{js}</script>
</body>
</html>
"""


__all__ = ["render_html"]
