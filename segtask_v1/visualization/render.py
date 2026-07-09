"""IR → 自包含 HTML 渲染器（结构化确定性布局）。

把若干 ``VisGraph`` 序列化为内嵌 JSON，配一段原生 HTML/CSS/JS：
* 顶部标签页切换 Data / Model / Predict；
* 布局由一套**只依赖图结构与追踪形状**的确定性版面规则在浏览器内计算：
  每个容器内最长路径分层；主干脊柱中轴对齐并按分辨率级（``res``）右缩进呈
  U 型；并联分支/输出头排脊柱右侧、超宽自动换行；跳连/反馈边走容器左侧车道、
  残差走右侧车道（区间打包分道）。无任何架构名/模块名特判，对任意新模块通用；
  折叠/展开任意容器即按可见锚点重新计算，结果幂等；
* 双击任意框 → 右侧详情抽屉展示完整参数；单击顶层框或其直接子模块（stem/stage 级）→ 聚焦高亮其直接连接。

输出是**单文件、离线可打开、零运行环境依赖**（无任何外部 JS 库）。
"""

from __future__ import annotations

import html
import json
from typing import Dict, List

from .graph import VisGraph


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
/* 画布：布局输出绝对坐标，所有框/容器绝对定位其上；连线 SVG 覆盖整画布。
   连线沿容器侧缘车道正交路由、绕开框体，故 SVG 层可安全置于框之上（z 20/30），
   聚焦上浮线与残差线始终可见、不再被容器背景遮挡。 */
.canvas { position: relative; }
svg.edges { position: absolute; left: 0; top: 0; pointer-events: none;
  z-index: 20; overflow: visible; }
svg.edges-top { z-index: 30; }
/* 隐藏量测箱：先把叶卡/容器标题渲染进来量出天然尺寸，再喂给布局定坐标。 */
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
  white-space: normal; flex-wrap: wrap; max-width: 100%; row-gap: 1px;
  overflow-wrap: anywhere;
}
.stage > .stage-head .caret { transition: transform .15s; font-size: 11px;
  cursor: pointer; }
.stage > .stage-head.collapsed .caret { transform: rotate(-90deg); }
/* 折叠态：卡片压窄、元信息换行，便于同层并列分支横排 */
.stage > .stage-head.collapsed { flex-wrap: wrap; white-space: normal;
  max-width: 300px; row-gap: 1px; }
.stage > .stage-head.collapsed .s-kv { flex-basis: 100%; margin-left: 0;
  white-space: normal; word-break: break-word; }
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
.main { display: flex; align-items: flex-start; }
.main > .canvas-wrap { flex: 1 1 auto; min-width: 0; }
.nav { position: sticky; top: 12px; left: 12px; z-index: 40;
  flex: 0 0 auto; margin: 18px 0 18px 12px;
  background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  box-shadow: 0 4px 16px rgba(15,23,42,.10); font-size: 12px;
  max-width: 200px; overflow: hidden; }
.nav-head { padding: 6px 10px; font-weight: 700; color: var(--c-stage);
  cursor: pointer; user-select: none; border-bottom: 1px solid var(--line); }
.nav.closed .nav-body { display: none; }
.nav.closed .nav-head { border-bottom: none; }
.nav-body { max-height: calc(100vh - 90px); overflow: auto; padding: 4px 0; }
.nav-item { padding: 3px 12px; cursor: pointer; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; color: var(--fg); }
.nav-item:hover { background: rgba(37,99,235,.08); color: var(--accent); }
.nav-item.lv1 { padding-left: 26px; }
@keyframes navflash { 0% { box-shadow: 0 0 0 3px rgba(37,99,235,.85); }
  100% { box-shadow: 0 0 0 3px rgba(37,99,235,0); } }
.nav-flash { animation: navflash 1.2s ease-out 2; }
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
# JS —— 结构化确定性布局 + 绝对定位 DOM + SVG 正交连线 + 折叠 + 聚焦 + 抽屉。
#
# 版面规则（只依赖图结构与追踪形状，无任何架构名/模块名特判）：
# 1. 每个容器内对兄弟做最长路径分层（rank）；主干脊柱 = 入口→主输出的最长
#    路径，脊柱节点中轴对齐，并按分辨率级（node.res，来自追踪形状）右缩进，
#    encoder↓/decoder↑ 自然呈 U 型；分辨率不变的流程退化为笔直单列；
# 2. 非脊柱兄弟（并联分支/输出头）排在脊柱右侧，同 rank 超宽自动换行；
# 3. 跳连/反馈边走最近公共容器的左侧车道、残差走右侧车道（短跨度贴内侧的
#    区间打包分道）；前向邻级边垂直直连。折叠/展开 = 按可见锚点重算，幂等。
# ---------------------------------------------------------------------------
_JS = r"""
const DATA = __DATA__;
// 连边线型/配色：forward 主流（实线灰）、skip 跳连（实线蓝）、
// residual 残差（琥珀虚线）。
const EDGE_STYLE = {
  forward:  { color: "#94a3b8", width: "1.6", dash: null,  marker: "arrow" },
  skip:     { color: "#2563eb", width: "1.8", dash: null,  marker: "arrow-skip" },
  residual: { color: "#d97706", width: "1.8", dash: "3 3", marker: "arrow-res" },
};
// 布局常量：行间距/列间距/车道宽/分辨率缩进/单行最大内容宽/容器内边距。
const LC = { GAPY: 40, GAPX: 26, LANE_W: 11, LANE_PAD: 8, INDENT: 26,
             MAXROW: 1500, PAD: 12, HEAD_GAP: 6 };
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
  const foc = st ? canFocus(node, st) : !node.parent_id;
  card.appendChild(el("div", "hint",
    foc ? "\u5355\u51FB\u9AD8\u4EAE \u00B7 \u53CC\u51FB\u8BE6\u60C5"
        : "\u53CC\u51FB\u8BE6\u60C5"));
  if (st) attachNodeHandlers(card, node, st, foc);
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
    // 折叠/展开仅由标题左侧三角触发；标题其余区域单击=聚焦（顶层及 stem/stage 级）、双击=详情。
    caret.addEventListener("click", (ev) => {
      ev.stopPropagation();
      if (st.collapsed.has(node.id)) st.collapsed.delete(node.id);
      else st.collapsed.add(node.id);
      relayout(st);
    });
    attachNodeHandlers(head, node, st, canFocus(node, st));
  }
  return box;
}

// 可聚焦层级：顶层模块 + 顶层容器的直接子模块（stem/stage 级）；
// 更深的子模块不参与聚焦，仅双击看详情。
function canFocus(node, st) {
  if (!node.parent_id) return true;
  const p = st.ix.byId[node.parent_id];
  return !!p && !p.parent_id;
}

let _clickTimer = null;
function attachNodeHandlers(elm, node, st, focusable) {
  if (focusable) {
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
    // 更深层子模块不参与聚焦：单击不做任何事，仅双击看详情。
    elm.addEventListener("click", (ev) => ev.stopPropagation());
    elm.addEventListener("dblclick", (ev) => {
      ev.stopPropagation(); openDrawer(node);
    });
  }
}

// ---------------- 尺寸量测（喂给布局） ----------------
// 叶卡按内容自适应（min/max-width 由 CSS 约束）；容器只量标题条（展开时
// 作为顶部内边距与最小宽，折叠时即整框尺寸）。结果按 node.id 缓存，
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

// ---------------- 结构化布局内核 ----------------
// 祖先链（自身在首位、最外层在末位）与最近公共容器（null = 画布根）。
function ancChainOf(st, id) {
  const out = [id];
  let cur = (st.ix.byId[id] || {}).parent_id;
  while (cur) { out.push(cur); cur = (st.ix.byId[cur] || {}).parent_id; }
  return out;
}
function lcaOf(st, a, b) {
  const pa = new Set(ancChainOf(st, a).slice(1));
  const cb = ancChainOf(st, b);
  for (let i = 1; i < cb.length; i++) if (pa.has(cb[i])) return cb[i];
  return null;
}
// 节点 id → 其在容器 container 直属子级中的代表（祖先或自身）。
function projTo(st, container, id) {
  let cur = id;
  while (cur != null) {
    const p = (st.ix.byId[cur] || {}).parent_id || null;
    if (p === container) return cur;
    cur = p;
  }
  return null;
}

// Kahn 最长路径分层（非 residual 边）；有环（反馈边）时贪心断环：
// 选已被环外前驱定到最高 rank 的入口强制入队。结果对节点序确定、幂等。
function layerRanks(ids, edges) {
  const idset = new Set(ids);
  const succ = {}, indeg = {}, seen = new Set();
  ids.forEach(i => { succ[i] = []; indeg[i] = 0; });
  edges.forEach(e => {
    if (e.kind === "residual" || e.s === e.t) return;
    const key = e.s + ">" + e.t;
    if (!idset.has(e.s) || !idset.has(e.t) || seen.has(key)) return;
    seen.add(key);
    succ[e.s].push(e.t);
    indeg[e.t]++;
  });
  const rank = {}, order = {};
  ids.forEach((i, k) => { rank[i] = 0; order[i] = k; });
  const indeg2 = Object.assign({}, indeg);
  let queue = ids.filter(i => indeg2[i] === 0);
  const done = new Set();
  const remaining = new Set(ids);
  while (remaining.size) {
    while (queue.length) {
      const cur = queue.pop();
      if (done.has(cur)) continue;
      done.add(cur);
      remaining.delete(cur);
      succ[cur].forEach(nx => {
        if (rank[cur] + 1 > rank[nx]) rank[nx] = rank[cur] + 1;
        if (--indeg2[nx] <= 0 && !done.has(nx)) queue.push(nx);
      });
    }
    if (remaining.size) {
      let best = null;
      remaining.forEach(i => {
        if (best === null || rank[i] > rank[best] ||
            (rank[i] === rank[best] && order[i] < order[best])) best = i;
      });
      indeg2[best] = 0;
      queue.push(best);
    }
  }
  // 纯 ASAP（就绪即排）：节点排在其全部数据源的下一层。同源后继严格同层
  // 并列（多 stem、MultiRF 分支、ds head 与下一级 dec 等），位置即计算次序。
  return rank;
}

// 主干脊柱：入口→主输出（优先 loss/output 类汇点）的最长路径节点集合。
function pickSpine(st, ids, edges, rank) {
  const pred = {}, seen = new Set();
  ids.forEach(i => { pred[i] = []; });
  edges.forEach(e => {
    if (e.kind === "residual" || e.s === e.t) return;
    if ((rank[e.t] || 0) <= (rank[e.s] || 0)) return;   // 忽略反馈/同层
    const key = e.s + ">" + e.t;
    if (seen.has(key)) return;
    seen.add(key);
    pred[e.t].push(e.s);
  });
  const order = {};
  ids.forEach((i, k) => { order[i] = k; });
  const byRank = ids.slice().sort((a, b) => (rank[a] - rank[b]) || (order[a] - order[b]));
  const len = {}, back = {};
  byRank.forEach(i => {
    len[i] = 1; back[i] = null;
    pred[i].forEach(p => {
      if (len[p] + 1 > len[i]) { len[i] = len[p] + 1; back[i] = p; }
    });
  });
  const kindPri = (i) => {
    const k = (st.ix.byId[i] || {}).kind;
    return k === "loss" ? 2 : (k === "output" ? 1 : 0);
  };
  let sink = null;
  ids.forEach(i => {
    if (sink === null) { sink = i; return; }
    const c = kindPri(i) - kindPri(sink);
    if (c > 0 || (c === 0 && (len[i] > len[sink] ||
        (len[i] === len[sink] && rank[i] > rank[sink])))) sink = i;
  });
  const spine = new Set();
  let cur = sink;
  while (cur != null && !spine.has(cur)) { spine.add(cur); cur = back[cur]; }
  return spine;
}

// 侧缘车道分配：区间打包（短跨度排前 → 贴内侧车道），返回每条边的道号与总道数。
function allocLanes(items) {
  const lanes = [], asg = {};
  items.slice().sort((a, b) => ((a.y1 - a.y0) - (b.y1 - b.y0)) || (a.y0 - b.y0))
    .forEach(it => {
      let l = 0;
      for (;; l++) {
        lanes[l] = lanes[l] || [];
        if (lanes[l].every(iv => it.y1 + 6 <= iv.y0 || it.y0 >= iv.y1 + 6)) break;
      }
      lanes[l].push({ y0: it.y0, y1: it.y1 });
      asg[it.key] = l;
    });
  return { asg, count: lanes.length };
}

// 计算整图布局：返回
//   geo   —— 可见节点 id → {x,y,w,h,depth,isBox} 画布绝对坐标；
//   routes—— 可见边 → 正交折点序列（画布绝对坐标）；
//   W/H   —— 画布尺寸。
function computeLayout(st, veds) {
  // 1) 每条可见边投影到最近公共容器下的兄弟对，按容器分桶。
  const sib = {};    // 容器 key（"$root" 或容器 id）→ [{s,t,kind,i}]
  const edgeCont = [];  // 边 i → {c, s, t}
  veds.forEach((e, i) => {
    const c = lcaOf(st, e.vs, e.vt);
    const s = projTo(st, c, e.vs), t = projTo(st, c, e.vt);
    edgeCont[i] = { c, s, t };
    if (s && t && s !== t) {
      const key = c == null ? "$root" : c;
      (sib[key] = sib[key] || []).push({ s, t, kind: e.kind, i });
    }
  });

  const geo = {};      // id → {x,y,w,h}（先相对父内容原点，走树后转绝对）
  const cinfo = {};    // 容器 key → {laneAsg, laneL, laneR, headH}
  const lineIdx = {};  // 节点 → 所在换行序（同 rank 第 0 行才允许垂直直连入边）

  // 2) 自底向上递归布局每个容器（相对坐标），并做本容器的车道分配。
  function layoutBox(id) {
    const nd = id == null ? null : st.ix.byId[id];
    const kidNodes = id == null
      ? (st.g.nodes || []).filter(n => !n.parent_id)
      : (st.ix.kids[id] || []);
    const expanded = kidNodes.length && (id == null || !st.collapsed.has(id));
    if (!expanded) {
      const m = measure(nd, kidNodes.length ? "head" : "leaf", st);
      return { w: m.w, h: m.h };
    }
    const ids = kidNodes.map(n => n.id);
    const size = {};
    ids.forEach(k => { size[k] = layoutBox(k); });
    const key = id == null ? "$root" : id;
    const edges = sib[key] || [];
    const rank = layerRanks(ids, edges);
    const spine = pickSpine(st, ids, edges, rank);
    const res = {};
    ids.forEach(i => { res[i] = (st.ix.byId[i] || {}).res || 0; });
    let resMin = Infinity;
    spine.forEach(i => { if (res[i] < resMin) resMin = res[i]; });
    if (!isFinite(resMin)) resMin = 0;

    // 分支 barycenter 定位用的前驱表（仅下行边）。
    const preds = {};
    ids.forEach(i => { preds[i] = []; });
    edges.forEach(e => {
      if (e.kind !== "residual" && rank[e.t] > rank[e.s]) preds[e.t].push(e.s);
    });

    // 「就近挂靠」：唯一数据源投影为**展开容器**的非脊柱节点（如 ds/aux 头），
    // 纵向对齐到容器内其真实源的下一层子行（二者同为该源的下一个计算），
    // 横向排在容器右缘外侧，位置即计算次序。容器折叠时自然退化为普通行内并列。
    const attach = {};
    edges.forEach(e => {
      if (e.kind === "residual" || rank[e.t] <= rank[e.s]) return;
      const n = e.t;
      if (spine.has(n)) return;
      if (attach[n] === undefined) attach[n] = { p: e.s, srcs: [] };
      else if (attach[n] && attach[n].p !== e.s) attach[n] = null;   // 多源不挂靠
      if (attach[n]) attach[n].srcs.push(veds[e.i].vs);
    });
    Object.keys(attach).forEach(n => {
      const a = attach[n];
      const del = () => { delete attach[n]; };
      if (!a) { del(); return; }
      const p = a.p, pc = cinfo[p];
      const pKids = st.ix.kids[p] || [];
      if (!pc || !pc.kidRank || !pKids.length || st.collapsed.has(p)) { del(); return; }
      const reps = a.srcs.map(sid => projTo(st, p, sid)).filter(x => x != null);
      if (!reps.length) { del(); return; }
      const nr = Math.max.apply(null, reps.map(rp => pc.kidRank[rp] || 0)) + 1;
      const band = pKids.map(k => k.id)
        .filter(k => pc.kidRank[k] === nr && geo[k]);
      if (!band.length) { del(); return; }   // 源已是容器内末层 → 留在普通行流
      let by0 = Infinity, by1 = -Infinity;
      band.forEach(k => {
        by0 = Math.min(by0, geo[k].y);
        by1 = Math.max(by1, geo[k].y + geo[k].h);
      });
      attach[n] = { p, y0: by0, y1: by1 };
    });

    const declOrder = {};
    ids.forEach((i, k) => { declOrder[i] = k; });
    const cx = {}, top = {}, rankLineCnt = {};
    let y = 0;
    const ranksSorted = [...new Set(ids.map(i => rank[i]))].sort((a, b) => a - b);
    ranksSorted.forEach(r => {
      const row = ids.filter(i => rank[i] === r);
      const sp = row.filter(i => spine.has(i));
      const rest = row.filter(i => !spine.has(i) && !attach[i]).sort((a, b) => {
        const bc = (n) => preds[n].length
          ? preds[n].reduce((s, p) => s + (cx[p] || 0), 0) / preds[n].length
          : Infinity;
        return (bc(a) - bc(b)) || (declOrder[a] - declOrder[b]);
      });
      // 换行：首行含脊柱节点，其余从脊柱右侧起排，超过 MAXROW 换行。
      const lines = [];
      let line = [], lineW = 0;
      if (sp.length) { line.push(sp[0]); lineW = size[sp[0]].w; }
      rest.forEach(i => {
        const w = size[i].w;
        if (line.length && lineW + LC.GAPX + w > LC.MAXROW) {
          lines.push(line); line = []; lineW = 0;
        }
        line.push(i);
        lineW += (line.length > 1 ? LC.GAPX : 0) + w;
      });
      if (line.length) lines.push(line);
      rankLineCnt[r] = lines.length;
      lines.forEach((ln, li) => {
        const lh = Math.max.apply(null, ln.map(i => size[i].h));
        ln.forEach(i => { lineIdx[i] = li; });
        if (li === 0 && sp.length) {
          const s0 = sp[0];
          cx[s0] = (res[s0] - resMin) * LC.INDENT;   // 分辨率缩进（中轴系）
          top[s0] = y + (lh - size[s0].h) / 2;
          let x = cx[s0] + size[s0].w / 2 + LC.GAPX;
          ln.forEach(i => {
            if (i === s0) return;
            cx[i] = x + size[i].w / 2;
            top[i] = y + (lh - size[i].h) / 2;
            x += size[i].w + LC.GAPX;
          });
        } else {
          // 无脊柱行（并联分支/换行续排）：按前驱 barycenter 居中。
          const totW = ln.reduce((s, i) => s + size[i].w, 0)
            + LC.GAPX * (ln.length - 1);
          let bx = 0, bn = 0;
          ln.forEach(i => preds[i].forEach(p => {
            if (cx[p] != null) { bx += cx[p]; bn++; }
          }));
          let x = (bn ? bx / bn : 0) - totW / 2;
          ln.forEach(i => {
            cx[i] = x + size[i].w / 2;
            top[i] = y + (lh - size[i].h) / 2;
            x += size[i].w + LC.GAPX;
          });
        }
        y += lh + LC.GAPY;
      });
    });
    // 挂靠节点定位：纵向对齐源的下一层带，横向从容器右缘外依次排开。
    const bandX = {};
    Object.keys(attach).forEach(n => {
      const a = attach[n];
      const bk = a.p + "|" + a.y0;
      if (bandX[bk] === undefined)
        bandX[bk] = cx[a.p] + size[a.p].w / 2 + LC.GAPX;
      top[n] = top[a.p] + (a.y0 + a.y1) / 2 - size[n].h / 2;
      cx[n] = bandX[bk] + size[n].w / 2;
      bandX[bk] += size[n].w + LC.GAPX;
    });
    let contentH = Math.max(0, y - LC.GAPY);
    Object.keys(attach).forEach(n => {
      contentH = Math.max(contentH, top[n] + size[n].h);
    });

    // 直连折线（源底→中线→目标顶）若穿过兄弟盒（如挂靠列内纵向穿框），
    // 也改走侧缘车道。用本容器兄弟级几何近似校验三段折线。
    const segHitsBox = (vert, coord, lo, hi, e) => {
      if (hi < lo) { const t = lo; lo = hi; hi = t; }
      return ids.some(b => {
        if (b === e.s || b === e.t) return false;
        const bx0 = cx[b] - size[b].w / 2, bx1 = cx[b] + size[b].w / 2;
        const by0 = top[b], by1 = top[b] + size[b].h;
        return vert
          ? (bx0 < coord && coord < bx1 && by0 < hi && lo < by1)
          : (by0 < coord && coord < by1 && bx0 < hi && lo < bx1);
      });
    };
    const directBlocked = (e) => {
      const y0 = top[e.s] + size[e.s].h, y1 = top[e.t];
      if (y1 < y0 + 4) return false;             // 走水平直连分支，另行处理
      const my = (y0 + y1) / 2;
      return segHitsBox(true, cx[e.s], y0, my, e)
          || segHitsBox(false, my, cx[e.s], cx[e.t], e)
          || segHitsBox(true, cx[e.t], my, y1, e);
    };

    // 3) 本容器侧缘车道：skip/反馈走左，residual 走右（用兄弟盒纵区间打包）。
    const leftIv = [], rightIv = [];
    edges.forEach(e => {
      if (!needSide(e, rank, rankLineCnt) && !directBlocked(e)) return;
      const y0 = Math.min(top[e.s] + size[e.s].h / 2, top[e.t] + size[e.t].h / 2);
      const y1 = Math.max(top[e.s] + size[e.s].h / 2, top[e.t] + size[e.t].h / 2);
      (e.kind === "residual" ? rightIv : leftIv).push({ key: e.i, y0, y1 });
    });
    const L = allocLanes(leftIv), R = allocLanes(rightIv);
    const laneAsg = Object.assign({}, L.asg, R.asg);

    // 4) 平移到容器内容区（预留左右车道 gutter），并记录相对几何。
    let minX = Infinity, maxX = -Infinity;
    ids.forEach(i => {
      minX = Math.min(minX, cx[i] - size[i].w / 2);
      maxX = Math.max(maxX, cx[i] + size[i].w / 2);
    });
    if (!isFinite(minX)) { minX = 0; maxX = 0; }
    const gutL = L.count * LC.LANE_W + (L.count ? LC.LANE_PAD : 0);
    const gutR = R.count * LC.LANE_W + (R.count ? LC.LANE_PAD : 0);
    const headM = id == null ? { w: 0, h: 0 } : measure(nd, "head", st);
    const padTop = id == null ? 0 : headM.h + LC.HEAD_GAP;
    const pad = id == null ? 4 : LC.PAD;
    const dx = pad + gutL - minX, dy = padTop + (id == null ? 0 : 0);
    ids.forEach(i => {
      geo[i] = { x: cx[i] - size[i].w / 2 + dx, y: top[i] + dy,
                 w: size[i].w, h: size[i].h };
    });
    const w = Math.max(pad * 2 + gutL + (maxX - minX) + gutR,
                       id == null ? 0 : headM.w + 4);
    const h = padTop + contentH + pad;
    cinfo[key] = { laneAsg, laneL: L.count, laneR: R.count,
                   pad, gutL, gutR, w, h, ids: ids.slice(), kidRank: rank };
    return { w, h };
  }

  function needSide(e, rank, rankLineCnt) {
    if (e.kind === "residual") return true;
    if (e.kind === "skip") return true;
    if ((rank[e.t] || 0) <= (rank[e.s] || 0)) return true;   // 反馈/同层
    return (rank[e.t] || 0) - (rank[e.s] || 0) > 1
        || (lineIdx[e.t] || 0) > 0 || (lineIdx[e.s] || 0) > 0
        || (rankLineCnt[rank[e.s]] || 1) > 1;   // 源层换行过 → 直连会穿行
  }

  const rootSize = layoutBox(null);

  // 5) 相对坐标 → 画布绝对坐标（父容器逐级累加）。
  const abs = {};
  function walkAbs(id, ox, oy, depth) {
    const kidNodes = id == null
      ? (st.g.nodes || []).filter(n => !n.parent_id)
      : (st.ix.kids[id] || []);
    const expanded = kidNodes.length && (id == null || !st.collapsed.has(id));
    if (!expanded) return;
    kidNodes.forEach(n => {
      const g = geo[n.id];
      if (!g) return;
      abs[n.id] = { x: ox + g.x, y: oy + g.y, w: g.w, h: g.h, depth,
                    isBox: (st.ix.kids[n.id] || []).length > 0 };
      walkAbs(n.id, ox + g.x, oy + g.y, depth + 1);
    });
  }
  walkAbs(null, 0, 0, 0);

  // 6) 连边路由（画布绝对坐标）。侧缘边端点取**实际可见盒**（可深入展开容器），
  // 车道取最近公共容器的 gutter；同盒多进/多出按序错开落点，避免线头重叠。
  const inCnt = {}, outCnt = {}, inSeen = {}, outSeen = {};
  const inGrp = {}, outGrp = {};   // 直连边扇出/扇入分组（用于按对侧 x 排序落点）
  veds.forEach((e, i) => {
    const pc = edgeCont[i];
    if (!pc.s || !pc.t || pc.s === pc.t) return;
    const key = pc.c == null ? "$root" : pc.c;
    const side = (cinfo[key] && cinfo[key].laneAsg[i] != null);
    const bt = abs[e.vt], bs = abs[e.vs];
    if (!bt || !bs) return;
    const dir = e.kind === "residual" ? "R" : "L";
    const kIn = e.vt + "|" + (side ? dir : "T");
    const kOut = e.vs + "|" + (side ? dir : "B");
    inCnt[kIn] = (inCnt[kIn] || 0) + 1;
    outCnt[kOut] = (outCnt[kOut] || 0) + 1;
    if (!side) {
      (outGrp[kOut] = outGrp[kOut] || []).push(i);
      (inGrp[kIn] = inGrp[kIn] || []).push(i);
    }
  });
  const stagger = (idx, n, span) => {
    if (n <= 1) return 0;
    const step = Math.min(10, span / (n + 1));
    return (idx - (n - 1) / 2) * step;
  };
  // 扇出落点按目标 x 排序、扇入落点按源 x 排序：出/入顺序与走向一致，
  // 从根上消除同源/同宿边在端点附近的相互交叉。
  const outIdx = {}, inIdx = {};
  Object.keys(outGrp).forEach(k => {
    outGrp[k].slice().sort((a, b) =>
      (abs[veds[a].vt].x + abs[veds[a].vt].w / 2)
      - (abs[veds[b].vt].x + abs[veds[b].vt].w / 2))
      .forEach((ei, j) => { outIdx[ei] = j; });
  });
  Object.keys(inGrp).forEach(k => {
    inGrp[k].slice().sort((a, b) =>
      (abs[veds[a].vs].x + abs[veds[a].vs].w / 2)
      - (abs[veds[b].vs].x + abs[veds[b].vs].w / 2))
      .forEach((ei, j) => { inIdx[ei] = j; });
  });

  // 水平段是否穿过同容器兄弟盒（穿框则改从上/下缘进出）。
  const blockedH = (ci, x0, x1, yy, skip) => {
    if (x1 < x0) { const t = x0; x0 = x1; x1 = t; }
    return (ci.ids || []).some(id => {
      if (skip.indexOf(id) >= 0) return false;
      const g = abs[id];
      return g && g.y < yy && yy < g.y + g.h && g.x < x1 && x0 < g.x + g.w;
    });
  };

  const routes = [];
  const folds = [];   // 层间折弯的直连边：先收集，再按互不交叉的次序统一分配中线 y
  veds.forEach((e, i) => {
    const pc = edgeCont[i];
    if (!pc.s || !pc.t || pc.s === pc.t) return;
    const key = pc.c == null ? "$root" : pc.c;
    const ci = cinfo[key];
    const A = abs[e.vs], B = abs[e.vt];
    if (!A || !B || !ci) return;
    const lane = ci.laneAsg[i];
    let pts;
    if (lane == null) {
      // 垂直直连：A 底 → B 顶，necessary 时在层间水平折弯。
      const kOut = e.vs + "|B", kIn = e.vt + "|T";
      const ax = A.x + A.w / 2 + stagger(outIdx[i] || 0, outCnt[kOut], A.w * 0.6);
      const bx = B.x + B.w / 2 + stagger(inIdx[i] || 0, inCnt[kIn], B.w * 0.6);
      const y0 = A.y + A.h, y1 = B.y;
      if (y1 < y0 + 4) {
        // 目标不在源下方（挂靠节点与源同带等）：改走水平直连（侧缘出入）。
        const toRight = B.x + B.w / 2 >= A.x + A.w / 2;
        const hx0 = toRight ? A.x + A.w : A.x;
        const hx1 = toRight ? B.x : B.x + B.w;
        const hy0 = A.y + A.h / 2, hy1 = B.y + B.h / 2;
        if (Math.abs(hy0 - hy1) < 1) pts = [[hx0, hy0], [hx1, hy1]];
        else {
          const mx = (hx0 + hx1) / 2;
          pts = [[hx0, hy0], [mx, hy0], [mx, hy1], [hx1, hy1]];
        }
      } else if (Math.abs(ax - bx) < 1) pts = [[ax, y0], [ax, y1]];
      else {
        // 折弯边：先占位，稍后按互不交叉次序统一分配中线 y。
        const r = { meta: e, pts: null };
        folds.push({ key, ax, bx, y0, y1, route: r, ord: i });
        routes.push(r);
        return;
      }
    } else {
      // 侧缘车道：左（skip/反馈）或右（residual）。
      const contAbs = pc.c == null ? { x: 0, w: rootSize.w }
                                   : abs[pc.c] || { x: 0, w: rootSize.w };
      const right = e.kind === "residual";
      const laneX = right
        ? contAbs.x + contAbs.w - ci.pad - (lane + 0.5) * LC.LANE_W
        : contAbs.x + ci.pad + ci.gutL - (lane + 0.5) * LC.LANE_W;
      const dir = right ? "R" : "L";
      const kOut = e.vs + "|" + dir, kIn = e.vt + "|" + dir;
      outSeen[kOut] = (outSeen[kOut] || 0) + 1;
      inSeen[kIn] = (inSeen[kIn] || 0) + 1;
      // 附加 lane 序微偏移：不同车道的边引出/引入水平段不共线，便于溯源。
      const lOff = ((lane % 3) - 1) * 5;
      const ay = A.y + A.h / 2 + lOff
        + stagger(outSeen[kOut] - 1, outCnt[kOut], A.h - 12);
      const by = B.y + B.h / 2 + lOff
        + stagger(inSeen[kIn] - 1, inCnt[kIn], B.h - 12);
      const ax = right ? A.x + A.w : A.x;
      const bx = right ? B.x + B.w : B.x;
      const skip = [e.vs, e.vt];
      pts = [];
      if (blockedH(ci, ax, laneX, ay, skip)) {
        // 源侧穿框：改从源底缘出，经行间空隙拐入车道。
        const ax2 = A.x + A.w / 2, gy = A.y + A.h + LC.GAPY / 2;
        pts.push([ax2, A.y + A.h], [ax2, gy], [laneX, gy]);
      } else {
        pts.push([ax, ay], [laneX, ay]);
      }
      if (blockedH(ci, laneX, bx, by, skip)) {
        // 目标侧穿框：改经行间空隙从目标顶缘进。
        const bx2 = B.x + B.w / 2, gy = B.y - LC.GAPY / 2;
        pts.push([laneX, gy], [bx2, gy], [bx2, B.y]);
      } else {
        pts.push([laneX, by], [bx, by]);
      }
    }
    routes.push({ meta: e, pts });
  });

  // 折弯边中线 y 分配：同容器内 [y0,y1] 纵向重叠的折弯边聚成一簇，
  // 依「水平段不得跨过他边竖直段」的成对约束拓扑排序，再在空隙带内
  // 均匀取 y——扇出/扇入/混合场景下折线互不交叉且不共线。
  const grpMap = {};
  folds.forEach(f => { (grpMap[f.key] = grpMap[f.key] || []).push(f); });
  Object.keys(grpMap).forEach(k => {
    const fs = grpMap[k].slice().sort((a, b) => a.y0 - b.y0 || a.ord - b.ord);
    let cluster = [], cMax = -Infinity;
    const flush = () => { if (cluster.length) assignFoldY(cluster); cluster = []; };
    fs.forEach(f => {
      if (cluster.length && f.y0 >= cMax) { flush(); cMax = -Infinity; }
      cluster.push(f);
      cMax = Math.max(cMax, f.y1);
    });
    flush();
  });
  function assignFoldY(fs) {
    const n = fs.length;
    const lo = (f) => Math.min(f.ax, f.bx), hi = (f) => Math.max(f.ax, f.bx);
    const after = fs.map(() => []), deg = fs.map(() => 0);
    for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) {
      if (a === b) continue;
      const e = fs[a], f = fs[b];
      // e 的水平段跨过 f 的上竖段（f.ax）→ f 须在 e 之上；跨过 f 的下竖段
      // （f.bx）→ e 须在 f 之上。
      if (lo(e) < f.ax && f.ax < hi(e)) { after[b].push(a); deg[a]++; }
      if (lo(e) < f.bx && f.bx < hi(e)) { after[a].push(b); deg[b]++; }
    }
    const order = [], q = [];
    for (let x = 0; x < n; x++) if (!deg[x]) q.push(x);
    q.sort((a, b) => fs[a].ord - fs[b].ord);
    while (q.length) {
      const u = q.shift();
      order.push(u);
      after[u].forEach(v => { if (--deg[v] === 0) q.push(v); });
      q.sort((a, b) => fs[a].ord - fs[b].ord);
    }
    for (let x = 0; x < n; x++)          // 约束成环时兜底保序
      if (order.indexOf(x) < 0) order.push(x);
    let gLo = Infinity, gHi = -Infinity;
    fs.forEach(f => { gLo = Math.min(gLo, f.y0); gHi = Math.max(gHi, f.y1); });
    order.forEach((u, rk) => {
      const f = fs[u];
      const t = gLo + (rk + 1) / (n + 1) * (gHi - gLo);
      const my = Math.max(f.y0 + 5, Math.min(f.y1 - 5, t));
      f.route.pts = [[f.ax, f.y0], [f.ax, my], [f.bx, my], [f.bx, f.y1]];
    });
  }

  // 兜底避框：跨容器边的竖直段可能纵穿无关实心盒（叶子/折叠容器；展开
  // 容器只是虚线外框，穿过无妨）。逐段检测，穿框处沿盒近侧缘绕行。
  const hasKids = {};
  Object.keys(abs).forEach(id => {
    const p = (st.ix.byId[id] || {}).parent_id;
    if (p) hasKids[p] = true;
  });
  const solid = Object.keys(abs).filter(id => !hasKids[id]);
  routes.forEach(r => {
    if (!r.pts) return;
    const out = [r.pts[0]];
    for (let s = 1; s < r.pts.length; s++) {
      const [x0, y0] = out[out.length - 1], [x1, y1] = r.pts[s];
      if (Math.abs(x1 - x0) < 0.5 && Math.abs(y1 - y0) > 1) {
        const lo = Math.min(y0, y1), hi = Math.max(y0, y1);
        const hit = solid
          .filter(id => id !== r.meta.vs && id !== r.meta.vt)
          .map(id => abs[id])
          .filter(g => g.x + 1 < x0 && x0 < g.x + g.w - 1
                    && g.y + 1 < hi && lo < g.y + g.h - 1)
          .sort((a, b) => (y1 > y0 ? a.y - b.y : b.y - a.y));
        hit.forEach(g => {
          const side = (x0 - g.x < g.x + g.w - x0) ? g.x - 7 : g.x + g.w + 7;
          const ya = y1 > y0 ? g.y - 6 : g.y + g.h + 6;
          const yb = y1 > y0 ? g.y + g.h + 6 : g.y - 6;
          out.push([x0, ya], [side, ya], [side, yb], [x0, yb]);
        });
      }
      out.push([x1, y1]);
    }
    r.pts = out;
  });

  return { geo: abs, routes, W: Math.ceil(rootSize.w) + 8,
           H: Math.ceil(rootSize.h) + 8 };
}

// ---------------- 布局 + 绘制 ----------------
function relayout(st) {
  const tops = (st.g.nodes || []).filter(n => !n.parent_id);
  const canvas = st.flowEl.querySelector(".canvas");
  if (!tops.length) {
    canvas.innerHTML = "";
    canvas.appendChild(el("div", "empty", "（此视图无节点）"));
    st.laidOut = true;
    return;
  }
  const eds = visEdges(st);
  const lay = computeLayout(st, eds);
  paint(st, lay);
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

function paint(st, lay) {
  const canvas = st.flowEl.querySelector(".canvas");
  canvas.innerHTML = "";
  const svg = svgNS("svg");
  svg.setAttribute("class", "edges");
  svg.innerHTML = svgDefs();
  const svgTop = svgNS("svg");
  svgTop.setAttribute("class", "edges edges-top");
  svgTop.innerHTML = svgDefs();
  canvas.appendChild(svg);

  // 节点绝对定位（布局已给画布绝对坐标）。容器 z 按深度递增（外层最低），
  // 叶卡恒在容器之上（z10），连线层再往上（z20/30）。
  const nodeEls = {};
  Object.entries(lay.geo).forEach(([id, gm]) => {
    const nd = st.ix.byId[id];
    const isBox = gm.isBox;
    const e = isBox ? stageBox(nd, st, st.collapsed.has(id))
                    : leafCard(nd, st);
    e.style.left = gm.x + "px";
    e.style.top = gm.y + "px";
    e.style.width = gm.w + "px";
    e.style.height = gm.h + "px";
    e.style.zIndex = isBox ? String(1 + gm.depth) : "10";
    canvas.appendChild(e);
    nodeEls[id] = e;
  });
  canvas.appendChild(svgTop);
  const W = lay.W, H = lay.H;
  canvas.style.width = W + "px";
  canvas.style.height = H + "px";
  [svg, svgTop].forEach(s => {
    s.setAttribute("width", W); s.setAttribute("height", H);
    s.style.width = W + "px"; s.style.height = H + "px";
  });

  // 连线：结构化布局已给出每条边的正交折点（画布坐标系）。
  const edgeEls = [];
  lay.routes.forEach(({ meta, pts }) => {
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
    // 透明宽命中层：hover 高亮单条边 + tooltip 显示端点，便于溯源。
    const hit = svgNS("path");
    hit.setAttribute("d", path.getAttribute("d"));
    hit.setAttribute("fill", "none");
    hit.setAttribute("stroke", "rgba(0,0,0,0)");
    hit.setAttribute("stroke-width", "11");
    hit.style.pointerEvents = "stroke";
    const nameOf = (id) => (st.ix.byId[id] || {}).label || id;
    const tip = svgNS("title");
    tip.textContent = nameOf(meta.vs) + " \u2192 " + nameOf(meta.vt)
      + (meta.kind !== "forward" ? "  [" + meta.kind + "]" : "");
    hit.appendChild(tip);
    svgTop.appendChild(hit);
    const it = { meta, path, txt, home, hit, baseW: String(style.width) };
    hit.addEventListener("mouseenter", () => hoverEdge(st, it, true));
    hit.addEventListener("mouseleave", () => hoverEdge(st, it, false));
    edgeEls.push(it);
  });
  st.rc = { nodeEls, edgeEls, svg, svgTop, canvas };
  applyFocus(st);
  if (st === activeState()) refreshNav(st);
}

// —— 锚点导航：按计算次序（y 坐标）列出顶层节点与已展开容器的直接子模块，
// 点击平滑滚动到目标并闪烁提示。纯数据驱动，与聚焦可达层级一致。 ——
function refreshNav(st) {
  const nav = document.getElementById("nav");
  const body = nav.querySelector(".nav-body");
  body.innerHTML = "";
  const rc = st.rc;
  if (!rc) { nav.style.display = "none"; return; }
  const geo = {};
  Object.keys(rc.nodeEls).forEach(id => {
    const e = rc.nodeEls[id];
    geo[id] = { y: parseFloat(e.style.top), x: parseFloat(e.style.left) };
  });
  const items = [];
  Object.keys(geo).forEach(id => {
    const nd = st.ix.byId[id] || {};
    if (nd.kind === "merge") return;
    if (!nd.parent_id) items.push({ id, lv: 0 });
    else if (!(st.ix.byId[nd.parent_id] || {}).parent_id
             && geo[nd.parent_id]) items.push({ id, lv: 1 });
  });
  items.sort((a, b) => geo[a.id].y - geo[b.id].y || geo[a.id].x - geo[b.id].x
                     || (a.lv - b.lv));
  if (items.length < 4) { nav.style.display = "none"; return; }
  nav.style.display = "";
  items.forEach(({ id, lv }) => {
    const nd = st.ix.byId[id] || {};
    const it = el("div", "nav-item" + (lv ? " lv1" : ""), nd.label || id);
    it.title = nd.label || id;
    it.addEventListener("click", () => {
      const tgt = (st.rc || {}).nodeEls && st.rc.nodeEls[id];
      if (!tgt) return;
      tgt.scrollIntoView({ behavior: "smooth", block: "center",
                           inline: "center" });
      tgt.classList.remove("nav-flash");
      void tgt.offsetWidth;
      tgt.classList.add("nav-flash");
    });
    body.appendChild(it);
  });
}

// —— 悬停溯源：单条边加粗上浮、其余边淡出；离开后恢复聚焦态 ——
function hoverEdge(st, it, on) {
  const rc = st.rc;
  if (!rc) return;
  if (!on) {
    rc.edgeEls.forEach(e => {
      e.path.setAttribute("stroke-width", e.baseW);
      e.path.style.opacity = "";
      if (e.txt) e.txt.style.opacity = "";
      if (e.path.parentNode !== e.home) e.home.appendChild(e.path);
      if (e.txt && e.txt.parentNode !== e.home) e.home.appendChild(e.txt);
    });
    applyFocus(st);
    return;
  }
  rc.edgeEls.forEach(e => {
    const h = e === it;
    e.path.style.opacity = h ? "1" : "0.10";
    if (e.txt) e.txt.style.opacity = h ? "1" : "0.10";
    if (h) {
      e.path.setAttribute("stroke-width", String(parseFloat(e.baseW) + 1.4));
      rc.svgTop.appendChild(e.path);
      if (e.txt) rc.svgTop.appendChild(e.txt);
    }
  });
}

// —— 聚焦高亮：单击顶层模块后，仅其本身+直接邻居+相连边醒目，其余淡出 ——
// 通用、数据驱动（基于折叠上卷后的可见邻接）：相连边几何不变（仍是车道路由），
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
                 laidOut: false, rc: null, measCache: {} };
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
  else refreshNav(st);
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
<div class="main">
<div class="nav" id="nav" style="display:none">
  <div class="nav-head" onclick="this.parentNode.classList.toggle('closed')">结构导航</div>
  <div class="nav-body"></div>
</div>
<div class="canvas-wrap" id="canvas-wrap"></div>
</div>
<div class="drawer" id="drawer">
  <div class="drawer-head">
    <h2 id="drawer-title">Details</h2>
    <button class="x" onclick="closeDrawer()">&times;</button>
  </div>
  <div class="drawer-body" id="drawer-body"></div>
</div>
<script>{js}</script>
</body>
</html>
"""


__all__ = ["render_html"]
