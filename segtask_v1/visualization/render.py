"""IR → 自包含 HTML 渲染器（零外部依赖）。

把若干 ``VisGraph`` 序列化为内嵌 JSON，配一段原生 HTML/CSS/JS：
* 顶部标签页切换 Data / Model / Predict；
* 纵向自上而下流式布局，stage 用可折叠大框包裹子框；
* SVG overlay 按实际 DOM 矩形绘制箭头（折叠/展开/切页/缩放后自动重算）；
* 双击任意框 → 右侧详情抽屉展示完整参数。

不使用 mermaid / d3 / graphviz 等任何第三方库，单文件可离线打开。
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
  --c-data: #0e7490; --c-data-bg: #ecfeff;
  --c-process: #b45309; --c-process-bg: #fffbeb;
  --c-input: #4338ca; --c-input-bg: #eef2ff;
  --c-stage: #334155; --c-stage-bg: #f1f5f9;
  --c-conv: #1d4ed8; --c-conv-bg: #eff6ff;
  --c-norm: #047857; --c-norm-bg: #ecfdf5;
  --c-act: #a21caf; --c-act-bg: #fdf4ff;
  --c-op: #475569; --c-op-bg: #f8fafc;
  --c-head: #be185d; --c-head-bg: #fdf2f8;
  --c-output: #15803d; --c-output-bg: #f0fdf4;
  --c-loss: #b91c1c; --c-loss-bg: #fef2f2;
  --c-model: #6d28d9; --c-model-bg: #f5f3ff;
}
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    "Helvetica Neue", Arial, "PingFang SC", "Microsoft YaHei", sans-serif;
  background: var(--bg); color: var(--ink); font-size: 13px;
}
header {
  background: var(--panel); border-bottom: 1px solid var(--line);
  padding: 12px 20px; position: sticky; top: 0; z-index: 30;
}
header h1 { margin: 0 0 6px; font-size: 16px; font-weight: 650; }
.meta { display: flex; flex-wrap: wrap; gap: 6px 14px; color: var(--muted);
  font-size: 12px; }
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
.canvas { position: relative; }
.flow { display: none; }
.flow.active { display: block; }
svg.edges { position: absolute; left: 0; top: 0; width: 100%; height: 100%;
  pointer-events: none; z-index: 1; overflow: visible; }
.col { position: relative; z-index: 2; display: flex; flex-direction: column;
  align-items: center; gap: 22px; }
.children { display: flex; flex-direction: column; align-items: center;
  gap: 16px; padding: 6px 14px 14px; }
.children.collapsed { display: none; }

/* leaf node card */
.node {
  background: var(--panel); border: 1px solid var(--line); border-left-width: 4px;
  border-radius: 8px; padding: 8px 12px; min-width: 200px; max-width: 460px;
  box-shadow: 0 1px 2px rgba(15,23,42,.06); cursor: pointer; position: relative;
}
.node:hover { box-shadow: 0 2px 10px rgba(37,99,235,.18); }
.node .title { font-weight: 650; font-size: 12.5px; display: flex;
  justify-content: space-between; gap: 10px; align-items: baseline; }
.node .badge { font-size: 10px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .04em; opacity: .8; }
.node .kv { margin-top: 4px; font-size: 11.5px; color: var(--muted);
  line-height: 1.5; }
.node .kv span { color: var(--ink); }
.node .hint { position: absolute; right: 8px; bottom: 4px; font-size: 9.5px;
  color: var(--muted); opacity: 0; }
.node:hover .hint { opacity: .7; }

/* stage container */
.stage {
  background: var(--c-stage-bg); border: 1px dashed var(--c-stage);
  border-radius: 12px; min-width: 280px; max-width: 1024px;
}
.stage > .stage-head {
  display: flex; align-items: center; gap: 8px; padding: 8px 12px;
  cursor: pointer; font-weight: 700; color: var(--c-stage); user-select: none;
}
.stage > .stage-head .caret { transition: transform .15s; font-size: 11px; }
.stage > .stage-head.collapsed .caret { transform: rotate(-90deg); }
.stage > .stage-head .s-kv { color: var(--muted); font-weight: 500;
  font-size: 11px; margin-left: auto; }

/* row: side-by-side grouping (并联分支 / 多分辨率输入) */
.row { display: flex; flex-direction: row; align-items: flex-start;
  justify-content: center; gap: 26px; flex-wrap: wrap; }

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
.k-head   { border-left-color: var(--c-head);   background: var(--c-head-bg); }
.k-output { border-left-color: var(--c-output); background: var(--c-output-bg); }
.k-loss   { border-left-color: var(--c-loss);   background: var(--c-loss-bg); }
.k-model  { border-left-color: var(--c-model);  background: var(--c-model-bg);
  min-width: 300px; }
.badge.k-data{color:var(--c-data)} .badge.k-process{color:var(--c-process)}
.badge.k-input{color:var(--c-input)} .badge.k-conv{color:var(--c-conv)}
.badge.k-norm{color:var(--c-norm)} .badge.k-act{color:var(--c-act)}
.badge.k-op{color:var(--c-op)} .badge.k-head{color:var(--c-head)}
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
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.legend { display: flex; flex-wrap: wrap; gap: 10px; padding: 6px 20px 0;
  color: var(--muted); font-size: 11px; }
.legend .dot { display: inline-block; width: 10px; height: 10px;
  border-radius: 3px; margin-right: 4px; vertical-align: middle; }
.empty { color: var(--muted); padding: 40px; text-align: center; }
"""

# ---------------------------------------------------------------------------
# JS —— DOM 树构建 + SVG 箭头叠加 + 折叠 + 详情抽屉。
# ---------------------------------------------------------------------------
_JS = r"""
const DATA = __DATA__;
const KINDS = ["data","process","input","conv","norm","act","op","head",
  "output","loss","model"];
// 连边线型/配色：forward 主流（实线灰）、skip 跳连（实线蓝，走侧缘嵌套车道）、
// residual 残差（琥珀虚线，就近局部弧）。
const EDGE_STYLE = {
  forward:  { color: "#94a3b8", width: "1.6", dash: null,  marker: "arrow" },
  skip:     { color: "#2563eb", width: "1.8", dash: null,  marker: "arrow-skip" },
  residual: { color: "#d97706", width: "1.8", dash: "3 3", marker: "arrow-res" },
};

function el(tag, cls, txt) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (txt != null) e.textContent = txt;
  return e;
}

function buildLeaf(node) {
  const card = el("div", "node k-" + node.kind);
  card.id = "n-" + node.id;
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
  card.appendChild(el("div", "hint", "double-click for details"));
  return card;
}

function buildStage(node, childrenOf) {
  const box = el("div", "stage k-stage");
  box.id = "n-" + node.id;
  box.dataset.nodeId = node.id;
  const head = el("div", "stage-head" + (node.collapsed ? " collapsed" : ""));
  head.appendChild(el("span", "caret", "\u25BC"));
  head.appendChild(el("span", null, node.label));
  const sk = Object.entries(node.key_info || {})
    .map(([k, v]) => k + " " + v).join("  ·  ");
  if (sk) head.appendChild(el("span", "s-kv", sk));
  box.appendChild(head);
  const body = el("div", "children" + (node.collapsed ? " collapsed" : ""));
  layoutInto(body, childrenOf[node.id] || [], childrenOf);
  box.appendChild(body);
  head.addEventListener("click", (ev) => {
    ev.stopPropagation();
    body.classList.toggle("collapsed");
    head.classList.toggle("collapsed");
    requestAnimationFrame(drawEdges);
  });
  return box;
}

// Lay a set of sibling nodes into `container`. If their `rank` values differ,
// group same-rank siblings into a horizontal `.row` (并联分支 side-by-side);
// otherwise fall back to vertical flow with consecutive `input` nodes rowed.
function layoutInto(container, nodes, childrenOf) {
  if (!nodes.length) return;
  const ranks = new Set(nodes.map(n => n.rank || 0));
  if (ranks.size > 1) {
    const arr = nodes.slice().sort((a, b) => (a.rank || 0) - (b.rank || 0));
    let i = 0;
    while (i < arr.length) {
      let j = i;
      while (j < arr.length && (arr[j].rank || 0) === (arr[i].rank || 0)) j++;
      if (j - i > 1) {
        const row = el("div", "row");
        // 同行内 head（如 deep-supervision 头）靠左摆，使其对应 decoder level 的
        // 右侧边留出给 encoder→decoder 跳连进框（不被旁分支遮挡）。
        const seg = arr.slice(i, j).sort((a, b) =>
          (a.kind === "head" ? 0 : 1) - (b.kind === "head" ? 0 : 1));
        for (const nd of seg) row.appendChild(render(nd, childrenOf));
        container.appendChild(row);
      } else {
        container.appendChild(render(arr[i], childrenOf));
      }
      i = j;
    }
    return;
  }
  // single rank → preserve order; row consecutive multi-resolution inputs
  let i = 0;
  while (i < nodes.length) {
    if (nodes[i].kind === "input") {
      let j = i;
      while (j < nodes.length && nodes[j].kind === "input") j++;
      if (j - i > 1) {
        const row = el("div", "row");
        for (let k = i; k < j; k++) row.appendChild(render(nodes[k], childrenOf));
        container.appendChild(row);
        i = j; continue;
      }
    }
    container.appendChild(render(nodes[i], childrenOf));
    i++;
  }
}

function render(node, childrenOf) {
  const isStage = (childrenOf[node.id] || []).length > 0;
  const wrap = el("div", "node-wrap");
  wrap.style.position = "relative";
  const elx = isStage ? buildStage(node, childrenOf) : buildLeaf(node);
  // double-click → detail drawer (works for both leaf & stage)
  elx.addEventListener("dblclick", (ev) => { ev.stopPropagation(); openDrawer(node); });
  wrap.appendChild(elx);
  return wrap;
}

let CUR = null;
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
function closeDrawer() { document.getElementById("drawer").classList.remove("open"); }

// Resolve a node id to its nearest VISIBLE DOM element (walk up collapsed stages).
function visibleAnchor(nodeId) {
  let e = document.getElementById("n-" + nodeId);
  if (!e) return null;
  // if inside a collapsed .children, climb to the owning stage box
  let cur = e;
  while (cur) {
    const par = cur.parentElement;
    if (par && par.classList.contains("children")
        && par.classList.contains("collapsed")) {
      // owning stage = the .stage that contains this children body
      const stage = par.closest(".stage");
      e = stage || e;
      cur = stage;
      continue;
    }
    cur = par;
  }
  return e;
}

function drawEdges() {
  const flow = document.querySelector(".flow.active");
  if (!flow) return;
  const svg = flow.querySelector("svg.edges");
  const canvas = flow.querySelector(".canvas");
  svg.innerHTML = svgDefs();
  const cr = canvas.getBoundingClientRect();
  const g = DATA.flows[flow.dataset.flow] || {};
  const edges = g.edges || [];
  const topIds = new Set((g.nodes || [])
    .filter(n => !n.parent_id).map(n => n.id));
  const seen = new Set();

  // 内容横向边界（所有可见框的并集）：外缘车道据此排到所有框之外，杜绝压框。
  let contentL = Infinity, contentR = -Infinity;
  flow.querySelectorAll(".node, .stage").forEach(elx => {
    const r = elx.getBoundingClientRect();
    if (r.width === 0) return;
    contentL = Math.min(contentL, r.left - cr.left);
    contentR = Math.max(contentR, r.right - cr.left);
  });
  if (!isFinite(contentL)) { contentL = 0; contentR = canvas.scrollWidth; }

  // 预解析每条边的几何 + 去重 + 归入侧边车道。
  //  band 'R'：顶层 encoder→decoder 跳连（同心嵌套环，最长跨度最外圈）。
  //  band 'L'：顶层 head→loss 汇聚束（ds 头上移后其 loss 边变长，单列左侧避免压框）。
  const items = [];
  edges.forEach(ed => {
    const a = visibleAnchor(ed.src), b = visibleAnchor(ed.dst);
    if (!a || !b || a === b) return;
    const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect();
    const cx1 = ra.left + ra.width / 2 - cr.left, yb1 = ra.bottom - cr.top;
    const cx2 = rb.left + rb.width / 2 - cr.left, yt2 = rb.top - cr.top;
    const kind = ed.kind || "forward";
    const key = Math.round(cx1) + "," + Math.round(yb1) + ">" +
                Math.round(cx2) + "," + Math.round(yt2) + ">" + kind;
    if (seen.has(key)) return; seen.add(key);
    let band = null;
    if (topIds.has(ed.src) && topIds.has(ed.dst)) {
      // 仅把"跨多行"的 head→loss（deep-supervision 头）引到左侧外缘束；
      // 与 loss 相邻的 seg/aux 头照常竖向短接，避免无谓绕到最左、标签挤叠。
      if (ed.dst === "loss") { if (yt2 - yb1 > 70) band = "L"; }
      else if (kind === "skip") band = "R";
    }
    items.push({ ed, ra, rb, cx1, yb1, cx2, yt2, kind, band });
  });

  // 车道分配：同侧按纵向跨度升序 → 跨度最大者拿最高车道（最外圈），
  // 从而嵌套区间同心不交叉（如 enc0→dec3 包含 enc1→dec2）。
  const STEP = 22, GAP = 26;
  ["R", "L"].forEach(side => {
    const grp = items.filter(it => it.band === side);
    grp.sort((p, q) => Math.abs(p.yt2 - p.yb1) - Math.abs(q.yt2 - q.yb1));
    grp.forEach((it, i) => { it.lane = i; });
  });
  let maxX = contentR;

  items.forEach(it => {
    const { ed, ra, rb, cx1, yb1, cx2, yt2, kind } = it;
    const style = EDGE_STYLE[kind] || EDGE_STYLE.forward;
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    let d, lx, ly;
    if (it.band === "R" || it.band === "L") {
      // 从两端框侧沿引出，绕到内容外缘的车道线（右/左）再折回，全程在框外。
      const right = it.band === "R";
      const x1 = (right ? ra.right : ra.left) - cr.left;
      const x2 = (right ? rb.right : rb.left) - cr.left;
      const y1 = ra.top + ra.height / 2 - cr.top;
      const y2 = rb.top + rb.height / 2 - cr.top;
      const sx = right ? (contentR + GAP + it.lane * STEP)
                       : Math.max(8, contentL - GAP - it.lane * STEP);
      maxX = Math.max(maxX, sx);
      d = `M ${x1} ${y1} C ${sx} ${y1}, ${sx} ${y2}, ${x2} ${y2}`;
      lx = right ? sx + 4 : sx - 4; ly = (y1 + y2) / 2;
    } else if (kind === "residual" || yt2 < yb1 - 2) {
      // 残差捷径 / 回流：就近的局部右侧小弧（不进外缘车道）。
      const off = 44 + Math.abs(cx2 - cx1) * 0.12 + (kind === "residual" ? 16 : 0);
      const sx = Math.max(cx1, cx2) + off;
      maxX = Math.max(maxX, sx);
      const up = yt2 < yb1 - 2;
      const ya = up ? (yb1 - ra.height / 2) : yb1;
      const yb = up ? (yt2 + rb.height / 2) : yt2;
      d = `M ${cx1} ${ya} C ${sx} ${ya}, ${sx} ${yb}, ${cx2} ${yb}`;
      lx = sx + 4; ly = (ya + yb) / 2;
    } else {
      // 前向相邻：竖向贝塞尔。
      const my = (yb1 + yt2) / 2;
      d = `M ${cx1} ${yb1} C ${cx1} ${my}, ${cx2} ${my}, ${cx2} ${yt2}`;
      lx = (cx1 + cx2) / 2 + 6; ly = my - 2;
    }
    path.setAttribute("d", d);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", style.color);
    path.setAttribute("stroke-width", style.width);
    if (style.dash) path.setAttribute("stroke-dasharray", style.dash);
    path.setAttribute("marker-end", "url(#" + style.marker + ")");
    svg.appendChild(path);
    if (ed.label) {
      const tx = document.createElementNS("http://www.w3.org/2000/svg", "text");
      tx.setAttribute("x", lx);
      tx.setAttribute("y", ly);
      tx.setAttribute("fill", style.color);
      tx.setAttribute("font-size", "10.5");
      tx.setAttribute("font-weight", kind === "forward" ? "400" : "700");
      if (it.band === "L") tx.setAttribute("text-anchor", "end");
      tx.textContent = ed.label;
      svg.appendChild(tx);
    }
  });
  // size svg to canvas（含外缘车道）
  svg.setAttribute("width", Math.max(canvas.scrollWidth, maxX + 12));
  svg.setAttribute("height", canvas.scrollHeight);
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

function buildFlow(flowKey) {
  const g = DATA.flows[flowKey];
  const flow = el("div", "flow");
  flow.dataset.flow = flowKey;
  const canvas = el("div", "canvas");
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "edges");
  canvas.appendChild(svg);
  const col = el("div", "col");
  canvas.appendChild(col);
  flow.appendChild(canvas);

  if (!g || !g.nodes.length) {
    col.appendChild(el("div", "empty", "（此视图无节点）"));
    return flow;
  }
  // index children by parent
  const childrenOf = {};
  const byId = {};
  g.nodes.forEach(n => { byId[n.id] = n; });
  g.nodes.forEach(n => {
    const p = n.parent_id;
    if (p != null && p !== "") { (childrenOf[p] = childrenOf[p] || []).push(n); }
  });
  // top-level layout: rank-aware (并联横排) with multi-res input fallback
  const tops = g.nodes.filter(n => !n.parent_id);
  layoutInto(col, tops, childrenOf);
  return flow;
}

function init() {
  const order = DATA.order.filter(k => DATA.flows[k]);
  const tabsEl = document.getElementById("tabs");
  const wrap = document.getElementById("canvas-wrap");
  order.forEach((k, idx) => {
    const tab = el("div", "tab" + (idx === 0 ? " active" : ""), DATA.titles[k] || k);
    tab.dataset.flow = k;
    tab.addEventListener("click", () => activate(k));
    tabsEl.appendChild(tab);
    const flow = buildFlow(k);
    if (idx === 0) flow.classList.add("active");
    wrap.appendChild(flow);
  });
  renderMeta(order[0]);
  requestAnimationFrame(drawEdges);
}
function activate(k) {
  document.querySelectorAll(".tab").forEach(t =>
    t.classList.toggle("active", t.dataset.flow === k));
  document.querySelectorAll(".flow").forEach(f =>
    f.classList.toggle("active", f.dataset.flow === k));
  renderMeta(k);
  requestAnimationFrame(drawEdges);
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

window.addEventListener("resize", () => requestAnimationFrame(drawEdges));
window.addEventListener("keydown", e => { if (e.key === "Escape") closeDrawer(); });
document.addEventListener("DOMContentLoaded", init);
"""


def _legend_html() -> str:
    items = [
        ("data", "数据"), ("process", "处理"), ("input", "输入"),
        ("stage", "阶段"), ("conv", "卷积"), ("norm", "归一化"),
        ("act", "激活"), ("op", "算子"), ("head", "头"),
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
        ("#2563eb", "solid", "跳连 skip（侧缘嵌套）"),
        ("#d97706", "dashed", "残差 residual"),
    ]
    spans.append('<span style="width:1px;height:14px;background:var(--line)">'
                 '</span>')
    for color, style, name in edges:
        spans.append(
            f'<span><i class="eline" style="border-top-color:{color};'
            f'border-top-style:{style}"></i>{html.escape(name)}</span>')
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
<script>{js}</script>
</body>
</html>
"""


__all__ = ["render_html"]
