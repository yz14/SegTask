"""仪表盘静态资源：CSS + 通用 SVG 折线图渲染 JS（零外部依赖）。

JS 接收一份已「渲染就绪」的 payload（见 ``charts.py``），通用地绘制折线图、
坐标轴、网格、悬停读数、图例开关、best-epoch 标记、逐类网格、best 指标卡片与
对比表，并支持训练中定时自动重载。配色 / 字体 token 与 ``visualization/render.py``
保持一致，全局视觉统一。
"""

from __future__ import annotations

CSS = """
:root {
  --bg: #f6f7f9; --panel: #ffffff; --ink: #1f2430; --muted: #6b7280;
  --line: #c7ccd6; --grid: #eef0f3; --accent: #2563eb; --best: #b45309;
  --good-bg: #f0fdf4; --good-ink: #15803d;
  --font-sans: "Inter", "SF Pro Text", -apple-system, BlinkMacSystemFont,
    "Segoe UI", Roboto, "Helvetica Neue", Arial, "PingFang SC",
    "Microsoft YaHei", sans-serif;
  --font-mono: "JetBrains Mono", "SF Mono", SFMono-Regular, ui-monospace,
    "Cascadia Code", Menlo, Consolas, "Liberation Mono", monospace;
}
* { box-sizing: border-box; }
html, body { margin: 0; }
body {
  font-family: var(--font-sans); background: var(--bg); color: var(--ink);
  font-size: 13px; line-height: 1.45;
  -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
  text-rendering: optimizeLegibility;
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
.live { color: var(--accent); font-weight: 600; }
.live .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  background: var(--accent); margin-right: 5px; vertical-align: middle;
  animation: pulse 1.6s ease-in-out infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .25; } }
.wrap { padding: 16px 20px 60px; max-width: 1400px; margin: 0 auto; }

/* sections + responsive panel grid */
.section { margin: 22px 0 8px; }
.section:first-of-type { margin-top: 4px; }
.section h2 { font-size: 11.5px; font-weight: 700; letter-spacing: .05em;
  text-transform: uppercase; color: var(--muted); margin: 0 0 8px;
  display: flex; align-items: center; gap: 12px; }
.section h2::after { content: ""; flex: 1; height: 1px; background: var(--line); }
.panel-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px; }
.panel-grid .panel { margin-bottom: 0; }
.span-full { grid-column: 1 / -1; }
@media (max-width: 880px) {
  .panel-grid { grid-template-columns: 1fr; }
  .span-full { grid-column: auto; }
}

/* best card */
.bestcard { background: var(--good-bg); border: 1px solid #bbf7d0;
  border-radius: 12px; padding: 14px 16px; margin-bottom: 18px; }
.bestcard .hl { font-size: 14px; font-weight: 700; color: var(--good-ink);
  margin-bottom: 10px; }
.bestcard .hl b { font-family: var(--font-mono); font-size: 18px; }
.bestcard .grid { display: grid; gap: 6px 18px;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); }
.bestcard .m { display: flex; justify-content: space-between; gap: 10px;
  border-bottom: 1px solid #dcfce7; padding: 3px 0; }
.bestcard .m .k { color: var(--muted); }
.bestcard .m .v { font-family: var(--font-mono); font-variant-numeric: tabular-nums;
  color: var(--ink); }

/* panels */
.panel { background: var(--panel); border: 1px solid var(--line);
  border-radius: 12px; padding: 12px 14px 8px; margin-bottom: 16px;
  box-shadow: 0 1px 2px rgba(15,23,42,.05); }
.panel > .ptitle { font-weight: 650; font-size: 13px; margin-bottom: 6px;
  display: flex; align-items: center; gap: 10px; }
.panel .ctrl { margin-left: auto; font-size: 11px; color: var(--muted);
  display: flex; gap: 8px; align-items: center; }
.panel .ctrl label { cursor: pointer; user-select: none; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 12px; margin: 2px 0 4px; }
.legend .it { display: inline-flex; align-items: center; gap: 5px; cursor: pointer;
  font-size: 11.5px; color: var(--ink); padding: 1px 4px; border-radius: 5px; }
.legend .it.off { color: var(--muted); opacity: .5; text-decoration: line-through; }
.legend .sw { width: 11px; height: 11px; border-radius: 3px; display: inline-block; }
.chart { position: relative; }
.chart svg { display: block; width: 100%; height: auto; }
.gridwrap { display: grid; gap: 12px;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); }
.gridwrap .sub { border: 1px solid var(--grid); border-radius: 8px; padding: 6px; }
.gridwrap .sub .st { font-size: 11.5px; font-weight: 600; color: var(--muted);
  margin-bottom: 2px; }
.tt { position: absolute; pointer-events: none; background: #0f172a; color: #fff;
  font-size: 11px; padding: 6px 8px; border-radius: 6px; white-space: nowrap;
  font-family: var(--font-mono); font-variant-numeric: tabular-nums; opacity: 0;
  transform: translate(-50%, -110%); transition: opacity .08s; z-index: 5;
  box-shadow: 0 4px 14px rgba(0,0,0,.25); }
.tt b { color: #93c5fd; font-weight: 600; }
.tt .row { display: flex; gap: 8px; justify-content: space-between; }
.tt .row .sw { width: 8px; height: 8px; border-radius: 2px; display: inline-block;
  margin-right: 4px; }

/* compare table */
.cmp-table { width: 100%; border-collapse: collapse; font-size: 12px;
  background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
  overflow: hidden; margin-bottom: 16px; }
.cmp-table th, .cmp-table td { padding: 7px 10px; text-align: left;
  border-bottom: 1px solid var(--grid); font-variant-numeric: tabular-nums; }
.cmp-table th { background: #f1f5f9; font-weight: 600; color: var(--muted);
  position: sticky; }
.cmp-table td:first-child, .cmp-table th:first-child { font-weight: 600; }
.cmp-table .runsw { width: 10px; height: 10px; border-radius: 3px;
  display: inline-block; margin-right: 6px; vertical-align: middle; }
.empty { color: var(--muted); padding: 48px; text-align: center; }
"""


JS = r"""
const P = __PAYLOAD__;
const SVGNS = "http://www.w3.org/2000/svg";

function el(tag, cls, txt) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (txt != null) e.textContent = txt;
  return e;
}
function svg(tag, attrs) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in (attrs || {})) e.setAttribute(k, attrs[k]);
  return e;
}
function fmt(v) {
  if (v == null || !isFinite(v)) return "-";
  const a = Math.abs(v);
  if (v !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return (Math.round(v * 1e4) / 1e4).toString();
}
function extent(series, key) {
  let lo = Infinity, hi = -Infinity;
  for (const s of series) {
    if (s.hidden) continue;
    for (const p of s.points) { const v = p[key]; if (v < lo) lo = v; if (v > hi) hi = v; }
  }
  return [lo, hi];
}
function niceTicks(lo, hi, n) {
  if (!isFinite(lo) || !isFinite(hi) || lo === hi) {
    const c = isFinite(lo) ? lo : 0;
    return [c];
  }
  const span = hi - lo;
  const step0 = span / Math.max(1, n);
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const norm = step0 / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const start = Math.ceil(lo / step) * step;
  const out = [];
  for (let t = start; t <= hi + step * 1e-6; t += step) out.push(t);
  return out;
}

// dims
const DIM = { w: 820, h: 300, ml: 56, mr: 16, mt: 12, mb: 30 };
// 半宽面板用更窄 viewBox，使其在两列网格里放大、文字仍清晰。
const HALFDIM = { w: 540, h: 300, ml: 52, mr: 14, mt: 12, mb: 30 };
const SUBDIM = { w: 360, h: 220, ml: 46, mr: 12, mt: 10, mb: 26 };

function makeChart(container, chart, dim) {
  const state = { series: chart.series.map(s => ({ ...s, hidden: !!s.hidden })),
                  log: !!chart.log };
  const root = el("div", "chart");
  const tt = el("div", "tt");
  const s = svg("svg", { viewBox: `0 0 ${dim.w} ${dim.h}`,
                         preserveAspectRatio: "none" });
  // keep aspect via wrapper height auto using viewBox ratio
  s.style.aspectRatio = `${dim.w} / ${dim.h}`;
  root.appendChild(s);
  root.appendChild(tt);
  container.appendChild(root);

  function px(x, xmin, xmax) {
    if (xmax === xmin) return dim.ml + (dim.w - dim.ml - dim.mr) / 2;
    return dim.ml + (x - xmin) / (xmax - xmin) * (dim.w - dim.ml - dim.mr);
  }
  function py(v, ymin, ymax, log) {
    const h = dim.h - dim.mt - dim.mb;
    if (log) {
      const l = Math.log10, a = l(ymin), b = l(ymax);
      if (b === a) return dim.mt + h / 2;
      return dim.mt + h - (l(v) - a) / (b - a) * h;
    }
    if (ymax === ymin) return dim.mt + h / 2;
    return dim.mt + h - (v - ymin) / (ymax - ymin) * h;
  }

  let scales = null;
  function redraw() {
    while (s.firstChild) s.removeChild(s.firstChild);
    const vis = state.series.filter(z => !z.hidden);
    let [xmin, xmax] = extent(state.series, 0);
    let [ymin, ymax] = extent(vis, 1);
    if (!isFinite(xmin)) { s.appendChild(svg("rect", {})); return; }
    let log = state.log && ymin > 0;
    // pad y
    if (ymin === ymax) { const e = Math.abs(ymin) || 1; ymin -= e * .1; ymax += e * .1; }
    else if (!log) { const pad = (ymax - ymin) * .06; ymin -= pad; ymax += pad; }
    scales = { xmin, xmax, ymin, ymax, log };

    // grid + y ticks
    const yticks = log ? logTicks(ymin, ymax) : niceTicks(ymin, ymax, 5);
    for (const t of yticks) {
      const y = py(t, ymin, ymax, log);
      s.appendChild(svg("line", { x1: dim.ml, y1: y, x2: dim.w - dim.mr, y2: y,
        stroke: "var(--grid)", "stroke-width": 1 }));
      const lab = svg("text", { x: dim.ml - 6, y: y + 3, "text-anchor": "end",
        "font-size": 10, fill: "var(--muted)", "font-family": "var(--font-mono)" });
      lab.textContent = fmt(t);
      s.appendChild(lab);
    }
    // x ticks
    const xticks = niceTicks(xmin, xmax, 6).filter(t => t >= xmin && t <= xmax);
    for (const t of xticks) {
      const x = px(t, xmin, xmax);
      const lab = svg("text", { x: x, y: dim.h - dim.mb + 14, "text-anchor": "middle",
        "font-size": 10, fill: "var(--muted)", "font-family": "var(--font-mono)" });
      lab.textContent = Math.round(t);
      s.appendChild(lab);
    }
    // axes
    s.appendChild(svg("line", { x1: dim.ml, y1: dim.mt, x2: dim.ml, y2: dim.h - dim.mb,
      stroke: "var(--line)", "stroke-width": 1 }));
    s.appendChild(svg("line", { x1: dim.ml, y1: dim.h - dim.mb, x2: dim.w - dim.mr,
      y2: dim.h - dim.mb, stroke: "var(--line)", "stroke-width": 1 }));

    // best marker
    if (chart.best_x != null && chart.best_x >= xmin && chart.best_x <= xmax) {
      const x = px(chart.best_x, xmin, xmax);
      s.appendChild(svg("line", { x1: x, y1: dim.mt, x2: x, y2: dim.h - dim.mb,
        stroke: "var(--best)", "stroke-width": 1.2, "stroke-dasharray": "4 3" }));
      const lab = svg("text", { x: x + 3, y: dim.mt + 10, "font-size": 9.5,
        fill: "var(--best)", "font-family": "var(--font-mono)" });
      lab.textContent = "best";
      s.appendChild(lab);
    }

    // series polylines（强调线最后画、加粗，置于顶层）。
    const ordered = vis.slice().sort((a, b) => (a.emphasis ? 1 : 0) - (b.emphasis ? 1 : 0));
    for (const z of ordered) {
      let d = "";
      z.points.forEach((p, i) => {
        const X = px(p[0], xmin, xmax), Y = py(p[1], ymin, ymax, log);
        d += (i ? " L" : "M") + X + " " + Y;
      });
      s.appendChild(svg("path", { d, fill: "none", stroke: z.color,
        "stroke-width": z.emphasis ? 3 : 1.8,
        "stroke-linejoin": "round", "stroke-linecap": "round" }));
      if (z.points.length === 1) {
        const p = z.points[0];
        s.appendChild(svg("circle", { cx: px(p[0], xmin, xmax),
          cy: py(p[1], ymin, ymax, log), r: 3, fill: z.color }));
      }
    }
    // hover group (drawn last)
    hoverG = svg("g", {}); s.appendChild(hoverG);
  }

  let hoverG = null;
  function onMove(ev) {
    if (!scales) return;
    const pt = s.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
    const loc = pt.matrixTransform(s.getScreenCTM().inverse());
    const { xmin, xmax, ymin, ymax, log } = scales;
    // invert x → data
    const frac = (loc.x - dim.ml) / (dim.w - dim.ml - dim.mr);
    const xd = xmin + frac * (xmax - xmin);
    const vis = state.series.filter(z => !z.hidden);
    while (hoverG.firstChild) hoverG.removeChild(hoverG.firstChild);
    if (loc.x < dim.ml || loc.x > dim.w - dim.mr || !vis.length) { tt.style.opacity = 0; return; }
    // nearest x among union of points (use first series' x grid)
    let nx = null, best = Infinity;
    for (const z of vis) for (const p of z.points) {
      const dd = Math.abs(p[0] - xd); if (dd < best) { best = dd; nx = p[0]; }
    }
    if (nx == null) { tt.style.opacity = 0; return; }
    const gx = px(nx, xmin, xmax);
    hoverG.appendChild(svg("line", { x1: gx, y1: dim.mt, x2: gx, y2: dim.h - dim.mb,
      stroke: "#94a3b8", "stroke-width": 1, "stroke-dasharray": "2 2" }));
    const rows = [];
    for (const z of vis) {
      const p = z.points.find(q => q[0] === nx);
      if (!p) continue;
      hoverG.appendChild(svg("circle", { cx: gx, cy: py(p[1], ymin, ymax, log),
        r: 3, fill: z.color, stroke: "#fff", "stroke-width": 1 }));
      rows.push(`<div class="row"><span><span class="sw" style="background:${z.color}"></span>${z.label}</span><span>${fmt(p[1])}</span></div>`);
    }
    tt.innerHTML = `<b>epoch ${Math.round(nx)}</b>` + rows.join("");
    const rect = root.getBoundingClientRect();
    tt.style.left = (gx / dim.w * rect.width) + "px";
    tt.style.top = (dim.mt / dim.h * rect.height) + "px";
    tt.style.opacity = 1;
  }
  s.addEventListener("mousemove", onMove);
  s.addEventListener("mouseleave", () => { tt.style.opacity = 0;
    if (hoverG) while (hoverG.firstChild) hoverG.removeChild(hoverG.firstChild); });

  redraw();
  return { state, redraw };
}

function logTicks(lo, hi) {
  const out = [], a = Math.floor(Math.log10(lo)), b = Math.ceil(Math.log10(hi));
  for (let e = a; e <= b; e++) { const v = Math.pow(10, e); if (v >= lo && v <= hi) out.push(v); }
  return out.length ? out : [lo, hi];
}

function legendFor(panel, chartObj) {
  if (chartObj.state.series.length < 2) return null;
  const lg = el("div", "legend");
  chartObj.state.series.forEach(z => {
    const it = el("div", "it" + (z.hidden ? " off" : ""));
    const sw = el("span", "sw"); sw.style.background = z.color;
    const lab = el("span", null, z.label);
    if (z.emphasis) lab.style.fontWeight = "700";
    it.appendChild(sw); it.appendChild(lab);
    it.addEventListener("click", () => {
      z.hidden = !z.hidden; it.classList.toggle("off");
      chartObj.redraw();
    });
    lg.appendChild(it);
  });
  return lg;
}

function buildPanel(panel) {
  const box = el("div", "panel");
  const title = el("div", "ptitle"); title.appendChild(el("span", null, panel.title));
  box.appendChild(title);

  if (panel.kind === "grid") {
    const gw = el("div", "gridwrap");
    box.appendChild(gw);
    (panel.charts || []).forEach(c => {
      const sub = el("div", "sub");
      sub.appendChild(el("div", "st", c.title));
      gw.appendChild(sub);
      makeChart(sub, c, SUBDIM);
    });
    return box;
  }

  // line panel（半宽面板用更窄 viewBox）
  const ctrl = el("div", "ctrl");
  title.appendChild(ctrl);
  const dim = panel.span === "half" ? HALFDIM : DIM;
  const chartObj = makeChart(box, panel, dim);
  // legend (insert before svg)
  const lg = legendFor(panel, chartObj);
  if (lg) box.insertBefore(lg, box.querySelector(".chart"));
  // log toggle
  if (panel.log_toggle) {
    const id = "log_" + panel.id;
    const lab = el("label");
    const cb = el("input"); cb.type = "checkbox"; cb.id = id;
    lab.appendChild(cb); lab.appendChild(document.createTextNode(" log y"));
    cb.addEventListener("change", () => { chartObj.state.log = cb.checked; chartObj.redraw(); });
    ctrl.appendChild(lab);
  }
  return box;
}

function renderMeta(meta) {
  const box = document.getElementById("meta");
  (meta || []).forEach(([k, v]) => {
    const s = el("span");
    s.appendChild(el("b", null, k + ": "));
    s.appendChild(document.createTextNode(v));
    box.appendChild(s);
  });
  if ((P.auto_reload_seconds | 0) > 0) {
    const live = el("span", "live");
    live.innerHTML = `<span class="dot"></span>live · ${P.auto_reload_seconds}s`;
    box.appendChild(live);
  }
}

function renderBestCard(bc) {
  if (!bc) return;
  const card = el("div", "bestcard");
  const hl = el("div", "hl");
  hl.innerHTML = `Best model · ${bc.headline.metric} = <b>${bc.headline.value}</b> @ epoch ${bc.headline.epoch}`;
  card.appendChild(hl);
  const grid = el("div", "grid");
  (bc.metrics || []).forEach(([k, v]) => {
    const m = el("div", "m");
    m.appendChild(el("span", "k", k));
    m.appendChild(el("span", "v", v));
    grid.appendChild(m);
  });
  card.appendChild(grid);
  document.getElementById("wrap").appendChild(card);
}

function renderRunsLegend(runs) {
  if (!runs || !runs.length) return;
  const card = el("div", "bestcard"); card.style.background = "var(--panel)";
  card.style.borderColor = "var(--line)";
  const hl = el("div", "hl"); hl.style.color = "var(--ink)"; hl.textContent = "Runs";
  card.appendChild(hl);
  const lg = el("div", "legend");
  runs.forEach(r => {
    const it = el("div", "it");
    const sw = el("span", "sw"); sw.style.background = r.color;
    it.appendChild(sw); it.appendChild(el("span", null, r.name));
    lg.appendChild(it);
  });
  card.appendChild(lg);
  document.getElementById("wrap").appendChild(card);
}

function renderTable(tbl) {
  if (!tbl || !tbl.columns) return;
  const t = el("table", "cmp-table");
  const thead = el("tr");
  tbl.columns.forEach(c => thead.appendChild(el("th", null, c)));
  const head = el("thead"); head.appendChild(thead); t.appendChild(head);
  const tb = el("tbody");
  (tbl.rows || []).forEach(r => {
    const tr = el("tr");
    r.forEach(c => tr.appendChild(el("td", null, c)));
    tb.appendChild(tr);
  });
  t.appendChild(tb);
  document.getElementById("wrap").appendChild(t);
}

function init() {
  document.getElementById("title").textContent = P.title || "Training Monitor";
  document.title = P.title || "Training Monitor";
  renderMeta(P.meta);
  if (P.mode === "compare") { renderRunsLegend(P.runs); renderTable(P.table); }
  else renderBestCard(P.best_card);
  const wrap = document.getElementById("wrap");
  const panels = P.panels || [];
  if (!panels.length) {
    wrap.appendChild(el("div", "empty", "暂无指标数据（还没有完成任何 epoch）。"));
  } else {
    // 按 group 连续分组：每组一个区块标题 + 两列响应式网格。
    const groups = [];
    let cur = null;
    panels.forEach(p => {
      const g = p.group || "";
      if (!cur || cur.name !== g) { cur = { name: g, items: [] }; groups.push(cur); }
      cur.items.push(p);
    });
    groups.forEach(grp => {
      if (grp.name) {
        const sec = el("div", "section");
        sec.appendChild(el("h2", null, grp.name));
        wrap.appendChild(sec);
      }
      const pg = el("div", "panel-grid");
      grp.items.forEach(p => {
        const box = buildPanel(p);
        box.classList.add(p.span === "full" ? "span-full" : "span-half");
        pg.appendChild(box);
      });
      wrap.appendChild(pg);
    });
  }
  // scroll restore (across auto-reload)
  const y = sessionStorage.getItem("mon_scrollY");
  if (y) window.scrollTo(0, parseInt(y, 10));
}
window.addEventListener("beforeunload", () =>
  sessionStorage.setItem("mon_scrollY", String(window.scrollY)));
window.addEventListener("resize", () => { /* svg scales via viewBox; no redraw needed */ });
document.addEventListener("DOMContentLoaded", init);
const __RELOAD = P.auto_reload_seconds | 0;
if (__RELOAD > 0) setTimeout(() => location.reload(), __RELOAD * 1000);
"""


__all__ = ["CSS", "JS"]
