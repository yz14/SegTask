"""把 ``MetricsHistory`` 整理成「渲染就绪」的图表载荷（纯函数，零外部依赖）。

设计：Python 端把历史解析成一份与具体指标语义无关的通用 payload（每条曲线已是
``[[x, y], ...]`` 点列），JS 端只做通用的「画线 / 网格 / 悬停 / 图例」渲染。
好处是分组/取数/best 逻辑全部可在 Python 侧单测，JS 保持精简通用。

x 轴一律用 1-based epoch（展示友好）；内部 0-based 索引在此 +1。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .history import MetricsHistory

# 克制的定性配色（与 visualization/render.py 同色系，便于全局视觉统一）。
PALETTE: List[str] = [
    "#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed",
    "#0891b2", "#db2777", "#65a30d", "#475569", "#ca8a04",
]


def _color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def _points(hist: MetricsHistory, key: str, source: str) -> List[List[float]]:
    """取 ``[[epoch_1based, value], ...]``。"""
    xs, ys = hist.series(key, source)
    return [[x + 1, y] for x, y in zip(xs, ys)]


def _line_series(label: str, color: str, points: List[List[float]]) -> Dict[str, Any]:
    return {"label": label, "color": color, "points": points}


# 概览指标：均值类（按出现顺序取存在者）。
_OVERVIEW_METRICS: List[str] = [
    "mean_dice", "mean_iou", "mean_recall", "mean_precision",
    "mean_mcc", "mean_vol_sim", "mean_surface_dice", "min_class_dice",
]

# 逐类指标前缀 → 人类可读名。[0,1] 同尺度的合并到一张图（线型区分指标）；
# 其它尺度（MCC∈[-1,1]、VolSim）各自单图，避免尺度混淆。
_UNIT_SCALE_PREFIXES: List[tuple] = [
    ("dice_class_", "Dice"),
    ("iou_class_", "IoU"),
    ("recall_class_", "Recall"),
    ("precision_class_", "Precision"),
]
_OTHER_SCALE_PREFIXES: List[tuple] = [
    ("mcc_class_", "MCC"),
    ("vol_sim_class_", "VolSim"),
]

# 逐类合并图（Per-class Dice/IoU/Recall/Precision）的配色：按「指标分色系、
# 类内分深浅」给每条线一个独立颜色——既区分指标又区分类，比线型(dash)更易读。
# 每个色系 4 档（class 0..3 由深到浅）；类数超过 4 时循环复用。
_PER_CLASS_UNIT_FAMILIES: List[List[str]] = [
    ["#1e3a8a", "#2563eb", "#3b82f6", "#60a5fa"],  # Dice    · 蓝
    ["#065f46", "#059669", "#10b981", "#34d399"],  # IoU     · 绿
    ["#9a3412", "#d97706", "#f59e0b", "#fbbf24"],  # Recall  · 橙
    ["#5b21b6", "#7c3aed", "#8b5cf6", "#a78bfa"],  # Prec.   · 紫
]


def _loss_component_keys(hist: MetricsHistory) -> List[str]:
    """训练损失分量键（主路 / aux / 多分辨率），排除权重 ``w_aux_*``。"""
    prefixes = ("L_main", "L_aux_res_", "L_aux_", "L_res_")
    keys = [k for k in hist.metric_keys("train")
            if any(k.startswith(p) for p in prefixes)]
    return sorted(keys)


def _best_x(hist: MetricsHistory) -> Optional[float]:
    best = hist.best
    ep = best.get("epoch")
    return (ep + 1) if isinstance(ep, int) else None


# 模型健康监测键（训练侧），由 trainer 逐 epoch 聚合后写入 train 指标。
_HEALTH_KEYS = (
    "grad_norm", "grad_norm_max", "weight_norm",
    "grad_clip_frac", "nonfinite_steps", "amp_scale", "update_ratio",
)


def _health_panels(
    hist: MetricsHistory,
    train_keys: set,
    best_x: Optional[float],
) -> List[Dict[str, Any]]:
    """「Model health」组面板：present-才画，老 run 无这些键时整组不出现。

    - 梯度范数（mean + max，默认对数纵轴）：发散/消失一眼可见。
    - 权重范数（L2）：权重是否异常增长 / 坍塌。
    - 裁剪比例：开启 grad_clip 时范数超阈值的优化步占比。
    - 非有限步计数：NaN/Inf loss 何时出现。
    - AMP loss scale：仅 fp16 scaler 启用时存在，反复回退=溢出频繁。
    - update/weight 比值（默认对数纵轴）：仅开启该开关时存在；健康区间约 1e-3，
      过大=可能发散/lr 偏高，过小=学不动/lr 偏低。
    """
    panels: List[Dict[str, Any]] = []
    grp = "Model health"

    gn_series: List[Dict[str, Any]] = []
    if "grad_norm" in train_keys:
        s = _line_series("grad_norm", _color(0), _points(hist, "grad_norm", "train"))
        s["emphasis"] = True
        gn_series.append(s)
    if "grad_norm_max" in train_keys:
        gn_series.append(_line_series(
            "grad_norm_max", _color(1), _points(hist, "grad_norm_max", "train")))
    if gn_series:
        panels.append({
            "id": "grad_norm", "title": "Gradient norm", "kind": "line",
            "group": grp, "span": "half", "log": True, "log_toggle": True,
            "best_x": best_x, "series": gn_series,
        })

    if "weight_norm" in train_keys:
        panels.append({
            "id": "weight_norm", "title": "Weight norm (L2)", "kind": "line",
            "group": grp, "span": "half", "best_x": best_x,
            "series": [_line_series("weight_norm", _color(4),
                                    _points(hist, "weight_norm", "train"))],
        })

    if "grad_clip_frac" in train_keys:
        panels.append({
            "id": "grad_clip_frac", "title": "Grad-clip fraction", "kind": "line",
            "group": grp, "span": "half", "best_x": best_x,
            "series": [_line_series("grad_clip_frac", _color(3),
                                    _points(hist, "grad_clip_frac", "train"))],
        })

    if "nonfinite_steps" in train_keys:
        panels.append({
            "id": "nonfinite_steps", "title": "Non-finite steps", "kind": "line",
            "group": grp, "span": "half", "best_x": best_x,
            "series": [_line_series("nonfinite_steps", _color(1),
                                    _points(hist, "nonfinite_steps", "train"))],
        })

    if "amp_scale" in train_keys:
        panels.append({
            "id": "amp_scale", "title": "AMP loss scale", "kind": "line",
            "group": grp, "span": "half", "log": True, "log_toggle": True,
            "best_x": best_x,
            "series": [_line_series("amp_scale", _color(5),
                                    _points(hist, "amp_scale", "train"))],
        })

    if "update_ratio" in train_keys:
        panels.append({
            "id": "update_ratio", "title": "Update/weight ratio", "kind": "line",
            "group": grp, "span": "half", "log": True, "log_toggle": True,
            "best_x": best_x,
            "series": [_line_series("update_ratio", _color(2),
                                    _points(hist, "update_ratio", "train"))],
        })

    return panels


def build_single_payload(
    hist: MetricsHistory,
    *,
    auto_reload_seconds: int = 0,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """单 run 仪表盘 payload。"""
    name = run_name or hist.run_name
    best_x = _best_x(hist)
    val_keys = set(hist.metric_keys("val"))
    train_keys = set(hist.metric_keys("train"))
    sel = hist.summary.get("save_best_metric")
    crit = hist.summary.get("save_best_criterion") or sel
    panels: List[Dict[str, Any]] = []

    # ============ Training ============
    # 损失曲线（train / val），占满整行。总损失（train/val）加粗高亮，训练损失
    # 分量（L_main / L_aux_* / L_res_*）以细线并入同一张图、各自一色，默认显示，
    # 可经图例逐条开关。
    loss_series: List[Dict[str, Any]] = []
    if "loss" in train_keys:
        s = _line_series("train loss", _color(0), _points(hist, "loss", "train"))
        s["emphasis"] = True
        loss_series.append(s)
    if "val_loss" in val_keys:
        s = _line_series("val loss", _color(1), _points(hist, "val_loss", "val"))
        s["emphasis"] = True
        loss_series.append(s)
    comp = _loss_component_keys(hist)
    for j, k in enumerate(comp):
        loss_series.append(_line_series(k, _color(2 + j),
                                        _points(hist, k, "train")))
    if loss_series:
        panels.append({
            "id": "loss", "title": "Loss", "kind": "line",
            "group": "Training", "span": "full",
            "log_toggle": True, "best_x": best_x, "series": loss_series,
        })

    # ============ Validation ============
    # 均值指标总览（[0,1] 同尺度多线），选模指标加粗高亮。占满整行。
    ov = [m for m in _OVERVIEW_METRICS if m in val_keys]
    if sel and sel in val_keys and sel not in ov:
        ov = [sel] + ov
    if ov:
        ov_series = []
        for i, m in enumerate(ov):
            s = _line_series(m, _color(i), _points(hist, m, "val"))
            if m == sel:
                # 选模指标无需文字标注：加粗 + 虚线即可一眼识别（图例同步显示）。
                s["emphasis"] = True
                s["dash"] = "6 4"
            ov_series.append(s)
        title = "Validation metrics (mean)"
        if crit and crit != sel:
            title += f"  ·  criterion = {crit}"
        panels.append({
            "id": "overview", "title": title, "kind": "line",
            "group": "Validation", "span": "full",
            "best_x": best_x, "series": ov_series,
        })

    # 逐类指标：同为 [0,1] 尺度的多个指标合并到「同一张图」——按指标分色系、
    # 类内分深浅，每条线一个独立颜色（图例区分），默认全部显示。
    # 非 [0,1] 尺度（MCC∈[-1,1]、VolSim）各自单图，避免尺度混淆。
    present_unit = [(p, n) for p, n in _UNIT_SCALE_PREFIXES
                    if hist.per_class_keys(p, "val")]
    if present_unit:
        series = []
        for mi, (prefix, nice) in enumerate(present_unit):
            family = _PER_CLASS_UNIT_FAMILIES[mi % len(_PER_CLASS_UNIT_FAMILIES)]
            for k in hist.per_class_keys(prefix, "val"):
                idx = k[len(prefix):]
                ci = int(idx) if idx.isdigit() else 0
                color = family[ci % len(family)]
                series.append(_line_series(f"class {idx} · {nice}", color,
                                           _points(hist, k, "val")))
        nices = [n for _, n in present_unit]
        title = ("Per-class " + nices[0]) if len(nices) == 1 \
            else "Per-class " + " / ".join(nices)
        panels.append({
            "id": "per_class_unit", "title": title, "kind": "line",
            "group": "Validation",
            "span": "half" if len(present_unit) == 1 else "full",
            "best_x": best_x, "series": series,
        })

    for prefix, nice in _OTHER_SCALE_PREFIXES:
        keys = hist.per_class_keys(prefix, "val")
        if not keys:
            continue
        series = []
        for k in keys:
            idx = k[len(prefix):]
            ci = int(idx) if idx.isdigit() else 0
            series.append(_line_series(f"class {idx}", _color(ci),
                                       _points(hist, k, "val")))
        panels.append({
            "id": f"per_class_{prefix.rstrip('_')}",
            "title": f"Per-class {nice}", "kind": "line",
            "group": "Validation", "span": "half",
            "best_x": best_x, "series": series,
        })

    # ============ System ============
    # 学习率（对数纵轴），半宽。
    if any(r.lr is not None for r in hist.records):
        panels.append({
            "id": "lr", "title": "Learning rate", "kind": "line",
            "group": "System", "span": "half",
            "log": True, "best_x": best_x,
            "series": [_line_series("lr", _color(8), _points(hist, "lr", "top"))],
        })

    # GPU 峰值显存，半宽。
    if any(r.gpu_peak_mib is not None for r in hist.records):
        panels.append({
            "id": "gpu", "title": "GPU peak memory (MiB)", "kind": "line",
            "group": "System", "span": "half", "best_x": best_x,
            "series": [_line_series("gpu_peak_mib", _color(3),
                                    _points(hist, "gpu_peak_mib", "top"))],
        })

    # ============ Model health ============
    panels.extend(_health_panels(hist, train_keys, best_x))

    return {
        "mode": "single",
        "title": name,
        "auto_reload_seconds": int(auto_reload_seconds),
        "meta": _single_meta(hist, name),
        "best_card": _best_card(hist),
        "panels": panels,
    }


def _single_meta(hist: MetricsHistory, name: str) -> List[List[str]]:
    s = hist.summary
    planned = s.get("total_epochs_planned") or 0
    recorded = len(hist)
    meta = [["Run", name], ["Status", str(s.get("status", "?"))]]
    meta.append(["Epochs", f"{recorded}" + (f" / {planned}" if planned else "")])
    if hist.num_classes:
        meta.append(["Classes", str(hist.num_classes)])
    best = hist.best
    if best:
        meta.append(["Best",
                     f"{best.get('metric_name')}={_fmt(best.get('metric_value'))}"
                     f" @ ep{(best.get('epoch') or 0) + 1}"])
    return meta


# best 卡片里逐类矩阵的列（前缀 → 列名），按既有图表同序排列。
_MATRIX_PREFIXES: List[tuple] = _UNIT_SCALE_PREFIXES + _OTHER_SCALE_PREFIXES


def _best_card(hist: MetricsHistory) -> Optional[Dict[str, Any]]:
    best = hist.best
    if not best:
        return None
    val = best.get("val") or {}
    sel = best.get("metric_name")

    # 1) 均值 / 聚合指标 → 高亮 stat 磁贴（按既定顺序，选模指标置顶并标记）。
    means_keys = [m for m in _OVERVIEW_METRICS if m in val]
    if sel in val and sel not in means_keys:
        means_keys = [sel] + means_keys
    means = [{"key": k, "value": _fmt(val[k]), "selected": k == sel}
             for k in means_keys]

    # 2) 逐类指标 → 矩阵（行=class，列=指标）；同时记录被矩阵吸收的 key。
    consumed = set(means_keys)
    columns: List[str] = []
    col_prefix: List[str] = []
    class_idx: List[int] = []
    for prefix, nice in _MATRIX_PREFIXES:
        idxs = []
        for k in val:
            if k.startswith(prefix):
                suf = k[len(prefix):]
                if suf.isdigit():
                    idxs.append(int(suf))
        if idxs:
            columns.append(nice)
            col_prefix.append(prefix)
            class_idx = sorted(set(class_idx) | set(idxs))
    matrix: Optional[Dict[str, Any]] = None
    if columns and class_idx:
        rows = []
        for ci in class_idx:
            cells = []
            for prefix in col_prefix:
                k = f"{prefix}{ci}"
                if k in val:
                    consumed.add(k)
                    try:
                        t = float(val[k])
                    except (TypeError, ValueError):
                        t = None
                    cells.append({"value": _fmt(val[k]),
                                  "t": t if t is not None and t == t else None})
                else:
                    cells.append({"value": "—", "t": None})
            rows.append({"label": f"class {ci}", "cells": cells})
        matrix = {"columns": columns, "rows": rows}

    # 3) 其余标量指标（既非均值磁贴、也未进矩阵）→ 次要小列表。
    rest = [[k, _fmt(val[k])] for k in sorted(val) if k not in consumed]

    return {
        "headline": {
            "metric": best.get("metric_name"),
            "value": _fmt(best.get("metric_value")),
            "epoch": (best.get("epoch") or 0) + 1,
        },
        "means": means,
        "matrix": matrix,
        "rest": rest,
    }


def build_compare_payload(
    histories: Sequence[MetricsHistory],
    *,
    run_names: Optional[Sequence[str]] = None,
    auto_reload_seconds: int = 0,
) -> Dict[str, Any]:
    """多 run 对比 payload：同名指标叠加 + best 对照表。"""
    hs = list(histories)
    names = list(run_names) if run_names else [h.run_name for h in hs]
    # 同名重复时加序号区分。
    seen: Dict[str, int] = {}
    uniq = []
    for n in names:
        if n in seen:
            seen[n] += 1
            uniq.append(f"{n}#{seen[n]}")
        else:
            seen[n] = 0
            uniq.append(n)
    names = uniq

    run_colors = [_color(i) for i in range(len(hs))]

    # 对比的指标集合：val_loss + 概览均值指标（取任一 run 出现过的）。
    val_union = set()
    for h in hs:
        val_union.update(h.metric_keys("val"))
    compare_metrics = (["val_loss"] if "val_loss" in val_union else []) \
        + [m for m in _OVERVIEW_METRICS if m in val_union]

    panels: List[Dict[str, Any]] = []
    for m in compare_metrics:
        series = []
        for i, h in enumerate(hs):
            pts = _points(h, m, "val")
            if pts:
                series.append(_line_series(names[i], run_colors[i], pts))
        if series:
            panels.append({"id": f"cmp_{m}", "title": m, "kind": "line",
                           "group": "Comparison", "span": "half",
                           "series": series})
    # 训练损失对比（train loss）。
    tr_series = []
    for i, h in enumerate(hs):
        pts = _points(h, "loss", "train")
        if pts:
            tr_series.append(_line_series(names[i], run_colors[i], pts))
    if tr_series:
        panels.insert(0, {"id": "cmp_train_loss", "title": "train loss",
                          "kind": "line", "group": "Comparison",
                          "span": "half", "series": tr_series})

    # best 对照表。
    table = _compare_table(hs, names)

    return {
        "mode": "compare",
        "title": "Run comparison",
        "auto_reload_seconds": int(auto_reload_seconds),
        "runs": [{"name": n, "color": run_colors[i]} for i, n in enumerate(names)],
        "panels": panels,
        "table": table,
    }


def _compare_table(histories: Sequence[MetricsHistory],
                   names: Sequence[str]) -> Dict[str, Any]:
    # 列：Run | best criterion 值 | best epoch | 概览指标在 best epoch 的值。
    metric_cols: List[str] = []
    for h in histories:
        for m in _OVERVIEW_METRICS:
            best_val = (h.best.get("val") or {})
            if m in best_val and m not in metric_cols:
                metric_cols.append(m)
    columns = ["Run", "best", "epoch"] + metric_cols
    rows: List[List[str]] = []
    for i, h in enumerate(histories):
        best = h.best
        bval = best.get("val") or {}
        row = [
            names[i],
            f"{best.get('metric_name', '?')}={_fmt(best.get('metric_value'))}"
            if best else "n/a",
            str((best.get("epoch") or 0) + 1) if best else "n/a",
        ]
        row += [_fmt(bval.get(m)) if m in bval else "—" for m in metric_cols]
        rows.append(row)
    return {"columns": columns, "rows": rows}


def _fmt(v: Any) -> str:
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return "—"
    if fv != 0 and (abs(fv) < 1e-3 or abs(fv) >= 1e5):
        return f"{fv:.2e}"
    return f"{fv:.4f}"


__all__ = [
    "PALETTE", "build_single_payload", "build_compare_payload",
]
