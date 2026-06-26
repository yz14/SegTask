"""训练监测工具（segtask_v1.monitor）测试。

数据层（history）零外部依赖，可独立运行；不触碰 torch / 训练流程。
"""

from __future__ import annotations

import json
import math

from segtask_v1.monitor import (
    EpochRecord,
    MetricsHistory,
    MetricsLogger,
    render_comparison,
    render_dashboard,
    write_dashboard,
)
from segtask_v1.monitor import charts


def _val(epoch: int) -> dict:
    return {
        "val_loss": 1.0 / (epoch + 1),
        "mean_dice": 0.5 + 0.04 * epoch,
        "dice_class_0": 0.9,
        "dice_class_1": 0.4 + 0.05 * epoch,
        "nan_metric": float("nan"),  # 应被过滤
    }


def _train(epoch: int) -> dict:
    return {"loss": 2.0 / (epoch + 1), "L_main": 1.0 / (epoch + 1)}


def test_logger_roundtrip_and_filtering(tmp_path):
    d = tmp_path / "run_a" / "monitor"
    lg = MetricsLogger(
        d, run_name="run_a", save_best_metric="mean_dice",
        save_best_mode="max", num_classes=2, total_epochs=5)
    for ep in range(5):
        lg.log_epoch(ep, train=_train(ep), val=_val(ep), lr=1e-3 * (0.9 ** ep),
                     gpu_peak_mib=1234.0 + ep, wall_time_s=10.0)
    lg.finalize("completed")

    hist = MetricsHistory.from_dir(d)
    assert len(hist) == 5
    # 序列正确、跳过缺失。
    xs, ys = hist.series("mean_dice", "val")
    assert xs == [0, 1, 2, 3, 4]
    assert math.isclose(ys[-1], 0.5 + 0.04 * 4)
    # NaN 指标被过滤，不出现在任何 key 集合中。
    assert "nan_metric" not in hist.metric_keys("val")
    # best：mean_dice 单调增 → best 是最后一个 epoch。
    best = hist.best
    assert best["epoch"] == 4
    assert best["metric_name"] == "mean_dice"
    # 逐类 key 数值后缀排序。
    assert hist.per_class_keys("dice_class_") == ["dice_class_0", "dice_class_1"]
    # summary 状态与计划 epoch 数。
    assert hist.summary["status"] == "completed"
    assert hist.summary["total_epochs_planned"] == 5


def test_resume_appends_without_duplication(tmp_path):
    d = tmp_path / "run_b" / "monitor"
    lg = MetricsLogger(d, run_name="run_b", save_best_metric="mean_dice",
                       save_best_mode="max", total_epochs=10)
    for ep in range(3):
        lg.log_epoch(ep, train=_train(ep), val=_val(ep))

    # 模拟续训：新 logger、resume=True，从 epoch 3 续写。
    lg2 = MetricsLogger(d, run_name="run_b", save_best_metric="mean_dice",
                        save_best_mode="max", total_epochs=10, resume=True)
    assert len(lg2.records) == 3  # 加载了已有历史
    for ep in range(3, 6):
        lg2.log_epoch(ep, train=_train(ep), val=_val(ep))

    hist = MetricsHistory.from_dir(d)
    epochs = [r.epoch for r in hist.records]
    assert epochs == [0, 1, 2, 3, 4, 5]  # 无重复、连续

    # jsonl 行数 == epoch 数（无半行 / 重复行）。
    lines = [ln for ln in (d / "metrics.jsonl").read_text().splitlines() if ln.strip()]
    assert len(lines) == 6


def test_overwrites_stale_run_when_not_resume(tmp_path):
    d = tmp_path / "run_c" / "monitor"
    lg = MetricsLogger(d, total_epochs=3)
    for ep in range(3):
        lg.log_epoch(ep, val=_val(ep))
    assert len(MetricsHistory.from_dir(d)) == 3

    # 全新 run（resume=False）应清空历史。
    lg2 = MetricsLogger(d, total_epochs=3)
    assert len(lg2.records) == 0
    assert len(MetricsHistory.from_dir(d)) == 0


def test_epoch_record_serialization():
    rec = EpochRecord(epoch=2, train={"loss": 0.1}, val={"mean_dice": 0.8},
                      lr=1e-3, gpu_peak_mib=100.0, is_best=True)
    d = rec.to_dict()
    rt = EpochRecord.from_dict(json.loads(json.dumps(d)))
    assert rt.epoch == 2 and rt.is_best is True
    assert rt.val["mean_dice"] == 0.8 and rt.lr == 1e-3


def test_missing_dir_yields_empty_history(tmp_path):
    hist = MetricsHistory.from_dir(tmp_path / "does_not_exist")
    assert len(hist) == 0
    assert hist.best == {}
    assert hist.last_epoch is None


# ---------------------------------------------------------------------------
# 步骤 2：渲染层（charts payload + dashboard HTML）
# ---------------------------------------------------------------------------
def _make_run(tmp_path, name, n=5, best_metric="mean_dice"):
    d = tmp_path / name / "monitor"
    lg = MetricsLogger(d, run_name=name, save_best_metric=best_metric,
                       save_best_mode="max", num_classes=2, total_epochs=n,
                       save_best_criterion=best_metric)
    for ep in range(n):
        lg.log_epoch(ep, train=_train(ep), val=_val(ep),
                     lr=1e-3 * (0.9 ** ep), gpu_peak_mib=1234.0 + ep)
    lg.finalize("completed")
    return MetricsHistory.from_dir(d)


def test_single_payload_structure(tmp_path):
    hist = _make_run(tmp_path, "run_a")
    payload = charts.build_single_payload(hist, auto_reload_seconds=7)
    assert payload["mode"] == "single"
    assert payload["auto_reload_seconds"] == 7
    ids = {p["id"] for p in payload["panels"]}
    # 关键面板存在：损失 / 概览 / 学习率 / GPU / 逐类 Dice。
    assert {"loss", "overview", "lr", "gpu"} <= ids
    # 已去掉冗余的独立「选模指标」面板（并入概览高亮）。
    assert "primary" not in ids
    # 逐类 [0,1] 指标合并到统一面板。
    assert "per_class_unit" in ids

    # 分组 / 跨列布局元数据齐全。
    by_id = {p["id"]: p for p in payload["panels"]}
    assert by_id["loss"]["group"] == "Training" and by_id["loss"]["span"] == "full"
    assert by_id["overview"]["group"] == "Validation"
    assert by_id["overview"]["span"] == "full"
    assert by_id["lr"]["group"] == "System" and by_id["lr"]["span"] == "half"

    # 选模指标并入概览并加粗高亮。
    ov_sel = [s for s in by_id["overview"]["series"] if s.get("emphasis")]
    assert len(ov_sel) == 1 and "(selection)" in ov_sel[0]["label"]

    # 逐类指标合并为「一图多线」：fixture 只有 Dice（num_classes=2 → 2 条），
    # 单指标时半宽。
    pc = by_id["per_class_unit"]
    assert pc["kind"] == "line" and len(pc["series"]) == 2
    assert pc["span"] == "half"
    assert pc["title"] == "Per-class Dice"

    # x 轴为 1-based epoch。
    loss = by_id["loss"]
    assert loss["series"][0]["points"][0][0] == 1
    # best 标记 / best 卡片。
    assert loss["best_x"] == 5  # mean_dice 单调增 → best 在最后 epoch（0-based 4 → 1-based 5）
    bc = payload["best_card"]
    assert bc["headline"]["metric"] == "mean_dice"
    assert bc["headline"]["epoch"] == 5


def test_per_class_unit_scale_merge(tmp_path):
    # 同为 [0,1] 尺度的 Dice + IoU 合并到一张图：每条线一个独立颜色（无 dash），
    # 按「指标分色系、类内分深浅」区分，默认全部显示。
    d = tmp_path / "merge" / "monitor"
    lg = MetricsLogger(d, run_name="merge", save_best_metric="mean_dice",
                       save_best_mode="max", num_classes=2, total_epochs=3)
    for ep in range(3):
        lg.log_epoch(ep, train=_train(ep), val={
            "mean_dice": 0.5 + 0.04 * ep,
            "dice_class_0": 0.9, "dice_class_1": 0.4 + 0.05 * ep,
            "iou_class_0": 0.8, "iou_class_1": 0.3 + 0.05 * ep,
        })
    lg.finalize("completed")
    hist = MetricsHistory.from_dir(d)
    payload = charts.build_single_payload(hist)
    by_id = {p["id"]: p for p in payload["panels"]}
    pc = by_id["per_class_unit"]
    # 两指标 → 占满整行；2 类 × 2 指标 = 4 条线。
    assert pc["span"] == "full"
    assert len(pc["series"]) == 4
    assert pc["title"] == "Per-class Dice / IoU"
    # 不再用线型(dash)区分，全部为独立实线颜色。
    assert all("dash" not in s for s in pc["series"])
    # 16 条以内每条线颜色互不相同（此处 4 条）。
    colors = [s["color"] for s in pc["series"]]
    assert len(set(colors)) == len(colors)
    # 同一类的 Dice / IoU 现在用不同颜色（不同色系）。
    dice = [s for s in pc["series"] if "Dice" in s["label"]]
    iou = [s for s in pc["series"] if "IoU" in s["label"]]
    c0_dice = next(s for s in dice if s["label"].startswith("class 0"))
    c0_iou = next(s for s in iou if s["label"].startswith("class 0"))
    assert c0_dice["color"] != c0_iou["color"]


def test_loss_components_merged_into_loss_panel(tmp_path):
    # 训练损失分量并入 Loss 面板（不再有独立 loss_components 面板）；
    # 总损失 train/val 加粗高亮，分量默认显示。
    d = tmp_path / "lossc" / "monitor"
    lg = MetricsLogger(d, run_name="lossc", save_best_metric="mean_dice",
                       save_best_mode="max", num_classes=2, total_epochs=3)
    for ep in range(3):
        lg.log_epoch(ep, train={"loss": 2.0 / (ep + 1), "L_main": 1.0 / (ep + 1),
                                "L_res_0": 0.5 / (ep + 1)},
                     val=_val(ep))
    lg.finalize("completed")
    hist = MetricsHistory.from_dir(d)
    payload = charts.build_single_payload(hist)
    ids = {p["id"] for p in payload["panels"]}
    assert "loss_components" not in ids  # 已合并
    by_id = {p["id"]: p for p in payload["panels"]}
    loss = by_id["loss"]
    labels = [s["label"] for s in loss["series"]]
    assert "train loss" in labels and "val loss" in labels
    assert "L_main" in labels and "L_res_0" in labels  # 分量并入
    # 总损失（train/val）加粗高亮，分量不强调。
    emph = {s["label"] for s in loss["series"] if s.get("emphasis")}
    assert emph == {"train loss", "val loss"}
    assert all(not s.get("emphasis") for s in loss["series"]
               if s["label"] in ("L_main", "L_res_0"))
    # 每条线颜色互不相同。
    colors = [s["color"] for s in loss["series"]]
    assert len(set(colors)) == len(colors)


def test_best_card_means_and_per_class_matrix(tmp_path):
    # best 卡片：均值磁贴 + 逐类矩阵（含数值用于 heatmap）+ 其余标量列表。
    d = tmp_path / "card" / "monitor"
    lg = MetricsLogger(d, run_name="card", save_best_metric="mean_dice",
                       save_best_mode="max", num_classes=2, total_epochs=2)
    for ep in range(2):
        lg.log_epoch(ep, train=_train(ep), val={
            "val_loss": 1.0 / (ep + 1),
            "mean_dice": 0.5 + 0.1 * ep, "mean_iou": 0.4 + 0.1 * ep,
            "dice_class_0": 0.8 + 0.05 * ep, "dice_class_1": 0.6 + 0.05 * ep,
            "iou_class_0": 0.7, "iou_class_1": 0.5,
            "mcc_class_0": 0.3, "mcc_class_1": 0.4,
        })
    lg.finalize("completed")
    hist = MetricsHistory.from_dir(d)
    bc = charts.build_single_payload(hist)["best_card"]

    # 均值磁贴：含 mean_dice / mean_iou，选模指标标 selected。
    mean_keys = [m["key"] for m in bc["means"]]
    assert "mean_dice" in mean_keys and "mean_iou" in mean_keys
    sel = [m for m in bc["means"] if m["selected"]]
    assert len(sel) == 1 and sel[0]["key"] == "mean_dice"

    # 逐类矩阵：列含 Dice / IoU / MCC，行=2 个 class，cell 带数值 t。
    mtx = bc["matrix"]
    assert mtx["columns"] == ["Dice", "IoU", "MCC"]
    assert [r["label"] for r in mtx["rows"]] == ["class 0", "class 1"]
    c0 = mtx["rows"][0]["cells"]
    assert len(c0) == 3 and c0[0]["t"] is not None

    # 逐类 key 被矩阵吸收，不再混入「其余」列表；val_loss 进入其余。
    rest_keys = [k for k, _ in bc["rest"]]
    assert "dice_class_0" not in rest_keys
    assert "val_loss" in rest_keys


def test_per_class_other_scale_separate(tmp_path):
    # 非 [0,1] 尺度（MCC）不并入合并图，单独成图。
    d = tmp_path / "mcc" / "monitor"
    lg = MetricsLogger(d, run_name="mcc", save_best_metric="mean_dice",
                       save_best_mode="max", num_classes=2, total_epochs=2)
    for ep in range(2):
        lg.log_epoch(ep, train=_train(ep), val={
            "mean_dice": 0.5 + 0.04 * ep,
            "dice_class_0": 0.9, "dice_class_1": 0.5,
            "mcc_class_0": -0.2 + 0.1 * ep, "mcc_class_1": 0.1,
        })
    lg.finalize("completed")
    hist = MetricsHistory.from_dir(d)
    payload = charts.build_single_payload(hist)
    ids = {p["id"] for p in payload["panels"]}
    assert "per_class_unit" in ids          # dice
    assert "per_class_mcc_class" in ids      # mcc 单独
    by_id = {p["id"]: p for p in payload["panels"]}
    assert all("dash" not in s for s in by_id["per_class_mcc_class"]["series"])


def test_single_payload_empty_history():
    hist = MetricsHistory(run_name="empty")
    payload = charts.build_single_payload(hist)
    assert payload["panels"] == []
    assert payload["best_card"] is None


def test_render_dashboard_html_self_contained(tmp_path):
    hist = _make_run(tmp_path, "run_a")
    htmlstr = render_dashboard(hist, auto_reload_seconds=5)
    # 自包含：内联 CSS/JS，无外部 http(s) 资源引用。
    assert "<!DOCTYPE html>" in htmlstr
    assert "<style>" in htmlstr and "<script>" in htmlstr
    assert "src=\"http" not in htmlstr and "href=\"http" not in htmlstr
    # payload 已注入，且 </ 被转义避免提前闭合 script。
    assert "const P = " in htmlstr
    assert "</script>" in htmlstr  # 仅结尾真正闭合标签
    # 标题含 run 名。
    assert "run_a" in htmlstr


def test_render_dashboard_persists_legend_state(tmp_path):
    # 图例隐藏 / log 开关状态跨自动重载持久化：HTML 内联的 JS 必须用
    # sessionStorage 存取（否则定时 location.reload() 会把隐藏曲线重置）。
    hist = _make_run(tmp_path, "run_a")
    htmlstr = render_dashboard(hist, auto_reload_seconds=5)
    assert "loadHidden" in htmlstr and "saveHidden" in htmlstr
    assert "mon_hid_" in htmlstr
    assert "loadLog" in htmlstr and "saveLog" in htmlstr


def test_render_dashboard_escapes_script_close(tmp_path):
    d = tmp_path / "weird" / "monitor"
    lg = MetricsLogger(d, run_name="</script><b>x", save_best_metric="mean_dice",
                       total_epochs=2)
    lg.log_epoch(0, val=_val(0))
    lg.finalize()
    hist = MetricsHistory.from_dir(d)
    htmlstr = render_dashboard(hist)
    # 整页应恰好只有一个真正闭合的 </script>（页面结尾那个）。
    assert htmlstr.count("</script>") == 1
    # run 名里的 </ 在注入 payload 中被转义。
    assert "<\\/script><b>x" in htmlstr


def test_compare_payload_and_html(tmp_path):
    h1 = _make_run(tmp_path, "run_a", n=5)
    h2 = _make_run(tmp_path, "run_b", n=4)
    payload = charts.build_compare_payload([h1, h2])
    assert payload["mode"] == "compare"
    assert len(payload["runs"]) == 2
    # 训练损失对比面板在最前。
    assert payload["panels"][0]["id"] == "cmp_train_loss"
    assert len(payload["panels"][0]["series"]) == 2
    # best 对照表两行。
    assert len(payload["table"]["rows"]) == 2
    htmlstr = render_comparison([h1, h2])
    assert "run_a" in htmlstr and "run_b" in htmlstr
    assert "cmp-table" in htmlstr


def test_compare_dedups_run_names(tmp_path):
    h1 = _make_run(tmp_path, "dup", n=3)
    h2 = _make_run(tmp_path, "dup", n=3)
    payload = charts.build_compare_payload([h1, h2], run_names=["dup", "dup"])
    names = [r["name"] for r in payload["runs"]]
    assert names[0] != names[1]


def test_write_dashboard_atomic(tmp_path):
    hist = _make_run(tmp_path, "run_a")
    out = tmp_path / "out" / "training_monitor.html"
    written = write_dashboard(hist, out, auto_reload_seconds=3)
    assert written == out and out.exists()
    assert not out.with_suffix(out.suffix + ".tmp").exists()  # 临时文件已清理
    assert "<!DOCTYPE html>" in out.read_text(encoding="utf-8")


# ===========================================================================
# Trainer 实时集成（步骤 3）—— 需要 torch 才能导入 Trainer，缺失则跳过。
# 不跑完整数据/模型管线：用 ``Trainer.__new__`` 构造最小实例，直接驱动新增的
# monitor 钩子（_init_monitor / _monitor_log_epoch / _monitor_finalize），
# 验证落盘节奏、best/末轮强制重渲染、收尾静态渲染与续训不重复。
# ===========================================================================
import pytest  # noqa: E402


def _bare_trainer(cfg, output_dir):
    """绕过重型 __init__，仅装配 monitor 钩子所需的最小属性。"""
    torch = pytest.importorskip("torch")
    from segtask_v1.trainer.trainer import Trainer

    t = Trainer.__new__(Trainer)
    t.cfg = cfg
    t.output_dir = output_dir
    t.num_fg = cfg.num_fg_classes
    t.device = torch.device("cpu")
    t._monitor = None
    t._monitor_html = None
    t._monitor_cfg = cfg.monitor
    return t


def _make_cfg_with_monitor(output_dir, *, update_every=2, epochs=5):
    pytest.importorskip("torch")
    from segtask_v1.config import Config

    cfg = Config()
    cfg.train.output_dir = str(output_dir)
    cfg.train.epochs = epochs
    cfg.train.save_best_criterion = "dice"  # → save_best_metric=mean_dice (max)
    cfg.monitor.enabled = True
    cfg.monitor.update_every = update_every
    cfg.monitor.auto_reload_seconds = 7
    return cfg


def test_trainer_monitor_logs_and_renders_on_cadence(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    cfg = _make_cfg_with_monitor(out, update_every=2, epochs=5)
    t = _bare_trainer(cfg, out)
    t._init_monitor(resume_active=False)

    assert t._monitor is not None  # 启用且初始化成功
    html = t._monitor_html
    mon_dir = t._monitor.dir
    jsonl = mon_dir / "metrics.jsonl"

    # mean_dice 在 epoch 3 取峰值，使 best 标记与据值计算的 best 一致。
    dice_curve = [0.50, 0.60, 0.70, 0.85, 0.80]

    rendered_at = []
    prev_html = None
    for epoch in range(5):
        is_best = (epoch == 3)
        val = {"val_loss": 1.0 / (epoch + 1),
               "mean_dice": dice_curve[epoch],
               "dice_class_0": 0.9, "dice_class_1": dice_curve[epoch] - 0.1}
        t._monitor_log_epoch(
            epoch, _train(epoch), val,
            lr=1e-3, gpu_peak_mib=None, wall_time_s=0.01,
            is_best=is_best, last_epoch=(epoch == 4))
        # jsonl 逐 epoch 增长。
        assert len(jsonl.read_text().strip().splitlines()) == epoch + 1
        cur_html = html.read_text(encoding="utf-8") if html.exists() else None
        if cur_html is not None and cur_html != prev_html:
            rendered_at.append(epoch)  # 内容变化 ⇒ 本轮发生了重渲染
        prev_html = cur_html

    # update_every=2 ⇒ epoch1,3 触发；epoch3 同时是 best；epoch4 是末轮 ⇒ 必渲染。
    assert {1, 3, 4}.issubset(set(rendered_at))
    assert 0 not in rendered_at and 2 not in rendered_at

    # 渲染时带自动刷新（训练进行中）。
    assert "auto_reload_seconds" in html.read_text(encoding="utf-8")

    # 收尾：状态写入 summary，并做一次无自动刷新的静态终渲染。
    t._monitor_finalize("completed")
    summary = json.loads((mon_dir / "metrics_summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["best"]["epoch"] == 3  # best epoch 落定
    assert "<!DOCTYPE html>" in html.read_text(encoding="utf-8")


def test_trainer_monitor_disabled_is_zero_side_effect(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    cfg = _make_cfg_with_monitor(out, epochs=3)
    cfg.monitor.enabled = False
    t = _bare_trainer(cfg, out)
    # 模拟 __init__ 中的守卫：disabled ⇒ 不初始化。
    if cfg.monitor.enabled:
        t._init_monitor(resume_active=False)

    for epoch in range(3):
        t._monitor_log_epoch(
            epoch, _train(epoch), _val(epoch), lr=1e-3,
            gpu_peak_mib=None, wall_time_s=0.0, is_best=False,
            last_epoch=(epoch == 2))
    t._monitor_finalize("completed")

    # 关闭时零文件、零副作用。
    assert not (out / "monitor").exists()
    assert not (out / "training_monitor.html").exists()


def test_trainer_monitor_resume_no_duplication(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    cfg = _make_cfg_with_monitor(out, update_every=1, epochs=6)

    # 第一段：epoch 0..2。
    t1 = _bare_trainer(cfg, out)
    t1._init_monitor(resume_active=False)
    for epoch in range(3):
        t1._monitor_log_epoch(
            epoch, _train(epoch), _val(epoch), lr=1e-3,
            gpu_peak_mib=None, wall_time_s=0.0, is_best=False,
            last_epoch=False)
    jsonl = t1._monitor.dir / "metrics.jsonl"
    assert len(jsonl.read_text().strip().splitlines()) == 3

    # 第二段：以 resume 续训 epoch 3..5，不应重复已有 epoch。
    t2 = _bare_trainer(cfg, out)
    t2._init_monitor(resume_active=True)
    for epoch in range(3, 6):
        t2._monitor_log_epoch(
            epoch, _train(epoch), _val(epoch), lr=1e-3,
            gpu_peak_mib=None, wall_time_s=0.0, is_best=False,
            last_epoch=(epoch == 5))

    epochs_logged = [
        json.loads(line)["epoch"]
        for line in jsonl.read_text().strip().splitlines()
    ]
    assert epochs_logged == [0, 1, 2, 3, 4, 5]  # 连续、无重复


# ===========================================================================
# CLI（步骤 4）：python -m segtask_v1.monitor —— 离线（重）渲染 + 多 run 对比。
# 零外部依赖，可独立运行（不触碰 torch）。
# ===========================================================================
from segtask_v1.monitor.__main__ import main as monitor_cli  # noqa: E402


def _persist_run(tmp_path, name, *, scale=0.05, n=5):
    """落盘一个真实 run 目录（含 metrics.jsonl + summary），返回 output_dir。"""
    out = tmp_path / name
    lg = MetricsLogger(
        out / "monitor", run_name=name, save_best_metric="mean_dice",
        save_best_mode="max", num_classes=2, total_epochs=n)
    for e in range(n):
        lg.log_epoch(
            e, train={"loss": 2.0 / (e + 1)},
            val={"val_loss": 1.0 / (e + 1), "mean_dice": 0.5 + scale * e,
                 "dice_class_0": 0.9, "dice_class_1": 0.4 + scale * e},
            lr=1e-3, is_best=(e == n - 1))
    lg.finalize("completed")
    return out


def test_cli_single_default_output(tmp_path):
    run = _persist_run(tmp_path, "exp_a")
    rc = monitor_cli([str(run)])
    assert rc == 0
    out = run / "training_monitor.html"  # 缺省写到 run 目录
    assert out.exists()
    assert "<!DOCTYPE html>" in out.read_text(encoding="utf-8")


def test_cli_single_accepts_monitor_subdir_and_explicit_out(tmp_path):
    run = _persist_run(tmp_path, "exp_a")
    out = tmp_path / "custom.html"
    rc = monitor_cli([str(run / "monitor"), "-o", str(out), "--auto-reload", "5"])
    assert rc == 0 and out.exists()
    assert "auto_reload_seconds" in out.read_text(encoding="utf-8")


def test_cli_compare_multi_run(tmp_path):
    a = _persist_run(tmp_path, "exp_a", scale=0.04)
    b = _persist_run(tmp_path, "exp_b", scale=0.07)
    out = tmp_path / "cmp.html"
    rc = monitor_cli([str(a), str(b), "-o", str(out),
                      "--names", "baseline", "aug"])
    assert rc == 0 and out.exists()
    htmlstr = out.read_text(encoding="utf-8")
    assert "baseline" in htmlstr and "aug" in htmlstr
    assert "cmp-table" in htmlstr  # best 对照表存在


def test_cli_names_count_mismatch_errors(tmp_path):
    a = _persist_run(tmp_path, "exp_a")
    b = _persist_run(tmp_path, "exp_b")
    rc = monitor_cli([str(a), str(b), "--names", "only-one"])
    assert rc == 2  # 用法错误


def test_cli_missing_run_errors(tmp_path):
    rc = monitor_cli([str(tmp_path / "does_not_exist")])
    assert rc == 1


def test_cli_empty_run_errors(tmp_path):
    empty = tmp_path / "empty" / "monitor"
    empty.mkdir(parents=True)
    rc = monitor_cli([str(tmp_path / "empty")])
    assert rc == 1  # 无 epoch 记录
