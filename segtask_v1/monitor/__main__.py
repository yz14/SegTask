"""训练监测 CLI —— 离线（重）渲染单 run 仪表盘或多 run 对比页。

训练期 Trainer 已自动实时刷新仪表盘；本 CLI 面向**训练之后**的场景：从已落盘的
``metrics.jsonl`` 重新生成 HTML、或把若干次实验叠加成一张对比页，便于 debug 与
横向比较。零外部依赖，产物为自包含 HTML、可离线打开。

用法
----
单 run（重渲染，默认写到该 run 目录下 ``training_monitor.html``）::

    python -m segtask_v1.monitor runs/exp_a

多 run 对比（叠加同名指标曲线 + best 对照表）::

    python -m segtask_v1.monitor runs/exp_a runs/exp_b runs/exp_c -o cmp.html \\
        --names baseline +aug +ema

每个 RUN 既可指向含 ``metrics.jsonl`` 的目录，也可指向其父级 ``output_dir``
（自动下探 ``monitor/`` 子目录）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .dashboard import write_comparison, write_dashboard
from .history import MetricsHistory

_DEFAULT_SINGLE_NAME = "training_monitor.html"
_DEFAULT_COMPARE_NAME = "training_compare.html"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m segtask_v1.monitor",
        description="离线渲染训练监测仪表盘（单 run）或多 run 对比页。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "runs", nargs="+", metavar="RUN",
        help="一个或多个 run 目录（output_dir 或其 monitor/ 子目录）。"
             "给出多个时生成对比页。")
    p.add_argument(
        "-o", "--out", metavar="PATH", default=None,
        help="输出 HTML 路径。单 run 缺省写到该 run 目录下 "
             f"{_DEFAULT_SINGLE_NAME}；多 run 缺省写到当前目录 "
             f"{_DEFAULT_COMPARE_NAME}。")
    p.add_argument(
        "--names", nargs="+", metavar="NAME", default=None,
        help="对比页中各 run 的显示名（顺序与 RUN 对应；缺省取 run_name）。")
    p.add_argument(
        "--auto-reload", type=int, default=0, metavar="SEC",
        help="页面自动重载间隔（秒）；0（缺省）表示静态、不自动刷新。")
    return p


def _load_runs(run_paths: Sequence[str]) -> List[MetricsHistory]:
    histories: List[MetricsHistory] = []
    for rp in run_paths:
        d = Path(rp)
        if not d.exists():
            raise FileNotFoundError(f"run 目录不存在：{rp}")
        hist = MetricsHistory.from_dir(d)
        if len(hist) == 0:
            raise ValueError(
                f"未在 {rp} 找到任何 epoch 记录（缺少或为空的 metrics.jsonl）。")
        histories.append(hist)
    return histories


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.names is not None and len(args.names) != len(args.runs):
        print(
            f"error: --names 数量（{len(args.names)}）与 RUN 数量"
            f"（{len(args.runs)}）不一致。", file=sys.stderr)
        return 2

    try:
        histories = _load_runs(args.runs)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if len(histories) == 1:
        hist = histories[0]
        if args.out:
            out = Path(args.out)
        else:
            src = Path(hist.source_dir or ".")
            base = src.parent if src.name == "monitor" else src
            out = base / _DEFAULT_SINGLE_NAME
        run_name = args.names[0] if args.names else None
        written = write_dashboard(
            hist, out, auto_reload_seconds=args.auto_reload, run_name=run_name)
        print(f"单 run 仪表盘已生成：{written}  ({len(hist)} epochs)")
    else:
        out = Path(args.out) if args.out else Path.cwd() / _DEFAULT_COMPARE_NAME
        written = write_comparison(
            histories, out, run_names=args.names,
            auto_reload_seconds=args.auto_reload)
        print(f"多 run 对比页已生成：{written}  （{len(histories)} runs）")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
