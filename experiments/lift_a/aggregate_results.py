"""Aggregate the three Plan-A baseline runs into a comparison table.

Parses each run's ``outputs/lift_a/<name>/train.log`` for:
  * Model params (M)
  * Per-epoch ``loss`` / ``val_dice`` (training-side rolling)
  * Pooled per-class dice + mean from the ``Val:`` line
  * Best epoch / best mean_dice
  * Total training time (from the final 'Training complete' line)
  * Per-class coverage (sanity)

Writes a Markdown summary to ``experiments/lift_a/results.md`` plus a CSV
of per-epoch curves (``epoch,run,train_loss,val_dice,...``) for plotting.

Note: GPU peak memory is not logged by the trainer; a follow-up patch can
hook ``torch.cuda.max_memory_allocated`` into the epoch summary if needed.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS = [
    ("baseline_2_5d",        "outputs/lift_a/baseline_2_5d"),
    ("planA_lift_r2plus1d",  "outputs/lift_a/planA_lift_r2plus1d"),
    ("ref_3d_zaxis",         "outputs/lift_a/ref_3d_zaxis"),
]

# ---------------------------------------------------------------------------
# Log regexes (tied to trainer.py's logger.info messages — keep in sync).
# ---------------------------------------------------------------------------
RE_PARAMS = re.compile(r"Model params:\s*([\d.]+)M")
RE_EPOCH = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s*\|\s*LR=([\d.eE+\-]+)\s*\|\s*loss=([\d.]+)\s*\|"
    r"\s*val_dice=([\d.]+)\s*\|\s*best="
)
RE_VAL = re.compile(
    r"Val:\s*loss=([\d.]+),\s*pooled_mean_dice=([\d.]+),\s*"
    r"per_class=\[([^\]]+)\],\s*coverage=\[([^\]]+)\]/(\d+)\s*samples"
)
RE_BEST = re.compile(r"New best:\s*mean_dice=([\d.]+)\s*at epoch\s*(\d+)")
RE_DONE = re.compile(r"Training complete.*?Time:\s*(.+)$")
RE_GPU = re.compile(r"GPU:\s*([^()]+)\(([\d.]+)\s*GB\)")
RE_GPU_PEAK = re.compile(r"GPU peak \(epoch\s*(\d+)\):\s*([\d.]+)\s*MiB")


@dataclass
class RunStats:
    name: str
    log_path: Path
    params_m: Optional[float] = None
    gpu_name: Optional[str] = None
    gpu_total_gb: Optional[float] = None
    best_mean_dice: Optional[float] = None
    best_epoch: Optional[int] = None
    final_val_per_class: List[float] = field(default_factory=list)
    coverage: List[int] = field(default_factory=list)
    n_val: Optional[int] = None
    elapsed: Optional[str] = None
    epochs: List[Dict[str, float]] = field(default_factory=list)
    gpu_peak_mib: Optional[float] = None  # max across epochs


def parse_log(name: str, run_dir: Path) -> RunStats:
    log_path = run_dir / "train.log"
    s = RunStats(name=name, log_path=log_path)
    if not log_path.is_file():
        print(f"[warn] {log_path} not found; skipping {name}")
        return s

    last_val: Optional[Tuple[float, float, List[float], List[int], int]] = None

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            if (m := RE_PARAMS.search(line)):
                s.params_m = float(m.group(1))
            elif (m := RE_GPU.search(line)):
                s.gpu_name = m.group(1).strip()
                s.gpu_total_gb = float(m.group(2))
            elif (m := RE_EPOCH.search(line)):
                s.epochs.append({
                    "epoch": int(m.group(1)),
                    "lr": float(m.group(3)),
                    "train_loss": float(m.group(4)),
                    "val_dice": float(m.group(5)),
                })
            elif (m := RE_VAL.search(line)):
                per_class = [float(x.strip()) for x in m.group(3).split(",")]
                cov = [int(x.strip()) for x in m.group(4).split(",")]
                last_val = (
                    float(m.group(1)), float(m.group(2)),
                    per_class, cov, int(m.group(5)))
            elif (m := RE_BEST.search(line)):
                d = float(m.group(1))
                if s.best_mean_dice is None or d > s.best_mean_dice:
                    s.best_mean_dice = d
                    s.best_epoch = int(m.group(2))
            elif (m := RE_GPU_PEAK.search(line)):
                v = float(m.group(2))
                if s.gpu_peak_mib is None or v > s.gpu_peak_mib:
                    s.gpu_peak_mib = v
            elif (m := RE_DONE.search(line)):
                s.elapsed = m.group(1).strip()

    if last_val is not None:
        _, _, per_class, cov, n_val = last_val
        s.final_val_per_class = per_class
        s.coverage = cov
        s.n_val = n_val
    return s


def write_curves_csv(runs: List[RunStats], out: Path) -> None:
    rows = []
    for r in runs:
        for ep in r.epochs:
            rows.append({
                "run": r.name, "epoch": ep["epoch"],
                "train_loss": ep["train_loss"],
                "val_dice": ep["val_dice"], "lr": ep["lr"],
            })
    if not rows:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote per-epoch curves to {out}")


def write_markdown(runs: List[RunStats], out: Path) -> None:
    lines: List[str] = []
    lines.append("# Plan A 升维收益对照实验结果\n")
    lines.append("自动由 `experiments/lift_a/aggregate_results.py` 生成。"
                 "metric 来源：每组 `outputs/lift_a/<name>/train.log`。\n")

    # Headline table
    lines.append("## 总览\n")
    lines.append("| Run | params (M) | best mean_dice | best epoch | final per-class dice | coverage / n_val | GPU peak (MiB) | total time |")
    lines.append("|---|---:|---:|---:|---|---|---:|---|")
    for r in runs:
        per_class = (
            "[" + ", ".join(f"{x:.4f}" for x in r.final_val_per_class) + "]"
            if r.final_val_per_class else "—")
        cov = (
            f"{r.coverage} / {r.n_val}"
            if r.coverage and r.n_val is not None else "—")
        params_cell = f"{r.params_m:.2f}" if r.params_m is not None else "—"
        best_cell = f"{r.best_mean_dice:.4f}" if r.best_mean_dice is not None else "—"
        epoch_cell = f"{r.best_epoch}" if r.best_epoch is not None else "—"
        gpu_cell = f"{r.gpu_peak_mib:.0f}" if r.gpu_peak_mib is not None else "—"
        lines.append(
            f"| `{r.name}` | {params_cell} | {best_cell} | {epoch_cell} "
            f"| {per_class} | {cov} | {gpu_cell} | {r.elapsed or '—'} |")

    # Per-class dice block
    lines.append("\n## Per-class dice（最后一次 val 报告）\n")
    lines.append("| Run | class_0 | class_1 | class_2 |")
    lines.append("|---|---:|---:|---:|")
    for r in runs:
        if r.final_val_per_class:
            cells = " | ".join(f"{x:.4f}" for x in r.final_val_per_class)
        else:
            cells = "— | — | —"
        lines.append(f"| `{r.name}` | {cells} |")

    # Quick notes
    lines.append("\n## 备注\n")
    lines.append("- `best mean_dice` 取自 `★ New best` 日志，与 `save_best_metric=mean_dice` 对齐。")
    lines.append("- `final per-class dice` 取最后一次 `Val:` 行（最后一个 epoch 的 EMA 指标）。")
    lines.append("- `GPU peak (MiB)` 取自 trainer 每 epoch 及时记录的 `torch.cuda.max_memory_allocated`（训练全程逐 epoch reset），表中为训练过程中的跨 epoch 最大值。")
    lines.append("- 三组 seed=42, EMA, bf16, 30 epochs, batch=2, samples_per_volume=4 严格一致；"
                 "唯一变量：`{patch_mode, block_type, lift_2_5d_to_3d}`。\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] wrote summary to {out}")


def main() -> None:
    runs: List[RunStats] = []
    for name, rel_dir in RUNS:
        run_dir = REPO_ROOT / rel_dir
        runs.append(parse_log(name, run_dir))

    out_md = REPO_ROOT / "experiments" / "lift_a" / "results.md"
    out_csv = REPO_ROOT / "experiments" / "lift_a" / "curves.csv"
    write_markdown(runs, out_md)
    write_curves_csv(runs, out_csv)

    # Console digest
    print("\n=== Digest ===")
    for r in runs:
        print(f"  {r.name:<24}  best_dice={r.best_mean_dice}  "
              f"best_ep={r.best_epoch}  params={r.params_m}M  "
              f"gpu_peak={r.gpu_peak_mib} MiB  time={r.elapsed}")


if __name__ == "__main__":
    main()
