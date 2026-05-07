"""Aux supervision controlled sweep on small_data (12 volumes).

What this script does
---------------------
Runs 4 short trainings (30 epochs each) on the small_data subset with the
SAME data / optimiser / aug / EMA / split seed, varying only:

  | id | aux_seg_supervision | context_fusion    | aux_head_mode |
  |----|---------------------|-------------------|---------------|
  | A0 | False               | multi_stem_proj   | linear        |  ← baseline
  | A1 | True                | multi_stem_proj   | linear        |  Plan A + aux
  | C1 | True                | hierarchical      | linear        |  Plan C + linear head
  | C2 | True                | hierarchical      | conv          |  Plan C + 3x3 head

Each run writes its full config + checkpoints + log under
``outputs/aux_exp/<id>/``. After all runs finish, we print a side-by-side
comparison of the best ``mean_dice`` and the final epoch's ``L_aux_k``
breakdown so you can see whether the aux signal is being learnt.

Usage::

    conda activate torch27_env
    python configs/experiments/run_aux_sweep.py
    python configs/experiments/run_aux_sweep.py --only A0,C2  # subset

Notes
-----
- Each run takes ~5–15 min depending on GPU. Total wall time ≈ 30–60 min.
- Reproducibility: fixed ``train.seed=42`` + ``data.split_seed=42`` so all
  4 runs see identical train/val partition.
- The base config lives in ``configs/experiments/seg2_5d_small_base.yaml``
  — edit it once to change shared settings (epochs, lr, etc.).
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from segtask_v1.config import load_config, save_config  # noqa: E402

BASE_CFG = ROOT / "configs" / "experiments" / "seg2_5d_small_base.yaml"
EXP_ROOT = ROOT / "outputs" / "aux_exp"

# -- Sweep matrix -------------------------------------------------------------
EXPERIMENTS: List[Dict] = [
    {
        "id": "A0_baseline",
        "label": "baseline (aux off, Plan A)",
        "model.aux_seg_supervision": False,
        "model.context_fusion": "multi_stem_proj",
        "model.aux_head_mode": "linear",
    },
    {
        "id": "A1_planA_aux",
        "label": "Plan A + aux (linear head)",
        "model.aux_seg_supervision": True,
        "model.context_fusion": "multi_stem_proj",
        "model.aux_head_mode": "linear",
    },
    {
        "id": "C1_planC_aux_linear",
        "label": "Plan C + aux (linear head)",
        "model.aux_seg_supervision": True,
        "model.context_fusion": "hierarchical",
        "model.aux_head_mode": "linear",
    },
    {
        "id": "C2_planC_aux_conv",
        "label": "Plan C + aux (3x3 conv head)",
        "model.aux_seg_supervision": True,
        "model.context_fusion": "hierarchical",
        "model.aux_head_mode": "conv",
    },
]


def _set_dotted(cfg, dotted: str, value):
    parent, _, last = dotted.rpartition(".")
    obj = cfg
    for part in parent.split("."):
        obj = getattr(obj, part)
    setattr(obj, last, value)


def _materialise_cfg(exp: Dict) -> Tuple[Path, "Config"]:
    """Build the per-experiment config (in-memory + on-disk YAML)."""
    cfg = load_config(BASE_CFG)
    out_dir = EXP_ROOT / exp["id"]
    cfg.train.output_dir = str(out_dir)
    for k, v in exp.items():
        if k in ("id", "label"):
            continue
        _set_dotted(cfg, k, v)
    cfg.sync()
    cfg.validate()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config.yaml"
    save_config(cfg, cfg_path)
    return cfg_path, cfg


def _setup_run_logger(out_dir: Path) -> logging.Handler:
    """Tee the root logger to ``out_dir/train.log`` for offline inspection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(out_dir / "train.log", mode="w",
                                  encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        "%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(handler)
    return handler


def _run_one(exp: Dict, only_smoke: bool = False) -> Dict[str, float]:
    """Build trainer in-process and run ``fit()``. Returns best metrics."""
    from segtask_v1.data.loader import build_dataloaders
    from segtask_v1.trainer import Trainer
    from segtask_v1.models.factory import build_model
    from segtask_v1.utils import seed_everything
    import torch

    cfg_path, cfg = _materialise_cfg(exp)
    out_dir = Path(cfg.train.output_dir)
    handler = _setup_run_logger(out_dir)
    log = logging.getLogger(__name__)
    try:
        log.info("=" * 70)
        log.info("[exp=%s] %s", exp["id"], exp["label"])
        log.info("Config saved to %s", cfg_path)
        log.info("=" * 70)
        seed_everything(cfg.train.seed, deterministic=cfg.train.deterministic)
        train_loader, val_loader = build_dataloaders(cfg)
        if only_smoke:
            log.info("[smoke] Loader built OK (train=%d, val=%d). Skipping fit().",
                     len(train_loader), len(val_loader))
            return {}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(cfg)
        trainer = Trainer(model, cfg, train_loader, val_loader, device)
        best = trainer.fit()
        return best
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


_RE_EPOCH = re.compile(
    r"Epoch\s+(?P<epoch>\d+)/\d+\s+\|\s+LR=[^|]+\|\s+loss=(?P<loss>[\d.eE+-]+)"
    r"\s+\|\s+val_dice=(?P<val>[\d.eE+-]+)"
)
_RE_AUX = re.compile(
    r"(L_main=(?P<lm>[\d.eE+-]+))?"
    r"(?:\s+L_aux_(?P<k>\d+)=(?P<lk>[\d.eE+-]+)\(w=(?P<wk>[\d.eE+-]+)\))?"
)


def _parse_log(log_path: Path) -> Dict[str, float]:
    """Parse ``train.log`` for the final-epoch summary line.

    Returns a dict like ``{"loss", "val_dice", "L_main", "L_aux_1", ...}``;
    all keys are present only when the corresponding fields exist in the
    log. Robust to missing aux fields (no-aux runs).
    """
    if not log_path.is_file():
        return {}
    last_epoch_line = None
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "Epoch " in line and "val_dice=" in line:
                last_epoch_line = line.strip()
    if not last_epoch_line:
        return {}
    out: Dict[str, float] = {}
    m = _RE_EPOCH.search(last_epoch_line)
    if m:
        out["loss"] = float(m["loss"])
        out["val_dice"] = float(m["val"])
    # Capture all L_main / L_aux_k tokens.
    for tok in re.finditer(r"(L_main|L_aux_\d+)=([\d.eE+-]+)", last_epoch_line):
        out[tok.group(1)] = float(tok.group(2))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="Comma-separated subset of "
                    "experiment ids (e.g. A0_baseline,C2_planC_aux_conv).")
    ap.add_argument("--smoke", action="store_true",
                    help="Build configs + dataloaders only; skip fit().")
    ap.add_argument("--clean", action="store_true",
                    help="Wipe outputs/aux_exp/<id>/ before each run.")
    args = ap.parse_args()

    # Root logger to console.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    selected = [e for e in EXPERIMENTS if not only or e["id"] in only]
    if not selected:
        print(f"No experiments match --only={args.only!r}. Valid ids: "
              f"{[e['id'] for e in EXPERIMENTS]}")
        sys.exit(2)

    print(f"\nRunning {len(selected)} experiment(s) under {EXP_ROOT}\n")
    summary: Dict[str, Dict[str, float]] = {}
    for exp in selected:
        out_dir = EXP_ROOT / exp["id"]
        if args.clean and out_dir.exists():
            shutil.rmtree(out_dir)
        t0 = time.time()
        try:
            best = _run_one(exp, only_smoke=args.smoke)
        except Exception as e:
            logging.exception("Experiment %s FAILED: %s", exp["id"], e)
            summary[exp["id"]] = {"status": "FAIL"}
            continue
        elapsed = time.time() - t0
        parsed = _parse_log(out_dir / "train.log")
        summary[exp["id"]] = {
            "label": exp["label"],
            "elapsed_min": elapsed / 60.0,
            "best_val_dice": float(best.get("mean_dice", float("nan")))
            if best else float("nan"),
            **{k: v for k, v in parsed.items() if k.startswith("L_")},
        }

    # -------------------- Side-by-side comparison ------------------------
    print("\n" + "=" * 84)
    print("AUX SWEEP RESULTS — small_data (30 epochs each)")
    print("=" * 84)
    header = (f"{'id':<24} {'best_val_dice':>14} {'elapsed_min':>12} "
              f"{'L_main':>10} {'L_aux_1':>10} {'L_aux_2':>10}")
    print(header)
    print("-" * len(header))
    for eid, m in summary.items():
        if m.get("status") == "FAIL":
            print(f"{eid:<24} FAILED")
            continue
        print(f"{eid:<24} "
              f"{m.get('best_val_dice', float('nan')):>14.4f} "
              f"{m.get('elapsed_min', 0):>12.1f} "
              f"{m.get('L_main', float('nan')):>10.4f} "
              f"{m.get('L_aux_1', float('nan')):>10.4f} "
              f"{m.get('L_aux_2', float('nan')):>10.4f}")
    print("=" * 84)
    print(f"Configs + per-run logs under {EXP_ROOT}")
    print("Inspect train.log for full epoch-by-epoch curves "
          "(L_main / L_aux_k breakdown is appended to each Epoch line).")


if __name__ == "__main__":
    main()
