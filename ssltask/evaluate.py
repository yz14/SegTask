"""P6 离线评测 CLI。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from .config import apply_overrides, load_config, validate_ssl
from .eval.pipeline import run_eval_pipeline


def _parse_entry(spec: str):
    if spec.lower() == "from_scratch":
        return ("B2-from-scratch", None)
    if "=" in spec:
        name, rhs = spec.split("=", 1)
    elif ":" in spec:
        name, rhs = spec.split(":", 1)
    else:
        name, rhs = spec, spec
    rhs = rhs.strip()
    if rhs.lower() in ("from_scratch", "none", "null"):
        rhs = None
    return (name.strip(), rhs)


def main() -> dict:
    parser = argparse.ArgumentParser(description="P6 offline evaluation harness (ssltask)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides (key=value); ssl.* routes to SSLConfig")
    parser.add_argument("--entry", action="append", default=[], help="Eval entry spec: NAME=CKPT or NAME=from_scratch (repeatable)")
    parser.add_argument("--shots", nargs="*", type=int, default=None, help="Few-shot list")
    parser.add_argument("--readout", nargs="*", default=None, help="Readouts: linear finetune")
    parser.add_argument("--task", nargs="*", default=None, help="Tasks: seg cls")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    cfg, ssl = load_config(args.config)
    if args.override:
        apply_overrides(cfg, ssl, args.override)
        cfg.sync()
        cfg.validate()
        validate_ssl(ssl, cfg)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    logger = logging.getLogger(__name__)
    logger.info("Device: %s", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    entries = [_parse_entry(s) for s in args.entry] if args.entry else None
    result = run_eval_pipeline(
        cfg,
        ssl,
        entries=entries,
        shots=args.shots,
        readouts=args.readout,
        tasks=args.task,
        out_dir=args.out_dir,
    )
    logger.info("Offline evaluation written to %s / %s", result["json_path"], result["csv_path"])
    return result


if __name__ == "__main__":
    main()
