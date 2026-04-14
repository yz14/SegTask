"""CLI entry point for training.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --train.epochs 100 --train.lr 0.001
"""

from __future__ import annotations

import argparse
import logging
import sys

import torch

from segtask.config import Config, load_config, save_config
from segtask.data.loader import build_dataloaders
from segtask.models.factory import build_model
from segtask.trainer import Trainer
from segtask.utils import seed_everything, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SegTask Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, cpu")

    # Allow overriding any config value via CLI
    # e.g. --train.epochs 100 --data.batch_size 16
    args, unknown = parser.parse_known_args()

    # Parse overrides
    overrides = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            key = unknown[i][2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                overrides[key] = unknown[i + 1]
                i += 2
            else:
                overrides[key] = "true"
                i += 1
        else:
            i += 1

    args.overrides = overrides
    return args


def apply_overrides(cfg: Config, overrides: dict) -> None:
    """Apply CLI overrides to config (e.g. 'train.lr' → cfg.train.lr)."""
    for key, value in overrides.items():
        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                logger.warning("Unknown config path: %s", key)
                break
        else:
            attr = parts[-1]
            if hasattr(obj, attr):
                current = getattr(obj, attr)
                # Type cast
                if isinstance(current, bool):
                    setattr(obj, attr, value.lower() in ("true", "1", "yes"))
                elif isinstance(current, int):
                    setattr(obj, attr, int(value))
                elif isinstance(current, float):
                    setattr(obj, attr, float(value))
                elif isinstance(current, str):
                    setattr(obj, attr, value)
                elif isinstance(current, list):
                    # Parse comma-separated values
                    if current and isinstance(current[0], int):
                        setattr(obj, attr, [int(x) for x in value.split(",")])
                    elif current and isinstance(current[0], float):
                        setattr(obj, attr, [float(x) for x in value.split(",")])
                    else:
                        setattr(obj, attr, value.split(","))
                else:
                    setattr(obj, attr, value)
                logger.info("Override: %s = %s", key, value)
            else:
                logger.warning("Unknown config key: %s", key)


def main():
    args = parse_args()

    # Load config
    cfg = load_config(args.config)
    apply_overrides(cfg, args.overrides)
    cfg.sync()

    # Setup
    setup_logging(cfg.train.output_dir)
    seed_everything(cfg.train.seed, cfg.train.deterministic)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    # Save effective config
    save_config(cfg, f"{cfg.train.output_dir}/config.yaml")

    # Build data
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # Build model
    model = build_model(cfg)

    # Train
    trainer = Trainer(model, cfg, train_loader, val_loader, device)
    best_metrics = trainer.fit()

    logger.info("Training finished. Best metrics: %s", best_metrics)


if __name__ == "__main__":
    main()
