"""3D 分割训练 CLI 入口。示例：`python -m segtask_v1.train --config configs/seg3d.yaml [--override train.epochs=50 ...]`。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from .config import load_config, save_config
from .data.loader import build_dataloaders
from .logging_utils import setup_logging as _setup_logging
from .models.factory import build_model
from .trainer import Trainer
from .utils import seed_everything


def setup_logging(output_dir: str, level: str = "INFO") -> None:
    """同时输出控制台（彩色）与文件（纯文本）日志。"""
    _setup_logging(output_dir=output_dir, level=level, log_filename="train.log")


def apply_overrides(cfg, overrides: list) -> None:
    """应用点记法 override。示例：--override train.epochs=50 model.backbone=convnext。"""
    for ov in overrides:
        if "=" not in ov:
            continue
        key, val = ov.split("=", 1)
        parts = key.split(".")
        obj = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        attr = parts[-1]
        old_val = getattr(obj, attr)
        # 转为原类型。
        if   isinstance(old_val, bool):
            new_val = val.lower() in ("true", "1", "yes")
        elif isinstance(old_val, int):
            new_val = int(val)
        elif isinstance(old_val, float):
            new_val = float(val)
        elif isinstance(old_val, list):
            import json
            new_val = json.loads(val)
        else:
            new_val = val
        setattr(obj, attr, new_val)
        logging.getLogger(__name__).info("Override: %s = %s → %s", key, old_val, new_val)


def main():
    parser = argparse.ArgumentParser(description="3D Segmentation Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides (key=value)")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    if args.override:
        apply_overrides(cfg, args.override)
        cfg.sync()
        cfg.validate()

    # Setup
    setup_logging(cfg.train.output_dir, args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("Config loaded from: %s", args.config)

    seed_everything(cfg.train.seed, cfg.train.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s (%.1f GB)",
                     torch.cuda.get_device_name(),
                     torch.cuda.get_device_properties(0).total_memory / 1e9)

    # Build data
    train_loader, val_loader = build_dataloaders(cfg)

    # Build model
    model = build_model(cfg)

    # Optional: export pipeline visualization HTML (TODO #2). Guarded by
    # cfg.vis.enabled; CPU-only, no GPU / real data needed. Zero-overhead when
    # disabled. Placed before Trainer so the model is still on CPU / uncompiled.
    if cfg.vis.enabled:
        from .visualization import generate_visualization
        try:
            out = generate_visualization(cfg, model)
            logger.info("Pipeline visualization written to: %s", out)
        except Exception as e:  # 可视化失败不应中断训练
            logger.warning("Visualization generation failed: %s", e)

    # Save resolved config
    save_config(cfg, Path(cfg.train.output_dir) / "resolved_config.yaml")

    # Train
    trainer      = Trainer(model, cfg, train_loader, val_loader, device)
    best_metrics = trainer.fit()

    logger.info("Best metrics: %s", best_metrics)
    return best_metrics


if __name__ == "__main__":
    main()
