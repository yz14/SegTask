"""分类训练 CLI 入口。

示例：``python -m clstask.train --config configs/cls3d.yaml``。

复用 SSL/分割编码器时：``--override cls.pretrained_ckpt=<ssl_best.pt>``——
只取 ``encoder.*`` 权重（strict=False），几何须与预训练一致
（patch_mode/spatial_dims/in_channels 耦合，见 ``config.validate_cls``）。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from segtask_v1.logging_utils import setup_logging as _setup_logging
from segtask_v1.utils import seed_everything

from .config import apply_overrides, load_config, save_config, validate_cls
from .data.loader import build_cls_dataloaders
from .models.factory import build_classifier
from .trainer.cls_trainer import ClsTrainer


def setup_logging(output_dir: str, level: str = "INFO") -> None:
    _setup_logging(output_dir=output_dir, level=level, log_filename="train.log")


def main():
    parser = argparse.ArgumentParser(description="Classification training (clstask)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Config overrides (key=value); cls.* routes to ClsConfig")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    cfg, cls = load_config(args.config)
    if args.override:
        apply_overrides(cfg, cls, args.override)
        cfg.sync()
        cfg.validate()
        validate_cls(cls, cfg)

    setup_logging(cfg.train.output_dir, args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("clstask config loaded from: %s (backbone=%s, granularity=%s)",
                args.config, cls.backbone, cls.label_granularity)

    seed_everything(cfg.train.seed, cfg.train.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    train_loader, val_loader = build_cls_dataloaders(cfg, cls)
    model = build_classifier(cfg, cls)

    save_config(cfg, cls, Path(cfg.train.output_dir) / "resolved_cls_config.yaml")

    trainer = ClsTrainer(model, cfg, cls, train_loader, val_loader, device)
    metrics = trainer.fit()
    logger.info("clstask training metrics: %s", metrics)
    return metrics


if __name__ == "__main__":
    main()
