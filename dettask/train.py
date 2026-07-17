"""检测训练 CLI 入口。

示例：``python -m dettask.train --config configs/det2_5d.yaml``。

复用 SSL/分割权重时：``--override det.pretrained_ckpt=<ssl_best.pt>``——
取 ``encoder.*``（重建式 SSL 亦命中 ``decoder.*``，strict=False），几何须
与预训练一致（patch_mode/spatial_dims/in_channels 耦合，见
``config.validate_det``）。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from taskcore.utils.logging_utils import setup_logging as _setup_logging
from taskcore.utils.common import seed_everything

from .config import apply_overrides, load_config, save_config, validate_det
from .data.loader import build_det_dataloaders
from .models.factory import build_detector
from .trainer.det_trainer import DetTrainer


def setup_logging(output_dir: str, level: str = "INFO") -> None:
    _setup_logging(output_dir=output_dir, level=level, log_filename="train.log")


def main():
    parser = argparse.ArgumentParser(description="Detection training (dettask)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Config overrides (key=value); det.* routes to DetConfig")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    cfg, det = load_config(args.config)
    if args.override:
        apply_overrides(cfg, det, args.override)
        cfg.sync()
        cfg.validate()
        validate_det(det, cfg)

    setup_logging(cfg.train.output_dir, args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("dettask config loaded from: %s (arch=%s, patch_mode=%s)",
                args.config, det.arch, cfg.data.patch_mode)

    seed_everything(cfg.train.seed, cfg.train.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    train_loader, val_loader = build_det_dataloaders(cfg, det)
    model = build_detector(cfg, det)

    save_config(cfg, det, Path(cfg.train.output_dir) / "resolved_det_config.yaml")

    trainer = DetTrainer(model, cfg, det, train_loader, val_loader, device)
    metrics = trainer.fit()
    logger.info("dettask training metrics: %s", metrics)
    return metrics


if __name__ == "__main__":
    main()
