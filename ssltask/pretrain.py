"""自监督预训练（SSL）CLI 入口。

示例：``python -m ssltask.pretrain --config configs/ssltask_genesis.yaml``。

独立 task：用与下游同构的骨干（``segtask_v1.models.factory.build_model`` 的 enc/dec）
在**无标注** image patch 上做自监督预训练，产出 ``<output_dir>/ssl_best.pt``。随后下游
分割/分类训练用 ``--override train.pretrain=<该路径>`` 即可经已有的非严格加载衔接
（enc(+dec) 命中、任务头随机）。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from segtask_v1.logging_utils import setup_logging as _setup_logging
from segtask_v1.utils import seed_everything

from .config import apply_overrides, load_config, save_config, validate_ssl
from .data.ssl_dataset import build_ssl_dataloader
from .methods import build_method
from .trainer import SSLTrainer


def setup_logging(output_dir: str, level: str = "INFO") -> None:
    _setup_logging(output_dir=output_dir, level=level, log_filename="pretrain.log")


def main():
    parser = argparse.ArgumentParser(description="Self-Supervised Pretraining (ssltask)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Config overrides (key=value); ssl.* routes to SSLConfig")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    cfg, ssl = load_config(args.config)
    if args.override:
        apply_overrides(cfg, ssl, args.override)
        cfg.sync()
        cfg.validate()
        validate_ssl(ssl, cfg)

    setup_logging(cfg.train.output_dir, args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("SSL config loaded from: %s (method=%s)", args.config, ssl.method)

    seed_everything(cfg.train.seed, cfg.train.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    train_loader = build_ssl_dataloader(cfg)
    method = build_method(cfg, ssl, device)

    save_config(cfg, ssl, Path(cfg.train.output_dir) / "resolved_ssl_config.yaml")

    trainer = SSLTrainer(method, cfg, ssl, train_loader, device)
    metrics = trainer.fit()
    logger.info("SSL pretrain metrics: %s", metrics)
    return metrics


if __name__ == "__main__":
    main()
