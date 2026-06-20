"""自监督预训练（SSL）CLI 入口。

示例：``python -m segtask_v1.pretrain --config configs/ssl_genesis.yaml``。

与 ``segtask_v1.train`` 对称的独立 task：用与分割同构的 UNet（仅换重建头）在
（无标注或忽略标签的）image patch 上做 Models Genesis 式重建预训练，产出
``<output_dir>/ssl_best.pt``。随后分割训练用 ``--override train.pretrain=<该路径>``
即可经已有的非严格加载衔接（enc+dec 命中、seg head 随机）。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from .config import load_config, save_config
from .data.loader import build_dataloaders
from .logging_utils import setup_logging as _setup_logging
from .models.factory import build_ssl_model
from .train import apply_overrides
from .trainer.ssl_trainer import SSLTrainer
from .utils import seed_everything


def setup_logging(output_dir: str, level: str = "INFO") -> None:
    _setup_logging(output_dir=output_dir, level=level, log_filename="pretrain.log")


def main():
    parser = argparse.ArgumentParser(description="Self-Supervised Pretraining (SSL)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides (key=value)")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # SSL 入口隐式开启 ssl.enabled（确保走 SSL 校验路径），用户也可在 YAML 显式设置。
    if not cfg.ssl.enabled:
        cfg.ssl.enabled = True
    if args.override:
        apply_overrides(cfg, args.override)
    cfg.sync()
    cfg.validate()

    setup_logging(cfg.train.output_dir, args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("SSL config loaded from: %s", args.config)

    seed_everything(cfg.train.seed, cfg.train.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # 复用分割数据管线（忽略 label，仅取 image patch 做重建）。
    train_loader, _val_loader = build_dataloaders(cfg)

    model = build_ssl_model(cfg)

    save_config(cfg, Path(cfg.train.output_dir) / "resolved_ssl_config.yaml")

    trainer = SSLTrainer(model, cfg, train_loader, device)
    metrics = trainer.fit()
    logger.info("SSL pretrain metrics: %s", metrics)
    return metrics


if __name__ == "__main__":
    main()
