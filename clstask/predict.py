"""分类推理 CLI 入口。

示例::

    python -m clstask.predict --config configs/cls3d.yaml \
        --ckpt outputs/cls3d/best_model.pth \
        --npz-dir data/npz --out-dir outputs/cls3d/preds
"""

from __future__ import annotations

import argparse
import logging

import torch

from taskcore.utils.logging_utils import setup_logging as _setup_logging

from .config import apply_overrides, load_config, validate_cls, validate_core
from .data.loader import discover_npz
from .models.factory import build_classifier
from .predictor.cls_predictor import ClsPredictor


def main():
    parser = argparse.ArgumentParser(description="Classification inference (clstask)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to best_model.pth")
    parser.add_argument("--npz-dir", type=str, default="",
                        help="npz dir to predict (default: data.npz_dir)")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--use-ema", action="store_true",
                        help="Prefer EMA weights from the checkpoint")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    cfg, cls = load_config(args.config)
    if args.override:
        apply_overrides(cfg, cls, args.override)
        cfg.sync()
        validate_core(cfg)
        validate_cls(cls, cfg)

    _setup_logging(output_dir=args.out_dir, level=args.log_level,
                   log_filename="predict.log")
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 构建时跳过训练期的预训练加载（权重马上被 ckpt 覆盖）。
    cls.pretrained_ckpt = ""
    model = build_classifier(cfg, cls)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=True)
    if args.use_ema and "ema_state_dict" in ckpt:
        ema = ckpt["ema_state_dict"].get("shadow", {})
        if ema:
            # shadow 只含参数（不含 buffer），叠加在完整权重之上。
            model.load_state_dict(ema, strict=False)
            logger.info("EMA weights overlaid from checkpoint.")
    logger.info("Checkpoint loaded from %s (epoch=%s)",
                args.ckpt, ckpt.get("epoch"))

    paths = discover_npz(args.npz_dir or cfg.data.npz_dir, cfg.data.npz_suffix)
    predictor = ClsPredictor(model, cfg, cls, device)
    predictor.predict_dir(paths, args.out_dir)
    logger.info("Predictions written to %s", args.out_dir)


if __name__ == "__main__":
    main()
