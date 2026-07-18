"""检测推理 CLI 入口。

示例::

    python -m dettask.predict --config configs/det2_5d.yaml \
        --ckpt outputs/det2_5d/best_model.pth \
        --npz-dir data/npz --out-dir outputs/det2_5d/preds
"""

from __future__ import annotations

import argparse
import logging

import torch

from taskcore.utils.logging_utils import setup_logging as _setup_logging

from taskcore.data.loader import discover_npz_recursive as discover_npz

from .config import apply_overrides, load_config, validate_det
from .models.factory import build_detector
from .predictor.det_predictor import DetPredictor


def main():
    parser = argparse.ArgumentParser(description="Detection inference (dettask)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to best_model.pth")
    parser.add_argument("--npz-dir", type=str, default="",
                        help="npz dir to predict (default: data.npz_dir)")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--use-ema", action="store_true",
                        help="Prefer EMA weights from the checkpoint")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip volume-level FROC evaluation")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    cfg, det = load_config(args.config)
    if args.override:
        apply_overrides(cfg, det, args.override)
        cfg.sync()
        cfg.validate()
        validate_det(det, cfg)

    _setup_logging(output_dir=args.out_dir, level=args.log_level,
                   log_filename="predict.log")
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 构建时跳过训练期的预训练加载（权重马上被 ckpt 覆盖）。
    det.pretrained_ckpt = ""
    model = build_detector(cfg, det)
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
    predictor = DetPredictor(model, cfg, det, device)
    metrics = predictor.predict_dir(paths, args.out_dir,
                                    evaluate=not args.no_eval)
    logger.info("Predictions written to %s; metrics: %s", args.out_dir, metrics)


if __name__ == "__main__":
    main()
