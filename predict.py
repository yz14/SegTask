"""CLI entry point for prediction / inference.

Usage:
    python predict.py --config configs/default.yaml --checkpoint outputs/best_model.pth --input path/to/image.nii.gz
    python predict.py --config configs/default.yaml --checkpoint outputs/best_model.pth --input-dir path/to/images/
"""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import torch

from segtask.config import load_config
from segtask.models.factory import build_model
from segtask.predictor import Predictor
from segtask.utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SegTask Prediction")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, default="", help="Single NIfTI file to predict")
    parser.add_argument("--input-dir", type=str, default="", help="Directory of NIfTI files")
    parser.add_argument("--output-dir", type=str, default="predictions", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, cpu")
    parser.add_argument("--save-prob", action="store_true", help="Save probability maps")
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config)
    setup_logging(args.output_dir, log_level="INFO")

    if args.tta:
        cfg.predict.tta_enabled = True

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    # Build model + predictor
    model = build_model(cfg)
    predictor = Predictor(model, cfg, device, checkpoint_path=args.checkpoint)

    # Gather input files
    input_files = []
    if args.input:
        input_files.append(args.input)
    elif args.input_dir:
        for ext in ["*.nii.gz", "*.nii"]:
            input_files.extend(glob.glob(str(Path(args.input_dir) / ext)))
        input_files.sort()

    if not input_files:
        logger.error("No input files found. Use --input or --input-dir.")
        return

    logger.info("Predicting %d files...", len(input_files))

    for i, fpath in enumerate(input_files):
        logger.info("[%d/%d] %s", i + 1, len(input_files), fpath)
        predictor.predict_and_save(
            image_path=fpath,
            output_dir=args.output_dir,
            label_values=cfg.data.label_values,
            save_probabilities=args.save_prob,
        )

    logger.info("All predictions saved to: %s", args.output_dir)


if __name__ == "__main__":
    main()
