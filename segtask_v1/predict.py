"""CLI entry point for 3D segmentation inference.

Usage:
    # Predict every .nii / .nii.gz file under a folder (recursive):
    python -m segtask_v1.predict --config configs/seg2_5d.yaml --checkpoint outputs/seg2_5d_resnet/best_model.pth --input F:/CT_data/lung_ves/nii --output F:/CT_data/lung_ves/body_pred

    # Predict a single file:
    python -m segtask_v1.predict --config configs/seg2_5d.yaml \
        --checkpoint outputs/best_model.pth --input case_001.nii.gz

    # Force EMA / online weights, save probability maps, override predict cfg:
    python -m segtask_v1.predict ... --weights ema --save-probs \
        --override predict.batch_size=4 predict.tta_flip=true
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from .config import load_config
from .predictor import run_inference
from .train import apply_overrides, setup_logging


def _gather_nifti(input_path: str, recursive: bool = True) -> List[str]:
    """Collect .nii / .nii.gz files from a file or directory."""
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.is_file():
        return [str(p)]
    pattern_iter = p.rglob("*") if recursive else p.glob("*")
    files = sorted(
        str(f) for f in pattern_iter
        if f.is_file() and (f.name.endswith(".nii") or f.name.endswith(".nii.gz"))
    )
    if not files:
        raise FileNotFoundError(f"No .nii/.nii.gz under {p}")
    return files


def main():
    parser = argparse.ArgumentParser(description="3D Segmentation Inference")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config (use the same one as training)")
    parser.add_argument("--checkpoint", "--ckpt", type=str, required=True,
                        help="Path to trained checkpoint, e.g. outputs/best_model.pth")
    parser.add_argument("--input", type=str, required=True,
                        help="A NIfTI file OR a directory containing .nii/.nii.gz")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (overrides cfg.predict.output_dir)")
    parser.add_argument("--weights", choices=["auto", "ema", "online"],
                        default="auto",
                        help="Which weights to load from checkpoint")
    parser.add_argument("--save-probs", action="store_true",
                        help="Also save per-class sigmoid probability maps")
    parser.add_argument("--no-recursive", action="store_true",
                        help="Do not search subdirectories of --input")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Config overrides (e.g. predict.batch_size=4)")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.override:
        apply_overrides(cfg, args.override)
        cfg.sync()
        cfg.validate()

    if args.output:
        cfg.predict.output_dir = args.output
    if args.save_probs:
        cfg.predict.save_probabilities = True

    Path(cfg.predict.output_dir).mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.predict.output_dir, args.log_level)
    logger = logging.getLogger(__name__)

    image_paths = _gather_nifti(args.input, recursive=not args.no_recursive)
    logger.info("Found %d NIfTI file(s) under %s", len(image_paths), args.input)
    logger.info("Checkpoint: %s (variant=%s)", args.checkpoint, args.weights)
    logger.info("Output dir: %s", cfg.predict.output_dir)

    run_inference(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        image_paths=image_paths,
        weight_variant=args.weights,
    )


if __name__ == "__main__":
    main()
