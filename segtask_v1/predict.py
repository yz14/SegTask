"""CLI entry point for 3D segmentation inference.

Usage:
    # Predict every .nii / .nii.gz file under a folder (recursive):
    python -m segtask_v1.predict --config configs/seg2_5d.yaml --checkpoint outputs/lung_bone_convnext/best_model.pth --input F:/CT_data/airway_segment_data/nii --output F:/CT_data/airway_segment_data/lung_bone_pred

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
from .data.loader import match_bbox_paths
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
    parser.add_argument("--bbox", type=str, default=None,
                        help="Optional ROI bbox NIfTI mask. Either a single "
                             "file (when --input is one file) or a directory "
                             "matching --input by filename. When omitted, "
                             "falls back to cfg.data.bbox_dir if set; "
                             "pass --bbox '' to force-disable.")
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

    # Resolve bbox source priority: explicit --bbox > cfg.data.bbox_dir.
    # Pass --bbox '' to force-disable even if cfg has bbox_dir set.
    bbox_paths = _resolve_bbox_paths(args.bbox, image_paths, cfg)
    if bbox_paths is not None:
        logger.info("BBox enabled: %d masks aligned with inputs.",
                    len(bbox_paths))

    run_inference(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        image_paths=image_paths,
        weight_variant=args.weights,
        bbox_paths=bbox_paths,
    )


def _resolve_bbox_paths(
    cli_bbox: object, image_paths: List[str], cfg) -> object:
    """Decide which bbox source to use and return a per-image path list
    (or None when bbox is disabled).

    Priority:
      1. ``--bbox`` explicit override:
         - empty string         → disable bbox (overrides cfg).
         - a single file path   → only valid when --input is one file.
         - a directory          → match by filename via ``match_bbox_paths``.
      2. Otherwise, fall back to ``cfg.data.bbox_dir`` (matched by name).
      3. If neither is set, return None (full-volume inference).
    """
    if cli_bbox is not None:
        if cli_bbox == "":
            return None
        p = Path(cli_bbox)
        if p.is_file():
            if len(image_paths) != 1:
                raise ValueError(
                    f"--bbox is a single file but --input expanded to "
                    f"{len(image_paths)} images; pass a directory instead.")
            return [str(p)]
        if not p.is_dir():
            raise FileNotFoundError(f"--bbox path not found: {p}")
        bbox_dir = str(p)
    else:
        bbox_dir = getattr(cfg.data, "bbox_dir", "") or ""
        if not bbox_dir:
            return None

    return match_bbox_paths(
        image_paths,
        bbox_dir,
        cfg.data.image_suffix,
        getattr(cfg.data, "bbox_suffix", ".nii.gz"))


if __name__ == "__main__":
    main()
