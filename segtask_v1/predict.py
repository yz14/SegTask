"""3D 分割推理 CLI 入口。示例：`python -m segtask_v1.predict --config <yaml> --checkpoint <pth> --input <file_or_dir> [--output <dir>] [--weights ema|online|auto] [--save-probs] [--override k=v ...]`。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from taskcore.config.core import load_config
from taskcore.data.loader import match_bbox_paths_lenient
from .predictor import run_inference
from .train import apply_overrides, setup_logging


def _gather_nifti(input_path: str, recursive: bool = True) -> List[str]:
    """从文件或目录收集 .nii/.nii.gz。"""
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
    parser.add_argument("--checkpoint", "--ckpt", type=str, default=None,
                        help="Checkpoint path; default <cfg.train.output_dir>/best_model.pth")
    parser.add_argument("--input", type=str, default=None,
                        help="NIfTI file or dir; default cfg.data.image_dir")
    parser.add_argument("--output", type=str, default=None,
                        help="Output dir; default <input_parent>/<task_name>_pred")
    parser.add_argument("--bbox", type=str, default=None,
                        help="ROI bbox file/dir; '' disables; default falls back to cfg.data.bbox_dir")
    parser.add_argument("--weights", choices=["auto", "ema", "online"],
                        default="auto",
                        help="Which weights to load from checkpoint")
    parser.add_argument("--precision",
                        choices=["auto", "fp32", "bf16", "fp16"],
                        default="auto",
                        help="Inference precision. auto: follow cfg.train.amp_dtype. fp16 may NaN with ConvNeXt+LN.")
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

    # 默认解析：ckpt=<train_out>/best_model.pth；input=cfg.data.image_dir；output=<input_parent>/<task_name>_pred (task_name=basename(train_out))。
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        train_out = cfg.train.output_dir or ""
        if not train_out:
            parser.error("--checkpoint not given and cfg.train.output_dir is empty.")
        checkpoint_path = str(Path(train_out) / "best_model.pth")

    input_path = args.input
    if not input_path:
        input_path = cfg.data.image_dir or ""
        if not input_path:
            parser.error("--input not given and cfg.data.image_dir is empty.")

    if args.output:
        cfg.predict.output_dir = args.output
    else:
        train_out = cfg.train.output_dir or ""
        if not train_out:
            parser.error("--output not given and cfg.train.output_dir is empty; "
                         "cannot derive task name for default output dir.")
        task_name = Path(train_out).name
        base_dir = Path(input_path).parent
        cfg.predict.output_dir = str(base_dir / f"{task_name}_pred")

    if args.save_probs:
        cfg.predict.save_probabilities = True

    Path(cfg.predict.output_dir).mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.predict.output_dir, args.log_level)
    logger = logging.getLogger(__name__)

    image_paths = _gather_nifti(input_path, recursive=not args.no_recursive)
    logger.info("Found %d NIfTI file(s) under %s", len(image_paths), input_path)
    logger.info("Checkpoint: %s (variant=%s)", checkpoint_path, args.weights)
    logger.info("Output dir: %s", cfg.predict.output_dir)

    # bbox 优先级：--bbox > cfg.data.bbox_dir；宽容匹配丢弃未匹配图像。
    image_paths, bbox_paths = _resolve_bbox_paths(
        args.bbox, image_paths, cfg)
    if bbox_paths is not None:
        logger.info("BBox enabled: %d masks aligned with inputs.",
                    len(bbox_paths))
    if not image_paths:
        logger.error("No images left to predict after bbox matching. "
                     "Exiting without running inference.")
        return

    run_inference(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        image_paths=image_paths,
        weight_variant=args.weights,
        bbox_paths=bbox_paths,
        precision=args.precision,
    )


def _resolve_bbox_paths(
    cli_bbox: object, image_paths: List[str], cfg):
    """解析 bbox 源，返 (image_paths, bbox_paths|None)；image 可被过滤。优先：--bbox (''=禁/file=单/dir=宽容匹配) > cfg.data.bbox_dir > None。"""
    if cli_bbox is not None:
        if cli_bbox == "":
            return image_paths, None
        p = Path(cli_bbox)
        if p.is_file():
            if len(image_paths) != 1:
                raise ValueError(
                    f"--bbox is a single file but --input expanded to "
                    f"{len(image_paths)} images; pass a directory instead.")
            return image_paths, [str(p)]
        if not p.is_dir():
            raise FileNotFoundError(f"--bbox path not found: {p}")
        bbox_dir = str(p)
    else:
        bbox_dir = cfg.data.bbox_dir or ""
        if not bbox_dir:
            return image_paths, None

    matched_images, matched_bboxes = match_bbox_paths_lenient(
        image_paths,
        bbox_dir,
        cfg.data.image_suffix,
        cfg.data.bbox_suffix)
    return matched_images, matched_bboxes


if __name__ == "__main__":
    main()
