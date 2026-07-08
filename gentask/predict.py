"""gentask super-resolution inference CLI entry point."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from .config import load_config
from .predictor import run_generation_inference
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
    parser = argparse.ArgumentParser(description="gentask super-resolution inference")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config (use the same one as training)")
    parser.add_argument("--checkpoint", "--ckpt", type=str, default=None,
                        help="Checkpoint path; default <cfg.train.output_dir>/best_model.pth")
    parser.add_argument("--input", type=str, default=None,
                        help="NIfTI file or dir; default cfg.data.image_dir")
    parser.add_argument("--output", type=str, default=None,
                        help="Output dir; default cfg.predict.output_dir")
    parser.add_argument("--weights", choices=["auto", "ema", "online"],
                        default="auto",
                        help="Which weights to load from checkpoint")
    parser.add_argument("--input-grid", choices=["hr", "lr"], default=None,
                        help="Input grid: 'hr' (already on HR grid) or 'lr' "
                             "(real low-res, resampled to HR grid before the "
                             "network); default cfg.predict.input_grid")
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
    if not cfg.is_generation:
        raise ValueError("gentask.predict expects task.type='generation'.")

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

    if args.input_grid:
        cfg.predict.input_grid = args.input_grid

    if args.output:
        cfg.predict.output_dir = args.output

    Path(cfg.predict.output_dir).mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.predict.output_dir, args.log_level)
    logger = logging.getLogger(__name__)

    image_paths = _gather_nifti(input_path, recursive=not args.no_recursive)
    logger.info("Found %d NIfTI file(s) under %s", len(image_paths), input_path)
    logger.info("Checkpoint: %s (variant=%s)", checkpoint_path, args.weights)
    logger.info("Output dir: %s", cfg.predict.output_dir)

    run_generation_inference(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        image_paths=image_paths,
        weight_variant=args.weights,
        output_dir=cfg.predict.output_dir,
    )

if __name__ == "__main__":
    main()
