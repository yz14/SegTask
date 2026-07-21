"""分类训练 CLI 入口。

示例：``python -m clstask.train --config configs/cls3d.yaml``。

复用 SSL/分割编码器时：``--override cls.pretrained_ckpt=<ssl_best.pt>``——
只取 ``encoder.*`` 权重（strict=False），几何须与预训练一致
（patch_mode/spatial_dims/in_channels 耦合，见 ``config.validate_cls``）。

多卡：在 YAML 配 ``train.gpus``（物理卡号列表）即可启用 DDP，本入口用
``torch.multiprocessing.spawn`` 按列表每卡起一个进程（NCCL），与 segtask
同模式；``len(train.gpus) <= 1``（或 CPU）时完全走单进程历史路径。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.multiprocessing as mp

from taskcore.engine.dist_utils import get_rank, get_world_size, is_main_process
from taskcore.engine.launch import (
    find_free_port, finalize_ddp_worker, init_ddp_worker,
    maybe_enable_expandable_segments,
)
from taskcore.utils.logging_utils import setup_logging as _setup_logging
from taskcore.utils.common import seed_everything

from .config import apply_overrides, load_config, save_config, validate_cls, validate_core
from .data.loader import build_cls_dataloaders
from .models.factory import build_classifier
from .trainer.cls_trainer import ClsTrainer


def setup_logging(output_dir: str, level: str = "INFO") -> None:
    _setup_logging(output_dir=output_dir, level=level, log_filename="train.log")


def _build_and_fit(cfg, cls, device: torch.device):
    """单进程内的"建数据/模型 → 训练"主体；单卡与各 DDP rank 共用。"""
    logger = logging.getLogger(__name__)
    rank, world_size = get_rank(), get_world_size()

    train_loader, val_loader = build_cls_dataloaders(
        cfg, cls, rank=rank, world_size=world_size)
    model = build_classifier(cfg, cls)

    if is_main_process():
        save_config(cfg, cls,
                    Path(cfg.train.output_dir) / "resolved_cls_config.yaml")

    trainer = ClsTrainer(model, cfg, cls, train_loader, val_loader, device)
    metrics = trainer.fit()
    logger.info("clstask training metrics: %s", metrics)
    return metrics


def _train_worker(local_rank: int, gpus: list, cfg, cls,
                  log_level: str, master_port: int) -> None:
    """每个 DDP 进程的入口（由 mp.spawn 调用，与 segtask train.py 同模式）。"""
    world_size = len(gpus)
    device = init_ddp_worker(local_rank, gpus, cfg, master_port)

    # 日志：rank0 写文件 + 彩色控制台；其余进程仅控制台 WARNING。
    if local_rank == 0:
        setup_logging(cfg.train.output_dir, log_level)
        logging.getLogger(__name__).info(
            "clstask DDP launched: world_size=%d on physical GPUs %s "
            "(MASTER_PORT=%s).", world_size, gpus, master_port)
    else:
        logging.basicConfig(level=logging.WARNING)

    # 逐 rank 偏移种子：解耦各 rank 的增强/采样流（数据切分由
    # DistributedSampler 保证，模型初值由 DDP 构造时从 rank0 广播）。
    seed_everything(cfg.train.seed + local_rank, cfg.train.deterministic)

    completed = False
    try:
        _build_and_fit(cfg, cls, device)
        completed = True
    finally:
        finalize_ddp_worker(completed)


def main():
    parser = argparse.ArgumentParser(description="Classification training (clstask)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Config overrides (key=value); cls.* routes to ClsConfig")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    cfg, cls = load_config(args.config)
    if args.override:
        apply_overrides(cfg, cls, args.override)
        cfg.sync()
        validate_core(cfg)
        validate_cls(cls, cfg)

    # 碎片治理：需在任何 CUDA 分配前注入（DDP 子进程经环境继承生效）。
    maybe_enable_expandable_segments(cfg)

    gpus = [int(g) for g in cfg.train.gpus]
    cuda_ok = torch.cuda.is_available()

    if cuda_ok and len(gpus) >= 2:
        master_port = int(cfg.train.ddp_master_port) or find_free_port()
        mp.spawn(
            _train_worker,
            args=(gpus, cfg, cls, args.log_level, master_port),
            nprocs=len(gpus),
            join=True)
        return None

    # ---- 单进程路径（CPU / 单卡），行为与历史一致 ----
    setup_logging(cfg.train.output_dir, args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("clstask config loaded from: %s (backbone=%s, granularity=%s)",
                args.config, cls.backbone, cls.label_granularity)

    seed_everything(cfg.train.seed, cfg.train.deterministic)
    if cuda_ok:
        gpu_index = gpus[0] if gpus else 0
        torch.cuda.set_device(gpu_index)
        device = torch.device(f"cuda:{gpu_index}")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    return _build_and_fit(cfg, cls, device)


if __name__ == "__main__":
    main()
