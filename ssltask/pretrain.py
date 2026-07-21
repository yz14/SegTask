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
import torch.distributed as dist
import torch.multiprocessing as mp

from taskcore.utils.logging_utils import setup_logging as _setup_logging
from taskcore.engine.launch import (
    find_free_port, finalize_ddp_worker, init_ddp_worker,
    maybe_enable_expandable_segments)
from taskcore.utils.common import seed_everything

from .config import apply_overrides, load_config, save_config, validate_ssl, validate_core
from .data.ssl_dataset import build_ssl_dataloader
from .methods import build_method
from .trainer import SSLTrainer


def setup_logging(output_dir: str, level: str = "INFO") -> None:
    _setup_logging(output_dir=output_dir, level=level, log_filename="pretrain.log")


def _build_and_fit(cfg, ssl, device: torch.device):
    logger = logging.getLogger(__name__)
    train_loader = build_ssl_dataloader(cfg, ssl)
    method = build_method(cfg, ssl, device)

    if not dist.is_initialized() or dist.get_rank() == 0:
        save_config(cfg, ssl,
                    Path(cfg.train.output_dir) / "resolved_ssl_config.yaml")

    trainer = SSLTrainer(method, cfg, ssl, train_loader, device)
    metrics = trainer.fit()
    logger.info("SSL pretrain metrics: %s", metrics)
    return metrics


def _pretrain_worker(local_rank: int, gpus: list, cfg, ssl,
                     log_level: str, master_port: int) -> None:
    """每个 DDP 进程的入口（由 mp.spawn 调用，与 segtask train.py 同模式）。"""
    world_size = len(gpus)
    device = init_ddp_worker(local_rank, gpus, cfg, master_port)

    if local_rank == 0:
        setup_logging(cfg.train.output_dir, log_level)
        logging.getLogger(__name__).info(
            "SSL DDP launched: world_size=%d on physical GPUs %s "
            "(MASTER_PORT=%s).", world_size, gpus, master_port)
    else:
        logging.basicConfig(level=logging.WARNING)

    # 逐 rank 偏移种子：解耦各 rank 的采样流（数据切分由 DistributedSampler
    # 保证，模型初值由 SSLTrainer 构造时从 rank0 广播）。
    seed_everything(cfg.train.seed + local_rank, cfg.train.deterministic)

    completed = False
    try:
        _build_and_fit(cfg, ssl, device)
        completed = True
    finally:
        finalize_ddp_worker(completed)


def main():
    parser = argparse.ArgumentParser(description="Self-Supervised Pretraining (ssltask)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides (key=value); ssl.* routes to SSLConfig")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    cfg, ssl = load_config(args.config)
    if args.override:
        apply_overrides(cfg, ssl, args.override)
        cfg.sync()
        validate_core(cfg)
        validate_ssl(ssl, cfg)

    maybe_enable_expandable_segments(cfg)

    gpus = [int(g) for g in cfg.train.gpus]
    cuda_ok = torch.cuda.is_available()
    if cuda_ok and len(gpus) >= 2:
        # 多卡 DDP：每卡 spawn 一个进程（与 segtask train.py 同模式）。
        master_port = int(cfg.train.ddp_master_port) or find_free_port()
        mp.spawn(
            _pretrain_worker,
            args=(gpus, cfg, ssl, args.log_level, master_port),
            nprocs=len(gpus),
            join=True)
        return None

    # ---- 单进程路径（CPU / 单卡），行为与历史一致 ----
    setup_logging(cfg.train.output_dir, args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("SSL config loaded from: %s (method=%s)", args.config, ssl.method)

    seed_everything(cfg.train.seed, cfg.train.deterministic)

    if cuda_ok:
        gpu_index = gpus[0] if gpus else 0
        torch.cuda.set_device(gpu_index)
        device = torch.device(f"cuda:{gpu_index}")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    return _build_and_fit(cfg, ssl, device)


if __name__ == "__main__":
    main()
