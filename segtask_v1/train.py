"""3D 分割训练 CLI 入口。示例：`python -m segtask_v1.train --config configs/seg3d.yaml [--override train.epochs=50 ...]`。

多卡：在 YAML 配 `train.gpus`（物理卡号列表，如 `[0, 2, 5, 7]`）即可启用 DDP，
本入口用 `torch.multiprocessing.spawn` 按列表每卡起一个进程（NCCL），只占指定卡，
照常 `python -m segtask_v1.train --config ...` 即可，无需手敲 torchrun / CUDA_VISIBLE_DEVICES。
`len(train.gpus) <= 1`（或 CPU）时完全走单进程历史路径，行为零变化。
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import yaml

# torch 2.4 的 torch.utils.checkpoint 内部仍调用已弃用的 torch.cpu.amp.autocast，
# 属 torch 自身告警；须在 import torch 之前注册，spawn 子进程 re-import 时同样生效。
warnings.filterwarnings(
    "ignore", message=r".*torch\.cpu\.amp\.autocast.*", category=FutureWarning)

import torch
import torch.multiprocessing as mp

from taskcore.config.core import Config, load_config, save_config
from taskcore.engine.launch import (
    find_free_port, finalize_ddp_worker, init_ddp_worker,
    maybe_enable_expandable_segments,
)
from taskcore.data.loader import build_dataloaders
from taskcore.utils.logging_utils import setup_logging as _setup_logging
from taskcore.models.factory import build_model
from .trainer import Trainer
from taskcore.engine.dist_utils import get_rank, get_world_size, is_main_process
from taskcore.utils.common import seed_everything


def setup_logging(output_dir: str, level: str = "INFO") -> None:
    """同时输出控制台（彩色）与文件（纯文本）日志。"""
    _setup_logging(output_dir=output_dir, level=level, log_filename="train.log")


def apply_overrides(cfg, overrides: list) -> None:
    """应用点记法 override。示例：--override train.epochs=50 model.backbone=convnext。"""
    for ov in overrides:
        if "=" not in ov:
            continue
        key, val = ov.split("=", 1)
        parts = key.split(".")
        obj = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        attr = parts[-1]
        old_val = getattr(obj, attr)
        # 转为原类型；默认值为 None 的 Optional 字段按 YAML 语义解析。
        if old_val is None:
            new_val = yaml.safe_load(val)
        elif isinstance(old_val, bool):
            new_val = val.lower() in ("true", "1", "yes")
        elif isinstance(old_val, int):
            new_val = int(val)
        elif isinstance(old_val, float):
            new_val = float(val)
        elif isinstance(old_val, list):
            import json
            new_val = json.loads(val)
        else:
            new_val = val
        setattr(obj, attr, new_val)
        logging.getLogger(__name__).info("Override: %s = %s → %s", key, old_val, new_val)


def _build_and_fit(cfg: Config, device: torch.device):
    """单进程内的"建数据/模型 → 训练"主体；单卡与各 DDP rank 共用。

    rank / world_size 取自已初始化的 process group（单进程为 0 / 1）。落盘类副作用
    （可视化导出、resolved_config）仅在 rank0 执行。
    """
    logger = logging.getLogger(__name__)
    rank, world_size = get_rank(), get_world_size()

    # Build data（DDP 下训练集经 DistributedSampler 切分到各 rank）。
    train_loader, val_loader = build_dataloaders(
        cfg, rank=rank, world_size=world_size)

    # Build model
    model = build_model(cfg)

    # Optional: export pipeline visualization HTML (TODO #2)。仅 rank0，CPU-only。
    if cfg.vis.enabled and is_main_process():
        from .visualization import generate_visualization
        try:
            out = generate_visualization(cfg, model)
            logger.info("Pipeline visualization written to: %s", out)
        except Exception as e:  # 可视化失败不应中断训练
            logger.warning("Visualization generation failed: %s", e)

    # Save resolved config（仅 rank0）。
    if is_main_process():
        save_config(cfg, Path(cfg.train.output_dir) / "resolved_config.yaml")

    trainer      = Trainer(model, cfg, train_loader, val_loader, device)
    best_metrics = trainer.fit()
    logger.info("Best metrics: %s", best_metrics)
    return best_metrics


def _train_worker(
    local_rank: int,
    gpus: list,
    cfg: Config,
    log_level: str,
    master_port: int,
) -> None:
    """每个 DDP 进程的入口（由 mp.spawn 调用，local_rank=0..world_size-1）。"""
    world_size = len(gpus)
    # 公用初始化（见 taskcore.engine.launch）：孤儿兜底 + 终止信号处理 +
    # NCCL 初始化 + 绑卡。
    device = init_ddp_worker(local_rank, gpus, cfg, master_port)

    # 日志：rank0 写文件 + 彩色控制台；其余进程仅控制台 WARNING，避免 N 倍刷屏。
    if local_rank == 0:
        setup_logging(cfg.train.output_dir, log_level)
        logger = logging.getLogger(__name__)
        logger.info(
            "DDP launched: world_size=%d on physical GPUs %s "
            "(MASTER_PORT=%s).", world_size, gpus, master_port)
    else:
        logging.basicConfig(level=logging.WARNING)

    # 逐 rank 偏移种子：解耦各 rank 的增强参数流与 DataLoader worker 采样流
    # （数据切分由 DistributedSampler 保证，模型初值由 DDP 构造时从 rank0 广播）。
    seed_everything(cfg.train.seed + local_rank, cfg.train.deterministic)

    completed = False
    try:
        _build_and_fit(cfg, device)
        completed = True
    finally:
        # 公用收尾：正常完成时同步屏障，异常路径直接销毁退出。
        finalize_ddp_worker(completed)


def main():
    parser = argparse.ArgumentParser(description="3D Segmentation Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides (key=value)")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    if args.override:
        apply_overrides(cfg, args.override)
        cfg.sync()
        cfg.validate()

    # 碎片治理：需在任何 CUDA 分配前注入（DDP 子进程经环境继承生效）。
    maybe_enable_expandable_segments(cfg)

    gpus    = [int(g) for g in cfg.train.gpus]
    cuda_ok = torch.cuda.is_available()
    use_ddp = cuda_ok and len(gpus) >= 2

    if use_ddp:
        # 多卡 DDP：每卡 spawn 一个进程。日志在各 worker 内按 rank 初始化。
        master_port = int(cfg.train.ddp_master_port) or find_free_port()
        mp.spawn(
            _train_worker,
            args=(gpus, cfg, args.log_level, master_port),
            nprocs=len(gpus),
            join=True)
        return None

    # ---- 单进程路径（CPU / 单卡），行为与历史一致 ----
    setup_logging(cfg.train.output_dir, args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("Config loaded from: %s", args.config)

    seed_everything(cfg.train.seed, cfg.train.deterministic)

    if cuda_ok:
        # 非空 gpus（单元素）时按指定物理卡跑（混用机选卡）；否则用默认 cuda:0。
        gpu_index = gpus[0] if gpus else 0
        torch.cuda.set_device(gpu_index)
        device = torch.device(f"cuda:{gpu_index}")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s (%.1f GB)",
                     torch.cuda.get_device_name(device),
                     torch.cuda.get_device_properties(device).total_memory / 1e9)

    return _build_and_fit(cfg, device)


if __name__ == "__main__":
    main()
