"""3D 分割训练 CLI 入口。示例：`python -m segtask_v1.train --config configs/seg3d.yaml [--override train.epochs=50 ...]`。

多卡：在 YAML 配 `train.gpus`（物理卡号列表，如 `[0, 2, 5, 7]`）即可启用 DDP，
本入口用 `torch.multiprocessing.spawn` 按列表每卡起一个进程（NCCL），只占指定卡，
照常 `python -m segtask_v1.train --config ...` 即可，无需手敲 torchrun / CUDA_VISIBLE_DEVICES。
`len(train.gpus) <= 1`（或 CPU）时完全走单进程历史路径，行为零变化。
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .config import Config, load_config, save_config
from .data.loader import build_dataloaders
from .logging_utils import setup_logging as _setup_logging
from .models.factory import build_model
from .trainer import Trainer
from .trainer.dist_utils import get_rank, get_world_size, is_main_process
from .utils import seed_everything


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
        # 转为原类型。
        if   isinstance(old_val, bool):
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


def _find_free_port() -> int:
    """挑一个空闲 TCP 端口作 DDP rendezvous（混用机避免端口冲突）。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


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


def _install_parent_death_signal() -> None:
    """Linux：父进程一旦死亡，内核立即向本子进程发 SIGKILL。

    `mp.spawn` 起的 worker 是非 daemon 进程；父进程若被硬杀 / 终端关闭 / OOM-kill /
    崩溃，子进程会被 init 收养成孤儿，且可能卡在 NCCL collective 上永久挂起、一直占
    显存。`PR_SET_PDEATHSIG` 让内核在父进程退出时无条件杀掉本进程，是孤儿挂死的兜底。
    含竞态处理：若设置前父进程已退出，主动自杀。非 Linux 平台为 no-op。
    """
    if sys.platform != "linux":
        return
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1  # <linux/prctl.h>
        parent_before = os.getppid()
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
        if os.getppid() != parent_before:  # 设置前父进程已死，错过信号 → 自尽
            os._exit(1)
    except Exception as e:  # 设置失败不应阻断训练，仅告警
        logging.getLogger(__name__).warning("PR_SET_PDEATHSIG 设置失败：%s", e)


def _install_term_handlers() -> None:
    """捕获 SIGTERM/SIGINT：尽力销毁 process group 后立即退出，避免半初始化占卡。"""
    def _handler(signum, _frame):
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        finally:
            os._exit(128 + signum)
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass  # 非主线程等场景无法注册，忽略


def _train_worker(
    local_rank: int,
    gpus: list,
    cfg: Config,
    log_level: str,
    master_port: int,
) -> None:
    """每个 DDP 进程的入口（由 mp.spawn 调用，local_rank=0..world_size-1）。"""
    physical_gpu = int(gpus[local_rank])
    world_size   = len(gpus)

    # 防孤儿挂死：父死即死 + 优雅响应终止信号（须在初始化 NCCL 之前装好）。
    _install_parent_death_signal()
    _install_term_handlers()
    # 让 NCCL watchdog 在 collective 超时时 abort 卡住的通信（使 timeout 真正生效）。
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(master_port)

    torch.cuda.set_device(physical_gpu)
    dist.init_process_group(
        backend="nccl", world_size=world_size, rank=local_rank,
        timeout=timedelta(minutes=int(cfg.train.ddp_timeout_minutes)))
    device = torch.device(f"cuda:{physical_gpu}")

    # 日志：rank0 写文件 + 彩色控制台；其余进程仅控制台 WARNING，避免 N 倍刷屏。
    if local_rank == 0:
        setup_logging(cfg.train.output_dir, log_level)
        logger = logging.getLogger(__name__)
        logger.info(
            "DDP launched: world_size=%d on physical GPUs %s "
            "(MASTER_PORT=%s).", world_size, gpus, master_port)
    else:
        logging.basicConfig(level=logging.WARNING)

    seed_everything(cfg.train.seed, cfg.train.deterministic)

    completed = False
    try:
        _build_and_fit(cfg, device)
        completed = True
    finally:
        if dist.is_initialized():
            # 仅在正常完成时同步收尾屏障；异常路径若 barrier 会卡等已死的 peer，
            # 故跳过、直接销毁退出（叠加 NCCL 超时双保险）。
            if completed:
                try:
                    dist.barrier()
                except Exception:
                    pass
            dist.destroy_process_group()


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

    gpus    = [int(g) for g in cfg.train.gpus]
    cuda_ok = torch.cuda.is_available()
    use_ddp = cuda_ok and len(gpus) >= 2

    if use_ddp:
        # 多卡 DDP：每卡 spawn 一个进程。日志在各 worker 内按 rank 初始化。
        master_port = int(cfg.train.ddp_master_port) or _find_free_port()
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
