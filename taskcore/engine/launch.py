"""多卡 DDP 启动公用件（mp.spawn 每卡一进程；单卡/CPU 路径零变化）。

各任务 CLI 入口共用的"启动工程"：空闲端口挑选、孤儿进程兜底、终止信号
处理、allocator 碎片治理、worker 进程的 NCCL 初始化/收尾。任务侧只需提供
模块级 worker 函数（mp.spawn 要求可 pickle），其主体为：

    def _train_worker(local_rank, gpus, cfg, log_level, master_port):
        device = init_ddp_worker(local_rank, gpus, cfg, master_port)
        ...  # rank0 配日志、seed 偏移、_build_and_fit
        finalize_ddp_worker(completed)

主入口尾部按 ``len(cfg.train.gpus) >= 2 and cuda`` 决定 spawn 还是单进程。
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import sys
from datetime import timedelta

import torch
import torch.distributed as dist

__all__ = [
    "find_free_port",
    "install_parent_death_signal",
    "install_term_handlers",
    "maybe_enable_expandable_segments",
    "init_ddp_worker",
    "finalize_ddp_worker",
]

logger = logging.getLogger(__name__)


def find_free_port() -> int:
    """挑一个空闲 TCP 端口作 DDP rendezvous（混用机避免端口冲突）。"""
    last_error = None
    for _ in range(8):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("", 0))
            return s.getsockname()[1]
        except OSError as exc:
            last_error = exc
        finally:
            s.close()
    raise OSError("could not allocate a rendezvous port") from last_error


def install_parent_death_signal() -> None:
    """Linux：父进程一旦死亡，内核立即向本子进程发 SIGKILL。

    ``mp.spawn`` 起的 worker 是非 daemon 进程；父进程若被硬杀 / 终端关闭 /
    OOM-kill / 崩溃，子进程会被 init 收养成孤儿，且可能卡在 NCCL collective
    上永久挂起、一直占显存。``PR_SET_PDEATHSIG`` 让内核在父进程退出时无条件
    杀掉本进程，是孤儿挂死的兜底。含竞态处理：若设置前父进程已退出，主动
    自杀。非 Linux 平台为 no-op。
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
        logger.warning("PR_SET_PDEATHSIG 设置失败：%s", e)


def install_term_handlers() -> None:
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


def maybe_enable_expandable_segments(cfg) -> None:
    """按 train.cuda_expandable_segments 注入 allocator 碎片治理配置。

    必须在首次 CUDA 分配前设置（含 DDP spawn 前，子进程继承环境）；已有
    PYTORCH_CUDA_ALLOC_CONF 时不覆盖，尊重用户显式设置。默认开关关闭时
    零副作用。
    """
    if not cfg.train.cuda_expandable_segments:
        return
    if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
        logger.warning(
            "train.cuda_expandable_segments=True 但环境已设置 "
            "PYTORCH_CUDA_ALLOC_CONF=%r，不覆盖。",
            os.environ["PYTORCH_CUDA_ALLOC_CONF"])
        return
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def init_ddp_worker(
    local_rank: int,
    gpus: list,
    cfg,
    master_port: int,
) -> torch.device:
    """DDP worker 进程的公共初始化；返回本 rank 绑定的 device。

    顺序固定：孤儿兜底/信号处理须在初始化 NCCL 之前装好。
    """
    physical_gpu = int(gpus[local_rank])
    world_size = len(gpus)

    install_parent_death_signal()
    install_term_handlers()
    # 让 NCCL watchdog 在 collective 超时时 abort 卡住的通信（使 timeout 真正生效）。
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(master_port)

    # DDP 后端固定 NCCL（CUDA 集合通信仅 NCCL 高效可靠）；NCCL 仅支持
    # Linux，其他平台在进坑前显式报错，而不是留给 init 深处的晦涩崩溃。
    if not dist.is_nccl_available():
        raise RuntimeError(
            "Multi-GPU DDP training requires the NCCL backend, which is "
            "only available on Linux (this PyTorch build reports NCCL "
            f"unavailable; platform={sys.platform!r}). On Windows, run "
            "single-GPU training (train.gpus with one entry), or use "
            "WSL2/Linux for multi-GPU.")

    torch.cuda.set_device(physical_gpu)
    dist.init_process_group(
        backend="nccl", world_size=world_size, rank=local_rank,
        timeout=timedelta(minutes=int(cfg.train.ddp_timeout_minutes)))
    return torch.device(f"cuda:{physical_gpu}")


def finalize_ddp_worker(completed: bool) -> None:
    """DDP worker 收尾：正常完成时同步屏障后销毁 process group。

    异常路径若 barrier 会卡等已死的 peer，故跳过、直接销毁退出（叠加
    NCCL 超时双保险）。
    """
    if dist.is_initialized():
        if completed:
            try:
                dist.barrier()
            except Exception:
                pass
        dist.destroy_process_group()
