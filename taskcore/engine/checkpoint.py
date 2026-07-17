"""Checkpoint 工具：state_dict 解析、前缀剥离、compile 包装拆解。

主流程方法（``_build_state_dict`` / ``_save_checkpoint`` / ``_load_checkpoint``
/ ``_load_pretrain``）保留在 ``Trainer`` 类上，便于现有测试通过
``inspect.getsource(Trainer._build_state_dict)`` 校验关键 token；本模块仅承载
完全静态的辅助函数。
"""

from __future__ import annotations

import logging
import os
import random
import threading
from pathlib import Path
from queue import Queue
from typing import Callable, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def atomic_torch_save(state, path) -> None:
    """原子写 checkpoint：同目录临时文件 + ``os.replace``。

    写盘中途崩溃/磁盘满时目标路径要么保留旧文件要么不存在，
    不会出现半写的 ``best_model.pth`` 覆盖掉可用旧 best。失败时清理
    临时文件后重抛。"""
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    try:
        torch.save(state, tmp)
        os.replace(tmp, p)
    except BaseException:
        try:
            if tmp.is_file():
                tmp.unlink()
        except OSError:
            logger.warning("Failed to remove temp checkpoint %s", tmp)
        raise


# ``_build_state_dict`` 写入的 RNG 快照键集合；``state_to_cpu`` 据此识别并走
# bytes 打包路径，避免 ``.clone()`` 把 ``ByteTensor`` 降级为普通 ``uint8 Tensor``。
_RNG_STATE_MARKERS = frozenset({"torch_cpu", "numpy", "python"})


def _looks_like_rng_state(obj: object) -> bool:
    if not isinstance(obj, dict):
        return False
    keys = set(obj.keys())
    return _RNG_STATE_MARKERS.issubset(keys)


def _tensor_to_rng_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().to(torch.uint8).contiguous().numpy().tobytes()


def _rng_bytes_to_cpu_tensor(data: object) -> torch.Tensor:
    """反序列化 RNG 字节或历史 Tensor 为 ``set_rng_state`` 可接受的 CPU uint8 张量。"""
    if isinstance(data, (bytes, bytearray)):
        return torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()
    if isinstance(data, torch.Tensor):
        return data.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    raise TypeError(
        f"RNG cpu state must be bytes or Tensor, got {type(data).__name__}")


def pack_rng_state_for_save(rng: dict) -> dict:
    """把 RNG 快照打成 pickle 安全的 bytes 布局（供 async ``state_to_cpu`` 使用）。"""
    out: dict = {}
    tc = rng.get("torch_cpu")
    if tc is not None:
        out["torch_cpu"] = _tensor_to_rng_bytes(tc)
    tcuda = rng.get("torch_cuda")
    if tcuda is not None:
        out["torch_cuda"] = [_tensor_to_rng_bytes(t) for t in tcuda]
    if "numpy" in rng:
        out["numpy"] = rng["numpy"]
    if "python" in rng:
        out["python"] = rng["python"]
    return out


def restore_rng_state(rng: dict) -> None:
    """从 checkpoint 恢复 RNG（兼容 bytes 打包与历史 Tensor 格式）。"""
    tc = rng.get("torch_cpu")
    if tc is not None:
        torch.set_rng_state(_rng_bytes_to_cpu_tensor(tc))
    tcuda = rng.get("torch_cuda")
    if tcuda is not None and torch.cuda.is_available():
        restored = [_rng_bytes_to_cpu_tensor(t) for t in tcuda]
        torch.cuda.set_rng_state_all(restored)
    np_state = rng.get("numpy")
    if np_state is not None:
        import numpy as np
        np.random.set_state(np_state)
    py_state = rng.get("python")
    if py_state is not None:
        import random
        random.setstate(py_state)


def _iter_leaf_optimizers(
    optimizer: torch.optim.Optimizer,
) -> Iterable[torch.optim.Optimizer]:
    """遍历实际持有 ``.state`` 的叶子优化器（含 ZeRO 内层 ``.optim``）。"""
    seen: set[int] = set()
    stack = [optimizer]
    while stack:
        opt = stack.pop()
        oid = id(opt)
        if oid in seen:
            continue
        seen.add(oid)
        inner = getattr(opt, "optim", None)
        if inner is not None and inner is not opt:
            stack.append(inner)
        if hasattr(opt, "param_groups") and hasattr(opt, "state"):
            yield opt


def relocate_optimizer_state(optimizer: torch.optim.Optimizer) -> int:
    """把 per-param optimizer state 张量搬到对应参数所在 device。

    Resume 时 checkpoint 经 CPU 落盘 + ``map_location`` 重载后，Adam(fused) 的
    ``step`` / ``exp_avg`` 等可能分裂在 CPU 与 GPU 上，ZeRO 重分片会加剧这一问题；
    在 ``load_state_dict`` 之后统一对齐可避免 fused kernel 设备不一致错误。
    """
    relocated = 0
    for opt in _iter_leaf_optimizers(optimizer):
        for group in opt.param_groups:
            for param in group["params"]:
                if param is None:
                    continue
                state = opt.state.get(param)
                if not state:
                    continue
                target = param.device
                for key, value in list(state.items()):
                    if (isinstance(value, torch.Tensor)
                            and value.device != target):
                        state[key] = value.to(device=target, non_blocking=True)
                        relocated += 1
    if relocated:
        logger.debug(
            "Relocated %d optimizer state tensor(s) onto param devices.",
            relocated)
    return relocated


def state_to_cpu(obj):
    """递归把嵌套 state 里的张量深拷贝到 CPU（``detach().clone().cpu()``）。

    异步保存前必须做：state_dict 中的张量与在线参数共享存储，训练继续推进会
    原地改写；拷贝后后台线程持有的快照与训练解耦。RNG 快照走 ``bytes`` 打包，
    避免破坏 ``torch.set_rng_state`` 所需的 uint8 语义。"""
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone().cpu()
    if isinstance(obj, dict):
        if _looks_like_rng_state(obj):
            return pack_rng_state_for_save(obj)
        return type(obj)((k, state_to_cpu(v)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return type(obj)(state_to_cpu(v) for v in obj)
    return obj


class AsyncCheckpointSaver:
    """后台单线程 checkpoint 写盘器。

    ``submit`` 前调用方需自行 ``state_to_cpu`` 深拷贝；队列无上限但提交频率
    受 ``save_every`` 约束，实际在飞任务 ≤1–2 个。``wait()`` 阻塞至全部写完
    （训练收尾 / 需要立即读回 ckpt 前调用）。写盘异常记入日志并在下次
    ``wait()`` 时重新抛出，避免静默丢 checkpoint。"""

    def __init__(self) -> None:
        self._queue: "Queue[Optional[tuple]]" = Queue()
        self._error: "Optional[BaseException]" = None
        self._worker = threading.Thread(
            target=self._run, name="ckpt-saver", daemon=True)
        self._worker.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                state, path, on_done = item
                try:
                    atomic_torch_save(state, path)
                    if on_done is not None:
                        on_done()
                except BaseException as exc:  # noqa: BLE001 — 记录后转交 wait()
                    self._error = exc
                    logger.error(
                        "Async checkpoint save failed for %s", path,
                        exc_info=True)
            finally:
                self._queue.task_done()

    def submit(self, state, path,
               on_done: "Optional[Callable[[], None]]" = None) -> None:
        self._queue.put((state, path, on_done))

    def wait(self) -> None:
        """阻塞至队列清空；若期间有写盘异常则抛出。"""
        self._queue.join()
        if self._error is not None:
            err, self._error = self._error, None
            raise RuntimeError("Async checkpoint save failed") from err

    def close(self) -> None:
        """排空队列并结束后台线程（幂等性不保证，只在收尾调用一次）。"""
        self.wait()
        self._queue.put(None)
        self._worker.join()


def unwrap_compile(m: nn.Module) -> nn.Module:
    """剥 ``torch.compile`` 的 ``_orig_mod`` 包装。"""
    return getattr(m, "_orig_mod", m)


def extract_model_state_dict(ckpt, prefer_ema: bool):
    """定位 ckpt 里的 model state_dict，兼容 3 种布局：

    * 本 trainer ckpt（含 ``model_state_dict`` / ``model_online_state_dict``）
    * 第三方 ``{"state_dict": ...}``
    * 裸 ``OrderedDict``

    Returns
    -------
    (state_dict, source_label)
    """
    # 裸 state_dict
    if not isinstance(ckpt, dict) or all(
            isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt, "raw_state_dict"

    # 优先 EMA shadow
    if prefer_ema and "ema_state_dict" in ckpt:
        ema_state = ckpt["ema_state_dict"]
        if isinstance(ema_state, dict) and "shadow" in ema_state:
            return ema_state["shadow"], "ema_shadow"

    # trainer-format 在线权重
    if "model_online_state_dict" in ckpt:
        return ckpt["model_online_state_dict"], "model_online_state_dict"
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], "model_state_dict"

    # 第三方 state_dict
    if "state_dict" in ckpt:
        return ckpt["state_dict"], "state_dict"

    raise KeyError(
        "Pretrain checkpoint does not contain a recognisable model "
        "state_dict. Expected one of: 'model_state_dict', "
        "'model_online_state_dict', 'state_dict', or a raw OrderedDict.")


def strip_common_prefixes(sd):
    """剥去 ``module.``（DDP）与 ``_orig_mod.``（torch.compile）前缀。"""
    if not isinstance(sd, dict):
        return sd
    prefixes = ("module.", "_orig_mod.")
    out = {}
    changed = False
    for k, v in sd.items():
        new_k = k
        # 反复剥防嵌套包装。
        while new_k.startswith(prefixes):
            for p in prefixes:
                if new_k.startswith(p):
                    new_k = new_k[len(p):]
                    changed = True
                    break
        out[new_k] = v
    return out if changed else sd


def _strip_compile_prefix(sd):
    """剥去 torch.compile 添加的 ``_orig_mod.`` 前缀。"""
    prefix = "_orig_mod."
    if isinstance(sd, dict) and any(k.startswith(prefix) for k in sd):
        return {(k[len(prefix):] if k.startswith(prefix) else k): v
                for k, v in sd.items()}
    return sd


def _unwrap_ema_state(ema_sd):
    """将 ``{shadow, decay}`` 拆为普通 state_dict；已是拆过的旧格式原返。"""
    if isinstance(ema_sd, dict) and "shadow" in ema_sd and isinstance(
            ema_sd["shadow"], dict):
        return ema_sd["shadow"]
    return ema_sd


def _select_state_dict(ckpt, variant: str):
    """从 ckpt 选权重。``variant``: ``'auto'`` (优 EMA) / ``'ema'`` / ``'online'``。

    返 ``(state_dict, label)``，``label`` 用于日志。
    """
    has_online = "model_online_state_dict" in ckpt
    has_ema = "ema_state_dict" in ckpt
    primary = ckpt["model_state_dict"]

    if variant == "online":
        return (ckpt["model_online_state_dict"] if has_online else primary,
                "online")
    if variant == "ema":
        if has_ema:
            return _unwrap_ema_state(ckpt["ema_state_dict"]), "ema"
        return (ckpt["model_online_state_dict"] if has_online else primary,
                "online")
    if has_ema:
        return _unwrap_ema_state(ckpt["ema_state_dict"]), "ema"
    return primary, "online"


def snapshot_rng_state() -> dict:
    """快照 RNG 状态以支持位精确 resume（torch CPU/CUDA + numpy + python）。"""
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else None),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


__all__ = [
    "AsyncCheckpointSaver",
    "state_to_cpu",
    "unwrap_compile",
    "extract_model_state_dict",
    "strip_common_prefixes",
    "pack_rng_state_for_save",
    "restore_rng_state",
    "relocate_optimizer_state",
    "_select_state_dict",
    "_strip_compile_prefix",
    "_unwrap_ema_state",
    "snapshot_rng_state",
]
