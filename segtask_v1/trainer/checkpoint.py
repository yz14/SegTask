"""Checkpoint 工具：state_dict 解析、前缀剥离、compile 包装拆解。

主流程方法（``_build_state_dict`` / ``_save_checkpoint`` / ``_load_checkpoint``
/ ``_load_pretrain``）保留在 ``Trainer`` 类上，便于现有测试通过
``inspect.getsource(Trainer._build_state_dict)`` 校验关键 token；本模块仅承载
完全静态的辅助函数。
"""

from __future__ import annotations

import logging
import threading
from queue import Queue
from typing import Callable, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def state_to_cpu(obj):
    """递归把嵌套 state 里的张量深拷贝到 CPU（``detach().clone().cpu()``）。

    异步保存前必须做：state_dict 中的张量与在线参数共享存储，训练继续推进会
    原地改写；拷贝后后台线程持有的快照与训练解耦。非张量对象原样返回（RNG
    state / 标量 / config 等本身即不可变或已是拷贝）。"""
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone().cpu()
    if isinstance(obj, dict):
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
                    torch.save(state, path)
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


__all__ = [
    "unwrap_compile",
    "extract_model_state_dict",
    "strip_common_prefixes",
]
