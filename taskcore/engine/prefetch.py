"""CUDA H2D 预取器：在 GPU 计算当前 batch 的同时，用独立 copy stream 把下一个
batch 从（pinned）host 内存异步拷到 device，隐藏 H2D 传输延迟。

数值完全等价 —— 仅改变拷贝与计算的重叠方式，不改变任何张量内容或顺序。
需 ``data.pin_memory=True`` 才能真正异步（pageable 内存下 cudaMemcpyAsync
会退化为同步拷贝，预取收益归零但仍正确）。

用法（``train.prefetch_to_gpu=True`` 时由 Trainer 包装 train_loader）::

    for batch in CudaPrefetcher(loader, device):
        image = batch["image"]          # 已在 device 上
        ...

流同步语义：

* copy stream 上发起 ``to(device, non_blocking=True)``；
* 交付 batch 前当前流 ``wait_stream(copy_stream)``，保证消费可见性；
* 对每个已交付张量 ``record_stream(current_stream)``，防止其显存块在
  当前流仍在使用时被 allocator 回收复用（PyTorch 跨流生命周期惯例）。

非 CUDA device 或非张量字段原样透传（此时退化为普通迭代，零行为差异）。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, Optional

import torch

__all__ = ["CudaPrefetcher"]


class CudaPrefetcher:
    """把 dict-of-tensors batch 迭代器包装为"提前一个 batch 上卡"的迭代器。

    仅移动 ``torch.Tensor`` 值；其余键值（如字符串 pid）原样保留。
    ``device`` 非 CUDA 时直接透传底层迭代（no-op 包装）。
    """

    def __init__(self, loader: Iterable[Dict[str, Any]],
                 device: torch.device) -> None:
        self.loader = loader
        self.device = device
        self._use_cuda = device.type == "cuda"
        self._stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=device) if self._use_cuda else None)

    def __len__(self) -> int:
        return len(self.loader)  # type: ignore[arg-type]

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        def move(v: Any) -> Any:
            if isinstance(v, torch.Tensor):
                return v.to(self.device, non_blocking=True)
            if isinstance(v, list) and v and all(
                    isinstance(t, torch.Tensor) for t in v):
                # fused 模式：native 面内 slab 以 list[Tensor] 交付。
                return [t.to(self.device, non_blocking=True) for t in v]
            return v
        return {k: move(v) for k, v in batch.items()}

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if not self._use_cuda:
            yield from self.loader
            return

        it = iter(self.loader)
        stream = self._stream
        assert stream is not None

        # 预取首个 batch。
        try:
            next_cpu = next(it)
        except StopIteration:
            return
        with torch.cuda.stream(stream):
            next_gpu = self._to_device(next_cpu)

        while True:
            # 当前流等待 copy stream 完成本 batch 的 H2D，随后即可安全消费。
            torch.cuda.current_stream(self.device).wait_stream(stream)
            batch = next_gpu
            for v in batch.values():
                if isinstance(v, torch.Tensor) and v.is_cuda:
                    v.record_stream(torch.cuda.current_stream(self.device))
                elif isinstance(v, list):
                    for t in v:
                        if isinstance(t, torch.Tensor) and t.is_cuda:
                            t.record_stream(
                                torch.cuda.current_stream(self.device))

            # 在 copy stream 上发起下一个 batch 的拷贝（与主流计算重叠）。
            try:
                next_cpu = next(it)
            except StopIteration:
                yield batch
                return
            with torch.cuda.stream(stream):
                next_gpu = self._to_device(next_cpu)

            yield batch
