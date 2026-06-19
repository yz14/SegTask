"""双批（金标准 + 粗标）混合训练采样工具。

提供两块互相独立、可单测的组件：

* :class:`SourceTaggedDataset` —— 轻量包装任意 ``Dataset``，在 ``__getitem__``
  返回的字典里加一个 ``"source"`` 标量张量（0=金标准 / 1=粗标），其余字段透传。
  现有 dataset 子类无需改动。
* :class:`MixedBatchSampler` —— 在 ``ConcatDataset`` 的全局索引空间上工作，保证
  每个 batch 同时含金标准与粗标，且按整数配额混合。粗标每个 epoch 顺序消费一遍，
  金标准循环重采样（适合"金少粗多"）。

二者均不依赖 trainer / specs，``DataLoader`` 通过 ``batch_sampler=`` 接入。
"""

from __future__ import annotations

import logging
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)

# source 标签约定。
SOURCE_PRIMARY   = 0   # 金标准
SOURCE_SECONDARY = 1   # 粗标


class SourceTaggedDataset(Dataset):
    """包装一个 dataset，向其样本字典注入 ``"source"`` 标量张量。

    透传 ``__len__`` 与底层 ``__getitem__`` 的全部字段；仅追加 ``"source"``。
    供 ``ConcatDataset`` 串联金/粗两源后，让 batch 携带来源信息，为后续 loss
    对粗标降权预留接口。
    """

    def __init__(self, base: Dataset, source_id: int) -> None:
        super().__init__()
        self.base = base
        self.source_id = int(source_id)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.base[idx]
        if not isinstance(sample, dict):
            raise TypeError(
                f"SourceTaggedDataset expects dict samples, got "
                f"{type(sample).__name__} from {type(self.base).__name__}.")
        sample["source"] = torch.tensor(self.source_id, dtype=torch.long)
        return sample

    # 暴露底层属性（如 _npz_paths），便于 loader 的缓存足迹估计等复用。
    def __getattr__(self, name: str):  # pragma: no cover - 仅转发
        # __getattr__ 仅在常规查找失败时触发，避免与 self.base/source_id 冲突。
        return getattr(self.__dict__["base"], name)


def resolve_per_batch_counts(
    mix_ratio: List[int], batch_size: int) -> tuple[int, int]:
    """把整数权重比 ``[gold, coarse]`` 解析为每 batch 的 (gold, coarse) 计数。

    要求 ``batch_size`` 能被 ``sum(mix_ratio)`` 整除、两元素均 >= 1。
    与 ``Config._validate_data`` 中的校验一致（此处再次断言以便独立单测）。
    """
    if len(mix_ratio) != 2:
        raise ValueError(f"mix_ratio must be [gold, coarse]; got {mix_ratio}.")
    g, c = int(mix_ratio[0]), int(mix_ratio[1])
    if g < 1 or c < 1:
        raise ValueError(
            f"mix_ratio elements must be >= 1 (each batch needs both "
            f"sources); got {mix_ratio}.")
    s = g + c
    if batch_size % s != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be divisible by sum(mix_ratio) "
            f"({s}).")
    k = batch_size // s
    return g * k, c * k


class MixedBatchSampler(Sampler[List[int]]):
    """在 concat 索引空间上产出"固定金/粗配比"的 batch。

    约定 concat 布局为 ``[primary | secondary]``：primary 占全局索引
    ``[0, n_primary)``，secondary 占 ``[n_primary, n_primary + n_secondary)``。

    每个 ``__iter__``（= 每个 epoch）：

    * 粗标(secondary)整体随机重排，按 ``coarse_per_batch`` 顺序消费，整轮恰好覆盖一遍；
    * 金标准(primary)按 ``gold_per_batch`` 取，耗尽即重排续取（循环过采样）；
    * epoch 长度 ``__len__ = n_secondary // coarse_per_batch``。

    Args:
        n_primary:        金标准展开样本数（= num_vols * samples_per_volume）。
        n_secondary:      粗标展开样本数。
        gold_per_batch:   每 batch 金标准数（>= 1）。
        coarse_per_batch: 每 batch 粗标数（>= 1）。
        seed:             基础随机种子；每 epoch 自增以获得可复现且各异的顺序。
    """

    def __init__(
        self,
        n_primary: int,
        n_secondary: int,
        gold_per_batch: int,
        coarse_per_batch: int,
        seed: int = 0) -> None:
        super().__init__(data_source=None)
        if gold_per_batch < 1 or coarse_per_batch < 1:
            raise ValueError(
                f"gold_per_batch and coarse_per_batch must be >= 1; got "
                f"{gold_per_batch}, {coarse_per_batch}.")
        if n_primary < gold_per_batch:
            raise ValueError(
                f"n_primary ({n_primary}) < gold_per_batch "
                f"({gold_per_batch}); need at least one full gold quota.")
        if n_secondary < coarse_per_batch:
            raise ValueError(
                f"n_secondary ({n_secondary}) < coarse_per_batch "
                f"({coarse_per_batch}); cannot form a single mixed batch.")

        self.n_primary        = int(n_primary)
        self.n_secondary      = int(n_secondary)
        self.gold_per_batch   = int(gold_per_batch)
        self.coarse_per_batch = int(coarse_per_batch)
        self._base_seed       = int(seed)
        self._epoch           = 0

        # secondary 在 concat 中的全局索引偏移。
        self._sec_offset = self.n_primary
        self._num_batches = self.n_secondary // self.coarse_per_batch

        logger.info(
            "MixedBatchSampler: gold=%d, coarse=%d samples; per-batch "
            "gold=%d + coarse=%d (batch_size=%d); %d batches/epoch "
            "(coarse-bound). Gold is cycled/oversampled ~%.2fx per epoch.",
            self.n_primary, self.n_secondary,
            self.gold_per_batch, self.coarse_per_batch,
            self.gold_per_batch + self.coarse_per_batch,
            self._num_batches,
            (self._num_batches * self.gold_per_batch)
            / max(self.n_primary, 1))

    def __len__(self) -> int:
        return self._num_batches

    def _gold_stream(self, rng: np.random.Generator) -> Iterator[int]:
        """无限金标准索引流：每耗尽一轮即重排续发。"""
        while True:
            perm = rng.permutation(self.n_primary)
            for i in perm:
                yield int(i)

    def __iter__(self) -> Iterator[List[int]]:
        # 每 epoch 用独立但可复现的 RNG。
        rng = np.random.default_rng(self._base_seed + self._epoch)
        self._epoch += 1

        sec_perm   = rng.permutation(self.n_secondary)
        gold_iter  = self._gold_stream(rng)

        for b in range(self._num_batches):
            start = b * self.coarse_per_batch
            sec_chunk = sec_perm[start:start + self.coarse_per_batch]
            # secondary 全局索引 = 局部索引 + 偏移。
            batch: List[int] = [self._sec_offset + int(i) for i in sec_chunk]
            for _ in range(self.gold_per_batch):
                batch.append(next(gold_iter))   # primary 全局索引 == 局部索引
            # 打散 batch 内顺序，避免来源位置固定带来的潜在偏置。
            rng.shuffle(batch)
            yield batch
