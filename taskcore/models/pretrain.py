"""共享的部分 checkpoint 权重加载工具。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from ..engine.checkpoint import extract_model_state_dict, strip_common_prefixes

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PretrainedLoadResult:
    """公共加载结果，保留每个命名空间是否存在及实际命中数。"""

    total_matched: int
    source: str
    prefix_has_keys: Mapping[str, bool]
    prefix_matched: Mapping[str, int]


def load_pretrained_modules(
    modules: Mapping[str, nn.Module],
    ckpt_path: str,
    *,
    zero_match_error: str,
    raise_on_zero: bool = True,
) -> PretrainedLoadResult:
    """按 ``prefix.`` 将 checkpoint 权重加载到多个模块。

    返回实际加载的 tensor 数及来源；调用方可在此基础上保留任务专属
    的前缀错误和 0 命中错误语义。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    full_sd, source = extract_model_state_dict(ckpt, prefer_ema=False)
    sd = strip_common_prefixes(full_sd)
    total = 0
    prefix_has_keys = {}
    prefix_matched = {}
    for prefix, module in modules.items():
        prefix = prefix.rstrip(".") + "."
        sub = {k[len(prefix):]: v for k, v in sd.items()
               if k.startswith(prefix)}
        prefix_has_keys[prefix] = bool(sub)
        own = module.state_dict()
        matched = {
            k: v for k, v in sub.items()
            if k in own and getattr(v, "shape", None) == own[k].shape}
        if matched:
            module.load_state_dict(matched, strict=False)
        logger.info(
            "Pretrained %s* matched %d/%d tensors from %s.",
            prefix, len(matched), len(own), ckpt_path)
        prefix_matched[prefix] = len(matched)
        total += len(matched)
    if total == 0 and raise_on_zero:
        raise RuntimeError(zero_match_error)
    return PretrainedLoadResult(
        total, source, prefix_has_keys, prefix_matched)


__all__ = ["PretrainedLoadResult", "load_pretrained_modules"]
