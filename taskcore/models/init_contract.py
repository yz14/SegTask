"""模块级初始化契约声明（3-2，timm ``init_weights`` / MMEngine ``init_cfg`` 思路）。

带自定义初始化的模块（zero-init 残差出口、ICNR、近零 offset、LayerScale 等）
在构造期调用 :func:`declare_no_reinit` 自声明；全局 ``model.init_strategy``
（``_apply_init_strategy``）只对未声明的参数生效，不越权覆盖。

设计：
* 声明落在**模块构造处**（契约与实现同址，单一真相源），工厂不再维护
  isinstance 白名单——新增自初始化模块只需在自身 ``__init__`` 声明一次。
* 注册表为 ``WeakSet``（弱引用），不延长参数生命周期、不进 state_dict、
  不影响 checkpoint 兼容性。
"""

from __future__ import annotations

import weakref
from typing import Union

import torch.nn as nn

# 全局弱引用注册表：已声明"由模块自身初始化、全局策略不得覆盖"的参数。
# 以 id(param) 为键、弱引用参数为值：参数被 GC 后条目自动移除（id 复用安全）；
# 不能用 WeakSet——其 __contains__ 走 Tensor.__eq__（逐元素比较）。
_SELF_INIT_PARAMS: "weakref.WeakValueDictionary[int, nn.Parameter]" = (
    weakref.WeakValueDictionary())


def declare_no_reinit(*items: Union[nn.Module, nn.Parameter]) -> None:
    """声明参数/模块（含全部子模块参数）带自定义初始化契约。

    在模块 ``__init__`` 中完成自定义初始化后调用；此后无论
    ``model.init_strategy`` 取何值，这些参数都不会被全局遍历重初始化。
    """
    for item in items:
        if isinstance(item, nn.Parameter):
            _SELF_INIT_PARAMS[id(item)] = item
        elif isinstance(item, nn.Module):
            for p in item.parameters():
                _SELF_INIT_PARAMS[id(p)] = p
        else:
            raise TypeError(
                f"declare_no_reinit expects nn.Module or nn.Parameter; "
                f"got {type(item).__name__}.")


def protected_param_ids(model: nn.Module) -> set:
    """返回 ``model`` 中已声明契约的参数 ``id`` 集合（供初始化遍历跳过）。"""
    return {id(p) for p in model.parameters()
            if _SELF_INIT_PARAMS.get(id(p)) is p}


def is_protected(param: nn.Parameter) -> bool:
    """单参数查询：是否已声明初始化契约。"""
    return _SELF_INIT_PARAMS.get(id(param)) is param


__all__ = ["declare_no_reinit", "protected_param_ids", "is_protected"]
