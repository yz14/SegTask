"""模型流 builder：``输入 → encoder → decoder → 各输出头 → 损失``。

与旧实现（按叶子 hook 的执行先后串成一条线性链）不同，这里**按真实张量数据流重建
DAG**，从而正确呈现：并联分支（如 multi-stem 三路并行）、残差捷径、encoder→decoder
跳连、多输入融合，以及各输出头的真实来源。

采集与重建策略：
1. 用一份 **CPU dummy 全零张量**（batch=1，尺寸取 pipeline 目标 patch）跑一次前向，
   对**所有模块**（含容器）注册 forward-pre / forward hook，记录每个模块输入/输出
   **张量的 ``id()``** 与真实形状；
2. 前向在 ``model.train()`` 下进行（``no_grad``），以便 aux / deep-supervision / topo
   等"仅训练期输出"的头也被激活、出现在图中；
3. 由"叶子输出张量 id → 产出叶子"建立产出映射；对每个图节点，用其**自身输入张量 id**
   反查上游产出节点 → 得到真实连边（天然含并联扇入扇出、跳连、多输入融合）；
4. 残差捷径按 block 类型结构化识别后单独画一条 ``residual`` 边；
5. 前向若抛错（自定义 forward 签名等），降级为 ``eval()`` 重试；仍失败则退化为
   **纯结构图**（只读 ``named_modules`` 层级，按声明序线性链）。

叶子按所属容器（encoder.stem / encoder.stages.k / decoder.levels.k / *_head）聚合成
可折叠 stage 大框；容器内再按 block（``blocks.k`` / ``stems.k`` / ``proj`` …）分组。
"""

from __future__ import annotations

import logging
import weakref
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn

from ..config import Config
from ..models.topology import ModelTopology, build_topology
from .data_flow import _model_input_shape, _target_patch_size
from .graph import VisGraph, VisNode, shape_str

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 叶子分类 / 参数抽取
# ---------------------------------------------------------------------------
_ACT_TYPES = (
    nn.ReLU, nn.LeakyReLU, nn.GELU, nn.SiLU, nn.ELU, nn.PReLU,
    nn.Sigmoid, nn.Tanh, nn.Hardswish, nn.Mish, nn.Softmax, nn.GLU,
)


def _leaf_kind(m: nn.Module) -> str:
    """把模块映射到 IR 语义类别（conv / norm / act / op）。"""
    if isinstance(m, nn.modules.conv._ConvNd):
        return "conv"
    norm_bases = (
        nn.modules.batchnorm._NormBase, nn.GroupNorm, nn.LayerNorm,
        nn.modules.instancenorm._InstanceNorm,
    )
    if isinstance(m, norm_bases):
        return "norm"
    if isinstance(m, _ACT_TYPES):
        return "act"
    return "op"


def _leaf_params(m: nn.Module) -> Dict[str, object]:
    """按模块类型抽取关键超参（用于详情抽屉）。"""
    p: Dict[str, object] = {}
    if isinstance(m, nn.modules.conv._ConvNd):
        p["in_channels"] = m.in_channels
        p["out_channels"] = m.out_channels
        p["kernel_size"] = m.kernel_size
        p["stride"] = m.stride
        p["padding"] = m.padding
        if any(d != 1 for d in m.dilation):
            p["dilation"] = m.dilation
        if m.groups != 1:
            p["groups"] = m.groups
        p["bias"] = m.bias is not None
    elif isinstance(m, nn.modules.batchnorm._NormBase):
        p["num_features"] = m.num_features
        p["eps"] = m.eps
        p["momentum"] = m.momentum
    elif isinstance(m, nn.GroupNorm):
        p["num_groups"] = m.num_groups
        p["num_channels"] = m.num_channels
        p["eps"] = m.eps
    elif isinstance(m, (nn.LayerNorm, nn.modules.instancenorm._InstanceNorm)):
        p["normalized"] = getattr(m, "normalized_shape",
                                  getattr(m, "num_features", "?"))
    elif isinstance(m, nn.Linear):
        p["in_features"] = m.in_features
        p["out_features"] = m.out_features
    n_param = sum(prm.numel() for prm in m.parameters(recurse=False))
    if n_param:
        p["params"] = n_param
    return p


# ---------------------------------------------------------------------------
# 容器 / block 分组
# ---------------------------------------------------------------------------
# 透明包裹容器名：分组时跳过（仅作语义包裹，无需单独成框）。
_WRAPPER_SEGS = {"stage"}


# encoder/decoder 下"按索引切分"的 ModuleList 容器段：每个索引各成一个顶层框。
# 覆盖标准 UNet（stages/downsamples/levels/upsamples）、ADM（levels）、
# EDM2（level_blocks/level_entries）、UNet++/UNet3+（blocks/upsamples/gates/
# branches/fusions，键可为 i_j 复合）。
_INDEXED_SEGS = {
    "stages", "downsamples", "levels", "upsamples", "level_blocks",
    "level_entries", "blocks", "gates", "branches", "fusions",
}


def _top_key(name: str) -> str:
    """叶子限定名 → 所属**顶层容器**键（encoder.stem / encoder.stages.k / …）。

    与旧 ``_stage_key`` 的关键区别：``encoder.stem.stems.k`` 不再各自成顶层框，而是
    统一归到单一 ``encoder.stem`` 容器（其内部三路并联另由 block 分组呈现）；并对各
    decoder 变体（UNet/UNet++/UNet3+/ADM/EDM2）按索引切分出每级一框。
    """
    parts = name.split(".")
    p0 = parts[0]
    if p0 in ("encoder", "decoder"):
        if p0 == "encoder" and len(parts) >= 2 and parts[1] == "stem":
            return "encoder.stem"
        if p0 == "encoder" and len(parts) >= 2 and parts[1] == "aux_fuse":
            return "encoder.aux_fuse"
        # 按索引切分的 ModuleList 段：索引可为数字或 i_j 复合键。
        if (len(parts) >= 3 and parts[1] in _INDEXED_SEGS
                and (parts[2].isdigit() or "_" in parts[2])):
            return f"{p0}.{parts[1]}.{parts[2]}"
        return ".".join(parts[:2]) if len(parts) >= 2 else p0
    if p0 in ("seg_head", "topo_head", "recon_head"):
        return p0
    if p0 in ("ds_heads", "aux_heads") and len(parts) >= 2:
        return f"{p0}.{parts[1]}"
    return p0


def _block_key(name: str, top: str) -> Optional[str]:
    """容器内的 block 级分组键（``blocks.k`` / ``stems.k`` / ``proj`` / ``upsample`` …）。

    返回的是**真实模块路径**（保留 ``stage`` 等包裹段，以便后续按路径取回模块做
    类型 / 残差判定）；显示标题由 ``_block_label`` 去掉包裹段。``None`` 表示该叶子
    直接挂在顶层容器下（无中间 block 层）。
    """
    if not name.startswith(top + "."):
        return None
    rem = name[len(top) + 1:].split(".")
    segs: List[str] = []
    i = 0
    while i < len(rem):
        seg = rem[i]
        segs.append(seg)
        if seg in _WRAPPER_SEGS:           # 'stage' 等透明包裹：保留路径但继续下探
            i += 1
            continue
        if i + 1 < len(rem) and rem[i + 1].isdigit():
            segs.append(rem[i + 1])        # 形如 blocks.0 / stems.1 的索引块
        break
    if not segs:
        return None
    block = top + "." + ".".join(segs)
    if block == name:                      # block 即叶子本身 → 视为直接子节点
        return None
    return block


def _top_label(key: str) -> str:
    """顶层容器键 → 人类可读标题。"""
    parts = key.split(".")
    pretty = {
        "encoder.stem": "Encoder Stem",
        "encoder.aux_fuse": "Encoder Aux-Fuse",
        "seg_head": "Seg Head (main)",
        "topo_head": "Topo Head",
        "recon_head": "Recon Head",
    }
    if key in pretty:
        return pretty[key]
    if len(parts) == 3:
        p0, seg, idx = parts
        enc = p0 == "encoder"
        seg_label = {
            "stages": "Encoder Stage", "downsamples": "Downsample",
            "levels": ("Encoder Level" if enc else "Decoder Level"),
            "upsamples": "Upsample",
            "level_blocks": ("Enc Blocks" if enc else "Dec Blocks"),
            "level_entries": ("Enc Entry" if enc else "Dec Entry"),
            "branches": "Dec Branches", "fusions": "Dec Fuse",
        }.get(seg)
        if seg_label:
            return f"{seg_label} {idx}"
        if p0 == "decoder" and seg in ("blocks", "gates"):
            # UNet++ 嵌套节点 X[i,j]（键形如 i_j）。
            return f"Decoder X[{idx.replace('_', ',')}]"
        if p0 == "decoder":
            return f"Decoder {seg} {idx}"
    if len(parts) == 2 and parts[0] == "ds_heads":
        return f"DS Head {parts[1]}"
    if len(parts) == 2 and parts[0] == "aux_heads":
        return f"Aux Head {parts[1]}"
    return key


def _block_label(block_key: str, top: str) -> str:
    """block 键 → 框内子标题（取相对 top 的尾段，去掉 ``stage`` 等包裹段）。"""
    rel = block_key[len(top) + 1:] if block_key.startswith(top + ".") else block_key
    parts = [p for p in rel.split(".") if p not in _WRAPPER_SEGS]
    return ".".join(parts) if parts else rel


def _is_head(key: str) -> bool:
    return (key in ("seg_head", "topo_head", "recon_head")
            or key.startswith("ds_heads.")
            or key.startswith("aux_heads."))


def _head_edge_label(key: str) -> str:
    if key == "seg_head":
        return "main"
    if key == "topo_head":
        return "topo"
    if key.startswith("ds_heads."):
        return "ds"
    if key.startswith("aux_heads."):
        return "aux"
    return ""


# ---------------------------------------------------------------------------
# 残差 block 识别
# ---------------------------------------------------------------------------
def _residual_block_types() -> Tuple[type, ...]:
    """已知"自带残差/捷径"的 block 类型（懒导入，缺失则跳过）。"""
    types: List[type] = []
    try:
        from ..models.resnet import (
            BottleneckBlock, PreActResNetBlock, R2Plus1DBlock, ResNetBlock,
        )
        types += [ResNetBlock, PreActResNetBlock, BottleneckBlock, R2Plus1DBlock]
    except Exception as e:  # pragma: no cover - 仅日志
        logger.debug("model_flow: 导入 resnet 残差块类型失败: %s", e)
    try:
        from ..models.convnext import ConvNeXtBlock
        types.append(ConvNeXtBlock)
    except Exception as e:  # pragma: no cover
        logger.debug("model_flow: 导入 convnext 残差块类型失败: %s", e)
    try:  # 扩散系 backbone 的残差块（输出 = 残差支 + skip(输入)）
        from ..models.adm_unet import _ResBlockNoEmb
        types.append(_ResBlockNoEmb)
    except Exception as e:  # pragma: no cover
        logger.debug("model_flow: 导入 adm 残差块类型失败: %s", e)
    try:
        from ..models.edm2_unet import _Block as _EDM2Block
        types.append(_EDM2Block)
    except Exception as e:  # pragma: no cover
        logger.debug("model_flow: 导入 edm2 残差块类型失败: %s", e)
    return tuple(types)


# 残差捷径子模块的常见命名（不同 backbone 各异）。
_SHORTCUT_ATTRS = ("shortcut", "skip_connection", "conv_skip")


def _has_residual(m: nn.Module, residual_types: Tuple[type, ...]) -> bool:
    """该 block 是否含残差捷径（``out = 残差支 + skip(x)`` 形态）。"""
    if residual_types and isinstance(m, residual_types):
        return True
    # 通用启发：含名为 shortcut / skip_connection / conv_skip 的子模块即视为残差块。
    return any(isinstance(getattr(m, a, None), nn.Module) for a in _SHORTCUT_ATTRS)


# ---------------------------------------------------------------------------
# 张量数据流追踪
# ---------------------------------------------------------------------------
class _ModuleRec:
    """单个模块一次前向的输入/输出张量 id 与形状记录。"""

    __slots__ = ("name", "module", "order", "in_ids", "out_ids",
                 "in_shape", "out_shape", "in_prov", "in_src", "in_op")

    def __init__(self, name: str, module: nn.Module):
        self.name = name
        self.module = module
        self.order: int = -1
        self.in_ids: Tuple[int, ...] = ()
        self.out_ids: Tuple[int, ...] = ()
        self.in_shape: Optional[Tuple[int, ...]] = None
        self.out_shape: Optional[Tuple[int, ...]] = None
        # 各输入张量在 pre_hook 时刻解析出的血缘（上游叶子名集合），与 in_ids 对齐。
        # 在 pre_hook 解析可保证此刻输入张量仍存活、id 唯一，从而 prov 命中正确，
        # 既免去张量钉住（省内存），又杜绝 id 复用导致的 build 期陈旧读。
        self.in_prov: Tuple["frozenset[str]", ...] = ()
        # 各输入张量的**直接产出叶子**名（weakref 校验，非直接产出则为 None），
        # 供 block 内叶子流式连边；同样在 pre_hook 解析以规避 id 复用。
        self.in_src: Tuple[Optional[str], ...] = ()
        # 各输入张量若由**多源融合算子**（cat/+/…）产生，则记其归一化符号（"cat"/
        # "+"/…），否则为 None；供步骤 D 在融合点插入显式 merge 节点并正确标注。
        self.in_op: Tuple[Optional[str], ...] = ()


def _iter_tensors(obj) -> List[torch.Tensor]:
    """从任意 args / kwargs 结构里扁平收集张量。"""
    out: List[torch.Tensor] = []
    if torch.is_tensor(obj):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            out += _iter_tensors(x)
    elif isinstance(obj, dict):
        for x in obj.values():
            out += _iter_tensors(x)
    return out


# 张量血缘（provenance）：tensor id → 产出它的上游叶子名集合（含哨兵 "input"）。
# functional 算子（cat / +/ interpolate / pool …）不产生 module，纯靠 module hook
# 无法连边；用 TorchFunctionMode 拦截这些算子，把输出张量的血缘并为各输入血缘之并，
# 从而下游 module 的输入张量能反查到真实的多个上游叶子（多输入融合 / 跳连 / 残差）。
Provenance = Dict[int, "frozenset[str]"]
# 融合算子标记：tensor id → 归一化符号。仅当某算子**实际合并 ≥ 2 个带血缘的输入**时
# 记录其输出张量，供步骤 D 在融合点插入显式 merge 节点并正确标注（cat / + / × …）。
MergeOps = Dict[int, str]
# torch 函数名 → 展示符号。未登记的多源融合算子回退到通用 "merge"。
_MERGE_OP_SYMBOLS: Dict[str, str] = {
    "cat": "cat", "concat": "cat", "concatenate": "cat", "stack": "cat",
    "hstack": "cat", "vstack": "cat", "dstack": "cat", "column_stack": "cat",
    "add": "+", "add_": "+", "__add__": "+", "__iadd__": "+", "__radd__": "+",
    "sub": "−", "sub_": "−", "subtract": "−", "__sub__": "−",
    "mul": "×", "mul_": "×", "multiply": "×", "__mul__": "×",
    "maximum": "max", "max": "max", "minimum": "min", "min": "min",
}


def _op_symbol(func) -> str:
    """torch 函数对象 → merge 节点展示符号（默认 ``merge``）。"""
    return _MERGE_OP_SYMBOLS.get(getattr(func, "__name__", ""), "merge")


def _make_prov_mode(prov: Provenance, merge_op: MergeOps):
    """构造一个累积张量血缘的 TorchFunctionMode（torch 不支持时返回 None）。

    每个流经模块/算子之间的张量在产生时都会被写入血缘（functional 输出在此累积，
    叶子输出在 forward hook 覆盖），因此 ``id()`` 复用不会留下被读取到的陈旧项，
    无需钉住全部张量（钉住会在大体积 3D 前向时撑爆内存）。
    """
    try:
        from torch.overrides import TorchFunctionMode
    except Exception:  # pragma: no cover - 老版本 torch
        return None

    class _ProvMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            ins = _iter_tensors(args) + _iter_tensors(kwargs)
            out = func(*args, **kwargs)
            try:
                srcs: "frozenset[str]" = frozenset()
                for t in ins:
                    s = prov.get(id(t))
                    if s:
                        srcs |= s
                in_ids = {id(t) for t in ins}
                # 是否为真正的多源融合：该算子本身接收 ≥ 2 个带血缘的输入。
                merge_sym = (_op_symbol(func)
                             if sum(1 for t in ins if prov.get(id(t))) >= 2
                             else None)
                for o in _iter_tensors(out):
                    if id(o) in in_ids:
                        # in-place / 透传：在该（存活）张量已有血缘上累加。
                        prov[id(o)] = prov.get(id(o), frozenset()) | srcs
                    else:
                        # 新张量：直接覆盖，杜绝其 id 命中已释放张量的陈旧血缘
                        # （陈旧读是 id 追踪下不确定性的根源）。
                        prov[id(o)] = srcs
                    if merge_sym is not None:
                        merge_op[id(o)] = merge_sym
                    else:
                        merge_op.pop(id(o), None)  # 新张量复用旧 id 时清陈旧标记
            except Exception:  # pragma: no cover - 血缘累积尽力而为
                pass
            return out

    return _ProvMode()


def _trace_modules(
    model: nn.Module, in_shape: Tuple[int, ...],
) -> Tuple[Dict[str, _ModuleRec], bool]:
    """注册全模块 hook 跑一次 dummy 前向，回填张量 id / 形状 / 输入血缘。

    返回 ``(recs, traced)``；``traced=False`` 表示前向失败。每个 rec 的 ``in_prov``
    在 pre_hook 时刻已解析好上游血缘，故无需把 ``prov`` 透出到 build 阶段。
    """
    recs: Dict[str, _ModuleRec] = {
        name: _ModuleRec(name, m)
        for name, m in model.named_modules() if name
    }
    leaf_names: Set[str] = {
        name for name, m in model.named_modules()
        if name and not list(m.children())
    }
    # 残差 block 容器：其输出 = 主路 + shortcut(输入)，会把"该 block 的输入血缘"
    # 经 identity 捷径透传到输出，污染同级/下游连边。这里在 block 出口把血缘**重封**
    # 为该 block 自身，使其对外只表现为单一产出单元（块内残差另由结构化残差边表达）。
    residual_types = _residual_block_types()
    reseal_names: Set[str] = {
        name for name, m in model.named_modules()
        if name and list(m.children()) and _has_residual(m, residual_types)
    }
    prov: Provenance = {}
    merge_op: MergeOps = {}
    # 叶子输出张量 id → (产出叶子名, 该张量弱引用)。weakref 用于在 pre_hook 查询时
    # 辨别"id 是否被复用"：若弱引用已失效或不指向当前输入张量，则该 id 实为陈旧项。
    live_out: Dict[int, Tuple[str, "weakref.ref"]] = {}

    counter = {"i": 0}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def _src_of(t: torch.Tensor) -> Optional[str]:
        entry = live_out.get(id(t))
        if entry is None:
            return None
        name, ref = entry
        return name if ref() is t else None  # 弱引用校验：防 id 复用误连

    def _mk_pre_hook(rec: _ModuleRec):
        def pre_hook(_mod, args, kwargs=None):
            if rec.order < 0:  # 仅记首次调用
                tensors = _iter_tensors(args) + _iter_tensors(kwargs or {})
                rec.in_ids = tuple(id(t) for t in tensors)
                # 此刻输入张量存活、id 唯一 → 立即快照血缘与直接产出叶子。
                rec.in_prov = tuple(prov.get(id(t), frozenset()) for t in tensors)
                rec.in_src = tuple(_src_of(t) for t in tensors)
                rec.in_op = tuple(merge_op.get(id(t)) for t in tensors)
                if tensors:
                    rec.in_shape = tuple(tensors[0].shape)
        return pre_hook

    def _mk_hook(rec: _ModuleRec):
        is_leaf = rec.name in leaf_names
        is_reseal = rec.name in reseal_names

        def hook(_mod, _inp, out):
            tensors = _iter_tensors(out)
            if rec.order < 0:
                rec.order = counter["i"]
                counter["i"] += 1
                rec.out_ids = tuple(id(t) for t in tensors)
                if tensors:
                    rec.out_shape = tuple(tensors[0].shape)
            if is_leaf:  # 叶子输出血缘归该叶子自身
                # 透传叶子（Identity / inplace 激活等）输出张量与输入同一对象：
                # 血缘不得抢占上游（否则冒名顶替真正的产出者），但**直接产出者**仍按
                # "最近写者"记录（含透传），以复刻 block 内 inplace 链的逐叶子连边。
                in_ids = {id(t) for t in _iter_tensors(_inp)}
                src = frozenset({rec.name})
                for t in tensors:
                    if id(t) not in in_ids:
                        prov[id(t)] = src
                    try:
                        live_out[id(t)] = (rec.name, weakref.ref(t))
                    except TypeError:  # pragma: no cover - 个别张量不可弱引用
                        pass
            elif is_reseal:  # 残差 block 出口：把输出血缘重封为该 block 自身
                src = frozenset({rec.name})
                for t in tensors:
                    prov[id(t)] = src
        return hook

    for rec in recs.values():
        handles.append(
            rec.module.register_forward_pre_hook(_mk_pre_hook(rec),
                                                  with_kwargs=True))
        handles.append(rec.module.register_forward_hook(_mk_hook(rec)))

    traced = False
    prev_training = model.training
    try:
        dummy = torch.zeros(*in_shape, dtype=torch.float32)
        for mode in (True, False):  # 先 train()（激活 aux/ds/topo），失败再 eval()
            try:
                prov.clear()
                merge_op.clear()
                prov[id(dummy)] = frozenset({"input"})
                model.train(mode)
                prov_mode = _make_prov_mode(prov, merge_op)
                with torch.no_grad():
                    if prov_mode is not None:
                        with prov_mode:
                            model(dummy)
                    else:
                        model(dummy)
                traced = True
                break
            except Exception as e:  # 换模式重试
                logger.debug("model_flow: forward(train=%s) 失败: %s", mode, e)
                counter["i"] = 0
                live_out.clear()
                for r in recs.values():  # 清空上次半成品记录
                    r.order = -1
                    r.in_ids = r.out_ids = ()
                    r.in_shape = r.out_shape = None
                    r.in_prov = ()
                    r.in_src = ()
                    r.in_op = ()
    finally:
        for h in handles:
            h.remove()
        model.train(prev_training)
    return recs, traced


# ---------------------------------------------------------------------------
# 损失节点
# ---------------------------------------------------------------------------
# 各损失分量 → 其真正消费的关键超参字段（顺序即展示顺序）。与 losses.factory 的
# 各 ``_build_*`` 构造器逐一对应；详情抽屉据此只展示与当前损失相关的参数。
_LOSS_COMPONENT_FIELDS: Dict[str, Tuple[str, ...]] = {
    "dice":          ("dice_smooth", "dice_squared", "batch_dice", "ignore_empty"),
    "bce":           (),
    "focal":         ("focal_alpha", "focal_gamma"),
    "tversky":       ("tversky_alpha", "tversky_beta", "dice_smooth", "batch_dice"),
    "gdl":           ("gdl_weight_type", "gdl_w_max", "dice_smooth", "batch_dice"),
    "focal_tversky": ("tversky_alpha", "tversky_beta", "focal_tversky_gamma",
                      "dice_smooth", "batch_dice"),
    "lovasz":        ("lovasz_per_sample",),
    "cldice":        ("cldice_iter", "cldice_smooth"),
}


def _loss_components(name: str) -> List[str]:
    """损失名 → 分量名列表（单一或复合）。从 losses.factory 反推，缺失则空。"""
    try:
        from ..losses.losses import _COMPOUND_BUILDERS, _SINGLE_BUILDERS
    except Exception:  # pragma: no cover - 仅日志
        return []
    pfx, nm = "_build_", name.lower()
    if nm in _SINGLE_BUILDERS:
        return [_SINGLE_BUILDERS[nm].__name__[len(pfx):]]
    if nm in _COMPOUND_BUILDERS:
        return [b.__name__[len(pfx):] for b in _COMPOUND_BUILDERS[nm]]
    return []


def _fmt_num(v: object) -> object:
    """浮点统一保留有效位，避免 1.3333333333 这类长尾。"""
    return round(v, 6) if isinstance(v, float) else v


def _loss_node_detail(cfg: Config) -> Dict[str, object]:
    """损失节点详情：展示**该损失实际使用的超参**（dice 平滑、focal α/γ、复合权重、
    深监督权重等），而非 pipeline/criterion 等与损失数值无关的元信息。"""
    lc = cfg.loss
    comps = _loss_components(lc.name)
    detail: Dict[str, object] = {"loss": lc.name}
    if comps:
        detail["components"] = " + ".join(comps)
        if len(comps) > 1:  # 复合损失：各分量相对权重
            try:
                from ..losses.losses import _compound_weights
                ws = _compound_weights(lc, len(comps))
            except Exception:  # pragma: no cover
                ws = list(getattr(lc, "compound_weights", []) or [])[:len(comps)]
            if ws:
                detail["compound_weights"] = ", ".join(
                    f"{c}={_fmt_num(w)}" for c, w in zip(comps, ws))
        seen: Set[str] = set()
        for c in comps:                  # 仅展示相关分量的关键超参（去重保序）
            for field in _LOSS_COMPONENT_FIELDS.get(c, ()):
                if field in seen or not hasattr(lc, field):
                    continue
                seen.add(field)
                detail[field] = _fmt_num(getattr(lc, field))
    if getattr(lc, "class_weights", None):
        detail["class_weights"] = list(lc.class_weights)
    # 监督结构相关的权重（仅在开启时展示）。
    if cfg.model.deep_supervision:
        detail["deep_supervision"] = True
        if getattr(lc, "deep_supervision_weights", None):
            detail["deep_supervision_weights"] = list(lc.deep_supervision_weights)
    if cfg.model.aux_seg_supervision:
        detail["aux_seg_supervision"] = True
        if getattr(lc, "aux_supervision_weights", None):
            detail["aux_supervision_weights"] = list(lc.aux_supervision_weights)
    if getattr(cfg.model, "aux_topo_head", False):
        detail["aux_topo_head"] = True
        detail["aux_topo_weight"] = _fmt_num(getattr(lc, "aux_topo_weight", 0.0))
    return detail


# ---------------------------------------------------------------------------
# DAG 重建：分组 → 节点 → 连边 → 层级
# ---------------------------------------------------------------------------
class _ModelGraphBuilder:
    """把 ``_ModuleRec`` 集合重建为 ``VisGraph`` 的 DAG。"""

    def __init__(self, g: VisGraph, recs: Dict[str, _ModuleRec],
                 traced: bool):
        self.g = g
        self.recs = recs
        self.traced = traced
        self.residual_types = _residual_block_types()

        # 叶子（被执行过；未追踪时取全部声明叶子）。
        self.leaves: Dict[str, _ModuleRec] = {
            name: rec for name, rec in recs.items()
            if not list(rec.module.children())
            and (rec.order >= 0 or not traced)
        }
        self.leaf_set: Set[str] = set(self.leaves)

        # 分组结构。
        self.top_leaves: Dict[str, List[_ModuleRec]] = {}
        self.top_order: List[str] = []
        for name, rec in self.leaves.items():
            tk = _top_key(name)
            if tk not in self.top_leaves:
                self.top_leaves[tk] = []
                self.top_order.append(tk)
            self.top_leaves[tk].append(rec)

        self.top_set: Set[str] = set(self.top_leaves)
        self._edges: Set[Tuple[str, str]] = set()
        # 叶子模块名 → 已发射的叶子节点 id（供同 block 内叶子连边反查）。
        self._leaf_node_id: Dict[str, str] = {}
        # (parent, tensor id, 输入序号) → 已发射的 merge 节点 id（同一融合张量被
        # 多下游消费时复用同一节点，避免重复建框）。
        self._merge_nodes: Dict[Tuple[str, int, int], str] = {}

    # -- 排序键 --------------------------------------------------------
    def _order_of(self, recs: Sequence[_ModuleRec]) -> Tuple[int, int]:
        orders = [r.order for r in recs if r.order >= 0]
        if orders:
            return (0, min(orders))
        return (1, 1 << 30)

    def _top_sort_key(self, key: str):
        oo = self._order_of(self.top_leaves[key])
        return (oo[0], oo[1], self.top_order.index(key))

    # -- 上游反查（基于 pre_hook 快照的输入血缘集合）----------------------
    def _producers_at_top(
        self, prov_sets: Sequence["frozenset[str]"], exclude: str,
    ) -> List[str]:
        """输入血缘集合 → 去重的上游顶层容器（或 'input'），按多源展开。"""
        srcs: List[str] = []
        seen: Set[str] = set()
        for s in prov_sets:
            for leaf in s:
                src = "input" if leaf == "input" else _top_key(leaf)
                if not src or src == exclude or src in seen:
                    continue
                if src == "input" or src in self.top_set:
                    seen.add(src)
                    srcs.append(src)
        return srcs

    def _producers_in_top(
        self, prov_sets: Sequence["frozenset[str]"], top: str, exclude: str,
    ) -> List[str]:
        """容器内：输入血缘集合 → 同容器内的上游 block 键（外部来源忽略）。"""
        srcs: List[str] = []
        seen: Set[str] = set()
        for s in prov_sets:
            for leaf in s:
                if leaf == "input" or _top_key(leaf) != top:
                    continue
                bk = _block_key(leaf, top) or leaf
                if bk != exclude and bk not in seen:
                    seen.add(bk)
                    srcs.append(bk)
        return srcs

    def _add_edge(self, src: str, dst: str, label: str = "",
                  kind: str = "forward") -> None:
        if src == dst or (src, dst) in self._edges:
            return
        self._edges.add((src, dst))
        self.g.add_edge(src, dst, label, kind=kind)

    def _emit_merge(self, parent: str, tid: int, idx: int, op: str,
                    out_shape: Optional[Tuple[int, ...]] = None
                    ) -> Tuple[str, bool]:
        """在融合点发射（或复用）一个显式 merge 算子节点；返回 ``(节点 id, 是否新建)``。

        同一 ``(parent, tid, idx)`` 复用同一节点，以免被多个下游消费时重复建框。
        节点 ``kind="merge"``，标题为融合符号（``cat`` / ``+`` / ``×`` …）。
        """
        mkey = (parent, tid, idx)
        mid = self._merge_nodes.get(mkey)
        if mid is not None:
            return mid, False
        mid = f"merge::{parent}::{tid}:{idx}"
        # 标题已是融合符号，故 key_info 只补出张量形状，避免框内重复显示算子名。
        ki: Dict[str, object] = {}
        if out_shape:
            ki["out"] = shape_str(out_shape)
        detail = {"op": op, "kind": "merge (functional fusion)"}
        self.g.add_node(mid, op, kind="merge", key_info=ki, detail=detail,
                        parent_id=parent)
        self._merge_nodes[mkey] = mid
        return mid, True


def build_model_flow(
    cfg: Config,
    model: nn.Module,
    topo: Optional[ModelTopology] = None,
    trace_shapes: bool = True,
) -> VisGraph:
    """构造模型流 ``VisGraph``。``trace_shapes=False`` 时跳过 dummy 前向、纯结构图。"""
    topo = topo or build_topology(cfg)
    target = _target_patch_size(cfg)
    in_shape = _model_input_shape(cfg, topo, target)
    in_shape = (1,) + tuple(in_shape[1:])  # 追踪 batch 固定为 1（省算力）

    g = VisGraph(title="模型流 Model Flow")

    recs: Dict[str, _ModuleRec] = {}
    traced = False
    if trace_shapes:
        try:
            recs, traced = _trace_modules(model, in_shape)
        except Exception as e:  # 追踪整体失败 → 纯结构
            logger.warning("model_flow: 数据流追踪失败，退化为纯结构图: %s", e)
    if not recs:
        recs = {name: _ModuleRec(name, m)
                for name, m in model.named_modules() if name}

    g.meta = {
        "arch": str(cfg.model.arch),
        "input": shape_str(in_shape),
        "trace_batch": "1",
        "shapes": "traced" if traced else "static (前向未执行/失败)",
        "in_channels": str(topo.in_channels),
        "out_classes": str(topo.out_classes),
        "spatial_dims": f"{topo.spatial_dims}D",
    }

    g.add_node(
        "input", "模型输入", kind="input",
        key_info={"shape": shape_str(in_shape)},
        detail={"in_channels": str(topo.in_channels),
                "spatial_dims": f"{topo.spatial_dims}D",
                "n_views": str(topo.n_views)})

    builder = _ModelGraphBuilder(g, recs, traced)
    _emit(builder, cfg)
    return g


# ---------------------------------------------------------------------------
# 节点与连边发射
# ---------------------------------------------------------------------------
def _emit(b: _ModelGraphBuilder, cfg: Config) -> None:
    g = b.g
    backbone = sorted(
        (k for k in b.top_leaves if not _is_head(k)), key=b._top_sort_key)
    heads = sorted(
        (k for k in b.top_leaves if _is_head(k)), key=b._top_sort_key)

    # 1) 发射各顶层容器（含内部 block / 叶子）。
    for key in backbone + heads:
        _emit_top(b, key)

    # 2) 顶层连边：按真实张量流反查上游。
    for key in backbone + heads:
        _emit_top_inbound_edges(b, key)

    # 2.5) 兜底：保证主干连通（追踪失败或反查缺失时）。
    _ensure_backbone_chain(b, backbone)

    # 3) 损失节点 + 各 head → loss。
    g.add_node(
        "loss", f"Loss: {cfg.loss.name}", kind="loss",
        key_info={"name": cfg.loss.name},
        detail=_loss_node_detail(cfg))
    if heads:
        for key in heads:
            b._add_edge(key, "loss", _head_edge_label(key))
    else:
        last = backbone[-1] if backbone else "input"
        b._add_edge(last, "loss")

    # 4) 标注跳连（src/dst 跨级且为 down 向）+ 计算层级 rank。
    _assign_ranks_and_skip(b, backbone, heads)
    # 5) 计算横向列位（col/colspan），供 renderer 做列对齐网格布局。
    _assign_columns(b)


def _called_direct_children(
    b: _ModelGraphBuilder, key: str,
) -> List[_ModuleRec]:
    """``key`` 的**直接子模块**中真正被前向调用过（有 order/形状）的那些，按执行序排序。
    ``nn.ModuleList`` 这类容器自身不被调用（无形状），但其子 block 会被逐个调用并记录到
    经过内部 functional reshape 之后的真实 in/out——比最深叶子更能代表容器的对外形状。"""
    pre = key + "."
    kids: List[_ModuleRec] = []
    for nm, r in b.recs.items():
        if nm.startswith(pre) and "." not in nm[len(pre):] and r.order >= 0:
            kids.append(r)
    kids.sort(key=lambda r: r.order)
    return kids


def _container_io(
    b: _ModelGraphBuilder, key: str, members: Sequence[_ModuleRec],
) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]]]:
    """容器/block 的 in/out 形状，按可信度逐级回退：
    ① **容器模块自身**被直接调用时记录的真实 in/out（如上采样容器 ``_Upsample`` 内含
       ``F.interpolate`` 这类 functional 算子，只有读容器自身才看得出分辨率变化）；
    ② 容器自身未被直接调用（如 ``nn.ModuleList`` 的 levels）时，取其**直接子模块**记录的
       in/out——子 block 的形状已包含其内部 functional reshape 的还原（如注意力块内把
       ``(B,C,H,W)`` 摊平成 ``(B,C,H·W)`` 做 ``Conv1d`` 再 reshape 回去，最深叶子 ``proj_out``
       只看到摊平后的 ``(B,C,H·W)``，而子 block 自身的 out 已是 reshape 回来的 ``(B,C,H,W)``）；
    ③ 最后才退化到最深成员叶子。"""
    crec = b.recs.get(key)
    ins = [r.in_shape for r in members if r.in_shape]
    outs = [r.out_shape for r in members if r.out_shape]
    in_sh = crec.in_shape if crec and crec.in_shape else None
    out_sh = crec.out_shape if crec and crec.out_shape else None
    if in_sh is None or out_sh is None:
        kids = _called_direct_children(b, key)
        if in_sh is None:
            kin = next((r.in_shape for r in kids if r.in_shape), None)
            in_sh = kin if kin is not None else (ins[0] if ins else None)
        if out_sh is None:
            kout = next(
                (r.out_shape for r in reversed(kids) if r.out_shape), None)
            out_sh = kout if kout is not None else (outs[-1] if outs else None)
    return in_sh, out_sh


def _emit_top(b: _ModelGraphBuilder, key: str) -> None:
    """发射一个顶层容器，及其内部 block / 叶子节点与容器内连边。"""
    g = b.g
    recs = sorted(b.top_leaves[key],
                  key=lambda r: r.order if r.order >= 0 else 1 << 30)
    kind = "head" if _is_head(key) else "stage"

    in_sh, out_sh = _container_io(b, key, recs)
    s_key: Dict[str, object] = {"ops": str(len(recs))}
    if in_sh:
        s_key["in"] = shape_str(in_sh)
    if out_sh:
        s_key["out"] = shape_str(out_sh)
    g.add_node(key, _top_label(key), kind=kind, key_info=s_key, collapsed=True)

    # 容器内按 block 分组。
    block_recs: Dict[Optional[str], List[_ModuleRec]] = {}
    block_order: List[Optional[str]] = []
    for rec in recs:
        bk = _block_key(rec.name, key)
        if bk not in block_recs:
            block_recs[bk] = []
            block_order.append(bk)
        block_recs[bk].append(rec)

    # 发射 block 框 + 叶子。
    for bk in block_order:
        members = block_recs[bk]
        if bk is None:
            # 未成 block 的散叶直接挂顶层容器；它们之间的真实张量流(如 Downsample
            # 的 op→norm)同样要连边，否则同框叶子会无边堆在一起。
            leaf_ids = [_emit_leaf(b, rec, parent=key) for rec in members]
            top_rec = b.recs.get(key)
            external = set(top_rec.in_ids) if top_rec else set()
            _emit_leaf_flow(b, members, leaf_ids, external, parent=key)
        else:
            _emit_block(b, bk, key, members)

    # 容器内 block 间连边：纯真实张量流（含穿过 functional 算子的血缘）。
    _emit_intra_edges(b, key, block_recs, block_order)


def _emit_leaf_flow(b: _ModelGraphBuilder, members: List[_ModuleRec],
                    leaf_ids: List[str], external: Set[int],
                    parent: str) -> None:
    """在一组**同级叶子**间按真实张量流连边：逐**输入张量**求其"直接上游叶子"。

    既服务于 block 框内的叶子(``_emit_block``)，也服务于顶层容器里未成 block 的散叶
    (``_emit_top`` 的 ``bk is None`` 组，如 ``Downsample`` 的 ``op→norm``)。

      · 该张量由某叶子直接产出(``in_src`` 非空，经 weakref 校验防 id 复用)→ 用它，
        从而保住顺序链与透传链；
      · 该张量由 functional 算子(torch.cat/+/split…)产生(``in_src`` 为 None)→ 退到
        血缘 ``in_prov``，取落在**本组内**的叶子集合，从而恢复并联分支的扇入
        (MultiRF 三支→fuse)与残差扇入(主路尾 + shortcut→输出)。
    外部输入(``external``)是本框的并联入口，不在组内连边；其上游由顶层入边处理。
    """
    name_to_cid = {rec.name: cid for rec, cid in zip(members, leaf_ids)}
    member_names = set(name_to_cid)

    def _is_shortcut(name: str) -> bool:
        return any(f".{a}" in name for a in _SHORTCUT_ATTRS)

    def _infer_op(srcs: List[str]) -> str:
        # 血缘未捕到算子符号时的回退：含捷径源 → 残差加，否则视为拼接。
        return "+" if any(_is_shortcut(s) for s in srcs) else "cat"

    def _block_prefix(name: str) -> str:
        # 捷径叶子所属残差子块的路径前缀（截到 shortcut 段之前），用于把悬空 shortcut
        # 连回**本子块**的输出叶，避免在多子块拍平成一个框时跨子块误连。
        for a in _SHORTCUT_ATTRS:
            idx = name.find(f".{a}")
            if idx >= 0:
                return name[:idx]
        return name

    if b.traced:
        name_rec = {rec.name: rec for rec in members}
        # 张量 id → 本 block 内**最后**产出它的叶子名（按前向序覆盖）。透传叶子(Identity、
        # 无参 attn、inplace 激活)的输出张量与其输入同对象，故会"接管"该 id。
        last_emit: Dict[int, str] = {}
        for rec in sorted(members, key=lambda r: r.order):
            for oid in rec.out_ids:
                last_emit[oid] = rec.name

        def _supersede(src: str, consumer: _ModuleRec) -> str:
            # functional 合流(如 `主路 + shortcut`)的扇入，其血缘只记到最早产出者；但真正
            # 喂入合流的是该张量**最后**的产出叶子(透传链尾，如 norm2→attn 的 attn)。把血缘
            # 来源上提到该尾叶，既得到正确的 norm2→attn→act2 链，又免去冗余的 norm2→act2。
            rec_src = name_rec.get(src)
            if rec_src is None or not rec_src.out_ids:
                return src
            cand = last_emit.get(rec_src.out_ids[0])
            cand_rec = name_rec.get(cand) if cand else None
            if (cand_rec is not None and cand != src
                    and cand in member_names and cand_rec.order < consumer.order):
                return cand
            return src

        has_out: Set[str] = set()  # 已连出（有下游）的叶子，供残差兜底判定悬空 shortcut
        for rec, cid in zip(members, leaf_ids):
            prov = rec.in_prov if len(rec.in_prov) == len(rec.in_ids) else ()
            in_op = rec.in_op if len(rec.in_op) == len(rec.in_ids) else ()
            for idx, (tid, prod) in enumerate(zip(rec.in_ids, rec.in_src)):
                if tid in external:          # 外部输入：并联入口，不连块内边
                    continue
                functional = False
                if prod is not None and prod != rec.name:
                    sources: Iterable[str] = (prod,)
                elif idx < len(prov):        # functional 产出 → 退到血缘
                    sources = [_supersede(s, rec) for s in prov[idx]]
                    functional = True
                else:
                    sources = ()
                # 去重 + 过滤到本组内、非自身的上游叶子。
                valid: List[str] = []
                for src in sources:
                    if src in member_names and src != rec.name and src not in valid:
                        valid.append(src)
                if not valid:
                    continue
                if functional and len(valid) >= 2:
                    # 多源融合点 → 插入显式 merge 节点：各上游叶→[op]→本叶。
                    op = (in_op[idx] if idx < len(in_op) else None) or _infer_op(valid)
                    out_sh = rec.in_shape if idx == 0 else None
                    mid, created = b._emit_merge(parent, tid, idx, op, out_sh)
                    if created:
                        for src in valid:
                            # shortcut 支汇入以残差线型凸显跳连；merge 节点本身已示算子
                            # 符号，故入 merge 的边不再重复标 "+"。
                            res = _is_shortcut(src) and not _is_shortcut(rec.name)
                            b._add_edge(name_to_cid[src], mid, "",
                                        kind="residual" if res else "forward")
                    for src in valid:
                        has_out.add(src)
                    b._add_edge(mid, cid, kind="forward")
                else:
                    for src in valid:
                        # 由 shortcut 子树**汇入主路**的边标注为残差(+)；shortcut 内部
                        # 链(conv→norm)与主路前向流仍为普通前向边。
                        residual_edge = _is_shortcut(src) and not _is_shortcut(rec.name)
                        label = "+" if residual_edge else ""
                        kind = "residual" if residual_edge else "forward"
                        b._add_edge(name_to_cid[src], cid, label, kind=kind)
                        has_out.add(src)
        # 残差兜底：`out = 主路 + shortcut(x)` 的相加是 functional 算子，透传(Identity)
        # 捷径叶子的输出张量与其输入(块外)同对象、血缘指向块外，prov 无法还原其汇入主路
        # 的边，留下悬空 shortcut 叶子(无下游)。主路上的透传叶子(无参 attn 等)已由
        # ``_supersede`` 上提到合流尾叶正确连边，故这里只需兜底**悬空的 shortcut 叶子**：
        # 把它们补连到汇点——其后最近的悬空主路叶子(相加结果承接叶 act2/主路末层)。
        # 以"存在悬空 shortcut 叶子"为触发条件(而非容器是否被判残差)，从而也能覆盖把多个
        # 残差子块拍平成一个框的情形(如 selfattn stage)。仅连 shortcut 叶子，不动其余。
        ordered = sorted(members, key=lambda r: r.order)
        for rec in members:
            if rec.name in has_out or not _is_shortcut(rec.name):
                continue
            # 汇点 = 与该 shortcut 同子块前缀、序最大的非 shortcut 叶(本子块输出叶 act2)；
            # 退而求其次取全组最后一个非 shortcut 叶。
            pref = _block_prefix(rec.name)
            sink = next((r.name for r in reversed(ordered)
                         if not _is_shortcut(r.name) and r.name.startswith(pref)),
                        None)
            sink = sink or next((r.name for r in reversed(ordered)
                                 if not _is_shortcut(r.name)), None)
            if sink is not None and sink != rec.name:
                b._add_edge(name_to_cid[rec.name], name_to_cid[sink],
                            "+", kind="residual")
    else:
        # 追踪失败的纯结构降级：按声明序把叶子连成一条线性链，至少给出可读骨架。
        for prev, cur in zip(leaf_ids, leaf_ids[1:]):
            b._add_edge(prev, cur)


def _emit_block(b: _ModelGraphBuilder, block_key: str, top: str,
                members: List[_ModuleRec]) -> None:
    g = b.g
    members = sorted(members, key=lambda r: r.order if r.order >= 0 else 1 << 30)
    mod = b.recs[block_key].module if block_key in b.recs else members[0].module
    ki: Dict[str, object] = {"type": type(mod).__name__}
    in_sh, out_sh = _container_io(b, block_key, members)
    if in_sh:
        ki["in"] = shape_str(in_sh)
    if out_sh:
        ki["out"] = shape_str(out_sh)
    residual = _has_residual(mod, b.residual_types)
    if residual:
        ki["skip"] = "residual (+)"
    g.add_node(block_key, _block_label(block_key, top),
               kind="stage", key_info=ki, parent_id=top, collapsed=True)

    leaf_ids = [_emit_leaf(b, rec, parent=block_key) for rec in members]
    # block 外部输入是本框的并联入口，不在块内连边；其上游由顶层入边处理。
    block_rec = b.recs.get(block_key)
    external: Set[int] = set(block_rec.in_ids) if block_rec else set()
    _emit_leaf_flow(b, members, leaf_ids, external, parent=block_key)


def _emit_leaf(b: _ModelGraphBuilder, rec: _ModuleRec, parent: str) -> str:
    g = b.g
    cid = f"leaf::{rec.name}"
    ki: Dict[str, object] = {"type": type(rec.module).__name__}
    if rec.out_shape:
        ki["out"] = shape_str(rec.out_shape)
    detail = _leaf_params(rec.module)
    detail["module"] = rec.name
    if rec.in_shape:
        detail["in_shape"] = shape_str(rec.in_shape)
    if rec.out_shape:
        detail["out_shape"] = shape_str(rec.out_shape)
    g.add_node(cid, type(rec.module).__name__, kind=_leaf_kind(rec.module),
               key_info=ki, detail=detail, parent_id=parent)
    b._leaf_node_id[rec.name] = cid
    return cid


def _emit_top_inbound_edges(b: _ModelGraphBuilder, key: str) -> None:
    """顶层容器的入边：反查"喂给本容器内任一叶子的外部张量"的上游容器。

    既用容器模块自身输入（被直接调用的容器），也并入**所有成员叶子的输入**——
    对 ``ModuleList`` 这类从不被调用、自身无输入记录的容器（ADM/EDM2 的 levels、
    UNet3+ 的 branches/fusions），后者是唯一能反查出真实入边（含 encoder→decoder
    跳连）的来源；容器内部来源由 ``_producers_at_top`` 的 exclude 自动滤除。
    """
    prov_sets: List["frozenset[str]"] = []
    rec = b.recs.get(key)
    if rec:
        prov_sets.extend(rec.in_prov)
    for r in b.top_leaves.get(key, []):
        prov_sets.extend(r.in_prov)
    for src in b._producers_at_top(prov_sets, exclude=key):
        b._add_edge(src, key)


def _emit_intra_edges(
    b: _ModelGraphBuilder, top: str,
    block_recs: Dict[Optional[str], List[_ModuleRec]],
    block_order: List[Optional[str]],
) -> None:
    """容器内 block 间连边：纯真实张量流。

    每个 block 用其模块输入的血缘 ``in_prov`` 反查同容器内上游 block。血缘由
    ``TorchFunctionMode`` 钩子穿过 ``torch.cat/+/split`` 等 functional 算子累积，
    因此 MultiStem 的三路 sub-stem→proj、DecoderLevel 的 upsample→首块等
    经 cat 融合的边都能被通用地还原，无需按容器类型写死补边。
    """
    real_blocks = [bk for bk in block_order if bk is not None]
    for bk in real_blocks:
        rec = b.recs.get(bk)
        if rec is None or not rec.in_prov:
            continue
        in_op = rec.in_op if len(rec.in_op) == len(rec.in_prov) else ()
        # 逐**输入张量**反查同容器上游 block：单上游直连；同一输入被多 block
        # 经 cat/+ 融合（如 MultiStem 三路 sub-stem→proj）则经显式 merge 节点汇入。
        for idx, pset in enumerate(rec.in_prov):
            srcs = b._producers_in_top([pset], top, exclude=bk)
            if not srcs:
                continue
            if len(srcs) >= 2:
                op = (in_op[idx] if idx < len(in_op) else None) or "cat"
                tid = rec.in_ids[idx] if idx < len(rec.in_ids) else -1 - idx
                out_sh = rec.in_shape if idx == 0 else None
                mid, created = b._emit_merge(top, tid, idx, op, out_sh)
                if created:
                    for src in srcs:
                        b._add_edge(src, mid)
                b._add_edge(mid, bk)
            else:
                b._add_edge(srcs[0], bk)


def _ensure_backbone_chain(b: _ModelGraphBuilder, backbone: List[str]) -> None:
    """兜底：任何无入边的非首 backbone 容器，按执行序从前一个连上；
    首容器若无入边则从 input 连上。保证图连通（追踪失败时尤为重要）。"""
    incoming = {dst for _, dst in b._edges}
    prev = "input"
    for key in backbone:
        if key not in incoming:
            b._add_edge(prev, key)
        prev = key


# ---------------------------------------------------------------------------
# 层级 rank 计算 + 跳连标注
# ---------------------------------------------------------------------------
def _assign_ranks_and_skip(
    b: _ModelGraphBuilder, backbone: List[str], heads: List[str],
) -> None:
    """对每个父容器内的兄弟节点做最长路径分层（rank），并把"跨级 down 边"标为 skip。"""
    g = b.g
    by_parent: Dict[Optional[str], List[str]] = {}
    for n in g.nodes:
        by_parent.setdefault(n.parent_id, []).append(n.id)

    # 各父容器内：用 forward 边做最长路径分层。
    node_rank: Dict[str, int] = {}
    for ids in by_parent.values():
        idset = set(ids)
        succ: Dict[str, List[str]] = {i: [] for i in ids}
        indeg: Dict[str, int] = {i: 0 for i in ids}
        for e in g.edges:
            if e.kind == "residual":
                continue
            if e.src in idset and e.dst in idset:
                succ[e.src].append(e.dst)
                indeg[e.dst] += 1
        # Kahn 最长路径。
        rank = {i: 0 for i in ids}
        queue = [i for i in ids if indeg[i] == 0]
        while queue:
            cur = queue.pop()
            for nx in succ[cur]:
                if rank[cur] + 1 > rank[nx]:
                    rank[nx] = rank[cur] + 1
                indeg[nx] -= 1
                if indeg[nx] == 0:
                    queue.append(nx)
        node_rank.update(rank)

    # 顶层 head 分层：
    #   * 主输出头（seg_head / aux_heads）从最后一个 decoder level 分叉，拍到同一末行；
    #   * deep-supervision 头（ds_heads.*）保留最长路径自然层级，从而与其分叉来源
    #     decoder level 的下一层**并列一行**（dec_k 同时发出 →dec_{k+1} 与 →ds_k），
    #     避免硬拍到底排后再拉长线自上而下穿过解码器框。loss 殿后。
    if heads:
        final_heads = [h for h in heads if not h.startswith("ds_heads.")]
        if final_heads:
            fr = max((node_rank.get(h, 0) for h in final_heads), default=0)
            for h in final_heads:
                node_rank[h] = fr
        if "loss" in node_rank:
            node_rank["loss"] = max(
                (node_rank.get(h, 0) for h in heads), default=0) + 1
    # 应用 rank。
    rankmap = {n.id: node_rank.get(n.id, 0) for n in g.nodes}
    for n in g.nodes:
        n.rank = rankmap[n.id]

    # 跨级 down 边（dst.rank - src.rank > 1）标为 skip（同一父容器内）。
    parent_of = {n.id: n.parent_id for n in g.nodes}
    for e in g.edges:
        if e.kind != "forward":
            continue
        if parent_of.get(e.src) != parent_of.get(e.dst):
            continue
        if rankmap.get(e.dst, 0) - rankmap.get(e.src, 0) > 1:
            e.kind = "skip"


def _assign_columns(b: _ModelGraphBuilder) -> None:
    """对每个父容器内的兄弟节点分配横向列位 ``(col, colspan)``，使并联路径各占一列、
    主链笔直，融合点（cat/+）居中覆盖其上游列。纯靠 forward 边血缘，不写死模块类型。

    逐 rank 自上而下：
      (a) 按已定稿的 forward 父（排除 residual/skip）定列——无父则为源、占新列；
          有父则 ``col=min(父.col)``、``colspan`` 覆盖父列区间（单父继承，使链笔直）。
      (b) 同 rank 内按 ``(col, 插入序)`` 从左到右扫描去重叠（右移），下游在各自 rank
          基于已定稿父列重算自动跟随，无累积漂移。
    """
    g = b.g
    by_parent: Dict[Optional[str], List[VisNode]] = {}
    for n in g.nodes:
        by_parent.setdefault(n.parent_id, []).append(n)
    # dst → forward 父列表（residual/skip 不计入列血缘）。
    fwd_parents: Dict[str, List[str]] = {}
    for e in g.edges:
        if e.kind == "forward":
            fwd_parents.setdefault(e.dst, []).append(e.src)

    for kids in by_parent.values():
        idset = {n.id for n in kids}
        order_idx = {n.id: i for i, n in enumerate(kids)}
        by_rank: Dict[int, List[str]] = {}
        for n in kids:
            by_rank.setdefault(n.rank, []).append(n.id)
        col: Dict[str, int] = {}
        span: Dict[str, int] = {}
        for r in sorted(by_rank):
            row = by_rank[r]
            for nid in row:
                ps = [p for p in fwd_parents.get(nid, ())
                      if p in idset and p in col and p != nid]
                if not ps:
                    col[nid] = 0          # 暂置，由扫描步打包成相邻列
                    span[nid] = 1
                else:
                    lo = min(col[p] for p in ps)
                    hi = max(col[p] + span[p] for p in ps)
                    col[nid] = lo
                    span[nid] = hi - lo
            # 同 rank 去重叠：按 (col, 插入序) 从左到右，必要时右移。
            cursor: Optional[int] = None
            for nid in sorted(row, key=lambda x: (col[x], order_idx[x])):
                if cursor is not None and col[nid] < cursor:
                    col[nid] = cursor
                cursor = col[nid] + span[nid]
        for n in kids:
            n.col = col[n.id]
            n.colspan = span[n.id]


# -- 小工具 ----------------------------------------------------------------


__all__ = ["build_model_flow"]
