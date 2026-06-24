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
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn

from ..config import Config
from ..models.topology import ModelTopology, build_topology
from .data_flow import _model_input_shape, _target_patch_size
from .graph import VisGraph, shape_str

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


def _top_key(name: str) -> str:
    """叶子限定名 → 所属**顶层容器**键（encoder.stem / encoder.stages.k / …）。

    与旧 ``_stage_key`` 的关键区别：``encoder.stem.stems.k`` 不再各自成顶层框，而是
    统一归到单一 ``encoder.stem`` 容器（其内部三路并联另由 block 分组呈现）。
    """
    parts = name.split(".")
    p0 = parts[0]
    if p0 == "encoder":
        if len(parts) >= 2 and parts[1] == "stem":
            return "encoder.stem"
        if (len(parts) >= 3 and parts[1] in ("stages", "downsamples")
                and parts[2].isdigit()):
            return f"encoder.{parts[1]}.{parts[2]}"
        if len(parts) >= 2 and parts[1] == "aux_fuse":
            return "encoder.aux_fuse"
        return ".".join(parts[:2]) if len(parts) >= 2 else p0
    if p0 == "decoder":
        if len(parts) >= 3 and parts[1] == "levels" and parts[2].isdigit():
            return f"decoder.levels.{parts[2]}"
        # UNet++/UNet3+ 的 ModuleDict（blocks/upsamples/gates，键形如 i_j）：按节点键聚合。
        if len(parts) >= 3 and parts[1] in ("blocks", "upsamples", "gates"):
            return f"decoder.{parts[1]}.{parts[2]}"
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
    if len(parts) == 3 and parts[0] == "encoder" and parts[1] == "stages":
        return f"Encoder Stage {parts[2]}"
    if len(parts) == 3 and parts[0] == "encoder" and parts[1] == "downsamples":
        return f"Downsample {parts[2]}"
    if len(parts) == 3 and parts[0] == "decoder" and parts[1] == "levels":
        return f"Decoder Level {parts[2]}"
    if len(parts) == 3 and parts[0] == "decoder":
        return f"Decoder X[{parts[2]}]"
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
    return tuple(types)


def _has_residual(m: nn.Module, residual_types: Tuple[type, ...]) -> bool:
    """该 block 是否含残差捷径（``out + shortcut(x)`` 形态）。"""
    if residual_types and isinstance(m, residual_types):
        return True
    # 通用启发：含名为 shortcut 的子模块即视为残差块。
    return hasattr(m, "shortcut") and isinstance(m.shortcut, nn.Module)


# ---------------------------------------------------------------------------
# 张量数据流追踪
# ---------------------------------------------------------------------------
class _ModuleRec:
    """单个模块一次前向的输入/输出张量 id 与形状记录。"""

    __slots__ = ("name", "module", "order", "in_ids", "out_ids",
                 "in_shape", "out_shape")

    def __init__(self, name: str, module: nn.Module):
        self.name = name
        self.module = module
        self.order: int = -1
        self.in_ids: Tuple[int, ...] = ()
        self.out_ids: Tuple[int, ...] = ()
        self.in_shape: Optional[Tuple[int, ...]] = None
        self.out_shape: Optional[Tuple[int, ...]] = None


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


def _trace_modules(
    model: nn.Module, in_shape: Tuple[int, ...],
) -> Tuple[Dict[str, _ModuleRec], Set[int], bool]:
    """注册全模块 hook 跑一次 dummy 前向，回填张量 id 与形状。

    返回 ``(recs, input_ids, traced)``；``traced=False`` 表示前向失败、流图不可用。
    """
    recs: Dict[str, _ModuleRec] = {
        name: _ModuleRec(name, m)
        for name, m in model.named_modules() if name
    }

    counter = {"i": 0}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def _mk_pre_hook(rec: _ModuleRec):
        def pre_hook(_mod, args, kwargs=None):
            if rec.order < 0:  # 仅记首次调用
                tensors = _iter_tensors(args) + _iter_tensors(kwargs or {})
                rec.in_ids = tuple(id(t) for t in tensors)
                if tensors:
                    rec.in_shape = tuple(tensors[0].shape)
        return pre_hook

    def _mk_hook(rec: _ModuleRec):
        def hook(_mod, _inp, out):
            if rec.order < 0:
                rec.order = counter["i"]
                counter["i"] += 1
                tensors = _iter_tensors(out)
                rec.out_ids = tuple(id(t) for t in tensors)
                if tensors:
                    rec.out_shape = tuple(tensors[0].shape)
        return hook

    for rec in recs.values():
        handles.append(
            rec.module.register_forward_pre_hook(_mk_pre_hook(rec),
                                                  with_kwargs=True))
        handles.append(rec.module.register_forward_hook(_mk_hook(rec)))

    traced = False
    input_ids: Set[int] = set()
    prev_training = model.training
    try:
        dummy = torch.zeros(*in_shape, dtype=torch.float32)
        input_ids = {id(dummy)}
        for mode in (True, False):  # 先 train()（激活 aux/ds/topo），失败再 eval()
            try:
                model.train(mode)
                with torch.no_grad():
                    model(dummy)
                traced = True
                break
            except Exception as e:  # 换模式重试
                logger.debug("model_flow: forward(train=%s) 失败: %s", mode, e)
                counter["i"] = 0
                for r in recs.values():  # 清空上次半成品记录
                    r.order = -1
                    r.in_ids = r.out_ids = ()
                    r.in_shape = r.out_shape = None
    finally:
        for h in handles:
            h.remove()
        model.train(prev_training)
    return recs, input_ids, traced


# ---------------------------------------------------------------------------
# 损失节点
# ---------------------------------------------------------------------------
def _loss_node_detail(cfg: Config) -> Dict[str, object]:
    detail: Dict[str, object] = {
        "loss.name": cfg.loss.name,
        "deep_supervision": bool(cfg.model.deep_supervision),
        "aux_seg_supervision": bool(cfg.model.aux_seg_supervision),
        "aux_topo_head": bool(getattr(cfg.model, "aux_topo_head", False)),
    }
    try:
        from ..losses.losses import build_loss
        from ..trainer.pipelines.factory import build_pipeline
        pipe = build_pipeline(cfg, build_loss(cfg.loss))
        detail["pipeline"] = type(pipe).__name__
        detail["criterion"] = type(getattr(pipe, "criterion", "")).__name__
        if getattr(pipe, "aux_weights", None):
            detail["aux_weights"] = list(pipe.aux_weights)
    except Exception as e:  # pipeline 构造失败仅缺补充信息，不致命
        logger.debug("model_flow: loss 详情 pipeline 构造失败: %s", e)
    return detail


# ---------------------------------------------------------------------------
# DAG 重建：分组 → 节点 → 连边 → 层级
# ---------------------------------------------------------------------------
class _ModelGraphBuilder:
    """把 ``_ModuleRec`` 集合重建为 ``VisGraph`` 的 DAG。"""

    def __init__(self, g: VisGraph, recs: Dict[str, _ModuleRec],
                 input_ids: Set[int], traced: bool):
        self.g = g
        self.recs = recs
        self.input_ids = input_ids
        self.traced = traced
        self.residual_types = _residual_block_types()

        # 叶子（被执行过；未追踪时取全部声明叶子）。
        self.leaves: Dict[str, _ModuleRec] = {
            name: rec for name, rec in recs.items()
            if not list(rec.module.children())
            and (rec.order >= 0 or not traced)
        }
        # 叶子输出张量 id → 产出叶子名（用于上游反查）。
        self.producer: Dict[int, str] = {}
        for name, rec in self.leaves.items():
            for tid in rec.out_ids:
                self.producer.setdefault(tid, name)

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

    # -- 排序键 --------------------------------------------------------
    def _order_of(self, recs: Sequence[_ModuleRec]) -> Tuple[int, int]:
        orders = [r.order for r in recs if r.order >= 0]
        if orders:
            return (0, min(orders))
        return (1, 1 << 30)

    def _top_sort_key(self, key: str):
        oo = self._order_of(self.top_leaves[key])
        return (oo[0], oo[1], self.top_order.index(key))

    # -- 上游反查 ------------------------------------------------------
    def _producers_at_top(self, in_ids: Sequence[int], exclude: str) -> List[str]:
        """输入张量 id 列表 → 去重的上游顶层容器（或 'input'）。"""
        srcs: List[str] = []
        seen: Set[str] = set()
        for tid in in_ids:
            if tid in self.input_ids:
                src = "input"
            else:
                prod = self.producer.get(tid)
                src = _top_key(prod) if prod else None
            if src and src != exclude and src not in seen:
                if src == "input" or src in self.top_set:
                    seen.add(src)
                    srcs.append(src)
        return srcs

    def _producers_in_top(
        self, in_ids: Sequence[int], top: str, exclude: str,
    ) -> List[str]:
        """容器内：输入张量 id → 同容器内的上游 block 键（外部来源忽略）。"""
        srcs: List[str] = []
        seen: Set[str] = set()
        for tid in in_ids:
            prod = self.producer.get(tid)
            if not prod or _top_key(prod) != top:
                continue
            bk = _block_key(prod, top) or prod
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
    input_ids: Set[int] = set()
    traced = False
    if trace_shapes:
        try:
            recs, input_ids, traced = _trace_modules(model, in_shape)
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

    builder = _ModelGraphBuilder(g, recs, input_ids, traced)
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


def _emit_top(b: _ModelGraphBuilder, key: str) -> None:
    """发射一个顶层容器，及其内部 block / 叶子节点与容器内连边。"""
    g = b.g
    recs = sorted(b.top_leaves[key],
                  key=lambda r: r.order if r.order >= 0 else 1 << 30)
    kind = "head" if _is_head(key) else "stage"

    ins = [r.in_shape for r in recs if r.in_shape]
    outs = [r.out_shape for r in recs if r.out_shape]
    s_key: Dict[str, object] = {"ops": str(len(recs))}
    if ins:
        s_key["in"] = shape_str(ins[0])
    if outs:
        s_key["out"] = shape_str(outs[-1])
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
            for rec in members:           # 直接挂顶层容器
                _emit_leaf(b, rec, parent=key)
        else:
            _emit_block(b, bk, key, members)

    # 容器内连边：block / 叶子间的真实流 + 结构化补边。
    _emit_intra_edges(b, key, block_recs, block_order)


def _emit_block(b: _ModelGraphBuilder, block_key: str, top: str,
                members: List[_ModuleRec]) -> None:
    g = b.g
    members = sorted(members, key=lambda r: r.order if r.order >= 0 else 1 << 30)
    mod = b.recs[block_key].module if block_key in b.recs else members[0].module
    ki: Dict[str, object] = {"type": type(mod).__name__}
    outs = [r.out_shape for r in members if r.out_shape]
    if outs:
        ki["out"] = shape_str(outs[-1])
    residual = _has_residual(mod, b.residual_types)
    if residual:
        ki["skip"] = "residual (+)"
    g.add_node(block_key, _block_label(block_key, top),
               kind="stage", key_info=ki, parent_id=top, collapsed=True)

    leaf_ids: List[str] = [_emit_leaf(b, rec, parent=block_key) for rec in members]

    # 叶子间真实流连边：按执行序流式解析"我的输入由哪个兄弟叶子产出"，
    # 正确处理 in-place / Identity 造成的张量 id 复用（避免 norm 扇出到多叶子）。
    # 把 block 外部输入 id 标记为 external：identity/shortcut 透传不得"抢占"它，
    # 从而 shortcut 与主路首叶都作为并联入口、互不串接。
    block_rec = b.recs.get(block_key)
    external: Set[int] = set(block_rec.in_ids) if block_rec else set()
    cur: Dict[int, str] = {}
    has_succ: Set[str] = set()
    for rec, cid in zip(members, leaf_ids):
        for tid in rec.in_ids:
            if tid in external:
                continue
            src = cur.get(tid)
            if src and src != cid:
                b._add_edge(src, cid)
                has_succ.add(src)
        for tid in rec.out_ids:
            if tid in external:           # 不抢占外部输入 id（透传算子）
                continue
            cur[tid] = cid

    # 残差/融合 functional 算子（+ / cat）不产生 module，会在主路与输出叶子间断链。
    # 结构化补回：把"悬空主路叶子"接到输出叶子；shortcut 叶子以 residual 边接入。
    if residual and len(leaf_ids) >= 2:
        out_leaf = leaf_ids[-1]              # 末叶（执行序最后）= block 输出
        shortcut_ids = {cid for rec, cid in zip(members, leaf_ids)
                        if ".shortcut" in rec.name}
        for rec, cid in zip(members, leaf_ids):
            if cid == out_leaf:
                continue
            if cid in shortcut_ids:
                b._add_edge(cid, out_leaf, "+", kind="residual")
            elif cid not in has_succ:        # 悬空主路末端 → 输出叶子
                b._add_edge(cid, out_leaf)
        if not shortcut_ids:                 # 无显式 shortcut（如 ConvNeXt 恒等残差）
            b._add_edge(leaf_ids[0], out_leaf, "+", kind="residual")


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
    """顶层容器的入边：用容器模块自身输入张量 id 反查上游顶层容器。"""
    rec = b.recs.get(key)
    if rec is None or not rec.in_ids:
        return
    for src in b._producers_at_top(rec.in_ids, exclude=key):
        b._add_edge(src, key)


def _emit_intra_edges(
    b: _ModelGraphBuilder, top: str,
    block_recs: Dict[Optional[str], List[_ModuleRec]],
    block_order: List[Optional[str]],
) -> None:
    """容器内 block 间连边：真实张量流 + 已知融合容器的结构化补边。"""
    # 真实流：每个 block 用其模块输入 id 反查同容器内上游 block。
    real_blocks = [bk for bk in block_order if bk is not None]
    for bk in real_blocks:
        rec = b.recs.get(bk)
        if rec is None or not rec.in_ids:
            continue
        for src in b._producers_in_top(rec.in_ids, top, exclude=bk):
            b._add_edge(src, bk)

    _structural_fusion_edges(b, top, real_blocks)


def _structural_fusion_edges(
    b: _ModelGraphBuilder, top: str, blocks: List[str],
) -> None:
    """对已知"含 functional 融合(cat/+)"的容器补结构化边。

    这些融合点（split/cat）会产生无 module 产出的新张量，纯张量 id 反查会断链；
    据容器类型补回并联分支 → 融合节点 的连边。
    """
    mod = b.recs[top].module if top in b.recs else None
    if mod is None:
        return

    # MultiStemProj / 分层 stem：三路 sub-stem 并联 → proj/fuse 融合。
    if _is_multistem(mod):
        stems = [bk for bk in blocks if _rel(bk, top).startswith("stems.")]
        fuses = [bk for bk in blocks if not _rel(bk, top).startswith("stems.")]
        for s in stems:
            for f in fuses:
                b._add_edge(s, f, "cat")
        return

    # DecoderLevel：upsample → 首个 stage block（skip 由顶层跳连边进入本框）。
    if _is_decoder_level(mod):
        ups = [bk for bk in blocks if "upsample" in _rel(bk, top)]
        body = [bk for bk in blocks
                if "upsample" not in _rel(bk, top)
                and "gate" not in _rel(bk, top).lower()]
        if ups and body:
            # body 内第一个（按已有顺序）即融合后主路入口。
            first_body = body[0]
            for u in ups:
                b._add_edge(u, first_body, "cat")
        return
    # 其余容器：依赖张量流真实边，无需结构化补边。


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

    # 顶层：heads 并排到同一行、loss 殿后。
    if heads:
        head_rank = max((node_rank.get(h, 0) for h in heads), default=0)
        for h in heads:
            node_rank[h] = head_rank
        if "loss" in node_rank:
            node_rank["loss"] = head_rank + 1
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


# -- 小工具 ----------------------------------------------------------------
def _rel(block_key: str, top: str) -> str:
    return block_key[len(top) + 1:] if block_key.startswith(top + ".") else block_key


def _is_multistem(m: nn.Module) -> bool:
    try:
        from ..models.stem import MultiStemProj
        if isinstance(m, MultiStemProj):
            return True
    except Exception:  # pragma: no cover
        pass
    return type(m).__name__ in ("MultiStemProj", "HierarchicalMultiStem",
                                "HierarchicalStems")


def _is_decoder_level(m: nn.Module) -> bool:
    try:
        from ..models.unet import DecoderLevel
        if isinstance(m, DecoderLevel):
            return True
    except Exception:  # pragma: no cover
        pass
    return type(m).__name__ == "DecoderLevel"


__all__ = ["build_model_flow"]
