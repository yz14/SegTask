"""模型流 builder：``输入 → encoder → decoder → 各输出头 → 损失``。

形状采集策略（与 PLAN 一致）：
1. 用一份 **CPU dummy 全零张量**（batch=1，尺寸取 pipeline 目标 patch）跑一次前向，
   通过 forward hook 抓取每个叶子模块（conv / norm / act / ...）的真实输入输出形状；
2. 前向在 ``model.train()`` 下进行（``no_grad``），以便 aux / deep-supervision / topo
   等"仅训练期输出"的头也被激活、出现在图中；
3. 前向若抛错（自定义 forward 签名等），降级为 ``eval()`` 重试；仍失败则退化为
   **纯结构图**（只读 ``named_modules`` 层级，不带形状）。

叶子按所属容器（encoder.stem / encoder.stages.k / decoder.levels.k / *_head）聚合成
可折叠 stage 大框，框内叶子按真实执行顺序排列；末端接一个损失框。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

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
# stage 聚合
# ---------------------------------------------------------------------------
def _stage_key(name: str) -> str:
    """叶子限定名 → 所属 stage 容器键（截到第一个整数索引）。"""
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part.isdigit():
            return ".".join(parts[: i + 1])
    if len(parts) >= 2 and parts[0] in ("encoder", "decoder"):
        return ".".join(parts[:2])
    return parts[0] if parts else name


def _stage_label(key: str) -> str:
    """stage 键 → 人类可读标题。"""
    parts = key.split(".")
    pretty = {
        "encoder.stem": "Encoder Stem",
        "encoder.aux_fuse": "Encoder Aux-Fuse",
        "seg_head": "Seg Head (main)",
        "topo_head": "Topo Head",
    }
    if key in pretty:
        return pretty[key]
    if len(parts) == 3 and parts[0] == "encoder" and parts[1] == "stages":
        return f"Encoder Stage {parts[2]}"
    if len(parts) == 3 and parts[0] == "encoder" and parts[1] == "downsamples":
        return f"Downsample {parts[2]}"
    if len(parts) == 3 and parts[0] == "decoder" and parts[1] == "levels":
        return f"Decoder Level {parts[2]}"
    if len(parts) == 2 and parts[0] == "ds_heads":
        return f"DS Head {parts[1]}"
    if len(parts) == 2 and parts[0] == "aux_heads":
        return f"Aux Head {parts[1]}"
    return key


def _is_head(key: str) -> bool:
    return (key in ("seg_head", "topo_head")
            or key.startswith("ds_heads.")
            or key.startswith("aux_heads."))


# ---------------------------------------------------------------------------
# 形状追踪
# ---------------------------------------------------------------------------
class _LeafRecord:
    __slots__ = ("name", "module", "order", "in_shape", "out_shape")

    def __init__(self, name: str, module: nn.Module):
        self.name = name
        self.module = module
        self.order: int = -1
        self.in_shape: Optional[Tuple[int, ...]] = None
        self.out_shape: Optional[Tuple[int, ...]] = None


def _trace_shapes(
    model: nn.Module, in_shape: Tuple[int, ...],
) -> Tuple[Dict[str, _LeafRecord], bool]:
    """注册 hook 跑一次 dummy 前向，回填每个叶子的真实形状。

    返回 ``(records, traced)``；``traced=False`` 表示前向失败、形状不可用。
    """
    leaves: Dict[str, _LeafRecord] = {}
    for name, m in model.named_modules():
        if name and not list(m.children()):  # 叶子（无子模块）
            leaves[name] = _LeafRecord(name, m)

    counter = {"i": 0}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def _mk_hook(rec: _LeafRecord):
        def hook(_mod, inp, out):
            if rec.order < 0:  # 仅记首次（避免被多次调用模块覆盖顺序）
                rec.order = counter["i"]
                counter["i"] += 1
                if isinstance(inp, (tuple, list)) and inp and torch.is_tensor(inp[0]):
                    rec.in_shape = tuple(inp[0].shape)
                if torch.is_tensor(out):
                    rec.out_shape = tuple(out.shape)
                elif isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]):
                    rec.out_shape = tuple(out[0].shape)
        return hook

    for name, rec in leaves.items():
        handles.append(rec.module.register_forward_hook(_mk_hook(rec)))

    traced = False
    prev_training = model.training
    try:
        dummy = torch.zeros(*in_shape, dtype=torch.float32)
        for mode in (True, False):  # 先 train()（激活 aux/ds/topo），失败再 eval()
            try:
                model.train(mode)
                with torch.no_grad():
                    model(dummy)
                traced = True
                break
            except Exception as e:  # 换模式重试
                logger.debug("model_flow: forward(train=%s) 失败: %s", mode, e)
    finally:
        for h in handles:
            h.remove()
        model.train(prev_training)
    return leaves, traced


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
# 主入口
# ---------------------------------------------------------------------------
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

    leaves: Dict[str, _LeafRecord] = {}
    traced = False
    if trace_shapes:
        try:
            leaves, traced = _trace_shapes(model, in_shape)
        except Exception as e:  # 追踪整体失败 → 纯结构
            logger.warning("model_flow: 形状追踪失败，退化为纯结构图: %s", e)
    if not leaves:
        # 纯结构降级：仅按 named_modules 收集叶子（无形状 / 无顺序）。
        for name, m in model.named_modules():
            if name and not list(m.children()):
                leaves[name] = _LeafRecord(name, m)

    g.meta = {
        "arch": str(cfg.model.arch),
        "input": shape_str(in_shape),
        "trace_batch": "1",
        "shapes": "traced" if traced else "static (前向未执行/失败)",
        "in_channels": str(topo.in_channels),
        "out_classes": str(topo.out_classes),
        "spatial_dims": f"{topo.spatial_dims}D",
    }

    # 模型输入框 -------------------------------------------------------
    g.add_node(
        "input", "模型输入", kind="input",
        key_info={"shape": shape_str(in_shape)},
        detail={"in_channels": str(topo.in_channels),
                "spatial_dims": f"{topo.spatial_dims}D",
                "n_views": str(topo.n_views)})

    # 按 stage 聚合叶子 ------------------------------------------------
    stage_leaves: Dict[str, List[_LeafRecord]] = {}
    stage_order_seen: List[str] = []
    for name, rec in leaves.items():
        key = _stage_key(name)
        if key not in stage_leaves:
            stage_leaves[key] = []
            stage_order_seen.append(key)
        stage_leaves[key].append(rec)

    def _stage_order(key: str) -> Tuple[int, int]:
        orders = [r.order for r in stage_leaves[key] if r.order >= 0]
        if orders:
            return (0, min(orders))               # 已追踪：按执行序
        return (1, stage_order_seen.index(key))   # 未追踪：按声明序

    backbone = sorted(
        (k for k in stage_leaves if not _is_head(k)), key=_stage_order)
    heads = sorted(
        (k for k in stage_leaves if _is_head(k)), key=_stage_order)

    def _emit_stage(key: str) -> None:
        recs = sorted(stage_leaves[key],
                      key=lambda r: r.order if r.order >= 0 else 1 << 30)
        kind = "head" if _is_head(key) else "stage"
        # stage 头部摘要：首叶入形状 → 末叶出形状。
        ins = [r.in_shape for r in recs if r.in_shape]
        outs = [r.out_shape for r in recs if r.out_shape]
        s_key: Dict[str, object] = {"ops": str(len(recs))}
        if ins:
            s_key["in"] = shape_str(ins[0])
        if outs:
            s_key["out"] = shape_str(outs[-1])
        g.add_node(key, _stage_label(key), kind=kind,
                   key_info=s_key, collapsed=True)
        prev_child: Optional[str] = None
        for j, rec in enumerate(recs):
            cid = f"{key}#{j}"
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
                       key_info=ki, detail=detail, parent_id=key)
            if prev_child is not None:
                g.add_edge(prev_child, cid)  # 框内执行序（折叠时自动隐藏）
            prev_child = cid

    for key in backbone:
        _emit_stage(key)
    for key in heads:
        _emit_stage(key)

    # 主干顺序连边（input → 各 backbone stage 链）---------------------
    prev = "input"
    for key in backbone:
        g.add_edge(prev, key)
        prev = key
    last_backbone = backbone[-1] if backbone else "input"

    # 损失节点 ---------------------------------------------------------
    g.add_node(
        "loss", f"Loss: {cfg.loss.name}", kind="loss",
        key_info={"name": cfg.loss.name},
        detail=_loss_node_detail(cfg))

    # 头分支：各 head 从最后主干特征引出 → 汇入 loss；无 head 时主干末端直连 loss。
    if heads:
        for key in heads:
            g.add_edge(last_backbone, key)
            label = "main" if key == "seg_head" else (
                "topo" if key == "topo_head" else
                "ds" if key.startswith("ds_heads.") else "aux")
            g.add_edge(key, "loss", label)
    else:
        g.add_edge(last_backbone, "loss")

    return g


__all__ = ["build_model_flow"]
