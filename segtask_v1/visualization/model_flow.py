"""模型流 builder：``输入 → encoder → decoder → 各输出头 → 损失``。

追踪引擎为 **torchlens**（eager 逐 op 追踪）：跑一次 dummy 前向，拿到
「op 级真值 DAG + 每个 op 的模块归属链 + 形状」，再映射到 ``VisGraph``：

* **分组 = 模块树本身**：容器框 = 前向中被真实调用的 nn.Module（含嵌套），
  不依赖任何容器命名白名单——新增模块自动获得正确的框；
* **连边 = op 级 DAG 收缩 + 按层级上提**：叶子模块内部的多个 op 收缩为单节点，
  functional 算子（reshape/permute/interpolate…）作为"导线"透传；每条收缩边
  上提到两端在**最近公共容器**下的兄弟层级发射，天然支持折叠展示；
* **残差/融合 = DAG 结构判定**：任何有 ≥2 个不同上游的 cat/+/× 算子即为融合点
  （显式 merge 节点）；若融合的某一路上游可沿 DAG 到达另一路，则该边为残差捷径。
  不依赖 block 类型清单与 shortcut 属性名。

torchlens 不可用或前向失败时，退化为**纯结构图**（按 ``named_modules`` 声明序
线性链），保证任何情况下都有可读骨架。
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from taskcore.config.core import Config
from taskcore.models.topology import ModelTopology, build_topology
from .data_flow import _model_input_shape
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


def _fmt_params(n: int) -> str:
    """参数量人读化：1234 → '1.2K'，3456789 → '3.46M'。"""
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


# torch 函数名 → 融合点展示符号。未登记的多源算子回退到通用 "merge"。
_MERGE_OP_SYMBOLS: Dict[str, str] = {
    "cat": "cat", "concat": "cat", "concatenate": "cat", "stack": "cat",
    "hstack": "cat", "vstack": "cat", "dstack": "cat", "column_stack": "cat",
    "add": "+", "add_": "+", "__add__": "+", "__iadd__": "+", "__radd__": "+",
    "sub": "−", "sub_": "−", "subtract": "−", "__sub__": "−",
    "mul": "×", "mul_": "×", "multiply": "×", "__mul__": "×",
    "maximum": "max", "max": "max", "minimum": "min", "min": "min",
    # lerp(a, b, w) = a + w·(b−a)：幅度保持网络（如 EDM2）用作加性融合。
    "lerp": "lerp", "lerp_": "lerp",
}
# 参与残差捷径判定的加性融合符号。乘法（×）是门控/调制（attention gate、SE、
# GLU 等：`y = x·g(x)` 中 x 是被门控的主信号而非恒等捷径），不属残差语义，
# 保持普通 forward + × 融合点展示。
_ADDITIVE_SYMBOLS = {"+", "−", "lerp"}


# ---------------------------------------------------------------------------
# torchlens 追踪
# ---------------------------------------------------------------------------
class _Op:
    """torchlens 单 op 的最小快照（脱离 ModelHistory 生命周期）。"""

    __slots__ = ("label", "step", "func", "shape", "mods",
                 "parents", "children", "is_input", "is_output")

    def __init__(self, label: str, step: int, func: str,
                 shape: Optional[Tuple[int, ...]],
                 mods: List[Tuple[str, int]],
                 parents: List[str], children: List[str],
                 is_input: bool, is_output: bool):
        self.label = label
        self.step = step
        self.func = func
        self.shape = shape
        self.mods = mods
        self.parents = parents
        self.children = children
        self.is_input = is_input
        self.is_output = is_output


def _parse_mods(mods) -> List[Tuple[str, int]]:
    """torchlens ``Layer.modules``（``["encoder:1", "encoder.stages.0:2", …]``，
    外层→内层）→ ``[(模块路径, 第几次调用), …]``。"""
    out: List[Tuple[str, int]] = []
    for m in mods or []:
        s = str(m)
        path, _, call = s.rpartition(":")
        if path and call.isdigit():
            out.append((path, int(call)))
        else:
            out.append((s, 1))
    return out


def _tl_trace(model: nn.Module, in_shape: Tuple[int, ...]) -> List[_Op]:
    """torchlens 跑一次 dummy 前向，抽取 op 快照列表（执行序）。

    先 ``train()``（激活 aux / deep-supervision / topo 等仅训练期输出的头），
    失败再 ``eval()``；两者都失败抛出最后一次异常，由上层降级为纯结构图。
    ``save=None`` 只留元数据、不驻留中间激活，避免 3D 大前向撑爆内存。
    """
    import copy

    import torchlens as tl

    dummy = torch.zeros(*in_shape, dtype=torch.float32)
    # train 模式前向自带副作用（BN running stats 更新、EDM2 强制 weight-norm 的
    # 就地 copy_、dropout 消耗全局随机数），`no_grad` 拦不住。追踪一份 deepcopy
    # 副本并隔离 RNG，保证可视化对待训练模型与随机数序列零副作用。
    try:
        target = copy.deepcopy(model)
    except Exception as e:
        logger.warning(
            "model_flow: 模型 deepcopy 失败（%s），退回原模型追踪（train 模式副作用"
            "可能泄漏到训练模型）。", e)
        target = model
    prev_training = target.training
    last_err: Optional[Exception] = None
    try:
        with torch.random.fork_rng(devices=[]):
            for mode in (True, False):
                target.train(mode)
                try:
                    with torch.no_grad():
                        hist = tl.trace(target, dummy, save=None)
                    break
                except Exception as e:
                    logger.debug(
                        "model_flow: torchlens(train=%s) 失败: %s", mode, e)
                    last_err = e
            else:
                raise last_err if last_err else RuntimeError("torchlens trace 失败")
    finally:
        target.train(prev_training)

    def _canon(lbl: str) -> str:
        # op 引用统一带 ":pass" 后缀（单 pass 引用常省略 ":1"）。
        return lbl if ":" in lbl else f"{lbl}:1"

    # 逐 **op**（执行实例）遍历而非逐 layer：同一 functional 层被多次执行
    # （如 SelfAttentionBlock 的两次 `x+…` 残差加法）时 layer 级视图会把
    # 多个 pass 合并成一个带自环的节点，丢失第二次加法。
    ops: List[_Op] = []
    for step, op in enumerate(hist):
        if getattr(op, "is_buffer", False):  # BN running-stats 等与数据流无关
            continue
        shape = tuple(op.shape) if op.shape else None
        ops.append(_Op(
            label=_canon(str(op.label)), step=step,
            func=str(op.func_name or ""),
            shape=shape, mods=_parse_mods(op.modules),
            parents=[_canon(str(p)) for p in (op.parents or [])],
            children=[_canon(str(c)) for c in (op.children or [])],
            is_input=bool(op.is_input), is_output=bool(op.is_output)))
    return ops


def _prune_to_dataflow(ops: List[_Op]) -> Dict[str, _Op]:
    """只保留位于「输入→输出」通路上的 op（剔除 buffer 更新链等旁支）。"""
    by_label = {op.label: op for op in ops}
    labels = set(by_label)

    def _reach(seeds: List[str], nbrs) -> Set[str]:
        seen: Set[str] = set()
        stack = [s for s in seeds if s in labels]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(n for n in nbrs(by_label[cur]) if n in labels)
        return seen

    fwd = _reach([o.label for o in ops if o.is_input],
                 lambda o: o.children)
    bwd = _reach([o.label for o in ops if o.is_output],
                 lambda o: o.parents)
    keep = fwd & bwd
    return {lbl: op for lbl, op in by_label.items() if lbl in keep}


# ---------------------------------------------------------------------------
# 适配层：op 级 DAG → VisGraph
# ---------------------------------------------------------------------------
def _inst_id(path: str, call: int) -> str:
    """模块实例（路径 + 第几次调用）→ 图节点 id。首个调用不带后缀。"""
    return path if call <= 1 else f"{path}@{call}"


class _Adapter:
    """把 torchlens op 快照收缩、分组并发射为 ``VisGraph``。"""

    def __init__(self, g: VisGraph, model: nn.Module, ops: Dict[str, _Op]):
        self.g = g
        self.ops = ops
        self.order = sorted(ops.values(), key=lambda o: o.step)
        self.modules: Dict[str, nn.Module] = {
            name: m for name, m in model.named_modules() if name}
        self.leaf_paths: Set[str] = {
            name for name, m in self.modules.items() if not list(m.children())}

        # op → 所属叶子实例 / 容器实例链。
        self.op_owner: Dict[str, Optional[str]] = {}        # 叶子实例 id
        self.op_containers: Dict[str, List[str]] = {}       # 容器实例 id 链（外→内）
        for op in self.order:
            chain: List[str] = []
            owner: Optional[str] = None
            for path, call in op.mods:
                iid = _inst_id(path, call)
                if path in self.leaf_paths and (path, call) == op.mods[-1]:
                    owner = iid
                else:
                    chain.append(iid)
            self.op_owner[op.label] = owner
            self.op_containers[op.label] = chain

        # 收缩：op → 代表节点集合（"input" / leaf 实例 / merge 节点 id）。
        self.op_reps: Dict[str, Set[str]] = {}
        # 材料化节点集合与其父容器实例链。
        self.node_chain: Dict[str, List[str]] = {}
        # 材料化 DAG 边（收缩后，去重保序）。
        self.cedges: List[Tuple[str, str]] = []
        self._cedge_set: Set[Tuple[str, str]] = set()
        # merge 节点元数据：id → (符号, 输出形状, 源节点列表)。
        self.merges: Dict[str, Tuple[str, Optional[Tuple[int, ...]], List[str]]] = {}
        # 叶子实例元数据。
        self.leaf_ops: Dict[str, List[_Op]] = {}

    # -- 收缩 op 级 DAG ---------------------------------------------------
    def contract(self) -> None:
        for op in self.order:
            up: Set[str] = set()
            for p in op.parents:
                up |= self.op_reps.get(p, set())
            if op.is_input:
                self.op_reps[op.label] = {"input"}
                self.node_chain.setdefault("input", [])
                continue
            owner = self.op_owner[op.label]
            if owner is not None:
                nid = f"leaf::{owner}"
                if nid not in self.leaf_ops:
                    self.leaf_ops[nid] = []
                    self.node_chain[nid] = self.op_containers[op.label]
                self.leaf_ops[nid].append(op)
                for src in up:
                    if src != nid:
                        self._add_cedge(src, nid)
                self.op_reps[op.label] = {nid}
                continue
            # functional op：≥2 个不同上游 → 显式融合节点；否则为导线透传。
            if len(up) >= 2:
                sym = _MERGE_OP_SYMBOLS.get(op.func, "merge")
                mid = f"merge::{op.label}"
                self.merges[mid] = (sym, op.shape, sorted(up))
                self.node_chain[mid] = self.op_containers[op.label]
                for src in up:
                    self._add_cedge(src, mid)
                self.op_reps[op.label] = {mid}
            else:
                self.op_reps[op.label] = up

    def _add_cedge(self, src: str, dst: str) -> None:
        if src != dst and (src, dst) not in self._cedge_set:
            self._cedge_set.add((src, dst))
            self.cedges.append((src, dst))

    # -- 残差判定：add 类融合中，可达另一路上游的那一路是捷径 ---------------
    def residual_edges(self) -> Set[Tuple[str, str]]:
        succ: Dict[str, List[str]] = {}
        for s, d in self.cedges:
            succ.setdefault(s, []).append(d)

        memo: Dict[Tuple[str, str], bool] = {}

        def reaches(a: str, b: str) -> bool:
            key = (a, b)
            if key in memo:
                return memo[key]
            memo[key] = False  # 防环（理论上 DAG 无环，防御式）
            seen: Set[str] = set()
            stack = [a]
            found = False
            while stack:
                cur = stack.pop()
                if cur == b:
                    found = True
                    break
                if cur in seen:
                    continue
                seen.add(cur)
                stack.extend(succ.get(cur, ()))
            memo[key] = found
            return found

        pred: Dict[str, List[str]] = {}
        for s, d in self.cedges:
            pred.setdefault(d, []).append(s)

        def back_dists(a: str) -> Dict[str, int]:
            """反向 BFS：a 的每个祖先 → 最短跳数（含 a 自身 0）。"""
            dist = {a: 0}
            frontier = [a]
            while frontier:
                nxt: List[str] = []
                for cur in frontier:
                    for p in pred.get(cur, ()):
                        if p not in dist:
                            dist[p] = dist[cur] + 1
                            nxt.append(p)
                frontier = nxt
            return dist

        out: Set[Tuple[str, str]] = set()
        for mid, (sym, _sh, srcs) in self.merges.items():
            if sym not in _ADDITIVE_SYMBOLS or len(srcs) < 2:
                continue
            # 规则 1：某一路上游 a 可沿 DAG 到达另一路 b ⇒ a 是恒等捷径。
            marked = False
            for a in srcs:
                if any(a != b and reaches(a, b) for b in srcs):
                    out.add((a, mid))
                    marked = True
            if marked:
                continue
            # 规则 2（投影捷径 ``out = f(x) + shortcut(x)``）：两路从公共分叉点
            # 出发，显著更短的一路（≤ 另一路一半深度）视为捷径。等长分支
            # （如 MultiRF 的多感受野加和）不标残差，保持普通融合。
            if len(srcs) == 2:
                da, db = back_dists(srcs[0]), back_dists(srcs[1])
                common = set(da) & set(db)
                if common:
                    fork = min(common, key=lambda c: da[c] + db[c])
                    la, lb = da[fork], db[fork]
                    if la * 2 <= lb and la < lb:
                        out.add((srcs[0], mid))
                    elif lb * 2 <= la and lb < la:
                        out.add((srcs[1], mid))
        return out

    # -- 容器树整理 --------------------------------------------------------
    def build_containers(self) -> Tuple[Dict[str, Optional[str]], List[str]]:
        """材料化节点的容器归属 → 实际需要的容器集合与父子关系。

        单子容器**向上合并**：若容器 A 的成员只有唯一的子容器 B（无其它叶/merge），
        B 的成员直接归入 A（B 不再单独成框），避免包裹层（如 Sequential(stage)）
        产生的空嵌套。返回 ``(节点/容器 → 父容器, 容器发射顺序)``。
        """
        members: Dict[Optional[str], List[str]] = {}   # 容器 → 直接成员（节点+子容器）
        parent: Dict[str, Optional[str]] = {}
        cont_order: List[str] = []
        seen_cont: Set[str] = set()
        for nid, chain in self.node_chain.items():
            for i, cid in enumerate(chain):
                if cid not in seen_cont:
                    seen_cont.add(cid)
                    cont_order.append(cid)
                    parent[cid] = chain[i - 1] if i else None
                    members.setdefault(chain[i - 1] if i else None, []).append(cid)
            p = chain[-1] if chain else None
            parent[nid] = p
            members.setdefault(p, []).append(nid)

        # 迭代合并"只有唯一子容器"的容器（把子并入父，保住外层语义框）。
        changed = True
        while changed:
            changed = False
            for cid in list(seen_cont):
                kids = members.get(cid, [])
                if len(kids) != 1 or kids[0] not in seen_cont:
                    continue
                child = kids[0]
                grandkids = members.pop(child, [])
                for gk in grandkids:
                    parent[gk] = cid
                members[cid] = grandkids
                seen_cont.discard(child)
                parent.pop(child, None)
                changed = True
        order = [c for c in cont_order if c in seen_cont]
        return parent, order

    # -- 形状 ---------------------------------------------------------------
    def leaf_io(self, nid: str) -> Tuple[Optional[Tuple[int, ...]],
                                         Optional[Tuple[int, ...]]]:
        group = self.leaf_ops.get(nid, [])
        gset = {o.label for o in group}
        in_sh = next(
            (self.ops[p].shape for o in group for p in o.parents
             if p in self.ops and p not in gset and self.ops[p].shape),
            None)
        out_sh = next(
            (o.shape for o in reversed(group)
             if o.shape and (o.is_output
                             or any(c not in gset for c in o.children))),
            None)
        if out_sh is None:
            out_sh = next((o.shape for o in reversed(group) if o.shape), None)
        return in_sh, out_sh

    def container_io(self, cid: str, node_parent: Dict[str, Optional[str]],
                     ) -> Tuple[Optional[Tuple[int, ...]],
                                Optional[Tuple[int, ...]], int]:
        """容器 in/out 形状与内部 op 数：取容器内首个「有外部输入」op 的入形状、
        末个「有外部输出」op 的出形状（直接读容器实例覆盖的全部 op）。"""
        inside = [op for op in self.order
                  if cid in self.op_containers[op.label]]
        iset = {o.label for o in inside}
        in_sh = next(
            (self.ops[p].shape for o in inside for p in o.parents
             if p in self.ops and p not in iset and self.ops[p].shape),
            None)
        out_sh = next(
            (o.shape for o in reversed(inside)
             if o.shape and (o.is_output
                             or any(c not in iset for c in o.children))),
            None)
        return in_sh, out_sh, len(inside)


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
    in_shape = _model_input_shape(cfg, topo)
    trace_shape = (1,) + tuple(in_shape[1:])  # 追踪 batch 固定为 1（省算力）

    g = VisGraph(title="模型流 Model Flow")

    ops: Dict[str, _Op] = {}
    if trace_shapes:
        try:
            ops = _prune_to_dataflow(_tl_trace(model, trace_shape))
        except Exception as e:  # 追踪整体失败 → 纯结构
            logger.warning("model_flow: torchlens 追踪失败，退化为纯结构图: %s", e)

    g.meta = {
        "arch": str(cfg.model.arch),
        "input": shape_str(in_shape),
        "trace_batch": "1",
        "shapes": "traced" if ops else "static (前向未执行/失败)",
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

    if ops:
        _emit_traced(g, cfg, model, ops, in_shape)
    else:
        _emit_structural(g, cfg, model)
    return g


def _res_level(out_sh: Optional[Tuple[int, ...]], in_w: int) -> int:
    """分辨率级：输出末维（W）相对模型输入缩小 2^level 倍（四舍五入取整）。

    只看空间末维，对通道数/维度数不敏感；形状未知或非正时回退 0（不缩进）。"""
    if not out_sh or in_w <= 0:
        return 0
    w = out_sh[-1]
    if not isinstance(w, int) or w <= 0 or w > in_w:
        return 0
    return max(0, int(round(math.log2(in_w / w))))


def _is_head_container_by_name(cid: str) -> bool:
    """命名启发式判定输出头，**仅供纯结构降级路径**（无追踪信息时无法知道
    哪个子模块产出模型输出）。追踪路径用结构判定：产出模型输出的顶层容器
    即输出头（见 ``_emit_traced``）。框着色用；不参与分组/连边判定。"""
    return "head" in cid.split(".")[0].lower()


def _emit_traced(g: VisGraph, cfg: Config, model: nn.Module,
                 ops: Dict[str, _Op], in_shape: Tuple[int, ...]) -> None:
    in_w = int(in_shape[-1]) if in_shape else 0
    a = _Adapter(g, model, ops)
    a.contract()
    residual = a.residual_edges()
    parent, cont_order = a.build_containers()

    # 输出头的结构化判定：产出模型输出（op.is_output）的顶层容器即输出头，
    # 不依赖 "head" 命名。
    def _top_of(nid: str) -> str:
        cur = nid
        while parent.get(cur) is not None:
            cur = parent[cur]
        return cur

    out_tops: List[str] = []
    for op in a.order:
        if not op.is_output:
            continue
        for r in a.op_reps.get(op.label) or set():
            top = _top_of(r)
            if top not in out_tops:
                out_tops.append(top)
    head_tops = set(out_tops) & set(cont_order)

    node_first_step: Dict[str, int] = {"input": -1}
    for nid, group in a.leaf_ops.items():
        node_first_step[nid] = min(o.step for o in group)
    for mid in a.merges:
        node_first_step[mid] = a.ops[mid[len("merge::"):]].step \
            if mid[len("merge::"):] in a.ops else 1 << 30

    # 1) 容器框（按前向首次进入顺序）。
    def _cont_step(cid: str) -> int:
        return min((op.step for op in a.order
                    if cid in a.op_containers[op.label]), default=1 << 30)

    for cid in sorted(cont_order, key=_cont_step):
        in_sh, out_sh, n_ops = a.container_io(cid, parent)
        mod = a.modules.get(cid.split("@")[0])
        ki: Dict[str, object] = {}
        if mod is not None:
            ki["type"] = type(mod).__name__
            n_p = sum(prm.numel() for prm in mod.parameters())
            if n_p:
                ki["params"] = _fmt_params(n_p)
        ki["ops"] = str(n_ops)
        if in_sh:
            ki["in"] = shape_str(in_sh)
        if out_sh:
            ki["out"] = shape_str(out_sh)
        pid = parent.get(cid)
        label = cid[len(pid) + 1:] if pid and cid.startswith(pid + ".") else cid
        kind = "head" if pid is None and cid in head_tops else "stage"
        # 容器 res 取**入口**形状：res 不参与布局横坐标（单一主轴），仅
        # 作为卡片分辨率徽标呈现；取入口形状使折叠框的徽标与其进入时的
        # 分辨率一致（出口形状会把 encoder 标到瓶颈级、decoder 标到全
        # 分辨率级，语义跳变）。
        g.add_node(cid, label, kind=kind, key_info=ki,
                   parent_id=pid, collapsed=True,
                   res=_res_level(in_sh or out_sh, in_w))

    # 2) 叶子节点。
    for nid, group in sorted(a.leaf_ops.items(),
                             key=lambda kv: node_first_step[kv[0]]):
        inst = nid[len("leaf::"):]
        path, _, call = inst.partition("@")
        mod = a.modules.get(path)
        tname = type(mod).__name__ if mod is not None else path.split(".")[-1]
        in_sh, out_sh = a.leaf_io(nid)
        ki = {"type": tname}
        if mod is not None:
            n_p = sum(prm.numel() for prm in mod.parameters())
            if n_p:
                ki["params"] = _fmt_params(n_p)
        if out_sh:
            ki["out"] = shape_str(out_sh)
        detail = _leaf_params(mod) if mod is not None else {}
        detail["module"] = path
        if call:
            detail["call"] = f"第 {call} 次调用（权重共享/复用）"
        if in_sh:
            detail["in_shape"] = shape_str(in_sh)
        if out_sh:
            detail["out_shape"] = shape_str(out_sh)
        label = tname if not call else f"{tname} ×{call}"
        g.add_node(nid, label,
                   kind=_leaf_kind(mod) if mod is not None else "op",
                   key_info=ki, detail=detail, parent_id=parent.get(nid),
                   res=_res_level(out_sh, in_w))

    # 3) 融合节点。
    for mid, (sym, out_sh, _srcs) in sorted(
            a.merges.items(), key=lambda kv: node_first_step[kv[0]]):
        ki = {}
        if out_sh:
            ki["out"] = shape_str(out_sh)
        g.add_node(mid, sym, kind="merge", key_info=ki,
                   detail={"op": sym, "kind": "merge (functional fusion)"},
                   parent_id=parent.get(mid),
                   res=_res_level(out_sh, in_w))

    # 4) 连边：直接在**最深的材料化端点**（叶子/merge/input）级别发射。
    # 渲染层的 visibleAnchor 会按折叠状态把端点动态上提到可见的容器框，
    # 因此无需（也不应）在 builder 预先上提——预上提会把多条跳连去重成一条。
    for src, dst in a.cedges:
        kind = "residual" if (src, dst) in residual else "forward"
        label = a.merges[dst][0] if kind == "residual" and dst in a.merges else ""
        g.add_edge(src, dst, label, kind=kind)

    # 5) 损失节点 + 顶层输出容器 → loss。
    g.add_node("loss", f"Loss: {cfg.loss.name}", kind="loss",
               key_info={"name": cfg.loss.name}, detail=_loss_node_detail(cfg))
    for top in out_tops:
        g.add_edge(top, "loss", top.split(".")[0] if top in head_tops else "")

    # 6) 分层 rank（builder 内部量）+ 跳连标注。
    _assign_ranks_and_skip(g)


# ---------------------------------------------------------------------------
# 纯结构降级（torchlens 不可用 / 前向失败）
# ---------------------------------------------------------------------------
def _emit_structural(g: VisGraph, cfg: Config, model: nn.Module) -> None:
    """按 ``named_modules`` 声明序：顶层子模块各成一框、叶子线性链。"""
    prev = "input"
    for top_name, top in model.named_children():
        kind = "head" if _is_head_container_by_name(top_name) else "stage"
        leaves = [(n, m) for n, m in top.named_modules() if not list(m.children())]
        g.add_node(top_name, top_name, kind=kind,
                   key_info={"type": type(top).__name__,
                             "ops": str(len(leaves))},
                   collapsed=True)
        g.add_edge(prev, top_name)
        prev_leaf: Optional[str] = None
        for name, m in leaves:
            nid = f"leaf::{top_name}.{name}" if name else f"leaf::{top_name}"
            detail = _leaf_params(m)
            detail["module"] = f"{top_name}.{name}" if name else top_name
            g.add_node(nid, type(m).__name__, kind=_leaf_kind(m),
                       key_info={"type": type(m).__name__},
                       detail=detail, parent_id=top_name)
            if prev_leaf:
                g.add_edge(prev_leaf, nid)
            prev_leaf = nid
        prev = top_name
    g.add_node("loss", f"Loss: {cfg.loss.name}", kind="loss",
               key_info={"name": cfg.loss.name}, detail=_loss_node_detail(cfg))
    g.add_edge(prev, "loss")
    _assign_ranks_and_skip(g)


# ---------------------------------------------------------------------------
# 层级 rank 计算 + 跳连标注（通用：只依赖图结构）
# ---------------------------------------------------------------------------
def _assign_ranks_and_skip(g: VisGraph) -> None:
    """对每个父容器内的兄弟节点做最长路径分层（rank），并把"跨级边"标为 skip。

    边可能发射在叶子级别（跨容器）：先把每条边投影为「两端在最近公共容器下的
    兄弟对」，投影边参与该层级的分层；原始边按其投影的 rank 跨度标注 skip。
    """
    parent_of = {n.id: n.parent_id for n in g.nodes}
    by_parent: Dict[Optional[str], List[str]] = {}
    for n in g.nodes:
        by_parent.setdefault(n.parent_id, []).append(n.id)

    anc_cache: Dict[str, List[str]] = {}

    def _anc(nid: str) -> List[str]:  # 自身在首位的祖先链
        if nid not in anc_cache:
            chain = [nid]
            cur = parent_of.get(nid)
            while cur is not None:
                chain.append(cur)
                cur = parent_of.get(cur)
            anc_cache[nid] = chain
        return anc_cache[nid]

    def _project(src: str, dst: str) -> Optional[Tuple[str, str]]:
        """边 → 最近公共容器下的兄弟对；同一节点内部则 None。"""
        if src not in parent_of or dst not in parent_of:
            return None
        sc, dc = _anc(src), _anc(dst)
        sset = set(sc)
        lca = next((c for c in dc[1:] if c in sset), None)
        s2 = next((c for c in sc if parent_of.get(c) == lca), sc[-1])
        d2 = next((c for c in dc if parent_of.get(c) == lca), dc[-1])
        return None if s2 == d2 else (s2, d2)

    proj: Dict[int, Tuple[str, str]] = {}
    for i, e in enumerate(g.edges):
        pr = _project(e.src, e.dst)
        if pr is not None:
            proj[i] = pr

    node_rank: Dict[str, int] = {}
    for ids in by_parent.values():
        idset = set(ids)
        succ: Dict[str, List[str]] = {i: [] for i in ids}
        indeg: Dict[str, int] = {i: 0 for i in ids}
        seen_pairs: Set[Tuple[str, str]] = set()
        for i, e in enumerate(g.edges):
            if e.kind == "residual" or i not in proj:
                continue
            s2, d2 = proj[i]
            if s2 in idset and d2 in idset and (s2, d2) not in seen_pairs:
                seen_pairs.add((s2, d2))
                succ[s2].append(d2)
                indeg[d2] += 1
        # Kahn 最长路径；图中存在环（如 hierarchical 融合的反馈边）时，
        # 朴素 Kahn 会让环内节点永远到不了 indeg 0、全卡在 rank 0。队列耗尽但仍有
        # 未处理节点时贪心断环：选「已被环外前驱定到最高 rank」的入口节点强制入队
        # （其回边自然退化为反向边/skip）。结果对节点序确定、幂等。
        rank = {i: 0 for i in ids}
        indeg2 = dict(indeg)
        queue = [i for i in ids if indeg2[i] == 0]
        done: Set[str] = set()
        remaining = set(ids)
        order = {i: k for k, i in enumerate(ids)}
        while remaining:
            while queue:
                cur = queue.pop()
                if cur in done:
                    continue
                done.add(cur)
                remaining.discard(cur)
                for nx in succ[cur]:
                    if rank[cur] + 1 > rank[nx]:
                        rank[nx] = rank[cur] + 1
                    indeg2[nx] -= 1
                    if indeg2[nx] <= 0 and nx not in done:
                        queue.append(nx)
            if remaining:
                nxt = max(remaining, key=lambda i: (rank[i], -order[i]))
                indeg2[nxt] = 0
                queue.append(nxt)
        node_rank.update(rank)

    # 顶层输出头拍到同一末行、loss 殿后（按 kind 判定，不依赖命名）。
    tops = by_parent.get(None, [])
    kind_of = {n.id: n.kind for n in g.nodes}
    head_ids = [i for i in tops if kind_of.get(i) == "head"]
    if head_ids:
        fr = max(node_rank.get(h, 0) for h in head_ids)
        for h in head_ids:
            node_rank[h] = fr
    if "loss" in node_rank:
        upstream = [node_rank.get(e.src, 0) for e in g.edges if e.dst == "loss"]
        node_rank["loss"] = (max(upstream) if upstream else
                             max(node_rank.values(), default=0)) + 1

    for n in g.nodes:
        n.rank = node_rank.get(n.id, 0)

    # skip 标注（前向边，按投影判定）：
    #  a) 投影后 rank 跨度 > 1（含上行反馈边：同样要跨行绕线）；
    #  b) 旁路边——两端点之间除这条直连外还存在更长的 DAG 路径
    #     （UNet 的 encoder→decoder 跳连即典型：主干也能从 src 走到 dst）。
    rankmap = {n.id: n.rank for n in g.nodes}
    succ_all: Dict[str, Set[str]] = {}
    for e in g.edges:
        succ_all.setdefault(e.src, set()).add(e.dst)

    bypass_memo: Dict[Tuple[str, str], bool] = {}

    def _has_alt_path(u: str, v: str) -> bool:
        key = (u, v)
        if key in bypass_memo:
            return bypass_memo[key]
        seen: Set[str] = {u}
        stack = [nx for nx in succ_all.get(u, ()) if nx != v]  # 不走直连
        found = False
        while stack:
            cur = stack.pop()
            if cur == v:
                found = True
                break
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(succ_all.get(cur, ()))
        bypass_memo[key] = found
        return found

    for i, e in enumerate(g.edges):
        if e.kind != "forward" or i not in proj:
            continue
        s2, d2 = proj[i]
        if (abs(rankmap.get(d2, 0) - rankmap.get(s2, 0)) > 1
                or _has_alt_path(e.src, e.dst)):
            e.kind = "skip"


__all__ = ["build_model_flow"]
