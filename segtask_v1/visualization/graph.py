"""可视化中间表示（IR）：``VisNode`` / ``VisEdge`` / ``VisGraph``。

与具体网络 / 流程无关的最小图模型，三个 builder（data_flow / model_flow /
predict_flow）把 ``cfg`` / ``model`` 翻译成 ``VisGraph``；renderer 只消费此 IR。
日后改网络或加流程，只动 builder、不动渲染层（职责分离）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


# 节点语义类别 —— renderer 据此着色 / 取图标。新增类别需在 render.py 的
# 调色板（_KIND_STYLES）中登记，否则回退到 "op" 默认样式。
NODE_KINDS = (
    "data",     # npz / 体积 / 张量数据态
    "process",  # 数据流的处理算子（取块 / 增强 / 裁剪 / 重塑）
    "input",    # 模型输入框（按分辨率分框）
    "stage",    # 模型阶段大框（encoder stage / decoder level / stem 等容器）
    "conv",     # 卷积类叶子
    "norm",     # 归一化类叶子
    "act",      # 激活类叶子
    "op",       # 其他叶子（pool / upsample / attention / add ...）
    "merge",    # 显式融合算子节点（cat / + / × ...：split→merge 的汇流点）
    "head",     # 输出头
    "output",   # 模型输出框
    "loss",     # 损失框
    "model",    # 预测流中抽象的"整模型"单框
)


@dataclass
class VisNode:
    """流程图节点。

    * ``id``        —— 图内唯一标识（builder 负责保证唯一）。
    * ``label``     —— 框内主标题。
    * ``kind``      —— 语义类别（见 ``NODE_KINDS``），决定配色 / 分组。
    * ``key_info``  —— 框内直接展示的关键信息（有序 dict，键值都转字符串）。
    * ``detail``    —— 双击详情抽屉里的完整参数（有序 dict）。
    * ``parent_id`` —— 所属容器节点 id（如 stage）；顶层节点为 ``None``。
    * ``collapsed`` —— 容器默认是否折叠（仅对有子节点的容器有意义）。
    * ``rank``      —— 同一父容器内的纵向层级（0 起）；同 rank 的兄弟节点横向并排（并联分支），
      不同 rank 自上而下排列。renderer 据此把并联结构排成一行。
    * ``col`` / ``colspan`` —— 同一父容器内的横向列位（0 起）与跨列宽：renderer 据 ``(rank, col,
      colspan)`` 把节点摆进 CSS Grid，使各路径独占列、笔直对齐，融合点（cat/+）居中覆盖其上游列。
    * ``res`` —— 分辨率级（0 起）：输出空间尺寸相对模型输入缩小 2^res 倍。由追踪
      形状计算，renderer 据此把主干按分辨率级右缩进，呈现 encoder↓/decoder↑ 的
      U 型结构；分辨率不变的流程（数据流等）恒为 0，布局自动退化为直列。
    """

    id: str
    label: str
    kind: str = "op"
    key_info: Dict[str, str] = field(default_factory=dict)
    detail: Dict[str, str] = field(default_factory=dict)
    parent_id: Optional[str] = None
    collapsed: bool = True
    rank: int = 0
    col: int = 0
    colspan: int = 1
    res: int = 0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# 连边语义类别 —— renderer 据此选线型 / 颜色。
#   forward  —— 主数据流（实线箭头，自上而下）。
#   residual —— 残差捷径 ``out + shortcut(x)``（弧线侧引，标注 ``+``）。
#   skip     —— encoder→decoder 跳连 / 长程上下文注入（虚线侧引）。
EDGE_KINDS = ("forward", "residual", "skip")


@dataclass
class VisEdge:
    """有向连边：``src → dst``，``label`` 可选（如标注张量形状 / 分支条件）。

    * ``kind`` —— 连边语义类别（见 ``EDGE_KINDS``），决定线型 / 颜色。
    """

    src: str
    dst: str
    label: str = ""
    kind: str = "forward"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class VisGraph:
    """单视图的图：节点 + 连边 + 顶部摘要元信息。"""

    title: str
    nodes: List[VisNode] = field(default_factory=list)
    edges: List[VisEdge] = field(default_factory=list)
    # 顶栏摘要（yaml 路径 / pipeline 名 / topology 关键量等），有序展示。
    meta: Dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Builder-facing helpers
    # ------------------------------------------------------------------
    def add_node(
        self,
        node_id: str,
        label: str,
        kind: str = "op",
        key_info: Optional[Dict[str, object]] = None,
        detail: Optional[Dict[str, object]] = None,
        parent_id: Optional[str] = None,
        collapsed: bool = True,
        rank: int = 0,
        col: int = 0,
        colspan: int = 1,
        res: int = 0,
    ) -> VisNode:
        """新增节点并返回（键值统一转字符串，避免 JSON 序列化歧义）。"""
        node = VisNode(
            id=node_id,
            label=label,
            kind=kind,
            key_info=_stringify(key_info),
            detail=_stringify(detail),
            parent_id=parent_id,
            collapsed=collapsed,
            rank=int(rank),
            col=int(col),
            colspan=int(colspan),
            res=int(res),
        )
        self.nodes.append(node)
        return node

    def add_edge(
        self, src: str, dst: str, label: object = "", kind: str = "forward",
    ) -> VisEdge:
        edge = VisEdge(
            src=src, dst=dst,
            label="" if label is None else str(label),
            kind=kind)
        self.edges.append(edge)
        return edge

    def to_dict(self) -> Dict[str, object]:
        return {
            "title": self.title,
            "meta": {str(k): str(v) for k, v in self.meta.items()},
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }


def assign_grid_layout(g: "VisGraph", *, assign_ranks: bool = False) -> None:
    """为图中每个父容器内的兄弟节点计算 ``(rank, col, colspan)``，供 renderer 摆进
    CSS Grid。纯靠图结构（forward 边血缘），与具体网络/流程无关，可被任一 builder 复用。

    * ``assign_ranks=True``：先按 forward 边做**最长路径分层**写入 ``rank``（线性流程
      因此自上而下逐级排开，不再因缺省 rank 全堆在同一格）；``residual`` 边不计入层级。
      模型流自带更复杂的 rank/skip 计算，故以 ``False`` 调用、只补列位。
    * 列位：逐 rank 自上而下，按已定稿的 forward 父定列——无父则占新列；有父则
      ``col=min(父.col)``、``colspan`` 覆盖父列区间（单父继承使主链笔直，多父跨列居中）；
      同 rank 内按 ``(col, 插入序)`` 从左到右去重叠（右移）。
    """
    by_parent: Dict[Optional[str], List[VisNode]] = {}
    for n in g.nodes:
        by_parent.setdefault(n.parent_id, []).append(n)

    if assign_ranks:
        node_rank: Dict[str, int] = {}
        for kids in by_parent.values():
            ids = [n.id for n in kids]
            idset = set(ids)
            succ: Dict[str, List[str]] = {i: [] for i in ids}
            indeg: Dict[str, int] = {i: 0 for i in ids}
            for e in g.edges:
                if e.kind == "residual":
                    continue
                if e.src in idset and e.dst in idset:
                    succ[e.src].append(e.dst)
                    indeg[e.dst] += 1
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
        for n in g.nodes:
            n.rank = node_rank.get(n.id, 0)

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
                    col[nid] = 0
                    span[nid] = 1
                else:
                    lo = min(col[p] for p in ps)
                    hi = max(col[p] + span[p] for p in ps)
                    col[nid] = lo
                    span[nid] = hi - lo
            cursor: Optional[int] = None
            for nid in sorted(row, key=lambda x: (col[x], order_idx[x])):
                if cursor is not None and col[nid] < cursor:
                    col[nid] = cursor
                cursor = col[nid] + span[nid]
        for n in kids:
            n.col = col[n.id]
            n.colspan = span[n.id]


def shape_str(shape) -> str:
    """张量形状 → 紧凑字符串 ``(B, C, D, H, W)``；``None`` → ``"?"``。"""
    if shape is None:
        return "?"
    try:
        return "(" + ", ".join(str(int(s)) for s in shape) + ")"
    except TypeError:
        return str(shape)


def _stringify(d: Optional[Dict[str, object]]) -> Dict[str, str]:
    """把任意 dict 的键值转成字符串，保持插入顺序。"""
    if not d:
        return {}
    return {str(k): _fmt_value(v) for k, v in d.items()}


def _fmt_value(v: object) -> str:
    if isinstance(v, float):
        # 紧凑浮点：去除多余尾零，但保留有效信息。
        return f"{v:g}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt_value(x) for x in v) + "]"
    return str(v)


__all__ = ["VisNode", "VisEdge", "VisGraph", "NODE_KINDS", "EDGE_KINDS",
           "shape_str", "assign_grid_layout"]
