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
    """

    id: str
    label: str
    kind: str = "op"
    key_info: Dict[str, str] = field(default_factory=dict)
    detail: Dict[str, str] = field(default_factory=dict)
    parent_id: Optional[str] = None
    collapsed: bool = True

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class VisEdge:
    """有向连边：``src → dst``，``label`` 可选（如标注张量形状 / 分支条件）。"""

    src: str
    dst: str
    label: str = ""

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
        )
        self.nodes.append(node)
        return node

    def add_edge(self, src: str, dst: str, label: object = "") -> VisEdge:
        edge = VisEdge(src=src, dst=dst, label="" if label is None else str(label))
        self.edges.append(edge)
        return edge

    def to_dict(self) -> Dict[str, object]:
        return {
            "title": self.title,
            "meta": {str(k): str(v) for k, v in self.meta.items()},
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }


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


__all__ = ["VisNode", "VisEdge", "VisGraph", "NODE_KINDS", "shape_str"]
