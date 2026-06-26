"""Unit tests for ``segtask_v1.visualization.model_flow`` 叶子级连边重建。

回归 TODO#3：MultiRF 等含**并联分支 / functional 融合(torch.cat) / 残差捷径**的
block，其叶子级连边曾因「只认直接产出叶子(in_src)、不认 functional 算子血缘」而断链
——并联分支与 shortcut 全部堆到 rank0（“5 个 conv 并排一行”）。这里校验逐输入张量
重建后的真实数据流：分支→fuse 扇入、主路成链、shortcut 残差汇入、且框内叶子全连通。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from segtask_v1.config import load_config
from segtask_v1.models.factory import build_model
from segtask_v1.visualization import model_flow as model_flow_mod
from segtask_v1.visualization.data_flow import build_data_flow
from segtask_v1.visualization.model_flow import build_model_flow
from segtask_v1.visualization.predict_flow import build_predict_flow

_CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def _build(cfg_name: str):
    cfg = load_config(str(_CONFIGS / cfg_name))
    model = build_model(cfg)
    return build_model_flow(cfg, model, trace_shapes=True)


def _children(g, parent_id):
    return [n for n in g.nodes if n.parent_id == parent_id]


def _intra_edges(g, ids):
    ids = set(ids)
    return [e for e in g.edges if e.src in ids and e.dst in ids]


def _num_components(ids, edges):
    """无向连通分量数（union-find）。"""
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in edges:
        parent[find(e.src)] = find(e.dst)
    return len({find(i) for i in ids})


def _leaf_groups(g):
    """所有「子节点全为叶子且≥2」的同级组 → {parent_id: [leaf_ids]}。"""
    by_parent = {}
    for n in g.nodes:
        by_parent.setdefault(n.parent_id, []).append(n)
    groups = {}
    for pid, kids in by_parent.items():
        if not pid:
            continue
        leaf_kids = [k.id for k in kids if k.id.startswith("leaf::")]
        if len(leaf_kids) == len(kids) and len(leaf_kids) >= 2:
            groups[pid] = leaf_kids
    return groups


# ---------------------------------------------------------------------------
# MultiRF block：并联分支 → cat 融合 → 主路链 + 残差捷径
# ---------------------------------------------------------------------------
def test_multirf_block_dataflow():
    g = _build("seg2_5d.yaml")
    blocks = [n for n in g.nodes
              if n.kind == "stage" and n.key_info.get("type") == "MultiRFBlock"]
    assert blocks, "seg2_5d 应含 MultiRFBlock"
    block = blocks[0]

    leaves = _children(g, block.id)
    leaf_ids = [n.id for n in leaves]
    edges = _intra_edges(g, leaf_ids)

    # (1) 框内叶子全连通：曾经分支/shortcut 断链 → 多分量。
    assert _num_components(leaf_ids, edges) == 1, "MultiRF block 叶子不应断链"

    # (2) 三条并联分支扇入同一**显式 merge 节点**(cat)，再由其汇入融合叶 fuse
    #     ——步骤 D：split→merge 的汇流点以独立算子节点呈现，而非分支直连 fuse。
    branch_ids = [n.id for n in leaves if ".branches." in n.id]
    assert len(branch_ids) >= 2, "MultiRF 应有多条并联分支"
    merge_targets = {e.dst for e in edges if e.src in set(branch_ids)}
    assert len(merge_targets) == 1, "所有分支应汇入同一 merge 节点"
    merge_id = next(iter(merge_targets))
    merge_node = next(n for n in leaves if n.id == merge_id)
    assert merge_node.kind == "merge" and merge_node.label == "cat"
    fuse_targets = {e.dst for e in edges if e.src == merge_id}
    assert any(".fuse" in t for t in fuse_targets), "merge 节点下游应为 fuse 叶"

    # (3) 残差捷径以 residual 边汇入主路（标注 +），而非伪造的前向跳连。
    sc_resid = [e for e in edges
                if ".shortcut" in e.src and e.kind == "residual"]
    assert sc_resid, "shortcut 应以 residual 边汇入主路"

    # (4) 分层后并联分支同 rank、主路成纵向链（不再全堆 rank0）。
    rank = {n.id: n.rank for n in leaves}
    assert len({rank[b] for b in branch_ids}) == 1, "并联分支应同 rank"
    assert max(rank.values()) >= 4, "主路应展开为多级链，而非全堆一行"


# ---------------------------------------------------------------------------
# 旧 PyTorch (<2.0) 兼容：register_forward_pre_hook 无 ``with_kwargs`` 参数时，
# 应回退到不带该参数的注册而非整体追踪失败退化为纯结构图。强制 fallback 路径后，
# 张量数据流追踪仍须成功（以 cat 融合产出的 merge 节点存在为证）。
# ---------------------------------------------------------------------------
def test_pre_hook_kwargs_fallback_still_traces(monkeypatch):
    monkeypatch.setattr(model_flow_mod, "_PRE_HOOK_SUPPORTS_KWARGS", False)
    g = _build("seg2_5d.yaml")
    merges = [n for n in g.nodes if n.kind == "merge"]
    assert merges, "无 with_kwargs 的回退注册下，数据流追踪仍应成功（merge 节点应在）"


# ---------------------------------------------------------------------------
# 顶层容器里**未成 block 的散叶**（如 Downsample 的 op→norm）也要连边。
# ---------------------------------------------------------------------------
def test_non_block_leaf_group_connected():
    g = _build("seg2_5d.yaml")
    ds_groups = {pid: ids for pid, ids in _leaf_groups(g).items()
                 if "downsample" in pid}
    assert ds_groups, "应存在 downsample 叶子组"
    for pid, ids in ds_groups.items():
        edges = _intra_edges(g, ids)
        assert _num_components(ids, edges) == 1, f"{pid} 的 op→norm 应连边"


# ---------------------------------------------------------------------------
# 通用性：单模型构图下，任一「全叶子同级组」都应连通（无孤立叶子）。
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cfg_name", ["seg2_5d.yaml", "lungves_multirf.yaml",
                                      "lungves_selfattn.yaml"])
def test_no_disconnected_leaf_groups(cfg_name):
    g = _build(cfg_name)
    disconnected = []
    for pid, ids in _leaf_groups(g).items():
        if _num_components(ids, _intra_edges(g, ids)) > 1:
            disconnected.append(pid)
    assert not disconnected, f"{cfg_name} 存在断链叶子组: {disconnected}"


# ---------------------------------------------------------------------------
# 步骤 D：cat / + 等融合点应呈现为显式 merge 算子节点。
# ---------------------------------------------------------------------------
def test_merge_nodes_at_fusion_points():
    g = _build("seg2_5d.yaml")
    merges = [n for n in g.nodes if n.kind == "merge"]
    assert merges, "seg2_5d（含 MultiRF cat 融合 + 解码 skip cat + 残差 +）应有 merge 节点"

    by_id = {n.id: n for n in g.nodes}
    in_edges, out_edges = {}, {}
    for e in g.edges:
        out_edges.setdefault(e.src, []).append(e)
        in_edges.setdefault(e.dst, []).append(e)

    for m in merges:
        # merge 标签为归一化算子符号；融合点须真正汇聚 ≥2 个上游、并有下游消费者。
        assert m.label in {"cat", "+", "−", "×", "max", "min", "merge"}
        ins = in_edges.get(m.id, [])
        outs = out_edges.get(m.id, [])
        assert len(ins) >= 2, f"merge {m.id} 应至少汇聚两路输入"
        assert len(outs) >= 1, f"merge {m.id} 应有下游消费者"
        # merge 节点与其上下游同处一个父框（局部汇流点，不跨框）。
        for e in ins + outs:
            assert by_id[e.src].parent_id == m.parent_id
            assert by_id[e.dst].parent_id == m.parent_id


# ---------------------------------------------------------------------------
# 列对齐网格布局：(col, colspan) 让并联路径各占一列、串联主链笔直、融合点跨列居中。
# ---------------------------------------------------------------------------
def _block_by_suffix(g, suffix):
    blocks = [n for n in g.nodes if n.id.endswith(suffix)]
    assert blocks, f"seg2_5d 应含 {suffix}"
    return blocks[0]


def test_block_column_layout():
    g = _build("seg2_5d.yaml")

    # (1) stage1.block0(ResNetBlock)：shortcut 独占左列，主路在右列且笔直一列。
    blk1 = _block_by_suffix(g, "encoder.stages.1.blocks.0")
    leaves1 = _children(g, blk1.id)
    sc1 = [n for n in leaves1 if ".shortcut" in n.id]
    main1 = [n for n in leaves1 if ".shortcut" not in n.id]
    assert sc1 and main1
    # shortcut 整列严格在主路左侧（列区间不相交）。
    assert max(n.col + n.colspan for n in sc1) <= min(n.col for n in main1), \
        "shortcut 应独占主路左侧的列"
    # 主路（非分支结构）同处一列、笔直：col 全相同、colspan==1。
    assert len({n.col for n in main1}) == 1 and all(
        n.colspan == 1 for n in main1), "stage1 主路应为笔直单列"

    # (2) stage2.block0(MultiRFBlock)：三分支同 rank 占相邻三列；其后融合·主路
    #     以 colspan>1 跨列居中覆盖分支列区间（即“单列后续居中对齐于分支组”）。
    blk2 = _block_by_suffix(g, "encoder.stages.2.blocks.0")
    leaves2 = _children(g, blk2.id)
    branches = [n for n in leaves2 if ".branches." in n.id]
    assert len(branches) >= 3, "MultiRF 应有≥3 条并联分支"
    assert len({n.rank for n in branches}) == 1, "分支应同 rank"
    assert len({n.col for n in branches}) == len(branches), "分支应各占独立列"
    b_lo = min(n.col for n in branches)
    b_hi = max(n.col + n.colspan for n in branches)
    fuse = next(n for n in leaves2 if ".fuse" in n.id)
    # fuse 跨列覆盖分支列区间（居中于三分支之下）。
    assert fuse.col == b_lo and fuse.col + fuse.colspan == b_hi, \
        "fuse 应跨列居中覆盖分支组"

    # (3) 通用性：任一父容器内、任一 rank 上，节点列区间互不重叠（清晰分列）。
    by_parent = {}
    for n in g.nodes:
        by_parent.setdefault(n.parent_id, []).append(n)
    for kids in by_parent.values():
        by_rank = {}
        for n in kids:
            by_rank.setdefault(n.rank, []).append(n)
        for nodes in by_rank.values():
            iv = sorted((n.col, n.col + n.colspan, n.id) for n in nodes)
            for a, b in zip(iv, iv[1:]):
                assert a[1] <= b[0], f"同 rank 列重叠: {a[2]} 与 {b[2]}"


# ---------------------------------------------------------------------------
# 数据流 / 预测流：线性链各节点应分到唯一 (rank, col)，不再全堆同一格。
# （回归：列布局改造后这两条线性流未调用 assign_grid_layout，缺省 rank=0/col=0
#  导致所有框堆叠在一起。）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cfg_name", ["seg2_5d.yaml", "lungves_multirf.yaml",
                                      "seg3d.yaml"])
def test_linear_flows_no_stacking(cfg_name):
    cfg = load_config(str(_CONFIGS / cfg_name))
    for builder in (build_data_flow, build_predict_flow):
        g = builder(cfg)
        cells = [(n.rank, n.col) for n in g.nodes]
        assert len(cells) == len(set(cells)), \
            f"{cfg_name} {builder.__name__}: 节点堆在同一 (rank,col) 格"
        # 线性链：rank 逐级递增、跨越所有节点（不止 rank0）。
        assert max(r for r, _ in cells) == len(g.nodes) - 1, \
            f"{cfg_name} {builder.__name__}: 应自上而下逐级排开"


# ---------------------------------------------------------------------------
# head 容器内「block 框 + 其后散叶分类器」应串成链（conv → classifier），
# 而非因无连边默认同 rank 并排堆叠。
# （回归：DS Head 内 ConvNormAct block 与 loose Conv2d classifier 并列。）
# ---------------------------------------------------------------------------
def test_head_block_then_loose_leaf_chained():
    g = _build("seg2_5d.yaml")
    by_parent = {}
    for n in g.nodes:
        by_parent.setdefault(n.parent_id, []).append(n)
    checked = 0
    for pid, kids in by_parent.items():
        if not pid or "ds_head" not in pid.lower() and "head" not in pid.lower():
            continue
        block_kids = [k for k in kids if k.kind == "stage"]
        loose_kids = [k for k in kids if k.id.startswith("leaf::")]
        if not (block_kids and loose_kids):
            continue
        checked += 1
        ids = {k.id for k in kids}
        edges = _intra_edges(g, ids)
        assert _num_components([k.id for k in kids], edges) == 1, \
            f"{pid}: block 与散叶分类器应连边成链"
        # block 与散叶不应同 rank（应是先后关系，纵向错开）。
        assert {k.rank for k in block_kids}.isdisjoint({k.rank for k in loose_kids}), \
            f"{pid}: block 与散叶分类器不应同 rank 并排"
    assert checked, "应存在含 block + 散叶分类器的 head 容器"
