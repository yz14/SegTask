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
from segtask_v1.visualization.model_flow import build_model_flow

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
