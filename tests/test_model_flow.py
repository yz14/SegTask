"""Unit tests for ``segtask_v1.visualization.model_flow``（torchlens 适配层）。

TODO#4 步骤 2 验收：模型流图改由 torchlens op 级 DAG 收缩生成——
叶子模块=单节点、functional 融合(cat/+/lerp…)=显式 merge 节点、
残差/跳连由通用 DAG 规则（可达性/深度不对称/旁路检测）标注，
不再依赖命名白名单、残差类型清单或属性名特判。

覆盖四类拓扑：unet(+selfattn+multirf) / unetpp / unet3p / 扩散 backbone(edm2)。
断言各配置的跳连数、残差标注与 stage in/out 形状与代码实际一致。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from segtask_v1.config import load_config
from segtask_v1.losses.losses import build_loss
from segtask_v1.models.factory import build_model
from segtask_v1.models.topology import build_topology
from segtask_v1.trainer.pipelines.factory import build_pipeline
from segtask_v1.visualization.data_flow import build_data_flow
from segtask_v1.visualization.model_flow import build_model_flow
from segtask_v1.visualization.predict_flow import build_predict_flow

_CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def _load(cfg_name: str, **overrides):
    cfg = load_config(str(_CONFIGS / cfg_name))
    for k, v in overrides.items():
        setattr(cfg.model, k, v)
    # 大 patch 追踪省内存（不影响图拓扑，仅缩小空间尺寸；层深维原样保留）。
    ps = list(cfg.data.patch_size)
    if any(p > 64 for p in ps):
        ps[-2:] = [max(32, p // 8) for p in ps[-2:]]
        if len(ps) == 3 and ps[0] > 16:
            ps[0] = max(8, ps[0] // 4)
        cfg.data.patch_size = ps
    return cfg


def _build(cfg_name: str, **overrides):
    cfg = _load(cfg_name, **overrides)
    model = build_model(cfg)
    return build_model_flow(cfg, model, trace_shapes=True)


def _model_input_shape(cfg):
    topo = build_topology(cfg)
    model = build_model(cfg)
    g = build_model_flow(cfg, model, trace_shapes=False)
    node = next(n for n in g.nodes if n.id == "input")
    shape = ast.literal_eval(node.key_info["shape"])
    return topo, shape


def _by_id(g):
    return {n.id: n for n in g.nodes}


def _descendants(g, root_id):
    """root 容器下（含嵌套）的全部节点 id。"""
    kids = {}
    for n in g.nodes:
        kids.setdefault(n.parent_id, []).append(n.id)
    out, stack = set(), [root_id]
    while stack:
        cur = stack.pop()
        for k in kids.get(cur, []):
            out.add(k)
            stack.append(k)
    return out


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
        if e.src in parent and e.dst in parent:
            parent[find(e.src)] = find(e.dst)
    return len({find(i) for i in ids})


def _materialized(g, ids):
    """ids 中真正参与连边的端点（叶子/merge/input），排除容器框。"""
    return [i for i in ids if i.startswith(("leaf::", "merge::")) or i == "input"]


# ---------------------------------------------------------------------------
# MultiRF block：并联分支 → cat 融合 → 主路链 + 残差捷径（seg2_5d）
# ---------------------------------------------------------------------------
def test_multirf_block_dataflow():
    g = _build("seg2_5d.yaml")
    blocks = [n for n in g.nodes
              if n.kind == "stage" and n.key_info.get("type") == "MultiRFBlock"]
    assert blocks, "seg2_5d 应含 MultiRFBlock"
    block = blocks[0]

    desc = _descendants(g, block.id)
    ends = _materialized(g, desc)
    edges = _intra_edges(g, ends)

    # (1) 框内数据流全连通：分支/shortcut 不断链。
    assert _num_components(ends, edges) == 1, "MultiRF block 内部不应断链"

    # (2) 并联分支扇入同一显式 cat merge 节点，其下游为 fuse 叶。
    branch_ids = {i for i in ends if ".branches." in i}
    assert len(branch_ids) >= 2, "MultiRF 应有多条并联分支"
    merge_targets = {e.dst for e in edges if e.src in branch_ids}
    # 分支后可有 branch_post 卷积，最终仍须汇入唯一 cat。
    cats = [i for i in ends
            if i.startswith("merge::") and _by_id(g)[i].label == "cat"]
    assert len(cats) == 1, "MultiRF 分支应汇入唯一 cat merge 节点"
    fuse_targets = {e.dst for e in edges if e.src == cats[0]}
    assert any(".fuse" in t for t in fuse_targets), "cat 下游应为 fuse 叶"

    # (3) shortcut 子树以 residual 边汇入加法 merge（通用规则识别，无类型清单）。
    resid = [e for e in edges if e.kind == "residual"]
    assert any(".shortcut" in e.src for e in resid), \
        "shortcut 应以 residual 边汇入主路"


# ---------------------------------------------------------------------------
# ResNet block：残差捷径由 DAG 规则标注（不依赖 _SHORTCUT_ATTRS）。
# ---------------------------------------------------------------------------
def test_resnet_block_residual():
    g = _build("seg2_5d.yaml")
    blocks = [n for n in g.nodes
              if n.kind == "stage" and n.key_info.get("type") == "ResNetBlock"]
    assert blocks, "seg2_5d 应含 ResNetBlock"
    for blk in blocks:
        ends = set(_materialized(g, _descendants(g, blk.id)))
        # 恒等捷径的残差源是块输入（位于块外），故只要求 dst 在块内。
        assert any(e.kind == "residual" and e.dst in ends for e in g.edges), \
            f"{blk.id} 应含 residual 边"


# ---------------------------------------------------------------------------
# 注意力块：整块为叶子/子叶链，内部不得出现虚假 merge（einsum/sdpa 的 q,k 同源
# 曾被误判为融合点）；两条残差（attn 残差 + FFN 残差）都应标出。
# ---------------------------------------------------------------------------
def _attn_cfg(cfg):
    n = len(cfg.model.encoder_channels)
    cfg.model.selfattn_enabled = True
    cfg.model.selfattn_ffn = True
    cfg.model.selfattn_encoder_stages = [0] * (n - 1) + ["softmax"]
    cfg.model.selfattn_decoder_stages = [0] * (n - 2) + ["linear"]
    return cfg


def test_selfattn_no_spurious_merge_and_residuals():
    cfg = _attn_cfg(_load("seg2_5d.yaml"))
    model = build_model(cfg)
    g = build_model_flow(cfg, model, trace_shapes=True)

    attn_leaves = [n for n in g.nodes
                   if n.id.startswith("leaf::") and n.id.endswith(".attn")]
    assert len(attn_leaves) == 2, "encoder/decoder 各应有 1 个注意力叶节点"

    by_id = _by_id(g)
    for leaf in attn_leaves:
        blk = leaf.parent_id  # SelfAttentionBlock 容器
        ends = _materialized(g, _descendants(g, blk))
        edges = _intra_edges(g, ends)
        merges = [i for i in ends if i.startswith("merge::")]
        # 唯一合法融合点是两个残差加法（x+h 与 x+ffn）；无 einsum/sdpa 虚假 merge。
        assert all(by_id[m].label == "+" for m in merges), \
            f"{blk} 内出现非加法 merge（虚假融合点）: {merges}"
        assert len(merges) == 2, f"{blk} 应恰有 2 个残差加法 merge"
        # 第一条残差 (x+h) 的源是块输入（位于块外），故按 dst 归属统计。
        ends_set = set(ends)
        resid = [e for e in g.edges
                 if e.kind == "residual" and e.dst in ends_set]
        assert len(resid) == 2, f"{blk} 的两条残差捷径都应标注"


# ---------------------------------------------------------------------------
# 跳连数与拓扑一致（通用旁路检测，不依赖 unetpp/unet3p 特判）。
# ---------------------------------------------------------------------------
def _skip_edges(g):
    return [e for e in g.edges if e.kind == "skip"]


def test_unet_skip_count_matches_topology():
    g = _build("seg2_5d.yaml")
    cfg = _load("seg2_5d.yaml")
    # 对称 UNet：每个解码 level 一条 encoder→decoder 跳连。
    expected = len(cfg.model.encoder_channels) - 1
    skips = _skip_edges(g)
    assert len(skips) == expected, \
        f"UNet 应有 {expected} 条跳连，实得 {len(skips)}"
    # 全部由 encoder 侧发出、汇入 cat 融合点。
    assert all(e.src.startswith("leaf::encoder.") for e in skips)
    assert all(e.dst.startswith("merge::") for e in skips)


def test_unetpp_skip_count_matches_topology():
    g = _build("seg2_5d.yaml", decoder_type="unetpp")
    cfg = _load("seg2_5d.yaml")
    L = len(cfg.model.encoder_channels) - 1  # 嵌套深度
    # UNet++ 节点 X_{i,j}（j≥1）的 cat 汇入 j 条同层稠密前驱 X_{i,0..j-1}
    # + 1 条上采样主干边；稠密前驱均为旁路跳连：sum_{j=1..L} j·(L+1−j)。
    expected = sum(j * (L + 1 - j) for j in range(1, L + 1))
    assert len(_skip_edges(g)) == expected, \
        f"UNet++ 应有 {expected} 条跳连，实得 {len(_skip_edges(g))}"


def test_unet3p_skip_count_matches_topology():
    g = _build("seg2_5d.yaml", decoder_type="unet3p")
    cfg = _load("seg2_5d.yaml")
    L = len(cfg.model.encoder_channels) - 1
    # UNet3+ 每个解码 level 聚合全尺度 L+1 路输入，其中 2 路为相邻主干边
    # （同级 encoder、下一级 decoder/瓶颈），其余 L−1 路为跳连：L·(L−1)。
    expected = L * (L - 1)
    skips = _skip_edges(g)
    assert len(skips) == expected, \
        f"UNet3+ 应有 {expected} 条跳连，实得 {len(skips)}"


# ---------------------------------------------------------------------------
# 扩散 backbone（EDM2）：lerp 幅度保持加法应识别为加性融合并标注残差。
# ---------------------------------------------------------------------------
def test_edm2_residuals_and_no_unknown_merges():
    cfg = _load("seg2_5d_edm2.yaml")
    try:
        model = build_model(cfg)
    except TypeError as e:
        pytest.skip(f"EDM2 builder 现存 bug（与可视化无关）: {e}")
    g = build_model_flow(cfg, model, trace_shapes=True)
    merges = [n for n in g.nodes if n.kind == "merge"]
    assert merges
    assert all(n.label != "merge" for n in merges), \
        "EDM2 不应出现未识别符号的 merge（lerp 应已登记）"
    assert any(e.kind == "residual" for e in g.edges), \
        "EDM2 res block 的 lerp 残差应标注"
    assert _skip_edges(g), "EDM2 UNet 的 encoder→decoder 跳连应标注"


# ---------------------------------------------------------------------------
# merge 节点健康性：≥2 路输入、≥1 个下游消费者、符号已归一化。
# ---------------------------------------------------------------------------
def test_merge_nodes_at_fusion_points():
    g = _build("seg2_5d.yaml")
    merges = [n for n in g.nodes if n.kind == "merge"]
    assert merges, "seg2_5d（MultiRF cat + 解码 skip cat + 残差 +）应有 merge 节点"

    in_edges, out_edges = {}, {}
    for e in g.edges:
        out_edges.setdefault(e.src, []).append(e)
        in_edges.setdefault(e.dst, []).append(e)

    for m in merges:
        assert m.label in {"cat", "+", "−", "×", "max", "min", "lerp", "merge"}
        assert len(in_edges.get(m.id, [])) >= 2, f"merge {m.id} 应汇聚≥2 路输入"
        assert len(out_edges.get(m.id, [])) >= 1, f"merge {m.id} 应有下游消费者"


# ---------------------------------------------------------------------------
# 容器 in/out 形状与拓扑声明一致（编码器逐 stage 通道数）。
# ---------------------------------------------------------------------------
def test_encoder_stage_shapes_match_topology():
    cfg = _load("seg2_5d.yaml")
    model = build_model(cfg)
    g = build_model_flow(cfg, model, trace_shapes=True)
    chans = cfg.model.encoder_channels
    for i, ch in enumerate(chans):
        node = next(n for n in g.nodes if n.id == f"encoder.stages.{i}")
        out_shape = ast.literal_eval(node.key_info["out"])
        assert out_shape[1] == ch, \
            f"stages.{i} 输出通道应为 {ch}，实得 {out_shape[1]}"


# ---------------------------------------------------------------------------
# 任一容器内数据流连通（无孤立节点）——泛化到多配置。
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cfg_name", ["seg2_5d.yaml", "lungves_multirf.yaml",
                                      "lungves_selfattn.yaml"])
def test_no_disconnected_subtrees(cfg_name):
    g = _build(cfg_name)
    disconnected = []
    for n in g.nodes:
        if n.kind != "stage":
            continue
        ends = _materialized(g, _descendants(g, n.id))
        if len(ends) < 2:
            continue
        # 容器内部子图允许经由容器外节点连通（如跨框 merge），故只检全图可达性。
        edges = _intra_edges(g, ends)
        if _num_components(ends, edges) > 1:
            # 端点若有跨容器边则不算孤立。
            ext = {e.src for e in g.edges} | {e.dst for e in g.edges}
            isolated = [i for i in ends if i not in ext]
            if isolated:
                disconnected.append((n.id, isolated))
    assert not disconnected, f"{cfg_name} 存在无任何连边的孤立节点: {disconnected}"


# ---------------------------------------------------------------------------
# 同 rank 列区间互不重叠（网格布局不变量，任意配置通用）。
# ---------------------------------------------------------------------------
def test_grid_no_column_overlap():
    g = _build("seg2_5d.yaml")
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
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cfg_name", ["seg2_5d.yaml", "lungves_multirf.yaml",
                                      "seg3d.yaml"])
def test_linear_flows_no_stacking(cfg_name):
    cfg = _load(cfg_name)
    for builder in (build_data_flow, build_predict_flow):
        g = builder(cfg)
        cells = [(n.rank, n.col) for n in g.nodes]
        assert len(cells) == len(set(cells)), \
            f"{cfg_name} {builder.__name__}: 节点堆在同一 (rank,col) 格"
        assert max(r for r, _ in cells) == len(g.nodes) - 1, \
            f"{cfg_name} {builder.__name__}: 应自上而下逐级排开"


@pytest.mark.parametrize(
    "cfg_name, expected_spatial",
    [
        ("seg3d.yaml", (16, 128, 128)),
        ("seg2_5d.yaml", (256, 256)),
        ("seg2_5d_planA.yaml", (16, 256, 256)),
    ],
)
def test_model_input_spatial_matches_patch_size(cfg_name, expected_spatial):
    cfg = load_config(str(_CONFIGS / cfg_name))
    topo, shape = _model_input_shape(cfg)
    assert shape[0] == cfg.data.batch_size
    assert shape[1] == topo.in_channels
    assert shape[2:] == expected_spatial


def test_model_input_shape_matches_real_pipeline_feed_seg3d():
    cfg = load_config(str(_CONFIGS / "seg3d.yaml"))
    topo, shape = _model_input_shape(cfg)
    pipe = build_pipeline(cfg, build_loss(cfg.loss))
    cube = torch.zeros(
        cfg.data.batch_size, 1, *pipe.target_patch_size, dtype=torch.float32)
    lab = torch.zeros_like(cube)
    img, _ = pipe.prepare_val_batch(cube, lab)
    assert shape == tuple(img.shape)
    assert shape[1] == topo.in_channels


def test_model_input_spatial_cubic_keep_native_multi_res_matches_patch_size():
    cfg = load_config(str(_CONFIGS / "seg3d.yaml"))
    cfg.data.patch_mode = "cubic"
    cfg.sync()
    cfg.validate()
    topo, shape = _model_input_shape(cfg)
    assert topo.spatial_dims == 3
    assert shape[2:] == tuple(cfg.data.patch_size)
