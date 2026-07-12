"""S9 回归测试：Frangi spacing-aware、multicrop 体积一致 scale、checkpoint 原子写+指纹。

覆盖 S9 阶段对 ssltask 的行为修正：
- ``frangi_vesselness`` 的可选 ``spacing``（物理尺度解释 + 各向异性高斯/Hessian）；
  ``spacing=None`` 与 ``spacing=[1,1,1]`` 等价（旧行为不回归）。
- ``_sample_box`` 改为体积一致采样（scale=体积占比，各向同性边长）。
- ``SSLTrainer._atomic_save`` / ``_state_fingerprint``：原子落盘 + 内容指纹校验。
"""

from __future__ import annotations

import torch

import torch.nn.functional as F

from ssltask.data.multicrop import (
    MultiCropGenerator, _affine_grid, _sample_box)
from ssltask.data.vesselness import frangi_vesselness
from ssltask.trainer.ssl_trainer import SSLTrainer


# ---------------------------------------------------------------------------
# Frangi spacing-aware
# ---------------------------------------------------------------------------
def _tube_vol() -> torch.Tensor:
    vol = torch.zeros(1, 1, 24, 32, 32)
    vol[0, 0, :, 16, 16] = 1.0
    vol[0, 0, :, 16, 17] = 1.0
    vol[0, 0, :, 17, 16] = 1.0
    return vol


def test_frangi_spacing_none_equals_isotropic_one():
    """spacing=None 与 spacing=[1,1,1] 数值一致（旧行为不回归）。"""
    vol = _tube_vol()
    a = frangi_vesselness(vol, scales=[1.0, 2.0], spatial_dims=3)
    b = frangi_vesselness(vol, scales=[1.0, 2.0], spatial_dims=3,
                          spacing=[1.0, 1.0, 1.0])
    assert torch.allclose(a, b, atol=1e-6)


def test_frangi_anisotropic_spacing_changes_output():
    """各向异性 spacing 改变响应（物理尺度不同 → 逐轴体素 sigma 不同）。"""
    vol = _tube_vol()
    iso = frangi_vesselness(vol, scales=[1.0, 2.0], spatial_dims=3,
                            spacing=[1.0, 1.0, 1.0])
    aniso = frangi_vesselness(vol, scales=[1.0, 2.0], spatial_dims=3,
                              spacing=[3.0, 0.7, 0.7])
    assert aniso.shape == vol.shape
    assert 0.0 <= float(aniso.min()) and float(aniso.max()) <= 1.0 + 1e-5
    assert not torch.allclose(iso, aniso, atol=1e-3)


def test_frangi_spacing_bad_length_raises():
    vol = _tube_vol()
    try:
        frangi_vesselness(vol, scales=[1.0], spatial_dims=3, spacing=[1.0, 1.0])
    except ValueError:
        return
    raise AssertionError("expected ValueError for wrong spacing length")


def test_frangi_spacing_nonpositive_raises():
    vol = _tube_vol()
    try:
        frangi_vesselness(vol, scales=[1.0], spatial_dims=3,
                          spacing=[1.0, 0.0, 1.0])
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-positive spacing")


# ---------------------------------------------------------------------------
# multicrop 体积一致 scale
# ---------------------------------------------------------------------------
def test_sample_box_volume_consistent_3d():
    """裁剪体积占比 ≈ 采样的体积占比 f∈[lo,hi]，且各轴边长各向同性。"""
    import random
    random.seed(0)
    spatial = (64, 64, 64)
    lo, hi = 0.5, 1.0
    fracs = []
    edge_ratios = []
    for _ in range(400):
        origins, sizes = _sample_box(spatial, lo, hi)
        assert len(sizes) == 3
        for o, s, d in zip(origins, sizes, spatial):
            assert 0 <= o and o + s <= d and s >= 1
        vol = 1.0
        for s, d in zip(sizes, spatial):
            vol *= s / d
        fracs.append(vol)
        e = [s / d for s, d in zip(sizes, spatial)]
        edge_ratios.append(max(e) - min(e))
    # 体积占比落在 [lo, hi] 附近（离散取整容差）。
    assert min(fracs) >= lo - 0.15
    assert max(fracs) <= hi + 1e-6
    # 各向同性：同一裁剪的各轴边长占比近似相等（离散取整下差异很小）。
    assert max(edge_ratios) < 0.1


def test_sample_box_local_scale_smaller_than_global():
    import random
    random.seed(1)
    spatial = (48, 48, 48)
    g = [1.0]
    ll = [1.0]
    for _ in range(200):
        _, gs = _sample_box(spatial, 0.5, 1.0)
        _, ls = _sample_box(spatial, 0.15, 0.5)
        g.append(gs[0] * gs[1] * gs[2] / (48 ** 3))
        ll.append(ls[0] * ls[1] * ls[2] / (48 ** 3))
    assert sum(g) / len(g) > sum(ll) / len(ll)


# ---------------------------------------------------------------------------
# 批量 grid_sample（_affine_grid）
# ---------------------------------------------------------------------------
def test_affine_grid_identity_3d():
    """full-box、out==in、无翻转 → grid_sample 还原输入（整数坐标 bilinear 恒等）。"""
    x = torch.randn(2, 1, 8, 10, 12)
    spatial = [8, 10, 12]
    B = 2
    origins = torch.zeros(B, 3)
    sizes = torch.tensor([spatial, spatial], dtype=torch.float32)
    flips = torch.zeros(B, 3, dtype=torch.bool)
    grid = _affine_grid(spatial, spatial, origins, sizes, flips, 3, x.device)
    out = F.grid_sample(x, grid, mode="bilinear", align_corners=False,
                        padding_mode="border")
    assert torch.allclose(out, x, atol=1e-4)


def test_affine_grid_flip_3d():
    """沿某轴翻转的采样网格 → 输出等于 torch.flip。"""
    x = torch.randn(1, 1, 6, 6, 6)
    spatial = [6, 6, 6]
    origins = torch.zeros(1, 3)
    sizes = torch.tensor([[6, 6, 6]], dtype=torch.float32)
    flips = torch.tensor([[False, True, False]])       # 翻转 H 轴
    grid = _affine_grid(spatial, spatial, origins, sizes, flips, 3, x.device)
    out = F.grid_sample(x, grid, mode="bilinear", align_corners=False,
                        padding_mode="border")
    assert torch.allclose(out, torch.flip(x, dims=[3]), atol=1e-4)


def test_multicrop_batched_output_range_and_shape():
    """批量 grid_sample 路径：输出形状正确、值域不超出输入范围（border+bilinear）。"""
    gen = MultiCropGenerator(
        3, global_size=[8, 8, 8], local_size=[4, 4, 4], n_global=2, n_local=3,
        intensity_scale=0.0, intensity_shift=0.0)
    x = torch.rand(2, 1, 16, 16, 16)
    out = gen(x)
    assert len(out["global"]) == 2 and len(out["local"]) == 3
    for c in out["global"]:
        assert c.shape == (2, 1, 8, 8, 8)
    for c in out["local"]:
        assert c.shape == (2, 1, 4, 4, 4)
        assert float(c.min()) >= float(x.min()) - 1e-4
        assert float(c.max()) <= float(x.max()) + 1e-4


# ---------------------------------------------------------------------------
# checkpoint 原子写 + 指纹
# ---------------------------------------------------------------------------
def test_atomic_save_roundtrip(tmp_path):
    from pathlib import Path
    state = {"a": torch.randn(3, 4), "b": torch.arange(5), "meta": 7}
    path = Path(tmp_path) / "ckpt.pt"
    SSLTrainer._atomic_save(state, path)
    assert path.exists()
    # 无残留临时文件。
    assert not list(Path(tmp_path).glob("ckpt.pt.tmp.*"))
    loaded = torch.load(path)
    assert torch.equal(loaded["a"], state["a"])
    assert torch.equal(loaded["b"], state["b"])
    assert loaded["meta"] == 7


def test_state_fingerprint_deterministic_and_sensitive():
    s1 = {"w": torch.zeros(2, 2), "b": torch.ones(3)}
    s2 = {"b": torch.ones(3), "w": torch.zeros(2, 2)}   # 顺序无关
    fp1 = SSLTrainer._state_fingerprint(s1)
    fp2 = SSLTrainer._state_fingerprint(s2)
    assert fp1 == fp2
    s3 = {"w": torch.zeros(2, 2), "b": torch.ones(3) * 2.0}
    assert SSLTrainer._state_fingerprint(s3) != fp1
    # dtype/shape 变化也应改变指纹。
    s4 = {"w": torch.zeros(2, 2), "b": torch.ones(4)}
    assert SSLTrainer._state_fingerprint(s4) != fp1
