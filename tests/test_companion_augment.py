"""P2e：Companion 空间变换合流 — 固定 seed CPU 等价性。"""

from __future__ import annotations

import torch

from taskcore.config import AugConfig
from taskcore.data.augment import (
    Companion,
    GPUAugmentor,
    _random_affine_elastic,
    _random_affine_elastic_companions,
    _random_flip,
    _random_flip_companions,
)
from gentask.data.augment import GPUAugmentor as GenGPUAugmentor


def _spatial_only_cfg(**kwargs) -> AugConfig:
    """关掉强度变换，只测空间路径。"""
    cfg = AugConfig(enabled=True)
    cfg.random_brightness_prob = 0.0
    cfg.random_contrast_prob = 0.0
    cfg.random_gamma_prob = 0.0
    cfg.gaussian_noise_prob = 0.0
    cfg.gaussian_blur_prob = 0.0
    cfg.simulate_lowres_prob = 0.0
    cfg.intensity_clamp = False
    cfg.inplace = False
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def test_flip_companions_match_legacy():
    torch.manual_seed(0)
    img = torch.randn(2, 1, 4, 8, 8)
    lbl = torch.randint(0, 3, (2, 1, 4, 8, 8)).float()
    wm = torch.rand(2, 1, 4, 8, 8)

    gen = torch.Generator().manual_seed(11)
    img_a, lbl_a, wm_a = _random_flip(
        img.clone(), lbl.clone(), prob=1.0, axes=[2, 3, 4],
        weight_map=wm.clone(), gen_cpu=gen)

    gen = torch.Generator().manual_seed(11)
    comps = [
        Companion(lbl.clone(), "nearest", None),
        Companion(wm.clone(), "nearest", None),
    ]
    img_b, comps = _random_flip_companions(
        img.clone(), prob=1.0, axes=[2, 3, 4],
        companions=comps, gen_cpu=gen)

    assert torch.equal(img_a, img_b)
    assert torch.equal(lbl_a, comps[0].tensor)
    assert torch.equal(wm_a, comps[1].tensor)


def test_affine_companions_match_legacy_oob_fill():
    torch.manual_seed(1)
    img = torch.randn(1, 1, 8, 16, 16)
    lbl = torch.full((1, 1, 8, 16, 16), 3.0)
    wm = torch.full((1, 1, 8, 16, 16), 2.0)

    gen = torch.Generator().manual_seed(22)
    img_a, lbl_a, wm_a = _random_affine_elastic(
        img.clone(), lbl.clone(),
        affine_prob=1.0, rotate_range=[0.0, 0.0],
        scale_range=[1.0, 1.0],
        elastic_prob=0.0, sigma=5.0, alpha=0.0,
        weight_map=wm.clone(), wmap_mode="nearest",
        translate_range=[0.5, 0.5], label_fill=0.0,
        gen_cpu=gen)

    gen = torch.Generator().manual_seed(22)
    comps = [
        Companion(lbl.clone(), "nearest", 0.0),
        Companion(wm.clone(), "nearest", 1.0),
    ]
    img_b, comps = _random_affine_elastic_companions(
        img.clone(),
        affine_prob=1.0, rotate_range=[0.0, 0.0],
        scale_range=[1.0, 1.0],
        elastic_prob=0.0, sigma=5.0, alpha=0.0,
        companions=comps, translate_range=[0.5, 0.5],
        gen_cpu=gen)

    assert torch.allclose(img_a, img_b)
    assert torch.equal(lbl_a, comps[0].tensor)
    assert torch.equal(wm_a, comps[1].tensor)
    assert (comps[0].tensor == 0.0).any()
    assert (comps[1].tensor == 1.0).any()


def test_gen_wmap_oob_preserves_border_not_neutral_one():
    """生成路径 wmap oob_fill=None：越界保留 border，不强制填 1.0。"""
    cfg = _spatial_only_cfg(
        random_flip_prob=0.0,
        random_affine_prob=1.0,
        random_rotate_range=[0.0, 0.0],
        random_scale_range=[1.0, 1.0],
        elastic_deform_prob=0.0,
        grid_dropout_prob=0.0,
        random_translate_range=[0.5, 0.5],
        wmap_interp_mode="nearest",
    )
    aug = GenGPUAugmentor(cfg, seed=7)
    img = torch.randn(1, 1, 8, 16, 16)
    # 全 2.0：若误用 seg 的 oob_fill=1.0 会看到 1；border 则仍为 2。
    wm = torch.full((1, 1, 8, 16, 16), 2.0)
    cond = torch.randn(1, 1, 8, 16, 16)
    _, wm_out, cond_out = aug(img, wm, cond)
    assert wm_out is not None and cond_out is not None
    assert set(wm_out.unique().tolist()) == {2.0}
    assert cond_out.shape == cond.shape


def test_seg_apply_matches_call_api():
    cfg = _spatial_only_cfg(
        random_flip_prob=1.0,
        random_flip_axes=[2],
        random_affine_prob=0.0,
        elastic_deform_prob=0.0,
        grid_dropout_prob=0.0,
    )
    aug = GPUAugmentor(cfg, label_fill=0.0, seed=99)
    img = torch.randn(2, 1, 4, 8, 8)
    lbl = torch.randint(0, 2, (2, 1, 4, 8, 8)).float()
    wm = torch.rand(2, 1, 4, 8, 8)

    img1, lbl1, wm1 = aug(img.clone(), lbl.clone(), wm.clone())

    aug2 = GPUAugmentor(cfg, label_fill=0.0, seed=99)
    comps = [
        Companion(lbl.clone(), "nearest", 0.0),
        Companion(wm.clone(), aug2.wmap_interp_mode, 1.0),
    ]
    img2, comps = aug2.apply(img.clone(), comps)

    assert torch.equal(img1, img2)
    assert torch.equal(lbl1, comps[0].tensor)
    assert torch.equal(wm1, comps[1].tensor)


def test_gen_rank4_squeeze_roundtrip():
    cfg = _spatial_only_cfg(
        random_flip_prob=1.0,
        random_flip_axes=[2],
        random_affine_prob=0.0,
        elastic_deform_prob=0.0,
        grid_dropout_prob=0.0,
    )
    aug = GenGPUAugmentor(cfg, seed=3)
    img = torch.randn(2, 4, 8, 8)  # rank-4
    wm = torch.ones(2, 4, 8, 8)
    cond = torch.randn(2, 4, 8, 8)
    out_i, out_w, out_c = aug(img, wm, cond)
    assert out_i.ndim == 4 and out_w.ndim == 4 and out_c.ndim == 4
    assert out_i.shape == img.shape
