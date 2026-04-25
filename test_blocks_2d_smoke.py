"""Smoke test for R2.1: blocks.py spatial_dims parameterization.

Verifies:
  1. 3D path (default spatial_dims=3) still works for every block.
  2. 2D path (spatial_dims=2) constructs and forwards for every block.
  3. CARAFE/DySample correctly reject spatial_dims=2 via Upsample.

Run:
    conda activate torch27_env
    python test_blocks_2d_smoke.py
"""
from __future__ import annotations

import sys

import torch

from segtask_v1.models.blocks import (
    INTERP_SMOOTH,
    AttentionGate3D,
    BlurPool3d,
    CBAM3D,
    CoordAttention3D,
    ConvNormAct,
    Downsample,
    ECA3D,
    PixelShuffle3d,
    PixelUnshuffle3d,
    SqueezeExcite3D,
    Upsample,
    get_conv,
    get_norm,
    icnr_init_,
    make_attention,
)


def _ok(name: str, msg: str = "") -> None:
    print(f"  [PASS] {name}{(' — ' + msg) if msg else ''}")


def _fail(name: str, err: Exception) -> None:
    print(f"  [FAIL] {name}: {type(err).__name__}: {err}")
    raise


# ---------------------------------------------------------------------------
# Test driver: build/forward each block in 3D and 2D and assert shapes.
# ---------------------------------------------------------------------------
def make_inputs(B: int, C: int, dims: int, size: int = 8):
    if dims == 3:
        return torch.randn(B, C, size, size, size)
    return torch.randn(B, C, size, size)


def test_get_norm():
    for d in (2, 3):
        for nt in ("batch", "instance", "group"):
            m = get_norm(nt, 16, num_groups=8, spatial_dims=d)
            x = make_inputs(2, 16, d)
            y = m(x)
            assert y.shape == x.shape, f"get_norm {nt} d={d} shape mismatch"
        _ok(f"get_norm spatial_dims={d}")


def test_conv_norm_act():
    for d in (2, 3):
        m = ConvNormAct(8, 16, kernel_size=3, stride=1, padding=1,
                        norm_type="instance", spatial_dims=d, dropout=0.1)
        x = make_inputs(2, 8, d)
        y = m(x)
        assert y.shape[1] == 16
        assert y.shape[2:] == x.shape[2:]
        _ok(f"ConvNormAct d={d}")


def test_attention_blocks():
    for d in (2, 3):
        for kind, cls in [
            ("se", SqueezeExcite3D),
            ("eca", ECA3D),
            ("cbam", CBAM3D),
            ("coord", CoordAttention3D),
        ]:
            m = make_attention(kind, 16, spatial_dims=d)
            x = make_inputs(2, 16, d)
            y = m(x)
            assert y.shape == x.shape, f"{kind} d={d} shape changed"
        _ok(f"attention factory d={d}")


def test_attention_gate():
    for d in (2, 3):
        ag = AttentionGate3D(x_ch=16, g_ch=32, spatial_dims=d)
        size = 8
        x = make_inputs(2, 16, d, size=size)
        # Coarser g — half the spatial size, will be resized inside
        if d == 3:
            g = torch.randn(2, 32, size // 2, size // 2, size // 2)
        else:
            g = torch.randn(2, 32, size // 2, size // 2)
        y = ag(x, g)
        assert y.shape == x.shape
        _ok(f"AttentionGate3D d={d}")


def test_blur_pool():
    for d in (2, 3):
        bp = BlurPool3d(channels=4, stride=2, filt_size=3, spatial_dims=d)
        x = make_inputs(2, 4, d, size=8)
        y = bp(x)
        # Expect spatial halved
        for s_in, s_out in zip(x.shape[2:], y.shape[2:]):
            assert s_out == s_in // 2, f"BlurPool d={d} expected halved"
        _ok(f"BlurPool3d d={d}")


def test_pixel_shuffle_roundtrip():
    """Unshuffle then shuffle should recover the input exactly."""
    for d in (2, 3):
        un = PixelUnshuffle3d(r=2, spatial_dims=d)
        sh = PixelShuffle3d(r=2, spatial_dims=d)
        x = make_inputs(2, 4, d, size=8)
        y = un(x)
        # Channel grew by 2^d
        assert y.shape[1] == 4 * (2 ** d)
        z = sh(y)
        assert torch.allclose(x, z), f"PixelShuffle roundtrip mismatch d={d}"
        _ok(f"PixelShuffle/Unshuffle3d d={d} (lossless roundtrip)")


def test_icnr_init():
    for d in (2, 3):
        # Simulate a conv weight for sub-pixel: out = in*r^d, kernel=1
        in_ch, base_out, r = 4, 8, 2
        rd = r ** d
        ksize = (1,) * d
        w = torch.empty(base_out * rd, in_ch, *ksize)
        icnr_init_(w, upscale=r, spatial_dims=d)
        # Check siblings replicated: every block of rd consecutive
        # output filters should be identical.
        first = w[0]
        for i in range(1, rd):
            assert torch.allclose(w[i], first), \
                f"ICNR siblings not equal d={d} block 0"
        _ok(f"icnr_init_ d={d}")


def test_downsample_modes():
    for d in (2, 3):
        for mode in ("conv", "maxpool", "avgpool", "blurpool", "pixelunshuffle"):
            m = Downsample(in_ch=4, out_ch=8, mode=mode, spatial_dims=d)
            x = make_inputs(2, 4, d, size=8)
            y = m(x)
            assert y.shape[1] == 8
            for s_in, s_out in zip(x.shape[2:], y.shape[2:]):
                assert s_out == s_in // 2
        _ok(f"Downsample all modes d={d}")


def test_upsample_modes():
    for d in (2, 3):
        modes = ["transpose", "trilinear", "nearest", "pixelshuffle"]
        if d == 3:
            modes += ["carafe", "dysample"]
        for mode in modes:
            m = Upsample(in_ch=8, out_ch=4, mode=mode, spatial_dims=d)
            x = make_inputs(2, 8, d, size=4)
            y = m(x)
            assert y.shape[1] == 4
            for s_in, s_out in zip(x.shape[2:], y.shape[2:]):
                assert s_out == s_in * 2, (
                    f"Upsample {mode} d={d} expected doubled, got {s_out}")
        _ok(f"Upsample all modes d={d}")


def test_upsample_2d_rejects_3d_only_modes():
    for mode in ("carafe", "dysample"):
        try:
            Upsample(in_ch=8, out_ch=4, mode=mode, spatial_dims=2)
        except ValueError as e:
            assert "spatial_dims=3" in str(e)
            continue
        raise AssertionError(f"Upsample {mode} 2D should have raised")
    _ok("Upsample 2D rejects carafe/dysample")


def test_get_conv_factory():
    assert get_conv(2) is torch.nn.Conv2d
    assert get_conv(3) is torch.nn.Conv3d
    try:
        get_conv(1)
    except ValueError:
        _ok("get_conv rejects spatial_dims not in {2,3}")
        return
    raise AssertionError("get_conv(1) should raise")


# ---------------------------------------------------------------------------
# R2.2: backbone stages + stems
# ---------------------------------------------------------------------------
def test_resnet_stages():
    from segtask_v1.models.resnet import ResNetStage
    for d in (2, 3):
        for block_type in ("basic", "preact", "bottleneck"):
            m = ResNetStage(
                in_ch=4, out_ch=8, num_blocks=2,
                norm_type="instance", activation="leakyrelu",
                attention_type="se", block_type=block_type,
                spatial_dims=d)
            x = make_inputs(2, 4, d, size=8)
            y = m(x)
            assert y.shape[1] == 8
            assert y.shape[2:] == x.shape[2:]
        _ok(f"ResNetStage all block types d={d}")


def test_convnext_stages():
    from segtask_v1.models.convnext import ConvNeXtStage
    for d in (2, 3):
        m = ConvNeXtStage(
            in_ch=4, out_ch=8, num_blocks=2,
            attention_type="eca", spatial_dims=d,
            drop_path_rates=[0.0, 0.05])
        x = make_inputs(2, 4, d, size=8)
        y = m(x)
        assert y.shape[1] == 8
        assert y.shape[2:] == x.shape[2:]
        _ok(f"ConvNeXtStage d={d}")


def test_stem_builders():
    from segtask_v1.models.stem import build_stem
    for d in (2, 3):
        for mode in ("conv3", "conv7", "dual", "patch2", "patch4"):
            stem, stride = build_stem(
                mode=mode, in_ch=1, out_ch=8,
                norm_type="instance", activation="leakyrelu",
                spatial_dims=d)
            x = make_inputs(2, 1, d, size=8)
            y = stem(x)
            expected_size = 8 // stride
            assert y.shape[1] == 8
            for s in y.shape[2:]:
                assert s == expected_size, (
                    f"stem={mode} d={d} stride={stride} "
                    f"expected {expected_size}, got {s}")
        _ok(f"build_stem all modes d={d}")


# ---------------------------------------------------------------------------
# R2.3: end-to-end UNet build via factory
# ---------------------------------------------------------------------------
def _build_test_cfg(spatial_dims: int, decoder_type: str = "unet",
                    backbone: str = "resnet"):
    from segtask_v1.config import Config
    cfg = Config()
    cfg.data.label_values = [0, 1, 2]
    cfg.data.num_classes = 3
    # Use small patches that survive 4 downsamples (32 / 16 = 2).
    if spatial_dims == 3:
        cfg.data.patch_size = [16, 32, 32]
    else:
        cfg.data.patch_size = [12, 32, 32]   # D will become input channels
    cfg.model.spatial_dims = spatial_dims
    cfg.model.backbone = backbone
    cfg.model.encoder_channels = [16, 32, 64, 128]   # 4 levels, fewer params
    cfg.model.decoder_type = decoder_type
    cfg.sync()
    cfg.validate()
    return cfg


def test_unet_factory_3d_default():
    """Backward-compat: original 3D path still produces correct shape."""
    from segtask_v1.models.factory import build_model
    cfg = _build_test_cfg(spatial_dims=3, decoder_type="unet")
    model = build_model(cfg).eval()
    x = torch.randn(1, 1, *cfg.data.patch_size)
    y = model(x)
    assert y.shape == (1, 2, *cfg.data.patch_size), (
        f"3D UNet expected (1, 2, *patch); got {tuple(y.shape)}")
    _ok("UNet factory 3D end-to-end (default decoder)")


def test_unet_factory_2d_unet():
    """End-to-end 2D UNet: input rank 4, output rank 4."""
    from segtask_v1.models.factory import build_model
    cfg = _build_test_cfg(spatial_dims=2, decoder_type="unet")
    model = build_model(cfg).eval()
    # 2D model expects (B, C_in, H, W); for 2.5D, C_in == D slices.
    # Use generic in_channels=1 here (no multi-res), H=W=32.
    x = torch.randn(1, cfg.model.in_channels, 32, 32)
    y = model(x)
    assert y.ndim == 4, f"2D UNet output must be rank-4, got {y.ndim}"
    assert y.shape[0] == 1
    assert y.shape[1] == cfg.num_fg_classes  # default num_res=1
    assert y.shape[2:] == (32, 32)
    _ok("UNet factory 2D end-to-end (unet decoder)")


def test_unet_factory_2d_unetpp():
    from segtask_v1.models.factory import build_model
    cfg = _build_test_cfg(spatial_dims=2, decoder_type="unetpp")
    model = build_model(cfg).eval()
    x = torch.randn(1, 1, 32, 32)
    y = model(x)
    assert y.shape == (1, 2, 32, 32)
    _ok("UNet factory 2D end-to-end (unetpp decoder)")


def test_unet_factory_2d_unet3p():
    from segtask_v1.models.factory import build_model
    cfg = _build_test_cfg(spatial_dims=2, decoder_type="unet3p")
    model = build_model(cfg).eval()
    x = torch.randn(1, 1, 32, 32)
    y = model(x)
    assert y.shape == (1, 2, 32, 32)
    _ok("UNet factory 2D end-to-end (unet3p decoder)")


def test_unet_factory_2d_convnext_with_attention():
    """ConvNeXt backbone + skip_attention + ECA in 2D, deep supervision on."""
    from segtask_v1.models.factory import build_model
    cfg = _build_test_cfg(spatial_dims=2, decoder_type="unet",
                          backbone="convnext")
    cfg.model.attention_type = "eca"
    cfg.model.skip_attention = True
    cfg.model.deep_supervision = True
    cfg.model.upsample_mode = "trilinear"  # bilinear in 2D path
    cfg.validate()
    model = build_model(cfg).train()  # DS only active in train mode
    x = torch.randn(1, 1, 32, 32)
    out = model(x)
    assert isinstance(out, list), "deep_supervision should return a list"
    assert out[0].shape == (1, 2, 32, 32), \
        f"main DS output shape mismatch: {out[0].shape}"
    for sub in out[1:]:
        assert sub.ndim == 4
        assert sub.shape[0] == 1 and sub.shape[1] == 2
    _ok("UNet factory 2D ConvNeXt + ECA + skip-attn + DS")


def test_unet_factory_2d_patch_stem_resolution_restored():
    """patch2 stem: encoder runs at 1/2 res; main output must be restored."""
    from segtask_v1.models.factory import build_model
    cfg = _build_test_cfg(spatial_dims=2, decoder_type="unet")
    cfg.model.stem_mode = "patch2"
    cfg.validate()
    model = build_model(cfg).eval()
    x = torch.randn(1, 1, 32, 32)
    y = model(x)
    assert y.shape == (1, 2, 32, 32), (
        f"patchN stem 2D should restore resolution; got {tuple(y.shape)}")
    _ok("UNet factory 2D patch-stem resolution restored")


def main():
    torch.manual_seed(0)
    print("R2.1 + R2.2 + R2.3 smoke test — spatial_dims parameterization")
    print("=" * 60)
    tests = [
        # R2.1 — blocks.py
        test_get_norm,
        test_get_conv_factory,
        test_conv_norm_act,
        test_attention_blocks,
        test_attention_gate,
        test_blur_pool,
        test_pixel_shuffle_roundtrip,
        test_icnr_init,
        test_downsample_modes,
        test_upsample_modes,
        test_upsample_2d_rejects_3d_only_modes,
        # R2.2 — backbone stages + stems
        test_resnet_stages,
        test_convnext_stages,
        test_stem_builders,
        # R2.3 — UNet factory end-to-end
        test_unet_factory_3d_default,
        test_unet_factory_2d_unet,
        test_unet_factory_2d_unetpp,
        test_unet_factory_2d_unet3p,
        test_unet_factory_2d_convnext_with_attention,
        test_unet_factory_2d_patch_stem_resolution_restored,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            _fail(t.__name__, e)
            return 1
    print("=" * 60)
    print("All R2.x smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
