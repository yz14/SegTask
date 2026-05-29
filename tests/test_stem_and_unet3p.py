"""Unit tests for stem upgrade + UNet3+ full-scale skip decoder.

Covers:
- build_stem factory: every mode returns a module + correct stride
- PatchEmbedStem / DualConvStem shapes
- UNet3D final upsample restores input resolution with patch stems
- UNet3PDecoder:
    * output shape matches highest-res encoder feature
    * out_channels attribute is UNet3D-compatible
    * full-scale aggregation: every decoder node depends on all encoder
      levels (gradient check)
    * skip_attention variant produces same-shape output
- End-to-end: every combination of stem × decoder_type × deep_supervision
- Config validation rejects unknown stem_mode / decoder_type
"""

from __future__ import annotations

import pytest
import torch

from segtask_v1.models.stem import (
    STEM_MODES,
    DualConvStem,
    PatchEmbedStem,
    build_stem,
)
from segtask_v1.models.unet3p import UNet3PDecoder


# ---------------------------------------------------------------------------
# build_stem factory
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode,expected_stride", [
    ("conv3", 1), ("conv7", 1), ("dual", 1),
    ("patch2", 2), ("patch4", 4),
])
def test_build_stem_stride(mode, expected_stride):
    stem, stride = build_stem(mode, in_ch=1, out_ch=16)
    assert stride == expected_stride
    x = torch.randn(1, 1, 8, 16, 16)
    y = stem(x)
    assert y.shape == (1, 16,
                       8 // expected_stride,
                       16 // expected_stride,
                       16 // expected_stride)


def test_build_stem_rejects_unknown():
    with pytest.raises(ValueError):
        build_stem("notamode", 1, 16)


def test_dual_conv_stem_shape():
    stem = DualConvStem(1, 16)
    x = torch.randn(2, 1, 8, 16, 16)
    y = stem(x)
    assert y.shape == (2, 16, 8, 16, 16)


def test_patch_embed_stem_patch4():
    stem = PatchEmbedStem(1, 32, patch_size=4)
    x = torch.randn(1, 1, 16, 32, 32)
    y = stem(x)
    assert y.shape == (1, 32, 4, 8, 8)


def test_patch_embed_stem_rejects_bad_patch():
    with pytest.raises(ValueError):
        PatchEmbedStem(1, 16, patch_size=0)


# ---------------------------------------------------------------------------
# UNet3D with patch stem restores input resolution
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stem_mode", ["conv3", "conv7", "dual", "patch2", "patch4"])
def test_unet_restores_resolution(stem_mode):
    from segtask_v1.config import Config
    from segtask_v1.models.factory import build_model

    cfg = Config()
    cfg.data.patch_mode = "z_axis"
    cfg.data.patch_size = [16, 64, 64]
    cfg.data.multi_res_scales = [1.0]
    cfg.data.label_values = [0, 1, 2]
    cfg.model.backbone = "resnet"
    cfg.model.encoder_channels = [16, 32, 64]
    cfg.model.blocks_per_level = 1
    cfg.model.stem_mode = stem_mode
    cfg.sync()
    cfg.validate()

    model = build_model(cfg).eval()
    x = torch.randn(1, cfg.model.in_channels, 16, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape[-3:] == (16, 64, 64), (
        f"Main output should match input resolution for {stem_mode}, got {y.shape}")


def test_unet_deep_supervision_with_patch_stem():
    """With patch stem: main output at input res, DS heads at decoder res."""
    from segtask_v1.config import Config
    from segtask_v1.models.factory import build_model

    cfg = Config()
    cfg.data.patch_mode = "z_axis"
    cfg.data.patch_size = [16, 64, 64]
    cfg.data.multi_res_scales = [1.0]
    cfg.data.label_values = [0, 1, 2]
    cfg.model.backbone = "resnet"
    cfg.model.encoder_channels = [16, 32, 64]
    cfg.model.blocks_per_level = 1
    cfg.model.stem_mode = "patch2"
    cfg.model.deep_supervision = True
    cfg.sync()
    cfg.validate()

    model = build_model(cfg).train()
    x = torch.randn(1, cfg.model.in_channels, 16, 64, 64)
    outs = model(x)
    assert isinstance(outs, list) and len(outs) >= 2
    # Main output restored to input resolution.
    assert outs[0].shape[-3:] == (16, 64, 64)
    # DS outputs at the decoder's native resolutions (≤ input resolution).
    for ds in outs[1:]:
        assert ds.shape[-3] <= 16
        assert ds.shape[-2] <= 64
        assert ds.shape[-1] <= 64


# ---------------------------------------------------------------------------
# UNet3PDecoder standalone
# ---------------------------------------------------------------------------
def _fake_encoder_features(channels, base_shape=(8, 16, 16)):
    feats = []
    d, h, w = base_shape
    for i, c in enumerate(channels):
        s = 2 ** i
        feats.append(torch.randn(1, c, max(d // s, 1), max(h // s, 1), max(w // s, 1)))
    return feats


class TestUNet3PDecoder:
    def test_output_shapes(self):
        enc_ch = [16, 32, 64, 128]
        dec = UNet3PDecoder(enc_ch, cat_channels=16)
        feats = _fake_encoder_features(enc_ch)
        out = dec(feats)
        # out is [low_res, ..., high_res], length = n - 1.
        assert len(out) == len(enc_ch) - 1
        assert out[-1].shape[2:] == feats[0].shape[2:]    # highest-res match
        assert out[0].shape[2:] == feats[-2].shape[2:]    # lowest decoder == 2nd deepest encoder
        for t in out:
            assert t.shape[1] == dec.fused_ch

    def test_out_channels_compat(self):
        enc_ch = [16, 32, 64, 128]
        dec = UNet3PDecoder(enc_ch, cat_channels=16, fused_channels=0)
        assert len(dec.out_channels) == len(enc_ch) - 1
        assert all(c == 16 * len(enc_ch) for c in dec.out_channels)

    def test_full_scale_dependency(self):
        """Every decoder node must depend on every encoder level (this is
        the defining property of full-scale skip)."""
        enc_ch = [16, 32, 64, 128]
        dec = UNet3PDecoder(enc_ch, cat_channels=16)
        feats = [f.detach().requires_grad_(True)
                 for f in _fake_encoder_features(enc_ch)]
        out = dec(feats)
        # Probe highest-res decoder node — it should receive gradient from
        # every single encoder level (both shallower pools and deeper
        # upsamples).
        probe = out[-1]
        probe.sum().backward()
        for i, f in enumerate(feats):
            assert f.grad is not None and f.grad.abs().sum() > 0, (
                f"Highest-res decoder node does not depend on encoder level {i}")

    def test_skip_attention_shape(self):
        enc_ch = [16, 32, 64]
        dec = UNet3PDecoder(enc_ch, cat_channels=8, skip_attention=True)
        feats = _fake_encoder_features(enc_ch)
        out = dec(feats)
        assert len(out) == 2
        assert out[-1].shape[2:] == feats[0].shape[2:]

    def test_rejects_single_level(self):
        with pytest.raises(ValueError):
            UNet3PDecoder([16])


# ---------------------------------------------------------------------------
# UNet3+ end-to-end
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stem_mode", ["conv3", "patch2"])
@pytest.mark.parametrize("backbone", ["resnet", "convnext"])
def test_unet3p_end_to_end(stem_mode, backbone):
    from segtask_v1.config import Config
    from segtask_v1.models.factory import build_model

    cfg = Config()
    cfg.data.patch_mode = "z_axis"
    cfg.data.patch_size = [16, 64, 64]
    cfg.data.multi_res_scales = [1.0]
    cfg.data.label_values = [0, 1, 2]
    cfg.model.backbone = backbone
    cfg.model.encoder_channels = [16, 32, 64]
    cfg.model.blocks_per_level = 1
    cfg.model.stem_mode = stem_mode
    cfg.model.decoder_type = "unet3p"
    cfg.model.unet3p_cat_channels = 16
    cfg.sync()
    cfg.validate()

    model = build_model(cfg).eval()
    x = torch.randn(1, cfg.model.in_channels, 16, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape[-3:] == (16, 64, 64)


def test_unet3p_with_deep_supervision_and_skip_attention():
    from segtask_v1.config import Config
    from segtask_v1.models.factory import build_model

    cfg = Config()
    cfg.data.patch_mode = "z_axis"
    cfg.data.patch_size = [16, 64, 64]
    cfg.data.multi_res_scales = [1.0]
    cfg.data.label_values = [0, 1, 2]
    cfg.model.backbone = "resnet"
    cfg.model.encoder_channels = [16, 32, 64, 96]
    cfg.model.blocks_per_level = 1
    cfg.model.decoder_type = "unet3p"
    cfg.model.unet3p_cat_channels = 16
    cfg.model.skip_attention = True
    cfg.model.deep_supervision = True
    cfg.sync()
    cfg.validate()

    model = build_model(cfg).train()
    x = torch.randn(1, cfg.model.in_channels, 16, 64, 64)
    outs = model(x)
    assert isinstance(outs, list)
    assert outs[0].shape[-3:] == (16, 64, 64)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
class TestConfigValidation:
    def test_rejects_bad_stem_mode(self):
        from segtask_v1.config import Config
        cfg = Config()
        cfg.model.stem_mode = "notastem"
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_rejects_bad_decoder_type(self):
        from segtask_v1.config import Config
        cfg = Config()
        cfg.model.decoder_type = "transformer"
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_rejects_bad_unet3p_cat_channels(self):
        from segtask_v1.config import Config
        cfg = Config()
        cfg.model.unet3p_cat_channels = 0
        with pytest.raises(AssertionError):
            cfg.validate()
