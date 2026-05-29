"""Unit tests for UNet++ nested dense decoder.

Covers:
- UNetPPDecoder standalone: shapes, ``out_channels`` compatibility, node count
- Dense dependency: highest-res decoder output depends on ALL encoder levels
  AND on all intermediate nested columns (topology correctness)
- skip_attention variant produces same-shape output
- End-to-end UNet3D build across (backbone × stem × decoder_type)
- Deep supervision with UNet++
- Config validation accepts "unetpp" and rejects unknown decoder_type
"""

from __future__ import annotations

import pytest
import torch

from segtask_v1.models.unetpp import UNetPPDecoder


def _fake_encoder_features(channels, base_shape=(8, 16, 16)):
    feats = []
    d, h, w = base_shape
    for i, c in enumerate(channels):
        s = 2 ** i
        feats.append(torch.randn(
            1, c, max(d // s, 1), max(h // s, 1), max(w // s, 1)))
    return feats


def _simple_stage_builder(in_ch, out_ch):
    """Minimal stage for testing: single 3x3x3 conv + BN + ReLU."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )


# ---------------------------------------------------------------------------
# Standalone decoder
# ---------------------------------------------------------------------------
class TestUNetPPDecoder:
    def test_output_shapes_and_channels(self):
        enc_ch = [16, 32, 64, 128]  # n = 4
        dec = UNetPPDecoder(enc_ch, stage_builder=_simple_stage_builder)
        feats = _fake_encoder_features(enc_ch)
        out = dec(feats)
        # Returns n-1 = 3 feature maps, low-res → high-res.
        assert len(out) == 3
        # Spatial resolutions mirror encoder levels 2, 1, 0 (low → high).
        assert out[0].shape[2:] == feats[2].shape[2:]
        assert out[1].shape[2:] == feats[1].shape[2:]
        assert out[2].shape[2:] == feats[0].shape[2:]
        # Channel widths: [enc[2], enc[1], enc[0]] = [64, 32, 16].
        assert [t.shape[1] for t in out] == [64, 32, 16]

    def test_out_channels_matches_classical_decoder(self):
        """UNet3D uses decoder.out_channels to build seg / DS heads; the
        widths must match the classical Decoder convention exactly."""
        enc_ch = [16, 32, 64, 128, 256]
        dec = UNetPPDecoder(enc_ch, stage_builder=_simple_stage_builder)
        # Classical Decoder: [enc[n-2], enc[n-3], ..., enc[0]]
        assert dec.out_channels == [128, 64, 32, 16]

    def test_node_count(self):
        """UNet++ creates n*(n-1)/2 nested nodes plus encoder on top."""
        enc_ch = [16, 32, 64, 128, 256]  # n = 5
        dec = UNetPPDecoder(enc_ch, stage_builder=_simple_stage_builder)
        # Nested nodes = sum_{i=0..n-2} (n-1-i) = 4+3+2+1 = 10
        assert len(dec.blocks) == 10
        assert len(dec.upsamples) == 10

    def test_highest_res_depends_on_all_encoder_levels(self):
        """Dense skip paths → X[0, n-1] must have gradient w.r.t. every E_i."""
        enc_ch = [16, 32, 64, 128]  # n = 4
        dec = UNetPPDecoder(enc_ch, stage_builder=_simple_stage_builder)
        feats = [f.detach().requires_grad_(True)
                 for f in _fake_encoder_features(enc_ch)]
        out = dec(feats)
        # Highest-res diagonal output = X[0, n-1] = X[0, 3]
        out[-1].sum().backward()
        for i, f in enumerate(feats):
            assert f.grad is not None and f.grad.abs().sum() > 0, (
                f"Highest-res X[0,{len(enc_ch)-1}] does not depend on E_{i}")

    def test_skip_attention_shape(self):
        enc_ch = [8, 16, 32]
        dec = UNetPPDecoder(enc_ch, stage_builder=_simple_stage_builder,
                            skip_attention=True)
        feats = _fake_encoder_features(enc_ch)
        out = dec(feats)
        assert len(out) == 2
        assert out[-1].shape[2:] == feats[0].shape[2:]

    def test_rejects_single_level(self):
        with pytest.raises(ValueError):
            UNetPPDecoder([16], stage_builder=_simple_stage_builder)


# ---------------------------------------------------------------------------
# End-to-end UNet3D
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backbone", ["resnet", "convnext"])
@pytest.mark.parametrize("stem_mode", ["conv3", "patch2"])
def test_unetpp_end_to_end(backbone, stem_mode):
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
    cfg.model.decoder_type = "unetpp"
    cfg.sync()
    cfg.validate()

    model = build_model(cfg).eval()
    x = torch.randn(1, cfg.model.in_channels, 16, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape[-3:] == (16, 64, 64)


def test_unetpp_with_deep_supervision_and_skip_attention():
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
    cfg.model.decoder_type = "unetpp"
    cfg.model.skip_attention = True
    cfg.model.deep_supervision = True
    cfg.model.upsample_mode = "pixelshuffle"
    cfg.sync()
    cfg.validate()

    model = build_model(cfg).train()
    x = torch.randn(1, cfg.model.in_channels, 16, 64, 64)
    outs = model(x)
    # 4 encoder levels → n-1 = 3 decoder features → 1 main + 2 DS = 3 outputs.
    assert isinstance(outs, list) and len(outs) == 3
    # Main at input resolution; DS at progressively lower resolutions.
    assert outs[0].shape[-3:] == (16, 64, 64)
    for i in range(1, len(outs)):
        # Each DS output must be at or below its predecessor's resolution
        assert outs[i].shape[-1] <= outs[i - 1].shape[-1]


def test_unetpp_trains_one_step():
    """Smoke test: one full forward + backward + optimizer step."""
    from segtask_v1.config import Config
    from segtask_v1.models.factory import build_model

    cfg = Config()
    cfg.data.patch_mode = "z_axis"
    cfg.data.patch_size = [8, 32, 32]
    cfg.data.multi_res_scales = [1.0]
    cfg.data.label_values = [0, 1, 2]
    cfg.model.backbone = "resnet"
    cfg.model.encoder_channels = [8, 16, 32]
    cfg.model.blocks_per_level = 1
    cfg.model.decoder_type = "unetpp"
    cfg.sync()
    cfg.validate()

    model = build_model(cfg).train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    x = torch.randn(1, cfg.model.in_channels, 8, 32, 32)
    tgt = torch.randint(0, 2, (1, cfg.num_fg_classes, 8, 32, 32)).float()

    opt.zero_grad()
    y = model(x)
    logits = y[0] if isinstance(y, list) else y
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, tgt)
    loss.backward()
    # All parameters (nested nodes included) should receive gradient.
    zero_grad_params = [n for n, p in model.named_parameters()
                        if p.grad is None or p.grad.abs().sum() == 0]
    assert not zero_grad_params, (
        f"Parameters with no gradient after backward: {zero_grad_params[:5]}")
    opt.step()


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
def test_config_accepts_unetpp():
    from segtask_v1.config import Config
    cfg = Config()
    cfg.model.decoder_type = "unetpp"
    cfg.sync()
    cfg.validate()  # should not raise


def test_config_still_rejects_unknown_decoder():
    from segtask_v1.config import Config
    cfg = Config()
    cfg.model.decoder_type = "transformer"
    with pytest.raises(AssertionError):
        cfg.validate()
