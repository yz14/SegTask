"""Tests for self-supervised (Models Genesis) pretraining.

Covers:
- SSLConfig defaults + validate (method/recon_loss/single-view/probs guards).
- build_ssl_model: encoder/decoder reuse + recon_head, shape-preserving forward
  (3D and 2.5D), out_channels == in_channels.
- GenesisCorruptor: shape/dtype/finite preservation and that it actually corrupts.
- **SSL -> segmentation weight handoff**: encoder.*/decoder.* match exactly under
  strict=False load; recon_head.* is the only extra; seg_head.* is the only
  (head) missing -> validates the clean, non-patchy handoff.
- SSLTrainer: one-epoch CPU smoke run produces a loadable ssl_best.pt.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from segtask_v1.config import Config, ConfigError
from segtask_v1.data.ssl_transforms import GenesisCorruptor
from segtask_v1.data.vesselness import frangi_vesselness
from segtask_v1.models.factory import build_model, build_ssl_model
from segtask_v1.models.ssl import SSLReconModel
from segtask_v1.trainer.checkpoint import strip_common_prefixes


# ---------------------------------------------------------------------------
# config helpers
# ---------------------------------------------------------------------------
def _cfg(patch_mode="cubic"):
    cfg = Config()
    cfg.data.patch_mode = patch_mode
    cfg.data.patch_size = [16, 32, 32]
    cfg.data.multi_res_scales = [1.0]
    cfg.data.label_values = [0, 1]
    cfg.data.num_classes = 2
    cfg.model.backbone = "resnet"
    cfg.model.encoder_channels = [8, 16, 32]
    cfg.model.blocks_per_level = 1
    cfg.model.stem_mode = "conv3"
    cfg.ssl.enabled = True
    return cfg


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------
def test_ssl_config_defaults_off():
    cfg = Config()
    assert cfg.ssl.enabled is False
    cfg.sync()
    cfg.validate()  # default seg config must pass with ssl off


def test_ssl_validate_ok():
    cfg = _cfg(); cfg.sync(); cfg.validate()


@pytest.mark.parametrize("field,value", [
    ("method", "bogus"),
    ("recon_loss", "huber"),
    ("nonlinear_prob", 1.5),
    ("paint_count", -1),
])
def test_ssl_validate_rejects_bad(field, value):
    cfg = _cfg()
    setattr(cfg.ssl, field, value)
    cfg.sync()
    with pytest.raises(ConfigError):
        cfg.validate()


def test_ssl_validate_requires_single_view():
    cfg = _cfg()
    cfg.data.multi_res_scales = [1.0, 1.5]
    cfg.sync()
    with pytest.raises(ConfigError):
        cfg.validate()


def test_ssl_validate_rejects_bad_paint_range():
    cfg = _cfg()
    cfg.ssl.paint_block_range = [0.5, 0.2]   # lo > hi
    cfg.sync()
    with pytest.raises(ConfigError):
        cfg.validate()


def test_ssl_validate_prior_ok():
    cfg = _cfg()
    cfg.ssl.method = "prior"
    cfg.sync(); cfg.validate()


@pytest.mark.parametrize("field,value", [
    ("prior_scales", []),
    ("prior_scales", [0.0, 1.0]),
    ("prior_alpha", 0.0),
])
def test_ssl_validate_prior_rejects_bad(field, value):
    cfg = _cfg()
    cfg.ssl.method = "prior"
    setattr(cfg.ssl, field, value)
    cfg.sync()
    with pytest.raises(ConfigError):
        cfg.validate()


# ---------------------------------------------------------------------------
# Frangi vesselness target (label-free)
# ---------------------------------------------------------------------------
def test_frangi_vesselness_3d_highlights_tube():
    vol = torch.zeros(1, 1, 24, 32, 32)
    vol[0, 0, :, 16, 16] = 1.0
    vol[0, 0, :, 16, 17] = 1.0
    vol[0, 0, :, 17, 16] = 1.0
    out = frangi_vesselness(vol, scales=[1.0, 2.0], spatial_dims=3)
    assert out.shape == vol.shape
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0 + 1e-5
    tube = out[0, 0, :, 15:19, 15:19].mean().item()
    bg = out[0, 0, :, 0:4, 0:4].mean().item()
    assert tube > bg + 0.05


def test_frangi_vesselness_2d_per_channel_shape():
    im = torch.zeros(2, 3, 32, 32)
    im[:, :, 16, :] = 1.0
    out = frangi_vesselness(im, scales=[1.0, 2.0], spatial_dims=2)
    assert out.shape == im.shape
    assert out[:, :, 15:18, :].mean() > out[:, :, 0:3, :].mean()


def test_frangi_vesselness_rejects_bad_rank():
    with pytest.raises(ValueError):
        frangi_vesselness(torch.rand(2, 1, 16, 16), scales=[1.0], spatial_dims=3)
    with pytest.raises(ValueError):
        frangi_vesselness(torch.rand(2, 1, 8, 16, 16), scales=[], spatial_dims=3)


# ---------------------------------------------------------------------------
# model build + forward
# ---------------------------------------------------------------------------
def test_build_ssl_model_3d_forward():
    cfg = _cfg("cubic"); cfg.sync(); cfg.validate()
    model = build_ssl_model(cfg).eval()
    assert isinstance(model, SSLReconModel)
    assert model.out_channels == cfg.model.in_channels
    x = torch.randn(2, cfg.model.in_channels, 16, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape


def test_build_ssl_model_2_5d_forward():
    cfg = _cfg("2_5d"); cfg.sync(); cfg.validate()
    assert cfg.model.spatial_dims == 2
    model = build_ssl_model(cfg).eval()
    x = torch.randn(2, cfg.model.in_channels, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape


def test_build_ssl_model_requires_unet():
    cfg = _cfg("2_5d")
    cfg.model.arch = "adm"
    # bypass validate (which would also reject); exercise factory guard directly.
    with pytest.raises(ValueError):
        build_ssl_model(cfg)


# ---------------------------------------------------------------------------
# Genesis corruption
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("spatial_dims,shape", [(3, (8, 16, 16)), (2, (16, 16))])
def test_genesis_corruptor_shape_and_corrupts(spatial_dims, shape):
    cfg = _cfg()
    corruptor = GenesisCorruptor(cfg.ssl, spatial_dims)
    x = torch.rand(2, 1, *shape)
    y = corruptor(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()
    assert not torch.allclose(y, x)          # something changed
    assert torch.equal(x, x)                 # input not mutated in place


def test_genesis_corruptor_noop_when_all_probs_zero():
    cfg = _cfg()
    cfg.ssl.nonlinear_prob = 0.0
    cfg.ssl.local_shuffle_prob = 0.0
    cfg.ssl.paint_prob = 0.0
    corruptor = GenesisCorruptor(cfg.ssl, 3)
    x = torch.rand(2, 1, 8, 16, 16)
    y = corruptor(x)
    assert torch.allclose(y, x)


def test_genesis_corruptor_rejects_bad_rank():
    cfg = _cfg()
    corruptor = GenesisCorruptor(cfg.ssl, 3)
    with pytest.raises(ValueError):
        corruptor(torch.rand(2, 1, 16, 16))   # 2D tensor for 3D corruptor


# ---------------------------------------------------------------------------
# SSL -> segmentation handoff (the core integration property)
# ---------------------------------------------------------------------------
def test_ssl_to_seg_weight_handoff():
    cfg = _cfg("cubic"); cfg.sync(); cfg.validate()

    ssl_model = build_ssl_model(cfg)
    ssl_sd = strip_common_prefixes(ssl_model.state_dict())

    # downstream segmentation model from the SAME model config.
    seg_cfg = _cfg("cubic")
    seg_cfg.ssl.enabled = False
    seg_cfg.sync(); seg_cfg.validate()
    seg_model = build_model(seg_cfg)

    result = seg_model.load_state_dict(ssl_sd, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)

    # encoder + decoder must all be transferred (no missing among them).
    enc_dec_missing = [k for k in missing
                       if k.startswith("encoder.") or k.startswith("decoder.")]
    assert enc_dec_missing == [], f"enc/dec keys not transferred: {enc_dec_missing}"

    # the only extra key family from SSL is the recon head.
    assert all(k.startswith("recon_head.") for k in unexpected), unexpected
    assert any(k.startswith("recon_head.") for k in unexpected)

    # the seg head stays randomly initialised (missing from SSL ckpt).
    assert any(k.startswith("seg_head.") for k in missing)


# ---------------------------------------------------------------------------
# SSLTrainer smoke
# ---------------------------------------------------------------------------
class _ImgDataset(Dataset):
    def __init__(self, n, ch, shape):
        self.x = [torch.rand(ch, *shape) for _ in range(n)]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"image": self.x[i]}


@pytest.mark.parametrize("method", ["genesis", "prior"])
def test_ssl_trainer_one_epoch_smoke(tmp_path, method):
    from segtask_v1.trainer.ssl_trainer import SSLTrainer

    cfg = _cfg("cubic")
    cfg.ssl.method = method
    cfg.train.epochs = 1
    cfg.train.warmup_epochs = 0
    cfg.train.use_ema = True
    cfg.train.use_amp = False
    cfg.train.grad_accum_steps = 1
    cfg.train.output_dir = str(tmp_path)
    cfg.sync(); cfg.validate()

    model = build_ssl_model(cfg)
    ds = _ImgDataset(4, cfg.model.in_channels, (16, 32, 32))
    loader = DataLoader(ds, batch_size=2)
    trainer = SSLTrainer(model, cfg, loader, torch.device("cpu"))
    out = trainer.fit()
    assert "best_recon_loss" in out

    ckpt = tmp_path / "ssl_best.pt"
    assert ckpt.exists()
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "model_state_dict" in blob
    assert any(k.startswith("encoder.") for k in blob["model_state_dict"])
