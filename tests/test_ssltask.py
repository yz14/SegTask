"""Tests for the standalone ssltask self-supervised pretraining package.

Covers:
- SSLConfig defaults + validate_ssl (method/recon_loss/single-view/probs guards).
- build_ssl_recon_model: encoder/decoder reuse + recon_head, shape-preserving
  forward (3D and 2.5D), out_channels == in_channels.
- GenesisCorruptor + frangi_vesselness (ported behaviour).
- **SSL -> downstream weight handoff**: encoder.*/decoder.* match exactly under
  strict=False load into a segtask build_model; recon_head.* is the only extra;
  seg_head.* is the only (head) missing.
- ImageOnlyPatchDataset: reads a temp image-only npz and yields shaped patches.
- SSLTrainer: one-epoch CPU smoke run (genesis & prior) -> loadable ssl_best.pt
  whose model_state_dict carries encoder.* keys.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from segtask_v1.config import Config as SegConfig
from segtask_v1.models.factory import build_model
from segtask_v1.trainer.checkpoint import strip_common_prefixes

from ssltask.config import SSLConfig, validate_ssl
from ssltask.data.corruptions import GenesisCorruptor
from ssltask.data.masking import (
    apply_mask_token,
    compute_grid_shape,
    densify,
    downsample_mask_to,
    make_unit_mask,
    masked_recon_loss,
    per_unit_normalize,
    sample_unit_mask,
    upsample_mask_to,
)
from ssltask.data.multicrop import (
    MultiCropGenerator,
    PairedCropGenerator,
    site_coords,
)
from ssltask.data.ssl_dataset import ImageOnlyPatchDataset, LabeledPatchDataset
from ssltask.data.vesselness import frangi_vesselness
from ssltask.eval.cls_probe import ClsProbe, macro_cls_metrics
from ssltask.eval.metrics import hd95
from ssltask.eval.pipeline import build_nested_shot_splits, run_eval_pipeline
from ssltask.eval.probe import SegProbe
from ssltask.methods import build_method
from ssltask.methods.dino_gram import DINOGramMethod
from ssltask.models.dino_modules import DINOHead, DINONet, build_dino_net
from ssltask.models.ibot_modules import build_ibot_head, dense_head_forward
from ssltask.models.jepa_modules import JEPAPredictor, build_jepa_predictor
from ssltask.models.ssl_models import (
    SSLMIMModel,
    SSLReconModel,
    build_ssl_mim_model,
    build_ssl_recon_model,
)
from ssltask.models.spark_modules import (
    MaskedInstanceNorm3d,
    SSLSparkModel,
    SparkLightDecoder,
    build_ssl_spark_model,
    enable_masked_instance_norm,
    spark_encode,
)
from ssltask.models.spark_modules import _MaskedInstanceNormMixin, _SparkVisHolder
from ssltask.trainer import SSLTrainer

try:
    from segtask_v1.config import ConfigError
except Exception:  # pragma: no cover
    ConfigError = Exception


# ---------------------------------------------------------------------------
# config helpers
# ---------------------------------------------------------------------------
def _cfg(patch_mode="cubic"):
    cfg = SegConfig()
    cfg.data.patch_mode = patch_mode
    cfg.data.patch_size = [16, 32, 32]
    cfg.data.multi_res_scales = [1.0]
    cfg.data.label_values = [0, 1]
    cfg.data.num_classes = 2
    cfg.model.backbone = "resnet"
    cfg.model.encoder_channels = [8, 16, 32]
    cfg.model.blocks_per_level = 1
    cfg.model.stem_mode = "conv3"
    cfg.sync()
    cfg.validate()
    return cfg


def _cls_cfg(patch_mode="cubic"):
    cfg = _cfg(patch_mode)
    cfg.data.label_values = [0, 1, 2]
    cfg.data.num_classes = 3
    cfg.sync()
    cfg.validate()
    return cfg


def _eval_cfg(patch_mode="cubic"):
    cfg = _cls_cfg(patch_mode)
    cfg.data.batch_size = 1
    cfg.sync()
    cfg.validate()
    return cfg


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------
def test_ssl_validate_ok():
    cfg = _cfg()
    validate_ssl(SSLConfig(), cfg)


def test_ssl_validate_prior_ok():
    cfg = _cfg()
    validate_ssl(SSLConfig(method="prior"), cfg)


def test_ssl_validate_simmim_ok():
    cfg = _cfg()
    validate_ssl(SSLConfig(method="simmim"), cfg)


@pytest.mark.parametrize("field,value", [
    ("mim_mask_ratio", 0.0),
    ("mim_mask_ratio", 1.0),
    ("mim_mask_unit", 0),
    ("mim_head_dim", -1),
])
def test_ssl_validate_simmim_rejects_bad(field, value):
    cfg = _cfg()
    ssl = SSLConfig(method="simmim")
    setattr(ssl, field, value)
    with pytest.raises(ConfigError):
        validate_ssl(ssl, cfg)


def test_ssl_validate_probe_requires_dir():
    cfg = _cfg()
    ssl = SSLConfig()
    ssl.probe_enabled = True            # but no probe_data_dir
    with pytest.raises(ConfigError):
        validate_ssl(ssl, cfg)


def test_ssl_validate_probe_accepts_2_5d():
    """2.5D 在线探针已支持（深度折通道，线性头逐 类×切片 输出）。"""
    cfg = _cfg("2_5d")
    assert cfg.model.spatial_dims == 2
    ssl = SSLConfig()
    ssl.probe_enabled = True
    ssl.probe_data_dir = "some/dir"
    validate_ssl(ssl, cfg)        # 不再抛出


@pytest.mark.parametrize("field,value", [
    ("probe_every", 0),
    ("probe_iters", 0),
    ("probe_val_ratio", 0.0),
    ("probe_val_ratio", 1.0),
    ("probe_samples_per_volume", 0),
])
def test_ssl_validate_probe_rejects_bad(field, value):
    cfg = _cfg()
    ssl = SSLConfig()
    ssl.probe_enabled = True
    ssl.probe_data_dir = "some/dir"
    setattr(ssl, field, value)
    with pytest.raises(ConfigError):
        validate_ssl(ssl, cfg)


@pytest.mark.parametrize("field,value", [
    ("method", "bogus"),
    ("recon_loss", "huber"),
    ("nonlinear_prob", 1.5),
    ("paint_count", -1),
])
def test_ssl_validate_rejects_bad(field, value):
    cfg = _cfg()
    ssl = SSLConfig()
    setattr(ssl, field, value)
    with pytest.raises(ConfigError):
        validate_ssl(ssl, cfg)


def test_ssl_validate_rejects_bad_paint_range():
    cfg = _cfg()
    ssl = SSLConfig()
    ssl.paint_block_range = [0.5, 0.2]   # lo > hi
    with pytest.raises(ConfigError):
        validate_ssl(ssl, cfg)


def test_ssl_validate_requires_single_view():
    cfg = _cfg()
    cfg.data.multi_res_scales = [1.0, 1.5]
    cfg.sync()
    with pytest.raises(ConfigError):
        validate_ssl(SSLConfig(), cfg)


@pytest.mark.parametrize("field,value", [
    ("prior_scales", []),
    ("prior_scales", [0.0, 1.0]),
    ("prior_alpha", 0.0),
])
def test_ssl_validate_prior_rejects_bad(field, value):
    cfg = _cfg()
    ssl = SSLConfig(method="prior")
    setattr(ssl, field, value)
    with pytest.raises(ConfigError):
        validate_ssl(ssl, cfg)


def test_legacy_sslconfig_pickle_shim_roundtrip(tmp_path, monkeypatch):
    import segtask_v1.config as legacy_cfg

    legacy_class = getattr(legacy_cfg, "SSLConfig")
    assert isinstance(legacy_class, type)
    with pytest.raises(AttributeError):
        getattr(legacy_cfg, "NopeNotReal")

    legacy_ref_class = type("SSLConfig", (), {})
    legacy_ref_class.__module__ = "segtask_v1.config"
    monkeypatch.setattr(legacy_cfg, "SSLConfig", legacy_ref_class, raising=False)
    ckpt = tmp_path / "legacy_sslconfig.pkl"
    torch.save({"config": legacy_ref_class()}, ckpt, pickle_protocol=2)
    monkeypatch.delattr(legacy_cfg, "SSLConfig", raising=False)

    loaded = torch.load(ckpt, weights_only=False)
    assert type(loaded["config"]) is legacy_class

    monkeypatch.setattr(legacy_cfg, "_LEGACY_MODULE_ATTRS", {}, raising=False)
    with pytest.raises(AttributeError):
        getattr(legacy_cfg, "SSLConfig")
    with pytest.raises(AttributeError):
        torch.load(ckpt, weights_only=False)


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


# ---------------------------------------------------------------------------
# model build + forward
# ---------------------------------------------------------------------------
def test_build_ssl_model_3d_forward():
    cfg = _cfg("cubic")
    model = build_ssl_recon_model(cfg).eval()
    assert isinstance(model, SSLReconModel)
    assert model.out_channels == cfg.model.in_channels
    x = torch.randn(2, cfg.model.in_channels, 16, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape


def test_build_ssl_model_2_5d_forward():
    cfg = _cfg("2_5d")
    assert cfg.model.spatial_dims == 2
    model = build_ssl_recon_model(cfg).eval()
    x = torch.randn(2, cfg.model.in_channels, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape


def test_build_ssl_model_requires_unet():
    cfg = _cfg("2_5d")
    cfg.model.arch = "adm"
    with pytest.raises(ValueError):
        build_ssl_recon_model(cfg)


# ---------------------------------------------------------------------------
# Genesis corruption
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("spatial_dims,shape", [(3, (8, 16, 16)), (2, (16, 16))])
def test_genesis_corruptor_shape_and_corrupts(spatial_dims, shape):
    corruptor = GenesisCorruptor(SSLConfig(), spatial_dims)
    x = torch.rand(2, 1, *shape)
    y = corruptor(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()
    assert not torch.allclose(y, x)          # something changed
    assert torch.equal(x, x)                 # input not mutated in place


def test_genesis_corruptor_noop_when_all_probs_zero():
    ssl = SSLConfig()
    ssl.nonlinear_prob = 0.0
    ssl.local_shuffle_prob = 0.0
    ssl.paint_prob = 0.0
    corruptor = GenesisCorruptor(ssl, 3)
    x = torch.rand(2, 1, 8, 16, 16)
    y = corruptor(x)
    assert torch.allclose(y, x)


# ---------------------------------------------------------------------------
# MIM masking utilities (SimMIM / SparK / iBOT / JEPA shared)
# ---------------------------------------------------------------------------
def test_compute_grid_shape_ceil():
    assert compute_grid_shape((96, 112, 112), 16) == (6, 7, 7)
    assert compute_grid_shape((30, 30), 16) == (2, 2)        # ceil
    assert compute_grid_shape((16, 16, 16), [16, 8, 4]) == (1, 2, 4)


@pytest.mark.parametrize("spatial,unit", [((8, 8, 8), 4), ((32, 32), 8)])
def test_sample_unit_mask_ratio_and_binary(spatial, unit):
    grid = compute_grid_shape(spatial, unit)
    n_units = int(np.prod(grid))
    mask = sample_unit_mask(4, grid, 0.6, torch.device("cpu"))
    assert mask.shape == (4, 1, *grid)
    assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})
    expected = min(max(int(round(0.6 * n_units)), 1), max(n_units - 1, 1))
    # per-sample masked-unit count is exact
    assert torch.allclose(mask.flatten(1).sum(1),
                          torch.full((4,), float(expected)))


def test_sample_unit_mask_never_all_or_none():
    grid = (2, 2, 2)  # 8 units
    mask = sample_unit_mask(16, grid, 0.99, torch.device("cpu"))
    s = mask.flatten(1).sum(1)
    assert (s >= 1).all() and (s <= 7).all()


def test_upsample_mask_to_preserves_binary_and_size():
    grid_mask = sample_unit_mask(2, (3, 4, 4), 0.5, torch.device("cpu"))
    up = upsample_mask_to(grid_mask, (12, 16, 16))
    assert up.shape == (2, 1, 12, 16, 16)
    assert set(torch.unique(up).tolist()).issubset({0.0, 1.0})


def test_apply_mask_token_replaces_only_masked():
    x = torch.zeros(1, 2, 4, 4, 4)
    mask = torch.zeros(1, 1, 4, 4, 4)
    mask[..., :2, :, :] = 1.0
    token = torch.tensor([5.0, 7.0]).view(1, 2, 1, 1, 1)
    out = apply_mask_token(x, mask, token)
    assert torch.allclose(out[0, 0, :2], torch.full((2, 4, 4), 5.0))
    assert torch.allclose(out[0, 1, :2], torch.full((2, 4, 4), 7.0))
    assert torch.allclose(out[..., 2:, :, :], torch.zeros(1, 2, 2, 4, 4))


def test_masked_recon_loss_only_on_masked():
    pred = torch.zeros(1, 1, 4, 4, 4)
    target = torch.ones(1, 1, 4, 4, 4)
    mask = torch.zeros(1, 1, 4, 4, 4)
    mask[..., :2, :, :] = 1.0           # half masked
    # only masked positions count; |0-1| = 1 over masked -> mean 1.0
    loss = masked_recon_loss(pred, target, mask, "l1")
    assert pytest.approx(float(loss), abs=1e-6) == 1.0
    # if visible region differs, loss must NOT change (masked-only)
    pred2 = pred.clone()
    pred2[..., 2:, :, :] = 99.0
    loss2 = masked_recon_loss(pred2, target, mask, "l1")
    assert pytest.approx(float(loss2), abs=1e-6) == 1.0


# ---------------------------------------------------------------------------
# SimMIM model build + forward
# ---------------------------------------------------------------------------
def test_build_ssl_mim_model_3d_forward():
    cfg = _cfg("cubic")
    model = build_ssl_mim_model(cfg).eval()
    assert isinstance(model, SSLMIMModel)
    assert model.out_channels == cfg.model.in_channels
    assert model.mask_token.shape == (1, cfg.model.in_channels, 1, 1, 1)
    x = torch.randn(2, cfg.model.in_channels, 16, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape


def test_build_ssl_mim_model_2_5d_forward():
    cfg = _cfg("2_5d")
    assert cfg.model.spatial_dims == 2
    model = build_ssl_mim_model(cfg).eval()
    assert model.mask_token.shape == (1, cfg.model.in_channels, 1, 1)
    x = torch.randn(2, cfg.model.in_channels, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape


def test_build_ssl_mim_model_requires_unet():
    cfg = _cfg("2_5d")
    cfg.model.arch = "adm"
    with pytest.raises(ValueError):
        build_ssl_mim_model(cfg)


def test_simmim_handoff_encoder_only():
    """SimMIM 仅迁移 encoder：encoder.* 全命中；decoder.*/seg_head.* missing；
    head.*/mask_token 作为 unexpected 丢弃。"""
    cfg = _cfg("cubic")
    mim_sd = strip_common_prefixes(build_ssl_mim_model(cfg).state_dict())

    seg_model = build_model(cfg)
    result = seg_model.load_state_dict(mim_sd, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)

    enc_missing = [k for k in missing if k.startswith("encoder.")]
    assert enc_missing == [], f"encoder keys not transferred: {enc_missing}"
    assert all(k.startswith("head.") or k == "mask_token" for k in unexpected), \
        unexpected
    assert any(k.startswith("decoder.") for k in missing)
    assert any(k.startswith("seg_head.") for k in missing)


# ---------------------------------------------------------------------------
# SSL -> downstream handoff (the core integration property)
# ---------------------------------------------------------------------------
def test_ssl_to_downstream_weight_handoff():
    cfg = _cfg("cubic")
    ssl_model = build_ssl_recon_model(cfg)
    ssl_sd = strip_common_prefixes(ssl_model.state_dict())

    seg_model = build_model(cfg)
    result = seg_model.load_state_dict(ssl_sd, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)

    enc_dec_missing = [k for k in missing
                       if k.startswith("encoder.") or k.startswith("decoder.")]
    assert enc_dec_missing == [], f"enc/dec keys not transferred: {enc_dec_missing}"
    assert all(k.startswith("recon_head.") for k in unexpected), unexpected
    assert any(k.startswith("recon_head.") for k in unexpected)
    assert any(k.startswith("seg_head.") for k in missing)


# ---------------------------------------------------------------------------
# image-only dataset
# ---------------------------------------------------------------------------
def test_image_only_dataset_yields_patches(tmp_path):
    # write two image-only npz volumes (int16 HU)
    paths = []
    for i in range(2):
        p = tmp_path / f"vol_{i}.npz"
        img = (np.random.rand(20, 40, 40) * 400 - 200).astype(np.int16)
        np.savez(p, image=img)
        paths.append(str(p))

    ds = ImageOnlyPatchDataset(
        paths, patch_size=[16, 32, 32],
        intensity_min=-1024.0, intensity_max=1024.0, normalize="minmax",
        samples_per_volume=3)
    assert len(ds) == 2 * 3
    sample = ds[0]
    assert set(sample.keys()) == {"image"}
    assert sample["image"].shape == (1, 16, 32, 32)
    assert sample["image"].dtype == torch.float32
    assert torch.isfinite(sample["image"]).all()


def test_image_only_dataset_2_5d_folds_depth_to_channels(tmp_path):
    """2.5D（spatial_dims=2）：深度 D 折进通道，样本形状为 (D, H, W)。"""
    p = tmp_path / "vol_0.npz"
    img = (np.random.rand(20, 40, 40) * 400 - 200).astype(np.int16)
    np.savez(p, image=img)

    ds = ImageOnlyPatchDataset(
        [str(p)], patch_size=[16, 32, 32],
        intensity_min=-1024.0, intensity_max=1024.0, normalize="minmax",
        samples_per_volume=2, spatial_dims=2)
    sample = ds[0]
    assert sample["image"].shape == (16, 32, 32)        # (C=D, H, W)
    assert sample["image"].dtype == torch.float32
    assert torch.isfinite(sample["image"]).all()


def test_build_ssl_dataloader_2_5d_batch_shape(tmp_path):
    """2.5D dataloader 产出 (B, D, H, W)，C=in_channels=patch_size[0]（单 FOV）。"""
    from ssltask.data.ssl_dataset import build_ssl_dataloader
    for i in range(2):
        img = (np.random.rand(20, 40, 40) * 400 - 200).astype(np.int16)
        np.savez(tmp_path / f"vol_{i}.npz", image=img)

    cfg = _cfg("2_5d")
    assert cfg.model.spatial_dims == 2
    assert cfg.model.in_channels == cfg.data.patch_size[0]   # single FOV
    cfg.data.npz_dir = str(tmp_path)
    cfg.data.batch_size = 2
    cfg.data.num_workers = 0

    loader = build_ssl_dataloader(cfg)
    batch = next(iter(loader))
    assert batch["image"].shape == (2, 16, 32, 32)           # (B, C=D, H, W)
    assert torch.isfinite(batch["image"]).all()


def test_build_ssl_dataloader_2_5d_rejects_multi_fov(tmp_path):
    """2.5D 多 FOV（in_channels != patch_size[0]）应被拒：image-only SSL 仅支持单 FOV。"""
    from ssltask.data.ssl_dataset import build_ssl_dataloader
    img = (np.random.rand(20, 40, 40) * 400 - 200).astype(np.int16)
    np.savez(tmp_path / "vol_0.npz", image=img)

    cfg = _cfg("2_5d")
    cfg.data.npz_dir = str(tmp_path)
    # 制造 in_channels != patch_size[0] 的多 FOV 错配（绕过 sync 直接触发守卫）。
    cfg.data.patch_size = [8, 32, 32]                        # D=8，但 in_channels 仍=16
    with pytest.raises(ValueError, match="single-FOV"):
        build_ssl_dataloader(cfg)


def _write_labeled_npz(out_dir, n=2, shape=(20, 40, 40), cls_labels=None):
    """写 n 个含 image+label 的 npz；label 为随机 {0,1}（含一些前景）。"""
    paths = []
    for i in range(n):
        p = out_dir / f"lab_{i}.npz"
        img = (np.random.rand(*shape) * 400 - 200).astype(np.int16)
        lbl = (np.random.rand(*shape) > 0.7).astype(np.int16)
        lbl[..., :5, :5] = 1                        # guarantee some foreground
        payload = {"image": img, "label": lbl}
        if cls_labels is not None:
            payload["cls_label"] = np.asarray(cls_labels[i])
        np.savez(p, **payload)
        paths.append(str(p))
    return paths


def test_labeled_dataset_yields_image_and_label(tmp_path):
    _write_labeled_npz(tmp_path, 2, (20, 40, 40))
    from ssltask.data.ssl_dataset import discover_image_npz
    ds = LabeledPatchDataset(
        discover_image_npz(str(tmp_path)),
        patch_size=[16, 32, 32],
        intensity_min=-1024.0, intensity_max=1024.0, normalize="minmax",
        samples_per_volume=2)
    sample = ds[0]
    assert set(sample.keys()) == {"image", "label"}
    assert sample["image"].shape == (1, 16, 32, 32)
    assert sample["label"].shape == (1, 16, 32, 32)
    assert torch.isfinite(sample["image"]).all()


def test_labeled_dataset_2_5d_folds_depth_to_channels(tmp_path):
    """2.5D 探针数据集：image/label 均把深度 D 折进通道，形状 (D, H, W)。"""
    _write_labeled_npz(tmp_path, 1, (20, 40, 40))
    from ssltask.data.ssl_dataset import discover_image_npz
    ds = LabeledPatchDataset(
        discover_image_npz(str(tmp_path)),
        patch_size=[16, 32, 32],
        intensity_min=-1024.0, intensity_max=1024.0, normalize="minmax",
        samples_per_volume=2, spatial_dims=2)
    sample = ds[0]
    assert sample["image"].shape == (16, 32, 32)        # (C=D, H, W)
    assert sample["label"].shape == (16, 32, 32)
    assert torch.isfinite(sample["image"]).all()


# ---------------------------------------------------------------------------
# online seg probe (§0.5)
# ---------------------------------------------------------------------------
def test_seg_probe_evaluate_returns_dice(tmp_path):
    _write_labeled_npz(tmp_path, 2, (20, 40, 40))
    cfg = _cfg("cubic")
    ssl = SSLConfig(method="genesis")
    ssl.probe_enabled = True
    ssl.probe_data_dir = str(tmp_path)
    ssl.probe_iters = 2
    ssl.probe_samples_per_volume = 2
    validate_ssl(ssl, cfg)

    probe = SegProbe(cfg, ssl, torch.device("cpu"))
    # feed a freshly-built SSL recon model's weights (encoder.* loadable strict)
    sd = build_ssl_recon_model(cfg).state_dict()
    out = probe.evaluate(sd)
    assert set(out) == {"probe_dice", "probe_hd95"}
    assert 0.0 <= out["probe_dice"] <= 1.0
    assert out["probe_hd95"] >= 0.0


def test_seg_probe_evaluate_returns_dice_2_5d(tmp_path):
    """2.5D 在线探针：折叠 D 进通道，线性头输出 num_fg*D 通道，返回合法 Dice。"""
    _write_labeled_npz(tmp_path, 2, (20, 40, 40))
    cfg = _cfg("2_5d")
    assert cfg.model.spatial_dims == 2
    ssl = SSLConfig(method="genesis")
    ssl.probe_enabled = True
    ssl.probe_data_dir = str(tmp_path)
    ssl.probe_iters = 2
    ssl.probe_samples_per_volume = 2
    validate_ssl(ssl, cfg)

    probe = SegProbe(cfg, ssl, torch.device("cpu"))
    assert probe.head_out == cfg.num_fg_classes * cfg.data.patch_size[0]
    sd = build_ssl_recon_model(cfg).state_dict()
    out = probe.evaluate(sd)
    assert set(out) == {"probe_dice", "probe_hd95"}
    assert 0.0 <= out["probe_dice"] <= 1.0
    assert out["probe_hd95"] >= 0.0


def test_seg_probe_frozen_and_finetune_gradients(tmp_path):
    _write_labeled_npz(tmp_path, 2, (20, 40, 40))
    cfg = _cfg("cubic")
    ssl = SSLConfig(method="genesis")
    ssl.probe_enabled = True
    ssl.probe_data_dir = str(tmp_path)
    ssl.probe_iters = 1
    ssl.probe_samples_per_volume = 1
    validate_ssl(ssl, cfg)
    batch = next(iter(SegProbe(cfg, ssl, torch.device("cpu")).train_loader))

    frozen_probe = SegProbe(cfg, ssl, torch.device("cpu"), finetune=False)
    frozen_probe._load_encoder(build_ssl_recon_model(cfg).state_dict())
    frozen_head = frozen_probe._build_head()
    frozen_opt = torch.optim.Adam(frozen_head.parameters(), lr=1e-3)
    frozen_probe._train_step(batch, frozen_head, frozen_opt, torch.nn.BCEWithLogitsLoss())
    assert all(p.grad is None for p in frozen_probe.encoder.parameters())
    assert any(p.grad is not None for p in frozen_head.parameters())

    finetune_probe = SegProbe(cfg, ssl, torch.device("cpu"), finetune=True)
    finetune_probe._load_encoder(build_ssl_recon_model(cfg).state_dict())
    finetune_head = finetune_probe._build_head()
    finetune_opt = torch.optim.Adam(
        [{"params": finetune_probe.encoder.parameters(), "lr": 1e-4},
         {"params": finetune_head.parameters(), "lr": 1e-3}]
    )
    finetune_probe._train_step(batch, finetune_head, finetune_opt, torch.nn.BCEWithLogitsLoss())
    assert any(p.grad is not None for p in finetune_probe.encoder.parameters())
    assert any(p.grad is not None for p in finetune_head.parameters())


def test_seg_probe_rejects_state_dict_without_encoder(tmp_path):
    _write_labeled_npz(tmp_path, 1, (20, 40, 40))
    cfg = _cfg("cubic")
    ssl = SSLConfig(method="genesis")
    ssl.probe_enabled = True
    ssl.probe_data_dir = str(tmp_path)
    ssl.probe_iters = 1
    ssl.probe_samples_per_volume = 1
    validate_ssl(ssl, cfg)
    probe = SegProbe(cfg, ssl, torch.device("cpu"))
    with pytest.raises(KeyError):
        probe.evaluate({"seg_head.conv.weight": torch.zeros(1)})


# ---------------------------------------------------------------------------
# offline metrics / sampling
# ---------------------------------------------------------------------------
def test_hd95_basic_cases():
    a = np.zeros((16, 16), dtype=np.uint8)
    a[4:8, 4:8] = 1
    assert hd95(a, a) == pytest.approx(0.0)

    b = np.zeros((16, 16), dtype=np.uint8)
    b[5:9, 4:8] = 1  # one-voxel shift along the first axis
    assert hd95(a, b) == pytest.approx(1.0, rel=0.25)

    empty = np.zeros((16, 16), dtype=np.uint8)
    assert hd95(a, empty) != hd95(a, empty)  # nan by convention
    assert hd95(empty, empty) == pytest.approx(0.0)


def test_nested_shot_splits_are_nested_and_reproducible():
    train_pool = [f"v{i}" for i in range(6)]
    a = build_nested_shot_splits(train_pool, [1, 2, 5, 99], seed=7)
    b = build_nested_shot_splits(train_pool, [1, 2, 5, 99], seed=7)
    assert a == b
    assert a[1] == a[2][:1]
    assert a[2] == a[5][:2]
    assert a[5] == a[99][:5]
    assert len(a[99]) == len(train_pool)


# ---------------------------------------------------------------------------
# online cls probe (§0.4)
# ---------------------------------------------------------------------------
def test_cls_probe_evaluate_returns_metrics_3d(tmp_path):
    _write_labeled_npz(tmp_path, 3, (20, 40, 40))
    cfg = _cls_cfg("cubic")
    ssl = SSLConfig(method="genesis")
    ssl.probe_data_dir = str(tmp_path)
    ssl.probe_iters = 2
    ssl.probe_samples_per_volume = 2
    validate_ssl(ssl, cfg)

    probe = ClsProbe(cfg, ssl, torch.device("cpu"))
    sd = build_ssl_recon_model(cfg).state_dict()
    out = probe.evaluate(sd)
    assert set(out) == {"cls_auc", "cls_f1"}
    assert 0.0 <= out["cls_auc"] <= 1.0
    assert 0.0 <= out["cls_f1"] <= 1.0


def test_cls_probe_evaluate_returns_metrics_2_5d(tmp_path):
    _write_labeled_npz(tmp_path, 3, (20, 40, 40))
    cfg = _cls_cfg("2_5d")
    assert cfg.model.spatial_dims == 2
    ssl = SSLConfig(method="genesis")
    ssl.probe_data_dir = str(tmp_path)
    ssl.probe_iters = 2
    ssl.probe_samples_per_volume = 2
    validate_ssl(ssl, cfg)

    probe = ClsProbe(cfg, ssl, torch.device("cpu"))
    sd = build_ssl_recon_model(cfg).state_dict()
    out = probe.evaluate(sd)
    assert set(out) == {"cls_auc", "cls_f1"}
    assert 0.0 <= out["cls_auc"] <= 1.0
    assert 0.0 <= out["cls_f1"] <= 1.0


def test_cls_probe_frozen_and_finetune_gradients(tmp_path):
    _write_labeled_npz(tmp_path, 2, (20, 40, 40))
    cfg = _cls_cfg("cubic")
    ssl = SSLConfig(method="genesis")
    ssl.probe_data_dir = str(tmp_path)
    ssl.probe_iters = 1
    ssl.probe_samples_per_volume = 1
    validate_ssl(ssl, cfg)
    batch = next(iter(ClsProbe(cfg, ssl, torch.device("cpu")).train_loader))

    frozen_probe = ClsProbe(cfg, ssl, torch.device("cpu"), finetune=False)
    frozen_probe._load_encoder(build_ssl_recon_model(cfg).state_dict())
    frozen_head = frozen_probe._build_head()
    frozen_opt = torch.optim.Adam(frozen_head.parameters(), lr=1e-3)
    frozen_probe._train_step(batch, frozen_head, frozen_opt, torch.nn.BCEWithLogitsLoss())
    assert all(p.grad is None for p in frozen_probe.encoder.parameters())
    assert any(p.grad is not None for p in frozen_head.parameters())

    finetune_probe = ClsProbe(cfg, ssl, torch.device("cpu"), finetune=True)
    finetune_probe._load_encoder(build_ssl_recon_model(cfg).state_dict())
    finetune_head = finetune_probe._build_head()
    finetune_opt = torch.optim.Adam(
        [{"params": finetune_probe.encoder.parameters(), "lr": 1e-4},
         {"params": finetune_head.parameters(), "lr": 1e-3}]
    )
    finetune_probe._train_step(batch, finetune_head, finetune_opt, torch.nn.BCEWithLogitsLoss())
    assert any(p.grad is not None for p in finetune_probe.encoder.parameters())
    assert any(p.grad is not None for p in finetune_head.parameters())


def test_cls_probe_encoder_only_load_and_missing_encoder_key(tmp_path):
    _write_labeled_npz(tmp_path, 1, (20, 40, 40))
    cfg = _cls_cfg("cubic")
    ssl = SSLConfig(method="genesis")
    ssl.probe_data_dir = str(tmp_path)
    ssl.probe_iters = 1
    ssl.probe_samples_per_volume = 1
    validate_ssl(ssl, cfg)
    probe = ClsProbe(cfg, ssl, torch.device("cpu"))

    sd = strip_common_prefixes(build_ssl_recon_model(cfg).state_dict())
    sd["decoder.fake"] = torch.zeros(1)
    probe._load_encoder(sd)

    with pytest.raises(KeyError):
        probe._load_encoder({"decoder.fake": torch.zeros(1)})


def test_cls_probe_metrics_helpers():
    y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
    perfect = np.array([[0.99, 0.01], [0.02, 0.98], [0.95, 0.05], [0.05, 0.95]], dtype=np.float32)
    reverse = 1.0 - perfect
    ties = np.full_like(perfect, 0.5)
    out = macro_cls_metrics(y_true, perfect)
    assert out["cls_auc"] == pytest.approx(1.0)
    assert out["cls_f1"] == pytest.approx(1.0)
    out = macro_cls_metrics(y_true, reverse)
    assert out["cls_auc"] == pytest.approx(0.0)
    out = macro_cls_metrics(y_true, ties)
    assert out["cls_auc"] == pytest.approx(0.5)


def test_offline_eval_pipeline_smoke(tmp_path):
    cls_labels = [
        np.array([1, 0], dtype=np.int16),
        np.array([0, 1], dtype=np.int16),
        np.array([1, 1], dtype=np.int16),
        np.array([0, 0], dtype=np.int16),
    ]
    _write_labeled_npz(tmp_path, 4, (12, 20, 20), cls_labels=cls_labels)
    cfg = _eval_cfg("cubic")
    ssl = SSLConfig(method="genesis")
    ssl.probe_enabled = True
    ssl.probe_data_dir = str(tmp_path)
    ssl.eval_data_dir = str(tmp_path)
    ssl.eval_shots = [1, 2]
    ssl.eval_readouts = ["linear", "finetune"]
    ssl.eval_tasks = ["seg", "cls"]
    ssl.eval_out_dir = str(tmp_path / "eval_out")
    ssl.eval_holdout_ratio = 0.25
    ssl.eval_seed = 3
    ssl.probe_iters = 1
    ssl.probe_samples_per_volume = 1
    ssl.cls_probe_iters = 1
    ssl.cls_probe_hidden_dim = 16
    ssl.cls_label_key = "cls_label"
    validate_ssl(ssl, cfg)

    ckpt = tmp_path / "pretrained.pt"
    torch.save({"model_state_dict": build_ssl_recon_model(cfg).state_dict()}, ckpt)

    res = run_eval_pipeline(
        cfg,
        ssl,
        entries=[("pretrained", ckpt), ("B2", None)],
        shots=[1, 2],
        readouts=["linear", "finetune"],
        tasks=["seg", "cls"],
        out_dir=tmp_path / "eval_out",
    )
    assert Path(res["json_path"]).exists()
    assert Path(res["csv_path"]).exists()
    assert set(res["nested"]) == {"pretrained", "B2"}
    assert len(res["rows"]) == 16
    for row in res["rows"]:
        assert row["entry"] in {"pretrained", "B2"}
        assert row["task"] in {"seg", "cls"}
        assert row["readout"] in {"linear", "finetune"}
        assert row["shots"] in {1, 2}
        if row["task"] == "seg":
            assert row["dice"] is not None and 0.0 <= row["dice"] <= 1.0
            assert row["hd95"] is not None and row["hd95"] >= 0.0
        else:
            assert row["auc"] is not None and 0.0 <= row["auc"] <= 1.0
            assert row["f1"] is not None and 0.0 <= row["f1"] <= 1.0
    seg_metrics = res["nested"]["pretrained"]["seg"]["linear"][1]
    assert set(seg_metrics) == {"probe_dice", "probe_hd95"}
    cls_metrics = res["nested"]["pretrained"]["cls"]["linear"][1]
    assert set(cls_metrics) == {"cls_auc", "cls_f1"}

    with open(res["json_path"], "r", encoding="utf-8") as f:
        blob = json.load(f)
    assert "nested" in blob and "rows" in blob
    with open(res["csv_path"], "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    assert header == ["entry", "task", "readout", "shots", "dice", "hd95", "auc", "f1"]


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


@pytest.mark.parametrize("method", ["genesis", "prior", "simmim", "byol", "moco"])
def test_ssl_trainer_one_epoch_smoke(tmp_path, method):
    cfg = _cfg("cubic")
    cfg.train.epochs = 1
    cfg.train.warmup_epochs = 0
    cfg.train.use_ema = method not in ("dino", "byol", "moco")
    cfg.train.use_amp = False
    cfg.train.grad_accum_steps = 1
    cfg.train.output_dir = str(tmp_path)
    cfg.sync()
    cfg.validate()

    ssl = SSLConfig(method=method)
    validate_ssl(ssl, cfg)

    device = torch.device("cpu")
    m = build_method(cfg, ssl, device)
    ds = _ImgDataset(4, cfg.model.in_channels, (16, 32, 32))
    loader = DataLoader(ds, batch_size=2)
    trainer = SSLTrainer(m, cfg, ssl, loader, device)
    out = trainer.fit()
    assert "best_loss" in out

    ckpt = tmp_path / "ssl_best.pt"
    assert ckpt.exists()
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "model_state_dict" in blob
    assert any(k.startswith("encoder.") for k in blob["model_state_dict"])


@pytest.mark.parametrize("method", ["genesis", "simmim", "spark", "dino", "byol", "moco"])
def test_ssl_trainer_2_5d_smoke_and_handoff(tmp_path, method):
    """2.5D（spatial_dims=2，深度折通道）端到端：单 epoch 训练 + 下游 encoder.* 交接。"""
    cfg = _cfg("2_5d")
    assert cfg.model.spatial_dims == 2
    cfg.train.epochs = 1
    cfg.train.warmup_epochs = 0
    cfg.train.use_ema = method not in ("dino", "byol", "moco")
    cfg.train.use_amp = False
    cfg.train.grad_accum_steps = 1
    cfg.train.output_dir = str(tmp_path)
    cfg.sync()
    cfg.validate()

    ssl = _dino_ssl() if method == "dino" else SSLConfig(method=method)
    validate_ssl(ssl, cfg)

    device = torch.device("cpu")
    m = build_method(cfg, ssl, device)
    ds = _ImgDataset(4, cfg.model.in_channels, (32, 32))     # (C=D, H, W)
    loader = DataLoader(ds, batch_size=2)
    trainer = SSLTrainer(m, cfg, ssl, loader, device)
    out = trainer.fit()
    assert "best_loss" in out

    ckpt = tmp_path / "ssl_best.pt"
    assert ckpt.exists()
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
    assert any(k.startswith("encoder.") for k in sd)

    # 下游交接：strict=False 载入 2.5D seg 模型，encoder.* 必须全命中。
    seg_model = build_model(cfg)
    result = seg_model.load_state_dict(strip_common_prefixes(sd), strict=False)
    enc_missing = [k for k in result.missing_keys if k.startswith("encoder.")]
    assert enc_missing == [], f"encoder keys not transferred: {enc_missing}"


def test_ssl_trainer_with_online_probe(tmp_path):
    """探针启用：fit 返回 best_probe，ssl_best.pt 由探针 Dice 选出且含 encoder.*。"""
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    _write_labeled_npz(probe_dir, 2, (20, 40, 40))
    out_dir = tmp_path / "out"

    cfg = _cfg("cubic")
    cfg.train.epochs = 1
    cfg.train.warmup_epochs = 0
    cfg.train.use_ema = True
    cfg.train.use_amp = False
    cfg.train.grad_accum_steps = 1
    cfg.train.output_dir = str(out_dir)
    cfg.sync()
    cfg.validate()

    ssl = SSLConfig(method="genesis")
    ssl.probe_enabled = True
    ssl.probe_data_dir = str(probe_dir)
    ssl.probe_every = 1
    ssl.probe_iters = 2
    ssl.probe_samples_per_volume = 2
    validate_ssl(ssl, cfg)

    device = torch.device("cpu")
    m = build_method(cfg, ssl, device)
    ds = _ImgDataset(4, cfg.model.in_channels, (16, 32, 32))
    loader = DataLoader(ds, batch_size=2)
    trainer = SSLTrainer(m, cfg, ssl, loader, device)
    out = trainer.fit()
    assert "best_probe" in out

    ckpt = out_dir / "ssl_best.pt"
    assert ckpt.exists()
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "best_probe" in blob
    assert any(k.startswith("encoder.") for k in blob["model_state_dict"])


# ---------------------------------------------------------------------------
# DINO④ — multi-crop + EMA-teacher self-distillation
# ---------------------------------------------------------------------------
def _dino_ssl(**kw):
    """Small DINO SSLConfig for fast CPU tests (tiny head / few local crops)."""
    ssl = SSLConfig(method="dino")
    ssl.dino_out_dim = 128
    ssl.dino_hidden_dim = 64
    ssl.dino_bottleneck_dim = 32
    ssl.dino_local_crops = 2
    for k, v in kw.items():
        setattr(ssl, k, v)
    return ssl


def test_ssl_validate_dino_ok():
    validate_ssl(_dino_ssl(), _cfg())


@pytest.mark.parametrize("field,value", [
    ("dino_out_dim", 0),
    ("dino_hidden_dim", 0),
    ("dino_head_layers", 0),
    ("dino_global_crops", 1),          # DINO needs >= 2 globals
    ("dino_local_crops", -1),
    ("dino_student_temp", 0.0),
    ("dino_center_momentum", 1.0),     # must be < 1
    ("dino_warmup_teacher_temp_frac", 0.0),
    ("dino_flip_prob", 1.5),
])
def test_ssl_validate_dino_rejects_bad(field, value):
    ssl = _dino_ssl()
    setattr(ssl, field, value)
    with pytest.raises(ConfigError):
        validate_ssl(ssl, _cfg())


def test_ssl_validate_dino_rejects_bad_momentum():
    ssl = _dino_ssl(dino_momentum_base=0.99, dino_momentum_final=0.9)  # base > final
    with pytest.raises(ConfigError):
        validate_ssl(ssl, _cfg())


def test_ssl_validate_dino_rejects_bad_scale():
    ssl = _dino_ssl(dino_local_scale=[0.5, 0.2])  # lo > hi
    with pytest.raises(ConfigError):
        validate_ssl(ssl, _cfg())


def test_ssl_validate_dino_rejects_bad_crop_size_length():
    ssl = _dino_ssl(dino_global_size=[16, 16])    # len 2 != spatial_dims 3
    with pytest.raises(ConfigError):
        validate_ssl(ssl, _cfg())


@pytest.mark.parametrize("spatial_dims,shape,gsz,lsz", [
    (3, (16, 32, 32), [16, 32, 32], [8, 16, 16]),
    (2, (32, 32), [32, 32], [16, 16]),
])
def test_multicrop_generator_shapes(spatial_dims, shape, gsz, lsz):
    gen = MultiCropGenerator(
        spatial_dims, global_size=gsz, local_size=lsz, n_global=2, n_local=3)
    x = torch.rand(2, 1, *shape)
    out = gen(x)
    assert len(out["global"]) == 2 and len(out["local"]) == 3
    for g in out["global"]:
        assert g.shape == (2, 1, *gsz)
    for l in out["local"]:
        assert l.shape == (2, 1, *lsz)
    flat = torch.cat([t.flatten() for t in out["global"] + out["local"]])
    assert torch.isfinite(flat).all()
    assert x.shape == (2, 1, *shape)              # input not mutated in place


def test_multicrop_generator_rejects_wrong_dims():
    gen = MultiCropGenerator(3, global_size=[16, 16, 16], local_size=[8, 8, 8])
    with pytest.raises(ValueError):
        gen(torch.rand(2, 1, 16, 16))             # 4D into a 3D generator


def test_dino_head_forward_and_frozen_g():
    head = DINOHead(in_dim=32, out_dim=64, hidden_dim=48, bottleneck_dim=16)
    x = torch.randn(4, 32)
    y = head(x)
    assert y.shape == (4, 64)
    assert head.last_layer.weight_g.requires_grad is False
    assert torch.allclose(head.last_layer.weight_g,
                          torch.ones_like(head.last_layer.weight_g))


def test_build_dino_net_3d_forward():
    cfg = _cfg("cubic")
    net = build_dino_net(cfg, out_dim=128, hidden_dim=64, bottleneck_dim=32).eval()
    assert isinstance(net, DINONet)
    x = torch.randn(2, cfg.model.in_channels, 16, 32, 32)
    with torch.no_grad():
        y = net(x)
    assert y.shape == (2, 128)


def test_build_dino_net_2_5d_forward():
    cfg = _cfg("2_5d")
    assert cfg.model.spatial_dims == 2
    net = build_dino_net(cfg, out_dim=64, hidden_dim=32, bottleneck_dim=16).eval()
    x = torch.randn(2, cfg.model.in_channels, 32, 32)
    with torch.no_grad():
        y = net(x)
    assert y.shape == (2, 64)


def test_build_dino_net_requires_unet():
    cfg = _cfg("2_5d")
    cfg.model.arch = "adm"
    with pytest.raises(ValueError):
        build_dino_net(cfg, out_dim=8)


def test_dino_method_loss_runs_and_backward():
    cfg = _cfg("cubic")
    ssl = _dino_ssl()
    validate_ssl(ssl, cfg)
    m = build_method(cfg, ssl, torch.device("cpu"))
    m.configure_schedule(10)
    m.train()
    batch = {"image": torch.rand(2, cfg.model.in_channels, 16, 32, 32)}
    loss, logs = m.compute_loss(batch)
    assert torch.isfinite(loss) and loss.requires_grad
    assert "dino_loss" in logs and "teacher_temp" in logs
    loss.backward()
    # student receives grads; frozen teacher never does.
    assert any(p.grad is not None for p in m.module.student.parameters())
    assert all(p.grad is None for p in m.module.teacher.parameters())


def test_dino_schedules_monotonic():
    cfg = _cfg()
    ssl = _dino_ssl()
    m = build_method(cfg, ssl, torch.device("cpu"))
    m.configure_schedule(100)
    m._step = 0
    t0, mom0 = m._teacher_temp(), m._momentum()
    m._step = m.total_steps
    tN, momN = m._teacher_temp(), m._momentum()
    assert t0 == pytest.approx(ssl.dino_teacher_temp_warmup)
    assert tN == pytest.approx(ssl.dino_teacher_temp)
    assert mom0 == pytest.approx(ssl.dino_momentum_base)
    assert momN == pytest.approx(ssl.dino_momentum_final)
    assert tN >= t0 and momN >= mom0


def test_dino_ema_teacher_update():
    cfg = _cfg()
    ssl = _dino_ssl()
    m = build_method(cfg, ssl, torch.device("cpu"))
    m.configure_schedule(10)
    with torch.no_grad():
        for p in m.module.student.parameters():
            p.fill_(1.0)
        for p in m.module.teacher.parameters():
            p.zero_()
    m.on_after_step(5)
    mom = m._momentum()                       # teacher <- mom*0 + (1-mom)*1
    for p in m.module.teacher.parameters():
        assert torch.allclose(p, torch.full_like(p, 1.0 - mom), atol=1e-6)


def test_dino_handoff_encoder_only():
    """DINO 仅迁移（教师）encoder：encoder.* 全命中；无 unexpected；decoder/seg_head missing。"""
    cfg = _cfg("cubic")
    m = build_method(cfg, _dino_ssl(), torch.device("cpu"))
    sd = m.export_backbone_state_dict()
    assert sd and all(k.startswith("encoder.") for k in sd)

    seg_model = build_model(cfg)
    result = seg_model.load_state_dict(sd, strict=False)
    enc_missing = [k for k in result.missing_keys if k.startswith("encoder.")]
    assert enc_missing == [], f"encoder keys not transferred: {enc_missing}"
    assert list(result.unexpected_keys) == [], list(result.unexpected_keys)
    assert any(k.startswith("decoder.") for k in result.missing_keys)
    assert any(k.startswith("seg_head.") for k in result.missing_keys)


def test_dino_trainer_smoke(tmp_path):
    cfg = _cfg("cubic")
    cfg.train.epochs = 1
    cfg.train.warmup_epochs = 0
    cfg.train.use_ema = False                 # DINO 教师即 EMA
    cfg.train.use_amp = False
    cfg.train.grad_accum_steps = 1
    cfg.train.output_dir = str(tmp_path)
    cfg.sync()
    cfg.validate()

    ssl = _dino_ssl()
    validate_ssl(ssl, cfg)

    device = torch.device("cpu")
    m = build_method(cfg, ssl, device)
    ds = _ImgDataset(4, cfg.model.in_channels, (16, 32, 32))
    loader = DataLoader(ds, batch_size=2)
    trainer = SSLTrainer(m, cfg, ssl, loader, device)
    out = trainer.fit()
    assert "best_loss" in out

    ckpt = tmp_path / "ssl_best.pt"
    assert ckpt.exists()
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "model_state_dict" in blob
    assert any(k.startswith("encoder.") for k in blob["model_state_dict"])


# ---------------------------------------------------------------------------
# B1 — BYOL-3D / MoCo-3D
# ---------------------------------------------------------------------------
def _byol_ssl(**kw):
    ssl = SSLConfig(method="byol")
    ssl.byol_proj_dim = 32
    ssl.byol_pred_hidden_dim = 48
    ssl.dino_hidden_dim = 64
    ssl.dino_global_scale = [0.5, 1.0]
    ssl.dino_local_scale = [0.15, 0.5]
    ssl.dino_flip_prob = 0.5
    ssl.dino_intensity_scale = 0.1
    ssl.dino_intensity_shift = 0.1
    for k, v in kw.items():
        setattr(ssl, k, v)
    return ssl


def _moco_ssl(**kw):
    ssl = SSLConfig(method="moco")
    ssl.moco_proj_dim = 32
    ssl.moco_queue_size = 8
    ssl.moco_temperature = 0.2
    ssl.dino_hidden_dim = 64
    ssl.dino_global_scale = [0.5, 1.0]
    ssl.dino_local_scale = [0.15, 0.5]
    ssl.dino_flip_prob = 0.5
    ssl.dino_intensity_scale = 0.1
    ssl.dino_intensity_shift = 0.1
    for k, v in kw.items():
        setattr(ssl, k, v)
    return ssl


def test_ssl_validate_byol_ok():
    validate_ssl(_byol_ssl(), _cfg())


def test_ssl_validate_moco_ok():
    validate_ssl(_moco_ssl(), _cfg())


@pytest.mark.parametrize("field,value", [
    ("byol_proj_dim", 0),
    ("byol_pred_hidden_dim", 0),
    ("byol_momentum_base", 0.0),
])
def test_ssl_validate_byol_rejects_bad(field, value):
    ssl = _byol_ssl()
    setattr(ssl, field, value)
    with pytest.raises(ConfigError):
        validate_ssl(ssl, _cfg())


@pytest.mark.parametrize("field,value", [
    ("moco_proj_dim", 0),
    ("moco_queue_size", 1),
    ("moco_temperature", 0.0),
])
def test_ssl_validate_moco_rejects_bad(field, value):
    ssl = _moco_ssl()
    setattr(ssl, field, value)
    with pytest.raises(ConfigError):
        validate_ssl(ssl, _cfg())


@pytest.mark.parametrize("method,ssl_factory", [
    ("byol", _byol_ssl),
    ("moco", _moco_ssl),
])
def test_byol_moco_loss_backward_and_handoff(method, ssl_factory):
    cfg = _cfg("cubic")
    ssl = ssl_factory()
    validate_ssl(ssl, cfg)
    m = build_method(cfg, ssl, torch.device("cpu"))
    m.configure_schedule(10)
    m.train()
    m._step = 3
    batch = {"image": torch.rand(2, cfg.model.in_channels, 16, 32, 32)}
    loss, logs = m.compute_loss(batch)
    assert torch.isfinite(loss) and loss.requires_grad
    assert any(k.endswith("_loss") for k in logs)
    loss.backward()
    if method == "byol":
        assert any(p.grad is not None for p in m.module.online.encoder.parameters())
        assert any(p.grad is not None for p in m.module.predictor.parameters())
        assert all(p.grad is None for p in m.module.target.parameters())
    else:
        assert any(p.grad is not None for p in m.module.query.encoder.parameters())
        assert all(p.grad is None for p in m.module.key.parameters())

    sd = m.export_backbone_state_dict()
    assert sd and all(k.startswith("encoder.") for k in sd)
    seg_model = build_model(cfg)
    result = seg_model.load_state_dict(sd, strict=False)
    enc_missing = [k for k in result.missing_keys if k.startswith("encoder.")]
    assert enc_missing == [], f"encoder keys not transferred: {enc_missing}"


def test_byol_moco_method_2_5d_forward_backward():
    cfg = _cfg("2_5d")
    assert cfg.model.spatial_dims == 2
    for ssl in (_byol_ssl(), _moco_ssl()):
        validate_ssl(ssl, cfg)
        m = build_method(cfg, ssl, torch.device("cpu"))
        m.configure_schedule(10)
        m.train()
        m._step = 3
        batch = {"image": torch.rand(2, cfg.model.in_channels, 32, 32)}
        loss, logs = m.compute_loss(batch)
        assert torch.isfinite(loss) and loss.requires_grad
        assert any(k.endswith("_loss") for k in logs)
        loss.backward()
        assert any(p.grad is not None for p in m.module.parameters())


def test_moco_queue_updates_and_wraps():
    cfg = _cfg("cubic")
    ssl = _moco_ssl(moco_queue_size=6)
    validate_ssl(ssl, cfg)
    m = build_method(cfg, ssl, torch.device("cpu"))
    m.configure_schedule(10)
    batch = {"image": torch.rand(2, cfg.model.in_channels, 16, 32, 32)}
    assert int(m.module.queue_ptr.item()) == 0

    _, _ = m.compute_loss(batch)
    assert int(m.module.queue_ptr.item()) == 4
    q = m.module.queue
    assert q.shape == (ssl.moco_proj_dim, ssl.moco_queue_size)
    norms = torch.linalg.norm(q, dim=0)
    assert torch.isfinite(norms).all()
    assert torch.all(norms > 0)
    _, _ = m.compute_loss(batch)
    assert int(m.module.queue_ptr.item()) == 2


def test_byol_moco_handoff_can_load_probe_encoder(tmp_path):
    _write_labeled_npz(tmp_path, 1, (20, 40, 40))
    cfg = _cls_cfg("cubic")
    ssl = SSLConfig(method="genesis")
    ssl.probe_data_dir = str(tmp_path)
    ssl.probe_iters = 1
    ssl.probe_samples_per_volume = 1
    validate_ssl(ssl, cfg)
    byol_sd = build_method(cfg, _byol_ssl(), torch.device("cpu")).export_backbone_state_dict()
    moco_sd = build_method(cfg, _moco_ssl(), torch.device("cpu")).export_backbone_state_dict()
    probe = ClsProbe(cfg, ssl, torch.device("cpu"))
    probe._load_encoder(byol_sd)
    probe._load_encoder(moco_sd)


# ---------------------------------------------------------------------------
# SparK① — mask-dense-equivalence MIM + lightweight hierarchical decoder
# ---------------------------------------------------------------------------
def _spark_ssl(**kw):
    """Small SparK SSLConfig for fast CPU tests (narrow decoder, small unit)."""
    ssl = SSLConfig(method="spark")
    ssl.spark_mask_unit = 8
    ssl.spark_decoder_dim_div = 4
    ssl.spark_decoder_min_dim = 8
    for k, v in kw.items():
        setattr(ssl, k, v)
    return ssl


# ---- config validation ----------------------------------------------------
def test_ssl_validate_spark_ok():
    validate_ssl(_spark_ssl(), _cfg())


@pytest.mark.parametrize("field,value", [
    ("spark_mask_ratio", 0.0),
    ("spark_mask_ratio", 1.0),
    ("spark_mask_unit", 0),
    ("spark_decoder_dim_div", 0),
    ("spark_decoder_min_dim", 0),
])
def test_ssl_validate_spark_rejects_bad(field, value):
    ssl = _spark_ssl()
    setattr(ssl, field, value)
    with pytest.raises(ConfigError):
        validate_ssl(ssl, _cfg())


# ---- shared MIM masking utilities SparK relies on --------------------------
def test_downsample_mask_to_preserves_binary_and_size():
    full = make_unit_mask(2, (16, 32, 32), 8, 0.6, torch.device("cpu"))
    down = downsample_mask_to(full, (4, 8, 8))
    assert down.shape == (2, 1, 4, 8, 8)
    assert set(torch.unique(down).tolist()).issubset({0.0, 1.0})
    # identity when target already matches
    assert downsample_mask_to(full, (16, 32, 32)) is full


def test_per_unit_normalize_shift_and_scale_invariant():
    # unit evenly divides spatial -> per-unit mean/std subtraction is
    # invariant to a global additive offset and a global multiplicative scale.
    x = torch.randn(2, 1, 8, 8, 8)
    base = per_unit_normalize(x, 4)
    assert base.shape == x.shape and torch.isfinite(base).all()
    shifted = per_unit_normalize(x + 5.0, 4)
    scaled = per_unit_normalize(3.0 * x, 4)
    assert torch.allclose(base, shifted, atol=1e-4)
    assert torch.allclose(base, scaled, atol=1e-4)


def test_densify_keeps_visible_fills_masked_with_embed():
    feat = torch.ones(1, 2, 4, 4, 4)
    visible = torch.zeros(1, 1, 4, 4, 4)
    visible[..., :2, :, :] = 1.0                       # first half visible
    embed = torch.tensor([5.0, 7.0]).view(1, 2, 1, 1, 1)
    out = densify(feat, visible, embed)
    assert torch.allclose(out[..., :2, :, :], torch.ones(1, 2, 2, 4, 4))
    assert torch.allclose(out[0, 0, 2:], torch.full((2, 4, 4), 5.0))
    assert torch.allclose(out[0, 1, 2:], torch.full((2, 4, 4), 7.0))


# ---- model build + shape-preserving forward --------------------------------
def test_build_ssl_spark_model_3d_forward():
    cfg = _cfg("cubic")
    model = build_ssl_spark_model(cfg, dim_div=4, min_dim=8).eval()
    assert isinstance(model, SSLSparkModel)
    assert isinstance(model.spark_decoder, SparkLightDecoder)
    assert model.out_channels == cfg.model.in_channels
    x = torch.randn(2, cfg.model.in_channels, 16, 32, 32)
    mask = make_unit_mask(2, (16, 32, 32), 8, 0.6, x.device)
    with torch.no_grad():
        y = model(x, mask)
    assert y.shape == x.shape


def test_build_ssl_spark_model_2_5d_forward():
    cfg = _cfg("2_5d")
    assert cfg.model.spatial_dims == 2
    model = build_ssl_spark_model(cfg, dim_div=4, min_dim=8).eval()
    x = torch.randn(2, cfg.model.in_channels, 32, 32)
    mask = make_unit_mask(2, (32, 32), 8, 0.6, x.device)
    with torch.no_grad():
        y = model(x, mask)
    assert y.shape == x.shape


def test_build_ssl_spark_model_requires_unet():
    cfg = _cfg("2_5d")
    cfg.model.arch = "adm"
    with pytest.raises(ValueError):
        build_ssl_spark_model(cfg)


def test_spark_decoder_is_lighter_than_encoder():
    cfg = _cfg("cubic")
    model = build_ssl_spark_model(cfg, dim_div=4, min_dim=8)
    pc = model.param_count()
    assert pc["spark_decoder"] < pc["encoder"]
    assert pc["total"] == pc["encoder"] + pc["spark_decoder"]


# ---- mask-dense equivalence (the core SparK property) ----------------------
def test_spark_encode_full_density_equals_dense_forward():
    """mask=0 (fully visible) => gated encode is bit-identical to plain dense
    encoder.forward — this is what lets pretrain(sparse)/downstream(dense) share
    one set of encoder weights with zero conversion."""
    cfg = _cfg("cubic")
    encoder = build_ssl_spark_model(cfg, dim_div=4, min_dim=8).encoder.eval()
    x = torch.randn(2, cfg.model.in_channels, 16, 32, 32)
    with torch.no_grad():
        dense = encoder(x)
        feats, vis = spark_encode(encoder, x, torch.zeros(2, 1, 16, 32, 32))
    assert len(feats) == len(dense)
    for f, d in zip(feats, dense):
        assert f.shape == d.shape
        assert torch.allclose(f, d, atol=1e-6)
    # full-density visibility masks are all-ones at every scale
    for v in vis:
        assert torch.equal(v, torch.ones_like(v))


def test_spark_encode_gates_masked_positions_to_zero():
    """At every scale, masked positions (visible==0) must be exactly zero so the
    receptive field never leaks into the occluded region."""
    cfg = _cfg("cubic")
    encoder = build_ssl_spark_model(cfg, dim_div=4, min_dim=8).encoder.eval()
    x = torch.randn(2, cfg.model.in_channels, 16, 32, 32)
    mask = make_unit_mask(2, (16, 32, 32), 8, 0.6, x.device)
    with torch.no_grad():
        feats, vis = spark_encode(encoder, x, mask)
    for f, v in zip(feats, vis):
        assert f.shape[2:] == v.shape[2:]
        masked = (v == 0)
        assert float((f * masked).abs().max()) == 0.0


# ---- masked InstanceNorm (visible-only statistics) --------------------------
def test_spark_masked_norm_preserves_state_dict_and_converts_layers():
    """Conversion keeps encoder.* keys/values identical (downstream handoff
    unaffected) and actually swaps in Masked InstanceNorm layers."""
    cfg = _cfg("cubic")
    ref = build_ssl_spark_model(cfg, dim_div=4, min_dim=8,
                                masked_norm=False).encoder
    enc = build_ssl_spark_model(cfg, dim_div=4, min_dim=8,
                                masked_norm=True).encoder
    assert set(enc.state_dict()) == set(ref.state_dict())
    assert any(isinstance(m_, _MaskedInstanceNormMixin) for m_ in enc.modules())
    assert not any(isinstance(m_, _MaskedInstanceNormMixin)
                   for m_ in ref.modules())
    # idempotent: re-running converts nothing new but keeps a positive count.
    n1 = enable_masked_instance_norm(enc)
    n2 = enable_masked_instance_norm(enc)
    assert n1 == n2 > 0


def test_spark_masked_norm_dense_path_matches_native():
    """holder empty (dense forward: probe/downstream/DINO branch) => output is
    numerically the native InstanceNorm output."""
    torch.manual_seed(0)
    native = torch.nn.InstanceNorm3d(4, affine=True)
    torch.nn.init.normal_(native.weight)
    torch.nn.init.normal_(native.bias)
    masked = MaskedInstanceNorm3d(4, affine=True)
    masked.load_state_dict(native.state_dict())
    masked._spark_holder = _SparkVisHolder()          # vis_full=None
    x = torch.randn(2, 4, 8, 8, 8)
    assert torch.allclose(masked(x), native(x), atol=1e-6)


def test_spark_masked_norm_visible_stats_and_gating():
    """With a mask installed, statistics come from visible positions only
    (visible mean~0 / var~1 per sample-channel) and masked positions are gated
    to zero — unlike the polluted native InstanceNorm on zeroed input."""
    torch.manual_seed(0)
    norm = MaskedInstanceNorm3d(4, affine=False)
    holder = _SparkVisHolder()
    norm._spark_holder = holder
    mask = make_unit_mask(2, (8, 8, 8), 4, 0.5, torch.device("cpu"))
    vis = 1.0 - mask                                  # (B,1,8,8,8), 1=visible
    x = torch.randn(2, 4, 8, 8, 8) * 3.0 + 5.0
    holder.vis_full = vis
    y = norm(x * vis)                                 # spark zeroes masked input
    holder.vis_full = None
    assert float((y * mask).abs().max()) == 0.0       # gated output
    b = vis.bool().expand_as(y)
    for s in range(2):
        for c in range(4):
            v = y[s, c][b[s, c]]
            assert abs(float(v.mean())) < 1e-4
            assert abs(float(v.var(unbiased=False)) - 1.0) < 1e-3
    # the native norm on the same zeroed input is visibly polluted:
    z = torch.nn.InstanceNorm3d(4, affine=False)(x * vis)
    vm = float((z * vis).sum() / (vis.sum() * 4))
    assert abs(vm) > 0.05, "expected polluted mean shift on native norm"


def test_spark_encode_rejects_hierarchical_stem():
    cfg = _cfg("cubic")
    encoder = build_ssl_spark_model(cfg, dim_div=4, min_dim=8).encoder
    encoder.aux_fuse = torch.nn.ModuleDict(            # simulate hierarchical stem
        {"dummy": torch.nn.Identity()})
    x = torch.randn(1, cfg.model.in_channels, 16, 32, 32)
    with pytest.raises(NotImplementedError):
        spark_encode(encoder, x, torch.zeros(1, 1, 16, 32, 32))


# ---- method: loss runs + grads flow to the shared encoder ------------------
@pytest.mark.parametrize("norm_pix", [True, False])
def test_spark_method_loss_runs_and_backward(norm_pix):
    cfg = _cfg("cubic")
    ssl = _spark_ssl(spark_norm_pix=norm_pix)
    validate_ssl(ssl, cfg)
    m = build_method(cfg, ssl, torch.device("cpu"))
    m.train()
    batch = {"image": torch.randn(2, cfg.model.in_channels, 16, 32, 32)}
    loss, logs = m.compute_loss(batch)
    assert torch.isfinite(loss) and loss.requires_grad
    assert "recon_loss" in logs and "mask_ratio" in logs
    assert pytest.approx(logs["mask_ratio"], abs=1e-9) == ssl.spark_mask_ratio
    loss.backward()
    # the shared encoder (the downstream transfer target) must receive grads.
    assert any(p.grad is not None for p in m.module.encoder.parameters())


# ---- SSL -> downstream handoff (encoder.* only) ----------------------------
def test_spark_handoff_encoder_only():
    """SparK transfers only the encoder: encoder.* all hit; spark_decoder.* are
    the only unexpected keys (decoder used-then-discarded); decoder.*/seg_head.*
    remain missing on the downstream seg model."""
    cfg = _cfg("cubic")
    m = build_method(cfg, _spark_ssl(), torch.device("cpu"))
    sd = strip_common_prefixes(m.export_backbone_state_dict())
    assert any(k.startswith("encoder.") for k in sd)

    seg_model = build_model(cfg)
    result = seg_model.load_state_dict(sd, strict=False)
    enc_missing = [k for k in result.missing_keys if k.startswith("encoder.")]
    assert enc_missing == [], f"encoder keys not transferred: {enc_missing}"
    assert all(k.startswith("spark_decoder.") for k in result.unexpected_keys), \
        list(result.unexpected_keys)
    assert any(k.startswith("spark_decoder.") for k in result.unexpected_keys)
    assert any(k.startswith("decoder.") for k in result.missing_keys)
    assert any(k.startswith("seg_head.") for k in result.missing_keys)


# ---- decoder_mode='seg': warm-start the real downstream decoder ------------
def test_spark_validate_rejects_bad_decoder_mode():
    cfg = _cfg("cubic")
    ssl = _spark_ssl()
    ssl.spark_decoder_mode = "bogus"
    with pytest.raises(ConfigError):
        validate_ssl(ssl, cfg)


def test_spark_seg_decoder_loss_runs_and_backward():
    cfg = _cfg("cubic")
    ssl = _spark_ssl()
    ssl.spark_decoder_mode = "seg"
    validate_ssl(ssl, cfg)
    m = build_method(cfg, ssl, torch.device("cpu"))
    m.train()
    loss, logs = m.compute_loss(
        {"image": torch.randn(2, cfg.model.in_channels, 16, 32, 32)})
    assert torch.isfinite(loss) and loss.requires_grad
    loss.backward()
    # both the encoder and the transferable real decoder must receive grads.
    assert any(p.grad is not None for p in m.module.encoder.parameters())
    assert any(p.grad is not None for p in m.module.decoder.parameters())


def test_spark_seg_decoder_handoff_warm_starts_decoder():
    """decoder_mode='seg' transfers encoder.* AND decoder.*; only seg_head.*
    stays missing; recon_head/mask_embed are the only unexpected keys."""
    cfg = _cfg("cubic")
    ssl = _spark_ssl()
    ssl.spark_decoder_mode = "seg"
    m = build_method(cfg, ssl, torch.device("cpu"))
    sd = strip_common_prefixes(m.export_backbone_state_dict())

    seg_model = build_model(cfg)
    result = seg_model.load_state_dict(sd, strict=False)
    missing = list(result.missing_keys)
    assert all(k.startswith("seg_head.") for k in missing), missing
    assert any(k.startswith("seg_head.") for k in missing)
    assert all(k.startswith(("recon_head.", "mask_embed."))
               for k in result.unexpected_keys), list(result.unexpected_keys)


# ---- one-epoch CPU trainer smoke -------------------------------------------
def test_spark_trainer_smoke(tmp_path):
    cfg = _cfg("cubic")
    cfg.train.epochs = 1
    cfg.train.warmup_epochs = 0
    cfg.train.use_ema = True
    cfg.train.use_amp = False
    cfg.train.grad_accum_steps = 1
    cfg.train.output_dir = str(tmp_path)
    cfg.sync()
    cfg.validate()

    ssl = _spark_ssl()
    validate_ssl(ssl, cfg)

    device = torch.device("cpu")
    m = build_method(cfg, ssl, device)
    ds = _ImgDataset(4, cfg.model.in_channels, (16, 32, 32))
    loader = DataLoader(ds, batch_size=2)
    trainer = SSLTrainer(m, cfg, ssl, loader, device)
    out = trainer.fit()
    assert "best_loss" in out

    ckpt = tmp_path / "ssl_best.pt"
    assert ckpt.exists()
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "model_state_dict" in blob
    assert any(k.startswith("encoder.") for k in blob["model_state_dict"])


# ---------------------------------------------------------------------------
# JEPA-3D⑦ — latent-space masked prediction (context/EMA-target + predictor)
# ---------------------------------------------------------------------------
def _jepa_ssl(**kw):
    """Small JEPA SSLConfig for fast CPU tests."""
    ssl = SSLConfig(method="jepa")
    ssl.jepa_mask_unit = 8
    ssl.jepa_mask_ratio = 0.5
    ssl.jepa_predictor_depth = 2
    for k, v in kw.items():
        setattr(ssl, k, v)
    return ssl


# ---- config validation -----------------------------------------------------
def test_ssl_validate_jepa_ok():
    validate_ssl(_jepa_ssl(), _cfg())


@pytest.mark.parametrize("field,value", [
    ("jepa_mask_ratio", 0.0),
    ("jepa_mask_ratio", 1.0),
    ("jepa_mask_unit", 0),
    ("jepa_predictor_depth", 0),
    ("jepa_predictor_hidden", -1),
    ("jepa_var_weight", -1.0),
    ("jepa_cov_weight", -1.0),
    ("jepa_feature_level", 99),                # out of encoder_channels range
])
def test_ssl_validate_jepa_rejects_bad(field, value):
    ssl = _jepa_ssl()
    setattr(ssl, field, value)
    with pytest.raises(ConfigError):
        validate_ssl(ssl, _cfg())


def test_ssl_validate_jepa_rejects_bad_momentum():
    ssl = _jepa_ssl(jepa_momentum_base=0.99, jepa_momentum_final=0.9)  # base>final
    with pytest.raises(ConfigError):
        validate_ssl(ssl, _cfg())


# ---- predictor module ------------------------------------------------------
@pytest.mark.parametrize("shape", [(2, 8, 4, 4, 4), (2, 8, 8, 8)])
def test_jepa_predictor_preserves_shape(shape):
    pred = JEPAPredictor(channels=shape[1], hidden=16, depth=2,
                         spatial_dims=len(shape) - 2)
    x = torch.randn(*shape)
    y = pred(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_build_jepa_predictor_hidden_defaults_to_channels():
    cfg = _cfg("cubic")
    ch = int(cfg.model.encoder_channels[-1])
    pred = build_jepa_predictor(cfg, channels=ch, hidden=0, depth=2)
    # hidden==0 -> uses `ch`; final 1x1 conv maps hidden->ch.
    assert pred.out.in_channels == ch and pred.out.out_channels == ch


# ---- loss / gradients ------------------------------------------------------
def test_jepa_method_loss_runs_and_backward():
    cfg = _cfg("cubic")
    ssl = _jepa_ssl()
    validate_ssl(ssl, cfg)
    m = build_method(cfg, ssl, torch.device("cpu"))
    m.configure_schedule(10)
    m.train()
    m._step = 3
    batch = {"image": torch.rand(2, cfg.model.in_channels, 16, 32, 32)}
    loss, logs = m.compute_loss(batch)
    assert torch.isfinite(loss) and loss.requires_grad
    assert "jepa_loss" in logs and "ema_momentum" in logs
    loss.backward()
    # context encoder + predictor + mask_token learn; EMA target never does.
    assert any(p.grad is not None for p in m.module.context_encoder.parameters())
    assert any(p.grad is not None for p in m.module.predictor.parameters())
    assert m.module.mask_token.grad is not None
    assert all(p.grad is None for p in m.module.target_encoder.parameters())


def test_jepa_vicreg_terms_only_when_enabled():
    cfg = _cfg("cubic")
    batch = {"image": torch.rand(2, cfg.model.in_channels, 16, 32, 32)}

    off = build_method(cfg, _jepa_ssl(), torch.device("cpu"))
    off.configure_schedule(10); off.train(); off._step = 1
    _, logs_off = off.compute_loss(batch)
    assert "vicreg_var" not in logs_off and "vicreg_cov" not in logs_off

    on = build_method(
        cfg, _jepa_ssl(jepa_var_weight=1.0, jepa_cov_weight=1.0),
        torch.device("cpu"))
    on.configure_schedule(10); on.train(); on._step = 1
    _, logs_on = on.compute_loss(batch)
    assert logs_on["vicreg_var"] >= 0.0 and logs_on["vicreg_cov"] >= 0.0


def test_jepa_target_is_stop_grad_and_initialized_to_context():
    """Target encoder starts == context encoder and stays grad-free (EMA only)."""
    cfg = _cfg("cubic")
    m = build_method(cfg, _jepa_ssl(), torch.device("cpu"))
    for pc, pt in zip(m.module.context_encoder.parameters(),
                      m.module.target_encoder.parameters()):
        assert torch.allclose(pc, pt)
        assert pt.requires_grad is False


# ---- EMA target update -----------------------------------------------------
def test_jepa_ema_target_update():
    cfg = _cfg()
    m = build_method(cfg, _jepa_ssl(), torch.device("cpu"))
    m.configure_schedule(10)
    with torch.no_grad():
        for p in m.module.context_encoder.parameters():
            p.fill_(1.0)
        for p in m.module.target_encoder.parameters():
            p.zero_()
    m.on_after_step(5)
    mom = m._momentum()                       # target <- mom*0 + (1-mom)*1
    for p in m.module.target_encoder.parameters():
        assert torch.allclose(p, torch.full_like(p, 1.0 - mom), atol=1e-6)


def test_jepa_momentum_monotonic():
    cfg = _cfg()
    ssl = _jepa_ssl()
    m = build_method(cfg, ssl, torch.device("cpu"))
    m.configure_schedule(100)
    m._step = 0
    mom0 = m._momentum()
    m._step = m.total_steps
    momN = m._momentum()
    assert mom0 == pytest.approx(ssl.jepa_momentum_base)
    assert momN == pytest.approx(ssl.jepa_momentum_final)
    assert momN >= mom0


# ---- SSL -> downstream handoff (EMA target encoder -> encoder.* only) -------
def test_jepa_handoff_encoder_only():
    cfg = _cfg("cubic")
    m = build_method(cfg, _jepa_ssl(), torch.device("cpu"))
    sd = m.export_backbone_state_dict()
    assert sd and all(k.startswith("encoder.") for k in sd)

    seg_model = build_model(cfg)
    result = seg_model.load_state_dict(sd, strict=False)
    enc_missing = [k for k in result.missing_keys if k.startswith("encoder.")]
    assert enc_missing == [], f"encoder keys not transferred: {enc_missing}"
    assert list(result.unexpected_keys) == [], list(result.unexpected_keys)
    assert any(k.startswith("decoder.") for k in result.missing_keys)
    assert any(k.startswith("seg_head.") for k in result.missing_keys)


def test_jepa_handoff_exports_target_not_context():
    """Export must reflect the EMA target encoder, not the context encoder."""
    cfg = _cfg("cubic")
    m = build_method(cfg, _jepa_ssl(), torch.device("cpu"))
    with torch.no_grad():
        for p in m.module.context_encoder.parameters():
            p.fill_(1.0)
        for p in m.module.target_encoder.parameters():
            p.fill_(0.25)
    sd = m.export_backbone_state_dict()
    sample = next(v for v in sd.values() if v.numel() > 0)
    assert torch.allclose(sample, torch.full_like(sample, 0.25))


# ---- 2D / 2.5D shape support ----------------------------------------------
def test_jepa_method_2_5d_forward_backward():
    cfg = _cfg("2_5d")
    assert cfg.model.spatial_dims == 2
    ssl = _jepa_ssl(jepa_mask_unit=8)
    validate_ssl(ssl, cfg)
    m = build_method(cfg, ssl, torch.device("cpu"))
    m.configure_schedule(10); m.train(); m._step = 3
    batch = {"image": torch.rand(2, cfg.model.in_channels, 32, 32)}
    loss, _ = m.compute_loss(batch)
    assert torch.isfinite(loss) and loss.requires_grad
    loss.backward()
    assert any(p.grad is not None for p in m.module.context_encoder.parameters())


# ---- one-epoch CPU trainer smoke -------------------------------------------
def test_jepa_trainer_smoke(tmp_path):
    cfg = _cfg("cubic")
    cfg.train.epochs = 1
    cfg.train.warmup_epochs = 0
    cfg.train.use_ema = False                 # JEPA target is already EMA
    cfg.train.use_amp = False
    cfg.train.grad_accum_steps = 1
    cfg.train.output_dir = str(tmp_path)
    cfg.sync()
    cfg.validate()

    ssl = _jepa_ssl(jepa_var_weight=1.0, jepa_cov_weight=1.0)
    validate_ssl(ssl, cfg)

    device = torch.device("cpu")
    m = build_method(cfg, ssl, device)
    ds = _ImgDataset(4, cfg.model.in_channels, (16, 32, 32))
    loader = DataLoader(ds, batch_size=2)
    trainer = SSLTrainer(m, cfg, ssl, loader, device)
    out = trainer.fit()
    assert "best_loss" in out

    ckpt = tmp_path / "ssl_best.pt"
    assert ckpt.exists()
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert any(k.startswith("encoder.") for k in blob["model_state_dict"])


# ---------------------------------------------------------------------------
# DINO+Gram⑤ — DINO self-distillation + Gram anchoring on dense features
# ---------------------------------------------------------------------------
def _dino_gram_ssl(**kw):
    """Small DINO+Gram SSLConfig for fast CPU tests."""
    ssl = SSLConfig(method="dino_gram")
    ssl.dino_out_dim = 128
    ssl.dino_hidden_dim = 64
    ssl.dino_bottleneck_dim = 32
    ssl.dino_local_crops = 2
    ssl.dino_gram_start_frac = 0.3
    ssl.dino_gram_refresh_steps = 1
    for k, v in kw.items():
        setattr(ssl, k, v)
    return ssl


# ---- config validation (reuses all dino_* checks + gram-specific) ----------
def test_ssl_validate_dino_gram_ok():
    validate_ssl(_dino_gram_ssl(), _cfg())


def test_ssl_validate_dino_gram_inherits_dino_checks():
    ssl = _dino_gram_ssl(dino_global_crops=1)      # DINO rule: needs >= 2 globals
    with pytest.raises(ConfigError):
        validate_ssl(ssl, _cfg())


@pytest.mark.parametrize("field,value", [
    ("dino_gram_weight", -1.0),
    ("dino_gram_start_frac", 1.5),
    ("dino_gram_start_frac", -0.1),
    ("dino_gram_refresh_steps", 0),
    ("dino_gram_feature_level", 99),               # out of encoder_channels range
])
def test_ssl_validate_dino_gram_rejects_bad(field, value):
    ssl = _dino_gram_ssl()
    setattr(ssl, field, value)
    with pytest.raises(ConfigError):
        validate_ssl(ssl, _cfg())


# ---- gram weight gating ----------------------------------------------------
def test_dino_gram_off_before_start_frac():
    """progress < start_frac => gram disabled: weight==0 and gram_loss==0."""
    cfg = _cfg("cubic")
    ssl = _dino_gram_ssl(dino_gram_start_frac=0.5)
    validate_ssl(ssl, cfg)
    m = build_method(cfg, ssl, torch.device("cpu"))
    m.configure_schedule(10)
    m.train()
    m._step = 0                                     # progress 0 < 0.5
    batch = {"image": torch.rand(2, cfg.model.in_channels, 16, 32, 32)}
    loss, logs = m.compute_loss(batch)
    assert logs["gram_weight"] == 0.0
    assert logs["gram_loss"] == 0.0
    assert torch.isfinite(loss) and loss.requires_grad


def test_dino_gram_on_after_start_frac_and_backward():
    """progress >= start_frac => gram active; once student diverges from the
    gram-teacher snapshot the term is > 0; grads reach only the student."""
    cfg = _cfg("cubic")
    ssl = _dino_gram_ssl(dino_gram_start_frac=0.3, dino_gram_weight=2.0)
    validate_ssl(ssl, cfg)
    m = build_method(cfg, ssl, torch.device("cpu"))
    m.configure_schedule(10)
    m.train()
    # perturb the (frozen) gram-teacher so student != snapshot => non-zero Gram.
    with torch.no_grad():
        for p in m.module.gram_teacher.parameters():
            p.add_(0.5)
    m._step = 5                                     # progress 0.5 >= 0.3
    batch = {"image": torch.rand(2, cfg.model.in_channels, 16, 32, 32)}
    loss, logs = m.compute_loss(batch)
    assert logs["gram_weight"] == pytest.approx(2.0)
    assert logs["gram_loss"] > 0.0
    loss.backward()
    assert any(p.grad is not None for p in m.module.student.parameters())
    assert all(p.grad is None for p in m.module.teacher.parameters())
    assert all(p.grad is None for p in m.module.gram_teacher.parameters())


def test_dino_gram_loss_zero_when_snapshot_matches_student():
    """gram-teacher initialized == teacher == student => identical dense feats =>
    Gram term is exactly 0 even when the schedule has it enabled."""
    cfg = _cfg("cubic")
    ssl = _dino_gram_ssl(dino_gram_start_frac=0.0)
    validate_ssl(ssl, cfg)
    m = build_method(cfg, ssl, torch.device("cpu"))
    m.configure_schedule(10)
    m.train()
    m._step = 5
    _, logs = m.compute_loss(
        {"image": torch.rand(2, cfg.model.in_channels, 16, 32, 32)})
    assert logs["gram_weight"] > 0.0
    assert logs["gram_loss"] == pytest.approx(0.0, abs=1e-6)


# ---- gram matrix property --------------------------------------------------
def test_dino_gram_matrix_is_normalized_gram():
    """Gram matrix diagonal == 1 (unit L2 rows) and entries in [-1, 1]."""
    feat = torch.randn(2, 8, 4, 4, 4)
    g = DINOGramMethod._gram_matrix(feat)
    assert g.shape == (2, 64, 64)
    diag = torch.diagonal(g, dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-5)
    assert float(g.max()) <= 1.0 + 1e-4 and float(g.min()) >= -1.0 - 1e-4


# ---- memory-safe gram distance (C×C identity path) -------------------------
def test_dino_gram_sq_dist_matches_materialized_gram():
    """_gram_sq_dist == mean((G_s-G_t)**2) for both the N<=C (materialized)
    and N>C (C×C identity, no (B,N,N) allocation) branches."""
    torch.manual_seed(0)
    for shape in [(2, 3, 5, 5, 5),    # N=125 > C=3  -> identity path
                  (2, 8, 2, 2)]:      # N=4  <= C=8  -> materialized path
        s = torch.randn(*shape).double()
        t = torch.randn(*shape).double()
        ref = (DINOGramMethod._gram_matrix(s)
               - DINOGramMethod._gram_matrix(t)).pow(2).mean()
        out = DINOGramMethod._gram_sq_dist(s, t)
        assert torch.allclose(out, ref, atol=1e-10), (shape, float(out), float(ref))


def test_dino_gram_sq_dist_identity_path_backward():
    """Gradients flow through the student side on the N>C identity path."""
    torch.manual_seed(0)
    s = torch.randn(2, 3, 4, 4, 4, requires_grad=True)   # N=64 > C=3
    t = torch.randn(2, 3, 4, 4, 4)
    out = DINOGramMethod._gram_sq_dist(s, t)
    out.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()


# ---- periodic gram-teacher refresh ----------------------------------------
def test_dino_gram_teacher_refresh_on_interval():
    cfg = _cfg("cubic")
    ssl = _dino_gram_ssl(dino_gram_refresh_steps=2)
    validate_ssl(ssl, cfg)
    m = build_method(cfg, ssl, torch.device("cpu"))
    m.configure_schedule(10)
    # student != teacher so the EMA update actually moves the teacher.
    with torch.no_grad():
        for p in m.module.student.parameters():
            p.fill_(1.0)
        for p in m.module.teacher.parameters():
            p.zero_()
        for p in m.module.gram_teacher.parameters():
            p.zero_()

    m.on_after_step(1)                              # 1 % 2 != 0 -> no refresh
    moved = any(not torch.allclose(g, t) for g, t in zip(
        m.module.gram_teacher.parameters(), m.module.teacher.parameters()))
    assert moved, "teacher EMA moved but gram-teacher refreshed off-interval"

    m.on_after_step(2)                              # 2 % 2 == 0 -> refresh
    assert all(torch.allclose(g, t) for g, t in zip(
        m.module.gram_teacher.parameters(), m.module.teacher.parameters()))


# ---- SSL -> downstream handoff (encoder.* only, same as DINO) --------------
def test_dino_gram_handoff_encoder_only():
    cfg = _cfg("cubic")
    m = build_method(cfg, _dino_gram_ssl(), torch.device("cpu"))
    sd = m.export_backbone_state_dict()
    assert sd and all(k.startswith("encoder.") for k in sd)

    seg_model = build_model(cfg)
    result = seg_model.load_state_dict(sd, strict=False)
    enc_missing = [k for k in result.missing_keys if k.startswith("encoder.")]
    assert enc_missing == [], f"encoder keys not transferred: {enc_missing}"
    assert list(result.unexpected_keys) == [], list(result.unexpected_keys)
    assert any(k.startswith("decoder.") for k in result.missing_keys)
    assert any(k.startswith("seg_head.") for k in result.missing_keys)


# ---- one-epoch CPU trainer smoke (covers gram-on via start_frac=0) ---------
def test_dino_gram_trainer_smoke(tmp_path):
    cfg = _cfg("cubic")
    cfg.train.epochs = 1
    cfg.train.warmup_epochs = 0
    cfg.train.use_ema = False                       # DINO teacher is already EMA
    cfg.train.use_amp = False
    cfg.train.grad_accum_steps = 1
    cfg.train.output_dir = str(tmp_path)
    cfg.sync()
    cfg.validate()

    ssl = _dino_gram_ssl(dino_gram_start_frac=0.0)  # gram active from step 0
    validate_ssl(ssl, cfg)

    device = torch.device("cpu")
    m = build_method(cfg, ssl, device)
    ds = _ImgDataset(4, cfg.model.in_channels, (16, 32, 32))
    loader = DataLoader(ds, batch_size=2)
    trainer = SSLTrainer(m, cfg, ssl, loader, device)
    out = trainer.fit()
    assert "best_loss" in out

    ckpt = tmp_path / "ssl_best.pt"
    assert ckpt.exists()
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert any(k.startswith("encoder.") for k in blob["model_state_dict"])


# ---------------------------------------------------------------------------
# iBOT/DINOv2⑥ — DINO global self-distillation + iBOT masked dense prediction
# ---------------------------------------------------------------------------
def _ibot_ssl(**kw):
    """Small iBOT SSLConfig for fast CPU tests (tiny heads / few local crops)."""
    ssl = SSLConfig(method="ibot")
    ssl.dino_out_dim = 128
    ssl.dino_hidden_dim = 64
    ssl.dino_bottleneck_dim = 32
    ssl.dino_local_crops = 2
    ssl.ibot_mask_unit = 8
    ssl.ibot_mask_ratio = 0.5
    for k, v in kw.items():
        setattr(ssl, k, v)
    return ssl


# ---- config validation (reuses all dino_* checks + ibot-specific) ----------
def test_ssl_validate_ibot_ok():
    validate_ssl(_ibot_ssl(), _cfg())


def test_ssl_validate_ibot_inherits_dino_checks():
    ssl = _ibot_ssl(dino_global_crops=1)            # DINO rule: needs >= 2 globals
    with pytest.raises(ConfigError):
        validate_ssl(ssl, _cfg())


@pytest.mark.parametrize("field,value", [
    ("ibot_weight", -1.0),
    ("ibot_mask_ratio", 0.0),
    ("ibot_mask_ratio", 1.0),
    ("ibot_mask_unit", 0),
    ("ibot_out_dim", -1),
    ("ibot_feature_level", 99),                     # out of encoder_channels range
])
def test_ssl_validate_ibot_rejects_bad(field, value):
    ssl = _ibot_ssl()
    setattr(ssl, field, value)
    with pytest.raises(ConfigError):
        validate_ssl(ssl, _cfg())


# ---- dense projection head -------------------------------------------------
@pytest.mark.parametrize("shape", [(2, 16, 4, 4, 4), (2, 16, 8, 8)])
def test_dense_head_forward_shape(shape):
    head = build_ibot_head(in_dim=shape[1], out_dim=48, hidden_dim=32,
                           bottleneck_dim=16, n_layers=2)
    feat = torch.randn(*shape)
    out = dense_head_forward(head, feat)
    n = 1
    for s in shape[2:]:
        n *= s
    assert out.shape == (shape[0], n, 48)
    assert torch.isfinite(out).all()


# ---- loss / gradients ------------------------------------------------------
def test_ibot_method_loss_runs_and_backward():
    cfg = _cfg("cubic")
    ssl = _ibot_ssl(ibot_out_dim=48)
    validate_ssl(ssl, cfg)
    m = build_method(cfg, ssl, torch.device("cpu"))
    m.configure_schedule(10)
    m.train()
    m._step = 3
    batch = {"image": torch.rand(2, cfg.model.in_channels, 16, 32, 32)}
    loss, logs = m.compute_loss(batch)
    assert torch.isfinite(loss) and loss.requires_grad
    assert "dino_loss" in logs and "ibot_loss" in logs
    assert logs["ibot_loss"] > 0.0                  # random init => non-zero CE
    loss.backward()
    # student encoder + iBOT student head + mask_token learn; teacher never does.
    assert any(p.grad is not None for p in m.module.student.encoder.parameters())
    assert any(p.grad is not None
               for p in m.module.ibot_student_head.parameters())
    assert m.module.mask_token.grad is not None
    assert all(p.grad is None for p in m.module.teacher.parameters())
    assert all(p.grad is None
               for p in m.module.ibot_teacher_head.parameters())


def test_ibot_weight_scales_extra_term():
    """ibot_weight=0 => total loss == bare DINO loss (iBOT term contributes 0)."""
    cfg = _cfg("cubic")
    batch = {"image": torch.rand(2, cfg.model.in_channels, 16, 32, 32)}
    torch.manual_seed(0)
    m0 = build_method(cfg, _ibot_ssl(ibot_weight=0.0), torch.device("cpu"))
    m0.configure_schedule(10); m0.train(); m0._step = 3
    loss0, logs0 = m0.compute_loss(batch)
    # iBOT term still computed/logged, but contributes nothing to the total.
    assert logs0["ibot_loss"] > 0.0
    assert float(loss0.detach()) == pytest.approx(logs0["dino_loss"], abs=1e-5)


# ---- independent vs shared dense head --------------------------------------
def test_ibot_independent_head_has_own_params_and_center():
    cfg = _cfg("cubic")
    m = build_method(cfg, _ibot_ssl(ibot_out_dim=48, ibot_share_head=False),
                     torch.device("cpu"))
    assert m.module.own_heads is True
    assert m.module.ibot_student_head is not m.module.student.head
    assert m.module.ibot_center.shape[-1] == 48


def test_ibot_shared_head_reuses_global_head():
    cfg = _cfg("cubic")
    ssl = _ibot_ssl(ibot_share_head=True)           # default level -1 == bottleneck
    m = build_method(cfg, ssl, torch.device("cpu"))
    assert m.module.own_heads is False
    assert m.module.ibot_student_head is m.module.student.head
    assert m.module.ibot_teacher_head is m.module.teacher.head
    assert m.module.ibot_center.shape[-1] == ssl.dino_out_dim


def test_ibot_shared_head_requires_bottleneck_level():
    cfg = _cfg("cubic")
    ssl = _ibot_ssl(ibot_share_head=True, ibot_feature_level=0)   # 8 != 32 channels
    validate_ssl(ssl, cfg)
    with pytest.raises(ValueError):
        build_method(cfg, ssl, torch.device("cpu"))


# ---- EMA of the independent iBOT teacher head ------------------------------
def test_ibot_teacher_head_ema_update():
    cfg = _cfg()
    m = build_method(cfg, _ibot_ssl(ibot_out_dim=48), torch.device("cpu"))
    m.configure_schedule(10)
    with torch.no_grad():
        for p in m.module.ibot_student_head.parameters():
            p.fill_(1.0)
        for p in m.module.ibot_teacher_head.parameters():
            p.zero_()
    m.on_after_step(5)
    mom = m._momentum()                             # teacher <- mom*0 + (1-mom)*1
    for p in m.module.ibot_teacher_head.parameters():
        assert torch.allclose(p, torch.full_like(p, 1.0 - mom), atol=1e-6)


# ---- SSL -> downstream handoff (encoder.* only, same as DINO) --------------
def test_ibot_handoff_encoder_only():
    cfg = _cfg("cubic")
    m = build_method(cfg, _ibot_ssl(), torch.device("cpu"))
    sd = m.export_backbone_state_dict()
    assert sd and all(k.startswith("encoder.") for k in sd)

    seg_model = build_model(cfg)
    result = seg_model.load_state_dict(sd, strict=False)
    enc_missing = [k for k in result.missing_keys if k.startswith("encoder.")]
    assert enc_missing == [], f"encoder keys not transferred: {enc_missing}"
    assert list(result.unexpected_keys) == [], list(result.unexpected_keys)
    assert any(k.startswith("decoder.") for k in result.missing_keys)
    assert any(k.startswith("seg_head.") for k in result.missing_keys)


# ---- 2.5D shape support ----------------------------------------------------
def test_ibot_method_2_5d_forward_backward():
    cfg = _cfg("2_5d")
    assert cfg.model.spatial_dims == 2
    ssl = _ibot_ssl(ibot_out_dim=48)
    validate_ssl(ssl, cfg)
    m = build_method(cfg, ssl, torch.device("cpu"))
    m.configure_schedule(10); m.train(); m._step = 3
    batch = {"image": torch.rand(2, cfg.model.in_channels, 32, 32)}
    loss, logs = m.compute_loss(batch)
    assert torch.isfinite(loss) and loss.requires_grad
    assert logs["ibot_loss"] > 0.0
    loss.backward()
    assert any(p.grad is not None for p in m.module.student.encoder.parameters())


# ---- one-epoch CPU trainer smoke -------------------------------------------
def test_ibot_trainer_smoke(tmp_path):
    cfg = _cfg("cubic")
    cfg.train.epochs = 1
    cfg.train.warmup_epochs = 0
    cfg.train.use_ema = False                       # DINO teacher is already EMA
    cfg.train.use_amp = False
    cfg.train.grad_accum_steps = 1
    cfg.train.output_dir = str(tmp_path)
    cfg.sync()
    cfg.validate()

    ssl = _ibot_ssl(ibot_out_dim=48)
    validate_ssl(ssl, cfg)

    device = torch.device("cpu")
    m = build_method(cfg, ssl, device)
    ds = _ImgDataset(4, cfg.model.in_channels, (16, 32, 32))
    loader = DataLoader(ds, batch_size=2)
    trainer = SSLTrainer(m, cfg, ssl, loader, device)
    out = trainer.fit()
    assert "best_loss" in out

    ckpt = tmp_path / "ssl_best.pt"
    assert ckpt.exists()
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert any(k.startswith("encoder.") for k in blob["model_state_dict"])


# ---------------------------------------------------------------------------
# SparK + DINO⑧ — pixel reconstruction (SparK①) + global distillation (DINO④)
# ---------------------------------------------------------------------------
def _sparkdino_ssl(**kw):
    """Small SparK+DINO SSLConfig for fast CPU tests."""
    ssl = SSLConfig(method="sparkdino")
    ssl.dino_out_dim = 128
    ssl.dino_hidden_dim = 64
    ssl.dino_bottleneck_dim = 32
    ssl.dino_local_crops = 2
    ssl.spark_mask_unit = 8
    ssl.recon_loss = "mse"
    for k, v in kw.items():
        setattr(ssl, k, v)
    return ssl


# ---- config validation (reuses both dino_* and spark_* checks) -------------
def test_ssl_validate_sparkdino_ok():
    validate_ssl(_sparkdino_ssl(), _cfg())


def test_ssl_validate_sparkdino_inherits_dino_checks():
    ssl = _sparkdino_ssl(dino_global_crops=1)       # DINO rule: needs >= 2 globals
    with pytest.raises(ConfigError):
        validate_ssl(ssl, _cfg())


def test_ssl_validate_sparkdino_inherits_spark_checks():
    ssl = _sparkdino_ssl(spark_mask_ratio=1.0)      # SparK rule: in (0,1)
    with pytest.raises(ConfigError):
        validate_ssl(ssl, _cfg())


@pytest.mark.parametrize("value", [-1.0, -0.01])
def test_ssl_validate_sparkdino_rejects_bad_weight(value):
    ssl = _sparkdino_ssl(sparkdino_dino_weight=value)
    with pytest.raises(ConfigError):
        validate_ssl(ssl, _cfg())


# ---- shared encoder receives gradients from BOTH branches ------------------
def test_sparkdino_loss_runs_and_backward():
    cfg = _cfg("cubic")
    ssl = _sparkdino_ssl()
    validate_ssl(ssl, cfg)
    m = build_method(cfg, ssl, torch.device("cpu"))
    m.configure_schedule(10)
    m.train()
    m._step = 3
    batch = {"image": torch.rand(2, cfg.model.in_channels, 16, 32, 32)}
    loss, logs = m.compute_loss(batch)
    assert torch.isfinite(loss) and loss.requires_grad
    assert logs["spark_loss"] > 0.0 and "dino_loss" in logs
    loss.backward()
    # one shared encoder trained by both branches; SparK decoder + DINO head learn.
    assert any(p.grad is not None for p in m.module.student.encoder.parameters())
    assert any(p.grad is not None for p in m.module.spark_decoder.parameters())
    assert any(p.grad is not None for p in m.module.student.head.parameters())
    assert all(p.grad is None for p in m.module.teacher.parameters())


def test_sparkdino_shares_single_encoder():
    """SparK branch and DINO student branch use the very same encoder instance."""
    cfg = _cfg("cubic")
    m = build_method(cfg, _sparkdino_ssl(), torch.device("cpu"))
    # decoder operates on student.encoder features (no second encoder built).
    assert hasattr(m.module, "spark_decoder")
    assert m.module.spark_decoder is not None
    # teacher has its own (EMA) encoder, distinct from the student/SparK encoder.
    assert m.module.teacher.encoder is not m.module.student.encoder


def test_sparkdino_weight_zero_is_pure_spark():
    """dino_weight=0 => total loss == SparK reconstruction loss alone."""
    cfg = _cfg("cubic")
    m = build_method(cfg, _sparkdino_ssl(sparkdino_dino_weight=0.0),
                     torch.device("cpu"))
    m.configure_schedule(10); m.train(); m._step = 3
    loss, logs = m.compute_loss(
        {"image": torch.rand(2, cfg.model.in_channels, 16, 32, 32)})
    assert logs["dino_loss"] > 0.0                  # still computed/logged
    assert float(loss.detach()) == pytest.approx(logs["spark_loss"], abs=1e-5)


def test_sparkdino_weight_scales_dino_term():
    """Total == spark_loss + mu * dino_loss for the logged components."""
    cfg = _cfg("cubic")
    m = build_method(cfg, _sparkdino_ssl(sparkdino_dino_weight=2.0),
                     torch.device("cpu"))
    m.configure_schedule(10); m.train(); m._step = 3
    loss, logs = m.compute_loss(
        {"image": torch.rand(2, cfg.model.in_channels, 16, 32, 32)})
    assert logs["dino_weight"] == pytest.approx(2.0)
    expect = logs["spark_loss"] + 2.0 * logs["dino_loss"]
    assert float(loss.detach()) == pytest.approx(expect, abs=1e-5)


# ---- SSL -> downstream handoff (encoder.* only) ----------------------------
def test_sparkdino_handoff_encoder_only():
    cfg = _cfg("cubic")
    m = build_method(cfg, _sparkdino_ssl(), torch.device("cpu"))
    sd = m.export_backbone_state_dict()
    assert sd and all(k.startswith("encoder.") for k in sd)

    seg_model = build_model(cfg)
    result = seg_model.load_state_dict(sd, strict=False)
    enc_missing = [k for k in result.missing_keys if k.startswith("encoder.")]
    assert enc_missing == [], f"encoder keys not transferred: {enc_missing}"
    assert list(result.unexpected_keys) == [], list(result.unexpected_keys)
    assert any(k.startswith("decoder.") for k in result.missing_keys)
    assert any(k.startswith("seg_head.") for k in result.missing_keys)


# ---- 2.5D shape support ----------------------------------------------------
def test_sparkdino_method_2_5d_forward_backward():
    cfg = _cfg("2_5d")
    assert cfg.model.spatial_dims == 2
    ssl = _sparkdino_ssl()
    validate_ssl(ssl, cfg)
    m = build_method(cfg, ssl, torch.device("cpu"))
    m.configure_schedule(10); m.train(); m._step = 3
    batch = {"image": torch.rand(2, cfg.model.in_channels, 32, 32)}
    loss, logs = m.compute_loss(batch)
    assert torch.isfinite(loss) and loss.requires_grad
    assert logs["spark_loss"] > 0.0
    loss.backward()
    assert any(p.grad is not None for p in m.module.student.encoder.parameters())
    assert any(p.grad is not None for p in m.module.spark_decoder.parameters())


# ---- one-epoch CPU trainer smoke -------------------------------------------
def test_sparkdino_trainer_smoke(tmp_path):
    cfg = _cfg("cubic")
    cfg.train.epochs = 1
    cfg.train.warmup_epochs = 0
    cfg.train.use_ema = False                       # DINO teacher is already EMA
    cfg.train.use_amp = False
    cfg.train.grad_accum_steps = 1
    cfg.train.output_dir = str(tmp_path)
    cfg.sync()
    cfg.validate()

    ssl = _sparkdino_ssl()
    validate_ssl(ssl, cfg)

    device = torch.device("cpu")
    m = build_method(cfg, ssl, device)
    ds = _ImgDataset(4, cfg.model.in_channels, (16, 32, 32))
    loader = DataLoader(ds, batch_size=2)
    trainer = SSLTrainer(m, cfg, ssl, loader, device)
    out = trainer.fit()
    assert "best_loss" in out

    ckpt = tmp_path / "ssl_best.pt"
    assert ckpt.exists()
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert any(k.startswith("encoder.") for k in blob["model_state_dict"])


# ---------------------------------------------------------------------------
# VICRegL — dense-correspondence SSL (global VIC + location-matched local VIC)
# ---------------------------------------------------------------------------
def _vicregl_ssl(**kw):
    """Small VICRegL SSLConfig for fast CPU tests."""
    ssl = SSLConfig(method="vicregl")
    ssl.vicregl_proj_dim = 32
    ssl.vicregl_hidden_dim = 32
    ssl.vicregl_dense_proj_dim = 16
    ssl.vicregl_num_matches = 8
    for k, v in kw.items():
        setattr(ssl, k, v)
    return ssl


# ---- config validation -----------------------------------------------------
def test_ssl_validate_vicregl_ok():
    validate_ssl(_vicregl_ssl(), _cfg())


@pytest.mark.parametrize("field,value", [
    ("vicregl_proj_dim", 0),
    ("vicregl_num_matches", 0),
    ("vicregl_sim_coeff", -1.0),
    ("vicregl_alpha", 1.5),
    ("vicregl_feature_level", 99),
    ("vicregl_crop_scale", [0.9, 0.5]),
])
def test_ssl_validate_vicregl_rejects(field, value):
    ssl = _vicregl_ssl(**{field: value})
    with pytest.raises(ConfigError):
        validate_ssl(ssl, _cfg())


# ---- paired crops + coordinate bookkeeping ---------------------------------
def test_paired_crop_shapes_and_meta():
    gen = PairedCropGenerator(spatial_dims=3, out_size=(16, 32, 32),
                              scale=(0.6, 1.0))
    x = torch.rand(2, 1, 16, 32, 32)
    (v1, v2), (m1, m2) = gen(x)
    for v in (v1, v2):
        assert v.shape == (2, 1, 16, 32, 32)
    for m in (m1, m2):
        assert m["origin"].shape == (2, 3)
        assert m["size"].shape == (2, 3)
        assert m["flip"].shape == (2, 3) and m["flip"].dtype == torch.bool
        assert bool((m["origin"] >= 0).all())
        hi = m["origin"] + m["size"]
        assert bool((hi <= torch.tensor([16., 32., 32.])).all())


def test_site_coords_known_box_and_flip():
    # 1 sample, 2 axes, feature map 2x2, box origin (4, 8) size (8, 16)
    meta = {
        "origin": torch.tensor([[4.0, 8.0]]),
        "size": torch.tensor([[8.0, 16.0]]),
        "flip": torch.tensor([[False, True]]),
    }
    coords = site_coords((2, 2), meta)               # (1, 4, 2)
    assert coords.shape == (1, 4, 2)
    # axis 0 (no flip): centers at 4 + {0.25, 0.75}*8 = {6, 10}
    # axis 1 (flipped): centers at 8 + {0.75, 0.25}*16 = {20, 12}
    expected = torch.tensor([[[6., 20.], [6., 12.], [10., 20.], [10., 12.]]])
    assert torch.allclose(coords, expected)


def test_site_coords_identical_boxes_match_identity():
    meta = {
        "origin": torch.tensor([[0.0, 0.0, 0.0]]),
        "size": torch.tensor([[16.0, 32.0, 32.0]]),
        "flip": torch.zeros(1, 3, dtype=torch.bool),
    }
    c1 = site_coords((2, 4, 4), meta)
    c2 = site_coords((2, 4, 4), meta)
    nn_idx = torch.cdist(c1[0], c2[0]).argmin(dim=1)
    assert torch.equal(nn_idx, torch.arange(c1.shape[1]))


# ---- loss + backward --------------------------------------------------------
def test_vicregl_loss_runs_and_backward():
    cfg = _cfg()
    ssl = _vicregl_ssl()
    validate_ssl(ssl, cfg)
    m = build_method(cfg, ssl, torch.device("cpu"))
    x = torch.rand(2, cfg.model.in_channels, 16, 32, 32)
    loss, logs = m.compute_loss({"image": x})
    assert torch.isfinite(loss)
    for k in ("vicregl_loss", "global_loss", "local_loss"):
        assert k in logs
    loss.backward()
    grads = [p.grad for p in m.module.encoder.parameters()
             if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)


def test_vicregl_alpha_one_is_pure_global():
    cfg = _cfg()
    m = build_method(cfg, _vicregl_ssl(vicregl_alpha=1.0), torch.device("cpu"))
    x = torch.rand(2, cfg.model.in_channels, 16, 32, 32)
    loss, logs = m.compute_loss({"image": x})
    assert abs(float(loss.detach()) - logs["global_loss"]) < 1e-5


# ---- handoff: encoder-only warm-start ---------------------------------------
def test_vicregl_handoff_encoder_only():
    cfg = _cfg()
    m = build_method(cfg, _vicregl_ssl(), torch.device("cpu"))
    sd = m.export_backbone_state_dict()
    assert sd and all(k.startswith("encoder.") for k in sd)

    seg_model = build_model(cfg)
    result = seg_model.load_state_dict(strip_common_prefixes(sd), strict=False)
    assert not result.unexpected_keys
    assert all(not k.startswith("encoder.") for k in result.missing_keys)


# ---- one-epoch CPU trainer smoke -------------------------------------------
def test_vicregl_trainer_smoke(tmp_path):
    cfg = _cfg("cubic")
    cfg.train.epochs = 1
    cfg.train.warmup_epochs = 0
    cfg.train.use_ema = False
    cfg.train.use_amp = False
    cfg.train.grad_accum_steps = 1
    cfg.train.output_dir = str(tmp_path)
    cfg.sync()
    cfg.validate()

    ssl = _vicregl_ssl()
    validate_ssl(ssl, cfg)

    device = torch.device("cpu")
    m = build_method(cfg, ssl, device)
    ds = _ImgDataset(4, cfg.model.in_channels, (16, 32, 32))
    loader = DataLoader(ds, batch_size=2)
    trainer = SSLTrainer(m, cfg, ssl, loader, device)
    out = trainer.fit()
    assert "best_loss" in out

    ckpt = tmp_path / "ssl_best.pt"
    assert ckpt.exists()
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert any(k.startswith("encoder.") for k in blob["model_state_dict"])


# ---------------------------------------------------------------------------
# probe UNet head (ssl.probe_head='unet')
# ---------------------------------------------------------------------------
def test_probe_head_validate_rejects_bad():
    ssl = SSLConfig()
    ssl.probe_head = "resnet"
    with pytest.raises(ConfigError):
        validate_ssl(ssl, _cfg())


def test_unet_probe_head_forward_shape():
    from ssltask.eval.probe import _UNetProbeHead
    head = _UNetProbeHead([8, 16, 32], out_channels=1, spatial_dims=3, width=8)
    feats = [torch.rand(2, 8, 8, 16, 16), torch.rand(2, 16, 4, 8, 8),
             torch.rand(2, 32, 2, 4, 4)]
    out = head(feats, (16, 32, 32))
    assert out.shape == (2, 1, 16, 32, 32)


def test_seg_probe_unet_head_evaluate(tmp_path):
    _write_labeled_npz(tmp_path, 2, (20, 40, 40))
    cfg = _cfg("cubic")
    ssl = SSLConfig(method="genesis")
    ssl.probe_enabled = True
    ssl.probe_data_dir = str(tmp_path)
    ssl.probe_iters = 2
    ssl.probe_samples_per_volume = 2
    ssl.probe_head = "unet"
    ssl.probe_head_width = 8
    validate_ssl(ssl, cfg)

    from ssltask.eval.probe import _UNetProbeHead
    probe = SegProbe(cfg, ssl, torch.device("cpu"))
    assert isinstance(probe._build_head(), _UNetProbeHead)
    sd = build_ssl_recon_model(cfg).state_dict()
    out = probe.evaluate(sd)
    assert set(out) == {"probe_dice", "probe_hd95"}
    assert 0.0 <= out["probe_dice"] <= 1.0
    assert out["probe_hd95"] >= 0.0
