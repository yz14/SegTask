"""模型子系统回归测试（todo2 第 6 批）：

* 3-2 —— 模块级初始化契约声明（declare_no_reinit / init_strategy 不越权）；
* 2-6 —— ResEnc 显存分档（预设联动通道表 / 显存预算估算 / auto batch）。
"""

import logging

import pytest
import torch
import torch.nn as nn

from taskcore.config.core import Config, ConfigError
from taskcore.engine.memory import estimate_resenc_train_memory_gb
from taskcore.models.blocks import (
    DySample3d, GlobalResponseNorm, SelfAttentionBlock, Upsample)
from taskcore.models.factory import build_model, _custom_init_param_ids
from taskcore.models.init_contract import (
    declare_no_reinit, is_protected, protected_param_ids)


def _base_cfg() -> Config:
    cfg = Config()
    cfg.data.patch_size = [16, 32, 32]
    cfg.data.aug_oversample_ratio = 1.25
    return cfg


def _finish(cfg: Config) -> Config:
    cfg.sync()
    cfg.validate()
    return cfg


# ---------------------------------------------------------------------------
# 3-2 初始化契约
# ---------------------------------------------------------------------------
class TestInitContractRegistry:

    def test_declare_param_and_module(self):
        p = nn.Parameter(torch.zeros(3))
        assert not is_protected(p)
        declare_no_reinit(p)
        assert is_protected(p)

        m = nn.Linear(2, 2)
        declare_no_reinit(m)
        assert all(is_protected(q) for q in m.parameters())

    def test_declare_rejects_non_module(self):
        with pytest.raises(TypeError):
            declare_no_reinit(torch.zeros(3))

    def test_protected_param_ids_scoped_to_model(self):
        m1 = nn.Linear(2, 2)
        m2 = nn.Linear(2, 2)
        declare_no_reinit(m1)
        ids = protected_param_ids(m2)
        assert ids == set()
        ids = protected_param_ids(m1)
        assert len(ids) == 2


class TestModuleDeclarations:

    def test_selfattn_block_declares(self):
        blk = SelfAttentionBlock(16, num_heads=2)
        assert all(is_protected(p) for p in blk.parameters())
        assert torch.all(blk.proj.weight == 0)

    def test_dysample_declares(self):
        up = DySample3d(8, 8)
        assert all(is_protected(p) for p in up.parameters())

    def test_pixelshuffle_icnr_declares(self):
        up = Upsample(8, 8, mode="pixelshuffle", spatial_dims=3)
        assert all(is_protected(p) for p in up.expand.parameters())

    def test_grn_declares(self):
        grn = GlobalResponseNorm(8)
        assert is_protected(grn.gamma) and is_protected(grn.beta)


class TestInitStrategyRespectsContracts:

    def test_trunc_normal_preserves_layerscale(self):
        cfg = _base_cfg()
        cfg.model.encoder_channels = [8, 16, 32]
        cfg.model.unet.backbone = "convnext"
        cfg.model.init_strategy = "trunc_normal"
        model = build_model(_finish(cfg))
        from taskcore.models.convnext import ConvNeXtBlock
        gammas = [m.gamma for m in model.modules()
                  if isinstance(m, ConvNeXtBlock) and m.gamma is not None]
        assert gammas, "expected LayerScale-enabled ConvNeXt blocks"
        for g in gammas:
            assert torch.allclose(g, torch.full_like(g, 1e-6))

    def test_kaiming_preserves_selfattn_zero_init(self):
        cfg = _base_cfg()
        cfg.model.encoder_channels = [8, 16, 32]
        cfg.model.unet.backbone = "resnet"
        cfg.model.unet.selfattn.enabled = True
        cfg.model.unet.selfattn.encoder_stages = [0, 0, 1]
        cfg.model.init_strategy = "kaiming"
        model = build_model(_finish(cfg))
        blocks = [m for m in model.modules()
                  if isinstance(m, SelfAttentionBlock)]
        assert blocks
        for blk in blocks:
            assert torch.all(blk.proj.weight == 0)

    def test_custom_init_param_ids_reads_registry(self):
        cfg = _base_cfg()
        cfg.model.encoder_channels = [8, 16, 32]
        cfg.model.unet.backbone = "convnext"
        model = build_model(_finish(cfg))
        ids = _custom_init_param_ids(model)
        assert ids == protected_param_ids(model)
        assert ids

    def test_legacy_untouched(self):
        cfg = _base_cfg()
        cfg.model.encoder_channels = [8, 16, 32]
        cfg.model.unet.backbone = "convnext"
        cfg.model.init_strategy = "legacy"
        model = build_model(_finish(cfg))
        from taskcore.models.convnext import ConvNeXtBlock
        for m in model.modules():
            if isinstance(m, ConvNeXtBlock) and m.gamma is not None:
                assert torch.allclose(
                    m.gamma, torch.full_like(m.gamma, 1e-6))
                return
        pytest.fail("no ConvNeXt block found")


# ---------------------------------------------------------------------------
# 2-6 ResEnc 显存分档
# ---------------------------------------------------------------------------
class TestResEncPresetChannels:

    def test_preset_sets_channels_when_default(self):
        cfg = Config()
        cfg.model.resenc_preset = "m"
        cfg.data.patch_size = [128, 128, 128]
        cfg.data.aug_oversample_ratio = 1.25
        cfg.sync(); cfg.validate()
        assert cfg.model.encoder_channels == [32, 64, 128, 256, 320, 320]
        assert cfg.model.encoder_blocks_per_stage == [1, 3, 4, 6, 6, 6]

    def test_explicit_channels_win(self):
        cfg = Config()
        cfg.model.resenc_preset = "m"
        cfg.model.encoder_channels = [16, 32, 64]
        cfg.data.aug_oversample_ratio = 1.25
        cfg.sync(); cfg.validate()
        assert cfg.model.encoder_channels == [16, 32, 64]
        assert cfg.model.encoder_blocks_per_stage == [1, 3, 4]

    def test_depth_capped_by_patch(self):
        cfg = Config()
        cfg.model.resenc_preset = "xl"          # 模板 8 级
        cfg.data.patch_size = [16, 32, 32]      # 最大轴 32：4 次减半上限
        cfg.data.aug_oversample_ratio = 1.25
        cfg.sync(); cfg.validate()
        assert len(cfg.model.encoder_channels) == 4

    def test_auto_batch_size(self):
        cfg = Config()
        cfg.model.resenc_preset = "l"
        cfg.model.resenc_auto_batch_size = True
        cfg.data.patch_size = [96, 160, 160]
        cfg.data.aug_oversample_ratio = 1.25
        cfg.sync(); cfg.validate()
        est = estimate_resenc_train_memory_gb(
            cfg.data.patch_size, cfg.model.encoder_channels,
            cfg.model.encoder_blocks_per_stage,
            cfg.model.decoder_blocks_per_stage,
            cfg.data.batch_size)["total_gb"]
        assert est <= 24.0
        # 再 +1 就超预算（确实选了最大值）。
        est_next = estimate_resenc_train_memory_gb(
            cfg.data.patch_size, cfg.model.encoder_channels,
            cfg.model.encoder_blocks_per_stage,
            cfg.model.decoder_blocks_per_stage,
            cfg.data.batch_size + 1)["total_gb"]
        assert est_next > 24.0

    def test_over_budget_warns(self, caplog):
        cfg = Config()
        cfg.model.resenc_preset = "m"
        cfg.model.resenc_vram_budget_gb = 1.0
        cfg.data.patch_size = [128, 128, 128]
        cfg.data.batch_size = 8
        cfg.data.aug_oversample_ratio = 1.25
        with caplog.at_level(logging.WARNING):
            cfg.sync(); cfg.validate()
        assert any("超出预算" in r.message for r in caplog.records)

    def test_auto_batch_requires_preset(self):
        cfg = Config()
        cfg.model.resenc_auto_batch_size = True
        cfg.data.aug_oversample_ratio = 1.25
        cfg.sync()
        with pytest.raises(ConfigError):
            cfg.validate()

    def test_negative_budget_rejected(self):
        cfg = Config()
        cfg.model.resenc_preset = "m"
        cfg.model.resenc_vram_budget_gb = -1.0
        cfg.data.aug_oversample_ratio = 1.25
        cfg.sync()
        with pytest.raises(ConfigError):
            cfg.validate()


class TestResEncMemoryEstimator:

    def test_monotone_in_batch(self):
        args = ([64, 64, 64], [32, 64, 128], [1, 3, 4], [1, 1])
        e1 = estimate_resenc_train_memory_gb(*args, 1)["total_gb"]
        e2 = estimate_resenc_train_memory_gb(*args, 2)["total_gb"]
        e4 = estimate_resenc_train_memory_gb(*args, 4)["total_gb"]
        assert e1 < e2 < e4

    def test_state_independent_of_batch(self):
        args = ([64, 64, 64], [32, 64, 128], [1, 3, 4], [1, 1])
        s1 = estimate_resenc_train_memory_gb(*args, 1)["state_gb"]
        s4 = estimate_resenc_train_memory_gb(*args, 4)["state_gb"]
        assert s1 == s4

    def test_amp_halves_activations(self):
        args = ([64, 64, 64], [32, 64, 128], [1, 3, 4], [1, 1])
        a_amp = estimate_resenc_train_memory_gb(
            *args, 2, amp=True)["activation_gb"]
        a_fp32 = estimate_resenc_train_memory_gb(
            *args, 2, amp=False)["activation_gb"]
        assert a_fp32 == pytest.approx(2 * a_amp)
