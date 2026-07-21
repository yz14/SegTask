"""P1b：build_backbone 与 build_model 共享路径，权重同名同形。"""

from __future__ import annotations

import torch

from taskcore.config.core import Config
from taskcore.models.factory import build_backbone, build_model


def _tiny_cfg() -> Config:
    cfg = Config()
    cfg.data.patch_mode = "cubic"
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


def test_build_backbone_encoder_matches_build_model_keys():
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    enc = build_backbone(cfg)
    torch.manual_seed(0)
    full = build_model(cfg)
    assert set(enc.state_dict()) == set(full.encoder.state_dict())
    for k, v in enc.state_dict().items():
        assert torch.equal(v, full.encoder.state_dict()[k]), k


def test_build_backbone_with_decoder_matches_build_model():
    cfg = _tiny_cfg()
    torch.manual_seed(1)
    enc, dec = build_backbone(cfg, with_decoder=True)
    torch.manual_seed(1)
    full = build_model(cfg)
    assert set(enc.state_dict()) == set(full.encoder.state_dict())
    assert set(dec.state_dict()) == set(full.decoder.state_dict())
    for k, v in dec.state_dict().items():
        assert torch.equal(v, full.decoder.state_dict()[k]), k


def test_build_backbone_forward_feature_list():
    cfg = _tiny_cfg()
    enc = build_backbone(cfg)
    x = torch.randn(1, 1, 16, 32, 32)
    feats = enc(x)
    assert isinstance(feats, list)
    assert len(feats) == len(cfg.model.encoder_channels)
    assert feats[-1].shape[1] == cfg.model.encoder_channels[-1]


def test_build_model_attn_gate_target_unetpp():
    cfg = _tiny_cfg()
    cfg.model.decoder_type = "unetpp"
    cfg.model.skip_attention = True
    cfg.sync()
    cfg.validate()

    seg = build_model(cfg, attn_gate_target="skips")
    gen = build_model(cfg, attn_gate_target="upsample")
    assert seg.decoder.attn_gate_target == "skips"
    assert gen.decoder.attn_gate_target == "upsample"
    assert set(seg.encoder.state_dict()) == set(gen.encoder.state_dict())
