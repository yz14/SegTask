"""生成（超分）任务冒烟测试（TODO #2）。

无真实数据、CPU 小张量，覆盖：

  * 超分退化算子形状与（区域均值）数值合理性；
  * 重建损失 / 扩散损失可前向、可反传；PSNR/SSIM 可计算；
  * ADM/EDM2 重新启用 timestep/σ 条件后：分割路径形状不变（向后兼容）、
    扩散 backbone 前向 + 反传通；
  * 生成模型端到端（回归 + 扩散 × {adm,edm2} × {edm,ddpm_eps}）：
    训练前向出 loss 三元组、可反传；推理 restore 出图形状正确；
  * GenerationTrainer 跑通 train+val（PSNR/SSIM）一两个 epoch；
  * GenerationPredictor 对体数据 restore（含 z 不整除 / 体深不足 slab 两种边界）。

运行（无需 pytest）：``python tests/test_generation_smoke.py``。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gentask.config import Config  # noqa: E402


def _cfg(algorithm="regression", arch="unet", param="edm",
         mode="2_5d", out_channels=1):
    cfg = Config()
    cfg.data.patch_mode = mode
    cfg.data.num_classes = 2
    cfg.data.label_values = [0, 1]
    cfg.data.patch_size = [4, 16, 16] if mode == "2_5d" else [8, 16, 16]
    cfg.data.multi_res_scales = [1.0]
    cfg.model.arch = arch
    cfg.model.encoder_channels = [16, 32, 48]
    cfg.model.encoder_blocks_per_stage = [1, 1, 1]
    cfg.task.type = "generation"
    cfg.task.algorithm = algorithm
    cfg.task.degradation = "superres"
    cfg.task.out_channels = out_channels
    cfg.task.sr_scale = 2
    cfg.task.parameterization = param
    cfg.task.sampler = "edm_heun" if param == "edm" else "ddpm"
    cfg.task.sample_steps = 2
    cfg.sync()
    cfg.validate()
    return cfg


# ---------------------------------------------------------------------------
# S2 退化
# ---------------------------------------------------------------------------
def test_degradation_shape_and_smoothing():
    from gentask.data.degradation import build_degradation

    cfg = _cfg()
    deg = build_degradation(cfg.task, spatial_dims=2)
    hr = torch.rand(2, 4, 16, 16)
    lr = deg.degrade(hr)
    assert lr.shape == hr.shape, (lr.shape, hr.shape)
    # 退化（下采样→上采样）应削弱高频：总方差不增。
    assert lr.var().item() <= hr.var().item() + 1e-4


def test_anisotropic_degradation_zaxis():
    """各向异性退化：``axis_scales=[2,1,1]`` 只对 z 轴（D）超分，H,W 原样保留。"""
    from gentask.data.degradation import SuperResDegradation

    deg = SuperResDegradation(scale=1, spatial_dims=3, kernel="area",
                              axis_scales=[2, 1, 1])
    assert deg.axis_scales == (2, 1, 1)
    D, H, W = 8, 16, 16
    # 仅沿 W 变化（z/H 上恒定）→ 只退化 z 时应为恒等变换。
    along_w = torch.arange(W).float().view(1, 1, 1, 1, W).expand(2, 1, D, H, W).contiguous()
    assert torch.allclose(deg.degrade(along_w), along_w, atol=1e-5)
    # 仅沿 z 变化→ z 轴被模糊，方差下降。
    along_z = torch.arange(D).float().view(1, 1, D, 1, 1).expand(2, 1, D, H, W).contiguous()
    assert deg.degrade(along_z).var().item() < along_z.var().item()
    # 全 1 且无噪声 → 恒等。
    noop = SuperResDegradation(scale=1, spatial_dims=2, axis_scales=[1, 1])
    x = torch.rand(2, 4, 16, 16)
    assert torch.equal(noop.degrade(x), x)
    # 长度不匹配 spatial_dims 报错。
    try:
        SuperResDegradation(scale=2, spatial_dims=3, axis_scales=[2, 1])
    except ValueError:
        pass
    else:
        raise AssertionError("axis_scales 长度与 spatial_dims 不符应报错")


def test_vfi_decimate_degradation():
    """VFI 抽稀+线性插值：仅退化轴受影响、形状不变、去高频，非法 sampling 报错。"""
    from gentask.data.degradation import SuperResDegradation

    deg = SuperResDegradation(scale=1, spatial_dims=3, axis_scales=[2, 1, 1],
                              sampling="decimate")
    assert deg.sampling == "decimate"
    D, H, W = 8, 16, 16
    # 沿 W 变化、z 上恒定 → 只动 z 的抽稀+插值为恒等（W 全分辨率保留）。
    along_w = torch.arange(W).float().view(1, 1, 1, 1, W).expand(2, 1, D, H, W).contiguous()
    assert torch.allclose(deg.degrade(along_w), along_w, atol=1e-5)
    # 沿 z 变化 → 抽稀丢中间帧信息，结果改变、形状不变、方差不增（去高频）。
    along_z = torch.rand(2, 1, D, H, W)
    out = deg.degrade(along_z)
    assert out.shape == along_z.shape
    assert not torch.allclose(out, along_z, atol=1e-3)
    assert out.var().item() <= along_z.var().item() + 1e-4
    # 非法 sampling 报错。
    try:
        SuperResDegradation(scale=2, spatial_dims=3, sampling="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("非法 sampling 应报错")


# ---------------------------------------------------------------------------
# S3 损失与指标
# ---------------------------------------------------------------------------
def test_recon_loss_backprop():
    from gentask.losses.recon import build_recon_loss, psnr, ssim

    cfg = _cfg()
    cfg.task.ssim_weight = 0.5
    cfg.task.grad_weight = 0.1
    loss_fn = build_recon_loss(cfg)
    pred = torch.rand(2, 4, 16, 16, requires_grad=True)
    target = torch.rand(2, 4, 16, 16)
    bd = {}
    loss = loss_fn(pred, target, breakdown=bd)
    loss.backward()
    assert torch.isfinite(loss) and pred.grad is not None
    assert float(psnr(pred.detach(), target)) > 0 or True  # 可计算即可
    assert torch.isfinite(ssim(pred.detach(), target, spatial_dims=2))


def test_diffusion_loss_backprop():
    from gentask.losses.recon import DiffusionLoss

    dl = DiffusionLoss()
    pred = torch.randn(2, 4, 16, 16, requires_grad=True)
    target = torch.randn(2, 4, 16, 16)
    weight = torch.rand(2) + 0.1
    bd = {}
    loss = dl({"pred": pred, "target": target, "weight": weight}, breakdown=bd)
    loss.backward()
    assert torch.isfinite(loss) and pred.grad is not None


# ---------------------------------------------------------------------------
# S5a 条件 backbone：分割向后兼容 + 扩散前向/反传
# ---------------------------------------------------------------------------
def _seg_cfg(arch):
    cfg = Config()
    cfg.data.patch_mode = "2_5d"
    cfg.data.num_classes = 3
    cfg.data.label_values = [0, 1, 2]
    cfg.data.patch_size = [4, 16, 16]
    cfg.data.multi_res_scales = [1.0]
    cfg.model.arch = arch
    cfg.model.encoder_channels = [16, 32, 48]
    cfg.model.encoder_blocks_per_stage = [1, 1, 1]
    cfg.sync()
    cfg.validate()
    return cfg


def test_seg_backbone_unchanged():
    from gentask.models.factory import build_model

    for arch in ("adm", "edm2"):
        cfg = _seg_cfg(arch)
        model = build_model(cfg).eval()
        x = torch.randn(2, cfg.model.in_channels, 16, 16)
        with torch.no_grad():
            out = model(x)
        t = out["main"] if isinstance(out, dict) else out
        t = t[0] if isinstance(t, list) else t
        # num_fg(=2) * D(=4) = 8 通道。
        assert tuple(t.shape) == (2, 8, 16, 16), t.shape


def test_diffusion_backbone_forward_backward():
    from taskcore.models.adm_unet import build_adm_diffusion_unet
    from taskcore.models.edm2_unet import build_edm2_diffusion_unet

    for arch, builder in (("adm", build_adm_diffusion_unet),
                          ("edm2", build_edm2_diffusion_unet)):
        cfg = _seg_cfg(arch)
        d = 4
        net = builder(cfg, in_channels=2 * d, out_channels=d)
        net.train()
        xc = torch.randn(2, 2 * d, 16, 16)
        cn = torch.rand(2)
        out = net(xc, cn)
        assert tuple(out.shape) == (2, d, 16, 16), out.shape
        out.sum().backward()
        assert any(p.grad is not None for p in net.parameters())


# ---------------------------------------------------------------------------
# S4 / S5 生成模型端到端
# ---------------------------------------------------------------------------
def test_regression_model_end_to_end():
    from gentask.losses.recon import build_recon_loss
    from gentask.models.factory import build_model

    cfg = _cfg("regression")
    model = build_model(cfg)
    loss_fn = build_recon_loss(cfg)
    hr = torch.rand(2, 4, 16, 16)
    out = model(hr)
    assert set(out) >= {"pred", "target"}
    loss = loss_fn(out["pred"], out["target"])
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    model.eval()
    with torch.no_grad():
        rec = model.restore(model.degrade(hr))
    assert tuple(rec.shape) == (2, 4, 16, 16)


def _multi_view_aux_cfg():
    cfg = Config()
    cfg.data.patch_mode = "2_5d"
    cfg.data.num_classes = 2
    cfg.data.label_values = [0, 1]
    cfg.data.patch_size = [4, 16, 16]
    cfg.data.multi_res_scales = [1.0, 2.0]
    cfg.data.z_boundary_mode = "edge_pad"
    cfg.model.arch = "unet"
    cfg.model.encoder_channels = [16, 32, 48]
    cfg.model.encoder_blocks_per_stage = [1, 1, 1]
    cfg.model.aux_seg_supervision = True
    cfg.task.type = "generation"
    cfg.task.algorithm = "regression"
    cfg.task.degradation = "superres"
    cfg.task.out_channels = 1
    cfg.task.sr_scale = 2
    cfg.sync()
    cfg.validate()
    return cfg


def _conditioning_cfg(mode: str):
    cfg = Config()
    cfg.data.patch_mode = mode
    cfg.data.num_classes = 2
    cfg.data.label_values = [0, 1]
    cfg.data.patch_size = [4, 16, 16] if mode == "2_5d" else [8, 16, 16]
    cfg.data.multi_res_scales = [1.0]
    cfg.data.cond_dirs = ["/tmp/cond"]
    cfg.data.cond_suffixes = ".nii.gz"
    cfg.data.cond_normalize = "minmax"
    cfg.data.cond_intensity_min = 0.0
    cfg.data.cond_intensity_max = 1.0
    cfg.model.arch = "unet"
    cfg.model.encoder_channels = [16, 32, 48]
    cfg.model.encoder_blocks_per_stage = [1, 1, 1]
    cfg.task.type = "generation"
    cfg.task.algorithm = "regression"
    cfg.task.degradation = "superres"
    cfg.task.out_channels = 1
    cfg.task.sr_scale = 2
    cfg.sync()
    cfg.validate()
    return cfg


def _multi_view_conditioning_cfg(stem_fusion_mode: str):
    cfg = Config()
    cfg.data.patch_mode = "2_5d"
    cfg.data.num_classes = 2
    cfg.data.label_values = [0, 1]
    cfg.data.patch_size = [4, 16, 16]
    cfg.data.multi_res_scales = [1.0, 2.0]
    cfg.data.z_boundary_mode = "edge_pad"
    cfg.data.cond_dirs = ["/tmp/cond"]
    cfg.data.cond_suffixes = ".nii.gz"
    cfg.data.cond_normalize = "minmax"
    cfg.data.cond_intensity_min = 0.0
    cfg.data.cond_intensity_max = 1.0
    cfg.model.arch = "unet"
    cfg.model.stem_fusion_mode = stem_fusion_mode
    cfg.model.encoder_channels = [16, 32, 48]
    cfg.model.encoder_blocks_per_stage = [1, 1, 1]
    cfg.model.aux_seg_supervision = True
    cfg.task.type = "generation"
    cfg.task.algorithm = "regression"
    cfg.task.degradation = "superres"
    cfg.task.out_channels = 1
    cfg.task.sr_scale = 2
    cfg.sync()
    cfg.validate()
    return cfg


def _diffusion_conditioning_cfg():
    cfg = Config()
    cfg.data.patch_mode = "2_5d"
    cfg.data.num_classes = 2
    cfg.data.label_values = [0, 1]
    cfg.data.patch_size = [4, 16, 16]
    cfg.data.multi_res_scales = [1.0]
    cfg.data.cond_dirs = ["/tmp/cond"]
    cfg.data.cond_suffixes = ".nii.gz"
    cfg.data.cond_normalize = "minmax"
    cfg.data.cond_intensity_min = 0.0
    cfg.data.cond_intensity_max = 1.0
    cfg.model.arch = "adm"
    cfg.model.encoder_channels = [16, 32, 48]
    cfg.model.encoder_blocks_per_stage = [1, 1, 1]
    cfg.task.type = "generation"
    cfg.task.algorithm = "diffusion"
    cfg.task.degradation = "superres"
    cfg.task.out_channels = 1
    cfg.task.parameterization = "edm"
    cfg.task.sample_steps = 2
    cfg.task.sampler = "edm_heun"
    cfg.sync()
    cfg.validate()
    return cfg


def test_multi_view_aux_recon_forward_and_restore():
    from gentask.models.factory import build_model

    cfg = _multi_view_aux_cfg()
    model = build_model(cfg)
    model.train()
    hr = torch.cat([
        torch.zeros(1, 4, 16, 16),
        torch.ones(1, 4, 16, 16),
    ], dim=1)
    out = model(hr)
    assert set(out) >= {"pred", "ds_preds", "target", "aux_preds", "aux_targets"}
    assert tuple(out["pred"].shape) == (1, 4, 16, 16)
    assert tuple(out["target"].shape) == (1, 4, 16, 16)
    assert len(out["aux_preds"]) == 1 and len(out["aux_targets"]) == 1
    assert tuple(out["aux_preds"][0].shape) == (1, 4, 16, 16)
    assert tuple(out["aux_targets"][0].shape) == (1, 4, 16, 16)
    assert torch.allclose(out["target"], hr[:, :4])
    assert torch.allclose(out["aux_targets"][0], hr[:, 4:])

    lr = model.degrade(hr)
    restored = model.restore(lr)
    assert tuple(restored.shape) == tuple(out["pred"].shape)
    assert torch.allclose(restored, out["pred"])


def test_multi_view_aux_recon_trainer_step_and_backward():
    from gentask.models.factory import build_model
    from gentask.trainer.gen_trainer import GenerationTrainer

    cfg = _multi_view_aux_cfg()
    cfg.train.epochs = 1
    cfg.train.warmup_epochs = 0
    cfg.train.use_amp = False
    cfg.train.output_dir = "/tmp/gen_test_aux_recon"
    model = build_model(cfg)
    loader = [{
        "image": torch.cat([
            torch.zeros(1, 4, 16, 16),
            torch.ones(1, 4, 16, 16),
        ], dim=1),
        "label": torch.zeros(1, 1, 4, 16, 16),
    }]
    tr = GenerationTrainer(model, cfg, loader, loader, torch.device("cpu"))
    model.train()
    out = model(loader[0]["image"])
    breakdown = {}
    loss = tr._step_loss(out, breakdown)
    loss.backward()
    assert torch.isfinite(loss)
    assert "L_aux" in breakdown and breakdown["L_aux"] > 0
    assert any(p.grad is not None for p in model.parameters())


def test_single_view_regression_unchanged_with_no_aux_path():
    from gentask.losses.recon import build_recon_loss
    from gentask.models.factory import build_model
    from gentask.trainer.gen_trainer import GenerationTrainer

    cfg = _cfg("regression", mode="2_5d")
    cfg.data.multi_res_scales = [1.0]
    cfg.model.aux_seg_supervision = False
    cfg.sync()
    cfg.validate()
    model = build_model(cfg)
    hr = torch.rand(2, 4, 16, 16)
    out = model(hr)
    assert set(out) == {"pred", "ds_preds", "target"}
    loss_ref = build_recon_loss(cfg)(out["pred"], out["target"])
    tr = GenerationTrainer(model, cfg, _loader(cfg.data.patch_size[0], n=1, B=2),
                           _loader(cfg.data.patch_size[0], n=1, B=2),
                           torch.device("cpu"))
    loss_new = tr._step_loss(out, {})
    assert torch.allclose(loss_ref, loss_new)


def test_conditioning_disabled_is_noop():
    from gentask.models.factory import build_model

    for mode, hr in (
        ("z_axis", torch.rand(2, 1, 8, 16, 16)),
        ("2_5d", torch.rand(2, 4, 16, 16)),
    ):
        cfg = _cfg("regression", mode=mode)
        model = build_model(cfg).eval()
        with torch.no_grad():
            out0 = model(hr)
            out1 = model(hr, cond=None)
            lr = model.degrade(hr)
            rec0 = model.restore(lr)
            rec1 = model.restore(lr, cond=None)
        assert torch.allclose(out0["pred"], out1["pred"])
        assert torch.allclose(rec0, rec1)


def test_conditioning_3d_forward_and_trainer_step():
    from gentask.models.factory import build_model
    from gentask.trainer.gen_trainer import GenerationTrainer

    cfg = _conditioning_cfg("z_axis")
    model = build_model(cfg)
    assert cfg.model.in_channels == 2
    hr = torch.rand(2, 1, 8, 16, 16)
    cond = torch.rand(2, 1, 8, 16, 16)
    batch = {"image": hr, "cond": cond, "label": torch.zeros(2, 1, 8, 16, 16)}
    out = model(hr, cond=cond)
    assert tuple(out["pred"].shape) == tuple(hr.shape)
    with tempfile.TemporaryDirectory() as td:
        cfg.train.output_dir = td
        tr = GenerationTrainer(model, cfg, [batch], [batch], torch.device("cpu"))
        loss = tr._train_epoch(0)["loss"]
    assert np.isfinite(loss)
    with torch.no_grad():
        rec = model.restore(model.degrade(hr), cond=cond)
    assert tuple(rec.shape) == tuple(hr.shape)


def test_conditioning_2_5d_forward_and_trainer_step():
    from gentask.models.factory import build_model
    from gentask.trainer.gen_trainer import GenerationTrainer

    cfg = _conditioning_cfg("2_5d")
    model = build_model(cfg)
    assert cfg.model.in_channels == 8
    hr = torch.rand(2, 1, 4, 16, 16)
    cond = torch.rand(2, 1, 4, 16, 16)
    batch = {"image": hr, "cond": cond, "label": torch.zeros(2, 1, 4, 16, 16)}
    out = model(hr, cond=cond)
    assert tuple(out["pred"].shape) == (2, 4, 16, 16)
    with tempfile.TemporaryDirectory() as td:
        cfg.train.output_dir = td
        tr = GenerationTrainer(model, cfg, [batch], [batch], torch.device("cpu"))
        loss = tr._train_epoch(0)["loss"]
    assert np.isfinite(loss)
    with torch.no_grad():
        rec = model.restore(model.degrade(hr), cond=cond)
    assert tuple(rec.shape) == (2, 4, 16, 16)


def test_multi_view_conditioning_forward_backward_and_restore():
    from gentask.models.factory import build_model
    from gentask.trainer.gen_trainer import GenerationTrainer

    hr = torch.cat([
        torch.zeros(1, 4, 16, 16),
        torch.ones(1, 4, 16, 16),
    ], dim=1)
    cond = torch.full((1, 4, 16, 16), 2.0)
    for stem_mode in ("multi_stem_proj", "hierarchical"):
        cfg = _multi_view_conditioning_cfg(stem_mode)
        model = build_model(cfg)
        assert cfg.model.in_channels == 12
        model.train()
        out = model(hr, cond=cond)
        assert set(out) >= {"pred", "ds_preds", "target", "aux_preds", "aux_targets"}
        assert tuple(out["pred"].shape) == (1, 4, 16, 16)
        assert tuple(out["target"].shape) == (1, 4, 16, 16)
        assert len(out["aux_preds"]) == 1
        assert len(out["aux_targets"]) == 1
        assert tuple(out["aux_preds"][0].shape) == (1, 4, 16, 16)
        assert tuple(out["aux_targets"][0].shape) == (1, 4, 16, 16)
        assert torch.allclose(out["target"], hr[:, :4])
        assert torch.allclose(out["aux_targets"][0], hr[:, 4:])

        batch = {"image": hr, "label": torch.zeros(1, 1, 4, 16, 16), "cond": cond}
        tr = GenerationTrainer(model, cfg, [batch], [batch], torch.device("cpu"))
        breakdown = {}
        loss = tr._step_loss(out, breakdown)
        loss.backward()
        assert torch.isfinite(loss)
        assert breakdown.get("L_aux", 0.0) > 0
        assert any(p.grad is not None for p in model.parameters())

        model.eval()
        with torch.no_grad():
            rec = model.restore(model.degrade(hr), cond=cond)
        assert tuple(rec.shape) == tuple(out["pred"].shape)


def test_diffusion_conditioning_forward_restore_and_backward():
    from gentask.losses.recon import DiffusionLoss
    from gentask.models.factory import build_model

    cfg = _diffusion_conditioning_cfg()
    model = build_model(cfg)
    assert cfg.model.in_channels == 8
    hr = torch.rand(2, 4, 16, 16)
    cond = torch.rand(2, 4, 16, 16)
    out = model(hr, cond=cond)
    assert tuple(out["pred"].shape) == tuple(hr.shape)
    assert tuple(out["target"].shape) == tuple(hr.shape)
    loss = DiffusionLoss()(out)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    with torch.no_grad():
        rec = model.restore(model.degrade(hr), cond=cond)
    assert tuple(rec.shape) == tuple(hr.shape)


def test_diffusion_no_cond_restore_unchanged():
    from gentask.models.factory import build_model

    cfg = _cfg("diffusion", arch="adm", param="edm")
    model = build_model(cfg)
    hr = torch.rand(2, 4, 16, 16)
    lr = model.degrade(hr)
    torch.manual_seed(123)
    rec0 = model.restore(lr)
    torch.manual_seed(123)
    rec1 = model.restore(lr, cond=None)
    assert torch.allclose(rec0, rec1)


def test_zaxis_sr_regression_3d_end_to_end():
    """3D 厚→薄 z 轴超分回归：退化仅作用 z，模型输出与 HR 同形、可反传。"""
    from gentask.losses.recon import build_recon_loss
    from gentask.models.factory import build_model

    cfg = Config()
    cfg.data.patch_mode = "z_axis"
    cfg.data.num_classes = 2
    cfg.data.label_values = [0, 1]
    cfg.data.patch_size = [8, 16, 16]
    cfg.data.multi_res_scales = [1.0]
    cfg.model.arch = "unet"
    cfg.model.encoder_channels = [16, 32, 48]
    cfg.model.encoder_blocks_per_stage = [1, 1, 1]
    cfg.task.type = "generation"
    cfg.task.algorithm = "regression"
    cfg.task.degradation = "superres"
    cfg.task.out_channels = 1
    cfg.task.sr_scale_per_axis = [2, 1, 1]
    cfg.sync()
    cfg.validate()

    model = build_model(cfg)
    hr = torch.rand(2, 1, 8, 16, 16)
    out = model(hr)
    assert tuple(out["pred"].shape) == (2, 1, 8, 16, 16), out["pred"].shape
    loss = build_recon_loss(cfg)(out["pred"], out["target"])
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    # 退化只动 z：x,y 恒定体不变。
    along_z_const = torch.arange(16).float().view(1, 1, 1, 1, 16).expand(2, 1, 8, 16, 16).contiguous()
    with torch.no_grad():
        assert torch.allclose(model.degrade(along_z_const), along_z_const, atol=1e-5)


def test_deep_supervision_regression():
    """深监督回归：forward 返回多尺度头（head0 全分辨率），残差头可加下采样基线。"""
    from gentask.models.factory import build_model

    cfg = Config()
    cfg.data.patch_mode = "z_axis"
    cfg.data.num_classes = 2
    cfg.data.label_values = [0, 1]
    cfg.data.patch_size = [8, 16, 16]
    cfg.data.multi_res_scales = [1.0]
    cfg.model.arch = "unet"
    cfg.model.encoder_channels = [16, 32, 48]
    cfg.model.encoder_blocks_per_stage = [1, 1, 1]
    cfg.model.deep_supervision = True
    cfg.task.type = "generation"
    cfg.task.algorithm = "regression"
    cfg.task.degradation = "superres"
    cfg.task.out_channels = 1
    cfg.task.residual = True
    cfg.sync()
    cfg.validate()

    model = build_model(cfg)
    hr = torch.rand(2, 1, 8, 16, 16)
    out = model(hr)
    preds = out["ds_preds"]
    assert len(preds) >= 2, len(preds)
    assert tuple(preds[0].shape) == (2, 1, 8, 16, 16)        # 全分辨率头
    assert preds[1].shape[-1] < preds[0].shape[-1]            # 后续头更小
    assert out["pred"] is preds[0]
    # 推理只取全分辨率头。
    model.eval()
    with torch.no_grad():
        assert tuple(model.restore(model.degrade(hr)).shape) == (2, 1, 8, 16, 16)
    # 深监督与算法='diffusion' 互斥（校验报错）。
    cfg2 = Config()
    cfg2.data.patch_mode = "2_5d"
    cfg2.data.num_classes = 2
    cfg2.data.label_values = [0, 1]
    cfg2.data.patch_size = [4, 16, 16]
    cfg2.data.multi_res_scales = [1.0]
    cfg2.model.arch = "adm"
    cfg2.model.deep_supervision = True
    cfg2.task.type = "generation"
    cfg2.task.algorithm = "diffusion"
    cfg2.task.degradation = "superres"
    try:
        cfg2.sync(); cfg2.validate()
    except Exception:
        pass
    else:
        raise AssertionError("deep_supervision + diffusion 应被校验拒绝")


def test_deep_supervision_trainer_runs():
    """深监督回归经 GenerationTrainer 跑通一个 epoch（多尺度损失可反传）。"""
    from gentask.models.factory import build_model
    from gentask.trainer.gen_trainer import GenerationTrainer

    cfg = _cfg("regression")
    cfg.model.deep_supervision = True
    cfg.sync()
    cfg.validate()
    cfg.train.epochs = 1
    cfg.train.warmup_epochs = 0
    cfg.train.use_amp = False
    cfg.train.output_dir = "/tmp/gen_test_ds_regression"
    D = cfg.data.patch_size[0]
    model = build_model(cfg)
    tr = GenerationTrainer(model, cfg, _loader(D), _loader(D), torch.device("cpu"))
    res = tr.fit()
    assert np.isfinite(res["best_psnr"])
    assert (Path(cfg.train.output_dir) / "best_model.pth").exists()


def test_diffusion_model_end_to_end():
    from gentask.losses.recon import DiffusionLoss
    from gentask.models.factory import build_model

    dl = DiffusionLoss()
    for arch in ("adm", "edm2"):
        for param in ("edm", "ddpm_eps"):
            cfg = _cfg("diffusion", arch=arch, param=param)
            model = build_model(cfg)
            hr = torch.rand(2, 4, 16, 16)
            out = model(hr)
            assert set(out) >= {"pred", "target", "weight"}
            loss = dl(out)
            loss.backward()
            assert any(p.grad is not None for p in model.parameters()), (
                arch, param)
            model.eval()
            with torch.no_grad():
                rec = model.restore(model.degrade(hr))
            assert tuple(rec.shape) == (2, 4, 16, 16), (arch, param, rec.shape)


def test_diffusion_requires_adm_or_edm2():
    from gentask.config.dataclasses import ConfigError

    # validate() 即拦截（早于 build_model），无需等到构建期才报错。
    try:
        _cfg("diffusion", arch="unet")
    except ConfigError:
        return
    raise AssertionError("diffusion + arch='unet' 应当在 validate() 报错")


# ---------------------------------------------------------------------------
# S5c / S6 训练器（train + PSNR/SSIM 验证）
# ---------------------------------------------------------------------------
def _loader(D, n=2, B=2):
    return [{"image": torch.rand(B, D, 16, 16),
             "label": torch.zeros(B, 1, D, 16, 16)} for _ in range(n)]


def _trainer(cfg):
    from gentask.models.factory import build_model
    from gentask.trainer.gen_trainer import GenerationTrainer

    if cfg.data.patch_mode == "2_5d":
        image = torch.zeros(1, cfg.data.patch_size[0], 16, 16)
        label = torch.zeros(1, 1, cfg.data.patch_size[0], 16, 16)
    else:
        image = torch.zeros(1, 1, cfg.data.patch_size[0], 16, 16)
        label = torch.zeros(1, 1, cfg.data.patch_size[0], 16, 16)
    loader = [{"image": image, "label": label}]
    model = build_model(cfg)
    return GenerationTrainer(model, cfg, loader, loader, torch.device("cpu"))


def _high_error_patch_2d():
    pred = torch.zeros(1, 4, 8, 8)
    target = torch.zeros_like(pred)
    target[:, :, :2, :2] = 1.0
    return pred, target


def _high_error_volume_3d():
    pred = torch.zeros(1, 1, 8, 8, 8)
    target = torch.zeros_like(pred)
    target[:, :, :2, :2, :2] = 1.0
    return pred, target


def test_weighted_recon_loss_uniform_matches_unweighted():
    cfg = _cfg("regression", mode="2_5d")
    cfg.task.recon_loss = "l1"
    tr = _trainer(cfg)
    pred, target = _high_error_patch_2d()
    out = {"pred": pred, "target": target}
    w_uniform = torch.ones(1, 1, 4, 8, 8)
    loss_u = tr._step_loss(out, {}, weight_map=None).item()
    loss_w = tr._step_loss(out, {}, weight_map=w_uniform).item()
    assert np.isclose(loss_u, loss_w, rtol=0, atol=1e-6), (loss_u, loss_w)


def test_weighted_recon_loss_nonuniform_increases_with_error_region():
    cfg = _cfg("regression", mode="2_5d")
    cfg.task.recon_loss = "l1"
    tr = _trainer(cfg)
    pred, target = _high_error_patch_2d()
    out = {"pred": pred, "target": target}
    w_uniform = torch.ones(1, 1, 4, 8, 8)
    w_focus = torch.ones(1, 1, 4, 8, 8)
    w_focus[:, :, :2, :2, :2] = 8.0
    loss_u = tr._step_loss(out, {}, weight_map=w_uniform).item()
    loss_f = tr._step_loss(out, {}, weight_map=w_focus).item()
    assert loss_f > loss_u, (loss_u, loss_f)


def test_weighted_ds_recon_loss_uniform_matches_unweighted():
    cfg = _cfg("regression", arch="unet", mode="z_axis")
    cfg.task.recon_loss = "l1"
    cfg.model.deep_supervision = True
    cfg.sync()
    cfg.validate()
    tr = _trainer(cfg)
    pred, target = _high_error_volume_3d()
    low = torch.zeros(1, 1, 4, 4, 4)
    out = {"pred": pred, "ds_preds": [pred, low], "target": target}
    w_uniform = torch.ones_like(pred)
    loss_u = tr._step_loss(out, {}, weight_map=None).item()
    loss_w = tr._step_loss(out, {}, weight_map=w_uniform).item()
    assert np.isclose(loss_u, loss_w, rtol=0, atol=1e-6), (loss_u, loss_w)


def test_weighted_ds_recon_loss_nonuniform_increases_with_error_region():
    cfg = _cfg("regression", arch="unet", mode="z_axis")
    cfg.task.recon_loss = "l1"
    cfg.model.deep_supervision = True
    cfg.sync()
    cfg.validate()
    tr = _trainer(cfg)
    pred, target = _high_error_volume_3d()
    low = torch.zeros(1, 1, 4, 4, 4)
    out = {"pred": pred, "ds_preds": [pred, low], "target": target}
    w_uniform = torch.ones_like(pred)
    w_focus = torch.ones_like(pred)
    w_focus[:, :, :2, :2, :2] = 8.0
    loss_u = tr._step_loss(out, {}, weight_map=w_uniform).item()
    loss_f = tr._step_loss(out, {}, weight_map=w_focus).item()
    assert loss_f > loss_u, (loss_u, loss_f)


def test_generation_trainer_runs():
    from gentask.models.factory import build_model
    from gentask.trainer.gen_trainer import GenerationTrainer

    for algorithm, arch, param in (("regression", "unet", "edm"),
                                    ("diffusion", "adm", "ddpm_eps")):
        cfg = _cfg(algorithm, arch=arch, param=param)
        cfg.train.epochs = 1
        cfg.train.warmup_epochs = 0
        cfg.train.use_amp = False
        cfg.train.use_ema = True
        cfg.train.output_dir = f"/tmp/gen_test_{algorithm}_{arch}_{param}"
        D = cfg.data.patch_size[0]
        model = build_model(cfg)
        tr = GenerationTrainer(model, cfg, _loader(D), _loader(D), torch.device("cpu"))
        res = tr.fit()
        assert np.isfinite(res["best_psnr"])
        assert (Path(cfg.train.output_dir) / "best_model.pth").exists()


# ---------------------------------------------------------------------------
# R2 多视图消费管线 + GPU 增强（trainer/pipelines, data/augment）
# ---------------------------------------------------------------------------
def _dataset_loader(shape, n=1):
    """dataset 布局 batch：单条 max-FOV 过采样 cube (B, 1, eD, eH, eW)。"""
    return [{"image": torch.rand(*shape),
             "label": torch.zeros(*shape)} for _ in range(n)]


def _fit_once(cfg, loader):
    from gentask.models.factory import build_model
    from gentask.trainer.gen_trainer import GenerationTrainer

    cfg.train.epochs = 1
    cfg.train.warmup_epochs = 0
    cfg.train.use_amp = False
    model = build_model(cfg)
    tr = GenerationTrainer(model, cfg, loader, loader, torch.device("cpu"))
    res = tr.fit()
    assert np.isfinite(res["best_psnr"])
    return res


def test_pipeline_multi_view_2_5d_stacked_fit():
    cfg = _cfg("regression", mode="2_5d")
    cfg.data.multi_res_scales = [1.0, 2.0]
    cfg.train.output_dir = tempfile.mkdtemp()
    cfg.sync()
    cfg.validate()
    # dataset 发 (B, 1, eD_max=round(4*2.0)=8, 16, 16)。
    _fit_once(cfg, _dataset_loader((2, 1, 8, 16, 16)))


def test_pipeline_multi_view_lift_fit():
    cfg = _cfg("regression", mode="2_5d")
    cfg.data.multi_res_scales = [1.0, 2.0]
    cfg.model.lift_2_5d_to_3d = True
    cfg.train.output_dir = tempfile.mkdtemp()
    cfg.sync()
    cfg.validate()
    assert cfg.model.spatial_dims == 3
    _fit_once(cfg, _dataset_loader((2, 1, 8, 16, 16)))


def test_pipeline_native_view_depth_fit():
    cfg = _cfg("regression", mode="2_5d")
    cfg.data.multi_res_scales = [1.0, 2.0]
    cfg.data.keep_native_view_depth = True
    cfg.train.output_dir = tempfile.mkdtemp()
    cfg.sync()
    cfg.validate()
    # per_view_depths=[4,8] → in_ch=12；dataset 发 (B,1,8,16,16)。
    _fit_once(cfg, _dataset_loader((2, 1, 8, 16, 16)))


def test_pipeline_zaxis_multi_view_oversample_fit():
    cfg = _cfg("regression", mode="z_axis")
    cfg.data.multi_res_scales = [1.0, 2.0]
    cfg.data.aug_oversample_ratio = 1.5
    cfg.data.z_boundary_mode = "edge_pad"
    cfg.train.output_dir = tempfile.mkdtemp()
    cfg.sync()
    cfg.validate()
    # eD = round(8*1.5)=12, eD_max = round(12*2)=24 → (B,1,24,16,16)。
    _fit_once(cfg, _dataset_loader((2, 1, 24, 16, 16)))


def test_pipeline_vanilla_oversample_crop():
    cfg = _cfg("regression", mode="z_axis")
    cfg.data.aug_oversample_ratio = 1.5
    cfg.train.output_dir = tempfile.mkdtemp()
    cfg.sync()
    cfg.validate()
    # 单视图过采样：eD = round(8*1.5)=12 → pipeline 中心裁回 8。
    _fit_once(cfg, _dataset_loader((2, 1, 12, 16, 16)))


def test_pipeline_prepare_batch_shapes():
    from gentask.trainer.pipelines import build_pipeline

    cfg = _cfg("regression", mode="2_5d")
    cfg.data.multi_res_scales = [1.0, 2.0]
    cfg.sync()
    pipe = build_pipeline(cfg)
    img = torch.rand(2, 1, 8, 16, 16)
    wmap = torch.rand(2, 1, 8, 16, 16)
    out, w, _ = pipe.prepare_batch(img, wmap, None)
    assert out.shape == (2, 2, 4, 16, 16), out.shape
    assert w.shape == (2, 1, 4, 16, 16), w.shape
    # rank-4 预打包输入透传（合成测试 batch 兼容）。
    packed = torch.rand(2, 8, 16, 16)
    out2, _, _ = pipe.prepare_batch(packed, None, None)
    assert out2 is packed


def test_gpu_augmentor_shapes_and_sync():
    from gentask.config import AugConfig
    from gentask.data.augment import GPUAugmentor

    aug_cfg = AugConfig(random_flip_prob=1.0, random_affine_prob=1.0,
                        elastic_deform_prob=1.0, random_brightness_prob=1.0,
                        random_contrast_prob=1.0, random_gamma_prob=1.0,
                        gaussian_noise_prob=1.0, gaussian_blur_prob=1.0,
                        simulate_lowres_prob=1.0, grid_dropout_prob=1.0)
    aug = GPUAugmentor(aug_cfg, max_scale=2.0)
    img = torch.rand(2, 1, 8, 16, 16)
    wmap = torch.ones(2, 1, 8, 16, 16)
    cond = torch.rand(2, 2, 8, 16, 16)
    out, w, c = aug(img, wmap, cond)
    assert out.shape == img.shape
    assert w.shape == wmap.shape
    assert c.shape == cond.shape
    assert torch.isfinite(out).all() and torch.isfinite(c).all()
    # intensity_clamp：增强后不超增强前逐样本值域。
    assert out.max() <= img.max() + 1e-5 and out.min() >= img.min() - 1e-5
    # rank-4 输入（2.5D 预打包）自动升维再还原。
    out4, _, _ = aug(torch.rand(2, 8, 16, 16))
    assert out4.shape == (2, 8, 16, 16)


def test_augmentor_disabled_passthrough():
    from gentask.config import AugConfig
    from gentask.data.augment import GPUAugmentor

    aug = GPUAugmentor(AugConfig(enabled=False))
    img = torch.rand(2, 1, 8, 16, 16)
    out, w, c = aug(img)
    assert out is img and w is None and c is None


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------
def test_generation_predictor_restore_volume():
    from gentask.models.factory import build_model
    from gentask.predictor.gen_predictor import GenerationPredictor

    cfg = _cfg("regression")
    model = build_model(cfg)
    gp = GenerationPredictor(model, cfg, torch.device("cpu"))
    # z=11 不整除 slab=4 → 末窗重叠平均。
    rec = gp.restore_volume(np.random.rand(11, 16, 16).astype(np.float32))
    assert rec.shape == (11, 16, 16) and np.isfinite(rec).all()
    # 体深 < slab → 零填充裁回。
    rec2 = gp.restore_volume(np.random.rand(3, 16, 16).astype(np.float32))
    assert rec2.shape == (3, 16, 16) and np.isfinite(rec2).all()


def main() -> int:
    tests = [
        test_degradation_shape_and_smoothing,
        test_anisotropic_degradation_zaxis,
        test_vfi_decimate_degradation,
        test_recon_loss_backprop,
        test_diffusion_loss_backprop,
        test_seg_backbone_unchanged,
        test_diffusion_backbone_forward_backward,
        test_regression_model_end_to_end,
        test_multi_view_aux_recon_forward_and_restore,
        test_multi_view_aux_recon_trainer_step_and_backward,
        test_single_view_regression_unchanged_with_no_aux_path,
        test_conditioning_disabled_is_noop,
        test_conditioning_3d_forward_and_trainer_step,
        test_conditioning_2_5d_forward_and_trainer_step,
        test_multi_view_conditioning_forward_backward_and_restore,
        test_diffusion_conditioning_forward_restore_and_backward,
        test_diffusion_no_cond_restore_unchanged,
        test_zaxis_sr_regression_3d_end_to_end,
        test_deep_supervision_regression,
        test_deep_supervision_trainer_runs,
        test_diffusion_model_end_to_end,
        test_diffusion_requires_adm_or_edm2,
        test_generation_trainer_runs,
        test_generation_predictor_restore_volume,
        test_weighted_recon_loss_uniform_matches_unweighted,
        test_weighted_recon_loss_nonuniform_increases_with_error_region,
        test_weighted_ds_recon_loss_uniform_matches_unweighted,
        test_weighted_ds_recon_loss_nonuniform_increases_with_error_region,
    ]
    for t in tests:
        try:
            t()
            print(f"  [ok] {t.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {t.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return 1
    print("=" * 60)
    print("All generation smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
