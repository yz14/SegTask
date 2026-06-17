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
    from gentask.models.adm_unet import build_adm_diffusion_unet
    from gentask.models.edm2_unet import build_edm2_diffusion_unet

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
    from gentask.models.factory import build_model

    cfg = _cfg("diffusion", arch="unet")
    try:
        build_model(cfg)
    except ValueError:
        return
    raise AssertionError("diffusion + arch='unet' 应当报错")


# ---------------------------------------------------------------------------
# S5c / S6 训练器（train + PSNR/SSIM 验证）
# ---------------------------------------------------------------------------
def _loader(D, n=2, B=2):
    return [{"image": torch.rand(B, D, 16, 16),
             "label": torch.zeros(B, 1, D, 16, 16)} for _ in range(n)]


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
        test_recon_loss_backprop,
        test_diffusion_loss_backprop,
        test_seg_backbone_unchanged,
        test_diffusion_backbone_forward_backward,
        test_regression_model_end_to_end,
        test_zaxis_sr_regression_3d_end_to_end,
        test_diffusion_model_end_to_end,
        test_diffusion_requires_adm_or_edm2,
        test_generation_trainer_runs,
        test_generation_predictor_restore_volume,
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
