"""R5: augment 固定 seed GPU 等价性 + gen 薄封装契约。

CPU 路径见 ``test_companion_augment.py``；本文件在 CUDA 可用时验证：
1. 同 seed 两次 GPU 前向 bit-identical；
2. seg apply 与 gen 封装对 image 同 seed 空间变换一致；
3. 短程 gen 训练一步（augment ON）不炸。
"""

from __future__ import annotations

import pytest
import torch

from taskcore.config import AugConfig
from taskcore.data.augment import Companion, GPUAugmentor
from gentask.data.augment import GPUAugmentor as GenGPUAugmentor

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for R5 GPU checks")


def _spatial_cfg(**kwargs) -> AugConfig:
    cfg = AugConfig(enabled=True)
    cfg.random_brightness_prob = 0.0
    cfg.random_contrast_prob = 0.0
    cfg.random_gamma_prob = 0.0
    cfg.gaussian_noise_prob = 0.0
    cfg.gaussian_blur_prob = 0.0
    cfg.simulate_lowres_prob = 0.0
    cfg.intensity_clamp = False
    cfg.inplace = False
    cfg.random_flip_prob = 1.0
    cfg.random_flip_axes = [2, 3, 4]
    cfg.random_affine_prob = 1.0
    cfg.random_rotate_range = [-5.0, 5.0]
    cfg.random_scale_range = [0.95, 1.05]
    cfg.random_translate_range = [-0.05, 0.05]
    cfg.elastic_deform_prob = 0.0
    cfg.grid_dropout_prob = 0.0
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def test_gpu_fixed_seed_bit_identical():
    device = torch.device("cuda")
    cfg = _spatial_cfg()
    img = torch.randn(2, 1, 8, 16, 16, device=device)
    lbl = torch.randint(0, 3, (2, 1, 8, 16, 16), device=device).float()
    wm = torch.rand(2, 1, 8, 16, 16, device=device)

    a1 = GPUAugmentor(cfg, label_fill=0.0, seed=12345)
    a2 = GPUAugmentor(cfg, label_fill=0.0, seed=12345)
    o1 = a1(img.clone(), lbl.clone(), wm.clone())
    o2 = a2(img.clone(), lbl.clone(), wm.clone())
    for x, y in zip(o1, o2):
        assert torch.equal(x, y)


def test_gpu_seg_vs_gen_image_match_same_seed():
    """同 seed：seg apply(image) 与 gen(image, wm, cond) 的 image 输出一致。"""
    device = torch.device("cuda")
    cfg = _spatial_cfg(wmap_interp_mode="nearest")
    img = torch.randn(1, 1, 8, 16, 16, device=device)
    wm = torch.ones(1, 1, 8, 16, 16, device=device)
    cond = torch.randn(1, 1, 8, 16, 16, device=device)

    seg = GPUAugmentor(cfg, label_fill=0.0, seed=77)
    gen = GenGPUAugmentor(cfg, seed=77)

    img_s, _ = seg.apply(img.clone(), [])
    img_g, _, _ = gen(img.clone(), wm.clone(), cond.clone())
    assert torch.allclose(img_s, img_g, rtol=0, atol=0)


def test_gpu_short_gen_train_step_with_augment():
    """短程 sanity：gen 增强 ON + 一步 forward/backward 不炸。"""
    from gentask.config import AugConfig as GenAugConfig, Config
    from gentask.data.augment import GPUAugmentor as GAug
    from gentask.models.factory import build_model
    from gentask.losses.recon import build_recon_loss

    device = torch.device("cuda")
    cfg = Config()
    cfg.data.patch_mode = "cubic"
    cfg.data.patch_size = [8, 16, 16]
    cfg.data.multi_res_scales = [1.0]
    cfg.data.label_values = [0, 1]
    cfg.data.num_classes = 2
    cfg.model.arch = "unet"
    cfg.model.backbone = "resnet"
    cfg.model.encoder_channels = [8, 16, 32]
    cfg.model.blocks_per_level = 1
    cfg.model.stem_mode = "conv3"
    cfg.task.algorithm = "regression"
    cfg.loss.name = "l1"
    cfg.augment = GenAugConfig(enabled=True)
    cfg.augment.random_flip_prob = 1.0
    cfg.sync()

    model = build_model(cfg).to(device)
    model.train()
    loss_fn = build_recon_loss(cfg)
    aug = GAug(cfg.augment, seed=3)

    hr = torch.rand(2, 1, 8, 16, 16, device=device)
    wm = torch.ones(2, 1, 8, 16, 16, device=device)
    hr_a, wm_a, _ = aug(hr, wm, None)
    lr = torch.nn.functional.interpolate(
        hr_a, scale_factor=0.5, mode="trilinear", align_corners=False)
    lr = torch.nn.functional.interpolate(
        lr, size=hr_a.shape[-3:], mode="trilinear", align_corners=False)
    pred = model(lr)
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    loss = loss_fn(pred, hr_a, weight=wm_a)
    loss.backward()
    assert torch.isfinite(loss).item()
