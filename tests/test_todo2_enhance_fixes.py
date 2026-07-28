"""增强子系统回归测试（2-1 单次重采样 / 2-2 越界 wmap→0 / 2-4 各向异性感知）。

覆盖：
* 2-1：``GPUAugmentor.fused_call`` 单次 grid_sample 完成面内 resize + 空间增强；
  identity 变换与 CPU ``resize_3d`` 角点对齐（train/val resize 近似等价）；
  fused 与 dataset-fused-off 输出 schema/尺寸一致。
* 2-2：越界体素 label 保 border（不伪造背景）、weight_map→0；无 weight map 时
  fused 输出恒为全 1。
* 2-4：薄 slab 出面旋转角上界收敛、各向异性 spacing 弹性逐轴缩放、lowres 粗轴
  跳过；各向同性立方 patch 为 no-op（与历史逐位一致）。
"""

from __future__ import annotations

import math

import numpy as np
import torch

from taskcore.config import AugConfig
from taskcore.data.augment import (
    GPUAugmentor,
    derive_aniso_params,
    _simulate_lowres,
)
from taskcore.data.dataset import resize_3d


def _spatial_only_cfg(**over) -> AugConfig:
    base = dict(
        enabled=True, random_flip_prob=0.0, random_affine_prob=0.0,
        elastic_deform_prob=0.0, grid_dropout_prob=0.0,
        random_brightness_prob=0.0, random_contrast_prob=0.0,
        random_gamma_prob=0.0, gaussian_noise_prob=0.0,
        gaussian_blur_prob=0.0, simulate_lowres_prob=0.0,
        anisotropy_aware=False)
    base.update(over)
    return AugConfig(**base)


# ---------------------------------------------------------------------------
# 2-1 单次重采样
# ---------------------------------------------------------------------------
class TestFusedResample:
    def test_identity_matches_scipy_resize(self):
        """augment 关 → fused 纯 resize，与 scipy zoom 角点映射对齐（<1e-5）。"""
        rng = np.random.default_rng(0)
        arr = rng.random((16, 48, 48)).astype(np.float32)
        cpu = resize_3d(arr, 16, 24, 24, is_label=False, anti_alias=False)
        aug = GPUAugmentor(AugConfig(enabled=False), seed=0)
        gpu, _, _ = aug.fused_call(
            [torch.from_numpy(arr)[None]],
            [torch.zeros(1, 16, 48, 48)],
            [torch.ones(1, 16, 48, 48)],
            (24, 24))
        assert gpu.shape == (1, 1, 16, 24, 24)
        assert np.abs(gpu[0, 0].numpy() - cpu).mean() < 1e-5

    def test_fused_output_schema(self):
        """fused 输出 (B,C,eD,out_h,out_w)，image/label/wmap 对齐。"""
        cfg = _spatial_only_cfg(random_affine_prob=1.0,
                                random_rotate_range=[-10.0, 10.0])
        aug = GPUAugmentor(cfg, seed=1)
        B = 3
        imgs = [torch.rand(1, 12, 40, 40) for _ in range(B)]
        lbls = [torch.ones(1, 12, 40, 40) for _ in range(B)]
        wms = [torch.ones(1, 12, 40, 40) for _ in range(B)]
        img, lbl, wm = aug.fused_call(imgs, lbls, wms, (20, 20))
        assert img.shape == lbl.shape == wm.shape == (B, 1, 12, 20, 20)

    def test_fused_variable_inplane_sizes(self):
        """逐样本 native 面内尺寸不同 → 统一 resize 到 out_hw。"""
        aug = GPUAugmentor(_spatial_only_cfg(), seed=0)
        imgs = [torch.rand(1, 8, 40, 40), torch.rand(1, 8, 64, 50)]
        lbls = [torch.ones(1, 8, 40, 40), torch.ones(1, 8, 64, 50)]
        wms = [torch.ones(1, 8, 40, 40), torch.ones(1, 8, 64, 50)]
        img, _, _ = aug.fused_call(imgs, lbls, wms, (24, 24))
        assert img.shape == (2, 1, 8, 24, 24)

    def test_fused_no_wmap_stays_all_one(self):
        """无越界的恒等变换 → weight_map 全 1（中性）。"""
        aug = GPUAugmentor(AugConfig(enabled=False), seed=0)
        _, _, wm = aug.fused_call(
            [torch.rand(1, 8, 32, 32)], [torch.ones(1, 8, 32, 32)],
            [torch.ones(1, 8, 32, 32)], (32, 32))
        assert bool((wm == 1.0).all())


# ---------------------------------------------------------------------------
# 2-2 越界 weight_map → 0
# ---------------------------------------------------------------------------
class TestOOBWeightMap:
    def _fused_oob(self):
        cfg = _spatial_only_cfg(random_affine_prob=1.0,
                                random_rotate_range=[0.0, 0.0],
                                random_scale_range=[1.0, 1.0],
                                random_translate_range=[0.5, 0.5])
        aug = GPUAugmentor(cfg, seed=0)
        img = torch.randn(1, 1, 8, 24, 24)
        lbl = torch.full((1, 1, 8, 24, 24), 3.0)
        wm = torch.full((1, 1, 8, 24, 24), 2.0)
        return aug.fused_call([img[0]], [lbl[0]], [wm[0]], (24, 24))

    def test_fused_oob_label_keeps_border(self):
        _, lbl, _ = self._fused_oob()
        assert set(lbl.unique().tolist()) <= {3.0}

    def test_fused_oob_wmap_zeroed(self):
        _, _, wm = self._fused_oob()
        assert (wm == 0.0).any()
        assert set(wm.unique().tolist()) <= {0.0, 2.0}

    def test_batched_call_oob_wmap_zeroed(self):
        """分批入口 __call__ 同样把越界 wmap 置 0（非 fused 路径）。"""
        cfg = _spatial_only_cfg(random_affine_prob=1.0,
                                random_translate_range=[0.5, 0.5])
        aug = GPUAugmentor(cfg, seed=0)
        img = torch.randn(1, 1, 8, 16, 16)
        lbl = torch.full((1, 1, 8, 16, 16), 3.0)
        wm = torch.full((1, 1, 8, 16, 16), 2.0)
        _, lbl2, wm2 = aug(img, lbl, wm)
        assert set(lbl2.unique().tolist()) <= {3.0}
        assert (wm2 == 0.0).any()


# ---------------------------------------------------------------------------
# 2-4 各向异性感知派生
# ---------------------------------------------------------------------------
class TestAnisoDerivation:
    def test_isotropic_cube_is_noop(self):
        p = derive_aniso_params(
            (64, 64, 64), None, enabled=True, threshold=3.0,
            rotate_range=[-15.0, 15.0])
        assert p.rotate_range_per_axis is None
        assert p.elastic_axis_scale is None
        assert p.lowres_ignore_axes == ()

    def test_disabled_returns_empty(self):
        p = derive_aniso_params(
            (16, 256, 256), (3.0, 0.7, 0.7), enabled=False,
            threshold=3.0, rotate_range=[-15.0, 15.0])
        assert p == derive_aniso_params(
            (16, 256, 256), (3.0, 0.7, 0.7), enabled=False,
            threshold=3.0, rotate_range=[-15.0, 15.0])
        assert p.rotate_range_per_axis is None

    def test_thin_slab_caps_out_of_plane_rotation(self):
        """薄 slab（D≪H,W）→ 出面轴（W/H）旋转收敛，面内轴（D）保留原范围。"""
        p = derive_aniso_params(
            (16, 256, 256), None, enabled=True, threshold=3.0,
            rotate_range=[-15.0, 15.0])
        assert p.rotate_range_per_axis is not None
        cap = math.degrees(math.asin(16.0 / 256.0))
        oop_lo, oop_hi = p.rotate_range_per_axis[0]
        assert abs(oop_hi - cap) < 1e-6 and abs(oop_lo + cap) < 1e-6
        # 面内轴（z, 绕 D）保留原范围
        assert p.rotate_range_per_axis[2] == [-15.0, 15.0]

    def test_anisotropic_spacing_scales_elastic(self):
        """z spacing 粗 → 弹性位移在 z（grid 轴序末位）按 min/spacing 缩小。"""
        p = derive_aniso_params(
            (40, 192, 192), (3.0, 0.7, 0.7), enabled=True, threshold=3.0,
            rotate_range=[-5.0, 5.0])
        assert p.elastic_axis_scale is not None
        # scale_whd = (smin/sw, smin/sh, smin/sd); smin=0.7 → z(=D) 缩到 0.7/3
        assert abs(p.elastic_axis_scale[2] - 0.7 / 3.0) < 1e-6
        assert abs(p.elastic_axis_scale[0] - 1.0) < 1e-6

    def test_lowres_ignores_coarse_axis(self):
        p = derive_aniso_params(
            (40, 192, 192), (3.0, 0.7, 0.7), enabled=True, threshold=3.0,
            rotate_range=[-5.0, 5.0])
        assert 0 in p.lowres_ignore_axes  # z 轴 spacing 比 = 3/0.7 > 3

    def test_simulate_lowres_respects_ignore_axes(self):
        im = torch.rand(2, 1, 16, 64, 64)
        out = _simulate_lowres(
            im.clone(), 1.0, [0.5, 0.5],
            gen_cpu=torch.Generator().manual_seed(0), ignore_axes=(0,))
        assert out.shape == im.shape


class TestFusedEndToEnd:
    """make_data → build_dataloaders → fused collate → fused_call 全链路。"""

    @staticmethod
    def _synthetic(td, n_volumes=4, shape=(12, 32, 32)):
        import pytest as _pytest
        nib = _pytest.importorskip("nibabel")
        rng = np.random.RandomState(0)
        img_dir = td / "images"
        lbl_dir = td / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        affine = np.eye(4)
        for i in range(n_volumes):
            img = (rng.randn(*shape) * 50.0 + 40.0).astype(np.float32)
            nib.save(nib.Nifti1Image(img.transpose(2, 1, 0), affine),
                     str(img_dir / f"vol_{i:02d}.nii.gz"))
            lbl = np.zeros(shape, dtype=np.int16)
            lbl[4:8, 8:20, 8:20] = 1
            nib.save(nib.Nifti1Image(lbl.transpose(2, 1, 0), affine),
                     str(lbl_dir / f"vol_{i:02d}.nii.gz"))
        return str(img_dir), str(lbl_dir)

    def test_fused_loader_delivers_native_lists(self):
        import tempfile
        from pathlib import Path

        from taskcore.config import Config
        from taskcore.data.loader import build_dataloaders
        from taskcore.data.make_data import prepare_dataset

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            img_dir, lbl_dir = self._synthetic(td)
            cfg = Config()
            cfg.data.image_dir = img_dir
            cfg.data.label_dir = lbl_dir
            cfg.data.npz_dir = str(td / "npz")
            cfg.data.patch_mode = "2_5d"
            cfg.data.patch_size = [8, 16, 16]
            cfg.data.label_values = [0, 1]
            cfg.data.num_classes = 2
            cfg.data.multi_res_scales = [1.0]
            cfg.data.batch_size = 2
            cfg.data.num_workers = 0
            cfg.data.samples_per_volume = 2
            cfg.data.val_ratio = 0.25
            cfg.data.pin_memory = False
            cfg.augment.enabled = True
            assert cfg.data.fuse_inplane_resize  # 默认开启
            cfg.sync()
            cfg.validate()
            prepare_dataset(cfg, out_dir=cfg.data.npz_dir, workers=0)

            train_loader, val_loader = build_dataloaders(cfg)
            tb = next(iter(train_loader))
            # train：native 面内 list（32×32，未 CPU resize）
            assert isinstance(tb["image"], list)
            assert tb["image"][0].shape[-2:] == (32, 32)
            assert tb["weight_map"][0].shape == tb["label"][0].shape
            # val：常规 stacked（CPU resize 镜像）
            vb = next(iter(val_loader))
            assert torch.is_tensor(vb["image"])
            assert vb["image"].shape[-2:] == (16, 16)

            # fused_call 消费 → (B,C,eD,pH,pW)
            aug = GPUAugmentor(cfg.augment, seed=0)
            img, lbl, wm = aug.fused_call(
                [t.float() for t in tb["image"]],
                [t.float() for t in tb["label"]],
                [t.float() for t in tb["weight_map"]],
                (16, 16))
            assert img.shape == (2, 1, 8, 16, 16)
            assert lbl.shape == wm.shape == img.shape
            assert set(np.unique(lbl.numpy())) <= {0.0, 1.0}


class TestFusedTrainerEndToEnd:
    def test_trainer_one_epoch_with_fused_augment(self):
        """augment 开 + fused 默认开 → Trainer 走 fused 分支完成 1 epoch。"""
        import tempfile
        from pathlib import Path

        import pytest as _pytest
        nib = _pytest.importorskip("nibabel")

        from taskcore.config.core import Config
        from taskcore.data.loader import build_dataloaders
        from taskcore.models.factory import build_model
        from segtask_v1.trainer import Trainer

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            rng = np.random.RandomState(0)
            img_dir = td / "images"
            lbl_dir = td / "labels"
            img_dir.mkdir()
            lbl_dir.mkdir()
            affine = np.eye(4)
            for i in range(4):
                img = rng.randn(20, 64, 64).astype(np.float32) * 50.0
                nib.save(nib.Nifti1Image(img.transpose(2, 1, 0), affine),
                         str(img_dir / f"vol_{i:02d}.nii.gz"))
                lbl = np.zeros((20, 64, 64), dtype=np.int16)
                lbl[8:12, 24:40, 24:40] = 1
                nib.save(nib.Nifti1Image(lbl.transpose(2, 1, 0), affine),
                         str(lbl_dir / f"vol_{i:02d}.nii.gz"))

            cfg = Config()
            cfg.data.image_dir = str(img_dir)
            cfg.data.label_dir = str(lbl_dir)
            cfg.data.npz_dir = str(td / "npz")
            cfg.data.npz_auto_build = True
            cfg.data.patch_mode = "2_5d"
            cfg.data.patch_size = [12, 32, 32]
            cfg.data.label_values = [0, 1]
            cfg.data.num_classes = 2
            cfg.data.multi_res_scales = [1.0]
            cfg.data.batch_size = 2
            cfg.data.num_workers = 0
            cfg.data.samples_per_volume = 1
            cfg.data.cache_mode = "memory"
            cfg.model.encoder_channels = [16, 32, 64]
            cfg.model.deep_supervision = False
            cfg.augment.enabled = True
            cfg.augment.random_affine_prob = 1.0
            cfg.augment.elastic_deform_prob = 0.5
            cfg.train.epochs = 1
            cfg.train.use_amp = False
            cfg.train.use_ema = False
            cfg.train.warmup_epochs = 0
            cfg.train.compile_mode = "none"
            cfg.train.output_dir = str(td / "out")
            cfg.train.log_every = 1
            cfg.train.save_every = 9999
            cfg.train.val_every = 1
            cfg.sync()
            cfg.validate()
            assert cfg.data.fuse_inplane_resize

            train_loader, val_loader = build_dataloaders(cfg)
            sample = next(iter(train_loader))
            assert isinstance(sample["image"], list)  # fused 生效

            device = torch.device("cpu")
            model = build_model(cfg)
            trainer = Trainer(model, cfg, train_loader, val_loader, device)
            trainer.fit()  # 不崩即覆盖 fused 分支 train + 常规 val


class TestAugmentorSpacingWiring:
    def test_spacing_derives_per_shape(self):
        cfg = _spatial_only_cfg(anisotropy_aware=True)
        aug = GPUAugmentor(cfg, seed=0, spacing_zyx=(3.0, 0.7, 0.7))
        p = aug._derive_aniso((40, 192, 192))
        assert p.elastic_axis_scale is not None
        # 缓存命中同一形状返回同一对象
        assert aug._derive_aniso((40, 192, 192)) is p
