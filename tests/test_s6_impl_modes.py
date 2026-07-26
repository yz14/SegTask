"""S6-B whole_oversample_mode='pad' 与 S6-C 增强 RNG 状态入 checkpoint。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from taskcore.config import AugConfig
from taskcore.data.augment import GPUAugmentor
from taskcore.data.dataset import SegDataset3DWhole


def _write_npz(path: Path, image: np.ndarray, label: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, image=image, label=label,
             fg_slices=np.arange(image.shape[0], dtype=np.int32))
    return str(path)


def _make_whole_ds(root: Path, *, oversample: float,
                   mode: str) -> SegDataset3DWhole:
    img = np.arange(6 * 8 * 10, dtype=np.int16).reshape(6, 8, 10)
    lbl = np.ones((6, 8, 10), dtype=np.int16)
    npz = _write_npz(root / "s.npz", img, lbl)
    return SegDataset3DWhole(
        image_paths=["dummy.nii.gz"],
        label_paths=["dummy.nii.gz"],
        label_values=[0, 1],
        patch_size=(4, 6, 6),
        aug_oversample_ratio=oversample,
        normalize="minmax",
        intensity_min=-1000.0,
        intensity_max=1000.0,
        is_train=True,
        cache_enabled=False,
        npz_paths=[npz],
        oversample_mode=mode)


class TestWholeOversampleMode:

    def test_invalid_mode_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(ValueError):
                _make_whole_ds(Path(td), oversample=1.25, mode="bogus")

    def test_legacy_and_pad_same_shape(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            s_leg = _make_whole_ds(root, oversample=1.25, mode="legacy")[0]
            s_pad = _make_whole_ds(root, oversample=1.25, mode="pad")[0]
            assert s_leg["image"].shape == s_pad["image"].shape
            assert s_leg["label"].shape == s_pad["label"].shape
            assert s_leg["weight_map"].shape == s_pad["weight_map"].shape

    def test_pad_wmap_zero_on_border_ones_in_center(self):
        with tempfile.TemporaryDirectory() as td:
            s = _make_whole_ds(Path(td), oversample=1.5, mode="pad")[0]
            wm = s["weight_map"]
            assert set(wm.unique().tolist()) == {0.0, 1.0}
            # 中心 patch_size 区域权重为 1，边缘 pad 区域为 0。
            eD, eH, eW = wm.shape[1:]
            d0 = (eD - 4) // 2
            h0 = (eH - 6) // 2
            w0 = (eW - 6) // 2
            assert torch.all(
                wm[0, d0:d0 + 4, h0:h0 + 6, w0:w0 + 6] == 1.0)
            assert wm[0, 0, 0, 0] == 0.0

    def test_legacy_wmap_all_ones(self):
        with tempfile.TemporaryDirectory() as td:
            s = _make_whole_ds(Path(td), oversample=1.5, mode="legacy")[0]
            assert torch.all(s["weight_map"] == 1.0)

    def test_pad_no_oversample_is_plain_resize(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            s_pad = _make_whole_ds(root, oversample=1.0, mode="pad")[0]
            s_leg = _make_whole_ds(root, oversample=1.0, mode="legacy")[0]
            assert torch.equal(s_pad["image"], s_leg["image"])
            assert torch.equal(s_pad["label"], s_leg["label"])
            assert torch.all(s_pad["weight_map"] == 1.0)

    def test_pad_preserves_dtypes(self):
        with tempfile.TemporaryDirectory() as td:
            s = _make_whole_ds(Path(td), oversample=1.25, mode="pad")[0]
            leg = _make_whole_ds(Path(td), oversample=1.25,
                                 mode="legacy")[0]
            assert s["image"].dtype == leg["image"].dtype
            assert s["label"].dtype == leg["label"].dtype
            assert s["weight_map"].dtype == torch.float32


class TestAugmentorRngState:

    def _cfg(self) -> AugConfig:
        cfg = AugConfig(enabled=True)
        cfg.random_flip_prob = 0.5
        cfg.random_rotate_prob = 0.5
        return cfg

    def _run(self, aug: GPUAugmentor, n: int):
        outs = []
        for i in range(n):
            torch.manual_seed(1234)  # 干扰全局 RNG，确保私有 generator 独立。
            img = torch.linspace(0, 1, 2 * 1 * 4 * 6 * 6).reshape(
                2, 1, 4, 6, 6)
            lbl = torch.zeros(2, 1, 4, 6, 6)
            lbl[:, :, 1:3, 2:4, 2:4] = 1.0
            o = aug(img.clone(), lbl.clone())
            outs.append(o[0].clone())
        return outs

    def test_state_roundtrip_bit_exact(self):
        aug1 = GPUAugmentor(self._cfg(), seed=7)
        self._run(aug1, 3)
        state = aug1.state_dict()
        cont = self._run(aug1, 2)

        aug2 = GPUAugmentor(self._cfg(), seed=7)
        self._run(aug2, 3)
        aug2.load_state_dict(state)
        resumed = self._run(aug2, 2)
        for a, b in zip(cont, resumed):
            assert torch.equal(a, b)

    def test_no_seed_state_is_empty_and_load_noop(self):
        aug = GPUAugmentor(self._cfg(), seed=None)
        assert aug.state_dict() == {}
        aug.load_state_dict({})  # 不应抛错。
