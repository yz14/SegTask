"""升级批次 3（推理子系统）回归测试。

覆盖
----
1. 1-7 TTA logit 域平均：
   - ``tta_logit_average=True`` 时结果 == sigmoid(逐变体 logits 串行均值)；
   - 饱和 logits 下 logit 域与概率域结果确实不同（行为差异存在）；
   - ``tta_logit_average=False`` 保留旧概率域行为。
2. 2-5 流式分块累加（stream_accumulate）：
   - ``_StreamZAccumulator`` 与 ``_DenseZAccumulator`` 在 z 型（权重沿 z 广播）
     与 cubic 型（权重全空间）布局下逐位一致，含跳窗空隙层与 fp16；
   - 乱序（z 回退）add fail-fast；
   - 端到端：真实 Predictor 的 sliding_window_z / sliding_window_cubic 在
     stream on/off 下输出逐位一致（含 TTA、skip_empty_windows、短尾窗）。
3. 配置默认值。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from taskcore.config.core import Config  # noqa: E402
from segtask_v1.predictor import forwards as F_  # noqa: E402
from segtask_v1.predictor import sliding as S_  # noqa: E402
from segtask_v1.predictor.sliding import (  # noqa: E402
    _DenseZAccumulator, _StreamZAccumulator,
)


# ===========================================================================
# 1-7 TTA logit 域平均
# ===========================================================================
class _Net3D(nn.Module):
    def __init__(self, cin: int, cout: int, gain: float = 1.0):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, 3, padding=1)
        self.gain = gain

    def forward(self, x):
        return self.conv(x) * self.gain


class _Stub:
    def __init__(self, model, *, num_fg, patch_D, tta_logit_average,
                 tta_batch_size=None, batch_size=2):
        self.model = model
        self.model_dtype = torch.float32
        self.channels_last = False
        self.num_fg = num_fg
        self.patch_D = patch_D
        self.batch_size = batch_size
        self.tta_batch_size = tta_batch_size
        self._adabn_estimating = False
        self.tta_logit_average = tta_logit_average


_FLIPS_3D = ([2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4])


@torch.no_grad()
def _serial_logit_ref(net, x, num_fg):
    """串行参考：逐变体 logits 均值后 sigmoid。"""
    total = net(x).float()[:, :num_fg].clone()
    for fd in _FLIPS_3D:
        total = total + torch.flip(
            net(torch.flip(x, fd)).float()[:, :num_fg], fd)
    return torch.sigmoid(total / (1.0 + len(_FLIPS_3D)))


@torch.no_grad()
def _serial_prob_ref(net, x, num_fg):
    total = torch.sigmoid(net(x).float())[:, :num_fg].clone()
    for fd in _FLIPS_3D:
        total = total + torch.flip(
            torch.sigmoid(net(torch.flip(x, fd)).float())[:, :num_fg], fd)
    return total / (1.0 + len(_FLIPS_3D))


@torch.no_grad()
def test_tta_logit_average_matches_serial_logit_reference():
    torch.manual_seed(0)
    net = _Net3D(1, 2).eval()
    x = torch.randn(2, 1, 4, 8, 8)
    base_logits = net(x).float()[:, :2]
    got = F_.tta_flip_ensemble(
        _Stub(net, num_fg=2, patch_D=4, tta_logit_average=True),
        x, base_logits)
    ref = _serial_logit_ref(net, x, 2)
    assert float((got - ref).abs().max()) < 1e-6


@torch.no_grad()
def test_tta_prob_domain_preserved_when_disabled():
    torch.manual_seed(1)
    net = _Net3D(1, 2).eval()
    x = torch.randn(2, 1, 4, 8, 8)
    base_logits = net(x).float()[:, :2]
    got = F_.tta_flip_ensemble(
        _Stub(net, num_fg=2, patch_D=4, tta_logit_average=False),
        x, base_logits)
    ref = _serial_prob_ref(net, x, 2)
    assert float((got - ref).abs().max()) < 1e-6


@torch.no_grad()
def test_tta_logit_vs_prob_differ_under_saturation():
    """饱和 logits（gain=20）下，概率域均值被拉向 (k*1.0+..)/n 的饱和偏差，
    logit 域均值保留原始证据量纲——两者必须可区分。"""
    torch.manual_seed(2)
    net = _Net3D(1, 2, gain=20.0).eval()
    x = torch.randn(2, 1, 4, 8, 8)
    base_logits = net(x).float()[:, :2]
    got_logit = F_.tta_flip_ensemble(
        _Stub(net, num_fg=2, patch_D=4, tta_logit_average=True),
        x, base_logits)
    got_prob = F_.tta_flip_ensemble(
        _Stub(net, num_fg=2, patch_D=4, tta_logit_average=False),
        x, base_logits)
    assert float((got_logit - got_prob).abs().max()) > 0.05
    # 两域输出均为合法概率。
    for t in (got_logit, got_prob):
        assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0


@torch.no_grad()
def test_tta_2_5d_logit_average():
    """2.5D folded TTA 的 logit 域路径：与串行 logit 参考一致。"""
    from einops import rearrange
    torch.manual_seed(3)
    num_fg, D = 2, 4
    net = nn.Conv2d(D, num_fg * D, 3, padding=1).eval()
    x2d = torch.randn(2, D, 8, 8)

    def _logits5(inp):
        return rearrange(net(inp).float(), 'b (c d) h w -> b c d h w',
                         c=num_fg, d=D)

    base_logits = _logits5(x2d)
    total = base_logits.clone()
    for fx, fp in (([2], [3]), ([3], [4]), ([2, 3], [3, 4])):
        total = total + torch.flip(_logits5(torch.flip(x2d, fx)), fp)
    ref = torch.sigmoid(total / 4.0)

    got = F_.tta_flip_ensemble_2_5d(
        _Stub(net, num_fg=num_fg, patch_D=D, tta_logit_average=True),
        x2d, base_logits)
    assert float((got - ref).abs().max()) < 1e-6


# ===========================================================================
# 2-5 流式分块累加 — 单元级
# ===========================================================================
def _random_windows(D, pD, stride, seed=0):
    """按 z 序生成 (zs, ze) 窗口序列（含短尾窗）。"""
    rng = np.random.RandomState(seed)
    wins = []
    z = 0
    while z < D:
        ze = min(z + pD, D)
        wins.append((z, ze))
        if ze >= D:
            break
        z += stride
    return wins, rng


@pytest.mark.parametrize("weight_hw", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_stream_matches_dense_unit(weight_hw, dtype):
    num_fg, D, H, W, pD = 2, 23, 6, 5, 8
    dev = torch.device("cpu")
    dense = _DenseZAccumulator(num_fg, D, H, W, dtype, dev, weight_hw=weight_hw)
    stream = _StreamZAccumulator(num_fg, D, H, W, dtype, dev,
                                 weight_hw=weight_hw)
    wins, rng = _random_windows(D, pD, stride=4, seed=7)
    for i, (zs, ze) in enumerate(wins):
        if i == 2:      # 制造跳窗空隙：该窗不累加
            stream.flush_below(zs)
            continue
        ad = ze - zs
        sub = torch.from_numpy(
            rng.rand(num_fg, ad, H, W).astype(np.float32)).to(dtype)
        if weight_hw:
            w = torch.from_numpy(
                rng.rand(1, ad, H, W).astype(np.float32)).to(dtype)
        else:
            w = torch.from_numpy(
                rng.rand(1, ad, 1, 1).astype(np.float32)).to(dtype)
        dense.add(sub, w, zs, ze)
        stream.add(sub.clone(), w.clone(), zs, ze)
        stream.flush_below(zs)   # 保守下界：低于当前窗起点的层已定
    out_d = dense.finalize()
    out_s = stream.finalize()
    np.testing.assert_array_equal(out_s, out_d)


def test_stream_out_of_order_add_raises():
    acc = _StreamZAccumulator(1, 10, 4, 4, torch.float32,
                              torch.device("cpu"), weight_hw=False)
    sub = torch.ones(1, 3, 4, 4)
    w = torch.ones(1, 3, 1, 1)
    acc.add(sub, w, 4, 7)
    acc.flush_below(5)
    with pytest.raises(RuntimeError, match="ascending z order"):
        acc.add(sub, w, 3, 6)


def test_stream_uncovered_layers_are_zero():
    acc = _StreamZAccumulator(1, 10, 2, 2, torch.float32,
                              torch.device("cpu"), weight_hw=False)
    acc.add(torch.ones(1, 2, 2, 2), torch.ones(1, 2, 1, 1), 6, 8)
    out = acc.finalize()
    assert out[:, :6].max() == 0.0 and out[:, 8:].max() == 0.0
    np.testing.assert_allclose(out[:, 6:8], 1.0, rtol=1e-6)


def test_stream_band_stays_small():
    """流式核心收益：device 侧 band 长度应保持在 ~pD 量级而非全 D。"""
    num_fg, D, H, W, pD, stride = 1, 200, 4, 4, 16, 8
    acc = _StreamZAccumulator(num_fg, D, H, W, torch.float32,
                              torch.device("cpu"), weight_hw=False)
    wins, _ = _random_windows(D, pD, stride)
    max_band = 0
    for zs, ze in wins:
        acc.flush_below(zs)
        acc.add(torch.ones(num_fg, ze - zs, H, W),
                torch.ones(1, ze - zs, 1, 1), zs, ze)
        max_band = max(max_band, acc.pred.shape[1])
    acc.finalize()
    assert max_band <= pD + stride, max_band


# ===========================================================================
# 2-5 流式分块累加 — 端到端 parity（真实 Predictor）
# ===========================================================================
def _make_predictor(patch_mode: str, *, tta=False, skip_empty=False):
    from taskcore.models.factory import build_model
    from segtask_v1.predictor import Predictor

    cfg = Config()
    cfg.data.patch_mode = patch_mode
    cfg.data.patch_size = [16, 16, 16]
    cfg.data.label_values = [0, 1, 2]
    cfg.data.num_classes = 3
    cfg.data.multi_res_scales = [1.0]
    cfg.data.intensity_min = 0.0
    cfg.data.intensity_max = 100.0
    cfg.model.encoder_channels = [8, 16, 32]
    cfg.model.deep_supervision = False
    cfg.predict.batch_size = 2
    cfg.predict.tta_flip = tta
    cfg.predict.z_overlap = 0.5
    cfg.predict.skip_empty_windows = skip_empty
    cfg.train.use_amp = False
    cfg.sync()
    cfg.validate()
    device = torch.device("cpu")
    torch.manual_seed(11)
    model = build_model(cfg).to(device).eval()
    return Predictor(model, cfg, device)


@torch.no_grad()
def _parity(p, fn, vol):
    p.stream_accumulate = False
    p._diag_first_batch_logged = True
    dense = fn(p, vol)
    p.stream_accumulate = True
    stream = fn(p, vol)
    np.testing.assert_array_equal(stream, dense)
    return dense


@torch.no_grad()
def test_sliding_z_stream_parity():
    p = _make_predictor("z_axis")
    rng = np.random.RandomState(0)
    vol = rng.rand(37, 16, 16).astype(np.float32)   # 短尾窗 + 多窗
    out = _parity(p, S_.sliding_window_z, vol)
    assert out.shape == (2, 37, 16, 16)


@torch.no_grad()
def test_sliding_z_stream_parity_with_tta_and_skips():
    p = _make_predictor("z_axis", tta=True, skip_empty=True)
    p.skip_empty_threshold = 0.5
    rng = np.random.RandomState(1)
    vol = rng.rand(40, 16, 16).astype(np.float32)
    vol[8:24] = 0.0                                  # 中段整窗低强度 → 跳窗
    out = _parity(p, S_.sliding_window_z, vol)
    assert out.shape == (2, 40, 16, 16)


@torch.no_grad()
def test_sliding_cubic_stream_parity():
    p = _make_predictor("cubic")
    rng = np.random.RandomState(2)
    vol = rng.rand(29, 25, 22).astype(np.float32)   # 三轴短尾
    out = _parity(p, S_.sliding_window_cubic, vol)
    assert out.shape == (2, 29, 25, 22)


@torch.no_grad()
def test_sliding_cubic_stream_parity_fp16_acc():
    p = _make_predictor("cubic")
    p.acc_dtype = torch.float16
    rng = np.random.RandomState(3)
    vol = rng.rand(20, 16, 16).astype(np.float32)
    out = _parity(p, S_.sliding_window_cubic, vol)
    assert out.dtype == np.float32


# ===========================================================================
# 配置默认值
# ===========================================================================
def test_config_defaults():
    cfg = Config()
    assert cfg.predict.tta_logit_average is True
    assert cfg.predict.stream_accumulate is True
    cfg.validate()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
