"""Step A — TTA 批量化 等价性回归测试。

覆盖
----
1. ``tta_flip_ensemble`` (3D, 7 flip) 批量化结果与逐变体串行参考实现逐像素等价
   (max abs diff < 1e-5)。
2. ``tta_flip_ensemble_2_5d`` (2.5D folded, 3 H/W flip) 同上。
3. 批量化确实减少了前向次数：3D ``tta_batch_size=3`` → ceil(7/3)=3 次 model() 调用，
   而串行参考需 7 次。
4. ``tta_batch_size=None`` 退化为 ``batch_size``，仍逐像素等价。
5. AdaBN 估计期护栏：``_adabn_estimating=True`` 时 ``_tta_chunk_size`` 强制返回 1，
   且结果仍等价。

模型用带 BatchNorm 的小网络（eval 模式：BN 用 running stats，与 batch 构成无关），
确保"批量化只改变前向顺序/批大小、不改变数值"这一等价前提成立。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from segtask_v1.predictor import forwards as F_  # noqa: E402


# ===========================================================================
# Helpers
# ===========================================================================
class _CallCounter(nn.Module):
    """包一层 model 以统计 forward 调用次数（验证批量化确实减少前向次数）。"""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.calls = 0

    def forward(self, x):  # noqa: D401
        self.calls += 1
        return self.net(x)


class _Net3D(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, 3, padding=1)
        self.bn = nn.BatchNorm3d(cout)

    def forward(self, x):
        return self.bn(self.conv(x))


class _Net2D(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn(self.conv(x))


class _Stub:
    """仅持有 TTA 函数读取的属性。"""

    def __init__(self, model, *, num_fg, patch_D, tta_batch_size,
                 batch_size=2, adabn_estimating=False,
                 tta_logit_average=False):
        self.model = model
        self.model_dtype = torch.float32
        self.channels_last = False
        self.num_fg = num_fg
        self.patch_D = patch_D
        self.batch_size = batch_size
        self.tta_batch_size = tta_batch_size
        self._adabn_estimating = adabn_estimating
        # 本文件验批量化等价性，默认用旧概率域（与串行参考实现同域）；
        # logit 域的行为回归见 test_upgrade_b3_inference.py。
        self.tta_logit_average = tta_logit_average


def _randomize_bn(module: nn.Module, seed: int) -> None:
    """给 BN 灌入非平凡的 running stats / affine 参数，使其不是恒等映射。"""
    g = torch.Generator().manual_seed(seed)
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            c = m.num_features
            m.running_mean.copy_(torch.randn(c, generator=g) * 0.5)
            m.running_var.copy_(torch.rand(c, generator=g) + 0.5)
            m.weight.data.copy_(torch.randn(c, generator=g) * 0.3 + 1.0)
            m.bias.data.copy_(torch.randn(c, generator=g) * 0.1)


# ---- serial reference implementations (mirror the pre-batching behavior) ----
def _serial_3d(s: _Stub, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    total = base.clone()
    count = 1.0
    for fd in ([2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]):
        pred = s.model(torch.flip(x, fd))
        pf = torch.sigmoid(pred.float())[:, :s.num_fg]
        total = total + torch.flip(pf, fd)
        count += 1.0
    return total / count


def _serial_2_5d(s: _Stub, x2d: torch.Tensor,
                 base: torch.Tensor) -> torch.Tensor:
    D = s.patch_D
    total = base.clone()
    count = 1.0
    for fx, fp in (([2], [3]), ([3], [4]), ([2, 3], [3, 4])):
        pred = s.model(torch.flip(x2d, fx))
        pred5 = rearrange(pred, 'b (c d) h w -> b c d h w', c=s.num_fg, d=D)
        pf = torch.sigmoid(pred5.float())
        total = total + torch.flip(pf, fp)
        count += 1.0
    return total / count


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


# ===========================================================================
# 3D
# ===========================================================================
@torch.no_grad()
def test_tta_3d_batched_matches_serial():
    torch.manual_seed(0)
    num_fg, pD, B = 2, 4, 2
    net = _Net3D(cin=1, cout=num_fg)
    _randomize_bn(net, seed=1)
    net.eval()

    x = torch.randn(B, 1, pD, 8, 8)
    base_logits = net(x).float()[:, :num_fg]
    ref = _serial_3d(_Stub(net, num_fg=num_fg, patch_D=pD,
                           tta_batch_size=1), x, torch.sigmoid(base_logits))

    for tbs in (2, 3, 8):
        counter = _CallCounter(net).eval()
        s = _Stub(counter, num_fg=num_fg, patch_D=pD, tta_batch_size=tbs)
        got = F_.tta_flip_ensemble(s, x, base_logits)
        d = _maxdiff(got, ref)
        if d >= 1e-5:
            raise AssertionError(
                f"3D TTA batched(tbs={tbs}) vs serial max abs diff={d:.3g}")
        expected_calls = (7 + tbs - 1) // tbs
        assert counter.calls == expected_calls, (
            f"tbs={tbs}: expected {expected_calls} model calls, "
            f"got {counter.calls}")


@torch.no_grad()
def test_tta_3d_none_falls_back_to_batch_size():
    torch.manual_seed(2)
    num_fg, pD, B = 2, 4, 2
    net = _Net3D(cin=1, cout=num_fg)
    _randomize_bn(net, seed=3)
    net.eval()

    x = torch.randn(B, 1, pD, 8, 8)
    base_logits = net(x).float()[:, :num_fg]
    ref = _serial_3d(_Stub(net, num_fg=num_fg, patch_D=pD,
                           tta_batch_size=1), x, torch.sigmoid(base_logits))

    counter = _CallCounter(net).eval()
    s = _Stub(counter, num_fg=num_fg, patch_D=pD,
              tta_batch_size=None, batch_size=4)
    got = F_.tta_flip_ensemble(s, x, base_logits)
    assert _maxdiff(got, ref) < 1e-5
    # None → batch_size=4 → ceil(7/4)=2 forwards
    assert counter.calls == 2, counter.calls


# ===========================================================================
# 2.5D
# ===========================================================================
@torch.no_grad()
def test_tta_2_5d_batched_matches_serial():
    torch.manual_seed(4)
    num_fg, pD, B = 2, 4, 2
    cin = pD                       # folded single-view layout (C_res*D = 1*D)
    net = _Net2D(cin=cin, cout=num_fg * pD)
    _randomize_bn(net, seed=5)
    net.eval()

    x2d = torch.randn(B, cin, 8, 8)
    pred = net(x2d)
    base_logits = rearrange(pred.float(), 'b (c d) h w -> b c d h w',
                            c=num_fg, d=pD)
    ref = _serial_2_5d(_Stub(net, num_fg=num_fg, patch_D=pD,
                             tta_batch_size=1), x2d,
                       torch.sigmoid(base_logits))

    for tbs in (2, 3):
        counter = _CallCounter(net).eval()
        s = _Stub(counter, num_fg=num_fg, patch_D=pD, tta_batch_size=tbs)
        got = F_.tta_flip_ensemble_2_5d(s, x2d, base_logits)
        d = _maxdiff(got, ref)
        if d >= 1e-5:
            raise AssertionError(
                f"2.5D TTA batched(tbs={tbs}) vs serial max abs diff={d:.3g}")
        expected_calls = (3 + tbs - 1) // tbs
        assert counter.calls == expected_calls, (
            f"tbs={tbs}: expected {expected_calls} calls, got {counter.calls}")


# ===========================================================================
# AdaBN estimation guard
# ===========================================================================
@torch.no_grad()
def test_adabn_estimating_forces_serial_chunk():
    net = _Net3D(cin=1, cout=2).eval()
    s = _Stub(net, num_fg=2, patch_D=4, tta_batch_size=8,
              adabn_estimating=True)
    assert F_._tta_chunk_size(s) == 1, (
        "AdaBN 估计期必须强制 chunk=1（串行）")
    s._adabn_estimating = False
    assert F_._tta_chunk_size(s) == 8


@torch.no_grad()
def test_adabn_estimating_still_equivalent():
    """估计期(chunk=1)结果也应与串行参考等价（eval BN 下恒等价）。"""
    torch.manual_seed(6)
    num_fg, pD, B = 2, 4, 2
    net = _Net3D(cin=1, cout=num_fg)
    _randomize_bn(net, seed=7)
    net.eval()

    x = torch.randn(B, 1, pD, 8, 8)
    base_logits = net(x).float()[:, :num_fg]
    ref = _serial_3d(_Stub(net, num_fg=num_fg, patch_D=pD,
                           tta_batch_size=1), x, torch.sigmoid(base_logits))

    counter = _CallCounter(net).eval()
    s = _Stub(counter, num_fg=num_fg, patch_D=pD, tta_batch_size=8,
              adabn_estimating=True)
    got = F_.tta_flip_ensemble(s, x, base_logits)
    assert _maxdiff(got, ref) < 1e-5
    assert counter.calls == 7, counter.calls  # chunk=1 → 7 serial forwards


# ===========================================================================
# Driver
# ===========================================================================
if __name__ == "__main__":
    tests = [
        test_tta_3d_batched_matches_serial,
        test_tta_3d_none_falls_back_to_batch_size,
        test_tta_2_5d_batched_matches_serial,
        test_adabn_estimating_forces_serial_chunk,
        test_adabn_estimating_still_equivalent,
    ]
    n_pass = 0
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {exc}")
        else:
            n_pass += 1
            print(f"  ok    {t.__name__}")
    print(f"\n{n_pass}/{len(tests)} passed")
    if n_pass != len(tests):
        sys.exit(1)
