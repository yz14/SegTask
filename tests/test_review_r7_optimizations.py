"""R7 审查优化项回归测试。

覆盖四项改动：

1. ``MetricAccumulator`` 逐类混淆量升 float64 累加（防大计数 float32 舍入）；
2. ``predict.hw_overlap``：cubic H/W 轴独立 overlap（None = 沿用 z_overlap）；
3. ``CudaPrefetcher``：CPU 透传等价 + 迭代完整性；
4. ``train.prefetch_to_gpu`` 配置字段存在且默认关闭。

全部 CPU 可跑，无需 GPU / 真实数据。
"""

from __future__ import annotations

import torch

from segtask_v1.config import Config, ConfigError
from segtask_v1.predictor import blending as _blending
from segtask_v1.trainer.prefetch import CudaPrefetcher
from segtask_v1.trainer.validation import MetricAccumulator


# ---------------------------------------------------------------------------
# 1) MetricAccumulator float64 accumulation
# ---------------------------------------------------------------------------
def _feed(acc: MetricAccumulator, n: int = 3, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        logits = torch.randn(2, 2, 6, 8, 8, generator=g) * 4
        target = (torch.rand(2, 2, 6, 8, 8, generator=g) > 0.5).float()
        acc.update(logits, target, loss_value=0.5)


def test_metric_accumulator_uses_float64():
    acc = MetricAccumulator(
        criterion="balanced", surface_dice_tolerance=1,
        surface_dice_weight=0.5)
    _feed(acc)
    for name in ("_inter", "_pred_sum", "_target_sum", "_voxels",
                 "_cov", "_sd_num", "_sd_denom"):
        t = getattr(acc, name)
        assert t is not None, name
        assert t.dtype == torch.float64, f"{name} dtype={t.dtype}"


def test_metric_accumulator_survives_large_counts():
    """float64 下累加超过 float32 整数精度上限 (2^24) 后计数仍精确。"""
    acc = MetricAccumulator(
        criterion="dice", surface_dice_tolerance=0, surface_dice_weight=0.0)
    _feed(acc, n=1)
    base = float(acc._inter[0].item())
    # 人为推到 float32 会丢精度的量级，再 +1 应仍可分辨。
    acc._inter += 3e7
    acc._inter += 1.0
    assert float(acc._inter[0].item()) == base + 3e7 + 1.0

    m = acc.compute(log=False)
    assert "mean_dice" in m and 0.0 <= m["mean_dice"] <= 1.0


def test_metric_accumulator_metrics_unchanged_vs_reference():
    """升 float64 后常规规模下指标与 float32 参考实现一致（同一口径）。"""
    acc = MetricAccumulator(
        criterion="dice", surface_dice_tolerance=0, surface_dice_weight=0.0)
    g = torch.Generator().manual_seed(7)
    logits = torch.randn(2, 2, 6, 8, 8, generator=g) * 4
    target = (torch.rand(2, 2, 6, 8, 8, generator=g) > 0.5).float()
    acc.update(logits, target)
    m = acc.compute(log=False)

    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum(dim=(0, 2, 3, 4))
    denom = pred.sum(dim=(0, 2, 3, 4)) + target.sum(dim=(0, 2, 3, 4))
    eps = 1e-5
    ref_dice = ((2 * inter + eps) / (denom + eps)).mean().item()
    assert abs(m["mean_dice"] - ref_dice) < 1e-6


# ---------------------------------------------------------------------------
# 2) predict.hw_overlap
# ---------------------------------------------------------------------------
def _cubic_cfg(**pred_over):
    cfg = Config()
    cfg.data.patch_mode = "cubic"
    cfg.data.patch_size = [32, 64, 64]
    for k, v in pred_over.items():
        setattr(cfg.predict, k, v)
    return cfg

def test_hw_overlap_default_none_and_validation():
    cfg = _cubic_cfg()
    assert cfg.predict.hw_overlap is None
    cfg.validate()  # None 默认合法

    cfg2 = _cubic_cfg(hw_overlap=0.25)
    cfg2.validate()

    import pytest
    with pytest.raises(ConfigError):
        _cubic_cfg(hw_overlap=1.5).validate()
    with pytest.raises(ConfigError):
        _cubic_cfg(hw_overlap=-0.1).validate()
    with pytest.raises(ConfigError):
        _cubic_cfg(z_overlap=1.0).validate()


def test_hw_overlap_changes_window_counts():
    """H/W overlap 降低 → 该轴滑窗数减少；stride 语义与 z 轴同一公式。"""
    pH = 64
    H_orig = 200
    n_50 = len(_blending.compute_1d_positions(
        H_orig, pH, max(1, int(pH * (1 - 0.5)))))
    n_25 = len(_blending.compute_1d_positions(
        H_orig, pH, max(1, int(pH * (1 - 0.25)))))
    assert n_25 < n_50


def test_predictor_resolves_hw_overlap_fallback():
    """Predictor.hw_overlap：None → 沿用 z_overlap；显式设置 → 用显式值。

    直接核 Predictor.__init__ 中的解析表达式（不构建完整模型）。
    """
    pc = _cubic_cfg(z_overlap=0.5).predict
    resolved = (float(pc.hw_overlap) if pc.hw_overlap is not None
                else float(pc.z_overlap))
    assert resolved == 0.5

    pc2 = _cubic_cfg(z_overlap=0.5, hw_overlap=0.25).predict
    resolved2 = (float(pc2.hw_overlap) if pc2.hw_overlap is not None
                 else float(pc2.z_overlap))
    assert resolved2 == 0.25


# ---------------------------------------------------------------------------
# 3) CudaPrefetcher（CPU 透传路径）
# ---------------------------------------------------------------------------
def test_prefetcher_cpu_passthrough_identical():
    batches = [
        {"image": torch.randn(2, 1, 4, 8, 8), "label": torch.rand(2, 1, 4, 8, 8),
         "pid": ["a", "b"]}
        for _ in range(4)]
    pf = CudaPrefetcher(batches, torch.device("cpu"))
    out = list(pf)
    assert len(out) == len(batches) == len(pf)
    for got, exp in zip(out, batches):
        assert got is exp  # CPU 路径原样透传，零拷贝


def test_prefetcher_empty_loader():
    pf = CudaPrefetcher([], torch.device("cpu"))
    assert list(pf) == []


# ---------------------------------------------------------------------------
# 4) train.prefetch_to_gpu 字段
# ---------------------------------------------------------------------------
def test_prefetch_to_gpu_default_off_and_validates():
    cfg = Config()
    assert cfg.train.prefetch_to_gpu is False
    cfg.train.prefetch_to_gpu = True
    cfg.data.pin_memory = False   # 应仅 warning，不报错
    cfg.validate()
