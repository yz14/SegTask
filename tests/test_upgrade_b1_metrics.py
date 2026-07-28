"""升级批次 1（指标子系统）回归测试。

覆盖：
* 1-1 per-case 指标：per_case_overlap / PerCaseAggregator（mean±std / p5 / 最差 k 例、
      NaN 空类剔除、多卡行合并等价）
* 1-2 HD95：per_case_hausdorff（物理 spacing、对称性、单侧空 → NaN）
* 2-8 clDice：per_case_cldice（完美预测=1、空类 NaN、2D/3D）+ 选模 criterion 映射
* 3-7 阈值标定：ThresholdSweep（最优阈值恢复、可加性、空类回退）、
      MetricAccumulator 集成、config 校验、Predictor 消费侧
"""

import numpy as np
import pytest
import torch

from taskcore.config.core import Config, ConfigError, _CRITERION_TO_METRIC
from taskcore.metrics import (
    PerCaseAggregator,
    ThresholdSweep,
    per_case_cldice,
    per_case_hausdorff,
    per_case_overlap,
)
from segtask_v1.trainer.validation import MetricAccumulator


# ---------------------------------------------------------------------------
# 1-1 per_case_overlap
# ---------------------------------------------------------------------------
def test_per_case_overlap_known_values():
    # 类 0：pred=GT 完美；类 1：pred 8 voxel、GT 8 voxel、交 4。
    pred = torch.zeros(2, 4, 4, 4)
    tgt = torch.zeros(2, 4, 4, 4)
    pred[0, :2] = 1; tgt[0, :2] = 1
    pred[1, 0, :2, :2] = 1  # 4 voxels... 用明确交并
    pred[1, 1, :2, :2] = 1  # 共 8
    tgt[1, 1, :2, :2] = 1
    tgt[1, 2, :2, :2] = 1   # 共 8，交 4
    m = per_case_overlap(pred, tgt)
    assert m["dice"][0] == pytest.approx(1.0, abs=1e-4)
    assert m["dice"][1] == pytest.approx(2 * 4 / 16, abs=1e-4)
    assert m["iou"][1] == pytest.approx(4 / 12, abs=1e-4)


def test_per_case_overlap_empty_class_is_nan():
    pred = torch.zeros(2, 4, 4, 4)
    tgt = torch.zeros(2, 4, 4, 4)
    pred[0, 0, 0, 0] = 1; tgt[0, 0, 0, 0] = 1
    m = per_case_overlap(pred, tgt)
    assert m["dice"][0] == pytest.approx(1.0, abs=1e-4)
    assert np.isnan(m["dice"][1]) and np.isnan(m["iou"][1])


# ---------------------------------------------------------------------------
# 1-2 per_case_hausdorff
# ---------------------------------------------------------------------------
def test_hd95_physical_spacing():
    # 单 voxel 掩码沿轴 0 偏移 1 voxel，spacing_z=3mm → 对称距离恒 3mm。
    pred = torch.zeros(1, 8, 8, 8)
    tgt = torch.zeros(1, 8, 8, 8)
    pred[0, 3, 4, 4] = 1
    tgt[0, 4, 4, 4] = 1
    hd = per_case_hausdorff(pred, tgt, spacing=[3.0, 1.0, 1.0])
    assert hd[0] == pytest.approx(3.0, abs=1e-6)
    # voxel 单位（spacing=None）→ 1.0
    hd_vox = per_case_hausdorff(pred, tgt, spacing=None)
    assert hd_vox[0] == pytest.approx(1.0, abs=1e-6)


def test_hd95_symmetric_and_perfect():
    pred = torch.zeros(1, 8, 8, 8)
    pred[0, 2:5, 2:5, 2:5] = 1
    hd = per_case_hausdorff(pred, pred.clone())
    assert hd[0] == pytest.approx(0.0, abs=1e-6)
    # 对称：交换 pred/tgt 结果一致。
    tgt = torch.zeros(1, 8, 8, 8)
    tgt[0, 4:7, 4:7, 4:7] = 1
    a = per_case_hausdorff(pred, tgt)
    b = per_case_hausdorff(tgt, pred)
    assert a[0] == pytest.approx(b[0], abs=1e-6)


def test_hd95_one_sided_empty_is_nan():
    pred = torch.zeros(1, 4, 4, 4)
    tgt = torch.zeros(1, 4, 4, 4)
    tgt[0, 1, 1, 1] = 1
    assert np.isnan(per_case_hausdorff(pred, tgt)[0])
    assert np.isnan(per_case_hausdorff(tgt, pred)[0])
    assert np.isnan(per_case_hausdorff(pred, pred.clone())[0])


# ---------------------------------------------------------------------------
# 2-8 per_case_cldice
# ---------------------------------------------------------------------------
def test_cldice_perfect_and_empty():
    pred = torch.zeros(2, 12, 12, 12)
    pred[0, 6, 6, 2:10] = 1  # 细管
    cl = per_case_cldice(pred, pred.clone())
    assert cl[0] == pytest.approx(1.0, abs=1e-3)
    assert np.isnan(cl[1])  # 双侧空


def test_cldice_2d_support():
    pred = torch.zeros(1, 16, 16)
    pred[0, 8, 2:14] = 1
    cl = per_case_cldice(pred, pred.clone())
    assert cl[0] == pytest.approx(1.0, abs=1e-3)


def test_cldice_penalizes_broken_centerline():
    tgt = torch.zeros(1, 16, 16)
    tgt[0, 8, 1:15] = 1
    pred = tgt.clone()
    pred[0, 8, 7:9] = 0  # 断线
    cl = per_case_cldice(pred, tgt)
    assert cl[0] < 1.0


def test_cldice_criterion_registered():
    assert _CRITERION_TO_METRIC["cldice"] == ("mean_cldice", "max")


# ---------------------------------------------------------------------------
# 1-1 PerCaseAggregator
# ---------------------------------------------------------------------------
def test_aggregator_summary_math():
    agg = PerCaseAggregator(num_classes=1, worst_k=2)
    for v in (0.9, 0.8, 0.5, 0.2):
        agg.update({"dice": np.array([v])})
    out = agg.compute()
    vals = np.array([0.9, 0.8, 0.5, 0.2])
    assert out["case_mean_dice"] == pytest.approx(vals.mean())
    assert out["case_std_dice"] == pytest.approx(vals.std())
    assert out["case_p5_dice"] == pytest.approx(np.percentile(vals, 5))
    assert out["case_worstk_dice"] == pytest.approx((0.2 + 0.5) / 2)


def test_aggregator_nan_excluded_and_distance_direction():
    agg = PerCaseAggregator(num_classes=2, worst_k=1)
    agg.update({"hd95": np.array([2.0, np.nan])})
    agg.update({"hd95": np.array([8.0, 4.0])})
    out = agg.compute()
    assert out["case_mean_hd95"] == pytest.approx((2.0 + 8.0 + 4.0) / 3)
    # 距离类：worst-k 取最大（病例分 = nanmean 类分：2.0 与 6.0）。
    assert out["case_worstk_hd95"] == pytest.approx(6.0)
    assert "case_p95_hd95" in out and "case_p5_hd95" not in out


def test_aggregator_merge_rows_equivalent_to_single():
    a1 = PerCaseAggregator(1, worst_k=2)
    a2 = PerCaseAggregator(1, worst_k=2)
    full = PerCaseAggregator(1, worst_k=2)
    vals = [0.9, 0.7, 0.3, 0.6]
    for v in vals[:2]:
        a1.update({"dice": np.array([v])})
        full.update({"dice": np.array([v])})
    for v in vals[2:]:
        a2.update({"dice": np.array([v])})
        full.update({"dice": np.array([v])})
    a1.merge_rows(a2.raw_rows)
    assert a1.compute() == full.compute()


def test_aggregator_all_nan_metric_omitted():
    agg = PerCaseAggregator(1, worst_k=1)
    agg.update({"hd95": np.array([np.nan])})
    assert agg.compute() == {}


# ---------------------------------------------------------------------------
# 3-7 ThresholdSweep
# ---------------------------------------------------------------------------
def test_sweep_recovers_generating_threshold():
    torch.manual_seed(0)
    prob = torch.rand(1, 2, 8, 16, 16)
    target = (prob > 0.6).float()
    sw = ThresholdSweep([0.1, 0.3, 0.5, 0.6, 0.7, 0.9], 2)
    sw.update(prob, target)
    best = sw.best_thresholds()
    # target = prob>0.6：dice 最优阈值即 0.6（网格上）。
    assert best == [0.6, 0.6]


def test_sweep_grid_values_exact():
    sw = ThresholdSweep([0.1, 0.5, 0.9], 1)
    sw.update(torch.rand(1, 1, 4, 4, 4), torch.ones(1, 1, 4, 4, 4))
    for t in sw.best_thresholds():
        assert t in (0.1, 0.5, 0.9)


def test_sweep_empty_class_falls_back_to_mid():
    sw = ThresholdSweep([0.2, 0.5, 0.8], 1)
    sw.update(torch.rand(1, 1, 4, 4, 4), torch.zeros(1, 1, 4, 4, 4))
    assert sw.best_thresholds() == [0.5]


def test_sweep_additivity_matches_single_pass():
    torch.manual_seed(1)
    a = torch.rand(1, 1, 6, 8, 8); ta = (a > 0.4).float()
    b = torch.rand(1, 1, 6, 8, 8); tb = (b > 0.4).float()
    grid = [0.2, 0.4, 0.6, 0.8]
    split = ThresholdSweep(grid, 1)
    split.update(a, ta); split.update(b, tb)
    joint = ThresholdSweep(grid, 1)
    joint.update(torch.cat([a, b]), torch.cat([ta, tb]))
    for s, j in zip(split.state_tensors(), joint.state_tensors()):
        assert torch.allclose(s, j)
    assert split.best_thresholds() == joint.best_thresholds()


def test_sweep_state_tensors_sum_equals_combined():
    # 模拟多卡 all-reduce：两 rank 统计相加后 == 全集单进程。
    torch.manual_seed(2)
    a = torch.rand(1, 1, 4, 8, 8); ta = (a > 0.5).float()
    b = torch.rand(1, 1, 4, 8, 8); tb = (b > 0.5).float()
    grid = [0.3, 0.5, 0.7]
    r0 = ThresholdSweep(grid, 1); r0.update(a, ta)
    r1 = ThresholdSweep(grid, 1); r1.update(b, tb)
    for t0, t1 in zip(r0.state_tensors(), r1.state_tensors()):
        t0 += t1  # 就地求和（all_reduce_sum 语义）
    joint = ThresholdSweep(grid, 1)
    joint.update(torch.cat([a, b]), torch.cat([ta, tb]))
    assert r0.best_thresholds() == joint.best_thresholds()


# ---------------------------------------------------------------------------
# MetricAccumulator 集成
# ---------------------------------------------------------------------------
def _run_acc(**kw):
    torch.manual_seed(0)
    C = 2
    acc = MetricAccumulator(
        criterion="dice", surface_dice_tolerance=1, surface_dice_weight=0.3,
        threshold=0.5, num_fg=C, **kw)
    for _ in range(3):
        prob = torch.rand(C, 6, 12, 12)
        tgt = (torch.rand(C, 6, 12, 12) > 0.7).float()
        pred_bin = (prob > 0.5).float()
        acc.update(pred_bin.unsqueeze(0), tgt.unsqueeze(0), pred_is_binary=True)
        acc.update_sweep(prob.unsqueeze(0), tgt.unsqueeze(0))
        acc.update_case(pred_bin, tgt)
    acc.all_reduce(C, torch.device("cpu"))
    return acc, acc.compute(log=False)


def test_accumulator_per_case_and_sweep_integration():
    acc, m = _run_acc(
        per_case=True, per_case_worst_k=2, compute_hd95=True,
        compute_cldice=True, calibrate_grid=[0.25, 0.5, 0.75],
        spacing=[3.0, 0.7, 0.7])
    for key in ("case_mean_dice", "case_std_dice", "case_p5_dice",
                "case_worstk_dice", "case_mean_hd95", "case_p95_hd95",
                "case_mean_cldice", "mean_cldice"):
        assert key in m, key
    assert acc.calibrated_thresholds is not None
    assert len(acc.calibrated_thresholds) == 2
    assert all(t in (0.25, 0.5, 0.75) for t in acc.calibrated_thresholds)


def test_accumulator_disabled_paths_unchanged():
    # per_case/calibrate 关闭时：无 case_* 键、无标定，update_case/sweep 为 no-op。
    acc, m = _run_acc(per_case=False, calibrate_grid=None)
    assert not any(k.startswith("case_") for k in m)
    assert "mean_cldice" not in m
    assert acc.calibrated_thresholds is None
    assert "mean_dice" in m  # pooled 指标不受影响


# ---------------------------------------------------------------------------
# Config 校验
# ---------------------------------------------------------------------------
def _base_cfg(**train_kw):
    cfg = Config()
    cfg.data.label_values = [0, 1]
    for k, v in train_kw.items():
        setattr(cfg.train, k, v)
    cfg.sync()
    return cfg


def test_config_defaults_valid():
    cfg = _base_cfg()
    cfg.validate()
    assert cfg.train.per_case_metrics is True
    assert cfg.train.calibrate_threshold is False
    assert cfg.predict.use_calibrated_threshold is True


def test_calibrate_requires_high_mode():
    cfg = _base_cfg(calibrate_threshold=True, val_metric_mode="medium")
    with pytest.raises(ConfigError, match="calibrate_threshold"):
        cfg.validate()
    cfg2 = _base_cfg(calibrate_threshold=True, val_metric_mode="high")
    cfg2.validate()


def test_cldice_criterion_requires_compute_and_high():
    cfg = _base_cfg(save_best_criterion="cldice", val_metric_mode="high")
    with pytest.raises(ConfigError, match="cldice"):
        cfg.validate()  # compute_cldice=False
    cfg2 = _base_cfg(save_best_criterion="cldice", val_metric_mode="high",
                     compute_cldice=True)
    cfg2.validate()
    assert cfg2.train.save_best_metric == "mean_cldice"
    assert cfg2.train.save_best_mode == "max"


def test_invalid_grid_rejected():
    for bad in ([], [0.0, 0.5], [0.5, 1.0], [1.5]):
        cfg = _base_cfg(calibrate_threshold_grid=bad)
        with pytest.raises(ConfigError, match="calibrate_threshold_grid"):
            cfg.validate()


def test_per_case_worst_k_validated():
    cfg = _base_cfg(per_case_worst_k=0)
    with pytest.raises(ConfigError, match="per_case_worst_k"):
        cfg.validate()


# ---------------------------------------------------------------------------
# Predictor 消费侧（apply_calibrated_thresholds）
# ---------------------------------------------------------------------------
def _fake_predictor(num_fg=2, use_cal=True, threshold=0.5):
    """绑定真实方法的最小替身：只依赖 cfg.predict/threshold/num_fg 字段。"""
    from segtask_v1.predictor.predictor import Predictor

    class _P:
        pass
    p = _P()
    p.cfg = Config()
    p.cfg.predict.use_calibrated_threshold = use_cal
    p.threshold = threshold
    p.threshold_min = float(np.min(threshold))
    p.num_fg = num_fg
    p.threshold_calibrated = False
    p.apply_calibrated_thresholds = (
        Predictor.apply_calibrated_thresholds.__get__(p))
    return p


def test_predictor_consumes_calibrated_thresholds():
    p = _fake_predictor()
    p.apply_calibrated_thresholds({"calibrated_thresholds": [0.35, 0.6]})
    assert p.threshold == [0.35, 0.6]
    assert p.threshold_min == pytest.approx(0.35)
    assert p.threshold_calibrated is True


def test_predictor_old_ckpt_noop():
    p = _fake_predictor()
    p.apply_calibrated_thresholds({})
    assert p.threshold == 0.5 and p.threshold_calibrated is False


def test_predictor_opt_out_keeps_configured():
    p = _fake_predictor(use_cal=False, threshold=[0.4, 0.7])
    p.apply_calibrated_thresholds({"calibrated_thresholds": [0.2, 0.2]})
    assert p.threshold == [0.4, 0.7] and p.threshold_calibrated is False


def test_predictor_length_mismatch_raises():
    p = _fake_predictor(num_fg=2)
    with pytest.raises(ValueError, match="calibrated_thresholds"):
        p.apply_calibrated_thresholds({"calibrated_thresholds": [0.5]})


# ---------------------------------------------------------------------------
# Trainer 落盘侧（_ckpt_extra_state 携带标定阈值）
# ---------------------------------------------------------------------------
def test_trainer_extra_state_carries_calibration():
    from segtask_v1.trainer.trainer import Trainer

    class _Aug:
        def state_dict(self):
            return {}

    class _Pipeline:
        def __init__(self):
            from segtask_v1.losses.balancer import StaticBalancer
            self.balancer = StaticBalancer({"main": 1.0}, normalize=True)

    class _T:
        pass
    t = _T()
    t.pipeline = _Pipeline()
    t.cfg = _base_cfg()
    t.augmentor = _Aug()
    t.has_best = True
    t.patience_counter = 0
    t.calibrated_thresholds = None
    extra = Trainer._ckpt_extra_state(t)
    assert "calibrated_thresholds" not in extra
    t.calibrated_thresholds = [0.45]
    extra = Trainer._ckpt_extra_state(t)
    assert extra["calibrated_thresholds"] == [0.45]
    assert "arch_fingerprint" in extra
