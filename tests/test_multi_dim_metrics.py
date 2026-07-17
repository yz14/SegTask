"""新增 pooled 多维度选模指标的单测：

覆盖三件事
----------
1. ``dice_batch_stats`` 新增字段 (``pred_sum`` / ``target_sum`` / ``voxels``)
   与已有字段满足恒等式：``denom == pred_sum + target_sum``；``voxels``
   等于 ``B * spatial_numel``。
2. ``derive_overlap_metrics`` 闭式输出与手工 numpy 计算逐 ε 一致：
   dice / iou / recall / precision / vol_sim / mcc 全覆盖；并验证 dice
   与现有 pooled-dice 公式 (2·inter+ε)/(denom+ε) 完全数值一致（向后兼容
   保证：换用新闭式不影响线上已选模型的轨迹）。
3. pooled 分子/分母对 reduction shape 不变（per_slice vs per_volume），
   等价于现有 ``test_pooled_dice_metric_invariant_under_reduction``，但
   覆盖到新增字段。
4. ``harmonic_mean_metrics`` 行为：全 1 ≈ 1，含 0 ≈ 0，常规中间值与
   解析调和均值一致。
5. ``Config.sync()`` 把新增 ``save_best_criterion`` 选项正确映射到
   ``(save_best_metric, save_best_mode)``，且 ``validate()`` 接受这些
   值；非法值仍报错。
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# 1. dice_batch_stats — 字段新增 & 恒等式
# ---------------------------------------------------------------------------
def test_dice_batch_stats_new_keys_identities():
    from taskcore.utils.common import dice_batch_stats

    torch.manual_seed(0)
    B, C, D, H, W = 2, 3, 4, 8, 8
    pred = torch.randn(B, C, D, H, W) * 2
    target = (torch.rand(B, C, D, H, W) > 0.7).float()

    stats = dice_batch_stats(pred, target)

    # 新键存在。
    for k in ("pred_sum", "target_sum", "voxels"):
        assert k in stats, f"missing key: {k}"

    # 形状。
    assert stats["pred_sum"].shape == (C,)
    assert stats["target_sum"].shape == (C,)
    assert stats["voxels"].ndim == 0  # scalar

    # denom 恒等式（新字段必须与旧 denom 自洽）。
    torch.testing.assert_close(
        stats["denom"], stats["pred_sum"] + stats["target_sum"],
        atol=1e-5, rtol=1e-5)

    # voxels = B * spatial_numel（dtype 兼容比较）。
    expected_voxels = float(B * D * H * W)
    assert math.isclose(float(stats["voxels"].item()), expected_voxels, rel_tol=0, abs_tol=0.0)


# ---------------------------------------------------------------------------
# 2. derive_overlap_metrics — 与 numpy 手工实现比对
# ---------------------------------------------------------------------------
def _np_metrics(tp, fp, fn, tn, eps=1e-5):
    """numpy 参考实现。返回 dict[name -> np.ndarray (C,)]。"""
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    recall = (tp + eps) / (tp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    vol_sim = 1.0 - np.abs(fp - fn) / (2 * tp + fp + fn + eps)
    den2 = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = np.where(
        den2 > 0,
        (tp * tn - fp * fn) / np.sqrt(np.maximum(den2, eps)),
        np.zeros_like(den2, dtype=np.float64))
    return dict(dice=dice, iou=iou, recall=recall, precision=precision,
                vol_sim=vol_sim, mcc=mcc)


def test_derive_overlap_metrics_matches_numpy():
    from taskcore.utils.common import derive_overlap_metrics

    # 显式构造已知混淆矩阵（3 个类，便于核对）。
    # 类 0：完全匹配；类 1：纯过分割；类 2：纯欠分割。
    tp = np.array([100.0, 50.0, 30.0])
    fp = np.array([0.0, 50.0, 0.0])
    fn = np.array([0.0, 0.0, 70.0])
    voxels_total = 1000.0
    tn = voxels_total - tp - fp - fn

    inter = torch.tensor(tp, dtype=torch.float32)
    pred_sum = torch.tensor(tp + fp, dtype=torch.float32)
    target_sum = torch.tensor(tp + fn, dtype=torch.float32)
    voxels = torch.tensor(voxels_total, dtype=torch.float64)

    out = derive_overlap_metrics(inter, pred_sum, target_sum, voxels)
    ref = _np_metrics(tp, fp, fn, tn)

    for k, ref_val in ref.items():
        got = out[k].numpy()
        np.testing.assert_allclose(got, ref_val, rtol=1e-4, atol=1e-5,
                                   err_msg=f"metric {k!r} mismatch")

    # 类 0 应给出近 1 的 dice/iou/mcc。
    assert out["dice"][0].item() > 0.999
    assert out["iou"][0].item() > 0.999
    assert out["mcc"][0].item() > 0.999

    # 类 1（仅 FP）应有 recall=1，precision<1，vol_sim<1。
    assert out["recall"][1].item() > 0.999
    assert out["precision"][1].item() < 0.6
    assert out["vol_sim"][1].item() < 1.0

    # 类 2（仅 FN）应有 precision=1，recall<1。
    assert out["precision"][2].item() > 0.999
    assert out["recall"][2].item() < 0.5


def test_derive_overlap_metrics_matches_pooled_dice_formula():
    """新闭式 dice 与历史 (2·inter+ε)/(denom+ε) 数值一致 — 不破坏已有 val 曲线。"""
    from taskcore.utils.common import derive_overlap_metrics, dice_batch_stats

    torch.manual_seed(1)
    B, C, D, H, W = 3, 2, 4, 8, 8
    pred = torch.randn(B, C, D, H, W)
    target = (torch.rand(B, C, D, H, W) > 0.6).float()

    stats = dice_batch_stats(pred, target)
    derived = derive_overlap_metrics(
        stats["inter"], stats["pred_sum"],
        stats["target_sum"], stats["voxels"])

    smooth = 1e-5
    legacy_dice = (2.0 * stats["inter"] + smooth) / (stats["denom"] + smooth)
    torch.testing.assert_close(
        derived["dice"], legacy_dice.float(), atol=1e-5, rtol=1e-5)


def test_derive_overlap_metrics_empty_class_no_nan():
    """既无 GT 又无 pred 的类必须返回有限值（dice/iou=1 因平滑，mcc=0）。"""
    from taskcore.utils.common import derive_overlap_metrics

    inter = torch.zeros(2)
    pred_sum = torch.zeros(2)
    target_sum = torch.zeros(2)
    voxels = torch.tensor(1000.0, dtype=torch.float64)

    out = derive_overlap_metrics(inter, pred_sum, target_sum, voxels)
    for k, v in out.items():
        assert torch.isfinite(v).all(), f"{k} contains NaN/Inf: {v}"


# ---------------------------------------------------------------------------
# 3. 新字段 reduction-invariance
# ---------------------------------------------------------------------------
def test_new_stats_invariant_under_reduction():
    """per_slice (B*D,C,H,W) 与 per_volume (B,C,D,H,W) 累加和必须一致。"""
    from taskcore.utils.common import dice_batch_stats

    torch.manual_seed(2)
    B, C, D, H, W = 2, 2, 3, 16, 16
    pred_v = torch.randn(B, C, D, H, W)
    target_v = (torch.rand(B, C, D, H, W) > 0.5).float()

    # per_slice 视角：把 D 维并到 batch。
    pred_s = pred_v.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
    target_s = target_v.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)

    s_v = dice_batch_stats(pred_v, target_v)
    s_s = dice_batch_stats(pred_s, target_s)

    for k in ("inter", "denom", "pred_sum", "target_sum"):
        torch.testing.assert_close(s_v[k], s_s[k], atol=1e-5, rtol=1e-5,
                                   msg=f"stat {k!r} differs across reductions")
    assert math.isclose(float(s_v["voxels"]), float(s_s["voxels"]),
                        rel_tol=0, abs_tol=0.0)


# ---------------------------------------------------------------------------
# 4. harmonic_mean_metrics 行为
# ---------------------------------------------------------------------------
def test_harmonic_mean_metrics_extremes_and_mid():
    from taskcore.utils.common import harmonic_mean_metrics

    # 全 1 ≈ 1。
    out_one = harmonic_mean_metrics([torch.tensor(1.0)] * 4).item()
    assert abs(out_one - 1.0) < 1e-4

    # 含 0：调和均值应 → 0。
    out_zero = harmonic_mean_metrics([
        torch.tensor(1.0), torch.tensor(1.0),
        torch.tensor(1.0), torch.tensor(0.0)]).item()
    assert out_zero < 1e-3, f"harmonic mean with a zero should be ~0, got {out_zero}"

    # 解析比对：H(0.5, 0.5, 0.5, 0.5) = 0.5。
    out_half = harmonic_mean_metrics([torch.tensor(0.5)] * 4).item()
    assert abs(out_half - 0.5) < 1e-3

    # 解析比对：H(0.8, 0.6, 0.4, 0.2) ≈ 0.384。
    vals = [0.8, 0.6, 0.4, 0.2]
    ref = len(vals) / sum(1.0 / v for v in vals)
    out_mix = harmonic_mean_metrics([torch.tensor(v) for v in vals]).item()
    assert abs(out_mix - ref) < 5e-3, f"got {out_mix}, ref {ref}"


# ---------------------------------------------------------------------------
# 5. Config 新选模标准映射 & 校验
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("crit,expected_metric,expected_mode", [
    ("loss",              "val_loss",       "min"),
    ("dice",              "mean_dice",      "max"),
    ("dice+surface_dice", "mean_combined",  "max"),
    ("iou",               "mean_iou",       "max"),
    ("mcc",               "mean_mcc",       "max"),
    ("min_dice",          "min_class_dice", "max"),
    ("balanced",          "mean_balanced",  "max"),
])
def test_save_best_criterion_mapping(crit, expected_metric, expected_mode):
    from taskcore.config.core import Config

    cfg = Config()
    cfg.train.save_best_criterion = crit
    cfg.sync()
    assert cfg.train.save_best_metric == expected_metric
    assert cfg.train.save_best_mode == expected_mode


def test_save_best_criterion_invalid_rejected():
    from taskcore.config.core import Config

    cfg = Config()
    cfg.train.save_best_criterion = "not_a_real_metric"
    cfg.sync()  # sync 本身不报错（只映射），validate() 报错。
    with pytest.raises(AssertionError, match="save_best_criterion"):
        cfg.validate()


# ---------------------------------------------------------------------------
# 6. 任务化推荐预设：save_best_preset → 覆盖底层字段
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("preset,exp_crit,exp_tol,exp_w,exp_metric,exp_mode", [
    ("lung",               "dice+surface_dice", 1, 0.5, "mean_combined",  "max"),
    ("vessel",             "balanced",          2, 0.5, "mean_balanced",  "max"),
    ("airway",             "balanced",          1, 0.5, "mean_balanced",  "max"),
    ("bone_multi",         "min_dice",          1, 0.5, "min_class_dice", "max"),
    ("lymph_node",         "mcc",               1, 0.5, "mean_mcc",       "max"),
    ("lesion_small",       "mcc",               1, 0.5, "mean_mcc",       "max"),
    ("oar_multi",          "min_dice",          1, 0.5, "min_class_dice", "max"),
    ("heart_chamber",      "dice+surface_dice", 1, 0.4, "mean_combined",  "max"),
    ("bone_lung_combined", "balanced",          1, 0.5, "mean_balanced",  "max"),
])
def test_save_best_preset_expands(
        preset, exp_crit, exp_tol, exp_w, exp_metric, exp_mode):
    """preset 必须覆盖 (criterion, sd_tol, sd_w) 并联动 (metric, mode)。"""
    from taskcore.config.core import Config

    cfg = Config()
    # 先显式塞入完全不同的值，证明 preset 真的覆盖了它们。
    cfg.train.save_best_criterion    = "loss"
    cfg.train.surface_dice_tolerance = 7
    cfg.train.surface_dice_weight    = 0.9
    cfg.train.save_best_preset = preset
    cfg.sync()
    assert cfg.train.save_best_criterion    == exp_crit
    assert cfg.train.surface_dice_tolerance == exp_tol
    assert abs(cfg.train.surface_dice_weight - exp_w) < 1e-9
    assert cfg.train.save_best_metric == exp_metric
    assert cfg.train.save_best_mode   == exp_mode
    cfg.validate()  # validate 必须接受 preset。


def test_save_best_preset_empty_is_noop():
    """空 preset 不得修改用户显式设置（向后兼容必须）。"""
    from taskcore.config.core import Config

    cfg = Config()
    cfg.train.save_best_preset       = ""  # 默认空。
    cfg.train.save_best_criterion    = "mcc"
    cfg.train.surface_dice_tolerance = 3
    cfg.train.surface_dice_weight    = 0.7
    cfg.sync()
    assert cfg.train.save_best_criterion    == "mcc"
    assert cfg.train.surface_dice_tolerance == 3
    assert abs(cfg.train.surface_dice_weight - 0.7) < 1e-9
    # criterion → metric/mode 仍正常解析。
    assert cfg.train.save_best_metric == "mean_mcc"
    assert cfg.train.save_best_mode   == "max"


def test_save_best_preset_case_insensitive_and_trim():
    """preset 名应大小写/空白不敏感（yaml 编辑常误带空格）。"""
    from taskcore.config.core import Config

    cfg = Config()
    cfg.train.save_best_preset = "  Vessel  "
    cfg.sync()
    assert cfg.train.save_best_criterion == "balanced"
    assert cfg.train.surface_dice_tolerance == 2


def test_save_best_preset_invalid_rejected():
    from taskcore.config.core import Config

    cfg = Config()
    cfg.train.save_best_preset = "definitely_not_a_preset"
    cfg.sync()  # sync 不报错（容错），validate 报错。
    with pytest.raises(AssertionError, match="save_best_preset"):
        cfg.validate()


def test_save_best_preset_yaml_roundtrip(tmp_path):
    """preset 必须能完整经过 yaml dump/load 后保留。"""
    import yaml

    from taskcore.config.core import Config, load_config

    cfg = Config()
    cfg.train.save_best_preset = "lung"
    cfg.sync()

    # 仅写出 train 段中我们关心的字段，模拟用户写的最小 yaml。
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "train:\n  save_best_preset: lung\n", encoding="utf-8")
    loaded = load_config(str(yaml_path))
    assert loaded.train.save_best_preset    == "lung"
    assert loaded.train.save_best_criterion == "dice+surface_dice"
    assert loaded.train.surface_dice_tolerance == 1
    assert abs(loaded.train.surface_dice_weight - 0.5) < 1e-9
    assert loaded.train.save_best_metric == "mean_combined"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
