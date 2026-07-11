"""Batch-4 修复回归测试。

覆盖：
1. P2-11 blending.prob_to_label 逐类阈值 eligible-mask 语义
2. P2-05 affine/elastic 越界区域 label→背景常数、weight_map→中性 1
3. P2-03 可选患者/组级划分（data.group_id_regex）
4. P3-02 验证损失键更名 val_loss → val_base_loss
5. P3-03 skip_empty_windows 跳窗比例告警
6. P2-06 pipeline 拓扑契约全组合参数化闭环（prepare_batch → loss →
   backward → extract_main_pred / split_for_metrics）

CPU-only、小张量、快速。
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from segtask_v1.config import Config, ConfigError, _CRITERION_TO_METRIC
from segtask_v1.data.augment import GPUAugmentor, _random_affine_elastic
from segtask_v1.data.loader import extract_group_ids, grouped_train_val_split
from segtask_v1.losses.losses import build_loss
from segtask_v1.predictor.blending import prob_to_label
from segtask_v1.predictor.sliding import _SKIP_RATIO_WARN, _log_skip_stats
from segtask_v1.trainer.pipelines import SupervisionPack, build_pipeline
from segtask_v1.trainer.validation import MetricAccumulator


# ===========================================================================
# 1. P2-11 — per-class threshold eligible-mask 语义
# ===========================================================================
class TestProbToLabelEligibleMask:
    LV = [0, 1, 2]  # bg=0, fg classes → label 1, 2

    def test_scalar_threshold_unchanged(self):
        # 标量阈值：argmax + 全局门控，行为保持不变。
        prob = np.zeros((2, 1, 1, 2), dtype=np.float32)
        prob[0, 0, 0, 0] = 0.7   # class0 wins voxel0
        prob[1, 0, 0, 1] = 0.4   # below thr → bg
        out = prob_to_label(prob, label_values=self.LV, num_fg=2,
                            threshold=0.5)
        assert out[0, 0, 0] == 1 and out[0, 0, 1] == 0

    def test_per_class_secondary_class_wins_when_argmax_ineligible(self):
        # 核心场景：argmax 类未过自身阈值，但次高类过了自己的阈值 →
        # 旧实现（先 argmax 再门控）输出背景；新 eligible-mask 输出次高类。
        prob = np.zeros((2, 1, 1, 1), dtype=np.float32)
        prob[0] = 0.60   # class0 prob 最高，但其阈值 0.9 → 不合格
        prob[1] = 0.55   # class1 阈值 0.5 → 合格
        out = prob_to_label(prob, label_values=self.LV, num_fg=2,
                            threshold=[0.9, 0.5])
        assert out[0, 0, 0] == 2

    def test_per_class_none_eligible_is_background(self):
        prob = np.full((2, 1, 1, 1), 0.4, dtype=np.float32)
        out = prob_to_label(prob, label_values=self.LV, num_fg=2,
                            threshold=[0.5, 0.5])
        assert out[0, 0, 0] == 0

    def test_per_class_best_among_eligible(self):
        # 两类都合格 → 取概率最高者。
        prob = np.zeros((2, 1, 1, 1), dtype=np.float32)
        prob[0] = 0.7
        prob[1] = 0.8
        out = prob_to_label(prob, label_values=self.LV, num_fg=2,
                            threshold=[0.5, 0.5])
        assert out[0, 0, 0] == 2

    def test_per_class_strictly_greater_contract(self):
        # prob == threshold 判背景（与验证侧 prob > threshold 同契约）。
        prob = np.full((1, 1, 1, 1), 0.5, dtype=np.float32)
        out = prob_to_label(prob, label_values=[0, 1], num_fg=1,
                            threshold=[0.5])
        assert out[0, 0, 0] == 0

    def test_per_class_nan_forced_background(self):
        prob = np.full((1, 1, 1, 2), 0.9, dtype=np.float32)
        prob[0, 0, 0, 1] = np.nan
        out = prob_to_label(prob, label_values=[0, 1], num_fg=1,
                            threshold=[0.5])
        assert out[0, 0, 0] == 1 and out[0, 0, 1] == 0

    def test_per_class_wrong_length_raises(self):
        prob = np.zeros((2, 1, 1, 1), dtype=np.float32)
        with pytest.raises(ValueError):
            prob_to_label(prob, label_values=self.LV, num_fg=2,
                          threshold=[0.5])


# ===========================================================================
# 2. P2-05 — 越界 label→背景、weight_map→中性 1
# ===========================================================================
class TestAffineOOBFill:
    @staticmethod
    def _run(label_fill: float):
        # 纯平移 0.5（归一化坐标）→ 体积约一半采样点越界，确定性触发。
        torch.manual_seed(0)
        B, D, H, W = 1, 8, 16, 16
        img = torch.randn(B, 1, D, H, W)
        lbl = torch.full((B, 1, D, H, W), 3.0)      # 全前景（label 值 3）
        wm = torch.full((B, 1, D, H, W), 2.0)       # 全 2 的权重图
        img2, lbl2, wm2 = _random_affine_elastic(
            img.clone(), lbl.clone(),
            affine_prob=1.0, rotate_range=[0.0, 0.0],
            scale_range=[1.0, 1.0],
            elastic_prob=0.0, sigma=5.0, alpha=0.0,
            weight_map=wm.clone(), wmap_mode="nearest",
            translate_range=[0.5, 0.5],
            label_fill=label_fill)
        return lbl2, wm2

    def test_oob_label_filled_with_background(self):
        lbl2, _ = self._run(label_fill=0.0)
        # 越界区域必须为背景 0，而不是 border 复制的前景 3。
        assert (lbl2 == 0.0).any(), "no OOB region filled — fix inactive"
        assert set(lbl2.unique().tolist()) <= {0.0, 3.0}

    def test_oob_label_fill_respects_custom_background(self):
        lbl2, _ = self._run(label_fill=7.0)
        assert (lbl2 == 7.0).any()
        assert set(lbl2.unique().tolist()) <= {7.0, 3.0}

    def test_oob_wmap_neutral_one(self):
        _, wm2 = self._run(label_fill=0.0)
        assert (wm2 == 1.0).any(), "no OOB wmap region neutralized"
        assert set(wm2.unique().tolist()) <= {1.0, 2.0}

    def test_in_bounds_transform_untouched(self):
        # 无平移/旋转/缩放（恒等变换）→ 无越界，label/wmap 逐位不变。
        B, D, H, W = 2, 4, 8, 8
        lbl = torch.randint(0, 3, (B, 1, D, H, W)).float()
        wm = torch.rand(B, 1, D, H, W) + 0.5
        img = torch.randn(B, 1, D, H, W)
        _, lbl2, wm2 = _random_affine_elastic(
            img.clone(), lbl.clone(),
            affine_prob=1.0, rotate_range=[0.0, 0.0],
            scale_range=[1.0, 1.0],
            elastic_prob=0.0, sigma=5.0, alpha=0.0,
            weight_map=wm.clone(), wmap_mode="nearest",
            label_fill=0.0)
        assert torch.equal(lbl2, lbl)
        assert torch.allclose(wm2, wm, atol=1e-5)

    def test_gpu_augmentor_accepts_label_fill(self):
        cfg = Config()
        cfg.augment.enabled = True
        aug = GPUAugmentor(cfg.augment, label_fill=5.0)
        assert aug.label_fill == 5.0


# ===========================================================================
# 3. P2-03 — 患者/组级划分
# ===========================================================================
class TestGroupAwareSplit:
    PATHS = [f"/d/P{pid:03d}_T{t}.npz"
             for pid in range(10) for t in (1, 2, 3)]
    REGEX = r"^(P\d+)"

    def test_extract_group_ids(self):
        gids = extract_group_ids(self.PATHS[:6], self.REGEX)
        assert gids == ["P000"] * 3 + ["P001"] * 3

    def test_extract_no_capture_group_uses_whole_match(self):
        gids = extract_group_ids(["/d/P007_T1.npz"], r"^P\d+")
        assert gids == ["P007"]

    def test_extract_mismatch_fail_fast(self):
        with pytest.raises(ValueError, match="does not match"):
            extract_group_ids(["/d/oops.npz"], self.REGEX)

    def test_groups_disjoint_and_complete(self):
        tr, va = grouped_train_val_split(
            self.PATHS, self.REGEX, val_ratio=0.2, seed=42)
        assert sorted(tr + va) == list(range(len(self.PATHS)))
        gids = extract_group_ids(self.PATHS, self.REGEX)
        assert not ({gids[i] for i in tr} & {gids[i] for i in va})
        # val_ratio 按组数应用：10 组 × 0.2 = 2 组 × 3 样本。
        assert len(va) == 6

    def test_deterministic_by_seed(self):
        a = grouped_train_val_split(self.PATHS, self.REGEX, 0.2, seed=1)
        b = grouped_train_val_split(self.PATHS, self.REGEX, 0.2, seed=1)
        c = grouped_train_val_split(self.PATHS, self.REGEX, 0.2, seed=2)
        assert a == b
        assert a != c

    def test_single_group_empty_val(self, caplog):
        paths = ["/d/P000_T1.npz", "/d/P000_T2.npz"]
        with caplog.at_level(logging.WARNING):
            tr, va = grouped_train_val_split(paths, self.REGEX, 0.2, seed=0)
        assert va == [] and sorted(tr) == [0, 1]
        assert any("validation set is empty" in r.message
                   for r in caplog.records)

    def test_config_default_disabled(self):
        cfg = Config()
        assert cfg.data.group_id_regex == ""
        cfg.validate()  # 默认关闭不影响校验

    def test_config_invalid_regex_rejected(self):
        cfg = Config()
        cfg.data.group_id_regex = "("
        with pytest.raises(ConfigError, match="group_id_regex"):
            cfg.validate()

    def test_config_valid_regex_accepted(self):
        cfg = Config()
        cfg.data.group_id_regex = r"^(P\d+)"
        cfg.validate()


# ===========================================================================
# 4. P3-02 — val_loss → val_base_loss
# ===========================================================================
class TestValBaseLossRename:
    def test_criterion_map_uses_new_key(self):
        assert _CRITERION_TO_METRIC["loss"] == ("val_base_loss", "min")

    def test_accumulator_emits_new_key_only(self):
        acc = MetricAccumulator(
            criterion="dice", surface_dice_tolerance=0,
            surface_dice_weight=0.5)
        pred = torch.randn(2, 2, 4, 8, 8)
        target = (torch.rand(2, 2, 4, 8, 8) > 0.5).float()
        acc.update(pred, target, loss_value=0.42)
        m = acc.compute(log=False)
        assert "val_base_loss" in m and "val_loss" not in m
        assert m["val_base_loss"] == pytest.approx(0.42)

    def test_empty_accumulator_degenerate_dict(self):
        acc = MetricAccumulator(
            criterion="dice", surface_dice_tolerance=0,
            surface_dice_weight=0.5)
        m = acc.compute(log=False)
        assert "val_base_loss" in m and "val_loss" not in m
        assert np.isnan(m["val_base_loss"])

    def test_save_best_metric_property(self):
        cfg = Config()
        cfg.train.save_best_criterion = "loss"
        assert cfg.train.save_best_metric == "val_base_loss"
        assert cfg.train.save_best_mode == "min"


# ===========================================================================
# 5. P3-03 — skip_empty_windows 比例告警
# ===========================================================================
class TestSkipRatioWarning:
    @staticmethod
    def _p(log_progress: bool):
        return SimpleNamespace(skip_empty_threshold=0.0,
                               log_progress=log_progress)

    def test_high_ratio_warns_even_without_log_progress(self, caplog):
        with caplog.at_level(logging.WARNING,
                             logger="segtask_v1.predictor.sliding"):
            _log_skip_stats(self._p(log_progress=False), 80, 100, "z")
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_low_ratio_info_only_with_log_progress(self, caplog):
        with caplog.at_level(logging.INFO,
                             logger="segtask_v1.predictor.sliding"):
            _log_skip_stats(self._p(log_progress=True), 10, 100, "cubic")
        assert any(r.levelno == logging.INFO for r in caplog.records)
        assert not any(r.levelno == logging.WARNING for r in caplog.records)

    def test_zero_skipped_silent(self, caplog):
        with caplog.at_level(logging.INFO,
                             logger="segtask_v1.predictor.sliding"):
            _log_skip_stats(self._p(log_progress=True), 0, 100, "z")
        assert not caplog.records

    def test_threshold_constant_sane(self):
        assert 0.0 < _SKIP_RATIO_WARN < 1.0


# ===========================================================================
# 6. P2-06 — 拓扑契约全组合参数化闭环
# ===========================================================================
B = 1
NUM_FG = 2
D, H, W = 4, 16, 16

# (id, patch_mode, scales, native_d, native_mr, lift, aux)
_COMBOS = [
    ("whole",          "whole",  [1.0],      False, False, False, False),
    ("z_axis",         "z_axis", [1.0],      False, False, False, False),
    ("cubic",          "cubic",  [1.0],      False, False, False, False),
    ("native_mr",      "z_axis", [1.0, 2.0], False, True,  False, False),
    ("slab",           "2_5d",   [1.0],      False, False, False, False),
    ("slab_aux",       "2_5d",   [1.0, 2.0], False, False, False, True),
    ("native_d_aux",   "2_5d",   [1.0, 2.0], True,  False, False, True),
    ("lift",           "2_5d",   [1.0],      False, False, True,  False),
    ("lift_aux",       "2_5d",   [1.0, 2.0], False, False, True,  True),
]


def _make_cfg(patch_mode, scales, native_d, native_mr, lift, aux, ds):
    cfg = Config()
    cfg.data.label_values = [0, 1, 2]
    cfg.data.num_classes = 3
    cfg.data.patch_size = [8 if native_mr else D, H, W]
    cfg.data.batch_size = 1
    cfg.data.patch_mode = patch_mode
    cfg.data.multi_res_scales = list(scales)
    cfg.data.keep_native_view_depth = native_d
    cfg.data.keep_native_multi_res = native_mr
    cfg.model.lift_2_5d_to_3d = lift
    cfg.model.aux_seg_supervision = aux
    if lift:
        cfg.model.encoder_channels = [32, 64]
    cfg.model.deep_supervision = ds
    cfg.loss.deep_supervision_weights = [1.0, 0.5] if ds else []
    if ds:
        cfg.model.encoder_channels = [32, 64, 128]
    cfg.sync()
    cfg.validate()
    return cfg


def _dataset_batch(cfg, native_mr):
    d = cfg.data.patch_size[0]
    if cfg.data.patch_mode == "2_5d" or native_mr:
        # dataset 发单个 max-FOV cube（z 轴乘 max_scale）；逐视图拆分在
        # pipeline.prepare_batch 内完成（folded 与 keep_native_* 皆然）。
        d = int(round(d * max(cfg.data.multi_res_scales)))
        c = 1
    else:
        c = len(cfg.data.multi_res_scales)
    img = torch.randn(B, c, d, H, W)
    lbl = torch.randint(0, 3, (B, c, d, H, W)).float()
    return img, lbl


def _synth_pred(pipe, cfg, model_in, sup, ds):
    """按 pipeline 类型构造与模型输出同构的合成预测（requires_grad）。"""
    d = cfg.data.patch_size[0]

    def rg(*shape):
        return torch.randn(*shape, requires_grad=True)

    name = type(pipe).__name__
    if name == "Vanilla3DPipeline":
        main = rg(B, NUM_FG, d, H, W)
    elif name == "Patch3DNativeMultiResPipeline":
        main = rg(B, pipe.n_views * NUM_FG, d, H, W)
    elif name in ("Slab2_5DPipeline", "Slab2_5DAuxPipeline",
                  "Slab2_5DNativeDPipeline"):
        main = rg(B, NUM_FG * d, H, W)
    else:  # Lift pipelines
        main = rg(B, NUM_FG, d, H, W)

    if ds:
        # 深监督：主头 + 1 个半分辨率 DS 头。
        low = torch.nn.functional.interpolate(
            main.detach(), scale_factor=0.5, mode="nearest")
        low.requires_grad_(True)
        main_path = [main, low]
        leaves = [main, low]
    else:
        main_path = main
        leaves = [main]

    aux = []
    if pipe.n_aux_views:
        for k in range(pipe.n_aux_views):
            if name == "Slab2_5DNativeDPipeline":
                d_aux = pipe.per_view_depths[k + 1]
                a = rg(B, NUM_FG * d_aux, H, W)
            elif name == "Slab2_5DAuxPipeline":
                a = rg(B, NUM_FG * d, H, W)
            else:  # Lift aux
                a = rg(B, NUM_FG, d, H, W)
            aux.append(a)
            leaves.append(a)
        return {"main": main_path, "aux": aux}, leaves
    if ds:
        return main_path, leaves
    return main, leaves


@pytest.mark.parametrize("ds", [False, True], ids=["no_ds", "ds"])
@pytest.mark.parametrize(
    "patch_mode,scales,native_d,native_mr,lift,aux",
    [c[1:] for c in _COMBOS], ids=[c[0] for c in _COMBOS])
def test_topology_closed_loop(patch_mode, scales, native_d, native_mr,
                              lift, aux, ds):
    """dataset 形状 → prepare_batch → 合成 pred → compute_loss（有限值 +
    所有输出叶子均有非零梯度）→ extract_main_pred / split_for_metrics
    契约在全部合法拓扑组合 × DS 开关下闭环成立。"""
    torch.manual_seed(0)
    cfg = _make_cfg(patch_mode, scales, native_d, native_mr, lift, aux, ds)
    pipe = build_pipeline(cfg, build_loss(cfg.loss))

    img, lbl = _dataset_batch(cfg, native_mr)
    model_in, sup = pipe.prepare_batch(img, lbl, None)
    assert isinstance(sup, SupervisionPack)
    assert torch.isfinite(model_in).all()

    pred, leaves = _synth_pred(pipe, cfg, model_in, sup, ds)

    loss = pipe.compute_loss(pred, sup)
    assert torch.isfinite(loss), f"non-finite loss for {type(pipe).__name__}"
    loss.backward()
    for i, leaf in enumerate(leaves):
        assert leaf.grad is not None and leaf.grad.norm() > 0, (
            f"dead gradient path: leaf {i} of {type(pipe).__name__}")

    # metrics 契约：extract_main_pred 得主路 tensor；split_for_metrics 输出
    # (B, num_fg, ...) 且与 target 同形。
    main = pipe.extract_main_pred(pred)
    assert isinstance(main, torch.Tensor)
    img_v, lbl_v = pipe.prepare_val_batch(img.clone(), lbl.clone())
    pred_1x, target_1x = pipe.split_for_metrics(main.detach(), lbl_v)
    assert pred_1x.shape == target_1x.shape
    assert pred_1x.shape[1] == NUM_FG
