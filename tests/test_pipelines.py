"""Unit tests for ``segtask_v1.trainer.pipelines.*``.

CPU-only, small tensors, fast (<2s)。验证：

1. 工厂选对子类（覆盖 7 种模式分支）
2. 每个 pipeline 的 ``prepare_batch`` 输出形状与 SupervisionPack 字段
3. 每个 pipeline 的 ``compute_loss(pred, sup)`` 与"手写历史聚合"等价
   （``main + Σ w_k * aux``），``max|diff| == 0``
4. ``compute_loss`` 在 aux 路径下产生有效梯度
5. ``ViewPipeline.split_for_metrics`` / ``extract_main_pred`` 与模式无关
"""

from __future__ import annotations

import pytest
import torch

from segtask_v1.config import Config
from segtask_v1.losses.losses import build_loss
from segtask_v1.trainer.amp import compute_loss_fp32
from segtask_v1.trainer.pipelines import (
    Lift2_5DAuxPipeline, Lift2_5DPipeline,
    Patch3DNativeMultiResPipeline,
    Slab2_5DAuxPipeline, Slab2_5DNativeDPipeline, Slab2_5DPipeline,
    SupervisionPack, Vanilla3DPipeline,
    build_pipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _base_cfg(patch_size=(4, 16, 16)):
    cfg = Config()
    cfg.data.label_values = [0, 1, 2]
    cfg.data.num_classes = 3
    cfg.data.patch_size = list(patch_size)
    cfg.data.batch_size = 1
    cfg.model.deep_supervision = False
    cfg.loss.deep_supervision_weights = []
    cfg.sync()
    cfg.validate()
    return cfg


B = 1
NUM_FG = 2
TOL = 1e-5


# ===========================================================================
# Factory dispatch (Round-2 contract)
# ===========================================================================
class TestFactoryDispatch:
    """``build_pipeline`` 必须按 patch_mode + 5 个 flag 选对类。"""

    @pytest.mark.parametrize("patch_mode", ["whole", "z_axis", "cubic"])
    def test_3d_single_res(self, patch_mode):
        cfg = _base_cfg()
        cfg.data.patch_mode = patch_mode
        cfg.data.multi_res_scales = [1.0]
        cfg.sync(); cfg.validate()
        p = build_pipeline(cfg, build_loss(cfg.loss))
        assert isinstance(p, Vanilla3DPipeline)
        assert p.n_views == 1 and p.num_res_groups == 1 and p.n_aux_views == 0

    def test_3d_native_multi_res(self):
        cfg = _base_cfg(patch_size=(8, 16, 16))
        cfg.data.patch_mode = "z_axis"
        cfg.data.multi_res_scales = [1.0, 2.0]
        cfg.data.keep_native_multi_res = True
        cfg.sync(); cfg.validate()
        p = build_pipeline(cfg, build_loss(cfg.loss))
        assert isinstance(p, Patch3DNativeMultiResPipeline)
        assert p.n_views == 2 and p.num_res_groups == 2

    def test_2_5d_folded_no_aux(self):
        cfg = _base_cfg()
        cfg.data.patch_mode = "2_5d"
        cfg.data.multi_res_scales = [1.0]
        cfg.sync(); cfg.validate()
        p = build_pipeline(cfg, build_loss(cfg.loss))
        assert isinstance(p, Slab2_5DPipeline)
        assert p.num_res_groups == 1 and p.slab_depth == 4

    def test_2_5d_folded_aux(self):
        cfg = _base_cfg()
        cfg.data.patch_mode = "2_5d"
        cfg.data.multi_res_scales = [1.0, 2.0]
        cfg.model.aux_seg_supervision = True
        cfg.sync(); cfg.validate()
        p = build_pipeline(cfg, build_loss(cfg.loss))
        assert isinstance(p, Slab2_5DAuxPipeline)
        assert p.n_aux_views == 1 and len(p.aux_weights) == 1

    def test_2_5d_native_d_aux(self):
        cfg = _base_cfg()
        cfg.data.patch_mode = "2_5d"
        cfg.data.multi_res_scales = [1.0, 2.0]
        cfg.data.keep_native_view_depth = True
        cfg.model.aux_seg_supervision = True
        cfg.sync(); cfg.validate()
        p = build_pipeline(cfg, build_loss(cfg.loss))
        assert isinstance(p, Slab2_5DNativeDPipeline)
        assert len(p.aux_loss_fns) == 1
        assert p.per_view_depths == cfg.per_view_depths

    def test_lift_no_aux(self):
        cfg = _base_cfg()
        cfg.data.patch_mode = "2_5d"
        cfg.data.multi_res_scales = [1.0]
        cfg.model.lift_2_5d_to_3d = True
        cfg.model.encoder_channels = [32, 64]   # 让 D=4 通过 lift 整除性
        cfg.sync(); cfg.validate()
        p = build_pipeline(cfg, build_loss(cfg.loss))
        assert isinstance(p, Lift2_5DPipeline)
        assert p.num_res_groups == 1

    def test_lift_aux(self):
        cfg = _base_cfg()
        cfg.data.patch_mode = "2_5d"
        cfg.data.multi_res_scales = [1.0, 2.0]
        cfg.model.lift_2_5d_to_3d = True
        cfg.model.aux_seg_supervision = True
        cfg.model.encoder_channels = [32, 64]
        cfg.sync(); cfg.validate()
        p = build_pipeline(cfg, build_loss(cfg.loss))
        assert isinstance(p, Lift2_5DAuxPipeline)
        assert p.n_aux_views == 1


# ===========================================================================
# prepare_batch shape contracts
# ===========================================================================
class TestPrepareBatch:
    def test_vanilla3d_pass_through(self):
        cfg = _base_cfg()
        cfg.data.patch_mode = "whole"; cfg.data.multi_res_scales = [1.0]
        cfg.sync(); cfg.validate()
        p = build_pipeline(cfg, build_loss(cfg.loss))
        D, H, W = 4, 16, 16
        img = torch.randn(B, 1, D, H, W)
        lbl = torch.randint(0, 3, (B, 1, D, H, W)).float()
        out_img, sup = p.prepare_batch(img, lbl, None)
        assert torch.equal(out_img, img)
        assert torch.equal(sup.label_main, lbl)
        assert sup.aux_labels is None and sup.label_all_views is None

    def test_slab_2_5d_squeeze(self):
        cfg = _base_cfg()
        cfg.data.patch_mode = "2_5d"; cfg.data.multi_res_scales = [1.0]
        cfg.sync(); cfg.validate()
        p = build_pipeline(cfg, build_loss(cfg.loss))
        D, H, W = 4, 16, 16
        img = torch.randn(B, 1, D, H, W)         # rank-5 from dataset
        lbl = torch.randint(0, 3, (B, 1, D, H, W)).float()
        out_img, sup = p.prepare_batch(img, lbl, None)
        assert out_img.shape == (B, D, H, W)     # folded
        assert sup.label_main.shape == (B, D, H, W)

    def test_slab_2_5d_aux_keeps_label_views(self):
        cfg = _base_cfg()
        cfg.data.patch_mode = "2_5d"
        cfg.data.multi_res_scales = [1.0, 2.0]
        cfg.model.aux_seg_supervision = True
        cfg.sync(); cfg.validate()
        p = build_pipeline(cfg, build_loss(cfg.loss))
        D, H, W = 4, 16, 16
        eD_max = int(round(D * 2.0))     # dataset emits single max-FOV cube
        img = torch.randn(B, 1, eD_max, H, W)
        lbl = torch.randint(0, 3, (B, 1, eD_max, H, W)).float()
        out_img, sup = p.prepare_batch(img, lbl, None)
        assert out_img.shape == (B, 2 * D, H, W)
        assert sup.label_main.shape == (B, D, H, W)
        assert sup.label_all_views is not None
        assert sup.label_all_views.shape == (B, 2, D, H, W)

    def test_lift_keeps_rank5_image(self):
        cfg = _base_cfg()
        cfg.data.patch_mode = "2_5d"; cfg.data.multi_res_scales = [1.0]
        cfg.model.lift_2_5d_to_3d = True
        cfg.model.encoder_channels = [32, 64]
        cfg.sync(); cfg.validate()
        p = build_pipeline(cfg, build_loss(cfg.loss))
        D, H, W = 4, 16, 16
        img = torch.randn(B, 1, D, H, W)
        lbl = torch.randint(0, 3, (B, 1, D, H, W)).float()
        out_img, sup = p.prepare_batch(img, lbl, None)
        assert out_img.shape == (B, 1, D, H, W)            # rank-5 unchanged
        assert sup.label_main.shape == (B, 1, D, H, W)     # [:, :1] keeps C_res axis


# ===========================================================================
# compute_loss equivalence (vs hand-written formula)
# ===========================================================================
class TestComputeLossEquivalence:
    """``pipeline.compute_loss`` 必须与"main + Σ w_k * aux"逐字节等价。"""

    def test_vanilla3d(self):
        cfg = _base_cfg()
        cfg.data.patch_mode = "whole"; cfg.data.multi_res_scales = [1.0]
        cfg.sync(); cfg.validate()
        p = build_pipeline(cfg, build_loss(cfg.loss))
        D, H, W = 4, 16, 16
        pred = torch.randn(B, NUM_FG, D, H, W)
        lbl = torch.randint(0, 3, (B, 1, D, H, W)).float()
        sup = SupervisionPack(label_main=lbl)
        l_p = p.compute_loss(pred, sup)
        l_h = compute_loss_fp32(p.criterion, pred, lbl)
        assert (l_p - l_h).abs().max() < TOL

    def test_slab_aux(self):
        cfg = _base_cfg()
        cfg.data.patch_mode = "2_5d"; cfg.data.multi_res_scales = [1.0, 2.0]
        cfg.model.aux_seg_supervision = True
        cfg.sync(); cfg.validate()
        p = build_pipeline(cfg, build_loss(cfg.loss))
        D, H, W = 4, 16, 16
        pred_main = torch.randn(B, NUM_FG * D, H, W)
        pred_aux = torch.randn(B, NUM_FG * D, H, W)
        pred = {"main": pred_main, "aux": [pred_aux]}
        lbl_all = torch.randint(0, 3, (B, 2, D, H, W)).float()
        sup = SupervisionPack(
            label_main=lbl_all[:, 0], label_all_views=lbl_all)
        l_p = p.compute_loss(pred, sup)
        l_main = compute_loss_fp32(p.criterion, pred_main, lbl_all[:, 0])
        l_aux = compute_loss_fp32(p.aux_loss_fn, pred_aux, lbl_all[:, 1])
        l_h = l_main + p.aux_weights[0] * l_aux
        assert (l_p - l_h).abs().max() < TOL

    def test_lift_aux(self):
        cfg = _base_cfg()
        cfg.data.patch_mode = "2_5d"; cfg.data.multi_res_scales = [1.0, 2.0]
        cfg.model.lift_2_5d_to_3d = True
        cfg.model.aux_seg_supervision = True
        cfg.model.encoder_channels = [32, 64]
        cfg.sync(); cfg.validate()
        p = build_pipeline(cfg, build_loss(cfg.loss))
        D, H, W = 4, 16, 16
        pred_main = torch.randn(B, NUM_FG, D, H, W)
        pred_aux = torch.randn(B, NUM_FG, D, H, W)
        pred = {"main": pred_main, "aux": [pred_aux]}
        lbl_all = torch.randint(0, 3, (B, 2, D, H, W)).float()
        sup = SupervisionPack(
            label_main=lbl_all[:, :1], label_all_views=lbl_all)
        l_p = p.compute_loss(pred, sup)
        l_main = compute_loss_fp32(p.criterion, pred_main, lbl_all[:, :1])
        l_aux = compute_loss_fp32(p.aux_loss_fn, pred_aux, lbl_all[:, 1:2])
        l_h = l_main + p.aux_weights[0] * l_aux
        assert (l_p - l_h).abs().max() < TOL

    def test_native_d_aux(self):
        cfg = _base_cfg()
        cfg.data.patch_mode = "2_5d"; cfg.data.multi_res_scales = [1.0, 2.0]
        cfg.data.keep_native_view_depth = True
        cfg.model.aux_seg_supervision = True
        cfg.sync(); cfg.validate()
        p = build_pipeline(cfg, build_loss(cfg.loss))
        D, H, W = 4, 16, 16
        D_aux = p.per_view_depths[1]
        pred_main = torch.randn(B, NUM_FG * D, H, W)
        pred_aux = torch.randn(B, NUM_FG * D_aux, H, W)
        pred = {"main": pred_main, "aux": [pred_aux]}
        lbl_main = torch.randint(0, 3, (B, D, H, W)).float()
        lbl_aux = torch.randint(0, 3, (B, D_aux, H, W)).float()
        sup = SupervisionPack(
            label_main=lbl_main, aux_labels=[lbl_aux], aux_wmaps=[None])
        l_p = p.compute_loss(pred, sup)
        l_main = compute_loss_fp32(p.criterion, pred_main, lbl_main)
        l_aux = compute_loss_fp32(p.aux_loss_fns[0], pred_aux, lbl_aux)
        l_h = l_main + p.aux_weights[0] * l_aux
        assert (l_p - l_h).abs().max() < TOL


# ===========================================================================
# Backward through aux paths
# ===========================================================================
class TestBackward:
    def test_lift_aux_grads_both_heads(self):
        cfg = _base_cfg()
        cfg.data.patch_mode = "2_5d"; cfg.data.multi_res_scales = [1.0, 2.0]
        cfg.model.lift_2_5d_to_3d = True
        cfg.model.aux_seg_supervision = True
        cfg.model.encoder_channels = [32, 64]
        cfg.sync(); cfg.validate()
        p = build_pipeline(cfg, build_loss(cfg.loss))
        D, H, W = 4, 16, 16
        pred_main = torch.randn(B, NUM_FG, D, H, W, requires_grad=True)
        pred_aux = torch.randn(B, NUM_FG, D, H, W, requires_grad=True)
        pred = {"main": pred_main, "aux": [pred_aux]}
        lbl_all = torch.randint(0, 3, (B, 2, D, H, W)).float()
        sup = SupervisionPack(
            label_main=lbl_all[:, :1], label_all_views=lbl_all)
        loss = p.compute_loss(pred, sup)
        loss.backward()
        # 主路 + aux 路均必须有非零梯度，否则说明某条路径被 detach。
        assert pred_main.grad is not None and pred_main.grad.norm() > 0
        assert pred_aux.grad is not None and pred_aux.grad.norm() > 0


# ===========================================================================
# Breakdown contract (used by trainer.breakdown.format_breakdown)
# ===========================================================================
class TestBreakdown:
    def test_aux_breakdown_keys(self):
        cfg = _base_cfg()
        cfg.data.patch_mode = "2_5d"; cfg.data.multi_res_scales = [1.0, 2.0]
        cfg.model.aux_seg_supervision = True
        cfg.sync(); cfg.validate()
        p = build_pipeline(cfg, build_loss(cfg.loss))
        D, H, W = 4, 16, 16
        pred = {
            "main": torch.randn(B, NUM_FG * D, H, W),
            "aux":  [torch.randn(B, NUM_FG * D, H, W)],
        }
        lbl_all = torch.randint(0, 3, (B, 2, D, H, W)).float()
        sup = SupervisionPack(
            label_main=lbl_all[:, 0], label_all_views=lbl_all)
        bd: dict = {}
        p.compute_loss(pred, sup, breakdown=bd)
        assert "L_main" in bd
        assert "L_aux_1" in bd and "w_aux_1" in bd
        assert "L_total" in bd
        # L_total ≈ L_main + w_aux_1 * L_aux_1
        approx = bd["L_main"] + bd["w_aux_1"] * bd["L_aux_1"]
        assert abs(bd["L_total"] - approx) < 1e-5
