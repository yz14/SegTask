"""监督头损失均衡器（1-6：权重归一化 + GradNorm）回归测试。

CPU-only、小张量、快速。覆盖：

1. ``StaticBalancer``：归一化 Σw=1、比例保持、normalize=False 恒等、
   全零/负权重拒绝、未知头 KeyError、breakdown 契约。
2. ``GradNormBalancer``：初始归一化、<2 头拒绝、零初始权重拒绝、头集合
   失配拒绝、update 改变权重且保持 Σw/正性、AMP scale 不影响更新、
   update_every 节奏、state_dict/load 往返、头集合变化时安全重置。
3. ``build_balancer``：由配置装配 main/aux/topo 头全集。
4. config 校验：gradnorm 参数区间、单头开 gradnorm 拒绝。
5. pipeline 集成：默认归一化下 compute_loss 数值 = Σ 归一化权重×头损失
   （见 test_pipelines.py 的等价性测试）；此处验证单头配置数值不变。

Run:
    python -m pytest tests/test_upgrade_b2_loss_balancer.py -q
"""

from __future__ import annotations

import copy
import sys

import pytest
import torch
import torch.nn as nn

from taskcore.config.core import Config
from segtask_v1.losses.balancer import (
    GradNormBalancer,
    StaticBalancer,
    build_balancer,
)


def _base_cfg(**data_over):
    cfg = Config()
    cfg.data.patch_size = [4, 16, 16]
    cfg.data.label_values = [0, 1, 2]
    cfg.model.encoder_channels = [8, 16]
    for k, v in data_over.items():
        setattr(cfg.data, k, v)
    return cfg


# ===========================================================================
# StaticBalancer
# ===========================================================================
class TestStaticBalancer:
    def test_normalize_sums_to_one_keeps_ratio(self):
        b = StaticBalancer({"main": 1.0, "aux_1": 0.5, "topo": 0.3},
                           normalize=True)
        w = [b.weight("main"), b.weight("aux_1"), b.weight("topo")]
        assert sum(w) == pytest.approx(1.0)
        assert b.weight("aux_1") / b.weight("main") == pytest.approx(0.5)
        assert b.weight("topo") / b.weight("main") == pytest.approx(0.3)

    def test_no_normalize_identity(self):
        b = StaticBalancer({"main": 1.0, "aux_1": 0.5}, normalize=False)
        assert b.weight("main") == 1.0 and b.weight("aux_1") == 0.5

    def test_single_head_unchanged(self):
        # 单头 main=1.0 归一化后仍为 1.0：无 aux/topo 配置行为不变。
        b = StaticBalancer({"main": 1.0}, normalize=True)
        assert b.weight("main") == pytest.approx(1.0)

    def test_all_zero_rejected(self):
        with pytest.raises(ValueError, match="all-zero"):
            StaticBalancer({"main": 0.0, "aux_1": 0.0}, normalize=True)

    def test_negative_rejected(self):
        with pytest.raises(ValueError, match=">= 0"):
            StaticBalancer({"main": 1.0, "aux_1": -0.1}, normalize=True)

    def test_unknown_head_keyerror(self):
        b = StaticBalancer({"main": 1.0}, normalize=True)
        with pytest.raises(KeyError, match="unknown supervision head"):
            b.weight("aux_9")

    def test_combine_breakdown_contract(self):
        b = StaticBalancer({"main": 1.0, "aux_1": 0.5}, normalize=True)
        l_main = torch.tensor(0.9)
        l_aux = torch.tensor(0.3)
        bd: dict = {}
        total = b.combine([("main", l_main), ("aux_1", l_aux)], bd)
        expect = b.weight("main") * 0.9 + b.weight("aux_1") * 0.3
        assert float(total) == pytest.approx(expect)
        assert bd["L_main"] == pytest.approx(0.9)
        assert bd["L_aux_1"] == pytest.approx(0.3)
        assert bd["w_main"] == pytest.approx(b.weight("main"))
        assert bd["w_aux_1"] == pytest.approx(b.weight("aux_1"))
        assert bd["L_total"] == pytest.approx(expect)

    def test_combine_empty_rejected(self):
        b = StaticBalancer({"main": 1.0}, normalize=True)
        with pytest.raises(ValueError, match="at least one head"):
            b.combine([])

    def test_wants_update_false(self):
        b = StaticBalancer({"main": 1.0}, normalize=True)
        assert not b.wants_update()
        assert b.state_dict() == {}


# ===========================================================================
# GradNormBalancer
# ===========================================================================
def _gn(update_every=1, alpha=1.5, lr=0.05, weights=None):
    return GradNormBalancer(
        weights or {"main": 1.0, "aux_1": 0.5},
        alpha=alpha, lr=lr, update_every=update_every, normalize=True)


def _two_head_losses(model: nn.Linear):
    x = torch.randn(4, 8)
    out = model(x)
    l_main = (out[:, 0] - 1.0).pow(2).mean()
    l_aux = (out[:, 1] + 1.0).pow(2).mean() * 3.0
    return l_main, l_aux


class TestGradNormBalancer:
    def test_init_normalized(self):
        b = _gn()
        assert float(b.w.detach().sum()) == pytest.approx(1.0)
        assert b.weight("aux_1") / b.weight("main") == pytest.approx(0.5)

    def test_needs_two_heads(self):
        with pytest.raises(ValueError, match=">= 2 supervision heads"):
            GradNormBalancer({"main": 1.0})

    def test_zero_init_weight_rejected(self):
        with pytest.raises(ValueError, match="must be > 0"):
            GradNormBalancer({"main": 1.0, "aux_1": 0.0})

    def test_head_set_mismatch_rejected(self):
        b = _gn()
        with pytest.raises(RuntimeError, match="head set mismatch"):
            b.combine([("main", torch.tensor(1.0))])

    def test_update_without_combine_raises(self):
        b = _gn()
        with pytest.raises(RuntimeError, match="without a preceding"):
            b.update([], torch.device("cpu"))

    def test_update_moves_weights_preserves_sum_and_positivity(self):
        torch.manual_seed(0)
        model = nn.Linear(8, 2)
        b = _gn(lr=0.1)
        w_before = b.w.detach().clone()
        for _ in range(5):
            l_main, l_aux = _two_head_losses(model)
            assert b.tick_boundary()
            b.arm_stash()
            b.combine([("main", l_main), ("aux_1", l_aux)])
            b.update(list(model.parameters()), torch.device("cpu"))
        w_after = b.w.detach()
        assert not torch.allclose(w_before, w_after)
        assert float(w_after.sum()) == pytest.approx(1.0)
        assert (w_after > 0).all()

    def test_main_backward_unaffected_after_update(self):
        # update 用 retain_graph 的 autograd.grad + 内部 w-backward，
        # 不得破坏随后主损失的 backward。
        torch.manual_seed(0)
        model = nn.Linear(8, 2)
        b = _gn()
        l_main, l_aux = _two_head_losses(model)
        b.arm_stash()
        total = b.combine([("main", l_main), ("aux_1", l_aux)])
        b.update(list(model.parameters()), torch.device("cpu"))
        total.backward()  # 不应抛"graph freed"
        assert model.weight.grad is not None
        assert torch.isfinite(model.weight.grad).all()
        # 模型参数不得从 GradNorm 内部 backward 收到梯度污染前的额外累加：
        # w 是唯一 requires_grad 的 GradNorm 参数，l_grad 对模型参数无梯度
        #（g_vec 为 detach 常数），故 model.grad 只来自 total.backward()。

    def test_amp_scale_invariant(self):
        torch.manual_seed(1)
        model = nn.Linear(8, 2)
        b1 = _gn(lr=0.1)
        b2 = copy.deepcopy(b1)
        model2 = copy.deepcopy(model)
        torch.manual_seed(2)
        l_main, l_aux = _two_head_losses(model)
        torch.manual_seed(2)
        l_main2, l_aux2 = _two_head_losses(model2)
        b1.arm_stash(); b1.combine([("main", l_main), ("aux_1", l_aux)])
        b1.update(list(model.parameters()), torch.device("cpu"), amp_scale=1.0)
        b2.arm_stash(); b2.combine([("main", l_main2), ("aux_1", l_aux2)])
        b2.update(list(model2.parameters()), torch.device("cpu"),
                  amp_scale=65536.0)
        assert torch.allclose(b1.w.detach(), b2.w.detach(), atol=1e-6)

    def test_nonfinite_loss_skips_update(self):
        b = _gn()
        w_before = b.w.detach().clone()
        b.arm_stash()
        heads = [("main", torch.tensor(float("nan"), requires_grad=True)),
                 ("aux_1", torch.tensor(1.0, requires_grad=True))]
        b._stash = heads  # combine 会算 nan total；直接注入 stash 验证守卫
        b.update([], torch.device("cpu"))
        assert torch.allclose(b.w.detach(), w_before)

    def test_tick_boundary_cadence(self):
        b = _gn(update_every=3)
        due = [b.tick_boundary() for _ in range(7)]
        assert due == [True, False, False, True, False, False, True]

    def test_state_dict_roundtrip(self):
        torch.manual_seed(0)
        model = nn.Linear(8, 2)
        b = _gn(lr=0.1)
        l_main, l_aux = _two_head_losses(model)
        b.tick_boundary()
        b.arm_stash()
        b.combine([("main", l_main), ("aux_1", l_aux)])
        b.update(list(model.parameters()), torch.device("cpu"))
        state = b.state_dict()
        b2 = _gn(lr=0.1)
        b2.load_state_dict(state)
        assert torch.allclose(b2.w.detach(), b.w.detach())
        assert torch.allclose(b2._l0, b._l0)
        assert b2._boundary_clock == b._boundary_clock

    def test_load_head_set_change_starts_fresh(self):
        b = _gn()
        w_before = b.w.detach().clone()
        b.load_state_dict({
            "names": ["main", "aux_1", "topo"],
            "w": torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64),
            "l0": None, "boundary_clock": 7, "opt": {}})
        assert torch.allclose(b.w.detach(), w_before)
        assert b._boundary_clock == 0

    def test_weights_flow_into_combine(self):
        b = _gn()
        with torch.no_grad():
            b.w.copy_(torch.tensor([0.8, 0.2], dtype=torch.float64))
        total = b.combine([("main", torch.tensor(1.0)),
                           ("aux_1", torch.tensor(1.0))])
        assert float(total) == pytest.approx(1.0)
        total = b.combine([("main", torch.tensor(2.0)),
                           ("aux_1", torch.tensor(0.0))])
        assert float(total) == pytest.approx(1.6)


# ===========================================================================
# build_balancer / config
# ===========================================================================
class TestBuildBalancer:
    def test_static_full_head_set(self):
        cfg = _base_cfg()
        b = build_balancer(cfg, [0.5], 0.3)
        assert isinstance(b, StaticBalancer)
        s = b.weight("main") + b.weight("aux_1") + b.weight("topo")
        assert s == pytest.approx(1.0)

    def test_no_normalize_option(self):
        cfg = _base_cfg()
        cfg.loss.normalize_supervision_weights = False
        b = build_balancer(cfg, [0.5], None)
        assert b.weight("main") == 1.0 and b.weight("aux_1") == 0.5

    def test_gradnorm_selected(self):
        cfg = _base_cfg()
        cfg.loss.gradnorm_enabled = True
        b = build_balancer(cfg, [0.5], None)
        assert isinstance(b, GradNormBalancer)
        assert b.names == ["main", "aux_1"]

    def test_validate_gradnorm_single_head_rejected(self):
        cfg = _base_cfg()
        cfg.loss.gradnorm_enabled = True
        cfg.sync()
        with pytest.raises(ValueError, match="gradnorm_enabled"):
            cfg.validate()

    def test_validate_gradnorm_ok_with_aux(self):
        cfg = _base_cfg()
        cfg.data.patch_mode = "2_5d"
        cfg.data.multi_res_scales = [1.0, 2.0]
        cfg.model.aux_seg_supervision = True
        cfg.loss.gradnorm_enabled = True
        cfg.sync()
        cfg.validate()

    def test_validate_param_ranges(self):
        for field, bad in (("gradnorm_alpha", -0.1),
                           ("gradnorm_lr", 0.0),
                           ("gradnorm_update_every", 0)):
            cfg = _base_cfg()
            setattr(cfg.loss, field, bad)
            cfg.sync()
            with pytest.raises(ValueError, match=field):
                cfg.validate()


# ===========================================================================
# Pipeline 集成：单头配置数值不变
# ===========================================================================
class TestPipelineIntegration:
    def test_single_head_loss_unchanged(self):
        from segtask_v1.trainer.pipelines.factory import build_pipeline
        from segtask_v1.trainer.pipelines.base import SupervisionPack
        from segtask_v1.losses.losses import build_loss
        from taskcore.engine.amp import compute_loss_fp32
        cfg = _base_cfg()
        cfg.sync(); cfg.validate()
        p = build_pipeline(cfg, build_loss(cfg.loss))
        pred = torch.randn(2, 2, 4, 16, 16)
        lbl = torch.randint(0, 3, (2, 1, 4, 16, 16)).float()
        sup = SupervisionPack(label_main=lbl)
        l_p = p.compute_loss(pred, sup)
        l_ref = compute_loss_fp32(p.criterion, pred, lbl)
        assert (l_p - l_ref).abs().max() < 1e-6


    def test_trainer_gradnorm_end_to_end(self, tmp_path):
        """1 epoch 合成数据训练：GradNorm 更新头权重 + 状态随 ckpt 落盘/恢复。"""
        from taskcore.data.loader import build_dataloaders
        from taskcore.models.factory import build_model
        from segtask_v1.trainer import Trainer
        from test_2_5d_smoke import _make_synthetic_dataset

        img_dir, lbl_dir = _make_synthetic_dataset(
            tmp_path, n_volumes=4, shape=(20, 64, 64), num_fg=2)
        cfg = Config()
        cfg.data.image_dir = img_dir
        cfg.data.label_dir = lbl_dir
        cfg.data.npz_dir = str(tmp_path / "npz")
        cfg.data.npz_auto_build = True
        cfg.data.patch_mode = "2_5d"
        cfg.data.patch_size = [12, 32, 32]
        cfg.data.label_values = [0, 1, 2]
        cfg.data.num_classes = 3
        cfg.data.multi_res_scales = [1.0, 2.0]
        cfg.data.batch_size = 1
        cfg.data.num_workers = 0
        cfg.data.samples_per_volume = 1
        cfg.data.foreground_oversample_ratio = 1.0
        cfg.data.intensity_min = -200.0
        cfg.data.intensity_max = 200.0
        cfg.data.cache_mode = "memory"
        cfg.model.encoder_channels = [16, 32, 64]
        cfg.model.deep_supervision = False
        cfg.model.aux_seg_supervision = True
        cfg.augment.enabled = False
        cfg.loss.gradnorm_enabled = True
        cfg.loss.gradnorm_update_every = 1
        cfg.loss.gradnorm_lr = 0.1
        cfg.train.epochs = 1
        cfg.train.use_amp = False
        cfg.train.use_ema = False
        cfg.train.warmup_epochs = 0
        cfg.train.compile_mode = "none"
        cfg.train.output_dir = str(tmp_path / "out")
        cfg.train.log_every = 1
        cfg.train.save_every = 1
        cfg.train.val_every = 1
        cfg.sync()
        cfg.validate()

        train_loader, val_loader = build_dataloaders(cfg)
        model = build_model(cfg)
        device = torch.device("cpu")
        trainer = Trainer(model, cfg, train_loader, val_loader, device)
        bal = trainer.pipeline.balancer
        assert isinstance(bal, GradNormBalancer)
        w0 = bal.w.detach().clone()
        trainer.fit()
        assert not torch.allclose(bal.w.detach(), w0)
        assert float(bal.w.detach().sum()) == pytest.approx(1.0)

        # 状态落盘 + 恢复
        extra = trainer._ckpt_extra_state()
        assert "loss_balancer" in extra
        model2 = build_model(cfg)
        trainer2 = Trainer(model2, cfg, train_loader, val_loader, device)
        trainer2.pipeline.balancer.load_state_dict(extra["loss_balancer"])
        assert torch.allclose(
            trainer2.pipeline.balancer.w.detach(), bal.w.detach())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
