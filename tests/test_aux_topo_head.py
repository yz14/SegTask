"""中心线 / 距离场辅助头（aux_topo_head）测试。

CPU-only、小张量、快速。覆盖：

1. ``AuxTopoLoss`` 目标生成（soft-skeleton / 距离场）在 2D / 3D 上形状、值域、
   可微（反传产生梯度）。
2. UNet3D forward：训练态返回含 ``"topo"`` 的 dict 且与主头同形；eval 态不输出 topo。
3. pipeline ``compute_loss``：``breakdown`` 含 ``L_topo``/``w_topo``，反传后辅助头收到梯度。
4. 向后兼容：``aux_topo_head=False`` 时 forward 不含 topo、损失无 topo 项。
5. 与多 FOV ``aux_seg_supervision`` 共存：dict 同时含 ``aux`` 与 ``topo``。

Run:
    python -m pytest tests/test_aux_topo_head.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segtask_v1.config import Config
from segtask_v1.losses.losses import build_loss
from segtask_v1.losses.topo_aux import (
    AuxTopoLoss, morph_distance_target, soft_skeleton_target,
)
from segtask_v1.models.factory import build_model
from segtask_v1.trainer.pipelines import build_pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cfg_2_5d(*, aux_topo=True, target="centerline", aux_seg=False,
              multi_res=(1.0,)):
    cfg = Config()
    cfg.data.label_values = [0, 1, 2]
    cfg.data.num_classes = 3
    cfg.data.patch_size = [4, 16, 16]
    cfg.data.batch_size = 1
    cfg.data.patch_mode = "2_5d"
    cfg.data.multi_res_scales = list(multi_res)
    cfg.model.encoder_channels = [16, 32]
    cfg.model.blocks_per_level = 1
    cfg.model.deep_supervision = False
    cfg.loss.deep_supervision_weights = []
    cfg.model.aux_seg_supervision = aux_seg
    cfg.model.aux_topo_head = aux_topo
    cfg.model.aux_topo_target = target
    cfg.loss.aux_topo_weight = 0.3
    cfg.loss.aux_topo_iter = 2
    cfg.sync()
    cfg.validate()
    return cfg


def _cfg_3d(*, aux_topo=True, target="distance"):
    cfg = Config()
    cfg.data.label_values = [0, 1, 2]
    cfg.data.num_classes = 3
    cfg.data.patch_size = [8, 16, 16]
    cfg.data.batch_size = 1
    cfg.data.patch_mode = "cubic"
    cfg.data.multi_res_scales = [1.0]
    cfg.model.encoder_channels = [16, 32]
    cfg.model.blocks_per_level = 1
    cfg.model.deep_supervision = False
    cfg.loss.deep_supervision_weights = []
    cfg.model.aux_topo_head = aux_topo
    cfg.model.aux_topo_target = target
    cfg.loss.aux_topo_weight = 0.3
    cfg.loss.aux_topo_iter = 2
    cfg.sync()
    cfg.validate()
    return cfg


# ===========================================================================
# 1. AuxTopoLoss target generation
# ===========================================================================
class TestTopoTargets:
    @pytest.mark.parametrize("ndim", [2, 3])
    def test_skeleton_target_shape_range(self, ndim):
        shape = (2, 3) + (12,) * ndim
        label = (torch.rand(*shape) > 0.5).float()
        skel = soft_skeleton_target(label, n_iter=2)
        assert skel.shape == label.shape
        assert skel.min() >= 0.0 and skel.max() <= 1.0 + 1e-5

    @pytest.mark.parametrize("ndim", [2, 3])
    def test_distance_target_shape_range(self, ndim):
        shape = (2, 3) + (12,) * ndim
        label = torch.zeros(*shape)
        # 中心实心块 → 内部距离 > 0。
        sl = (slice(None), slice(None)) + (slice(3, 9),) * ndim
        label[sl] = 1.0
        dist = morph_distance_target(label, max_iter=2)
        assert dist.shape == label.shape
        assert dist.min() >= 0.0 and dist.max() <= 1.0 + 1e-5
        assert dist.max() > 0.0  # 厚块内部应有正距离

    @pytest.mark.parametrize("target,loss", [
        ("centerline", "auto"), ("centerline", "bce"),
        ("distance", "auto"), ("distance", "mse"),
    ])
    def test_loss_finite_and_backward(self, target, loss):
        fn = AuxTopoLoss(target=target, loss=loss, iter_=2)
        pred = torch.randn(2, 3, 12, 12, requires_grad=True)
        tgt = (torch.rand(2, 3, 12, 12) > 0.5).float()
        out = fn(pred, tgt)
        assert torch.isfinite(out)
        out.backward()
        assert pred.grad is not None and torch.isfinite(pred.grad).all()


# ===========================================================================
# 2 + 3. Model forward + pipeline compute_loss
# ===========================================================================
def _make_pipeline_inputs(cfg):
    """构造与 dataset 输出口径一致的 (image, label)，按 native 模式给最大 FOV cube。"""
    D, H, W = cfg.data.patch_size
    n_views = len(cfg.data.multi_res_scales)
    max_scale = max(cfg.data.multi_res_scales)
    if cfg.data.keep_native_multi_res:                 # cubic native：(1,1,eD,eH,eW)
        shape = (1, 1, round(D * max_scale),
                 round(H * max_scale), round(W * max_scale))
    elif cfg.data.keep_native_view_depth:              # 2.5D native-d：(1,1,eD,H,W)
        shape = (1, 1, round(D * max_scale), H, W)
    else:                                              # 折叠多视图：(1,n_views,D,H,W)
        shape = (1, n_views, D, H, W)
    img = torch.randn(*shape)
    lbl = torch.randint(0, 3, shape).float()
    lbl[..., 0, 0, 0] = 1                               # 保证至少一个前景体素
    return img, lbl


def _run_pipeline_case(cfg, *, expect_aux):
    model = build_model(cfg)
    pipeline = build_pipeline(cfg, build_loss(cfg.loss))

    img, lbl = _make_pipeline_inputs(cfg)
    image, sup = pipeline.prepare_batch(img, lbl, None)

    # ---- train forward: dict 含 topo ----
    model.train()
    out = model(image)
    assert isinstance(out, dict) and "topo" in out
    assert ("aux" in out) == expect_aux
    main = out["main"]
    main_t = main[0] if isinstance(main, list) else main
    assert out["topo"].shape == main_t.shape

    # ---- compute_loss + breakdown ----
    breakdown = {}
    loss = pipeline.compute_loss(out, sup, breakdown=breakdown)
    assert torch.isfinite(loss)
    assert "L_topo" in breakdown and "w_topo" in breakdown
    assert breakdown["w_topo"] == pytest.approx(cfg.loss.aux_topo_weight)

    loss.backward()
    grads = [p.grad for p in model.topo_head.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum().item() > 0 for g in grads), (
        "topo head 未收到非零梯度")

    # ---- eval: 不输出 topo ----
    model.eval()
    with torch.no_grad():
        out_eval = model(image)
    if isinstance(out_eval, dict):
        assert "topo" not in out_eval


class TestPipelineTopo:
    def test_2_5d_centerline(self):
        _run_pipeline_case(_cfg_2_5d(target="centerline"), expect_aux=False)

    def test_2_5d_distance(self):
        _run_pipeline_case(_cfg_2_5d(target="distance"), expect_aux=False)

    def test_3d_distance(self):
        _run_pipeline_case(_cfg_3d(target="distance"), expect_aux=False)

    def test_3d_centerline(self):
        _run_pipeline_case(_cfg_3d(target="centerline"), expect_aux=False)

    def test_2_5d_with_multi_fov_aux(self):
        # topo 与多 FOV aux 共存：dict 同时含 aux 与 topo。
        cfg = _cfg_2_5d(target="centerline", aux_seg=True, multi_res=(1.0, 2.0))
        _run_pipeline_case(cfg, expect_aux=True)

    def test_2_5d_native_depth_aux(self):
        # segves2_5d.yaml 同款：Slab2_5DNativeDPipeline（SliceChannelLoss 主路）。
        cfg = _cfg_2_5d(target="centerline", aux_seg=True, multi_res=(1.0, 2.0))
        cfg.data.keep_native_view_depth = True
        cfg.sync(); cfg.validate()
        from segtask_v1.trainer.pipelines import Slab2_5DNativeDPipeline
        pipe = build_pipeline(cfg, build_loss(cfg.loss))
        assert isinstance(pipe, Slab2_5DNativeDPipeline)
        _run_pipeline_case(cfg, expect_aux=True)

    def test_3d_native_multi_res(self):
        # segves3d.yaml 同款：Patch3DNativeMultiResPipeline（MultiResolutionLoss 主路）。
        cfg = _cfg_3d(target="distance")
        cfg.data.multi_res_scales = [1.0, 2.0]
        cfg.data.keep_native_multi_res = True
        cfg.sync(); cfg.validate()
        from segtask_v1.trainer.pipelines import Patch3DNativeMultiResPipeline
        pipe = build_pipeline(cfg, build_loss(cfg.loss))
        assert isinstance(pipe, Patch3DNativeMultiResPipeline)
        _run_pipeline_case(cfg, expect_aux=False)


# ===========================================================================
# 4. Backward compatibility
# ===========================================================================
class TestBackwardCompat:
    def test_no_topo_head_no_dict(self):
        cfg = _cfg_3d(aux_topo=False)
        assert not cfg.model.aux_topo_head
        model = build_model(cfg)
        pipeline = build_pipeline(cfg, build_loss(cfg.loss))
        assert pipeline.aux_topo_loss_fn is None
        D, H, W = cfg.data.patch_size
        img = torch.randn(1, 1, D, H, W)
        lbl = torch.randint(0, 3, (1, 1, D, H, W)).float()
        image, sup = pipeline.prepare_batch(img, lbl, None)
        model.train()
        out = model(image)
        assert not isinstance(out, dict)  # 无 aux/topo → tensor
        breakdown = {}
        loss = pipeline.compute_loss(out, sup, breakdown=breakdown)
        assert torch.isfinite(loss) and "L_topo" not in breakdown


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
