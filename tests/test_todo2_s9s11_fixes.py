"""TODO2 S9–S11 修复的回归测试。

覆盖：
* S9-A  class_weights 长度 fail-fast（config 层 + MultiResolutionLoss 构造期）
* S9-B  2.5D per-slice 物理 NSD 收到体积级 (z,y,x) spacing 时取面内两轴
* S9-E  空类（无 GT 无 pred）overlap 指标返回 0 而非平滑出的 1.0
* S9-G  val_volume_cache_max_gb 配置字段
* S10-C z_boundary_mode 白名单只收 edge_pad；sync() 自动升级 stretch
* S10-D/E 推理侧 resize_trilinear 与训练侧 resize_3d（scipy zoom）语义对拍
* S10-F/S11-B checkpoint 预处理镜像硬拦截 + allow_preprocess_mismatch 降级
* S10-H 诊断分位数抽样版
* 轻微项：递归输入同名输出自动加子目录前缀
"""

import numpy as np
import pytest
import torch

from taskcore.config.core import Config
from taskcore.metrics import derive_overlap_metrics, _nsd_stats_spacing_aware


# ---------------------------------------------------------------------------
# S9-E: 空类 overlap 指标返回 0
# ---------------------------------------------------------------------------
def test_empty_class_overlap_metrics_are_zero():
    # 类 0：完美预测；类 1：GT 与 pred 皆空。
    inter = torch.tensor([100.0, 0.0])
    pred_sum = torch.tensor([100.0, 0.0])
    target_sum = torch.tensor([100.0, 0.0])
    voxels = torch.tensor(1000.0, dtype=torch.float64)
    m = derive_overlap_metrics(inter, pred_sum, target_sum, voxels)
    for key in ("dice", "iou", "recall", "precision", "vol_sim"):
        assert m[key][0].item() == pytest.approx(1.0, abs=1e-3), key
        assert m[key][1].item() == 0.0, f"{key} for empty class must be 0"


def test_nonempty_class_metrics_unaffected():
    inter = torch.tensor([50.0])
    pred_sum = torch.tensor([80.0])
    target_sum = torch.tensor([70.0])
    voxels = torch.tensor(1000.0, dtype=torch.float64)
    m = derive_overlap_metrics(inter, pred_sum, target_sum, voxels)
    assert m["dice"][0].item() == pytest.approx(2 * 50 / 150, abs=1e-3)


# ---------------------------------------------------------------------------
# S9-B: per-slice 2D 输入 + 体积级 (z,y,x) spacing
# ---------------------------------------------------------------------------
def test_nsd_spacing_rank3_on_2d_slices_uses_inplane_axes():
    torch.manual_seed(0)
    pred = (torch.rand(2, 1, 16, 16) > 0.5).float()
    tgt = (torch.rand(2, 1, 16, 16) > 0.5).float()
    # (z, y, x) spacing 供 2D 切片：应等价于直接传 (y, x)。
    s3 = _nsd_stats_spacing_aware(pred, tgt, 2.0, [5.0, 0.7, 0.7])
    s2 = _nsd_stats_spacing_aware(pred, tgt, 2.0, [0.7, 0.7])
    assert torch.equal(s3["sd_num"], s2["sd_num"])
    assert torch.equal(s3["sd_denom"], s2["sd_denom"])


def test_nsd_spacing_rank_mismatch_raises():
    pred = (torch.rand(1, 1, 8, 8, 8) > 0.5).float()
    tgt = (torch.rand(1, 1, 8, 8, 8) > 0.5).float()
    with pytest.raises(ValueError, match="spacing length"):
        _nsd_stats_spacing_aware(pred, tgt, 2.0, [0.7, 0.7])


# ---------------------------------------------------------------------------
# S9-A: class_weights 长度校验
# ---------------------------------------------------------------------------
def test_config_rejects_class_weights_length_mismatch():
    cfg = Config()
    cfg.data.label_values = [0, 1]          # 1 个前景类
    cfg.loss.class_weights = [1.0, 1.0]     # 2 个权重 → 拒
    cfg.sync()
    with pytest.raises(Exception, match="class_weights"):
        cfg.validate()


def test_config_accepts_matching_class_weights():
    cfg = Config()
    cfg.data.label_values = [0, 1]
    cfg.loss.class_weights = [1.0]
    cfg.sync()
    cfg.validate()


def test_multi_resolution_loss_rejects_weight_length_mismatch():
    from segtask_v1.losses.losses import BinaryDiceLoss, MultiResolutionLoss

    base = BinaryDiceLoss(class_weights=[1.0, 2.0])
    with pytest.raises(ValueError, match="class_weights"):
        MultiResolutionLoss(
            base_loss=base, num_fg_classes=1, num_res=1,
            label_values=[0, 1])


# ---------------------------------------------------------------------------
# S9-G: 验证卷缓存字节上限配置
# ---------------------------------------------------------------------------
def test_val_volume_cache_max_gb_field():
    cfg = Config()
    assert cfg.train.val_volume_cache_max_gb == pytest.approx(8.0)
    cfg.train.val_volume_cache_max_gb = 0.5
    cfg.sync()
    cfg.validate()


# ---------------------------------------------------------------------------
# S10-C: stretch 白名单删除
# ---------------------------------------------------------------------------
def test_validate_rejects_stretch():
    cfg = Config()
    cfg.data.z_boundary_mode = "stretch"
    with pytest.raises(Exception, match="z_boundary_mode"):
        cfg.validate()


def test_sync_upgrades_stretch_to_edge_pad():
    cfg = Config()
    cfg.data.z_boundary_mode = "stretch"
    cfg.sync()
    assert cfg.data.z_boundary_mode == "edge_pad"
    cfg.validate()


# ---------------------------------------------------------------------------
# S10-D/E: 推理 GPU resize 与训练 scipy zoom 语义对拍
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("antialias", [False, True])
@pytest.mark.parametrize("shape,target", [
    ((9, 17, 13), (5, 9, 7)),      # 三轴下采样
    ((5, 9, 7), (9, 17, 13)),      # 三轴上采样
    ((12, 20, 16), (6, 10, 8)),    # 整数因子下采样
    ((7, 16, 16), (12, 8, 24)),    # 混合上/下采样
])
def test_resize_trilinear_matches_training_resize_3d(shape, target, antialias):
    from taskcore.data.dataset import resize_3d
    from segtask_v1.predictor.inputs import resize_trilinear

    rng = np.random.default_rng(42)
    vol = rng.normal(size=shape).astype(np.float32)
    ref = resize_3d(vol, *target, is_label=False, anti_alias=antialias)
    out = resize_trilinear(
        torch.from_numpy(vol)[None, None], target,
        antialias=antialias)[0, 0].numpy()
    assert np.abs(ref - out).max() < 1e-4


def test_resize_trilinear_identity_passthrough():
    from segtask_v1.predictor.inputs import resize_trilinear

    x = torch.rand(1, 1, 4, 5, 6)
    assert resize_trilinear(x, (4, 5, 6), antialias=True) is x


# ---------------------------------------------------------------------------
# S10-F/S11-B: 预处理镜像硬拦截
# ---------------------------------------------------------------------------
def _mk_cfgs():
    train_cfg = Config()
    train_cfg.sync()
    infer_cfg = Config()
    infer_cfg.sync()
    return train_cfg, infer_cfg


def test_preprocess_mirror_pass_when_identical():
    from segtask_v1.predictor.io import _check_preprocess_mirror

    train_cfg, infer_cfg = _mk_cfgs()
    _check_preprocess_mirror({"config": train_cfg}, infer_cfg, "ckpt.pth")


def test_preprocess_mirror_hard_mismatch_raises():
    from segtask_v1.predictor.io import _check_preprocess_mirror

    train_cfg, infer_cfg = _mk_cfgs()
    infer_cfg.data.normalize = (
        "zscore" if train_cfg.data.normalize != "zscore" else "minmax")
    with pytest.raises(RuntimeError, match="mirror mismatch"):
        _check_preprocess_mirror({"config": train_cfg}, infer_cfg, "ckpt.pth")


def test_preprocess_mirror_allow_flag_downgrades_to_warning():
    from segtask_v1.predictor.io import _check_preprocess_mirror

    train_cfg, infer_cfg = _mk_cfgs()
    infer_cfg.data.normalize = (
        "zscore" if train_cfg.data.normalize != "zscore" else "minmax")
    infer_cfg.predict.allow_preprocess_mismatch = True
    _check_preprocess_mirror({"config": train_cfg}, infer_cfg, "ckpt.pth")


def test_preprocess_mirror_skips_without_config():
    from segtask_v1.predictor.io import _check_preprocess_mirror

    _, infer_cfg = _mk_cfgs()
    _check_preprocess_mirror({}, infer_cfg, "ckpt.pth")


def test_preprocess_mirror_soft_key_patch_size_only_warns():
    from segtask_v1.predictor.io import _check_preprocess_mirror

    train_cfg, infer_cfg = _mk_cfgs()
    infer_cfg.data.patch_size = [
        int(s) * 2 for s in train_cfg.data.patch_size]
    _check_preprocess_mirror({"config": train_cfg}, infer_cfg, "ckpt.pth")


# ---------------------------------------------------------------------------
# S10-H: 诊断分位数抽样版
# ---------------------------------------------------------------------------
def test_sampled_quantile_small_array_exact():
    from segtask_v1.predictor.predictor import _sampled_quantile

    arr = np.arange(101, dtype=np.float32)
    q = _sampled_quantile(arr, [0.0, 0.5, 1.0])
    assert list(q) == [0.0, 50.0, 100.0]


def test_sampled_quantile_large_array_subsamples():
    from segtask_v1.predictor.predictor import _sampled_quantile

    rng = np.random.default_rng(0)
    arr = rng.normal(size=3_000_000).astype(np.float32)
    q = _sampled_quantile(arr, [0.5], cap=100_000)
    full = np.quantile(arr, [0.5])
    assert abs(float(q[0]) - float(full[0])) < 0.02


# ---------------------------------------------------------------------------
# S10-A: AdaBN per_volume 端到端冒烟（BN 统计确实被目标卷重估）
# ---------------------------------------------------------------------------
def test_adabn_per_volume_smoke(tmp_path):
    import SimpleITK as sitk
    from taskcore.config.seg_bundle import make_test_config
    from segtask_v1.predictor.predictor import Predictor

    class TinyBN(torch.nn.Module):
        def __init__(self, num_fg=1):
            super().__init__()
            self.conv = torch.nn.Conv3d(1, 4, 3, padding=1)
            self.bn = torch.nn.BatchNorm3d(4)
            self.head = torch.nn.Conv3d(4, num_fg, 1)

        def forward(self, x):
            return self.head(self.bn(self.conv(x)))

    cfg = make_test_config()
    cfg.data.patch_mode = "z_axis"
    cfg.data.patch_size = [16, 32, 32]
    cfg.data.multi_res_scales = [1.0]
    cfg.data.label_values = [0, 1]
    cfg.model.norm_type = "batch"
    cfg.predict.adabn_enabled = True
    cfg.predict.adabn_mode = "per_volume"
    cfg.sync()
    cfg.validate()

    model = TinyBN().eval()
    p = Predictor(model, cfg, torch.device("cpu"))
    rm0 = model.bn.running_mean.clone()

    vol = (np.random.default_rng(0).normal(size=(40, 32, 32))
           * 100 + 50).astype(np.float32)
    path = str(tmp_path / "v.nii.gz")
    sitk.WriteImage(sitk.GetImageFromArray(vol), path)
    out = p.predict_volume(path)
    assert out["label_map"].shape == (40, 32, 32)
    # per_volume 估计前向应改变 BN running stats（transductive BN 生效）。
    assert not torch.allclose(rm0, model.bn.running_mean)


# ---------------------------------------------------------------------------
# 轻微项：递归输入同名输出自动加子目录前缀
# ---------------------------------------------------------------------------
def test_unique_output_stems_dedups_by_subdir():
    from segtask_v1.predictor.io import _unique_output_stems

    paths = ["/data/a/ct.nii.gz", "/data/b/ct.nii.gz", "/data/other.nii"]
    stems = _unique_output_stems(paths)
    assert stems["/data/other.nii"] == "other"
    assert stems["/data/a/ct.nii.gz"] == "a__ct"
    assert stems["/data/b/ct.nii.gz"] == "b__ct"
    assert len(set(stems.values())) == 3


def test_unique_output_stems_no_conflict_untouched():
    from segtask_v1.predictor.io import _unique_output_stems

    paths = ["/x/one.nii.gz", "/x/two.nii.gz"]
    assert _unique_output_stems(paths) == {
        "/x/one.nii.gz": "one", "/x/two.nii.gz": "two"}
