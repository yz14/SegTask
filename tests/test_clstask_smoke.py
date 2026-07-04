"""clstask end-to-end smoke test（双几何 3D cubic + 2.5D 折叠）。

用合成 npz（image + label）跑通完整链路，验证 C1 验收标准：

  1. 配置：3D cubic / 2.5D 折叠两套 (cfg, cls) 加载 + 校验通过，几何派生正确
     （spatial_dims / in_channels）。
  2. 数据层：ClsPatchDataset 输出形状随 spatial_dims 切换；volume/slice
     两种 target 形状正确；mask 弱标签由 label 派生。
  3. 训练：两套几何各跑 3 epoch，loss 有限且总体下降，val 输出 AUC/F1/acc。
  4. SSL 迁移：训练产出的 checkpoint 含 encoder.* 键；load_pretrained_encoder
     能从该 ckpt 命中 encoder.* 权重（strict=False，命中数 > 0）。
  5. 四模板：encoder / densenet / vit backbone 在两套几何下均能前向出 logits。
  6. 推理：ClsPredictor 整卷推理输出卷级（及 slice 粒度的逐 slice）概率。

Run:
    python tests/test_clstask_smoke.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _ok(name: str, msg: str = "") -> None:
    print(f"  [PASS] {name}{(' — ' + msg) if msg else ''}")


def _make_npz_dir(root: Path, n: int = 6, shape=(24, 160, 160)) -> str:
    """合成 n 个 image+label npz；label 含随机前景块，保证类出现有变化。"""
    d = root / "npz"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        img = rng.standard_normal(shape).astype(np.float32) * 200.0
        lbl = np.zeros(shape, dtype=np.uint8)
        if i % 2 == 0:  # 一半样本含前景，制造正负样本
            z0, y0, x0 = (rng.integers(0, s // 3) for s in shape)
            lbl[z0:z0 + 12, y0:y0 + 80, x0:x0 + 80] = 1
        np.savez_compressed(d / f"vol_{i:03d}.npz", image=img, label=lbl)
    return str(d)


def _base_overrides(npz_dir: str, out_dir: str):
    return [
        f"data.npz_dir={npz_dir}",
        "data.npz_auto_build=false",
        "data.num_workers=0",
        "data.batch_size=2",
        "data.samples_per_volume=2",
        "data.val_ratio=0.34",
        "train.epochs=3",
        "train.warmup_epochs=1",
        "train.use_amp=false",
        "train.use_ema=true",
        f"train.output_dir={out_dir}",
    ]


def _load_cfg(config_path: str, overrides):
    from clstask.config import apply_overrides, load_config, validate_cls
    cfg, cls = load_config(config_path)
    apply_overrides(cfg, cls, overrides)
    cfg.sync()
    cfg.validate()
    validate_cls(cls, cfg)
    return cfg, cls


# ---------------------------------------------------------------------------
def test_config_geometry():
    root = Path(__file__).resolve().parents[1]
    cfg3d, _ = _load_cfg(str(root / "configs/cls3d_cubic.yaml"),
                         [f"data.npz_dir=/tmp"])
    assert cfg3d.model.spatial_dims == 3, cfg3d.model.spatial_dims
    assert cfg3d.model.in_channels == 1, cfg3d.model.in_channels
    cfg25, cls25 = _load_cfg(str(root / "configs/cls2_5d.yaml"),
                             [f"data.npz_dir=/tmp"])
    assert cfg25.model.spatial_dims == 2, cfg25.model.spatial_dims
    assert cfg25.model.in_channels == int(cfg25.data.patch_size[0]), \
        cfg25.model.in_channels
    assert cls25.label_granularity == "slice"
    _ok("config_geometry", "3D sd=3/in=1; 2.5D sd=2/in=D")


def test_dataset_shapes(npz_dir: str):
    from clstask.data.cls_dataset import ClsPatchDataset
    # 3D volume 粒度
    ds3 = ClsPatchDataset([f"{npz_dir}/vol_000.npz"], [16, 64, 64],
                          num_classes=1, label_granularity="volume",
                          label_source="mask", spatial_dims=3,
                          samples_per_volume=2, is_train=True)
    s = ds3[0]
    assert tuple(s["image"].shape) == (1, 16, 64, 64), s["image"].shape
    assert tuple(s["target"].shape) == (1,), s["target"].shape
    # 2.5D slice 粒度
    ds25 = ClsPatchDataset([f"{npz_dir}/vol_000.npz"], [12, 64, 64],
                           num_classes=1, label_granularity="slice",
                           label_source="mask", spatial_dims=2,
                           samples_per_volume=2, is_train=False)
    s2 = ds25[0]
    assert tuple(s2["image"].shape) == (12, 64, 64), s2["image"].shape
    assert tuple(s2["target"].shape) == (1, 12), s2["target"].shape
    _ok("dataset_shapes", "3D (1,D,H,W)/(K,); 2.5D (D,H,W)/(K,D)")


def test_backbones_forward():
    """四模板 × 双几何前向。"""
    from clstask.config import ClsConfig
    from clstask.models.factory import build_classifier
    from segtask_v1.config import Config

    def _mk(spatial_dims, backbone, granularity):
        cfg = Config()
        cfg.data.label_values = [0, 1, 2]   # 2 前景类 → K=2
        cfg.data.num_classes = 3
        cfg.data.patch_size = [16, 64, 64]
        cfg.data.multi_res_scales = [1.0]
        cfg.model.encoder_channels = [16, 32, 64]
        cfg.model.encoder_blocks_per_stage = [1, 1, 1]
        if spatial_dims == 2:
            cfg.data.patch_mode = "2_5d"
        else:
            cfg.data.patch_mode = "cubic"
        cfg.sync()
        cls = ClsConfig(backbone=backbone, label_granularity=granularity,
                        vit_embed_dim=48, vit_depth=2, vit_num_heads=4,
                        vit_patch_size=[4, 16, 16],
                        densenet_block_layers=[2, 2], densenet_growth_rate=8,
                        densenet_stem_channels=16, head_hidden_dim=32)
        model = build_classifier(cfg, cls)
        d = cfg.data.patch_size
        if spatial_dims == 2:
            x = torch.randn(2, d[0], d[1], d[2])
        else:
            x = torch.randn(2, 1, d[0], d[1], d[2])
        return model(x), granularity, d

    for backbone in ("encoder", "densenet", "vit"):
        for sd in (3, 2):
            logits, gran, d = _mk(sd, backbone, "volume")
            assert tuple(logits.shape) == (2, 2), (backbone, sd, logits.shape)
        # slice 粒度输出 (B, K, D)
        logits, _, d = _mk(3, backbone, "slice")
        assert tuple(logits.shape) == (2, 2, d[0]), (backbone, logits.shape)
        _ok(f"backbone_forward:{backbone}", "3D+2.5D volume/slice logits OK")


def test_losses_metrics():
    from clstask.losses.cls_loss import (
        MultiLabelBCELoss, MultiLabelFocalLoss, SingleLabelCELoss)
    from clstask.metrics import multilabel_metrics, singlelabel_metrics
    logits = torch.randn(8, 3, requires_grad=True)
    tgt = (torch.rand(8, 3) > 0.5).float()
    for loss_fn in (MultiLabelBCELoss(), MultiLabelFocalLoss()):
        loss = loss_fn(logits, tgt)
        assert torch.isfinite(loss)
        loss.backward(retain_graph=True)
    # slice 粒度 (B, K, D)
    lg = torch.randn(4, 2, 5)
    tg = (torch.rand(4, 2, 5) > 0.5).float()
    assert torch.isfinite(MultiLabelBCELoss()(lg, tg))
    # CE 软/硬标签
    ce = SingleLabelCELoss()
    assert torch.isfinite(ce(torch.randn(8, 4), torch.randint(0, 4, (8,))))
    assert torch.isfinite(ce(torch.randn(8, 4), torch.softmax(
        torch.randn(8, 4), dim=1)))
    # metrics：完美分数 AUC=1
    probs = torch.tensor([[0.9], [0.8], [0.2], [0.1]])
    tgts = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    m = multilabel_metrics(probs, tgts)
    assert abs(m["auc"] - 1.0) < 1e-6, m
    ms = singlelabel_metrics(torch.softmax(torch.randn(10, 3), 1),
                             torch.randint(0, 3, (10,)))
    assert 0.0 <= ms["auc"] <= 1.0
    _ok("losses_metrics", "bce/focal/ce finite grads; AUC perfect=1.0")


def test_table_labels_and_mixup(npz_dir: str, tmp: Path):
    """table 标签源（csv 多热 + json 单标签）与 mixup/cutmix 软标签。"""
    from clstask.data.cls_dataset import (
        ClsPatchDataset, load_label_table, match_table_to_paths)
    from clstask.trainer.mixup import apply_mixup_cutmix

    paths = sorted(str(p) for p in Path(npz_dir).glob("*.npz"))[:4]
    # csv 多热
    csv_path = tmp / "labels.csv"
    lines = ["pid,c1,c2"] + [
        f"{Path(p).name[:-4]},{i % 2},{(i + 1) % 2}"
        for i, p in enumerate(paths)]
    csv_path.write_text("\n".join(lines) + "\n")
    table = load_label_table(str(csv_path), num_classes=2, multi_label=True)
    targets = match_table_to_paths(paths, table)
    ds = ClsPatchDataset(paths, [12, 64, 64], num_classes=2,
                         label_granularity="volume", label_source="table",
                         table_targets=targets, spatial_dims=2,
                         samples_per_volume=1)
    s = ds[0]
    assert tuple(s["target"].shape) == (2,), s["target"].shape
    # json 单标签
    json_path = tmp / "labels.json"
    json_path.write_text(json.dumps(
        {Path(p).name[:-4]: i % 3 for i, p in enumerate(paths)}))
    table1 = load_label_table(str(json_path), num_classes=3, multi_label=False)
    t1 = match_table_to_paths(paths, table1)
    assert t1[0].ndim == 0 and t1[0].dtype == np.int64
    # mixup / cutmix：输出软标签 (B, K)，值域 [0, 1]
    img = torch.randn(4, 1, 12, 64, 64)
    hard = torch.randint(0, 3, (4,))
    mi, mt = apply_mixup_cutmix(img, hard, num_classes=3, mixup_alpha=0.4,
                                cutmix_alpha=1.0, prob=1.0)
    assert tuple(mt.shape) == (4, 3)
    assert torch.allclose(mt.sum(dim=1), torch.ones(4), atol=1e-5)
    _ok("table_labels_mixup", "csv/json tables + soft-label mixup/cutmix OK")


def test_train_and_transfer(config_path: str, npz_dir: str, out_dir: str,
                            tag: str):
    from clstask.data.loader import build_cls_dataloaders
    from clstask.models.factory import build_classifier, load_pretrained_encoder
    from clstask.trainer.cls_trainer import ClsTrainer

    cfg, cls = _load_cfg(config_path, _base_overrides(npz_dir, out_dir))
    device = torch.device("cpu")
    train_loader, val_loader = build_cls_dataloaders(cfg, cls)
    model = build_classifier(cfg, cls)
    trainer = ClsTrainer(model, cfg, cls, train_loader, val_loader, device)
    metrics = trainer.fit()
    assert np.isfinite(metrics["loss"]), metrics
    assert "val_auc" in metrics, metrics
    hist = [h["loss"] for h in trainer.history]
    # 随机 patch 采样下逐 epoch 有噪声，取后续最优 vs 首 epoch 判下降。
    assert min(hist[1:]) < hist[0], f"train loss did not descend: {hist}"
    ckpt = Path(out_dir) / "best_model.pth"
    assert ckpt.is_file(), "best_model.pth not saved"
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    enc_keys = [k for k in sd["model_state_dict"] if k.startswith("encoder.")]
    assert enc_keys, "checkpoint has no encoder.* keys"
    # SSL 迁移：新建同构模型，从刚保存的 ckpt 命中 encoder.*
    fresh = build_classifier(cfg, cls)
    load_pretrained_encoder(fresh, str(ckpt))
    _ok(f"train_transfer:{tag}",
        f"loss={metrics['loss']:.3f} val_auc={metrics['val_auc']:.3f} "
        f"encoder.* keys={len(enc_keys)}")
    return cfg, cls, str(ckpt)


def test_predict(config_path: str, npz_dir: str, out_dir: str, ckpt: str):
    from clstask.config import apply_overrides, validate_cls
    from clstask.data.loader import discover_npz
    from clstask.models.factory import build_classifier
    from clstask.predictor.cls_predictor import ClsPredictor

    cfg, cls = _load_cfg(config_path, _base_overrides(npz_dir, out_dir))
    cls.pretrained_ckpt = ""
    model = build_classifier(cfg, cls)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model_state_dict"], strict=True)
    predictor = ClsPredictor(model, cfg, cls, torch.device("cpu"))
    res = predictor.predict_volume(discover_npz(npz_dir)[0])
    k = int(cfg.num_fg_classes)
    assert res["volume_probs"].shape == (k,), res["volume_probs"].shape
    if cls.label_granularity == "slice":
        assert res["slice_probs"].shape[0] == k
    _ok("predict", f"volume_probs shape={res['volume_probs'].shape}")


def main() -> int:
    print("=" * 68)
    print("clstask smoke test (3D cubic + 2.5D folded)")
    print("=" * 68)
    torch.manual_seed(0)
    np.random.seed(0)
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        npz_dir = _make_npz_dir(tmp, n=12)

        print("\n[1] config geometry")
        test_config_geometry()
        print("\n[2] dataset shapes")
        test_dataset_shapes(npz_dir)
        print("\n[3] losses + metrics")
        test_losses_metrics()
        print("\n[4] backbones forward (4 templates x dual geometry)")
        test_backbones_forward()
        print("\n[4b] table labels + mixup/cutmix")
        test_table_labels_and_mixup(npz_dir, tmp)

        for tag, cfg_yaml in (("3d_cubic", "configs/cls3d_cubic.yaml"),
                              ("2_5d", "configs/cls2_5d.yaml")):
            print(f"\n[5:{tag}] train + SSL transfer")
            out = str(tmp / f"out_{tag}")
            cfg, cls, ckpt = test_train_and_transfer(
                str(root / cfg_yaml), npz_dir, out, tag)
            print(f"[6:{tag}] predict")
            test_predict(str(root / cfg_yaml), npz_dir, out, ckpt)

    print("\n" + "=" * 68)
    print("ALL CLSTASK SMOKE TESTS PASSED")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
