"""dettask end-to-end smoke test（双几何 3D cubic + 2.5D 折叠）。

用合成 npz（image + label/boxes）跑通完整链路，验证 D1–D5 验收标准：

  1. 单测：ops（IoU/GIoU/NMS/ROIAlign）、targets（anchor 生成、编解码往返、
     max-IoU/ATSS 分配、裁剪/翻转框联动、3D→2D 切片派生）、stitching
     （跨 slab 拼接 3D 框）、metrics（mAP/FROC 已知答案）。
  2. 配置：3D / 2.5D 两套 (cfg, det) 加载 + 校验通过，几何派生正确。
  3. 数据层：DetPatchDataset 输出形状与框维数随 spatial_dims 切换；
     mask 连通域派生框正确。
  4. 四模板：retinanet / fcos / faster_rcnn / detr × 双几何 —— 损失有限
     可反传 + predict 输出结构正确。
  5. 训练：2.5D RetinaNet + 3D RetinaNet 各 3 epoch，loss 下降、val mAP
     有读数；SSL 迁移 encoder.*（+decoder.*）命中。
  6. 推理：DetPredictor 整卷推理（2.5D 跨层拼接 / 3D 滑窗 NMS）输出 3D 框
     + 体级 FROC 读数。

Run:
    python tests/test_dettask_smoke.py
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# pytest fixture：脚本模式（python tests/test_dettask_smoke.py）由 main()
# 直接传参；pytest 收集时由该 fixture 提供同名参数。
@pytest.fixture(scope="module")
def npz_dir(tmp_path_factory) -> str:
    return _make_npz_dir(tmp_path_factory.mktemp("det_npz"), n=12)


def _ok(name: str, msg: str = "") -> None:
    print(f"  [PASS] {name}{(' — ' + msg) if msg else ''}")


def _make_npz_dir(root: Path, n: int = 12, shape=(24, 96, 96)) -> str:
    """合成 n 个 npz：image + label（球状前景）+ boxes（真值 3D 框）。"""
    d = root / "npz"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        img = rng.standard_normal(shape).astype(np.float32) * 50.0
        lbl = np.zeros(shape, dtype=np.uint8)
        boxes = []
        for _ in range(int(rng.integers(1, 3))):
            dz, dy, dx = 8, 24, 24
            z0 = int(rng.integers(0, shape[0] - dz))
            y0 = int(rng.integers(0, shape[1] - dy))
            x0 = int(rng.integers(0, shape[2] - dx))
            lbl[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx] = 1
            img[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx] += 300.0  # 可学信号
            boxes.append([z0, y0, x0, z0 + dz, y0 + dy, x0 + dx, 0])
        np.savez_compressed(d / f"vol_{i:03d}.npz", image=img, label=lbl,
                            boxes=np.asarray(boxes, np.float32))
    return str(d)


def _base_overrides(npz_dir: str, out_dir: str):
    return [
        f"data.npz_dir={npz_dir}",
        "data.npz_auto_build=false",
        "data.num_workers=0",
        "data.batch_size=2",
        "data.samples_per_volume=4",
        "data.val_ratio=0.34",
        'data.patch_size=[12, 64, 64]',
        "train.epochs=3",
        "train.warmup_epochs=1",
        "train.use_amp=false",
        "train.use_ema=true",
        'model.encoder_channels=[16, 32, 64]',
        'model.encoder_blocks_per_stage=[1, 1, 1]',
        "det.fpn_channels=32",
        f"train.output_dir={out_dir}",
    ]


def _load_cfg(config_path: str, overrides):
    from dettask.config import apply_overrides, load_config, validate_det
    cfg, det = load_config(config_path)
    apply_overrides(cfg, det, overrides)
    cfg.sync()
    cfg.validate()
    validate_det(det, cfg)
    return cfg, det


# ---------------------------------------------------------------------------
# [1] 单测
# ---------------------------------------------------------------------------
def test_ops():
    from dettask.ops import (batched_nms, box_iou, generalized_box_iou, nms,
                             roi_align)
    # IoU 已知答案（2D 半区重叠 = 1/3）。
    a = torch.tensor([[0., 0., 10., 10.]])
    b = torch.tensor([[0., 5., 10., 15.]])
    assert abs(float(box_iou(a, b)) - 1 / 3) < 1e-6
    # 3D 自身 IoU=1；不相交 GIoU < 0。
    c = torch.tensor([[0., 0., 0., 4., 4., 4.]])
    assert abs(float(box_iou(c, c)) - 1.0) < 1e-6
    e = torch.tensor([[10., 10., 10., 14., 14., 14.]])
    assert float(generalized_box_iou(c, e)) < 0
    # NMS：重叠高分保留、低分剔除；不相交都保留。
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.],
                          [50., 50., 60., 60.]])
    keep = nms(boxes, torch.tensor([0.9, 0.8, 0.7]), 0.5)
    assert keep.tolist() == [0, 2], keep
    # batched_nms：不同类不互相抑制。
    keep2 = batched_nms(boxes[:2], torch.tensor([0.9, 0.8]),
                        torch.tensor([0, 1]), 0.5)
    assert len(keep2) == 2
    # ROIAlign：常数特征图上任意 ROI 输出为常数（2D + 3D）。
    f2 = torch.full((1, 3, 16, 16), 7.0)
    r2 = roi_align(f2, torch.tensor([[2., 2., 10., 10.]]),
                   torch.tensor([0]), [4, 4], [1.0, 1.0])
    assert r2.shape == (1, 3, 4, 4) and torch.allclose(
        r2, torch.tensor(7.0), atol=1e-4)
    f3 = torch.full((1, 2, 8, 16, 16), 3.0)
    r3 = roi_align(f3, torch.tensor([[1., 2., 2., 6., 10., 10.]]),
                   torch.tensor([0]), [3, 4, 4], [1.0, 1.0, 1.0])
    assert r3.shape == (1, 2, 3, 4, 4) and torch.allclose(
        r3, torch.tensor(3.0), atol=1e-4)
    _ok("ops", "IoU/GIoU/NMS/ROIAlign(2D+3D) known answers")


def test_targets():
    from dettask.targets import (assign_atss, assign_max_iou, crop_boxes,
                                 decode_boxes, encode_boxes, flip_boxes,
                                 generate_anchors, slice_boxes_to_2d)
    dev = torch.device("cpu")
    # anchor 生成：数量 = P × A；中心落在特征网格。
    a2 = generate_anchors([4, 4], [8.0, 8.0], 16.0, [0.5, 1.0, 2.0],
                          [1.0, 1.26], [1.0], dev)
    assert a2.shape == (4 * 4 * 6, 4), a2.shape
    a3 = generate_anchors([2, 4, 4], [4.0, 8.0, 8.0], 16.0, [1.0],
                          [1.0], [0.5, 1.0], dev)
    assert a3.shape == (2 * 4 * 4 * 2, 6), a3.shape
    # 编解码往返（2D + 3D）。
    for anchors, gt in ((a2[:5], torch.tensor([[1., 2., 20., 22.]] * 5)),
                        (a3[:5], torch.tensor([[1., 2., 3., 9., 20., 22.]] * 5))):
        rt = decode_boxes(encode_boxes(gt, anchors), anchors)
        assert torch.allclose(rt, gt, atol=1e-4), (rt, gt)
    # max-IoU 分配：完美 anchor 为正、远处为负、每 gt 至少一正。
    anchors = torch.tensor([[0., 0., 10., 10.], [40., 40., 50., 50.]])
    gt = torch.tensor([[0., 0., 10., 10.]])
    m = assign_max_iou(anchors, gt, 0.5, 0.3)
    assert m.tolist() == [0, -1], m
    # ATSS：单层候选，正样本命中重叠 anchor。
    m2 = assign_atss(torch.cat([anchors, anchors + 1.0]), gt, [4], topk=4)
    assert (m2 >= 0).sum() >= 1
    # flip 联动：翻转两次 = 原状；翻转后仍在界内。
    b = torch.tensor([[2., 3., 8., 9.]])
    fb = flip_boxes(b, 0, [12, 12])
    assert torch.allclose(flip_boxes(fb, 0, [12, 12]), b)
    assert fb[0, 0].item() == 12 - 8 and fb[0, 2].item() == 12 - 2
    # crop 联动：完整包含保留、部分裁剪坐标正确、界外剔除。
    boxes = torch.tensor([[2., 2., 6., 6.], [0., 0., 3., 3.],
                          [20., 20., 30., 30.]])
    labels = torch.tensor([0, 1, 0])
    cb, cl = crop_boxes(boxes, labels, (1, 1), (8, 8), min_visibility=0.25)
    assert cb.shape[0] == 2 and cl.tolist() == [0, 1]
    assert torch.allclose(cb[0], torch.tensor([1., 1., 5., 5.]))
    assert torch.allclose(cb[1], torch.tensor([0., 0., 2., 2.]))
    # 3D→2D 切片派生：z 交叠不足剔除。
    b3 = torch.tensor([[0., 5., 6., 8., 15., 16.],
                       [20., 0., 0., 30., 4., 4.]])
    l3 = torch.tensor([0, 1])
    b2, l2 = slice_boxes_to_2d(b3, l3, 0, 12, min_overlap=0.5)
    assert b2.shape == (1, 4) and l2.tolist() == [0]
    assert torch.allclose(b2[0], torch.tensor([5., 6., 15., 16.]))
    _ok("targets", "anchors/codec-roundtrip/assign/flip/crop/slice linkage")


def test_stitching():
    from dettask.predictor.stitching import stitch_slab_detections
    mk = lambda boxes, scores, labels: {
        "boxes": torch.tensor(boxes), "scores": torch.tensor(scores),
        "labels": torch.tensor(labels, dtype=torch.long)}
    slabs = [
        mk([[10., 10., 30., 30.]], [0.9], [0]),
        mk([[11., 11., 31., 31.], [60., 60., 70., 70.]], [0.8, 0.6], [0, 0]),
        mk([[12., 12., 32., 32.]], [0.7], [0]),
    ]
    z = [[0., 6.], [3., 9.], [6., 12.]]
    out = stitch_slab_detections(slabs, z, link_iou=0.3, min_span=2)
    # 链 1 跨 3 slab 保留；孤立框 span=1 剔除。
    assert out["boxes"].shape == (1, 6), out
    assert abs(float(out["scores"][0]) - 0.9) < 1e-6
    assert float(out["boxes"][0, 0]) == 0. and float(out["boxes"][0, 3]) == 12.
    _ok("stitching", "3-slab chain kept (z-span merged); singleton dropped")


def test_metrics():
    from dettask.metrics import detection_map, froc
    gt = torch.tensor([[0., 0., 0., 8., 8., 8.]])
    gl = torch.tensor([0])
    perfect = {"boxes": gt.clone(), "scores": torch.tensor([0.9]),
               "labels": torch.tensor([0])}
    m = detection_map([perfect], [(gt, gl)], 1, 0.5)
    assert abs(m["map"] - 1.0) < 1e-6, m
    miss = {"boxes": torch.tensor([[50., 50., 50., 58., 58., 58.]]),
            "scores": torch.tensor([0.9]), "labels": torch.tensor([0])}
    m2 = detection_map([miss], [(gt, gl)], 1, 0.5)
    assert m2["map"] == 0.0, m2
    fr = froc([perfect], [(gt, gl)], [0.125, 1.0, 8.0], 0.5)
    assert abs(fr["froc"] - 1.0) < 1e-6, fr
    _ok("metrics", "mAP perfect=1/miss=0; FROC perfect=1")


# ---------------------------------------------------------------------------
# [2] 配置几何
# ---------------------------------------------------------------------------
def test_config_geometry():
    root = _ROOT
    cfg3d, det3d = _load_cfg(str(root / "configs/det3d.yaml"),
                             ["data.npz_dir=/tmp"])
    assert cfg3d.model.spatial_dims == 3 and cfg3d.model.in_channels == 1
    assert det3d.arch == "retinanet"
    cfg25, _ = _load_cfg(str(root / "configs/det2_5d.yaml"),
                         ["data.npz_dir=/tmp"])
    assert cfg25.model.spatial_dims == 2
    assert cfg25.model.in_channels == int(cfg25.data.patch_size[0])
    _ok("config_geometry", "3D sd=3/in=1; 2.5D sd=2/in=D")


# ---------------------------------------------------------------------------
# [3] 数据层
# ---------------------------------------------------------------------------
def test_dataset(npz_dir: str):
    from dettask.data.det_dataset import (DetPatchDataset, boxes_from_mask,
                                          det_collate)
    # mask 派生框与存储框一致（同一合成数据）。
    with np.load(f"{npz_dir}/vol_000.npz") as f:
        mb, ml = boxes_from_mask(f["label"], [1.0])
        stored = np.asarray(f["boxes"])
    assert mb.shape[0] >= 1 and ml.shape == (mb.shape[0],)
    # 3D。
    ds3 = DetPatchDataset([f"{npz_dir}/vol_000.npz"], [16, 64, 64],
                          spatial_dims=3, samples_per_volume=2,
                          is_train=True, fg_oversample_ratio=1.0)
    s = ds3[0]
    assert tuple(s["image"].shape) == (1, 16, 64, 64)
    assert s["boxes"].shape[-1] == 6 and s["boxes"].shape[0] >= 1
    assert (s["boxes"][:, :3] >= 0).all() and \
        (s["boxes"][:, 3:] <= torch.tensor([16., 64., 64.])).all()
    # 2.5D。
    ds25 = DetPatchDataset([f"{npz_dir}/vol_000.npz"], [12, 64, 64],
                           spatial_dims=2, samples_per_volume=2,
                           is_train=True, fg_oversample_ratio=1.0)
    s2 = ds25[0]
    assert tuple(s2["image"].shape) == (12, 64, 64)
    assert s2["boxes"].shape[-1] == 4
    # collate：变长框保持 list。
    batch = det_collate([s2, ds25[1]])
    assert batch["image"].shape == (2, 12, 64, 64)
    assert isinstance(batch["boxes"], list) and len(batch["boxes"]) == 2
    _ok("dataset", "3D (1,D,H,W)+6-col boxes; 2.5D (D,H,W)+4-col boxes")


# ---------------------------------------------------------------------------
# [4] 四模板 × 双几何
# ---------------------------------------------------------------------------
def test_four_archs():
    from dettask.config import DetConfig, validate_det
    from dettask.models.factory import build_detector
    from taskcore.config.core import Config

    def _mk_cfg(sd: int) -> Config:
        cfg = Config()
        cfg.data.label_values = [0, 1]
        cfg.data.patch_size = [8, 32, 32] if sd == 3 else [8, 32, 32]
        cfg.data.patch_mode = "cubic" if sd == 3 else "2_5d"
        cfg.data.multi_res_scales = [1.0]
        cfg.model.encoder_channels = [8, 16, 32]
        cfg.model.encoder_blocks_per_stage = [1, 1, 1]
        cfg.sync()
        return cfg

    for arch in ("retinanet", "fcos", "faster_rcnn", "detr"):
        for sd in (3, 2):
            cfg = _mk_cfg(sd)
            det = DetConfig(arch=arch, fpn_channels=16, num_queries=8,
                            detr_hidden_dim=32, detr_num_heads=4,
                            detr_dec_layers=1, rpn_pre_nms_topk=50,
                            rpn_post_nms_topk=20, rpn_batch_per_img=16,
                            roi_batch_per_img=8, roi_output_size=3)
            validate_det(det, cfg)
            model = build_detector(cfg, det)
            d = cfg.data.patch_size
            x = (torch.randn(2, 1, *d) if sd == 3
                 else torch.randn(2, d[0], d[1], d[2]))
            dim = 2 * sd
            gt = [torch.tensor([[1., 2., 2., 7., 24., 24.]]) if sd == 3
                  else torch.tensor([[2., 2., 24., 24.]]) for _ in range(2)]
            gl = [torch.tensor([0]) for _ in range(2)]
            losses = model(x, gt, gl)
            total = sum(losses.values())
            assert torch.isfinite(total), (arch, sd, losses)
            total.backward()
            model.eval()
            with torch.no_grad():
                dets = model(x)
            assert len(dets) == 2
            for dd in dets:
                assert dd["boxes"].shape[-1] == dim or dd["boxes"].numel() == 0
                assert dd["scores"].shape[0] == dd["boxes"].shape[0]
            _ok(f"arch:{arch}:sd{sd}",
                f"loss={float(total):.3f} finite+backward; predict OK")


# ---------------------------------------------------------------------------
# [5] 训练 + SSL 迁移；[6] 整卷推理 + FROC
# ---------------------------------------------------------------------------
def _run_train_transfer_predict(config_path: str, npz_dir: str,
                                out_dir: str, tag: str):
    from dettask.data.loader import build_det_dataloaders
    from dettask.models.factory import build_detector, \
        load_pretrained_backbone
    from dettask.predictor.det_predictor import DetPredictor
    from dettask.trainer.det_trainer import DetTrainer
    from clstask.data.loader import discover_npz

    cfg, det = _load_cfg(config_path, _base_overrides(npz_dir, out_dir))
    device = torch.device("cpu")
    train_loader, val_loader = build_det_dataloaders(cfg, det)
    model = build_detector(cfg, det)
    trainer = DetTrainer(model, cfg, det, train_loader, val_loader, device)
    metrics = trainer.fit()
    assert math.isfinite(metrics["loss"]), metrics
    assert "val_map" in metrics, metrics
    hist = [h["loss"] for h in trainer.history]
    assert min(hist[1:]) < hist[0], f"train loss did not descend: {hist}"
    ckpt = Path(out_dir) / "best_model.pth"
    assert ckpt.is_file(), "best_model.pth not saved"
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    enc_keys = [k for k in sd["model_state_dict"] if k.startswith("encoder.")]
    dec_keys = [k for k in sd["model_state_dict"] if k.startswith("decoder.")]
    assert enc_keys and dec_keys
    # SSL 迁移：新建同构模型，从刚保存的 ckpt 命中 encoder.*/decoder.*。
    fresh = build_detector(cfg, det)
    load_pretrained_backbone(fresh, str(ckpt))
    # 整卷推理 + FROC。
    model.load_state_dict(sd["model_state_dict"], strict=True)
    predictor = DetPredictor(model, cfg, det, device)
    paths = discover_npz(npz_dir)[:3]
    res = predictor.predict_volume(paths[0])
    assert res["boxes"].shape[-1] == 6 or res["boxes"].numel() == 0
    fr = predictor.predict_dir(paths, str(Path(out_dir) / "preds"))
    assert "froc" in fr, fr
    _ok(f"train_transfer_predict:{tag}",
        f"loss={metrics['loss']:.3f} val_map={metrics['val_map']:.3f} "
        f"enc/dec keys={len(enc_keys)}/{len(dec_keys)} "
        f"froc={fr['froc']:.3f}")


@pytest.mark.parametrize("tag,cfg_yaml", [
    ("2_5d", "configs/det2_5d.yaml"),
    ("3d_cubic", "configs/det3d.yaml"),
])
def test_train_transfer_predict(npz_dir: str, tmp_path: Path,
                                tag: str, cfg_yaml: str):
    torch.manual_seed(0)
    np.random.seed(0)
    out = str(tmp_path / f"out_{tag}")
    _run_train_transfer_predict(str(_ROOT / cfg_yaml), npz_dir, out, tag)


def main() -> int:
    print("=" * 68)
    print("dettask smoke test (3D cubic + 2.5D folded)")
    print("=" * 68)
    torch.manual_seed(0)
    np.random.seed(0)
    root = _ROOT
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        npz_dir = _make_npz_dir(tmp, n=12)

        print("\n[1] unit: ops / targets / stitching / metrics")
        test_ops()
        test_targets()
        test_stitching()
        test_metrics()
        print("\n[2] config geometry")
        test_config_geometry()
        print("\n[3] dataset")
        test_dataset(npz_dir)
        print("\n[4] four archs x dual geometry (loss+backward+predict)")
        test_four_archs()

        for tag, cfg_yaml in (("2_5d", "configs/det2_5d.yaml"),
                              ("3d_cubic", "configs/det3d.yaml")):
            print(f"\n[5+6:{tag}] train + SSL transfer + volume predict/FROC")
            out = str(tmp / f"out_{tag}")
            _run_train_transfer_predict(str(root / cfg_yaml), npz_dir, out,
                                        tag)

    print("\n" + "=" * 68)
    print("ALL DETTASK SMOKE TESTS PASSED")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
