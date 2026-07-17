from __future__ import annotations

import contextlib
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _write_npz(
    path: Path,
    image: np.ndarray,
    label: np.ndarray,
    fg_slices: np.ndarray | None = None,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fg_slices is None:
        fg_slices = np.arange(image.shape[0], dtype=np.int32)
    np.savez(path, image=image, label=label, fg_slices=fg_slices)
    return str(path)


def test_keep_last_k_checkpoint_pruning_preserves_best_and_can_disable():
    from taskcore.config.core import Config
    from segtask_v1.trainer import Trainer

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        for epoch in range(1, 6):
            (out / f"checkpoint_epoch_{epoch}.pth").write_text(f"ckpt{epoch}")
        (out / "best_model.pth").write_text("best")

        cfg = Config()
        cfg.train.save_keep_last = 2
        trainer = Trainer.__new__(Trainer)
        trainer.cfg = cfg
        trainer.output_dir = out

        trainer._prune_old_checkpoints()

        remain = sorted(p.name for p in out.glob("checkpoint_epoch_*.pth"))
        assert remain == ["checkpoint_epoch_4.pth", "checkpoint_epoch_5.pth"]
        assert (out / "best_model.pth").read_text() == "best"

        for epoch in range(1, 4):
            (out / f"checkpoint_epoch_{epoch}.pth").write_text(f"ckpt{epoch}")
        cfg.train.save_keep_last = 0
        trainer._prune_old_checkpoints()
        remain = sorted(p.name for p in out.glob("checkpoint_epoch_*.pth"))
        assert remain == [
            "checkpoint_epoch_1.pth",
            "checkpoint_epoch_2.pth",
            "checkpoint_epoch_3.pth",
            "checkpoint_epoch_4.pth",
            "checkpoint_epoch_5.pth",
        ]


def test_affine_aspect_correct_preserves_integer_voxel_mapping():
    from taskcore.data.augment import _build_rotation_matrices

    d, h, w = 2, 4, 6
    image = torch.empty(1, 1, d, h, w, dtype=torch.float32)
    for z in range(d):
        for y in range(h):
            for x in range(w):
                image[0, 0, z, y, x] = z * 100.0 + y * 10.0 + x

    angles = torch.tensor([[0.0, 0.0, math.pi / 2]], dtype=torch.float32)
    scales = torch.ones(1, 1, dtype=torch.float32)

    aspect = torch.tensor([float(w), float(h), float(d)], dtype=torch.float32)
    aff = _build_rotation_matrices(angles, scales, aspect=aspect)
    grid = F.affine_grid(aff, image.shape, align_corners=False)
    coords_x = (grid[..., 0] + 1.0) * w / 2.0 - 0.5
    coords_y = (grid[..., 1] + 1.0) * h / 2.0 - 0.5
    assert max(
        (coords_x - coords_x.round()).abs().max().item(),
        (coords_y - coords_y.round()).abs().max().item(),
    ) < 1e-5

    rotated = F.grid_sample(
        image, grid, mode="bilinear", padding_mode="border", align_corners=False)
    exp = torch.empty_like(rotated)
    x_idx = coords_x.round().clamp(0, w - 1).long()
    y_idx = coords_y.round().clamp(0, h - 1).long()
    for z in range(d):
        exp[0, 0, z] = image[0, 0, z][y_idx[0, z], x_idx[0, z]]
    assert torch.allclose(rotated, exp, atol=1e-5, rtol=0.0)

    aff_no = _build_rotation_matrices(angles, scales, aspect=None)
    grid_no = F.affine_grid(aff_no, image.shape, align_corners=False)
    coords_x_no = (grid_no[..., 0] + 1.0) * w / 2.0 - 0.5
    coords_y_no = (grid_no[..., 1] + 1.0) * h / 2.0 - 0.5
    assert max(
        (coords_x_no - coords_x_no.round()).abs().max().item(),
        (coords_y_no - coords_y_no.round()).abs().max().item(),
    ) >= 0.49
    rotated_no = F.grid_sample(
        image, grid_no, mode="bilinear", padding_mode="border", align_corners=False)
    assert not torch.allclose(rotated_no, exp, atol=1e-4, rtol=0.0)


def test_scheduler_horizon_respects_optimizer_step_accumulation():
    from taskcore.config.core import Config
    from segtask_v1.trainer import build_scheduler

    cfg = Config()
    cfg.data.label_values = [0, 1]
    cfg.data.num_classes = 2
    cfg.train.scheduler = "one_cycle"
    cfg.train.epochs = 4
    cfg.train.warmup_epochs = 1
    cfg.sync()

    opt = torch.optim.Adam([torch.randn(2, requires_grad=True)], lr=0.01)
    loader_len = 9
    accum1 = 1
    accum3 = 3
    steps1 = math.ceil(loader_len / accum1)
    steps3 = math.ceil(loader_len / accum3)

    sched1 = build_scheduler(opt, cfg, steps_per_epoch=steps1, post_warmup_steps=0)
    sched3 = build_scheduler(opt, cfg, steps_per_epoch=steps3, post_warmup_steps=0)
    assert sched1.total_steps == cfg.train.epochs * steps1
    assert sched3.total_steps == cfg.train.epochs * steps3
    assert sched1.total_steps == 36
    assert sched3.total_steps == 12

    cfg.train.scheduler = "cosine"
    cfg.sync()
    post1 = max((cfg.train.epochs - cfg.train.warmup_epochs) * steps1, 1)
    post3 = max((cfg.train.epochs - cfg.train.warmup_epochs) * steps3, 1)
    cosine1 = build_scheduler(opt, cfg, steps_per_epoch=steps1, post_warmup_steps=post1)
    cosine3 = build_scheduler(opt, cfg, steps_per_epoch=steps3, post_warmup_steps=post3)
    assert cosine1.T_max == post1
    assert cosine3.T_max == post3
    assert cosine1.T_max == 27
    assert cosine3.T_max == 9


def test_ddp_no_sync_guard_only_triggers_on_nonboundary_microsteps():
    from segtask_v1.trainer.trainer import Trainer

    class _SpyModel:
        def __init__(self):
            self.calls = []

        @contextlib.contextmanager
        def no_sync(self):
            self.calls.append("enter")
            try:
                yield
            finally:
                self.calls.append("exit")

    def _sync_ctx(is_dist: bool, is_step_boundary: bool, model: _SpyModel):
        return (model.no_sync() if (is_dist and not is_step_boundary)
                else contextlib.nullcontext())

    total_steps = 5
    accum = 2
    model = _SpyModel()

    for step in range(total_steps):
        is_step_boundary = ((step + 1) % accum == 0 or (step + 1) == total_steps)
        with _sync_ctx(False, is_step_boundary, model):
            pass
    assert model.calls == []

    model.calls.clear()
    for step in range(total_steps):
        is_step_boundary = ((step + 1) % accum == 0 or (step + 1) == total_steps)
        with _sync_ctx(True, is_step_boundary, model):
            pass
    assert model.calls == ["enter", "exit", "enter", "exit"]
    # 非边界步（step 0 / 2）才走 no_sync；step 1 / 3 / 4 保持正常同步。
    assert Trainer._effective_accum(0, total_steps, accum) == 2
    assert Trainer._effective_accum(4, total_steps, accum) == 1


def test_val_patch_sampling_is_deterministic_for_same_sample_index():
    from taskcore.data.dataset import SegDataset3D

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        npz = _write_npz(
            root / "sample.npz",
            image=np.arange(5 * 4 * 6, dtype=np.int16).reshape(5, 4, 6),
            label=np.zeros((5, 4, 6), dtype=np.int16),
        )

        ds = SegDataset3D(
            image_paths=["dummy.nii.gz"],
            label_paths=["dummy.nii.gz"],
            label_values=[0, 1],
            patch_size=(3, 4, 4),
            aug_oversample_ratio=1.0,
            multi_res_scales=[1.0],
            intensity_min=-1000.0,
            intensity_max=1000.0,
            normalize="minmax",
            foreground_oversample_ratio=0.0,
            samples_per_volume=1,
            is_train=False,
            cache_enabled=False,
            npz_paths=[npz],
        )

        a = ds[0]
        b = ds[0]
        assert torch.equal(a["image"], b["image"])
        assert torch.equal(a["label"], b["label"])


def test_per_class_thresholds_apply_to_matching_class():
    from segtask_v1.predictor.blending import prob_to_label

    prob = np.zeros((2, 2, 2, 2), dtype=np.float32)
    prob[0] = 0.6
    prob[1] = 0.4
    prob[:, 0, 0, 0] = [0.55, 0.58]  # class 1 wins but below its threshold.
    prob[:, 0, 0, 1] = [0.72, 0.10]  # class 0 above threshold.
    prob[:, 0, 1, 0] = [0.10, 0.90]  # class 1 above threshold.

    labels = prob_to_label(
        prob, label_values=[0, 1, 2], num_fg=2, threshold=[0.7, 0.8])
    assert labels[0, 0, 0] == 0
    assert labels[0, 0, 1] == 1
    assert labels[0, 1, 0] == 2
    assert labels[1, 1, 1] == 0
