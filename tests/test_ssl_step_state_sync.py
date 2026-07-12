"""S4: 方法状态（EMA/center/queue）与优化步严格同步；跳步不推进状态。"""

from __future__ import annotations

import pytest
import torch

from segtask_v1.config import Config as SegConfig
from ssltask.config import SSLConfig, validate_ssl
from ssltask.methods import build_method


def _cfg(patch_mode: str = "cubic"):
    cfg = SegConfig()
    cfg.data.patch_mode = patch_mode
    cfg.data.patch_size = [16, 32, 32]
    cfg.data.multi_res_scales = [1.0]
    cfg.data.label_values = [0, 1]
    cfg.data.num_classes = 2
    cfg.model.backbone = "resnet"
    cfg.model.encoder_channels = [8, 16, 32]
    cfg.model.blocks_per_level = 1
    cfg.model.stem_mode = "conv3"
    cfg.sync()
    cfg.validate()
    return cfg


def _dino_ssl(**kw):
    ssl = SSLConfig(method="dino")
    ssl.dino_out_dim = 128
    ssl.dino_hidden_dim = 64
    ssl.dino_bottleneck_dim = 32
    ssl.dino_local_crops = 2
    for k, v in kw.items():
        setattr(ssl, k, v)
    return ssl


def _snapshot(module: torch.nn.Module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _same(a, b) -> bool:
    return all(torch.equal(a[k], b[k]) for k in a)


# ---------------------------------------------------------------------------
# stepped=False：不推进 EMA / center / queue，但推进调度计数
# ---------------------------------------------------------------------------
def test_dino_skipped_step_freezes_teacher_and_center():
    cfg = _cfg()
    ssl = _dino_ssl()
    validate_ssl(ssl, cfg)
    m = build_method(cfg, ssl, torch.device("cpu"))
    m.configure_schedule(10)
    m.train()
    batch = {"image": torch.rand(2, 1, 16, 32, 32)}
    with torch.no_grad():
        for p in m.module.student.parameters():
            p.add_(0.1)                       # 学生 != 教师，EMA 若跑会动教师
    m.compute_loss(batch)                     # 缓存 pending center
    teacher0 = _snapshot(m.module.teacher)
    center0 = m.module.center.clone()

    m.on_after_step(1, stepped=False)         # 跳步
    assert m._step == 1                       # 调度计数照常推进
    assert _same(teacher0, _snapshot(m.module.teacher))
    assert torch.equal(center0, m.module.center)
    assert m._pending_center_n == 0           # pending 被丢弃

    m.compute_loss(batch)
    m.on_after_step(2, stepped=True)          # 正常步
    assert not _same(teacher0, _snapshot(m.module.teacher))
    assert not torch.equal(center0, m.module.center)


def test_dino_center_applied_once_per_boundary_mean_of_microbatches():
    """accum 组内多个 micro-batch 只在边界施加一次 center EMA（组均值）。"""
    cfg = _cfg()
    ssl = _dino_ssl()
    validate_ssl(ssl, cfg)
    m = build_method(cfg, ssl, torch.device("cpu"))
    m.configure_schedule(10)
    m.train()
    torch.manual_seed(0)
    batch = {"image": torch.rand(2, 1, 16, 32, 32)}
    m.compute_loss(batch)
    m.compute_loss(batch)
    assert m._pending_center_n == 2
    expected = (m._pending_center_sum / 2.0).clone()
    mom = m.center_momentum
    center0 = m.module.center.clone()
    m.on_after_step(1, stepped=True)
    want = center0 * mom + expected.to(center0.dtype) * (1.0 - mom)
    assert torch.allclose(m.module.center, want, atol=1e-6)
    assert m._pending_center_n == 0


def test_moco_skipped_step_drops_keys_and_freezes_key_encoder():
    cfg = _cfg()
    ssl = SSLConfig(method="moco")
    ssl.moco_proj_dim = 32
    ssl.moco_queue_size = 8
    ssl.dino_hidden_dim = 64
    validate_ssl(ssl, cfg)
    m = build_method(cfg, ssl, torch.device("cpu"))
    m.configure_schedule(10)
    m.train()
    batch = {"image": torch.rand(2, 1, 16, 32, 32)}
    with torch.no_grad():
        for p in m.module.query.parameters():
            p.add_(0.1)
    m.compute_loss(batch)
    assert len(m._pending_keys) == 1
    key0 = _snapshot(m.module.key)
    queue0 = m.module.queue.clone()

    m.on_after_step(1, stepped=False)
    assert not m._pending_keys                # key 丢弃
    assert torch.equal(queue0, m.module.queue)
    assert int(m.module.queue_ptr.item()) == 0
    assert _same(key0, _snapshot(m.module.key))

    m.compute_loss(batch)
    m.on_after_step(2, stepped=True)
    assert int(m.module.queue_ptr.item()) == 4
    assert not _same(key0, _snapshot(m.module.key))


@pytest.mark.parametrize("method,extra", [
    ("byol", {}),
    ("jepa", {}),
])
def test_ema_methods_skipped_step_freezes_target(method, extra):
    cfg = _cfg()
    ssl = SSLConfig(method=method)
    ssl.dino_hidden_dim = 64
    for k, v in extra.items():
        setattr(ssl, k, v)
    validate_ssl(ssl, cfg)
    m = build_method(cfg, ssl, torch.device("cpu"))
    m.configure_schedule(10)
    online = (m.module.online if method == "byol"
              else m.module.context_encoder)
    target = (m.module.target if method == "byol"
              else m.module.target_encoder)
    with torch.no_grad():
        for p in online.parameters():
            p.add_(0.1)
    t0 = _snapshot(target)
    m.on_after_step(1, stepped=False)
    assert m._step == 1
    assert _same(t0, _snapshot(target))
    m.on_after_step(2, stepped=True)
    assert not _same(t0, _snapshot(target))


def test_trainer_nonfinite_group_passes_stepped_false(tmp_path):
    """trainer 集成：非有限 loss 跳步的边界，方法状态冻结但时钟照常推进。"""
    from torch.utils.data import DataLoader, Dataset

    from ssltask.trainer import SSLTrainer

    class _ImgDataset(Dataset):
        def __init__(self, n):
            self.x = [torch.rand(1, 16, 32, 32) for _ in range(n)]

        def __len__(self):
            return len(self.x)

        def __getitem__(self, i):
            return {"image": self.x[i]}

    cfg = _cfg()
    cfg.train.epochs = 1
    cfg.train.use_ema = False
    cfg.train.use_amp = False
    cfg.train.grad_accum_steps = 2
    cfg.train.output_dir = str(tmp_path)
    cfg.sync()
    cfg.validate()
    ssl = _dino_ssl()
    validate_ssl(ssl, cfg)
    m = build_method(cfg, ssl, torch.device("cpu"))
    loader = DataLoader(_ImgDataset(4), batch_size=2)
    trainer = SSLTrainer(m, cfg, ssl, loader, torch.device("cpu"))

    orig = m.compute_loss

    def _nan_loss(batch):
        loss, logs = orig(batch)
        return loss * float("nan"), logs

    m.compute_loss = _nan_loss
    teacher0 = _snapshot(m.module.teacher)
    center0 = m.module.center.clone()
    trainer.fit()

    # 时钟照常推进；方法状态（EMA 教师 / center / pending）全部冻结。
    assert trainer.scheduler.current_step == trainer._total_opt_steps
    assert trainer._global_step == trainer._total_opt_steps
    assert m._step == trainer._total_opt_steps
    assert _same(teacher0, _snapshot(m.module.teacher))
    assert torch.equal(center0, m.module.center)
    assert m._pending_center_n == 0


def test_ibot_skipped_step_freezes_ibot_center():
    cfg = _cfg()
    ssl = _dino_ssl()
    ssl.method = "ibot"
    ssl.ibot_share_head = False
    ssl.ibot_out_dim = 48
    validate_ssl(ssl, cfg)
    m = build_method(cfg, ssl, torch.device("cpu"))
    m.configure_schedule(10)
    m.train()
    batch = {"image": torch.rand(2, 1, 16, 32, 32)}
    m.compute_loss(batch)
    assert m._pending_ibot_center_n >= 1
    c0 = m.module.ibot_center.clone()
    m.on_after_step(1, stepped=False)
    assert m._pending_ibot_center_n == 0
    assert torch.equal(c0, m.module.ibot_center)
    m.compute_loss(batch)
    m.on_after_step(2, stepped=True)
    assert not torch.equal(c0, m.module.ibot_center)
