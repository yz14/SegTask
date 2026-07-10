"""方案 B1：MoCo-3D（query encoder/projector + EMA key encoder + 队列）。"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from segtask_v1.models.factory import build_model
from segtask_v1.trainer.checkpoint import unwrap_compile
from segtask_v1.trainer.dist_utils import (
    get_world_size, is_dist_avail_and_initialized)

from ..data.multicrop import MultiCropGenerator
from ..models.dino_modules import DINOHead
from .base import SSLMethod


def _pool_feat(feats):
    return feats[-1].mean(dim=tuple(range(2, feats[-1].ndim)))


def _concat_all_gather(t: torch.Tensor) -> torch.Tensor:
    """跨 rank 收集张量并沿 batch 维拼接（非分布式时原样返回）。

    各 rank 等长（DistributedSampler drop_last 保证）；key 来自 no_grad 分支，
    无需梯度穿过 gather。所有 rank 得到同一拼接结果（按 rank 序），使隔离的
    queue/queue_ptr buffer 在各副本间保持逐步一致，同时把负样本扩充到全局 batch。
    """
    if not (is_dist_avail_and_initialized() and get_world_size() > 1):
        return t
    out = [torch.empty_like(t) for _ in range(get_world_size())]
    dist.all_gather(out, t.contiguous())
    return torch.cat(out, dim=0)


class _ProjectedEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, projector: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = _pool_feat(self.encoder(x))
        z = self.projector(z)
        return F.normalize(z, dim=-1)


class _MoCoModule(nn.Module):
    def __init__(self, query: _ProjectedEncoder, key: _ProjectedEncoder,
                 queue_size: int, proj_dim: int):
        super().__init__()
        self.query = query
        self.key = key
        for p in self.key.parameters():
            p.requires_grad_(False)
        self.register_buffer("queue", F.normalize(torch.randn(int(proj_dim), int(queue_size)), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


class MoCoMethod(SSLMethod):
    name = "moco"

    def __init__(self, cfg, ssl, device: torch.device):
        super().__init__(cfg, ssl, device)
        self.spatial_dims = int(cfg.model.spatial_dims)
        self.proj_dim = int(ssl.moco_proj_dim)
        self.queue_size = int(ssl.moco_queue_size)
        self.temperature = float(ssl.moco_temperature)
        self.momentum_base = float(ssl.moco_momentum_base)
        self.momentum_final = float(ssl.moco_momentum_final)
        self.dino_hidden_dim = int(ssl.dino_hidden_dim)
        self._step = 0
        self.total_steps = 1

        patch = [int(s) for s in cfg.data.patch_size]
        model_spatial = patch if self.spatial_dims == 3 else patch[1:]
        self.multicrop = MultiCropGenerator(
            spatial_dims=self.spatial_dims,
            global_size=model_spatial,
            local_size=model_spatial,
            n_global=2,
            n_local=0,
            global_scale=tuple(ssl.dino_global_scale),
            local_scale=tuple(ssl.dino_local_scale),
            flip_prob=float(ssl.dino_flip_prob),
            intensity_scale=float(ssl.dino_intensity_scale),
            intensity_shift=float(ssl.dino_intensity_shift),
        )

    def build_modules(self) -> nn.Module:
        proj_dim = int(self.ssl.moco_proj_dim)
        hidden_dim = int(self.ssl.dino_hidden_dim)
        q_enc = build_model(self.cfg).encoder
        k_enc = build_model(self.cfg).encoder
        k_enc.load_state_dict(q_enc.state_dict())   # key 初始 = query
        q_proj = DINOHead(
            in_dim=int(self.cfg.model.encoder_channels[-1]),
            out_dim=proj_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=proj_dim,
            n_layers=2,
            use_bn=False,
        )
        k_proj = DINOHead(
            in_dim=int(self.cfg.model.encoder_channels[-1]),
            out_dim=proj_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=proj_dim,
            n_layers=2,
            use_bn=False,
        )
        k_proj.load_state_dict(q_proj.state_dict())
        query = _ProjectedEncoder(q_enc, q_proj)
        key = _ProjectedEncoder(k_enc, k_proj)
        return _MoCoModule(query, key, int(self.ssl.moco_queue_size), proj_dim)

    def configure_schedule(self, total_steps: int) -> None:
        self.total_steps = max(int(total_steps), 1)

    def _momentum(self) -> float:
        progress = min(self._step / self.total_steps, 1.0)
        return self.momentum_final - (self.momentum_final - self.momentum_base) * (
            math.cos(math.pi * progress) + 1.0) / 2.0

    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        keys = F.normalize(keys.detach(), dim=-1)
        keys = keys.reshape(-1, keys.shape[-1])
        keys = _concat_all_gather(keys)               # 全局 key：各 rank 同步入队
        k = keys.shape[0]
        queue = self.module.queue
        ptr = int(self.module.queue_ptr.item())
        if k >= queue.shape[1]:
            keys = keys[-queue.shape[1]:]
            k = keys.shape[0]
        end = ptr + k
        if end <= queue.shape[1]:
            queue[:, ptr:end] = keys.T
        else:
            first = queue.shape[1] - ptr
            queue[:, ptr:] = keys[:first].T
            queue[:, :end % queue.shape[1]] = keys[first:].T
        ptr = (ptr + k) % queue.shape[1]
        self.module.queue_ptr[0] = ptr

    def compute_loss(self, batch: Dict[str, torch.Tensor]
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
        crops = self.multicrop(batch["image"])["global"]
        q1 = self.module.query(crops[0])
        q2 = self.module.query(crops[1])
        with torch.no_grad():
            k1 = self.module.key(crops[0])
            k2 = self.module.key(crops[1])
        queue = self.module.queue.detach().clone().T  # (K, D)
        l1_pos = torch.einsum("bd,bd->b", q1, k2).unsqueeze(1)
        l2_pos = torch.einsum("bd,bd->b", q2, k1).unsqueeze(1)
        l1_neg = q1 @ queue.T
        l2_neg = q2 @ queue.T
        logits1 = torch.cat([l1_pos, l1_neg], dim=1) / self.temperature
        logits2 = torch.cat([l2_pos, l2_neg], dim=1) / self.temperature
        labels = torch.zeros(logits1.shape[0], dtype=torch.long, device=logits1.device)
        loss = 0.5 * (F.cross_entropy(logits1, labels) + F.cross_entropy(logits2, labels))
        self._dequeue_and_enqueue(torch.cat([k1, k2], dim=0))
        return loss, {"moco_loss": float(loss.detach()), "queue_ptr": float(self.module.queue_ptr.item())}

    def on_resume(self, global_step: int) -> None:
        self._step = int(global_step)

    def on_after_step(self, global_step: int) -> None:
        self._step = int(global_step)
        m = self._momentum()
        with torch.no_grad():
            for pq, pk in zip(self.module.query.parameters(), self.module.key.parameters()):
                pk.mul_(m).add_(pq.detach(), alpha=1.0 - m)
            for bq, bk in zip(self.module.query.buffers(), self.module.key.buffers()):
                bk.copy_(bq)

    def export_backbone_state_dict(self) -> Dict[str, torch.Tensor]:
        query = unwrap_compile(self.module).query
        return {f"encoder.{k}": v.detach().cpu().clone()
                for k, v in query.encoder.state_dict().items()}


__all__ = ["MoCoMethod"]
