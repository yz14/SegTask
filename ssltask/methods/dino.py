"""方案④ DINO-3D：多裁剪 + EMA 教师自蒸馏（image-only，无标注）。

学生网络 ``student`` 看全部裁剪（global + local），EMA 教师 ``teacher`` 只看 global
裁剪；目标是让学生在每个裁剪上的软分配匹配教师对 global 裁剪的软分配（跨裁剪一致性
→ 学到视图不变的语义表征）。防坍缩靠两点：教师输出做 **centering**（减去 batch 级
EMA 中心）+ **sharpening**（低温 softmax），学生用较高温。教师权重由学生参数的
cosine 动量 EMA 更新（``on_after_step``）。

与重建/掩码族（②①③）正交：DINO 不预训练解码器，下游 ``train.pretrain``
（strict=False）仅命中 ``encoder.*``。其 EMA 教师 / 多裁剪 / 投影头 / center-sharpen /
温度·动量调度等机制为 ⑤⑥⑧ 与对比基线复用的公共基础设施。
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from segtask_v1.trainer.dist_utils import (
    all_reduce_sum_, get_world_size, is_dist_avail_and_initialized)

from ..data.multicrop import MultiCropGenerator
from ..models.dino_modules import build_dino_net
from .base import SSLMethod


def _global_batch_mean(batch_center: torch.Tensor) -> torch.Tensor:
    """把本 rank 的 batch 均值归约为全局均值（非分布式时原样返回）。

    手动 DDP 只在初始化时广播一次 buffer，之后各 rank 独立更新；center 若只用
    本地 batch 会逐步发散。各 rank 等长 batch（DistributedSampler drop_last
    保证），故 all-reduce 均值即全局 batch 均值，center 在各副本间保持一致。
    """
    if not (is_dist_avail_and_initialized() and get_world_size() > 1):
        return batch_center
    batch_center = batch_center.contiguous()
    all_reduce_sum_(batch_center)
    return batch_center / float(get_world_size())


class _DINOModule(nn.Module):
    """承载 student / teacher（冻结）与 centering 缓冲的容器（统一搬设备 / 存取）。"""

    def __init__(self, student: nn.Module, teacher: nn.Module, out_dim: int):
        super().__init__()
        self.student = student
        self.teacher = teacher
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.register_buffer("center", torch.zeros(1, int(out_dim)))


class DINOMethod(SSLMethod):
    name = "dino"

    def __init__(self, cfg, ssl, device: torch.device):
        super().__init__(cfg, ssl, device)
        self.spatial_dims = int(cfg.model.spatial_dims)
        self.student_temp = float(ssl.dino_student_temp)
        self.teacher_temp_final = float(ssl.dino_teacher_temp)
        self.teacher_temp_warmup = float(ssl.dino_teacher_temp_warmup)
        self.center_momentum = float(ssl.dino_center_momentum)
        self.momentum_base = float(ssl.dino_momentum_base)
        self.momentum_final = float(ssl.dino_momentum_final)
        self.warmup_temp_frac = float(ssl.dino_warmup_teacher_temp_frac)

        # 多裁剪输出尺寸：未显式给定则由 patch_size 推导（global=patch，local≈半）。
        patch = [int(s) for s in cfg.data.patch_size]                 # [D,H,W]
        model_spatial = patch if self.spatial_dims == 3 else patch[1:]
        global_size = (list(ssl.dino_global_size) or model_spatial)
        local_size = (list(ssl.dino_local_size)
                      or [max(int(s) // 2, 8) for s in model_spatial])
        self.multicrop = MultiCropGenerator(
            spatial_dims    = self.spatial_dims,
            global_size     = global_size,
            local_size      = local_size,
            n_global        = int(ssl.dino_global_crops),
            n_local         = int(ssl.dino_local_crops),
            global_scale    = tuple(ssl.dino_global_scale),
            local_scale     = tuple(ssl.dino_local_scale),
            flip_prob       = float(ssl.dino_flip_prob),
            intensity_scale = float(ssl.dino_intensity_scale),
            intensity_shift = float(ssl.dino_intensity_shift))
        self.n_global = int(ssl.dino_global_crops)

        # 调度（configure_schedule 由 SSLTrainer 在训练前调用以填实总步数）。
        self._step = 0
        self.total_steps = 1
        self.warmup_temp_steps = 1

    # ---- modules ----------------------------------------------------------
    def build_modules(self) -> nn.Module:
        ssl = self.ssl
        out_dim = int(ssl.dino_out_dim)
        kw = dict(out_dim=out_dim, hidden_dim=int(ssl.dino_hidden_dim),
                  bottleneck_dim=int(ssl.dino_bottleneck_dim),
                  n_layers=int(ssl.dino_head_layers),
                  use_bn=bool(ssl.dino_head_use_bn))
        student = build_dino_net(self.cfg, **kw)
        teacher = build_dino_net(self.cfg, **kw)
        teacher.load_state_dict(student.state_dict())     # 教师初始 = 学生
        return _DINOModule(student, teacher, out_dim)

    # ---- schedules --------------------------------------------------------
    def configure_schedule(self, total_steps: int) -> None:
        self.total_steps = max(int(total_steps), 1)
        self.warmup_temp_steps = max(
            int(self.warmup_temp_frac * self.total_steps), 1)

    def _teacher_temp(self) -> float:
        """teacher 温度从 warmup 起点线性升到 final（前 warmup_temp_steps 步）。"""
        if self._step >= self.warmup_temp_steps:
            return self.teacher_temp_final
        alpha = self._step / max(self.warmup_temp_steps, 1)
        return self.teacher_temp_warmup + alpha * (
            self.teacher_temp_final - self.teacher_temp_warmup)

    def _momentum(self) -> float:
        """EMA 教师动量：cosine 从 base 升到 final。"""
        progress = min(self._step / self.total_steps, 1.0)
        return self.momentum_final - (self.momentum_final - self.momentum_base) * (
            math.cos(math.pi * progress) + 1.0) / 2.0

    # ---- loss -------------------------------------------------------------
    def compute_loss(self, batch: Dict[str, torch.Tensor]
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
        crops = self.multicrop(batch["image"])
        global_crops: List[torch.Tensor] = crops["global"]
        all_crops = global_crops + crops["local"]

        student_out = [self.module.student(c).float() for c in all_crops]
        with torch.no_grad():
            teacher_out = [self.module.teacher(c).float() for c in global_crops]

        teacher_temp = self._teacher_temp()
        center = self.module.center.float()
        teacher_probs = [
            F.softmax((t - center) / teacher_temp, dim=-1).detach()
            for t in teacher_out]

        total = batch["image"].new_zeros(())
        n_pairs = 0
        for ti, tp in enumerate(teacher_probs):
            for si, s in enumerate(student_out):
                if si == ti:                      # 跳过同一 global 裁剪自配
                    continue
                logp = F.log_softmax(s / self.student_temp, dim=-1)
                total = total + (-(tp * logp).sum(dim=-1).mean())
                n_pairs += 1
        loss = total / max(n_pairs, 1)

        self._update_center(teacher_out)
        return loss, {"dino_loss": float(loss.detach()),
                      "teacher_temp": teacher_temp,
                      "ema_momentum": self._momentum()}

    @torch.no_grad()
    def _update_center(self, teacher_out: List[torch.Tensor]) -> None:
        batch_center = _global_batch_mean(
            torch.cat(teacher_out, dim=0).mean(dim=0, keepdim=True))
        self.module.center.mul_(self.center_momentum).add_(
            batch_center.to(self.module.center.dtype),
            alpha=1.0 - self.center_momentum)

    # ---- EMA teacher update ----------------------------------------------
    def on_resume(self, global_step: int) -> None:
        self._step = int(global_step)

    def on_after_step(self, global_step: int) -> None:
        self._step = int(global_step)
        m = self._momentum()
        with torch.no_grad():
            for ps, pt in zip(self.module.student.parameters(),
                              self.module.teacher.parameters()):
                pt.mul_(m).add_(ps.detach(), alpha=1.0 - m)
            for bs, bt in zip(self.module.student.buffers(),
                              self.module.teacher.buffers()):
                bt.copy_(bs)

    # ---- export -----------------------------------------------------------
    def export_backbone_state_dict(self) -> Dict[str, torch.Tensor]:
        """导出 **教师** encoder（DINO 惯例：教师表征更稳），键命名为 ``encoder.*``。"""
        from segtask_v1.trainer.checkpoint import unwrap_compile
        teacher = unwrap_compile(self.module).teacher
        enc_sd = teacher.encoder.state_dict()
        return {f"encoder.{k}": v.detach().cpu().clone()
                for k, v in enc_sd.items()}


__all__ = ["DINOMethod"]
