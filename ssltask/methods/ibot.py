"""方案⑥ iBOT/DINOv2：在 ④DINO 全局自蒸馏之上叠加 **iBOT 掩码密集特征预测**。

将图像级自蒸馏（④DINO）与 patch 级掩码特征预测（iBOT）结合（DINOv2 = DINO + iBOT）：
在 DINO 全局损失之外，遮住学生输入的若干单元，对每个**被遮位点**要求学生投影后的密集
特征匹配教师（看完整输入）在该位点的密集特征——以共享原型上的逐位点 softmax 交叉熵实现，
同样配 centering + sharpening。目标是单一模型内同时获得全局判别性与密集特征（分类+分割双强）。

相对 ④ 仅多一个变量：``L = L_DINO(global) + λ·L_iBOT(masked dense)``。实现上 ``compute_loss``
先调父类得 ``L_DINO``（行为与 ④ 完全一致），再在新采样的 global 裁剪上加 iBOT 项——干净隔离
"iBOT 开/关"。CNN 不能丢 token，故掩码分支用 SimMIM 式 mask-token 稠密输入（``apply_mask_token``）。
密集投影头默认独立原型（DINOv2 默认；``ibot_share_head=True`` 时与全局头共享）；其教师侧随
④ 的 EMA 教师一并更新（共享时自动；独立时在 ``on_after_step`` 显式 EMA）。下游交接与 ④ 一致：
只导出 **教师** ``encoder.*``，两个投影头 / mask-token / center 用完即弃。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.masking import apply_mask_token, downsample_mask_to, make_unit_mask
from ..models.ibot_modules import build_ibot_head, dense_head_forward
from .base import SSLMethod  # noqa: F401  (保持与同级方法一致的导入面)
from .dino import DINOMethod, _DINOModule


class _IBOTModule(_DINOModule):
    """在 ④ 的 student/teacher/center 之上挂 iBOT 密集双头 + mask-token + iBOT center。"""

    def __init__(self, student: nn.Module, teacher: nn.Module, global_out_dim: int,
                 ibot_student_head: nn.Module, ibot_teacher_head: nn.Module,
                 in_channels: int, spatial_dims: int, ibot_out_dim: int,
                 own_heads: bool):
        super().__init__(student, teacher, global_out_dim)
        self.ibot_student_head = ibot_student_head
        self.ibot_teacher_head = ibot_teacher_head
        self.own_heads = bool(own_heads)
        if self.own_heads:                          # 独立 iBOT 头：教师侧冻结（靠 EMA 更新）
            for p in self.ibot_teacher_head.parameters():
                p.requires_grad_(False)
        # mask-token：可广播到 (1, C, *1) 的可学习向量。
        self.mask_token = nn.Parameter(
            torch.zeros(1, int(in_channels), *([1] * int(spatial_dims))))
        self.register_buffer("ibot_center", torch.zeros(1, int(ibot_out_dim)))


class IBOTMethod(DINOMethod):
    name = "ibot"

    def __init__(self, cfg, ssl, device: torch.device):
        super().__init__(cfg, ssl, device)
        self.ibot_weight = float(ssl.ibot_weight)
        self.ibot_mask_ratio = float(ssl.ibot_mask_ratio)
        self.ibot_mask_unit = int(ssl.ibot_mask_unit)
        self.ibot_level = int(ssl.ibot_feature_level)
        self.ibot_share_head = bool(ssl.ibot_share_head)

    # ---- modules ----------------------------------------------------------
    def build_modules(self) -> nn.Module:
        base = super().build_modules()                # _DINOModule(student, teacher)
        cfg, ssl = self.cfg, self.ssl
        global_out = int(ssl.dino_out_dim)
        feat_ch = int(cfg.model.encoder_channels[int(ssl.ibot_feature_level)])
        in_ch = int(cfg.model.in_channels)
        spatial = int(cfg.model.spatial_dims)

        if bool(ssl.ibot_share_head):
            # 与全局头共享原型：要求密集特征通道 == 全局头输入维（瓶颈），原型数 = dino_out_dim。
            head_in = int(cfg.model.encoder_channels[-1])
            if feat_ch != head_in:
                raise ValueError(
                    f"ibot_share_head=True requires ibot_feature_level to point at "
                    f"the bottleneck (channels {head_in}); level "
                    f"{ssl.ibot_feature_level} has {feat_ch} channels.")
            s_head, t_head = base.student.head, base.teacher.head
            ibot_out, own = global_out, False
        else:
            ibot_out = int(ssl.ibot_out_dim) or global_out
            kw = dict(in_dim=feat_ch, out_dim=ibot_out,
                      hidden_dim=int(ssl.dino_hidden_dim),
                      bottleneck_dim=int(ssl.dino_bottleneck_dim),
                      n_layers=int(ssl.dino_head_layers),
                      use_bn=bool(ssl.dino_head_use_bn))
            s_head = build_ibot_head(**kw)
            t_head = build_ibot_head(**kw)
            t_head.load_state_dict(s_head.state_dict())   # 教师初始 = 学生
            own = True

        return _IBOTModule(
            base.student, base.teacher, global_out, s_head, t_head,
            in_ch, spatial, ibot_out, own_heads=own)

    # ---- iBOT masked dense loss ------------------------------------------
    def _ibot_loss(self, image: torch.Tensor) -> torch.Tensor:
        """遮学生 global 裁剪输入，对被遮位点做学生/教师密集特征的交叉熵（特征空间）。"""
        global_crops: List[torch.Tensor] = self.multicrop(image)["global"]
        teacher_temp = self._teacher_temp()
        center = self.module.ibot_center.float()
        terms: List[torch.Tensor] = []
        teacher_logits: List[torch.Tensor] = []
        for g in global_crops:
            b = g.shape[0]
            mask = make_unit_mask(                       # (B,1,*spatial) 1=被遮
                b, g.shape[2:], self.ibot_mask_unit, self.ibot_mask_ratio,
                device=g.device)
            g_masked = apply_mask_token(g, mask, self.module.mask_token)
            s_feat = self.module.student.encoder(g_masked)[self.ibot_level].float()
            with torch.no_grad():
                t_feat = self.module.teacher.encoder(g)[self.ibot_level].float()
            s_logits = dense_head_forward(self.module.ibot_student_head, s_feat)
            with torch.no_grad():
                t_logits = dense_head_forward(self.module.ibot_teacher_head, t_feat)

            grid = s_feat.shape[2:]
            fmask = downsample_mask_to(mask, grid).flatten(2).squeeze(1)   # (B, N)
            t_prob = F.softmax(
                (t_logits - center) / teacher_temp, dim=-1).detach()       # (B,N,K)
            logp = F.log_softmax(s_logits / self.student_temp, dim=-1)
            ce = -(t_prob * logp).sum(dim=-1)                              # (B, N)
            terms.append((ce * fmask).sum() / fmask.sum().clamp_min(1.0))
            teacher_logits.append(t_logits.detach())

        self._update_ibot_center(teacher_logits)
        return torch.stack(terms).mean()

    @torch.no_grad()
    def _update_ibot_center(self, teacher_logits: List[torch.Tensor]) -> None:
        cat = torch.cat([t.reshape(-1, t.shape[-1]) for t in teacher_logits], dim=0)
        batch_center = cat.mean(dim=0, keepdim=True)
        self.module.ibot_center.mul_(self.center_momentum).add_(
            batch_center.to(self.module.ibot_center.dtype),
            alpha=1.0 - self.center_momentum)

    # ---- loss -------------------------------------------------------------
    def compute_loss(self, batch: Dict[str, torch.Tensor]
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
        dino_loss, logs = super().compute_loss(batch)
        ibot = self._ibot_loss(batch["image"])
        loss = dino_loss + self.ibot_weight * ibot
        logs["ibot_loss"] = float(ibot.detach())
        logs["ibot_weight"] = self.ibot_weight
        return loss, logs

    # ---- EMA teacher update (encoder/global head via parent; iBOT head here) ---
    def on_after_step(self, global_step: int) -> None:
        super().on_after_step(global_step)            # EMA 教师 encoder + 全局头 + self._step
        if self.module.own_heads:                     # 独立 iBOT 头需单独 EMA
            m = self._momentum()
            with torch.no_grad():
                for ps, pt in zip(self.module.ibot_student_head.parameters(),
                                  self.module.ibot_teacher_head.parameters()):
                    pt.mul_(m).add_(ps.detach(), alpha=1.0 - m)
                for bs, bt in zip(self.module.ibot_student_head.buffers(),
                                  self.module.ibot_teacher_head.buffers()):
                    bt.copy_(bs)


__all__ = ["IBOTMethod"]
