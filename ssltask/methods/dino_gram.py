"""方案⑤ DINO+Gram：在 ④DINO 自蒸馏之上加 **Gram anchoring**（DINOv3）。

DINO④ 的图像级自蒸馏在长 schedule 下会让**密集**（逐位点）特征退化——全局表征仍好，
但局部特征图变糊，伤分割等密集下游。Gram anchoring 的修法：额外约束学生的密集特征
**Gram 矩阵**（位点-位点的余弦相似度矩阵）逼近一个"Gram 教师"——它是较早期、密集特征
尚佳的 EMA 教师**快照**，周期性刷新。Gram 只看相对相似度结构、不看绝对尺度，故能在不破坏
DINO 语义聚类的前提下把局部几何"拉回"清晰。

相对 ④ 仅多一个变量：``L = L_DINO + λ·L_gram``。λ 在训练进度达到 ``dino_gram_start_frac``
前为 0（先让密集特征成形），之后启用。Gram 仅在 global 裁剪上计算（保证学生/Gram 教师
位点数一致）。下游交接与 ④ 完全一致：只导出 **教师** ``encoder.*``（见 ``DINOMethod``）。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.dino_modules import build_dino_net
from .base import SSLMethod  # noqa: F401  (保持与同级方法一致的导入面)
from .dino import DINOMethod, _DINOModule


class _DINOGramModule(_DINOModule):
    """在 ④ 的 student/teacher/center 之上再挂一个冻结的 ``gram_teacher`` 快照。"""

    def __init__(self, student: nn.Module, teacher: nn.Module,
                 gram_teacher: nn.Module, out_dim: int):
        super().__init__(student, teacher, out_dim)
        self.gram_teacher = gram_teacher
        for p in self.gram_teacher.parameters():
            p.requires_grad_(False)


class DINOGramMethod(DINOMethod):
    name = "dino_gram"

    def __init__(self, cfg, ssl, device: torch.device):
        super().__init__(cfg, ssl, device)
        self.gram_weight = float(ssl.dino_gram_weight)
        self.gram_start_frac = float(ssl.dino_gram_start_frac)
        self.gram_refresh_steps = max(int(ssl.dino_gram_refresh_steps), 1)
        self.gram_level = int(ssl.dino_gram_feature_level)

    # ---- modules ----------------------------------------------------------
    def build_modules(self) -> nn.Module:
        base = super().build_modules()                # _DINOModule(student, teacher)
        ssl = self.ssl
        # 与 student/teacher 同构（weight_norm 头不支持 deepcopy，故重建后拷权重）。
        gram_teacher = build_dino_net(
            self.cfg, out_dim=int(ssl.dino_out_dim),
            hidden_dim=int(ssl.dino_hidden_dim),
            bottleneck_dim=int(ssl.dino_bottleneck_dim),
            n_layers=int(ssl.dino_head_layers),
            use_bn=bool(ssl.dino_head_use_bn))
        gram_teacher.load_state_dict(base.teacher.state_dict())  # 初始快照 = 教师
        return _DINOGramModule(
            base.student, base.teacher, gram_teacher, int(ssl.dino_out_dim))

    # ---- gram weight schedule --------------------------------------------
    def _gram_weight(self) -> float:
        """进度 < start_frac 时关闭 Gram（只跑纯 DINO），之后取 ``dino_gram_weight``。"""
        progress = min(self._step / self.total_steps, 1.0)
        return self.gram_weight if progress >= self.gram_start_frac else 0.0

    # ---- gram loss --------------------------------------------------------
    @staticmethod
    def _gram_matrix(feat: torch.Tensor) -> torch.Tensor:
        """密集特征 (B,C,*spatial) → 位点-位点余弦相似度 Gram 矩阵 (B,N,N)。"""
        x = feat.flatten(2).transpose(1, 2)           # (B, N, C)
        x = F.normalize(x, dim=-1, p=2)               # 每位点向量 L2 归一化
        return x @ x.transpose(1, 2)                  # (B, N, N)

    def _gram_loss(self, image: torch.Tensor) -> torch.Tensor:
        """学生 vs Gram 教师在 global 裁剪密集特征上的 Gram 矩阵 Frobenius² 距离。"""
        global_crops: List[torch.Tensor] = self.multicrop(image)["global"]
        terms = []
        for g in global_crops:
            s_feat = self.module.student.encoder(g)[self.gram_level].float()
            with torch.no_grad():
                t_feat = self.module.gram_teacher.encoder(g)[self.gram_level].float()
            gs = self._gram_matrix(s_feat)
            gt = self._gram_matrix(t_feat).detach()
            # mean over (B,N,N) == 每样本 Frobenius² / N²，再对 batch 取均值。
            terms.append((gs - gt).pow(2).mean())
        return torch.stack(terms).mean()

    # ---- loss -------------------------------------------------------------
    def compute_loss(self, batch: Dict[str, torch.Tensor]
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
        dino_loss, logs = super().compute_loss(batch)
        lam = self._gram_weight()
        if lam > 0.0:
            gram = self._gram_loss(batch["image"])
            loss = dino_loss + lam * gram
            logs["gram_loss"] = float(gram.detach())
        else:
            loss = dino_loss
            logs["gram_loss"] = 0.0
        logs["gram_weight"] = lam
        return loss, logs

    # ---- EMA teacher update + periodic gram-teacher refresh ---------------
    def on_after_step(self, global_step: int) -> None:
        super().on_after_step(global_step)            # 更新 EMA 教师 + self._step
        if int(global_step) % self.gram_refresh_steps == 0:
            self._refresh_gram_teacher()

    @torch.no_grad()
    def _refresh_gram_teacher(self) -> None:
        """用当前 EMA 教师整份覆盖 Gram 教师快照。"""
        for pg, pt in zip(self.module.gram_teacher.parameters(),
                          self.module.teacher.parameters()):
            pg.copy_(pt)
        for bg, bt in zip(self.module.gram_teacher.buffers(),
                          self.module.teacher.buffers()):
            bg.copy_(bt)


__all__ = ["DINOGramMethod"]
