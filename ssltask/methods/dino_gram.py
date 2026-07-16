"""方案⑤ DINO+Gram：在 ④DINO 自蒸馏之上加 **Gram anchoring**（DINOv3）。

DINO④ 的图像级自蒸馏在长 schedule 下会让**密集**（逐位点）特征退化——全局表征仍好，
但局部特征图变糊，伤分割等密集下游。Gram anchoring 的修法：额外约束学生的密集特征
**Gram 矩阵**（位点-位点的余弦相似度矩阵）逼近一个"Gram 教师"——它是较早期、密集特征
尚佳的 EMA 教师**快照**，周期性刷新。Gram 只看相对相似度结构、不看绝对尺度，故能在不破坏
DINO 语义聚类的前提下把局部几何"拉回"清晰。

相对 ④ 仅多一个变量：``L = L_DINO + λ·L_gram``。staged recipe：
1. 进度 < ``dino_gram_start_frac``：λ=0，纯 DINO（密集特征成形期），Gram 教师不刷新；
2. λ 首次生效的时刻，把**当时**的 EMA 教师快照为 Gram 教师（锚定"密集特征尚佳"的
   早期教师），之后保持冻结；
3. 此后每 ``dino_gram_refresh_steps`` 个优化步（以锚定时刻为原点）才整份刷新一次快照。

锚定时刻以 buffer ``gram_anchor_step`` 随 method state 入 checkpoint：断点续训后 Gram
教师快照与刷新节律逐位延续（旧 ckpt 无该键时回退为未锚定，首个 λ>0 步重新锚定）。
Gram 仅在 global 裁剪上计算（保证学生/Gram 教师位点数一致）。下游交接与 ④ 完全一致：
只导出 **教师** ``encoder.*``（见 ``DINOMethod``）。
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
        self.gram_teacher.eval()
        # λ 首次生效时的优化步（Gram 教师锚定时刻）；-1 = 尚未启用。入 buffer 使
        # resume 后不重新锚定、刷新节律延续（快照权重本身随 submodule 持久化）。
        self.register_buffer("gram_anchor_step",
                             torch.tensor(-1, dtype=torch.long))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # 旧 ckpt 无 gram_anchor_step：保持默认 -1（未锚定），不报 missing key。
        key = prefix + "gram_anchor_step"
        if key not in state_dict:
            state_dict[key] = self.gram_anchor_step.detach().clone()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def train(self, mode: bool = True):
        super().train(mode)          # 已保持 teacher.eval()
        self.gram_teacher.eval()
        return self


class DINOGramMethod(DINOMethod):
    name = "dino_gram"

    def __init__(self, cfg, ssl, device: torch.device):
        super().__init__(cfg, ssl, device)
        self.gram_weight = float(ssl.dino_gram_weight)
        self.gram_start_frac = float(ssl.dino_gram_start_frac)
        self.gram_refresh_steps = max(int(ssl.dino_gram_refresh_steps), 1)
        self.gram_level = int(ssl.dino_gram_feature_level)
        # λ 首次生效时的优化步（Gram 教师锚定时刻）的进程内镜像；持久化真相源为
        # module buffer ``gram_anchor_step``（见 on_resume / compute_loss）。
        self._gram_anchor_step = -1

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

    @staticmethod
    def _gram_sq_dist(s_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        """``mean((G_s - G_t)**2)``（即逐样本 Frobenius²/N² 的 batch 均值）。

        N > C 时不物化 (B,N,N) Gram，而用恒等式
        ``||XXᵀ - YYᵀ||²_F = ||XᵀX||²_F - 2||XᵀY||²_F + ||YᵀY||²_F``
        在 (C,C) 空间计算（数值精确等价，显存 O(N²)→O(C²)），使浅层高分辨率
        特征级（N 可达数万）也可安全作 Gram anchoring。
        """
        xs = F.normalize(s_feat.flatten(2).transpose(1, 2), dim=-1, p=2)  # (B,N,C)
        xt = F.normalize(t_feat.flatten(2).transpose(1, 2), dim=-1, p=2)
        _, n, c = xs.shape
        if n <= c:                                     # 小 N：直接 N×N 更便宜
            gs = xs @ xs.transpose(1, 2)
            gt = xt @ xt.transpose(1, 2)
            return (gs - gt).pow(2).mean()
        ss = xs.transpose(1, 2) @ xs                   # (B, C, C)
        st = xs.transpose(1, 2) @ xt
        tt = xt.transpose(1, 2) @ xt
        frob_sq = (ss.pow(2).sum(dim=(1, 2))
                   - 2.0 * st.pow(2).sum(dim=(1, 2))
                   + tt.pow(2).sum(dim=(1, 2)))        # (B,) 逐样本 ||G_s-G_t||²_F
        return frob_sq.clamp_min(0.0).mean() / float(n * n)

    def _gram_loss(self, global_crops: List[torch.Tensor]) -> torch.Tensor:
        """学生 vs Gram 教师在 global 裁剪密集特征上的 Gram 矩阵 Frobenius² 距离。

        复用 DINO 主损失的 global 裁剪（同一批视图），不再额外多裁剪。"""
        terms = []
        for g in global_crops:
            s_feat = self.module.student.encoder(g)[self.gram_level].float()
            with torch.no_grad():
                t_feat = self.module.gram_teacher.encoder(g)[self.gram_level].float()
            terms.append(self._gram_sq_dist(s_feat, t_feat.detach()))
        return torch.stack(terms).mean()

    # ---- loss -------------------------------------------------------------
    def compute_loss(self, batch: Dict[str, torch.Tensor]
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
        dino_loss, logs = super().compute_loss(batch)
        crops = self._cached_global_crops or []
        self._cached_global_crops = None
        lam = self._gram_weight()
        if lam > 0.0:
            if self._gram_anchor_step < 0:      # λ 首次生效：锚定当前 EMA 教师
                self._refresh_gram_teacher()
                self._gram_anchor_step = int(self._step)
                with torch.no_grad():
                    self.module.gram_anchor_step.fill_(self._gram_anchor_step)
            gram = self._gram_loss(crops)
            loss = dino_loss + lam * gram
            logs["gram_loss"] = gram.detach()
        else:
            loss = dino_loss
            logs["gram_loss"] = 0.0
        logs["gram_weight"] = lam
        return loss, logs

    # ---- resume ------------------------------------------------------------
    def on_resume(self, global_step: int) -> None:
        super().on_resume(global_step)
        # 从持久化 buffer 恢复锚定时刻（一次 .item() 同步）：已锚定的运行不会在
        # resume 后用当前教师覆盖 Gram 快照，刷新节律以原锚定时刻延续。
        self._gram_anchor_step = int(self.module.gram_anchor_step.item())

    # ---- EMA teacher update + periodic gram-teacher refresh ---------------
    def on_after_step(self, global_step: int, stepped: bool = True) -> None:
        super().on_after_step(global_step, stepped)   # 更新 EMA 教师 + self._step
        if not stepped or self._gram_anchor_step < 0:
            return                                    # 跳步或 Gram 尚未启用：不刷新
        elapsed = int(global_step) - self._gram_anchor_step
        if elapsed > 0 and elapsed % self.gram_refresh_steps == 0:
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
