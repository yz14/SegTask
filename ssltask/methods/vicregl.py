"""方案 B2：VICRegL-3D —— 稠密对应自监督（面向下游分割）。

VICRegL（Bardes et al., NeurIPS 2022）在 VICReg 的全局
invariance/variance/covariance 三项之上，增加**稠密局部项**：对两视图特征图上
按位置匹配（location-based，本实现）配对的位点，施加同套 VIC 损失。孪生结构
（两视图共享权重，无 EMA 教师、无负样本队列），实现简单且与现有代码栈契合。

对下游分割的价值：全局项学到判别性瓶颈表示，局部项显式对齐**逐位点密集特征**，
使 encoder 各尺度特征具备平移/裁剪一致的稠密语义 —— 正是密集预测（分割）所需。

分工遵循仓库约定：成对裁剪 + 坐标元数据在 data 层（:class:`PairedCropGenerator`
/ :func:`site_coords`），匹配与损失在本方法层。下游仅迁移 ``encoder.*``。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from segtask_v1.trainer.checkpoint import unwrap_compile

from ..data.multicrop import PairedCropGenerator, site_coords
from ..models.vicregl_modules import build_vicregl_net
from .base import SSLMethod


def _variance_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """方差 hinge：鼓励每维标准差 >= 1（抗表示坍缩）。z: (N, D)。"""
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std))


def _covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """协方差项：非对角元平方和 / D（去相关各维）。z: (N, D)。"""
    n, d = z.shape
    if n < 2:
        return z.new_zeros(())
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / (n - 1)
    off_diag = cov - torch.diag_embed(torch.diagonal(cov))
    return off_diag.pow(2).sum() / d


def _vic_terms(za: torch.Tensor, zb: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """一对嵌入 (N,D) 的 (invariance, variance, covariance) 三项。"""
    inv = F.mse_loss(za, zb)
    var = 0.5 * (_variance_loss(za) + _variance_loss(zb))
    cov = 0.5 * (_covariance_loss(za) + _covariance_loss(zb))
    return inv, var, cov


class VICRegLMethod(SSLMethod):
    """VICRegL-3D：全局 VIC + 位置匹配稠密 VIC（孪生、无 EMA/负样本）。"""

    name = "vicregl"

    def __init__(self, cfg, ssl, device: torch.device):
        super().__init__(cfg, ssl, device)
        self.spatial_dims = int(cfg.model.spatial_dims)
        self.sim_coeff = float(ssl.vicregl_sim_coeff)
        self.var_coeff = float(ssl.vicregl_var_coeff)
        self.cov_coeff = float(ssl.vicregl_cov_coeff)
        self.alpha = float(ssl.vicregl_alpha)          # 全局/局部加权（1=纯全局）
        self.num_matches = int(ssl.vicregl_num_matches)
        self.feature_level = int(ssl.vicregl_feature_level)

        patch = [int(s) for s in cfg.data.patch_size]
        model_spatial = patch if self.spatial_dims == 3 else patch[1:]
        self.paired = PairedCropGenerator(
            spatial_dims=self.spatial_dims,
            out_size=model_spatial,
            scale=tuple(ssl.vicregl_crop_scale),
            flip_prob=float(ssl.dino_flip_prob),
            intensity_scale=float(ssl.dino_intensity_scale),
            intensity_shift=float(ssl.dino_intensity_shift))

    def build_modules(self) -> nn.Module:
        return build_vicregl_net(
            self.cfg,
            proj_dim=int(self.ssl.vicregl_proj_dim),
            hidden_dim=int(self.ssl.vicregl_hidden_dim),
            dense_proj_dim=int(self.ssl.vicregl_dense_proj_dim),
            feature_level=int(self.ssl.vicregl_feature_level))

    # ---- location-based dense matching -----------------------------------
    def _matched_local(self, da: torch.Tensor, db: torch.Tensor,
                       ma: Dict[str, torch.Tensor], mb: Dict[str, torch.Tensor]
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """按原坐标最近邻，为每样本配 ``num_matches`` 对局部嵌入 (M,D)。

        da/db: (B, D, *sp) 两视图稠密嵌入；ma/mb: 对应裁剪元数据。返回沿所有
        样本堆叠的配对嵌入 ``(sum_b M_b, D)``，供稠密 VIC 项。
        """
        B, D = da.shape[0], da.shape[1]
        ca = site_coords(da.shape[2:], ma)             # (B, Na, D_ax)
        cb = site_coords(db.shape[2:], mb)
        fa = da.reshape(B, D, -1).transpose(1, 2)      # (B, Na, D)
        fb = db.reshape(B, D, -1).transpose(1, 2)
        na, nb = fa.shape[1], fb.shape[1]
        m = min(self.num_matches, na, nb)
        out_a: List[torch.Tensor] = []
        out_b: List[torch.Tensor] = []
        for b in range(B):
            dist = torch.cdist(ca[b], cb[b])           # (Na, Nb)
            j = dist.argmin(dim=1)                      # a→最近 b
            best = dist.gather(1, j.unsqueeze(1)).squeeze(1)  # (Na,)
            sel = torch.topk(best, m, largest=False).indices  # 最可靠 m 个 a
            out_a.append(fa[b, sel])
            out_b.append(fb[b, j[sel]])
        return torch.cat(out_a, dim=0), torch.cat(out_b, dim=0)

    def compute_loss(self, batch: Dict[str, torch.Tensor]
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
        (v1, v2), (m1, m2) = self.paired(batch["image"])
        g1, d1 = self.module(v1)
        g2, d2 = self.module(v2)
        g1, g2 = g1.float(), g2.float()

        g_inv, g_var, g_cov = _vic_terms(g1, g2)
        global_loss = (self.sim_coeff * g_inv + self.var_coeff * g_var
                       + self.cov_coeff * g_cov)

        la, lb = self._matched_local(d1.float(), d2.float(), m1, m2)
        l_inv, l_var, l_cov = _vic_terms(la, lb)
        local_loss = (self.sim_coeff * l_inv + self.var_coeff * l_var
                      + self.cov_coeff * l_cov)

        loss = self.alpha * global_loss + (1.0 - self.alpha) * local_loss
        return loss, {
            "vicregl_loss": float(loss.detach()),
            "global_loss": float(global_loss.detach()),
            "local_loss": float(local_loss.detach()),
            "inv": float(g_inv.detach()), "var": float(g_var.detach()),
            "cov": float(g_cov.detach())}

    def export_backbone_state_dict(self) -> Dict[str, torch.Tensor]:
        net = unwrap_compile(self.module)
        return {f"encoder.{k}": v.detach().cpu().clone()
                for k, v in net.encoder.state_dict().items()}


__all__ = ["VICRegLMethod"]
