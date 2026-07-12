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

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from segtask_v1.trainer.checkpoint import unwrap_compile

from ..data.multicrop import PairedCropGenerator, site_coords
from ..models.vicregl_modules import build_vicregl_net
from .base import SSLMethod


def _variance_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """方差 hinge：鼓励每维标准差 >= 1（抗表示坍缩）。z: (N, D)。

    总体方差（unbiased=False）：N=1 时无分母 0 的 NaN；但单样本的方差无
    统计意义（恒为 0，hinge 会给出虚假的满额惩罚），直接返回 0。"""
    if z.shape[0] < 2:
        return z.new_zeros(())
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
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
        self.feature_matches = int(ssl.vicregl_feature_matches)
        self.match_radius = float(ssl.vicregl_match_radius)
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

    # ---- dense matching ----------------------------------------------------
    @staticmethod
    def _dir_pairs(f_src: torch.Tensor, f_dst: torch.Tensor,
                   dist: torch.Tensor, m: int,
                   max_dist: Optional[torch.Tensor]
                   ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """单向最近邻配对：src 位点 → dst 最近位点，保留最可靠 top-m 对。

        ``max_dist`` 非 None 时丢弃超出半径的对（overlap-aware：两裁剪框不
        重叠时不强造正样本）；无有效对时返回 None。
        """
        best, j = dist.min(dim=1)                       # (Ns,)
        if max_dist is not None:
            idx = (best <= max_dist).nonzero(as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                return None
        else:
            idx = torch.arange(best.shape[0], device=best.device)
        k = min(m, int(idx.numel()))
        sel = idx[torch.topk(best[idx], k, largest=False).indices]
        return f_src[sel], f_dst[j[sel]]

    def _matched_local(self, da: torch.Tensor, db: torch.Tensor,
                       ma: Dict[str, torch.Tensor], mb: Dict[str, torch.Tensor]
                       ) -> Tuple[Optional[torch.Tensor],
                                  Optional[torch.Tensor], int, int]:
        """双向 location-based + 双向 feature-based 配对（VICRegL 语义）。

        da/db: (B, D, *sp) 两视图稠密嵌入；ma/mb: 对应裁剪元数据。
        位置配对只在两裁剪框重叠区内生效（距离 ≤ match_radius × 目标视图位点
        间距）；特征配对无空间限制（匹配语义相似的远位点）。返回
        ``(za, zb, n_loc, n_feat)``；整批无有效对时 za/zb 为 None。
        """
        B, D = da.shape[0], da.shape[1]
        ca = site_coords(da.shape[2:], ma)             # (B, Na, D_ax)
        cb = site_coords(db.shape[2:], mb)
        fa = da.reshape(B, D, -1).transpose(1, 2)      # (B, Na, D)
        fb = db.reshape(B, D, -1).transpose(1, 2)
        na, nb = fa.shape[1], fb.shape[1]
        m_loc = min(self.num_matches, na, nb)
        m_feat = min(self.feature_matches, na, nb)
        # 目标视图位点间距（原体素）：裁剪框尺寸 / 特征图轴长，取轴向最大。
        sp_a = torch.tensor([float(s) for s in da.shape[2:]],
                            device=ma["size"].device)
        sp_b = torch.tensor([float(s) for s in db.shape[2:]],
                            device=mb["size"].device)
        pitch_a = (ma["size"] / sp_a).max(dim=1).values          # (B,)
        pitch_b = (mb["size"] / sp_b).max(dim=1).values
        out_a: List[torch.Tensor] = []
        out_b: List[torch.Tensor] = []
        n_loc = n_feat = 0
        for b in range(B):
            dist = torch.cdist(ca[b], cb[b])           # (Na, Nb) 原体素距离
            for f_s, f_d, d, pitch in (
                    (fa[b], fb[b], dist, pitch_b[b]),
                    (fb[b], fa[b], dist.T, pitch_a[b])):
                max_d = (self.match_radius * pitch
                         if self.match_radius > 0 else None)
                pair = self._dir_pairs(f_s, f_d, d, m_loc, max_d)
                if pair is not None:
                    out_a.append(pair[0])
                    out_b.append(pair[1])
                    n_loc += pair[0].shape[0]
            if m_feat > 0:
                fdist = torch.cdist(fa[b], fb[b])      # 嵌入空间距离
                for f_s, f_d, d in ((fa[b], fb[b], fdist),
                                    (fb[b], fa[b], fdist.T)):
                    pair = self._dir_pairs(f_s, f_d, d, m_feat, None)
                    if pair is not None:
                        out_a.append(pair[0])
                        out_b.append(pair[1])
                        n_feat += pair[0].shape[0]
        if not out_a:
            return None, None, 0, 0
        return torch.cat(out_a, dim=0), torch.cat(out_b, dim=0), n_loc, n_feat

    def compute_loss(self, batch: Dict[str, torch.Tensor]
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
        (v1, v2), (m1, m2) = self.paired(batch["image"])
        g1, d1 = self.module(v1)
        g2, d2 = self.module(v2)
        g1, g2 = g1.float(), g2.float()

        g_inv, g_var, g_cov = _vic_terms(g1, g2)
        global_loss = (self.sim_coeff * g_inv + self.var_coeff * g_var
                       + self.cov_coeff * g_cov)

        la, lb, n_loc, n_feat = self._matched_local(
            d1.float(), d2.float(), m1, m2)
        if la is None:                                  # 两视图无任何有效配对
            local_loss = g1.new_zeros(())
        else:
            l_inv, l_var, l_cov = _vic_terms(la, lb)
            local_loss = (self.sim_coeff * l_inv + self.var_coeff * l_var
                          + self.cov_coeff * l_cov)

        loss = self.alpha * global_loss + (1.0 - self.alpha) * local_loss
        return loss, {
            "vicregl_loss": float(loss.detach()),
            "global_loss": float(global_loss.detach()),
            "local_loss": float(local_loss.detach()),
            "n_loc_matches": float(n_loc), "n_feat_matches": float(n_feat),
            "inv": float(g_inv.detach()), "var": float(g_var.detach()),
            "cov": float(g_cov.detach())}

    def export_backbone_state_dict(self) -> Dict[str, torch.Tensor]:
        net = unwrap_compile(self.module)
        return {f"encoder.{k}": v.detach().cpu().clone()
                for k, v in net.encoder.state_dict().items()}


__all__ = ["VICRegLMethod"]
