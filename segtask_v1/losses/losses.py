"""2D/3D 分割逐类二值损失（per-class 独立 sigmoid）。

pred/target：(B, num_fg, *spatial)，逐前景类独立二值任务。背景隐含。
weight_map (可选)：(B, 1, *spatial)，逐体素权重，跨通道广播。
包含 Dice/BCE/Focal/Tversky/GDL/FocalTversky/Lovasz/clDice + Compound + DeepSupervision
+ MultiResolutionLoss + SliceChannelLoss + build_loss 工厂。
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..config import LossConfig

logger = logging.getLogger(__name__)

EPS = 1e-8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _check_inputs(
    pred: torch.Tensor, target: torch.Tensor, weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
    """校验形状、将 target 转为 pred dtype（AMP 安全）。"""
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target shape mismatch: "
            f"{tuple(pred.shape)} vs {tuple(target.shape)}")
    if weight_map is not None:
        expected = (pred.shape[0], 1) + tuple(pred.shape[2:])
        if tuple(weight_map.shape) != expected:
            raise ValueError(
                f"weight_map must have shape {expected}, "
                f"got {tuple(weight_map.shape)}")
    if target.dtype != pred.dtype:
        target = target.to(pred.dtype)
    return target


def _register_class_weights(
    module: nn.Module, class_weights: Optional[Sequence[float]]) -> None:
    """将 class_weights 注册为 buffer（保 state_dict 一致）。"""
    if class_weights:
        module.register_buffer(
            "class_weights",
            torch.tensor(list(class_weights), dtype=torch.float32))
    else:
        module.register_buffer("class_weights", None)


def _weighted_mean_over_classes(
    per_class: torch.Tensor, class_weights: Optional[torch.Tensor]) -> torch.Tensor:
    """最后一维（类别）加权均值：(..., C) → (...,)。"""
    if class_weights is None:
        return per_class.mean(dim=-1)
    w = class_weights.to(per_class.device).to(per_class.dtype)
    return (per_class * w).sum(dim=-1) / w.sum().clamp(min=EPS)


def _weighted_voxel_mean(
    per_voxel: torch.Tensor, weight_map: Optional[torch.Tensor], class_weights: Optional[torch.Tensor]) -> torch.Tensor:
    """逐体素损失的归一化加权均值 = sum(loss*w)/sum(w)（幅值与总权重无关）。。"""
    if weight_map is None and class_weights is None:
        return per_voxel.mean()

    weight = per_voxel.new_ones(per_voxel.shape)
    if weight_map is not None:
        weight = weight * weight_map  # broadcast (B,1,*) → (B,C,*)
    if class_weights is not None:
        cw = class_weights.to(per_voxel.device).to(per_voxel.dtype)
        # 动态阐 (C,) → (1, C, 1, ..., 1) 适应 per_voxel.ndim。
        cw_pat = 'c -> 1 c' + ' 1' * (per_voxel.ndim - 2)
        weight = weight * rearrange(cw, cw_pat)

    return (per_voxel * weight).sum() / weight.sum().clamp(min=EPS)


def _interp_mode_smooth(spatial_ndim: int) -> str:
    return {1: "linear", 2: "bilinear", 3: "trilinear"}[spatial_ndim]


# ---------------------------------------------------------------------------
# Binary Dice Loss
# ---------------------------------------------------------------------------
class BinaryDiceLoss(nn.Module):
    """逐通道二值 Dice (sigmoid)。

    参数：smooth（平滑）; squared（V-Net 平方分母）; batch_dice（跨 batch+空间汇总后除，
    稀疏前景更稳；nnU-Net 默认）; ignore_empty（仅 per-sample：排除无 GT 类避免 dice≈1 掩错）。。"""

    def __init__(
        self,
        smooth: float = 1e-5,
        squared: bool = False,
        batch_dice: bool = False,
        ignore_empty: bool = False,
        class_weights: Optional[Sequence[float]] = None):
        super().__init__()
        self.smooth = smooth
        self.squared = squared
        self.batch_dice = batch_dice
        # ignore_empty only meaningful in per-sample mode
        self.ignore_empty = ignore_empty and not batch_dice
        _register_class_weights(self, class_weights)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        target    = _check_inputs(pred, target, weight_map)
        pred_prob = torch.sigmoid(pred)
        p = rearrange(pred_prob, 'b c ... -> b c (...)')
        t = rearrange(target,    'b c ... -> b c (...)')
        # squared=True 时分母用 p²（V-Net 原文变体），梯度对低置信预测更平缓。
        p_den = p * p if self.squared else p

        sum_dims: Tuple[int, ...] = (0, 2) if self.batch_dice else (2,)

        if weight_map is not None:
            # weight_map acts as a SUMMATION weight: each voxel's contribution
            # to numerator and denominator is scaled by w consistently.
            w = rearrange(weight_map, 'b ... -> b 1 (...)')  # broadcasts over C
            intersection = (w * p * t).sum(dim=sum_dims)
            denom = (w * p_den).sum(dim=sum_dims) + (w * t).sum(dim=sum_dims)
        else:
            intersection = (p * t).sum(dim=sum_dims)
            denom = p_den.sum(dim=sum_dims) + t.sum(dim=sum_dims)

        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        # dice shape: (C,) in batch_dice mode, (B, C) in per-sample mode

        if self.batch_dice:
            return 1.0 - _weighted_mean_over_classes(dice, self.class_weights)

        per_class_loss = 1.0 - dice  # (B, C)

        if self.ignore_empty:
            has_gt = (t.sum(dim=2) > 0).to(per_class_loss.dtype)  # (B, C)
            if self.class_weights is not None:
                cw = self.class_weights.to(per_class_loss.device).to(
                    per_class_loss.dtype)
                w_cls = has_gt * cw.unsqueeze(0)
            else:
                w_cls = has_gt
            num = (per_class_loss * w_cls).sum(dim=1)
            den = w_cls.sum(dim=1).clamp(min=EPS)
            return (num / den).mean()

        return _weighted_mean_over_classes(
            per_class_loss, self.class_weights).mean()


# ---------------------------------------------------------------------------
# Binary Cross-Entropy Loss
# ---------------------------------------------------------------------------
class BCELoss(nn.Module):
    """逐通道 BCE-with-logits。class_weights 以归一化加权均值作用，使幅值与无权一致
    （与 Dice 复合时重要）。。"""

    def __init__(self, class_weights: Optional[Sequence[float]] = None):
        super().__init__()
        _register_class_weights(self, class_weights)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        target = _check_inputs(pred, target, weight_map)
        per_voxel = F.binary_cross_entropy_with_logits(
            pred, target, reduction="none")
        return _weighted_voxel_mean(per_voxel, weight_map, self.class_weights)


# ---------------------------------------------------------------------------
# Binary Focal Loss
# ---------------------------------------------------------------------------
class BinaryFocalLoss(nn.Module):
    """逐通道二值 Focal。FL = -α_t (1-p_t)^γ log(p_t)；α_t 及 alpha (正) / 1-alpha (负)。。"""

    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, class_weights: Optional[Sequence[float]] = None):
        super().__init__()
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.gamma = gamma
        _register_class_weights(self, class_weights)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target = _check_inputs(pred, target, weight_map)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        # p_t = exp(-bce) — numerically stable, saves one sigmoid call.
        pt = torch.exp(-bce)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        per_voxel = alpha_t * (1.0 - pt).pow(self.gamma) * bce
        return _weighted_voxel_mean(per_voxel, weight_map, self.class_weights)


# ---------------------------------------------------------------------------
# Binary Tversky Loss
# ---------------------------------------------------------------------------
class BinaryTverskyLoss(nn.Module):
    """逐通道 Tversky（不对称 Dice）：TI=(TP+s)/(TP+αFP+βFN+s)。默认 α=0.3 β=0.7 偏召回。。"""

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-5,
        batch_dice: bool = False,
        class_weights: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.batch_dice = batch_dice
        _register_class_weights(self, class_weights)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target = _check_inputs(pred, target, weight_map)
        pred_prob = torch.sigmoid(pred)
        p = rearrange(pred_prob, 'b c ... -> b c (...)')
        t = rearrange(target,    'b c ... -> b c (...)')
        sum_dims: Tuple[int, ...] = (0, 2) if self.batch_dice else (2,)

        if weight_map is not None:
            w = rearrange(weight_map, 'b ... -> b 1 (...)')
            tp = (w * p * t).sum(dim=sum_dims)
            fp = (w * p * (1 - t)).sum(dim=sum_dims)
            fn = (w * (1 - p) * t).sum(dim=sum_dims)
        else:
            tp = (p * t).sum(dim=sum_dims)
            fp = (p * (1 - t)).sum(dim=sum_dims)
            fn = ((1 - p) * t).sum(dim=sum_dims)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        per_class_loss = 1.0 - tversky  # (C,) or (B, C)

        if self.batch_dice:
            return _weighted_mean_over_classes(per_class_loss, self.class_weights)
        return _weighted_mean_over_classes(
            per_class_loss, self.class_weights
        ).mean()


# ---------------------------------------------------------------------------
# Compound Loss
# ---------------------------------------------------------------------------
class CompoundLoss(nn.Module):
    """多损失加权和"""

    def __init__(self, losses: Sequence[nn.Module], weights: Sequence[float]):
        super().__init__()

        self.losses  = nn.ModuleList(losses)
        self.weights = list(weights)

    def forward(
        self,
        pred      : torch.Tensor,
        target    : torch.Tensor,
        weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        total = pred.new_zeros(())
        for fn, w in zip(self.losses, self.weights):
            total = total + w * fn(pred, target, weight_map=weight_map)
        return total


# ---------------------------------------------------------------------------
# Deep Supervision Wrapper
# ---------------------------------------------------------------------------
class DeepSupervisionLoss(nn.Module):
    """多尺度深监督。默认将 target 近邻下采样到每个 pred 尺寸"""

    def __init__(
        self,
        base_loss        : nn.Module,
        weights          : Sequence[float],
        normalize_weights: bool = True,
        upsample_pred    : bool = False):
        super().__init__()

        self.base_loss = base_loss
        w = list(weights)
        if normalize_weights:
            s = sum(w)
            if s <= 0:
                raise ValueError(f"DS weights must sum to positive, got {s}")
            w = [wi / s for wi in w]
        self.weights = w
        self.upsample_pred = upsample_pred

    def forward(
        self,
        preds     : Union[torch.Tensor, List[torch.Tensor]],
        target    : torch.Tensor,
        weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:

        if isinstance(preds, torch.Tensor):  # 单分辨率
            return self.base_loss(preds, target, weight_map=weight_map)

        if len(preds) != len(self.weights):
            raise ValueError(
                f"Number of predictions ({len(preds)}) must match number "
                f"of DS weights ({len(self.weights)})")

        total = preds[0].new_zeros(())
        for w, pred in zip(self.weights, preds):  # 依次对每个dec特征监督
            tgt_i, wm_i = target, weight_map
            if pred.shape[2:] != target.shape[2:]:
                spatial_ndim = pred.ndim - 2
                if self.upsample_pred:
                    pred = F.interpolate(
                        pred,
                        size=target.shape[2:],
                        mode=_interp_mode_smooth(spatial_ndim),
                        align_corners=False)
                else:
                    tgt_i = F.interpolate(
                        target, size=pred.shape[2:], mode="nearest")
                    if weight_map is not None:
                        wm_i = F.interpolate(
                            weight_map, size=pred.shape[2:], mode="nearest")
            total = total + w * self.base_loss(pred, tgt_i, weight_map=wm_i)
        return total


# ---------------------------------------------------------------------------
# Generalized Dice Loss  (Sudre et al., DLMIA 2017)
# ---------------------------------------------------------------------------
class GeneralizedDiceLoss(nn.Module):
    """Generalized Dice Loss（Sudre+ DLMIA 2017）：w_c = 1/(Σ t_c)^2、
    GDL = 1 - 2Σ w_c TP_c / Σ w_c (P_c+T_c)；自动以体积倒数补偿类别不平衡。

    weight_type: 'square'（论文） / 'simple' (1/Σt) / 'uniform'（禁体积加权）。
    class_weights 在体积权后额外叠加；w_max 夹住 1/volume 防爆炸。。"""

    def __init__(
        self,
        smooth: float = 1e-5,
        batch_dice: bool = True,
        weight_type: str = "square",
        class_weights: Optional[Sequence[float]] = None,
        w_max: float = 1e5):
        super().__init__()
        if weight_type not in ("square", "simple", "uniform"):
            raise ValueError(
                f"weight_type must be one of square/simple/uniform, "
                f"got {weight_type!r}")
        self.smooth = smooth
        self.batch_dice = batch_dice
        self.weight_type = weight_type
        self.w_max = w_max
        _register_class_weights(self, class_weights)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        target = _check_inputs(pred, target, weight_map)
        pred_prob = torch.sigmoid(pred)
        p = rearrange(pred_prob, 'b c ... -> b c (...)')
        t = rearrange(target,    'b c ... -> b c (...)')
        sum_dims: Tuple[int, ...] = (0, 2) if self.batch_dice else (2,)

        if weight_map is not None:
            w_vox = rearrange(weight_map, 'b ... -> b 1 (...)')
            t_vol = (w_vox * t).sum(dim=sum_dims)
            tp = (w_vox * p * t).sum(dim=sum_dims)
            denom = (w_vox * (p + t)).sum(dim=sum_dims)
        else:
            t_vol = t.sum(dim=sum_dims)
            tp = (p * t).sum(dim=sum_dims)
            denom = (p + t).sum(dim=sum_dims)

        # Volume-based class weights. Shapes: (C,) if batch_dice else (B, C).
        t_safe = t_vol.clamp(min=EPS)
        if self.weight_type == "square":
            wc = 1.0 / (t_safe * t_safe)
        elif self.weight_type == "simple":
            wc = 1.0 / t_safe
        else:  # uniform
            wc = torch.ones_like(t_safe)
        wc = wc.clamp(max=self.w_max)

        if self.class_weights is not None:
            cw = self.class_weights.to(wc.device).to(wc.dtype)
            wc = wc * cw  # broadcast over class axis (last)

        # Weighted aggregate along the class axis (last dim).
        num = (wc * tp).sum(dim=-1)
        den = (wc * denom).sum(dim=-1)
        gdl = 1.0 - (2.0 * num + self.smooth) / (den + self.smooth)
        # () if batch_dice else (B,)
        return gdl if gdl.ndim == 0 else gdl.mean()


# ---------------------------------------------------------------------------
# Focal Tversky Loss  (Abraham & Khan, ISBI 2019)
# ---------------------------------------------------------------------------
class BinaryFocalTverskyLoss(nn.Module):
    """Focal Tversky（Abraham & Khan ISBI 2019）：FTL = mean_c((1-TI_c)^γ)。
    α<β (默认0.3/0.7) 偏召回；γ>1 集中梯度于难类。
    默认 γ=4/3 对应原论文 γ_paper=0.75（实验采用值）。。"""

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 4.0 / 3.0,
        smooth: float = 1e-5,
        batch_dice: bool = False,
        class_weights: Optional[Sequence[float]] = None):
        super().__init__()
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.batch_dice = batch_dice
        _register_class_weights(self, class_weights)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        target = _check_inputs(pred, target, weight_map)
        pred_prob = torch.sigmoid(pred)
        p = rearrange(pred_prob, 'b c ... -> b c (...)')
        t = rearrange(target,    'b c ... -> b c (...)')
        sum_dims: Tuple[int, ...] = (0, 2) if self.batch_dice else (2,)

        if weight_map is not None:
            w = rearrange(weight_map, 'b ... -> b 1 (...)')
            tp = (w * p * t).sum(dim=sum_dims)
            fp = (w * p * (1 - t)).sum(dim=sum_dims)
            fn = (w * (1 - p) * t).sum(dim=sum_dims)
        else:
            tp = (p * t).sum(dim=sum_dims)
            fp = (p * (1 - t)).sum(dim=sum_dims)
            fn = ((1 - p) * t).sum(dim=sum_dims)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth)
        # Clamp (1 - TI) into [0, 1] for numerical safety under fractional powers.
        focal = (1.0 - tversky).clamp(min=0.0, max=1.0).pow(self.gamma)

        if self.batch_dice:
            return _weighted_mean_over_classes(focal, self.class_weights)
        return _weighted_mean_over_classes(focal, self.class_weights).mean()


# ---------------------------------------------------------------------------
# Lovász-Hinge Loss  (Berman et al., CVPR 2018)
# ---------------------------------------------------------------------------
def _lovasz_grad_batched(gt_sorted: torch.Tensor) -> torch.Tensor:
    """向量化 Lovász 扩展梯度。gt_sorted (..., L) 按误差降序排列。。"""
    gts = gt_sorted.sum(dim=-1, keepdim=True)
    intersection = gts - gt_sorted.cumsum(dim=-1)
    union = gts + (1.0 - gt_sorted).cumsum(dim=-1)
    jaccard = 1.0 - intersection / union.clamp(min=EPS)
    # 沿 L 轴差分（Lovász 阶跃梯度）。
    if jaccard.shape[-1] > 1:
        shifted = jaccard[..., 1:] - jaccard[..., :-1]
        jaccard = torch.cat([jaccard[..., :1], shifted], dim=-1)
    return jaccard


class LovaszHingeLoss(nn.Module):
    """逐类二值 Lovász-Hinge（Berman+ CVPR 2018）——IoU 直接代理。输入为原始 logits。
    weight_map 启发式以逐体素权乘非负 hinge 项（保排序梯度结构）；严格理论使用时建议禁用。
    per_sample：True 逐 (B,C) 独立排序取均；False 跨 batch 拼接后排序（小 patch 更平滑）。。"""

    def __init__(
        self, per_sample: bool = True,
        class_weights: Optional[Sequence[float]] = None):
        super().__init__()
        self.per_sample = per_sample
        _register_class_weights(self, class_weights)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        target = _check_inputs(pred, target, weight_map)
        logits = rearrange(pred,   'b c ... -> b c (...)')   # operate on logits
        t      = rearrange(target, 'b c ... -> b c (...)')
        B, C = logits.shape[:2]

        # Hinge 误差（logit 空间）：e = 1 - s*z，s=2t-1∈{-1,+1}；Lovász 需 ReLU 前的有符号边际。
        signed = 2.0 * t - 1.0
        errors = 1.0 - signed * logits

        if weight_map is not None:
            w_vox = rearrange(weight_map, 'b ... -> b 1 (...)').clamp(min=0)
            # 仅加权被惩罚侧；负边际 = 正确，ReLU 后为 0。
            errors = torch.where(
                errors > 0, errors * w_vox, errors)

        if self.per_sample:
            # 沿空间轴排序，以同 perm 取 target。
            err_sorted, perm = torch.sort(errors, dim=-1, descending=True)
            gt_sorted = t.gather(dim=-1, index=perm)
            grad = _lovasz_grad_batched(gt_sorted)
            per_class = (F.relu(err_sorted) * grad).sum(dim=-1)  # (B, C)
            return _weighted_mean_over_classes(
                per_class, self.class_weights).mean()

        # Batch-级 Lovász：逐通道 reshape 为 (B*L)，一次排序。
        errors_bc = rearrange(errors, 'b c l -> c (b l)')  # (C, B*L)
        t_bc      = rearrange(t,      'b c l -> c (b l)')
        err_sorted, perm = torch.sort(errors_bc, dim=-1, descending=True)
        gt_sorted = t_bc.gather(dim=-1, index=perm)
        grad = _lovasz_grad_batched(gt_sorted)
        per_class = (F.relu(err_sorted) * grad).sum(dim=-1)  # (C,)
        return _weighted_mean_over_classes(per_class, self.class_weights)


# ---------------------------------------------------------------------------
# Soft clDice  (Shit et al., CVPR 2021) — topology-preserving
# ---------------------------------------------------------------------------
def _soft_erode(x: torch.Tensor, spatial_ndim: int) -> torch.Tensor:
    """可微形态学腐蚀（-max_pool，kernel=3）。"""
    if spatial_ndim == 3:
        return -F.max_pool3d(-x, kernel_size=3, stride=1, padding=1)
    if spatial_ndim == 2:
        return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
    raise ValueError(f"spatial_ndim must be 2 or 3, got {spatial_ndim}")


def _soft_dilate(x: torch.Tensor, spatial_ndim: int) -> torch.Tensor:
    """可微形态学膨胀（max_pool，kernel=3）。"""
    if spatial_ndim == 3:
        return F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
    if spatial_ndim == 2:
        return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    raise ValueError(f"spatial_ndim must be 2 or 3, got {spatial_ndim}")


def _soft_skeletonize(
    img: torch.Tensor, n_iter: int, spatial_ndim: int) -> torch.Tensor:
    """迭代 soft skeletonization（Shit+ Alg. 1）；img 在 [0,1]，返同形 soft 骨架。。"""
    def _open(y: torch.Tensor) -> torch.Tensor:
        return _soft_dilate(_soft_erode(y, spatial_ndim), spatial_ndim)

    skel = F.relu(img - _open(img))
    for _ in range(n_iter):
        img = _soft_erode(img, spatial_ndim)
        delta = F.relu(img - _open(img))
        # (1-skel) 门控：避免重复计入已在 skel 中的体素。
        skel = skel + (1.0 - skel).clamp(min=0.0) * delta
    return skel


class SoftCLDiceLoss(nn.Module):
    """Soft centerline (clDice) 保拓损失（Shit+ CVPR 2021）。与 Dice 复合使用（dice_cldice）最佳。
    weight_map 仅 API 统一接受但忽略（clDice 为拓扑指标，逐体素权重无一致语义）。。"""

    def __init__(
        self,
        iter_: int = 3,
        smooth: float = 1.0,
        class_weights: Optional[Sequence[float]] = None):
        super().__init__()
        if iter_ < 1:
            raise ValueError(f"iter_ must be >= 1, got {iter_}")
        self.iter = iter_
        self.smooth = smooth
        _register_class_weights(self, class_weights)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 有意忽略 weight_map（见类 docstring）。
        del weight_map
        _check_inputs(pred, target)
        spatial_ndim = pred.ndim - 2
        if spatial_ndim not in (2, 3):
            raise ValueError(
                f"SoftCLDiceLoss expects 2D or 3D spatial input; got "
                f"pred.ndim={pred.ndim}")
        target = target.to(pred.dtype)
        pred_prob = torch.sigmoid(pred)

        skel_pred = _soft_skeletonize(pred_prob, self.iter, spatial_ndim)
        skel_t = _soft_skeletonize(target, self.iter, spatial_ndim)

        sp = rearrange(skel_pred, 'b c ... -> b c (...)')
        st = rearrange(skel_t,    'b c ... -> b c (...)')
        p  = rearrange(pred_prob, 'b c ... -> b c (...)')
        t  = rearrange(target,    'b c ... -> b c (...)')

        tprec = ((sp * t).sum(dim=-1) + self.smooth) / (
            sp.sum(dim=-1) + self.smooth)
        tsens = ((st * p).sum(dim=-1) + self.smooth) / (
            st.sum(dim=-1) + self.smooth)
        # 调和均值分母不加 smooth（官方 clDice 定义），否则完美预测损失不为 0；
        # smooth>0 时 tprec/tsens 严格 >0，clamp 仅作 smooth=0 时的除零兜底。
        cldice = 2.0 * tprec * tsens / (tprec + tsens).clamp(min=1e-8)
        per_class_loss = 1.0 - cldice  # (B, C)
        return _weighted_mean_over_classes(
            per_class_loss, self.class_weights).mean()


# ---------------------------------------------------------------------------
# Loss factory
# ---------------------------------------------------------------------------
def _build_dice(cfg: LossConfig, cw: Optional[List[float]]) -> BinaryDiceLoss:
    return BinaryDiceLoss(
        smooth        = cfg.dice_smooth,
        squared       = cfg.dice_squared,
        batch_dice    = cfg.batch_dice,
        ignore_empty  = cfg.ignore_empty,
        class_weights = cw)


def _build_bce(cfg: LossConfig, cw: Optional[List[float]]) -> BCELoss:
    return BCELoss(class_weights=cw)


def _build_focal(cfg: LossConfig, cw: Optional[List[float]]) -> BinaryFocalLoss:
    return BinaryFocalLoss(
        alpha=cfg.focal_alpha,
        gamma=cfg.focal_gamma,
        class_weights=cw,
    )


def _build_tversky(
    cfg: LossConfig, cw: Optional[List[float]]
) -> BinaryTverskyLoss:
    return BinaryTverskyLoss(
        alpha=cfg.tversky_alpha,
        beta=cfg.tversky_beta,
        smooth=cfg.dice_smooth,
        batch_dice=cfg.batch_dice,
        class_weights=cw,
    )


def _build_gdl(
    cfg: LossConfig, cw: Optional[List[float]]) -> GeneralizedDiceLoss:
    return GeneralizedDiceLoss(
        smooth=cfg.dice_smooth,
        batch_dice=cfg.batch_dice,
        weight_type=cfg.gdl_weight_type,
        class_weights=cw,
        w_max=cfg.gdl_w_max,
    )


def _build_focal_tversky(
    cfg: LossConfig, cw: Optional[List[float]]) -> BinaryFocalTverskyLoss:
    return BinaryFocalTverskyLoss(
        alpha=cfg.tversky_alpha,
        beta=cfg.tversky_beta,
        gamma=cfg.focal_tversky_gamma,
        smooth=cfg.dice_smooth,
        batch_dice=cfg.batch_dice,
        class_weights=cw,
    )


def _build_lovasz(
    cfg: LossConfig, cw: Optional[List[float]]) -> LovaszHingeLoss:
    return LovaszHingeLoss(
        per_sample=cfg.lovasz_per_sample,
        class_weights=cw,
    )


def _build_cldice(
    cfg: LossConfig, cw: Optional[List[float]]) -> SoftCLDiceLoss:
    return SoftCLDiceLoss(
        iter_=cfg.cldice_iter,
        smooth=cfg.cldice_smooth,
        class_weights=cw,
    )


_SINGLE_BUILDERS = {
    "dice"         : _build_dice,
    "bce"          : _build_bce,
    "focal"        : _build_focal,
    "tversky"      : _build_tversky,
    "gdl"          : _build_gdl,
    "focal_tversky": _build_focal_tversky,
    "lovasz"       : _build_lovasz,
    "cldice"       : _build_cldice}

_COMPOUND_BUILDERS = {
    "dice_bce"          : (_build_dice, _build_bce),
    "dice_focal"        : (_build_dice, _build_focal),
    "dice_tversky"      : (_build_dice, _build_tversky),
    "focal_plus_tversky": (_build_focal, _build_tversky),
    "dice_cldice"       : (_build_dice, _build_cldice),          # Shit et al. recipe
    "dice_focal_tversky": (_build_dice, _build_focal_tversky),
    "dice_lovasz"       : (_build_dice, _build_lovasz),
    "bce_lovasz"        : (_build_bce, _build_lovasz),
    "gdl_bce"           : (_build_gdl, _build_bce),
    "gdl_focal"         : (_build_gdl, _build_focal)}


def _compound_weights(cfg: LossConfig, n: int) -> List[float]:
    ws = list(cfg.compound_weights or [])
    if len(ws) >= n:
        return ws[:n]  # 自动适配长度
    logger.warning(
        "compound_weights has %d entries, need %d; defaulting missing to 1.0",
        len(ws), n)
    return (ws + [1.0] * n)[:n]


class MultiResolutionLoss(nn.Module):
    """多分辨率损失 
    输入预测值 (B, num_fg*C_res, D,H,W)、标签 (B, C_res, D,H,W) 。
    按 C_res 拆 pred、逐尺度 binary 化 label、逐分辨率 base_loss 后取均。"""

    def __init__(self, base_loss: nn.Module, num_fg_classes: int, num_res: int, label_values: List[int]):
        super().__init__()
        self.base_loss    = base_loss
        self.num_fg       = num_fg_classes
        self.num_res      = num_res
        self.label_values = label_values
        self.fg_values    = label_values[1:]  # exclude background

        # 诊断：每次 forward 把每个分辨率的损失以 detached tensor 追加到 history
        # （不在热路径上 .item() 同步）。被 DeepSupervisionLoss 多次调用时，
        # history 会累积多次（每个 DS 尺度一行）。训练循环仅在日志步
        # 调用 pop_per_res_diag() 取行均值并清空（单次同步），因此诊断值
        # 是自上次日志步以来的窗口均值，history 长度上限 ≈ log_every×DS 尺度数。
        self._per_res_history: List[torch.Tensor] = []

    def pop_per_res_diag(self) -> Optional[List[float]]:
        """取 history 行均（对 DS 尺度做平均），清空并返回；history 为空返 None。

        GPU→CPU 同步仅在此处发生一次，不在 forward 热路径上。"""
        if not self._per_res_history:
            return None
        avg = torch.stack(self._per_res_history).mean(dim=0)
        self._per_res_history = []
        return [float(v) for v in avg.tolist()]

    def forward(
        self, pred: torch.Tensor, label_raw: torch.Tensor, weight_map: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """跨全部分辨率计算损失并取均。pred (B,num_fg*C_res,*); label_raw (B,C_res,*) 。"""
        total = pred.new_zeros(())
        per_res_row: List[torch.Tensor] = []

        for r in range(self.num_res):  # 依次算每个分辨率的监督
            pred_r   = pred[:, r * self.num_fg:(r + 1) * self.num_fg]
            lbl_r    = label_raw[:, r]
            target_r = self._label_to_binary(lbl_r)

            # 逐分辨率 weight_map：(B,D,H,W) → (B,1,D,H,W)。
            wm_r = None
            if weight_map is not None:
                wm_r = weight_map[:, r:r + 1]  # (B, 1, D, H, W)

            l_r = self.base_loss(pred_r, target_r, weight_map=wm_r)
            total = total + l_r
            per_res_row.append(l_r.detach())

        self._per_res_history.append(torch.stack(per_res_row))
        return total / self.num_res

    def _label_to_binary(self, label: torch.Tensor) -> torch.Tensor:
        """整数 label (B,D,H,W) → 二值掊叠 (B,num_fg,D,H,W)，GPU 向量化。。"""
        fg = torch.tensor(self.fg_values, device=label.device, dtype=label.dtype)
        # label (B,D,H,W) → (B,1,D,H,W); fg (num_fg,) → (1,num_fg,1,1,1)。
        label_exp = label.unsqueeze(1)
        fg_pat    = 'c -> 1 c' + ' 1' * (label.ndim - 1)
        fg_exp    = rearrange(fg, fg_pat)
        return (label_exp == fg_exp).float()

    def split_for_metrics(
        self, pred: torch.Tensor, label_raw: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """为指标抽取首分辨率（3D）二值化：返 (pred_1x, target_1x)，均 (B,num_fg,*spatial)。。"""
        pred_1x   = pred[:, :self.num_fg]
        target_1x = self._label_to_binary(label_raw[:, 0])
        return pred_1x, target_1x

    def binarize_full(self, label_raw: torch.Tensor) -> torch.Tensor:
        """整数 label (B,C_res,*spatial) → 与主头输出同形二值 (B,num_fg*C_res,*spatial)。

        逐分辨率二值化后按通道拼接，layout 与 ``forward`` 中
        ``pred[:, r*num_fg:(r+1)*num_fg]`` 切片一致；供中心线/距离场辅助头构造目标。
        """
        parts = [self._label_to_binary(label_raw[:, r]) for r in range(self.num_res)]
        return torch.cat(parts, dim=1)


# ---------------------------------------------------------------------------
# 2.5D Slice-Channel Loss Wrapper
# ---------------------------------------------------------------------------
class SliceChannelLoss(nn.Module):
    """2.5D patch 模式包装。输入 (B, num_fg*D, H, W) logits + (B, D, H, W) 整数 label。
    逐类 pred_c=pred[:, c*D:(c+1)*D]、target_c=(label==fg_values[c])，逐类平均为总损失。

    reduction：'per_slice' → (B*D,1,H,W) 逐切独立 2D Dice/Tversky；
              'per_volume' → (B,1,D,H,W) 逐窗口 3D Dice/Tversky（空切与非空切共享分母）。
              BCE/Focal/Lovász 下二者结果一致（逐体素均值）。

    class_weights：逐类迭代使 base_loss 内部 cw 折叠为无操作，因此在包装层重读 base_loss.class_weights
    以归一化加权均合 per-class 损失；cw=None 时与简单均值位精确一致。。"""

    _VALID_REDUCTIONS = ("per_slice", "per_volume")

    def __init__(
        self, base_loss: nn.Module, num_fg_classes: int, num_slices: int, label_values: List[int], reduction: str = "per_slice"):
        super().__init__()
        if num_slices < 1:
            raise ValueError(f"num_slices must be >= 1, got {num_slices}")
        if reduction not in self._VALID_REDUCTIONS:
            raise ValueError(
                f"reduction must be one of {self._VALID_REDUCTIONS}, "
                f"got {reduction!r}")

        self.base_loss    = base_loss
        self.num_fg       = num_fg_classes
        self.num_slices   = num_slices
        self.label_values = label_values
        self.fg_values    = label_values[1:]
        self.reduction    = reduction

        # 构造时验长：forward 重读 base_loss.class_weights（跟随 device 定位，不复制 buffer）。
        cw_buf = getattr(base_loss, "class_weights", None)
        if cw_buf is not None and cw_buf.numel() != num_fg_classes:
            raise ValueError(
                f"SliceChannelLoss: base_loss.class_weights has "
                f"{cw_buf.numel()} entries but num_fg_classes="
                f"{num_fg_classes}. Provide ``cfg.loss.class_weights`` "
                f"with exactly num_fg_classes entries (one per foreground "
                f"class).")

    # ------------------------------------------------------------------
    # Per-slice reshape helpers (rank-4 contract: (B*D, num_fg, H, W))
    # ------------------------------------------------------------------
    def _label_to_binary(self, label_raw: torch.Tensor) -> torch.Tensor:
        """(B,D,H,W) 整数 → (B*D, num_fg, H, W) 二值（rank-4 供 2D base loss）。。"""
        if label_raw.ndim != 4:
            raise ValueError(
                f"SliceChannelLoss expects (B, D, H, W) raw label, "
                f"got rank-{label_raw.ndim}")

        B, D, H, W = label_raw.shape
        if D != self.num_slices:
            raise ValueError(
                f"label slice count {D} != configured num_slices "
                f"{self.num_slices}")

        fg   = torch.tensor(self.fg_values, device=label_raw.device, dtype=label_raw.dtype)
        fg_b = rearrange(fg, 'c -> 1 c 1 1')                        # (1, num_fg, 1, 1)
        flat = rearrange(label_raw, 'b d h w -> (b d) 1 h w')        # (B*D, 1, H, W)
        return (flat == fg_b).float()                               # (B*D, num_fg, H, W)

    def _split_pred(self, pred: torch.Tensor) -> torch.Tensor:
        """(B, num_fg*D, H, W) → (B*D, num_fg, H, W)。"""
        if pred.ndim != 4:
            raise ValueError(
                f"SliceChannelLoss expects (B, num_fg*D, H, W) pred, "
                f"got rank-{pred.ndim}")

        B, total_c, H, W = pred.shape
        D                = self.num_slices
        if total_c != self.num_fg * D:
            raise ValueError(
                f"pred channel count {total_c} != num_fg*D = "
                f"{self.num_fg}*{D} = {self.num_fg * D}")
        # (B, num_fg*D, H, W) → (B*D, num_fg, H, W)
        return rearrange(
            pred, 'b (c d) h w -> (b d) c h w',
            c=self.num_fg, d=D)

    @staticmethod
    def _flatten_weight_map(
        weight_map: Optional[torch.Tensor], num_slices: int,
    ) -> Optional[torch.Tensor]:
        """(B,D,H,W) → (B*D,1,H,W) 供 base loss 广播。"""
        if weight_map is None:
            return None
        if weight_map.ndim != 4:
            raise ValueError(
                f"SliceChannelLoss expects (B, D, H, W) weight_map, "
                f"got rank-{weight_map.ndim}")

        B, D, H, W = weight_map.shape
        if D != num_slices:
            raise ValueError(
                f"weight_map slice count {D} != num_slices {num_slices}")
        return rearrange(weight_map, 'b d h w -> (b d) 1 h w')

    # ------------------------------------------------------------------
    # Per-volume reshape helpers (rank-5 contract: (B, num_fg, D, H, W))
    # ------------------------------------------------------------------
    def _split_pred_5d(self, pred: torch.Tensor) -> torch.Tensor:
        """(B, num_fg*D, H, W) → (B, num_fg, D, H, W)。"""
        if pred.ndim != 4:
            raise ValueError(
                f"SliceChannelLoss expects (B, num_fg*D, H, W) pred, "
                f"got rank-{pred.ndim}")

        B, total_c, H, W = pred.shape
        D                = self.num_slices
        if total_c != self.num_fg * D:
            raise ValueError(
                f"pred channel count {total_c} != num_fg*D = "
                f"{self.num_fg}*{D} = {self.num_fg * D}")
        return rearrange(
            pred, 'b (c d) h w -> b c d h w',
            c=self.num_fg, d=D)

    def _label_to_binary_5d(self, label_raw: torch.Tensor) -> torch.Tensor:
        """(B,D,H,W) 整数 → (B, num_fg, D, H, W) 二值。"""
        if label_raw.ndim != 4:
            raise ValueError(
                f"SliceChannelLoss expects (B, D, H, W) raw label, "
                f"got rank-{label_raw.ndim}")

        B, D, H, W = label_raw.shape
        if D != self.num_slices:
            raise ValueError(
                f"label slice count {D} != configured num_slices "
                f"{self.num_slices}")

        fg   = torch.tensor(self.fg_values, device=label_raw.device, dtype=label_raw.dtype)
        fg_b = rearrange(fg, 'c -> 1 c 1 1 1')                       # (1, num_fg, 1, 1, 1)
        flat = label_raw.unsqueeze(1)                                # (B, 1, D, H, W)
        return (flat == fg_b).float()                                # (B, num_fg, D, H, W)

    def binarize_full(self, label_raw: torch.Tensor) -> torch.Tensor:
        """整数 label (B,D,H,W) → 与主头输出同形二值 (B,num_fg*D,H,W)，layout ``b (c d) h w``。

        与 ``_split_pred_5d`` 的折叠口径一致（class-major, slice-minor）；
        供中心线/距离场辅助头在折叠 2.5D 表示上逐 (类,切片) 构造 2D 目标。
        """
        bin5d = self._label_to_binary_5d(label_raw)                  # (B, num_fg, D, H, W)
        return rearrange(bin5d, 'b c d h w -> b (c d) h w')

    @staticmethod
    def _wmap_to_5d(
        weight_map: Optional[torch.Tensor], num_slices: int
        ) -> Optional[torch.Tensor]:
        """(B,D,H,W) → (B,1,D,H,W) 供 base loss 广播。"""
        if weight_map is None:
            return None
        if weight_map.ndim != 4:
            raise ValueError(
                f"SliceChannelLoss expects (B, D, H, W) weight_map, "
                f"got rank-{weight_map.ndim}")

        B, D, H, W = weight_map.shape
        if D != num_slices:
            raise ValueError(
                f"weight_map slice count {D} != num_slices {num_slices}")
        return weight_map.unsqueeze(1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _aggregate_per_class(self, terms: List[torch.Tensor]) -> torch.Tensor:
        """逐类损失汇总为最终标量：cw=None 时简单均值；否则 Σ w_c·L_c/Σ w_c（幅值与 cw 选择无关）。。"""
        if not terms:
            # num_fg==0 会被 Config.validate 拒；保留分支避免退化构造崩。
            raise RuntimeError(
                "SliceChannelLoss._aggregate_per_class got 0 terms")
        stacked = torch.stack(terms)  # (num_fg,)
        cw_buf  = getattr(self.base_loss, "class_weights", None)
        if cw_buf is None:
            return stacked.mean()
        cw = cw_buf.to(stacked.device).to(stacked.dtype)
        return (stacked * cw).sum() / cw.sum().clamp(min=EPS)

    def forward(
        self, pred: torch.Tensor, label_raw: torch.Tensor, weight_map: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """按配置的 reduction 计逐类均值损失。"""
        if self.reduction == "per_volume":  # resize到(B, num_fg, D, H, W)再算
            pred_5d   = self._split_pred_5d(pred)              # (B, num_fg, D, H, W)
            target_5d = self._label_to_binary_5d(label_raw)    # (B, num_fg, D, H, W)
            wm_5d     = self._wmap_to_5d(weight_map, self.num_slices)

            # 单前景类：逐类循环退化为单次调用，直接短路（与聚合路径数学恒等：
            # 均值/加权均值对单项都是恒等变换）。
            if self.num_fg == 1:
                return self.base_loss(pred_5d, target_5d, weight_map=wm_5d)

            # 3D 二值分割逐类循环：Dice/Tversky 跨 (D,H,W) 汇总，空切与非空切共享分母。
            terms: List[torch.Tensor] = []
            for c in range(self.num_fg):
                pred_c   = pred_5d[:, c:c + 1]               # (B, 1, D, H, W)
                target_c = target_5d[:, c:c + 1]             # (B, 1, D, H, W)
                terms.append(self.base_loss(pred_c, target_c, weight_map=wm_5d))
            return self._aggregate_per_class(terms)

        # 默认 per_slice
        pred_flat   = self._split_pred(pred)                # (B*D, num_fg, H, W)
        target_flat = self._label_to_binary(label_raw)      # (B*D, num_fg, H, W)
        wm_flat     = self._flatten_weight_map(weight_map, self.num_slices)

        # 单前景类短路（同 per_volume 分支，数学恒等）。
        if self.num_fg == 1:
            return self.base_loss(pred_flat, target_flat, weight_map=wm_flat)

        # 逐 fg 类传单通道二值张量使 base_loss 作为独立 2D 二值分割。
        terms = []
        for c in range(self.num_fg):
            pred_c   = pred_flat[:, c:c + 1]                # (B*D, 1, H, W)
            target_c = target_flat[:, c:c + 1]              # (B*D, 1, H, W)
            terms.append(self.base_loss(pred_c, target_c, weight_map=wm_flat))
        return self._aggregate_per_class(terms)

    def split_for_metrics(
        self, pred: torch.Tensor, label_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返与 reduction 一致的形状供指标与验证损失：
        per_slice → (B*D,num_fg,H,W)；per_volume → (B,num_fg,D,H,W)。
        compute_dice_per_class / dice_batch_stats 均 rank 无关。。"""
        if self.reduction == "per_volume":
            return (self._split_pred_5d(pred),
                    self._label_to_binary_5d(label_raw))
        return self._split_pred(pred), self._label_to_binary(label_raw)


def build_loss(cfg: LossConfig) -> nn.Module:
    """按 cfg 构造损失（全部逐类独立 sigmoid 二值）。"""
    cw   = list(cfg.class_weights) if cfg.class_weights else None
    name = cfg.name.lower()

    if name in _SINGLE_BUILDERS:
        return _SINGLE_BUILDERS[name](cfg, cw)

    if name in _COMPOUND_BUILDERS:
        builders   = _COMPOUND_BUILDERS[name]
        components = [b(cfg, cw) for b in builders]
        weights    = _compound_weights(cfg, len(components))
        return CompoundLoss(components, weights)

    supported = sorted(
        list(_SINGLE_BUILDERS.keys()) + list(_COMPOUND_BUILDERS.keys()))
    raise ValueError(f"Unknown loss: {cfg.name!r}. Supported: {supported}")