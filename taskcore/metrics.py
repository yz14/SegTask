"""分割/概率图指标数学（seg 语义，任务层与验证/探针共用）。

由 ``taskcore.utils.common`` 下沉的独立模块：dice / overlap 派生指标 /
调和均值 / spacing 感知 surface dice。通用工具（seed/EMA/Timer 等）仍在
``taskcore.utils.common``；旧导入路径经 re-export 保持兼容。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

@torch.no_grad()
def compute_dice_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-5,
    ignore_empty: bool = True,
) -> torch.Tensor:
    """(B,C,D,H,W) 逐类 sigmoid Dice。ignore_empty=True (nnU-Net)：空 GT 样本不入均；batch 全空类返 0。"""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    p = rearrange(pred_bin, 'b c ... -> b c (...)')
    t = rearrange(target, 'b c ... -> b c (...)')

    intersection = (p * t).sum(dim=2)
    denom = p.sum(dim=2) + t.sum(dim=2)
    dice = (2.0 * intersection + smooth) / (denom + smooth)  # (B, C)

    if not ignore_empty:
        return dice.mean(dim=0)

    has_gt = (t.sum(dim=2) > 0).to(dice.dtype)
    num = (dice * has_gt).sum(dim=0)
    den = has_gt.sum(dim=0).clamp(min=1)
    mean_dice = torch.where(
        has_gt.sum(dim=0) > 0, num / den, torch.zeros_like(num))
    return mean_dice


@torch.no_grad()
def dice_batch_stats(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: Union[float, torch.Tensor] = 0.5,
    pred_is_binary: bool = False,
) -> Dict[str, torch.Tensor]:
    """逐类汇总 batch 级混淆量，供 nnU-Net 风格 pooled 指标（Σ分子/Σ分母）。

    返回（每键为长度 C 的张量，``voxels`` 为标量）：
      * ``inter``     = ΣTP                （= Σ(p·t)）
      * ``denom``     = Σ(|p|+|t|)         （= 2·ΣTP + ΣFP + ΣFN）
      * ``pred_sum``  = Σ|p|               （= ΣTP + ΣFP）
      * ``target_sum``= Σ|t|               （= ΣTP + ΣFN）
      * ``voxels``    = Σ每样本空间体素数 （= ΣTP + ΣFP + ΣFN + ΣTN，类共享）
      * ``n_with_gt`` = 该 batch 中含正例 GT 的样本数（per-class）

    由这五个量可零额外通信地导出 dice / iou / recall / precision /
    volume_similarity / mcc 等指标，详见 ``derive_overlap_metrics``。
    旧字段 ``inter``/``denom``/``n_with_gt`` 完全保留以兼容已有调用方。

    ``threshold`` 支持标量或可广播到 ``(C, 1, ..., 1)`` 的逐类阈值张量。

    ``pred_is_binary=True`` 时 ``pred`` 已是 {0,1} 二值体，跳过 sigmoid+阈值
    （与喂饱和 logits 经 sigmoid 阈值后逐位一致，省两次逐元素 pass）。
    """
    pred_bin = (pred.float() if pred_is_binary
                else (torch.sigmoid(pred) > threshold).float())
    p = rearrange(pred_bin, 'b c ... -> b c (...)')
    t = rearrange(target, 'b c ... -> b c (...)')
    pred_sum   = p.sum(dim=(0, 2))
    target_sum = t.sum(dim=(0, 2))
    inter      = (p * t).sum(dim=(0, 2))
    denom      = pred_sum + target_sum
    n_with_gt  = (t.sum(dim=2) > 0).sum(dim=0).float()
    # 类共享：B * spatial_numel。用 float64 累加，防止超大体素数在长 val 集上
    # 触及 float32 精度上限（>16M 后整数误差非零）。
    voxels = torch.tensor(
        float(p.shape[0]) * float(p.shape[2]),
        dtype=torch.float64, device=p.device)
    return {
        "inter"     : inter,
        "denom"     : denom,
        "pred_sum"  : pred_sum,
        "target_sum": target_sum,
        "voxels"    : voxels,
        "n_with_gt" : n_with_gt}


@torch.no_grad()
def derive_overlap_metrics(
    inter:      torch.Tensor,
    pred_sum:   torch.Tensor,
    target_sum: torch.Tensor,
    voxels:     torch.Tensor,
    smooth:     float = 1e-5,
) -> Dict[str, torch.Tensor]:
    """由 pooled 混淆量闭式导出多维度逐类指标（全部为长度 C 的张量，GPU 上）。

    参数对应 ``dice_batch_stats`` 累加后的总和；``voxels`` 为类共享标量
    （B*spatial_numel 的总和），用于推算 TN。

    返回键：
      * ``dice``       = (2·TP+ε)/(2·TP+FP+FN+ε)        — 与现有 pooled dice 一致。
      * ``iou``        = (TP+ε)/(TP+FP+FN+ε)            — Jaccard，比 Dice 更严格。
      * ``recall``     = (TP+ε)/(TP+FN+ε)               — 灵敏度，反映欠分割。
      * ``precision``  = (TP+ε)/(TP+FP+ε)               — PPV，反映过分割。
      * ``vol_sim``    = 1 − |FP−FN|/(2·TP+FP+FN+ε)     — 与空间重叠解耦的体积一致性。
      * ``mcc``        = (TP·TN−FP·FN)/√((TP+FP)(TP+FN)(TN+FP)(TN+FN)+ε)
                                                          — 类极不平衡下最稳健的单指标，∈[−1,1]。

    所有除法平滑过；分母全 0 的类（既无 GT 又无 pred）返回 0 而非 NaN。
    """
    inter      = inter.double()
    pred_sum   = pred_sum.double()
    target_sum = target_sum.double()
    voxels_d   = voxels.double()

    tp = inter
    fp = (pred_sum - inter).clamp(min=0)
    fn = (target_sum - inter).clamp(min=0)
    tn = (voxels_d - tp - fp - fn).clamp(min=0)

    eps = float(smooth)
    dice      = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou       = (tp + eps) / (tp + fp + fn + eps)
    recall    = (tp + eps) / (tp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    vol_sim   = 1.0 - (fp - fn).abs() / (2.0 * tp + fp + fn + eps)

    # MCC：四个边际全部非零才有定义；任一为零按 0 处理（避免 NaN 影响选模）。
    mcc_num   = tp * tn - fp * fn
    mcc_den2  = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    valid     = mcc_den2 > 0
    mcc       = torch.where(
        valid, mcc_num / torch.sqrt(mcc_den2.clamp(min=eps)),
        torch.zeros_like(mcc_num))

    return {
        "dice"     : dice.float(),
        "iou"      : iou.float(),
        "recall"   : recall.float(),
        "precision": precision.float(),
        "vol_sim"  : vol_sim.float(),
        "mcc"      : mcc.float()}


@torch.no_grad()
def harmonic_mean_metrics(values: List[torch.Tensor], smooth: float = 1e-5) -> torch.Tensor:
    """多个 [0,1] 标量指标的调和均值。任一指标接近 0 会强力拉低结果，
    适合做"短板放大"的综合选模标准。所有输入需为同 device 的 0-d 张量。"""
    if not values:
        return torch.zeros((), dtype=torch.float32)
    stacked = torch.stack([v.float().clamp(min=0.0, max=1.0) for v in values])
    inv_mean = (1.0 / (stacked + smooth)).mean()
    return (1.0 / inv_mean).clamp(min=0.0, max=1.0)


def _binary_erosion_pool(mask: torch.Tensor, ndim: int) -> torch.Tensor:
    """3x3(x3) 二值腐蚀。外侧按背景 0 处理（先 zero-pad 再 maxpool 实现 minpool）。"""
    pad_amt = [1] * (2 * ndim)
    m = F.pad(mask, pad_amt, mode="constant", value=0.0)
    pool = F.max_pool2d if ndim == 2 else F.max_pool3d
    return -pool(-m, kernel_size=3, stride=1, padding=0)


def _binary_dilate_pool(mask: torch.Tensor, ndim: int, tol: int) -> torch.Tensor:
    """Chebyshev-τ 膨胀（kernel=2τ+1 maxpool）。τ=0 直接返回。

    τ≥2 时按轴分离（max 可分离，与全核严格等价）：计算量 k^d → d·k；
    τ=1 小核单次 pool 更快（分离的 kernel-launch/写回开销占主导）。"""
    if tol <= 0:
        return mask
    k = 2 * int(tol) + 1
    pool = F.max_pool2d if ndim == 2 else F.max_pool3d
    if tol < 2:
        return pool(mask, kernel_size=k, stride=1, padding=int(tol))
    out = mask
    for ax in range(ndim):
        ks = [1] * ndim
        pd = [0] * ndim
        ks[ax] = k
        pd[ax] = int(tol)
        out = pool(out, kernel_size=tuple(ks), stride=1, padding=tuple(pd))
    return out


def _nsd_stats_spacing_aware(
    pred_bin: torch.Tensor,
    target_f: torch.Tensor,
    tolerance_mm: float,
    spacing: "Union[float, tuple, list]",
) -> Dict[str, torch.Tensor]:
    """物理空间 (mm) 双向 Normalized Surface Dice 的可池化统计量。

    与 voxel-Chebyshev 版返回同义的 (sd_num, sd_denom, n_with_gt)，但"匹配"
    由各向异性 **欧氏** 表面距离判定：以 ``spacing``（每轴 mm，长度=ndim）为
    采样距离做 EDT，gt/pred 表面体素到对侧表面的最近欧氏距离 <= ``tolerance_mm``
    记为匹配。定义同 MONAI Surface Dice（distance-based, symmetric）：
    NSD[c] = Σ(|{b∈B_p: d(b,B_t)<=τ}| + |{b∈B_t: d(b,B_p)<=τ}|) / Σ(|B_p|+|B_t|)。
    分子分母跨样本/类求和后闭式导出，与单进程全集累加严格相等（可 all-reduce）。"""
    from scipy.ndimage import binary_erosion, distance_transform_edt

    device = pred_bin.device
    B, C = pred_bin.shape[:2]
    ndim = pred_bin.ndim - 2
    sp = ([float(spacing)] * ndim if isinstance(spacing, (int, float))
          else [float(s) for s in spacing])
    if len(sp) != ndim:
        raise ValueError(
            f"spacing length {len(sp)} != spatial rank {ndim}")
    tol = float(tolerance_mm)
    struct = np.ones((3,) * ndim, dtype=bool)   # 全连通（同 GPU Chebyshev 腐蚀）

    pred_np = (pred_bin.detach().cpu().numpy() > 0.5)
    tgt_np = (target_f.detach().cpu().numpy() > 0.5)

    sd_num = np.zeros(C, dtype=np.float64)
    sd_denom = np.zeros(C, dtype=np.float64)
    n_with_gt = np.zeros(C, dtype=np.float64)

    def _boundary(mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return np.zeros_like(mask)
        # border_value=0：卷体外按背景，边缘前景计入表面（同 GPU 版 zero-pad）。
        eroded = binary_erosion(mask, structure=struct, border_value=0)
        return mask & (~eroded)

    for b in range(B):
        for c in range(C):
            pm = pred_np[b, c]
            gm = tgt_np[b, c]
            if gm.any():
                n_with_gt[c] += 1.0
            pb = _boundary(pm)
            tb = _boundary(gm)
            n_pb = int(pb.sum())
            n_tb = int(tb.sum())
            sd_denom[c] += n_pb + n_tb
            if n_pb == 0 or n_tb == 0:
                continue   # 一侧无表面 → 无匹配（分母仍计，惩罚缺失）。
            # EDT(~boundary) 在每个体素给出到最近 boundary 体素的欧氏 mm 距离。
            dt_to_pred = distance_transform_edt(~pb, sampling=sp)
            dt_to_gt = distance_transform_edt(~tb, sampling=sp)
            sd_num[c] += float((dt_to_pred[tb] <= tol).sum())
            sd_num[c] += float((dt_to_gt[pb] <= tol).sum())

    return {
        "sd_num": torch.as_tensor(sd_num, dtype=torch.float32, device=device),
        "sd_denom": torch.as_tensor(
            sd_denom, dtype=torch.float32, device=device),
        "n_with_gt": torch.as_tensor(
            n_with_gt, dtype=torch.float32, device=device)}


@torch.no_grad()
def surface_dice_batch_stats(
    pred: torch.Tensor,
    target: torch.Tensor,
    tolerance: int = 1,
    threshold: Union[float, torch.Tensor] = 0.5,
    pred_is_binary: bool = False,
    tolerance_mm: float = 0.0,
    spacing: Optional[Union[float, tuple, list]] = None,
) -> Dict[str, torch.Tensor]:
    """逐类汇总 (sd_num, sd_denom, n_with_gt)，供 pooled surface-dice：
    SD[c] = Σ(|B_p ∩ Dil_τ(B_t)| + |B_t ∩ Dil_τ(B_p)|) / Σ(|B_p|+|B_t|)。
    支持 2D (B,C,H,W) 与 3D (B,C,D,H,W)；外侧体素按背景计入边界。
    ``pred_is_binary`` 含义同 ``dice_batch_stats``。

    ``tolerance_mm>0`` 且 ``spacing`` 非空时切换到物理空间各向异性 **欧氏** NSD
    （见 :func:`_nsd_stats_spacing_aware`）；否则用 voxel-Chebyshev@``tolerance``。"""
    pred_bin = (pred.float() if pred_is_binary
                else (torch.sigmoid(pred) > threshold).float())
    target_f = target.float()
    ndim = pred_bin.ndim - 2
    if ndim not in (2, 3):
        raise ValueError(
            f"surface_dice expects 2D/3D spatial, got rank {pred_bin.ndim}")

    if tolerance_mm > 0.0 and spacing is not None:
        return _nsd_stats_spacing_aware(
            pred_bin, target_f, tolerance_mm, spacing)

    p_er = _binary_erosion_pool(pred_bin, ndim)
    t_er = _binary_erosion_pool(target_f, ndim)
    pb = pred_bin * (1.0 - p_er)
    tb = target_f * (1.0 - t_er)

    pb_dil = _binary_dilate_pool(pb, ndim, tolerance)
    tb_dil = _binary_dilate_pool(tb, ndim, tolerance)

    spatial_dims = tuple(range(2, pb.ndim))
    reduce_dims = (0,) + spatial_dims

    sd_num = (pb * tb_dil).sum(dim=reduce_dims) + (tb * pb_dil).sum(dim=reduce_dims)
    sd_denom = pb.sum(dim=reduce_dims) + tb.sum(dim=reduce_dims)
    n_with_gt = (target_f.flatten(2).sum(dim=2) > 0).sum(dim=0).float()
    return {"sd_num": sd_num, "sd_denom": sd_denom, "n_with_gt": n_with_gt}
