"""训练工具：AverageMeter、ModelEMA、Timer、dice 指标、随机性。"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)


class AverageMeter:
    """跟踪运行均值。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum   += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


class ModelEMA:
    """参数 EMA，支持原地 apply/restore。

    兼容单卡与 DDP：DDP 反传 all-reduce 后各 rank 参数一致，逐 rank 独立维护
    shadow 数学上等价于单卡。不兼容 FSDP（参数分片，state_dict key/形状不完整）。
    所有方法须传入裸模型（``torch.compile`` 包装前 / ``unwrap_compile`` 后），
    否则 ``_orig_mod.`` 前缀会与 shadow key 不匹配。

    注意：浮点 buffer（含 BatchNorm running_mean/var）也按同一 decay 平滑，
    且无 SWA 式收尾 BN 重校准。默认架构用 instance/group norm（无 running
    stats）不受影响；若引入 BN backbone，建议以 EMA 权重评估前重估 BN 统计。"""

    def __init__(self, model: nn.Module, decay: float = 0.999,
                 warmup: bool = True,
                 offload_device: Optional[str] = None):
        self.decay = decay
        # decay warmup（timm 式）：有效 decay = min(decay, (1+n)/(10+n))，
        # 避免从零训练时 shadow 被随机初始权重长时间拖累。
        self.warmup = warmup
        self.num_updates = 0
        # shadow 存放设备：None = 跟随模型（现状）；"cpu" = 常驻 CPU，省 1×
        # 参数量 GPU 显存（update 时经 pinned staging 异步 D2H + 一次流同步，
        # 数学与跟随模型严格等价）。
        self.offload_device: Optional[torch.device] = (
            torch.device(offload_device) if offload_device else None)
        self.shadow: Dict[str, torch.Tensor] = {
            k: (v.detach().to(self.offload_device, copy=True)
                if self.offload_device is not None else v.detach().clone())
            for k, v in model.state_dict().items()
        }
        self._backup: Dict[str, torch.Tensor] = {}
        self._swapped: bool = False
        # update 热路径缓存：按 dtype 分组的 float 列表 + int 配对，避免每步
        # 逐张量 Python 循环。state_dict 返回参数/buffer 本体引用，同一
        # model 实例下跨 step 稳定。
        self._float_groups: Optional[list] = None
        self._int_pairs: Optional[list] = None
        self._pairs_model_id: Optional[int] = None

    def _build_pairs(self, model: nn.Module) -> None:
        float_groups = {}
        int_pairs = []
        for k, v in model.state_dict().items():
            shadow = self.shadow[k]
            if v.is_floating_point():
                key = (v.device, v.dtype)
                if key not in float_groups:
                    float_groups[key] = ([], [])
                float_groups[key][0].append(shadow)
                float_groups[key][1].append(v)
            else:
                int_pairs.append((shadow, v))
        # CPU offload 且 live 在 CUDA 时：逐组预分配 pinned staging 缓冲，
        # update 时先异步 D2H 到 staging，同步一次后在 CPU 上 foreach 更新。
        groups = []
        for shadow_list, live_list in float_groups.values():
            staging = None
            if (self.offload_device is not None
                    and self.offload_device.type == "cpu"
                    and live_list and live_list[0].is_cuda):
                staging = [
                    torch.empty_like(v, device="cpu", pin_memory=True)
                    for v in live_list]
            groups.append((shadow_list, live_list, staging))
        self._float_groups = groups
        self._int_pairs = int_pairs
        self._pairs_model_id = id(model)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        if self._float_groups is None or self._pairs_model_id != id(model):
            self._build_pairs(model)
        self.num_updates += 1
        decay = self.decay
        if self.warmup:
            decay = min(decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        # 先集中发起全部异步 D2H，再一次同步，避免逐组同步的开销。
        any_staged = False
        for _, live_list, staging in self._float_groups:
            if staging is not None:
                for s, live in zip(staging, live_list):
                    s.copy_(live, non_blocking=True)
                any_staged = True
        if any_staged:
            torch.cuda.current_stream().synchronize()
        for shadow_list, live_list, staging in self._float_groups:
            src = staging if staging is not None else live_list
            torch._foreach_mul_(shadow_list, decay)
            torch._foreach_add_(shadow_list, src, alpha=1.0 - decay)
        for shadow, live in self._int_pairs:
            # 整型 buffer（如 BN num_batches_tracked）直接跟随最新。
            shadow.copy_(live)

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module) -> None:
        """将 shadow 权重换入 model；live 存入 backup 供 restore()。"""
        if self._swapped:
            return
        sd = model.state_dict()
        if not self._backup:
            # offload 模式下 backup 也落 CPU，避免验证换入期间额外占 1× 参数 GPU 显存。
            self._backup = {
                k: (torch.empty_like(v, device=self.offload_device)
                    if self.offload_device is not None else torch.empty_like(v))
                for k, v in sd.items()}
        for k, live in sd.items():
            self._backup[k].copy_(live)
            live.copy_(self.shadow[k])
        self._swapped = True

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if not self._swapped:
            return
        sd = model.state_dict()
        for k, live in sd.items():
            live.copy_(self._backup[k])
        self._swapped = False

    def state_dict(self) -> Dict:
        return {"shadow": self.shadow, "decay": self.decay,
                "warmup": self.warmup, "num_updates": self.num_updates}

    def load_state_dict(self, state: Dict) -> None:
        loaded = state["shadow"]
        if set(loaded.keys()) == set(self.shadow.keys()):
            for k, v in loaded.items():
                self.shadow[k].copy_(v)
        else:
            # key 不一致：从零重建 shadow。
            logger.warning(
                "ModelEMA.load_state_dict: shadow keys mismatch current "
                "model (loaded=%d, current=%d) — rebuilding shadow from "
                "checkpoint. EMA history continuity is preserved only if "
                "the checkpoint matches the intended architecture.",
                len(loaded), len(self.shadow))
            self.shadow = {
                k: (v.detach().to(self.offload_device, copy=True)
                    if self.offload_device is not None else v.detach().clone())
                for k, v in loaded.items()}
            self._backup = {}
            self._swapped = False
        self._float_groups = None
        self._int_pairs = None
        self._pairs_model_id = None
        self.decay = state.get("decay", self.decay)
        self.warmup = state.get("warmup", self.warmup)
        self.num_updates = int(state.get("num_updates", self.num_updates))


class ModelSWA:
    """尾段等权权重平均（SWA, Izmailov 2018），支持原地 apply/restore。

    训练尾段每 epoch ``update`` 一次，对在线权重做等权平均：收敛盆地内多点
    平均落在更平坦区域，泛化通常优于任一单点。与 ModelEMA（指数加权）正交，
    可同时开启。shadow 常驻 CPU 并以 fp32 累积（零 GPU 显存占用、数值稳）。
    DDP 下各 rank 参数一致，逐 rank 独立维护数学等价。所有方法须传入裸模型
    （``unwrap_compile`` 后），否则 ``_orig_mod.`` 前缀会与 shadow key 不匹配。
    权重换入后若模型含 BatchNorm，须重估 running stats（平均权重下激活分布
    改变）；InstanceNorm/GroupNorm 无此需要。"""

    def __init__(self, model: nn.Module):
        self.n_averaged = 0
        # 浮点参数/缓冲以 fp32 在 CPU 累积；整型 buffer（如 BN
        # num_batches_tracked）跟随最新。首次 update 直接覆盖此占位快照。
        self.shadow: Dict[str, torch.Tensor] = {
            k: (v.detach().to("cpu", torch.float32, copy=True)
                if v.is_floating_point()
                else v.detach().to("cpu", copy=True))
            for k, v in model.state_dict().items()}
        self._backup: Dict[str, torch.Tensor] = {}
        self._swapped: bool = False

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """纳入一次快照：avg ← avg + (w − avg)/n（每 epoch 一次，无需热路径优化）。"""
        self.n_averaged += 1
        n = self.n_averaged
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if not v.is_floating_point():
                s.copy_(v.detach())
            elif n == 1:
                s.copy_(v.detach().to(torch.float32))
            else:
                s.add_((v.detach().to("cpu", torch.float32) - s) / n)

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module) -> None:
        """将平均权重换入 model；live 权重存 CPU backup 供 restore()。"""
        if self._swapped:
            return
        sd = model.state_dict()
        if not self._backup:
            self._backup = {k: torch.empty_like(v, device="cpu")
                            for k, v in sd.items()}
        for k, live in sd.items():
            self._backup[k].copy_(live)
            live.copy_(self.shadow[k])  # copy_ 自动转回 live dtype/device
        self._swapped = True

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if not self._swapped:
            return
        for k, live in model.state_dict().items():
            live.copy_(self._backup[k])
        self._swapped = False

    def state_dict(self) -> Dict:
        return {"shadow": self.shadow, "n_averaged": self.n_averaged}

    def load_state_dict(self, state: Dict) -> None:
        loaded = state["shadow"]
        if set(loaded.keys()) == set(self.shadow.keys()):
            for k, v in loaded.items():
                self.shadow[k].copy_(v)
        else:
            logger.warning(
                "ModelSWA.load_state_dict: shadow keys mismatch current "
                "model (loaded=%d, current=%d) — rebuilding shadow from "
                "checkpoint.", len(loaded), len(self.shadow))
            self.shadow = {k: v.detach().to("cpu", copy=True)
                           for k, v in loaded.items()}
        self.n_averaged = int(state.get("n_averaged", 0))
        self._backup = {}
        self._swapped = False


class Timer:
    """计时器。"""

    def __init__(self):
        self.start = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start

    def elapsed_str(self) -> str:
        s = int(self.elapsed())
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


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


@torch.no_grad()
def surface_dice_batch_stats(
    pred: torch.Tensor,
    target: torch.Tensor,
    tolerance: int = 1,
    threshold: Union[float, torch.Tensor] = 0.5,
    pred_is_binary: bool = False,
) -> Dict[str, torch.Tensor]:
    """逐类汇总 (sd_num, sd_denom, n_with_gt)，供 pooled surface-dice@τ：
    SD[c] = Σ(|B_p ∩ Dil_τ(B_t)| + |B_t ∩ Dil_τ(B_p)|) / Σ(|B_p|+|B_t|)。
    支持 2D (B,C,H,W) 与 3D (B,C,D,H,W)；外侧体素按背景计入边界。
    ``pred_is_binary`` 含义同 ``dice_batch_stats``。"""
    pred_bin = (pred.float() if pred_is_binary
                else (torch.sigmoid(pred) > threshold).float())
    target_f = target.float()
    ndim = pred_bin.ndim - 2
    assert ndim in (2, 3), f"surface_dice expects 2D/3D spatial, got rank {pred_bin.ndim}"

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


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """设置随机种子。deterministic=True 强制 cudnn deterministic（较慢）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Ampere+ 上对残余 fp32 matmul/conv 免费加速；deterministic 下关闭以保证可复现。
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    logger.info("Seed set to %d (deterministic=%s)", seed, deterministic)
