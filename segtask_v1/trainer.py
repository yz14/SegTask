"""Training pipeline for 3D segmentation.

Handles:
- Mixed precision (AMP, fp16 + bf16) with scaler disabled in bf16
- EMA with context-manager-based swap (exception-safe)
- Learning rate scheduling with warmup (step-aligned with base scheduler)
- Gradient clipping + gradient accumulation (partial-tail corrected)
- torch.compile acceleration (state_dict unwrapping on save / load)
- Validation and per-class Dice tracking (DS-safe loss path)
- Full-state checkpointing (model/ema/optimizer/scheduler/scaler/early-stop)
- Early stopping
- GPU data augmentation
"""

from __future__ import annotations

import logging
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
# PyTorch ≥ 2.3 使用 torch.amp.GradScaler（首位参数为 device）；旧版在 torch.cuda.amp 不接受该参数。探测后选用。
import inspect as _inspect
try:
    from torch.amp import GradScaler as _GradScaler  # type: ignore
    from torch.amp import autocast  # type: ignore
except ImportError:  # pragma: no cover
    from torch.cuda.amp import GradScaler as _GradScaler  # type: ignore
    from torch.amp import autocast  # type: ignore


def GradScaler(device: str = "cuda", **kwargs):  # noqa: N802
    """版本无关的 GradScaler 构造：仅在新 API 接受 ``device`` 时传入。"""
    try:
        params = _inspect.signature(_GradScaler).parameters
    except (TypeError, ValueError):
        params = {}
    if "device" in params:
        return _GradScaler(device, **kwargs)
    return _GradScaler(**kwargs)
from torch.utils.data import DataLoader

from .config import Config
from .data.augment import GPUAugmentor
from .losses.losses import (
    build_loss, DeepSupervisionLoss, MultiResolutionLoss, SliceChannelLoss)
from .models.unet import UNet3D
from .utils import (
    AverageMeter, ModelEMA, Timer,
    compute_dice_per_class, dice_batch_stats,
)

logger = logging.getLogger(__name__)


_AMP_DTYPES = {
    "float16": torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _unwrap_compile(m: nn.Module) -> nn.Module:
    """剥去 torch.compile 加的 ``_orig_mod`` 包装，使 state_dict 键不带 ``_orig_mod.`` 前缀。"""
    return getattr(m, "_orig_mod", m)


def _cuda_supports_bf16() -> bool:
    """当前 CUDA 设备是否原生支持 bf16 autocast（Ampere+ 或 ROCm）。"""
    if not torch.cuda.is_available():
        return False
    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(is_bf16_supported):
        try:
            return bool(is_bf16_supported())
        except Exception:  # pragma: no cover - defensive
            pass
    try:
        major, _minor = torch.cuda.get_device_capability()
        return major >= 8
    except Exception:  # pragma: no cover - defensive
        return False


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------
def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    tc = cfg.train
    params = [p for p in model.parameters() if p.requires_grad]
    if tc.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=tc.lr, weight_decay=tc.weight_decay)
    elif tc.optimizer == "adam":
        return torch.optim.Adam(params, lr=tc.lr, weight_decay=tc.weight_decay)
    elif tc.optimizer == "sgd":
        return torch.optim.SGD(
            params, lr=tc.lr, weight_decay=tc.weight_decay,
            momentum=tc.momentum, nesterov=tc.nesterov,
        )
    raise ValueError(f"Unknown optimizer: {tc.optimizer}")


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------
def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    steps_per_epoch: int,
    post_warmup_steps: int,
):
    """Build the base LR scheduler that runs AFTER warmup.

    `post_warmup_steps` is the number of optimizer steps the base scheduler
    will actually see, so `T_max` / poly's horizon / step milestones are
    aligned with the warmup-excluded segment of training.
    """
    tc = cfg.train
    horizon = max(post_warmup_steps, 1)

    if tc.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=horizon, eta_min=tc.cosine_min_lr)
    elif tc.scheduler == "poly":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: (1 - step / horizon) ** tc.poly_power)
    elif tc.scheduler == "step":
        milestones = list(range(
            tc.step_size * steps_per_epoch, horizon,
            tc.step_size * steps_per_epoch))
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=tc.step_gamma)
    elif tc.scheduler == "plateau":
        # Plateau 方向跟随 save_best_mode，以在真实优化目标停滞时降 LR。
        plateau_mode = tc.save_best_mode if tc.save_best_mode in ("max", "min") else "max"
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=plateau_mode, patience=tc.plateau_patience,
            factor=tc.plateau_factor)
    elif tc.scheduler == "cosine_warm_restarts":
        T_0 = tc.cosine_restart_period * steps_per_epoch
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(T_0, 1), T_mult=tc.cosine_restart_mult,
            eta_min=tc.cosine_min_lr)
    elif tc.scheduler == "one_cycle":
        # OneCycleLR 自带 warmup（pct_start）；Trainer 拒绝与 WarmupScheduler 叠加。
        total_steps = tc.epochs * steps_per_epoch
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=tc.lr, total_steps=total_steps,
            pct_start=max(tc.warmup_epochs, 1) / max(tc.epochs, 1))
    raise ValueError(f"Unknown scheduler: {tc.scheduler}")


# ---------------------------------------------------------------------------
# Warmup wrapper
# ---------------------------------------------------------------------------
class WarmupScheduler:
    """线性 warmup 后委托 base scheduler。

    Warmup 期：LR 从 warmup_lr 线性上升到 base_lr（base scheduler 不动）。
    Warmup 后：base scheduler 驱动。Plateau 逐 epoch step，其余逐 step 动。
    horizon 需以 ``post_warmup_steps`` 构建（见 build_scheduler）。
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler,
        warmup_steps: int,
        warmup_lr: float,
        base_lr: float,
    ):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr
        self.base_lr = base_lr
        self.current_step = 0
        self._is_plateau = isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        if warmup_steps > 0:
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

    def step(self) -> None:
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            alpha = self.current_step / max(self.warmup_steps, 1)
            lr = self.warmup_lr + alpha * (self.base_lr - self.warmup_lr)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        elif self.scheduler is not None and not self._is_plateau:
            self.scheduler.step()

    def step_epoch(self, metric: Optional[float] = None) -> None:
        if (self._is_plateau
                and self.scheduler is not None
                and self.current_step > self.warmup_steps
                and metric is not None):
            self.scheduler.step(metric)

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self) -> Dict:
        # 同时持久化 warmup 参数，以便 load 时检出配置漂移。
        return {
            "current_step": self.current_step,
            "warmup_steps": self.warmup_steps,
            "warmup_lr": self.warmup_lr,
            "base_lr": self.base_lr,
            "base_scheduler": (self.scheduler.state_dict()
                               if self.scheduler is not None else None),
        }

    def load_state_dict(self, state: Dict) -> None:
        ckpt_warmup_steps = state.get("warmup_steps")
        ckpt_warmup_lr = state.get("warmup_lr")
        ckpt_base_lr = state.get("base_lr")
        # warmup 参数漂移会导致 current_step 套入不同 shape，响亮警告。
        mismatches = []
        if (ckpt_warmup_steps is not None
                and int(ckpt_warmup_steps) != int(self.warmup_steps)):
            mismatches.append(
                f"warmup_steps: ckpt={ckpt_warmup_steps} vs cfg={self.warmup_steps}")
        if ckpt_warmup_lr is not None and float(ckpt_warmup_lr) != float(self.warmup_lr):
            mismatches.append(
                f"warmup_lr: ckpt={ckpt_warmup_lr} vs cfg={self.warmup_lr}")
        if ckpt_base_lr is not None and float(ckpt_base_lr) != float(self.base_lr):
            mismatches.append(
                f"base_lr: ckpt={ckpt_base_lr} vs cfg={self.base_lr}")
        if mismatches:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "Warmup config drift on resume (%s); current_step restored "
                "but schedule shape differs.", "; ".join(mismatches))

        self.current_step = int(state.get("current_step", 0))
        base_state = state.get("base_scheduler", None)
        if base_state is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(base_state)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    """Full training pipeline for 3D segmentation."""

    def __init__(
        self,
        model       : UNet3D,
        cfg         : Config,
        train_loader: DataLoader,
        val_loader  : DataLoader,
        device      : torch.device):
        self.cfg          = cfg
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        tc                = cfg.train

        # 设备放置优先：optimizer/EMA 需绑定已迁移参数；torch.compile 放到最后。
        self.model = model.to(device)

        # --- Loss ------------------------------------------------------
        # base_loss 独立保留供验证使用；训练时 criterion 再包 DeepSupervisionLoss / MultiResolutionLoss。
        self.base_loss = build_loss(cfg.loss)
        self.is_2_5d   = cfg.data.patch_mode == "2_5d"
        # Plan A lift：2.5D 走真 3D 形状 (B, num_fg, D, H, W)，D 作为空间轴。
        # 与 aux_keep_native_d 互斥；可与 aux_seg_supervision 合成。
        self.lift_2_5d_to_3d = bool(
            getattr(cfg.model, "lift_2_5d_to_3d", False) and self.is_2_5d)

        # 提前决定 aux_keep_native_d，以便下面按视图构造 SliceChannelLoss。
        # aux_view_depths 与 target_patch_size 在下方裁剪部分最终设定。
        self.aux_keep_native_d = bool(
            getattr(cfg.data, "aux_keep_native_d", False)
            and self.is_2_5d
            and len(cfg.data.multi_res_scales) > 1)
        self.aux_view_depths: List[int] = (
            list(cfg.aux_view_depths) if self.aux_keep_native_d else [])

        # keep_native_multi_res：aux_keep_native_d 的 3D 对应 (z_axis/cubic)。
        # dataset 发单 max-FOV cube；trainer 逐视图中心裁并 resize 回 patch_size。
        self.keep_native_multi_res = bool(
            getattr(cfg.data, "keep_native_multi_res", False)
            and not self.is_2_5d
            and cfg.data.patch_mode in ("z_axis", "cubic")
            and len(cfg.data.multi_res_scales) > 1)
        if self.keep_native_multi_res:
            # 预算逐视图原生尺寸 (D_k,H_k,W_k)；z_axis 只缩 D，cubic 缩 3 轴。
            pD, pH, pW = (int(x) for x in cfg.data.patch_size)
            self._mr_native_sizes: List[Tuple[int, int, int]] = []
            for s in cfg.data.multi_res_scales:
                D_k = int(round(pD * float(s)))
                if cfg.data.patch_mode == "z_axis":
                    H_k, W_k = pH, pW
                else:  # cubic
                    H_k = int(round(pH * float(s)))
                    W_k = int(round(pW * float(s)))
                self._mr_native_sizes.append((D_k, H_k, W_k))
            # view 0 强制对齐 patch_size，防浮点漂移。
            self._mr_native_sizes[0] = (pD, pH, pW)
        else:
            self._mr_native_sizes = []

        # 损失复合：INNER = MultiResolutionLoss 或 SliceChannelLoss；OUTER = DeepSupervisionLoss(INNER)。
        # 3D: pred (B, num_fg*C_res, ...)；2.5D: pred (B, num_fg*D, H, W) + label (B, D, H, W)。
        if self.is_2_5d and not self.lift_2_5d_to_3d:
            num_slices = int(cfg.data.patch_size[0])
            inner = SliceChannelLoss(
                base_loss=self.base_loss,
                num_fg_classes=cfg.num_fg_classes,
                num_slices=num_slices,
                label_values=cfg.data.label_values,
                reduction=cfg.loss.slice_loss_reduction,
            )
            num_res = 1   # for logging only; SliceChannelLoss has C_res==1
            logger.info(
                "Loss: %s [2.5D, reduction=%s], num_slices=%d, fg_classes=%d",
                cfg.loss.name, cfg.loss.slice_loss_reduction,
                num_slices, cfg.num_fg_classes)
        else:
            # 3D 或 lifted-2.5D：shape 合同 (B, num_fg*C_res, D, H, W)。
            # lift 强制 num_res=1（aux 视图不作主监督目标）；3D 按 multi_res_scales 长度。
            if self.lift_2_5d_to_3d:
                num_res = 1
            else:
                num_res = len(cfg.data.multi_res_scales)
            inner = MultiResolutionLoss(
                base_loss=self.base_loss,
                num_fg_classes=cfg.num_fg_classes,
                num_res=num_res,
                label_values=cfg.data.label_values,
            )
            logger.info(
                "Loss: %s, scales=%d, fg_classes=%d%s",
                cfg.loss.name, num_res, cfg.num_fg_classes,
                " [2.5D LIFTED to 3D]" if self.lift_2_5d_to_3d else "")

        if cfg.model.deep_supervision and cfg.loss.deep_supervision_weights:
            self.criterion = DeepSupervisionLoss(
                inner, cfg.loss.deep_supervision_weights)
        else:
            self.criterion = inner
        # 保留 INNER 句柄供指标 reshape（split_for_metrics 统一 3D / 2.5D 路径）。
        self._inner_loss = inner

        # ---- 2.5D 多 FOV aux 分割监督：仅 2.5D + n_views>1 + opt-in 时启用 ----
        n_views_data = len(cfg.data.multi_res_scales)
        self.aux_seg_supervision = bool(
            getattr(cfg.model, "aux_seg_supervision", False)
            and self.is_2_5d
            and n_views_data > 1)
        if self.aux_seg_supervision:
            n_aux = n_views_data - 1
            user_w = list(getattr(cfg.loss, "aux_supervision_weights", []))
            if not user_w:
                # 几何衰减：更宽 FOV 对齐较差，权重越小。
                user_w = [0.5 ** (k + 1) for k in range(n_aux)]
            elif len(user_w) != n_aux:
                # 防手造配置绕过 YAML；快速失败。
                raise ValueError(
                    f"loss.aux_supervision_weights length ({len(user_w)}) "
                    f"must equal n_views-1 ({n_aux}); got {user_w}.")
            self.aux_weights = [float(w) for w in user_w]
            # Aux 均为单分辨率逐视图损失（不走 DS）。两种布局：
            #   - aux_keep_native_d=False：所有 view num_slices=D，共享单 SliceChannelLoss。
            #   - aux_keep_native_d=True ：view k 的 num_slices=D_k，逐视图独立构造。
            if self.lift_2_5d_to_3d:
                # Lift+aux：aux 头发 (B, num_fg, D, H, W)；走 MultiResolutionLoss(num_res=1)。
                self.aux_inner_loss = MultiResolutionLoss(
                    base_loss=self.base_loss,
                    num_fg_classes=cfg.num_fg_classes,
                    num_res=1,
                    label_values=cfg.data.label_values,
                )
                self.aux_inner_losses = None
                logger.info(
                    "Aux seg supervision: ENABLED [LIFT], n_aux_views=%d, "
                    "weights=%s, fusion=%s",
                    n_aux, self.aux_weights, cfg.model.context_fusion)
            elif getattr(self, "aux_keep_native_d", False):
                # 逐视图 SliceChannelLoss（num_slices=D_k）。
                aux_depths = self.aux_view_depths[1:]  # skip view 0
                assert len(aux_depths) == n_aux, (
                    f"aux_view_depths excluding view 0 has length "
                    f"{len(aux_depths)}; expected {n_aux}.")
                self.aux_inner_loss = None
                self.aux_inner_losses = [
                    SliceChannelLoss(
                        base_loss=self.base_loss,
                        num_fg_classes=cfg.num_fg_classes,
                        num_slices=int(d_k),
                        label_values=cfg.data.label_values,
                        reduction=cfg.loss.slice_loss_reduction,
                    )
                    for d_k in aux_depths
                ]
                logger.info(
                    "Aux seg supervision: ENABLED (native depth), "
                    "n_aux_views=%d, per-view depths=%s, weights=%s, "
                    "fusion=%s",
                    n_aux, aux_depths, self.aux_weights,
                    cfg.model.context_fusion)
            else:
                self.aux_inner_loss = SliceChannelLoss(
                    base_loss=self.base_loss,
                    num_fg_classes=cfg.num_fg_classes,
                    num_slices=int(cfg.data.patch_size[0]),
                    label_values=cfg.data.label_values,
                    reduction=cfg.loss.slice_loss_reduction,
                )
                self.aux_inner_losses = None
                logger.info(
                    "Aux seg supervision: ENABLED, n_aux_views=%d, weights=%s, "
                    "fusion=%s",
                    n_aux, self.aux_weights, cfg.model.context_fusion)
        else:
            self.aux_weights = []
            self.aux_inner_loss = None
            self.aux_inner_losses = None

        # --- Optimizer + scheduler ------------------------------------
        self.optimizer  = build_optimizer(self.model, cfg)
        steps_per_epoch = len(train_loader)
        warmup_steps    = tc.warmup_epochs * steps_per_epoch
        total_steps     = tc.epochs * steps_per_epoch
        post_warmup     = total_steps - warmup_steps

        # OneCycleLR 自带 warmup，与 WarmupScheduler 叠加会双 warmup 且 total_steps 失准。
        if tc.scheduler == "one_cycle" and warmup_steps > 0:
            raise ValueError(
                "OneCycleLR has built-in warmup (pct_start). "
                "Set train.warmup_epochs=0 when using scheduler='one_cycle'.")

        base_scheduler = build_scheduler(
            self.optimizer, cfg, steps_per_epoch,
            post_warmup_steps=post_warmup)
        self.scheduler = WarmupScheduler(
            self.optimizer, base_scheduler,
            warmup_steps=warmup_steps,
            warmup_lr=tc.warmup_lr, base_lr=tc.lr)

        # --- AMP -------------------------------------------------------
        # "auto"：设备原生支持 bf16 则选 bf16，否则 fp16。
        amp_dtype_cfg = tc.amp_dtype
        if amp_dtype_cfg == "auto":
            amp_dtype_cfg = self._resolve_auto_amp_dtype(device)
            logger.info(
                "amp_dtype='auto' resolved to %r (device=%s).",
                amp_dtype_cfg, device)
        if amp_dtype_cfg not in _AMP_DTYPES:
            raise ValueError(
                f"Unknown amp_dtype: {tc.amp_dtype!r}. "
                f"Expected one of {sorted(_AMP_DTYPES) + ['auto']}.")
        self.amp_dtype = _AMP_DTYPES[amp_dtype_cfg]
        self._amp_dtype_name = amp_dtype_cfg
        self.use_amp = tc.use_amp and device.type == "cuda"
        # GradScaler 仅 fp16 需要；bf16 不需 loss scaling。
        self._scaler_active = self.use_amp and self.amp_dtype == torch.float16
        self.scaler = GradScaler("cuda", enabled=self._scaler_active)

        # --- EMA (bind to placed, not-yet-compiled model) -------------
        self.ema = ModelEMA(self.model, tc.ema_decay) if tc.use_ema else None

        # --- torch.compile (最后) -------------------------------------
        # Inductor CUDA 后端需 Triton（Windows 无官方轮子）；提前探测以避免首次 forward 报错。
        self._compile_enabled = False
        # 首次 step 完整后一次性记录 GPU 峰值（仅 CUDA）。
        self._first_step_mem_logged = False
        if tc.compile_mode != "none" and hasattr(torch, "compile"):
            triton_ok = True
            if device.type == "cuda":
                import importlib.util
                if importlib.util.find_spec("triton") is None:
                    triton_ok = False
                    logger.warning(
                        "torch.compile (mode='%s') requested but Triton not installed; "
                        "falling back to eager. Install Triton or set compile_mode='none'.",
                        tc.compile_mode,
                    )
            if triton_ok:
                logger.info("Compiling model with mode='%s'", tc.compile_mode)
                self.model = torch.compile(self.model, mode=tc.compile_mode)
                self._compile_enabled = True

        # --- 增强 ---------------------------------------------
        # 对 image/label/wmap 同步空间变换；label 近邻，wmap 按 cfg.augment.wmap_interp_mode。
        # 传入 max_scale 以保持弹性形变物理一致（单分辨率时与旧行为一致）。
        _scales = cfg.data.multi_res_scales or [1.0]
        self.augmentor = GPUAugmentor(cfg.augment, max_scale=max(_scales))

        # --- 裁剪（过采样 patch）---------------------------
        # z_axis/cubic 遵 aug_oversample_ratio：dataset 发超尺寸 patch，增强后中心裁回 patch_size。
        # aux_keep_native_d (2.5D)：dataset 发 max-FOV cube，裁剪保留整个 max-FOV，aux 视图需更宽 z。
        if self.aux_keep_native_d:
            max_scale = max(cfg.data.multi_res_scales)
            target_d_native = int(round(int(cfg.data.patch_size[0]) * max_scale))
            self.target_patch_size = (target_d_native,
                                      int(cfg.data.patch_size[1]),
                                      int(cfg.data.patch_size[2]))
            assert self.aux_view_depths[0] == int(cfg.data.patch_size[0]), (
                "aux_view_depths[0] must equal patch_size[0]; got "
                f"{self.aux_view_depths[0]} vs {cfg.data.patch_size[0]}.")
            assert sum(self.aux_view_depths) == int(cfg.model.in_channels), (
                f"sum(aux_view_depths)={sum(self.aux_view_depths)} must "
                f"equal model.in_channels={cfg.model.in_channels}.")
            logger.info(
                "Trainer aux_keep_native_d=True: max-FOV crop D=%d, "
                "per-view depths=%s, channel layout sum=%d.",
                target_d_native, self.aux_view_depths,
                int(cfg.model.in_channels))
        elif self.keep_native_multi_res:
            # 3D 懒抽取：dataset 发 max-FOV cube，裁剪后仍保留全 max-FOV，逐视图裁并 resize 到 patch_size。
            # z_axis 仅缩 z，cubic 缩 3 轴。
            max_scale = max(cfg.data.multi_res_scales)
            pD, pH, pW = (int(x) for x in cfg.data.patch_size)
            if cfg.data.patch_mode == "z_axis":
                self.target_patch_size = (
                    int(round(pD * max_scale)), pH, pW)
            else:  # cubic
                self.target_patch_size = (
                    int(round(pD * max_scale)),
                    int(round(pH * max_scale)),
                    int(round(pW * max_scale)))
            # 交叉检查：逐视图原生尺寸必须全部不超过 max-FOV 目标。
            for k, (D_k, H_k, W_k) in enumerate(self._mr_native_sizes):
                tD, tH, tW = self.target_patch_size
                if D_k > tD or H_k > tH or W_k > tW:
                    raise ValueError(
                        f"keep_native_multi_res: view {k} native size "
                        f"({D_k},{H_k},{W_k}) exceeds max-FOV target "
                        f"{self.target_patch_size}. Check multi_res_scales / "
                        "patch_size for floating-point drift.")
            logger.info(
                "Trainer keep_native_multi_res=True (%s): max-FOV crop "
                "target=%s, per-view native sizes=%s, n_views=%d.",
                cfg.data.patch_mode, self.target_patch_size,
                self._mr_native_sizes, len(cfg.data.multi_res_scales))
        else:
            self.target_patch_size = tuple(cfg.data.patch_size)
        self.needs_crop = cfg.data.aug_oversample_ratio > 1.0

        # --- Gradient accumulation ------------------------------------
        self.grad_accum_steps = max(tc.grad_accum_steps, 1)

        # --- Tracking --------------------------------------------------
        self.num_fg = cfg.num_fg_classes
        self._best_mode = tc.save_best_mode  # "max" or "min"
        self.best_metric: float = (
            -math.inf if self._best_mode == "max" else math.inf)
        self.has_best = False
        self.best_epoch = 0
        self.start_epoch = 0
        self.patience_counter = 0

        # --- Output directory -----------------------------------------
        self.output_dir = Path(tc.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- Resume / Pretrain ----------------------------------------
        # resume = 完整状态恢复；pretrain = 仅加载 model 权重。两者同设且文件存在时 resume 优先。
        resume_active = bool(tc.resume) and os.path.isfile(tc.resume)
        pretrain_active = bool(tc.pretrain) and os.path.isfile(tc.pretrain)

        if resume_active:
            if tc.pretrain:
                logger.warning(
                    "Both `train.resume` and `train.pretrain` are set; "
                    "using resume (%s). Pretrain weights from %s are ignored.",
                    tc.resume, tc.pretrain)
            self._load_checkpoint(tc.resume)
        elif pretrain_active:
            self._load_pretrain(
                tc.pretrain,
                strict=tc.pretrain_strict,
                load_ema=tc.pretrain_load_ema)
        else:
            # 路径误配时提前警告，不静默忽略。
            if tc.resume and not os.path.isfile(tc.resume):
                logger.warning(
                    "`train.resume` is set but file not found: %s. "
                    "Training will start from scratch.", tc.resume)
            if tc.pretrain and not os.path.isfile(tc.pretrain):
                logger.warning(
                    "`train.pretrain` is set but file not found: %s. "
                    "Training will start from scratch.", tc.pretrain)

    # ------------------------------------------------------------------
    # AMP helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_auto_amp_dtype(device: torch.device) -> str:
        """将 ``amp_dtype="auto"`` 解析为 "bfloat16"（CUDA 原生支持时）或 "float16"。"""
        if device.type == "cuda" and _cuda_supports_bf16():
            return "bfloat16"
        return "float16"

    # ------------------------------------------------------------------
    # Memory accounting
    # ------------------------------------------------------------------
    def _estimate_train_memory(self) -> Dict[str, float]:
        """静态估计训练侧持久 GPU 内存（MiB）：包含 params/grads/optim_state/EMA。

        不含激活/cuDNN workspace/dataloader staging。真实峰值见 fit() 逐 epoch 的 'GPU peak'。
        Optimizer mult 推断：Adam 系=2、SGD(momentum)=1、Lion=1、未知默认=2。
        """
        MIB = 1 << 20
        params = list(self.model.parameters())

        param_bytes = sum(p.numel() * p.element_size() for p in params)
        grad_bytes = sum(p.numel() * p.element_size()
                         for p in params if p.requires_grad)

        optim_name = type(self.optimizer).__name__
        n_train = sum(p.numel() for p in params if p.requires_grad)
        adam_family = {"Adam", "AdamW", "RAdam", "NAdam", "Adamax"}
        if optim_name in adam_family:
            optim_mult = 2
        elif optim_name == "SGD":
            has_momentum = any(g.get("momentum", 0) > 0
                               for g in self.optimizer.param_groups)
            optim_mult = 1 if has_momentum else 0
        elif optim_name == "Lion":
            optim_mult = 1
        else:
            optim_mult = 2  # 保守默认
        optim_bytes = optim_mult * n_train * 4  # fp32

        ema_bytes = 0
        if self.ema is not None:
            ema_bytes = sum(t.numel() * t.element_size()
                            for t in self.ema.shadow.values())

        persistent = param_bytes + grad_bytes + optim_bytes + ema_bytes
        return {
            "param_mib": param_bytes / MIB,
            "grad_mib": grad_bytes / MIB,
            "optim_mib": optim_bytes / MIB,
            "optim_mult": optim_mult,
            "optim_name": optim_name,
            "ema_mib": ema_bytes / MIB,
            "persistent_mib": persistent / MIB,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self) -> Dict[str, float]:
        """Run the full training loop. Returns best validation metrics."""
        tc    = self.cfg.train
        timer = Timer()

        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info("=" * 60)
        logger.info("Training: %d epochs, device=%s", tc.epochs, self.device)
        logger.info("Model params: %.2fM", total_params)
        mem = self._estimate_train_memory()
        ema_part = (f" + ema={mem['ema_mib']:.1f}" if mem["ema_mib"] > 0 else "")
        logger.info(
            "Static GPU mem (persistent, excl. activations): "
            "param=%.1f + grad=%.1f + optim(%s,%dx)=%.1f%s "
            "= %.1f MiB (real peak reported per-epoch as 'GPU peak')",
            mem["param_mib"], mem["grad_mib"],
            mem["optim_name"], mem["optim_mult"], mem["optim_mib"],
            ema_part, mem["persistent_mib"])
        if self.device.type == "cuda":
            cur_alloc  = torch.cuda.memory_allocated(self.device) / (1 << 20)
            cur_reserv = torch.cuda.memory_reserved(self.device) / (1 << 20)
            logger.info(
                "CUDA mem at training start: allocated=%.1f MiB, "
                "reserved=%.1f MiB (model already on device; "
                "activations/workspace will add on top during forward).",
                cur_alloc, cur_reserv)
        logger.info("Train batches: %d, Val batches: %d",
                    len(self.train_loader), len(self.val_loader))
        logger.info("AMP=%s (dtype=%s, resolved=%s, scaler=%s), "
                    "EMA=%s (decay=%.4f)",
                    self.use_amp, tc.amp_dtype, self._amp_dtype_name,
                    self._scaler_active, tc.use_ema, tc.ema_decay)
        logger.info("Grad accum=%d, Effective batch=%d",
                    self.grad_accum_steps,
                    self.cfg.data.batch_size * self.grad_accum_steps)
        logger.info("Foreground classes: %d, Loss: %s",
                    self.num_fg, self.cfg.loss.name)
        if tc.compile_mode != "none":
            logger.info(
                "torch.compile mode: %s (active=%s)",
                tc.compile_mode, self._compile_enabled)
        logger.info("=" * 60)

        best_metrics: Dict[str, float] = {}

        for epoch in range(self.start_epoch, tc.epochs):
            train_metrics = self._train_epoch(epoch)

            val_metrics: Dict[str, float] = {}
            if (epoch + 1) % tc.val_every == 0 or epoch == tc.epochs - 1:
                val_metrics = self._validate(epoch)

            # Plateau is the only base scheduler driven per-epoch.
            plateau_metric = val_metrics.get(tc.save_best_metric, None)
            self.scheduler.step_epoch(metric=plateau_metric)

            # --- Best-checkpoint decision (no magic >0 guard) ----------
            is_best = False
            if tc.save_best_metric in val_metrics:
                tracked = val_metrics[tc.save_best_metric]
                if not self.has_best:
                    is_best = True
                elif self._best_mode == "max":
                    is_best = tracked > self.best_metric
                else:
                    is_best = tracked < self.best_metric

                if is_best:
                    self.best_metric = tracked
                    self.best_epoch = epoch
                    self.has_best = True
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, is_best=True)
                    best_metrics = val_metrics
                    logger.info("★ New best: %s=%.4f at epoch %d",
                                tc.save_best_metric, tracked, epoch + 1)
                else:
                    self.patience_counter += 1

            # --- Epoch summary ----------------------------------------
            best_str = (f"{self.best_metric:.4f} (ep{self.best_epoch + 1})"
                        if self.has_best else "n/a")
            # Aggregate the aux component averages into a compact suffix
            # using the same renderer as the per-step debug line. Empty
            # for non-aux runs → epoch line is bit-identical to legacy.
            aux_summary_dict = {
                k: v for k, v in train_metrics.items()
                if k.startswith("L_main") or k.startswith("L_aux_")
                or k.startswith("w_aux_")
            }
            aux_msg = self._format_breakdown(aux_summary_dict)
            logger.info(
                "Epoch %d/%d | LR=%.2e | loss=%.4f | val_dice=%.4f | "
                "best=%s | %s%s",
                epoch + 1, tc.epochs, self.scheduler.get_lr(),
                train_metrics.get("loss", 0.0),
                val_metrics.get("mean_dice", 0.0),
                best_str,
                timer.elapsed_str(),
                aux_msg,
            )
            # Per-epoch peak GPU memory. Logged on a SEPARATE line so the
            # legacy "Epoch X/Y | ..." log regex stays bit-compatible with
            # downstream parsers; new aggregators (e.g. lift_a) can pick
            # this up to compare displays across runs. Reset the peak each
            # epoch so the value is per-epoch, not cumulative — the run-
            # level peak is then ``max`` across all epoch lines.
            if self.device.type == "cuda":
                peak_mib = torch.cuda.max_memory_allocated(self.device) / (1 << 20)
                logger.info("  GPU peak (epoch %d): %.1f MiB", epoch + 1, peak_mib)
                torch.cuda.reset_peak_memory_stats(self.device)

            # --- Periodic checkpoint ----------------------------------
            if (epoch + 1) % tc.save_every == 0:
                self._save_checkpoint(epoch, is_best=False)

            # --- Early stopping ---------------------------------------
            if tc.early_stopping > 0 and self.patience_counter >= tc.early_stopping:
                logger.info("Early stopping at epoch %d (patience=%d)",
                            epoch + 1, tc.early_stopping)
                break

        logger.info("=" * 60)
        if self.has_best:
            logger.info(
                "Training complete. Best %s=%.4f at epoch %d. Time: %s",
                tc.save_best_metric, self.best_metric,
                self.best_epoch + 1, timer.elapsed_str())
        else:
            logger.info("Training complete. No validation best recorded. "
                        "Time: %s", timer.elapsed_str())
        logger.info("=" * 60)
        return best_metrics

    # ------------------------------------------------------------------
    # EMA swap helper (exception-safe)
    # ------------------------------------------------------------------
    @contextmanager
    def _ema_swapped(self) -> Iterator[None]:
        """Temporarily swap EMA weights into the model. `try/finally`
        guarantees the online weights are restored even if the enclosed
        block raises — without this, an OOM during validation would leave
        the trainer running on EMA weights for the rest of training."""
        if self.ema is None:
            yield
            return
        self.ema.apply_shadow(self.model)
        try:
            yield
        finally:
            self.ema.restore(self.model)

    # ------------------------------------------------------------------
    # Training / validation loops
    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with optional gradient accumulation."""
        self.model.train()
        loss_meter = AverageMeter()
        dice_meter = AverageMeter()
        # Per-component meters for the multi-FOV aux supervision diagnostic
        # log. Populated lazily on the first batch (we don't know the aux
        # view names until ``_compute_loss_aux_fp32`` runs once); from then
        # on each component is averaged across the epoch independently.
        # Keys: "L_main", "L_aux_1", "L_aux_2", ...
        component_meters: Dict[str, AverageMeter] = {}
        tc    = self.cfg.train
        accum = self.grad_accum_steps

        total_steps = len(self.train_loader)
        # Any steps beyond `partial_start` belong to a partial accumulation
        # tail (len(loader) not divisible by accum). Divide those by the
        # real tail length so the effective LR doesn't shrink on them.
        remainder     = total_steps % accum if accum > 1 else 0
        partial_start = total_steps - remainder

        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(self.train_loader):
            image = batch["image"].to(self.device, non_blocking=True)
            # P8: the dataset emits ``label`` as int16 to halve CPU→GPU
            # bandwidth. Cast to float32 on the GPU immediately so the
            # augmentor (``F.grid_sample`` requires float input) and the
            # loss stack (``SliceChannelLoss`` / ``MultiResolutionLoss``)
            # see the same float32 tensor they have always seen. The
            # ``.float()`` op is a no-op / identity-return on tensors
            # already in float32, so this is forward-compatible with
            # configs that pin label dtype upstream.
            label = batch["label"].to(self.device, non_blocking=True).float()
            wmap  = batch.get("weight_map")
            if wmap is not None:
                wmap = wmap.to(self.device, non_blocking=True)
                if wmap.numel() == 0 or wmap.shape[1] == 0:
                    wmap = None  # treat empty collation sentinels as absent

            # --- GPU augmentation: image, label and (optional) weight_map
            #     share one sampled spatial transform so alignment holds.
            #     The augmentor uses nearest interpolation for label and
            #     bilinear for weight_map (continuous values).
            image, label, wmap = self.augmentor(image, label, wmap)

            # --- Center-crop when dataset returned oversampled patches
            if self.needs_crop:
                image, label, wmap = self._center_crop(image, label, wmap)

            # --- 3D lazy multi-res: rebuild the per-view C_res stack
            #     from the single max-FOV cube. After this step the
            #     contract is bit-identical to the legacy False-path
            #     3D dataset emission ((B, C_res, pD, pH, pW)), so the
            #     downstream forward / loss / metrics path is unchanged.
            if self.keep_native_multi_res:
                image, label, wmap = self._split_views_native_3d(
                    image, label, wmap)

            # --- 2.5D adaptation: collapse the C_res=1 channel so the D
            #     axis becomes the model's input-channel dimension.
            #     With aux seg supervision active the label/wmap tensors
            #     are kept at rank-5 ``(B, C_res, D, H, W)`` so per-view
            #     losses can index ``label[:, k]`` for aux head ``k``.
            label_all_views: Optional[torch.Tensor] = None
            wmap_all_views : Optional[torch.Tensor] = None
            # ``aux_view_labels[k]`` / ``aux_view_wmaps[k]`` carry the per-
            # view native-depth supervision targets when aux_keep_native_d
            # is on. They are list-form because views have varying D_k —
            # the legacy rank-5 ``label_all_views`` cannot represent that.
            aux_view_labels: Optional[List[torch.Tensor]] = None
            aux_view_wmaps: Optional[List[Optional[torch.Tensor]]] = None
            if self.is_2_5d:
                if self.lift_2_5d_to_3d and self.aux_seg_supervision:
                    # Lift+aux: image stays rank-5 ``(B, n_views, D, H, W)``
                    # (no squeeze; D is a spatial axis). Label/wmap stay
                    # rank-5 ``(B, C_res, D, H, W)`` so per-view aux
                    # supervision can index ``label[:, k:k+1]``. ``label`` /
                    # ``wmap`` are shadowed to view 0 (kept rank-5 via
                    # ``[:, :1]`` to match the lift main path's
                    # MultiResolutionLoss(num_res=1) contract — both the
                    # _compute_loss_aux_fp32 helper and the metric block
                    # ``self._inner_loss.split_for_metrics`` expect rank-5
                    # ``(B, 1, D, H, W)`` here).
                    label_all_views = label
                    wmap_all_views = wmap
                    label = label_all_views[:, :1].contiguous()
                    wmap = (wmap_all_views[:, :1].contiguous()
                            if wmap_all_views is not None else None)
                elif self.lift_2_5d_to_3d:
                    # Lift mode: D stays a real spatial axis. Image flows
                    # through unchanged — ``(B, n_views, D, H, W)``. The
                    # supervision target is view 0 only (``label[:, :1]``
                    # keeps the C_res axis at length 1 so the
                    # MultiResolutionLoss(num_res=1) contract is matched).
                    label = label[:, :1].contiguous()
                    if wmap is not None:
                        wmap = wmap[:, :1].contiguous()
                elif self.aux_seg_supervision and self.aux_keep_native_d:
                    # Native-depth path: dataset emits a single max-FOV
                    # cube; we center-crop per view BEFORE forward, with
                    # view 0 = the standard ``D``-deep main supervision
                    # and views 1..K = native ``D_k``-deep aux targets.
                    (image, label, wmap,
                     aux_view_labels, aux_view_wmaps) = (
                        self._split_views_native_d(image, label, wmap))
                elif self.aux_seg_supervision:
                    image, label_all_views, wmap_all_views = (
                        self._squeeze_2_5d_keep_views(image, label, wmap))
                    # ``label`` / ``wmap`` shadow view 0 so the dice metric
                    # block below — which is shared with the no-aux path —
                    # operates on the same supervision target as the main
                    # head (bit-equivalent metric definition).
                    label = label_all_views[:, 0]
                    wmap = (wmap_all_views[:, 0]
                            if wmap_all_views is not None else None)
                else:
                    image, label, wmap = self._squeeze_2_5d(image, label, wmap)

            # --- Effective accumulation denominator for this step
            if remainder > 0 and step >= partial_start:
                effective_accum = remainder
            else:
                effective_accum = accum

            # --- Forward (AMP) + loss (fp32)
            #
            # Model forward stays under autocast for the fp16 speedup, but
            # the loss is computed in fp32. Dice / BCE compounds reduce
            # over millions of voxels per batch (e.g. B*D*H*W ≈ 3.1M in
            # 2.5D); the running sums in the dice numerator/denominator
            # quickly exceed fp16's ±65504 range once the model starts
            # confidently predicting foreground, producing inf → NaN
            # losses and poisoning training (root cause of the epoch-20
            # NaN explosion observed historically). Casting pred to fp32
            # outside autocast is the standard nnU-Net-style fix.
            with autocast(device_type="cuda", enabled=self.use_amp,
                          dtype=self.amp_dtype):
                pred = self.model(image)
            breakdown: Dict[str, float] = {}
            if self.aux_seg_supervision and self.aux_keep_native_d:
                # Native-depth aux path: ``aux_view_labels[k]`` /
                # ``aux_view_wmaps[k]`` carry view-k targets at native
                # depth ``D_k``. ``label`` / ``wmap`` are view 0 (D-deep).
                loss = self._compute_loss_aux_native_d_fp32(
                    pred, label, wmap,
                    aux_view_labels, aux_view_wmaps,
                    breakdown=breakdown)
            elif self.aux_seg_supervision:
                # Aux-aware path: pred is a dict and we route per-view
                # supervision through ``_compute_loss_aux_fp32``. The main
                # path inside that helper still goes through the
                # DS-wrapped criterion, so deep_supervision composes with
                # aux supervision orthogonally. The ``breakdown`` dict is
                # filled in-place with detached scalars per component
                # (L_main, L_aux_k, w_aux_k, L_total) for diagnostic logs.
                loss = self._compute_loss_aux_fp32(
                    pred, label_all_views, wmap_all_views,
                    breakdown=breakdown)
            else:
                loss = self._compute_loss_fp32(
                    self.criterion, pred, label, weight_map=wmap)
            if effective_accum > 1:
                loss = loss / effective_accum

            # --- Backward (accumulates into .grad)
            self.scaler.scale(loss).backward()

            # --- Step boundary: every `accum` micro-steps, or at end of
            #     epoch to flush the partial tail.
            is_step_boundary = (
                (step + 1) % accum == 0 or (step + 1) == total_steps)
            if is_step_boundary:
                if tc.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), tc.grad_clip_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                self.scheduler.step()
                if self.ema is not None:
                    self.ema.update(self.model)

                # One-shot real-memory diagnostic: after the FIRST complete
                # optimizer-step cycle (i.e. ``accum`` micro-batches of
                # forward+backward + 1 optimizer.step + EMA update), report
                # the actual CUDA memory peak observed since the start of
                # this epoch. This includes activations / autograd saved
                # tensors / cuDNN+cuBLAS workspace / optimizer state — i.e.
                # everything the static estimate explicitly excludes. Fires
                # exactly once per ``Trainer.fit()`` call, before the user
                # has to wait for the full epoch to finish to see the
                # ``GPU peak (epoch N)`` line.
                if (not self._first_step_mem_logged
                        and self.device.type == "cuda"):
                    one_step_peak = (
                        torch.cuda.max_memory_allocated(self.device)
                        / (1 << 20))
                    logger.info(
                        "Actual one-step GPU peak: %.1f MiB "
                        "(forward + backward + optimizer.step + EMA "
                        "update; accum=%d micro-batches). Steady-state "
                        "training peak should stay close to this; the "
                        "full-epoch peak is reported separately at end "
                        "of each epoch as 'GPU peak (epoch N)'.",
                        one_step_peak, accum,
                    )
                    self._first_step_mem_logged = True

            # --- Metrics (log unscaled loss)
            loss_val = (loss.item() * effective_accum
                        if effective_accum > 1 else loss.item())
            # Guard the running average against single-batch non-finite
            # losses. A single NaN/Inf would otherwise poison ``sum`` and
            # render every later ``loss_meter.avg`` NaN for the rest of the
            # epoch (and all subsequent epochs' logs) even though training
            # itself continues healthily via GradScaler's inf-skip path.
            if math.isfinite(loss_val):
                loss_meter.update(loss_val, image.shape[0])
                # Aux breakdown averaging — only update meters with finite
                # scalars to mirror the main loss meter's NaN guard.
                for name, val in breakdown.items():
                    if not math.isfinite(val):
                        continue
                    if name not in component_meters:
                        component_meters[name] = AverageMeter()
                    component_meters[name].update(val, image.shape[0])
            else:
                logger.warning(
                    "Non-finite train loss (%s) at epoch %d step %d/%d; "
                    "skipping meter update. GradScaler will skip this "
                    "optimizer step.",
                    loss_val, epoch + 1, step + 1, total_steps)

            if (step + 1) % tc.log_every == 0 or step == 0:
                with torch.no_grad():
                    # Unify dict / list / tensor outputs to the main full-
                    # res tensor before metric splitting. ``label`` is
                    # already view 0 in both aux and no-aux paths above.
                    p = self._extract_main_pred(pred)
                    # Mode-agnostic via the inner wrapper's contract:
                    #   3D : returns (B, num_fg, *spatial), (B, num_fg, *spatial)
                    #   2.5D: returns (B*D, num_fg, H, W),  (B*D, num_fg, H, W)
                    p_1x, lbl_1x = self._inner_loss.split_for_metrics(
                        p.detach(), label)
                    dice = compute_dice_per_class(p_1x, lbl_1x)
                    mean_dice = dice.mean().item()
                    dice_meter.update(mean_dice, image.shape[0])
                # Compact diagnostic line: when aux supervision is on, append
                # "L_main / L_aux_k=val(w=...)" so the user can immediately
                # see whether each aux head is contributing meaningful
                # gradient (similar magnitude to L_main scaled by w_k).
                aux_msg = self._format_breakdown(breakdown)
                logger.debug(
                    "  [%d/%d] loss=%.4f dice=%.4f lr=%.2e%s",
                    step + 1, total_steps,
                    loss_val, mean_dice, self.scheduler.get_lr(),
                    aux_msg)

        # Surface the epoch-mean of each aux component alongside the
        # standard loss/dice metrics. ``fit()`` formats them into the
        # epoch summary line so the user sees aux contributions at a
        # cadence that matches the main metric log.
        out: Dict[str, float] = {"loss": loss_meter.avg, "dice": dice_meter.avg}
        for name, meter in component_meters.items():
            out[name] = meter.avg
        return out

    @staticmethod
    def _format_breakdown(breakdown: Dict[str, float]) -> str:
        """Render a compact " | L_main=... L_aux_1=...(w=...) ..." string.

        Returns "" when the breakdown is empty (single-FOV or aux disabled),
        keeping legacy log lines bit-identical for non-aux runs.
        """
        if not breakdown:
            return ""
        parts: List[str] = []
        # Stable ordering: L_main first, then L_aux_k in ascending k.
        if "L_main" in breakdown:
            parts.append(f"L_main={breakdown['L_main']:.4f}")
        aux_keys = sorted(
            (k for k in breakdown if k.startswith("L_aux_")),
            key=lambda k: int(k.split("_")[-1]))
        for k in aux_keys:
            view_k = k.split("_")[-1]
            w_key = f"w_aux_{view_k}"
            if w_key in breakdown:
                parts.append(
                    f"{k}={breakdown[k]:.4f}(w={breakdown[w_key]:.3g})")
            else:
                parts.append(f"{k}={breakdown[k]:.4f}")
        return " | " + " ".join(parts)

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """Validate on the validation set under EMA weights (if enabled).

        Uses POOLED per-class dice:
            dice[c] = 2 * Σ_batches intersection[c] / Σ_batches denom[c]
        This matches the nnU-Net convention and avoids the negative bias of
        averaging per-batch dice when some classes are empty in some batches.
        """
        self.model.eval()
        loss_meter = AverageMeter()
        inter_sum: Optional[torch.Tensor] = None  # (C,)
        denom_sum: Optional[torch.Tensor] = None  # (C,)
        cov_sum:   Optional[torch.Tensor] = None  # (C,) number of samples with non-empty GT per class

        n_samples = 0

        with self._ema_swapped():
            for batch in self.val_loader:
                image = batch["image"].to(self.device, non_blocking=True)
                # P8: match train-loop cast so loss / metric see float32
                # labels regardless of the dataset emission dtype.
                label = batch["label"].to(self.device, non_blocking=True).float()

                # 2.5D: squeeze C_res=1 for both image and label before
                # forward. (No GPU augmentation in val — directly squeeze.)
                # In aux_keep_native_d mode the dataset emits a single
                # max-FOV cube; we run the same per-view split as the
                # training loop but discard aux targets (val metric only
                # exercises view 0 = main supervision).
                if self.is_2_5d:
                    if self.lift_2_5d_to_3d:
                        # Lift mode (val): keep image at (B, n_views, D, H, W);
                        # supervise on view 0 only (matches train loop).
                        label = label[:, :1].contiguous()
                    elif self.aux_keep_native_d:
                        image, label, _, _, _ = (
                            self._split_views_native_d(image, label, None))
                    else:
                        image, label, _ = self._squeeze_2_5d(image, label, None)
                elif self.keep_native_multi_res:
                    # 3D lazy-multi-res path: dataset emits a single
                    # max-FOV cube; rebuild the per-view C_res stack
                    # so the model sees the standard input contract.
                    # Val skips augment/oversample-crop (no augmentor
                    # called), so the cube already arrives at the
                    # max-FOV target_patch_size — split directly.
                    image, label, _ = self._split_views_native_3d(
                        image, label, None)

                with autocast(device_type="cuda", enabled=self.use_amp,
                              dtype=self.amp_dtype):
                    pred = self.model(image)
                    # ``_extract_main_pred`` is dict/list/tensor-safe.
                    # In eval mode UNet3D returns a tensor (aux gated by
                    # ``self.training``), but the helper keeps the val
                    # contract robust if a future change relaxes that.
                    pred = self._extract_main_pred(pred)
                    pred_1x, target_1x = self._inner_loss.split_for_metrics(
                        pred, label)
                # Loss in fp32 — see `_train_epoch` for rationale (fp16 dice
                # reductions overflow on large patches and produce NaN).
                loss = self._compute_loss_fp32(
                    self.base_loss, pred_1x, target_1x)

                loss_val = loss.item()
                if math.isfinite(loss_val):
                    loss_meter.update(loss_val, image.shape[0])
                else:
                    logger.warning(
                        "Non-finite val loss (%s) at epoch %d; skipping "
                        "meter update.", loss_val, epoch + 1)
                stats = dice_batch_stats(pred_1x.float(), target_1x)
                if inter_sum is None:
                    inter_sum = stats["inter"].clone()
                    denom_sum = stats["denom"].clone()
                    cov_sum   = stats["n_with_gt"].clone()
                else:
                    inter_sum += stats["inter"]
                    denom_sum += stats["denom"]
                    cov_sum   += stats["n_with_gt"]
                n_samples += image.shape[0]

        if inter_sum is None:
            logger.warning("Validation loader yielded no batches.")
            return {"val_loss": float("nan"), "mean_dice": 0.0}

        # Pooled dice with 1e-5 smoothing to match training loss behaviour.
        smooth = 1e-5
        dice_per_class = (2.0 * inter_sum + smooth) / (denom_sum + smooth)
        dice_per_class = dice_per_class.cpu()

        metrics: Dict[str, float] = {"val_loss": loss_meter.avg}
        for c in range(len(dice_per_class)):
            metrics[f"dice_class_{c}"] = dice_per_class[c].item()
        metrics["mean_dice"] = dice_per_class.mean().item()

        # Per-class coverage helps diagnose "val dice is low because this
        # class barely appears in the val set" vs. genuine model failure.
        cov = cov_sum.cpu().tolist()
        logger.info(
            "  Val: loss=%.4f, pooled_mean_dice=%.4f, per_class=%s, "
            "coverage=%s/%d samples",
            metrics["val_loss"], metrics["mean_dice"],
            [f"{d:.4f}" for d in dice_per_class.tolist()],
            [int(c) for c in cov], n_samples)
        return metrics

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    # Maximum absolute logit value fed into the loss. AMP fp16 can produce
    # ``±inf`` in the forward path (instance-norm variance underflow,
    # saturated activations, large late-training weights). ``pred.float()``
    # does NOT recover those: ``inf`` in fp16 stays ``inf`` in fp32. The
    # numerically-stable BCE-with-logits kernel then evaluates
    #     max(x,0) - x*z + log1p(exp(-|x|))
    # which for ``x=+inf, target=1`` is ``inf - inf + 0 = NaN`` — the
    # dominant cause of single-batch NaN losses late in training. Clamping
    # to ±_LOGIT_CLAMP wipes that failure mode at zero cost: ``sigmoid(50)``
    # is already indistinguishable from 1.0 in fp32 (Dice is unaffected),
    # and BCE's gradient ``sigmoid(x) - target`` at ``|x|=50`` is 0 or 1 —
    # the same as the unclamped limit, so healthy training (|x| << 20) sees
    # bit-identical behaviour.
    _LOGIT_CLAMP: float = 50.0

    @staticmethod
    def _compute_loss_fp32(
        loss_fn: nn.Module,
        pred,
        target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run ``loss_fn`` outside autocast with fp32 inputs.

        Casts every prediction tensor (single tensor or list-of-tensors as
        produced by deep supervision) to ``float32`` and disables autocast
        for the loss call. The label tensor is cast to fp32 only if it is
        a float dtype — integer label tensors must remain integer-typed
        because some compound losses round / index them. ``weight_map``
        is cast to fp32 when provided.

        Prediction logits are additionally clamped to
        ``[-_LOGIT_CLAMP, +_LOGIT_CLAMP]`` to defuse occasional ±inf from
        the fp16 autocast forward path (see ``_LOGIT_CLAMP`` docstring).
        """
        c = Trainer._LOGIT_CLAMP
        if isinstance(pred, list):
            pred_fp32 = [p.float().clamp(-c, c) for p in pred]
        else:
            pred_fp32 = pred.float().clamp(-c, c)
        target_fp32 = target.float() if target.is_floating_point() else target
        wmap_fp32 = weight_map.float() if weight_map is not None else None
        with autocast(device_type="cuda", enabled=False):
            if wmap_fp32 is None:
                return loss_fn(pred_fp32, target_fp32)
            return loss_fn(pred_fp32, target_fp32, weight_map=wmap_fp32)

    # ------------------------------------------------------------------
    # Aux-aware helpers (multi-FOV deep supervision in 2.5D mode)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_main_pred(pred):
        """Unwrap the model output to its main-path tensor.

        Accepts:
          - dict {"main": tensor|list, "aux": [...]} → returns tensor
            (DS list is collapsed to its head, the main full-res output).
          - list [main_out, ds_2nd, ...] → returns list[0].
          - tensor → returns as-is.

        Mirrors how ``_train_epoch`` historically picked the main scale
        before the aux-supervision dict contract was added; centralising
        the logic here keeps the metric / log path branch-free.
        """
        if isinstance(pred, dict):
            pred = pred["main"]
        if isinstance(pred, list):
            pred = pred[0]
        return pred

    def _split_views_native_3d(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        wmap: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Per-view native crop + resize for the 3D lazy-extraction path.

        Input contract (post-augment, post-center-crop):
          image : ``(B, 1, eD_max, eH_max, eW_max)`` — single max-FOV cube
                  emitted by ``SegDataset3D._getitem_native_multi_res_z``
                  (z_axis: eH_max=pH, eW_max=pW) or
                  ``SegDataset3DCubic._getitem_native_multi_res_cubic``
                  (cubic: all three axes scaled by ``max_scale``).
                  ``(eD_max, eH_max, eW_max) == self.target_patch_size``.
          label : ``(B, 1, eD_max, eH_max, eW_max)`` raw integer labels.
          wmap  : optional ``(B, 1, eD_max, eH_max, eW_max)`` continuous
                  weights.

        Output (matches the legacy False-path 3D dataset emission):
          image_3d : ``(B, C_res, pD, pH, pW)`` — view k centered native
                      crop ``(D_k, H_k, W_k)`` resized trilinearly to
                      ``(pD, pH, pW)``, stacked along the new C_res axis.
          label_3d : ``(B, C_res, pD, pH, pW)`` raw integer labels —
                      same crop, but resized with ``mode="nearest"`` to
                      preserve discrete label values.
          wmap_3d  : ``(B, C_res, pD, pH, pW)`` or ``None`` — same crop,
                      trilinear resize (continuous weights).

        Geometric note
        --------------
        All views share the same physical centre by construction (the
        dataset extracted ONE max-FOV cube around a single sampled
        centre). Center-cropping each view's native size from this cube
        and resizing to canonical ``patch_size`` reproduces voxel-for-
        voxel the per-view independent extraction used by the
        False-path — modulo a single torch ``F.interpolate`` pass per
        view instead of two (one ``scipy.ndimage.zoom`` in the dataset +
        one shared augment grid_sample). The augment grid_sample now
        operates on the full max-FOV cube so all views receive the SAME
        warp at their native physical resolution; pre-resize information
        loss for aux views (False-path's ``round(eD*s) → eD`` zoom
        before augment) is eliminated.
        """
        if not self.keep_native_multi_res:
            raise RuntimeError(
                "_split_views_native_3d called but "
                "keep_native_multi_res=False")
        if image.ndim != 5 or image.shape[1] != 1:
            raise ValueError(
                "native-3d split expects (B, 1, eD_max, eH_max, eW_max); "
                f"got image.shape={tuple(image.shape)}")
        if (label.shape[:2] != image.shape[:2]
                or label.shape[2:] != image.shape[2:]):
            raise ValueError(
                "image / label shape mismatch: "
                f"image={tuple(image.shape)}, label={tuple(label.shape)}")
        B, _, tD, tH, tW = image.shape
        if (tD, tH, tW) != tuple(self.target_patch_size):
            raise ValueError(
                f"native-3d split expects spatial dims == target_patch_size"
                f"={self.target_patch_size}; got {(tD, tH, tW)}. The "
                "post-augment center crop should already have removed "
                "the augment oversample margin.")

        pD, pH, pW = (int(x) for x in self.cfg.data.patch_size)

        def _center_crop_3d(t: torch.Tensor, sizes: Tuple[int, int, int]
                             ) -> torch.Tensor:
            """Return the center ``(d_k, h_k, w_k)`` crop along axes 2/3/4
            of a ``(B, 1, D, H, W)`` tensor."""
            d_k, h_k, w_k = sizes
            d0 = (tD - d_k) // 2
            h0 = (tH - h_k) // 2
            w0 = (tW - w_k) // 2
            return t[:, :, d0:d0 + d_k, h0:h0 + h_k, w0:w0 + w_k]

        img_views: List[torch.Tensor] = []
        lbl_views: List[torch.Tensor] = []
        wmap_views: List[torch.Tensor] = []
        for k, sizes in enumerate(self._mr_native_sizes):
            img_k = _center_crop_3d(image, sizes)
            lbl_k = _center_crop_3d(label, sizes)
            wmap_k = (_center_crop_3d(wmap, sizes)
                      if wmap is not None else None)

            # Resize per-view native crop back to canonical ``patch_size``
            # so all views can be stacked along the C_res axis. Skip the
            # F.interpolate call when sizes already match (true for view 0
            # and any view where ``round(p_axis * s_k) == p_axis``).
            if sizes != (pD, pH, pW):
                img_k = F.interpolate(
                    img_k, size=(pD, pH, pW),
                    mode="trilinear", align_corners=False)
                # Labels: nearest preserves discrete integer values
                # (mirrors ``resize_3d(..., is_label=True)`` in the
                # OFF-path dataset).
                lbl_k = F.interpolate(lbl_k, size=(pD, pH, pW), mode="nearest")
                if wmap_k is not None:
                    wmap_k = F.interpolate(
                        wmap_k, size=(pD, pH, pW),
                        mode="trilinear", align_corners=False)

            # Each view contributes ONE channel to the C_res axis. Strip
            # the existing leading "1" via ``squeeze(1)`` so ``stack(dim=1)``
            # produces ``(B, C_res, pD, pH, pW)`` — bit-equivalent shape
            # to the legacy False-path 3D dataset emission.
            img_views.append(img_k.squeeze(1))
            lbl_views.append(lbl_k.squeeze(1))
            if wmap_k is not None:
                wmap_views.append(wmap_k.squeeze(1))

        image_out = torch.stack(img_views, dim=1).contiguous()
        label_out = torch.stack(lbl_views, dim=1).contiguous()
        wmap_out: Optional[torch.Tensor] = None
        if wmap_views:
            wmap_out = torch.stack(wmap_views, dim=1).contiguous()
        return image_out, label_out, wmap_out

    def _split_views_native_d(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        wmap: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        List[torch.Tensor],
        List[Optional[torch.Tensor]],
    ]:
        """Per-view native-depth center crop for the 2.5D simplified path.

        Input contract (post-augment, post-center-crop):
          image : ``(B, 1, eD_max, H, W)`` — single max-FOV cube emitted
                  by ``SegDataset3D._getitem_native_d``. Here
                  ``eD_max == round(D * max_scale) == self.target_patch_size[0]``.
          label : ``(B, 1, eD_max, H, W)`` raw integer labels.
          wmap  : optional ``(B, 1, eD_max, H, W)`` continuous weights.

        Output:
          image_2d : ``(B, sum_k D_k, H, W)`` — per-view native-depth
                      slices concatenated along the channel axis. View 0
                      occupies channels ``[0, D_0)``; view k occupies
                      ``[Σ_{<k} D_j, Σ_{<=k} D_j)``. The 2D model consumes
                      this directly. ``D_k = round(D * s_k)`` and view 0
                      sits at the centered ``D_0 == D`` slices.
          label_main : ``(B, D, H, W)`` — view-0 supervision (raw int).
          wmap_main  : ``(B, D, H, W)`` or ``None`` — view-0 weight map.
          aux_labels : list of ``(B, D_k, H, W)`` for k=1..K-1.
          aux_wmaps  : list (same length) of weight maps or None per view.

        Geometric note
        --------------
        All views share the same z-center by construction (``_sample_z``
        runs once before extraction). Center-cropping ``D_k`` slices out
        of ``eD_max`` reproduces, voxel-for-voxel, the per-view
        independent extraction used by the legacy False-path — but with
        a single shared augmentation grid_sample applied to the whole
        cube, eliminating cross-view warp drift.
        """
        if not self.aux_keep_native_d:
            raise RuntimeError(
                "_split_views_native_d called but aux_keep_native_d=False")
        if image.ndim != 5 or image.shape[1] != 1:
            raise ValueError(
                "native-d split expects (B, 1, eD_max, H, W); got "
                f"image.shape={tuple(image.shape)}")
        if label.shape[:2] != image.shape[:2] or label.shape[2:] != image.shape[2:]:
            raise ValueError(
                "image / label shape mismatch: "
                f"image={tuple(image.shape)}, label={tuple(label.shape)}")
        B, _, eD_max, H, W = image.shape
        depths = self.aux_view_depths
        D = depths[0]
        if eD_max != int(self.target_patch_size[0]):
            raise ValueError(
                f"native-d split expects depth axis == target_patch_size[0]"
                f"={self.target_patch_size[0]}; got {eD_max}. The post-"
                "augment center crop should already have removed the "
                "augment oversample margin.")
        if max(depths) > eD_max:
            raise ValueError(
                f"max(aux_view_depths)={max(depths)} exceeds eD_max={eD_max}; "
                "this indicates a multi_res_scales / patch_size mismatch.")

        def _center_slab(t: torch.Tensor, d_k: int) -> torch.Tensor:
            """Return the center ``d_k`` slices along axis=2 of ``t`` (B, 1, D, H, W)."""
            d0 = (eD_max - d_k) // 2
            return t[:, 0, d0:d0 + d_k].contiguous()  # (B, d_k, H, W)

        # ---- View 0: main supervision target ---------------------------
        # ``image_view_0 == _center_slab(image, D)`` is exactly the (B, D, H, W)
        # tensor the legacy single-FOV / view-0 path would consume.
        view0_img = _center_slab(image, D)
        label_main = _center_slab(label, D)
        wmap_main = _center_slab(wmap, D) if wmap is not None else None

        # ---- Aux views ---------------------------------------------------
        aux_imgs: List[torch.Tensor] = []
        aux_labels: List[torch.Tensor] = []
        aux_wmaps: List[Optional[torch.Tensor]] = []
        for d_k in depths[1:]:
            aux_imgs.append(_center_slab(image, d_k))
            aux_labels.append(_center_slab(label, d_k))
            aux_wmaps.append(_center_slab(wmap, d_k) if wmap is not None else None)

        # ---- Concatenate views along the channel axis for the model -----
        if aux_imgs:
            image_2d = torch.cat([view0_img] + aux_imgs, dim=1).contiguous()
        else:
            # Defensive: aux_keep_native_d implies n_views > 1, but degrade
            # gracefully to single-view layout if someone bypasses validate.
            image_2d = view0_img.contiguous()
        expected_in = sum(depths)
        if image_2d.shape[1] != expected_in:
            raise RuntimeError(
                f"native-d split produced {image_2d.shape[1]} input "
                f"channels; expected sum(depths)={expected_in}.")
        return image_2d, label_main, wmap_main, aux_labels, aux_wmaps

    def _compute_loss_aux_native_d_fp32(
        self,
        pred,
        label_main: torch.Tensor,
        wmap_main: Optional[torch.Tensor],
        aux_labels: Optional[List[torch.Tensor]],
        aux_wmaps: Optional[List[Optional[torch.Tensor]]],
        breakdown: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """Native-depth aux loss aggregator (mirror of ``_compute_loss_aux_fp32``).

        Differences from the False-path counterpart:
          * Targets arrive as PER-VIEW LIST tensors (one (B, D_k, H, W)
            per view) instead of rank-5 ``(B, C_res, D, H, W)``, because
            views have varying depths.
          * Each view dispatches to its OWN ``SliceChannelLoss`` from
            ``self.aux_inner_losses[k-1]`` (constructed with
            ``num_slices = D_k`` so the slice/channel reshape is exact).

        Loss formula and ``breakdown`` schema are bit-identical to the
        False-path so downstream logging code (``_format_breakdown`` and
        the per-component meters in ``_train_epoch``) need no changes.
        """
        if isinstance(pred, dict):
            main_pred = pred["main"]
            aux_preds = pred.get("aux", []) or []
        else:
            main_pred, aux_preds = pred, []

        main_l = self._compute_loss_fp32(
            self.criterion, main_pred, label_main, weight_map=wmap_main)
        total = main_l
        if breakdown is not None:
            breakdown["L_main"] = float(main_l.detach().item())

        if not aux_preds or not self.aux_inner_losses:
            if breakdown is not None:
                breakdown["L_total"] = float(total.detach().item())
            return total
        if aux_labels is None:
            raise RuntimeError(
                "aux_keep_native_d aux loss path requires aux_labels list "
                "but received None — likely a missing _split_views_native_d "
                "call upstream.")
        if not (len(aux_preds) == len(self.aux_weights)
                == len(self.aux_inner_losses) == len(aux_labels)):
            raise RuntimeError(
                "aux_keep_native_d arity mismatch: "
                f"preds={len(aux_preds)}, weights={len(self.aux_weights)}, "
                f"losses={len(self.aux_inner_losses)}, "
                f"labels={len(aux_labels)}.")
        for k_idx, (ap, w_k, loss_k, lbl_k) in enumerate(zip(
                aux_preds, self.aux_weights, self.aux_inner_losses, aux_labels)):
            view_k = k_idx + 1
            wm_k = (aux_wmaps[k_idx]
                    if aux_wmaps is not None else None)
            aux_l = self._compute_loss_fp32(
                loss_k, ap, lbl_k, weight_map=wm_k)
            total = total + w_k * aux_l
            if breakdown is not None:
                breakdown[f"L_aux_{view_k}"] = float(aux_l.detach().item())
                breakdown[f"w_aux_{view_k}"] = float(w_k)
        if breakdown is not None:
            breakdown["L_total"] = float(total.detach().item())
        return total

    def _squeeze_2_5d_keep_views(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        wmap: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Aux-supervision variant of :meth:`_squeeze_2_5d`.

        The image is reshaped exactly as in the legacy single-view path —
        ``(B, C_res, D, H, W) → (B, C_res*D, H, W)`` — but ``label`` and
        ``wmap`` are returned UNCHANGED at rank-5 ``(B, C_res, D, H, W)``.
        The training loop then slices view 0 for the main supervision and
        view k for the k-th auxiliary head, ensuring every view's
        physically-resampled label can drive its corresponding output.
        """
        if image.ndim != 5:
            raise ValueError(
                f"2.5D _squeeze_keep_views expects rank-5 image "
                f"(B, C_res, D, H, W); got shape={tuple(image.shape)}")
        if label.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"image / label batch+C_res mismatch: image="
                f"{tuple(image.shape)}, label={tuple(label.shape)}")
        B, C_res, D, H, W = image.shape
        image_2d = image.reshape(B, C_res * D, H, W).contiguous()
        return image_2d, label, wmap

    def _compute_loss_aux_fp32(
        self,
        pred,
        label_all: torch.Tensor,
        wmap_all: Optional[torch.Tensor],
        breakdown: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """Aux-aware loss aggregator for the 2.5D multi-FOV training step.

        Layout
        ------
        ``pred`` is normally a dict ``{"main": ..., "aux": [...]}`` produced
        by :class:`UNet3D` when aux supervision is active. As a defensive
        fallback we accept the legacy tensor / list form too — in that
        case only the main loss is computed (zero aux contribution), so
        a configuration drift can never silently inflate the loss.

        ``label_all`` is rank-5 ``(B, C_res, D, H, W)``: view 0 supervises
        the main head, view ``k`` supervises aux head ``k``. ``wmap_all``
        carries the per-view loss weight maps (or None).

        Loss formula
        ------------
            L = L_main(view_0) + Σ_{k=1..K} w_k * L_aux(view_k)

        where ``L_main`` is the full DS+SliceChannel pipeline and each
        ``L_aux`` is a single-resolution :class:`SliceChannelLoss` on the
        view-k label. Weights ``w_k`` come from ``self.aux_weights``.
        """
        # ---- Slice the supervision tensors per view --------------------
        # Folded 2.5D : SliceChannelLoss wants raw rank-4 ``(B, D, H, W)``.
        # Lift (3D)   : MultiResolutionLoss wants rank-5 ``(B, 1, D, H, W)``
        #               (the C_res axis is the resolution stack; num_res=1
        #               preserves it at length 1). Slicing as
        #               ``[:, k:k+1]`` instead of ``[:, k]`` keeps that
        #               axis intact for the lift contract.
        if self.lift_2_5d_to_3d:
            label_main = label_all[:, :1]
            wmap_main = wmap_all[:, :1] if wmap_all is not None else None
        else:
            label_main = label_all[:, 0]
            wmap_main = wmap_all[:, 0] if wmap_all is not None else None

        if isinstance(pred, dict):
            main_pred = pred["main"]
            aux_preds = pred.get("aux", []) or []
        else:
            # Defensive: model didn't produce aux outputs (e.g. eval mode
            # leakage or aux disabled at the model side). Compute main-only.
            main_pred, aux_preds = pred, []

        # Main path uses the full criterion (DS-wrapped if enabled). The
        # outer ``_compute_loss_fp32`` handles fp32 cast + logit clamp.
        main_l = self._compute_loss_fp32(
            self.criterion, main_pred, label_main, weight_map=wmap_main)
        total = main_l
        if breakdown is not None:
            # Detach to avoid keeping an extra autograd reference; ``.item``
            # synchronises but is acceptable at log_every cadence (and we
            # already do the same for the main loss meter).
            breakdown["L_main"] = float(main_l.detach().item())

        # Aux paths — each contributes ``w_k * L_aux(view_k)``.
        if not aux_preds or self.aux_inner_loss is None:
            if breakdown is not None:
                breakdown["L_total"] = float(total.detach().item())
            return total
        if len(aux_preds) != len(self.aux_weights):
            raise RuntimeError(
                f"Number of aux predictions ({len(aux_preds)}) does not "
                f"match number of aux weights ({len(self.aux_weights)}). "
                f"This indicates a model/config divergence.")
        for k_idx, (ap, w_k) in enumerate(zip(aux_preds, self.aux_weights)):
            view_k = k_idx + 1   # k=1..n_views-1 in label channel space
            # Same rank-4 / rank-5 dichotomy as the main slice above.
            if self.lift_2_5d_to_3d:
                lbl_k = label_all[:, view_k:view_k + 1]
                wm_k = (wmap_all[:, view_k:view_k + 1]
                        if wmap_all is not None else None)
            else:
                lbl_k = label_all[:, view_k]
                wm_k = wmap_all[:, view_k] if wmap_all is not None else None
            aux_l = self._compute_loss_fp32(
                self.aux_inner_loss, ap, lbl_k, weight_map=wm_k)
            total = total + w_k * aux_l
            if breakdown is not None:
                # Log the RAW aux loss (un-weighted) so the user can see
                # whether the aux head is actually being learnt; the
                # contribution to the optimiser is ``w_k * L_aux``.
                breakdown[f"L_aux_{view_k}"] = float(aux_l.detach().item())
                breakdown[f"w_aux_{view_k}"] = float(w_k)
        if breakdown is not None:
            breakdown["L_total"] = float(total.detach().item())
        return total

    def _squeeze_2_5d(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        wmap: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Collapse the C_res (multi-FOV) axis for 2.5D mode.

        Input shapes (post-augment, post-crop):
          image: (B, C_res, D, H, W) → (B, C_res * D, H, W)
              Each FOV view contributes D channels stacked contiguously.
              View 0 occupies channels [0, D); view i occupies
              [i*D, (i+1)*D). The 2D model receives this layout directly;
              ``MultiStemProj`` knows the chunk boundaries via the
              ``in_ch_per_view`` it was built with (= D).
          label: (B, C_res, D, H, W) raw int labels → (B, D, H, W)
              Only view 0 (the 1× FOV, true geometry) is used for
              supervision and metrics. Wider-FOV labels would correspond
              to a different physical slab and have no defined alignment
              with the 1× output.
          wmap : (B, C_res, D, H, W) or None → (B, D, H, W)
              Same view-0 selection rule as label, for the same reason.

        ``D becomes the slice axis`` of ``SliceChannelLoss`` and the
        per-view input-channel slab of the 2D model.

        With ``C_res == 1`` (legacy single-FOV) the reshape collapses to
        the previous ``squeeze(1)`` and is bit-identical to pre-multi-FOV
        behaviour.
        """
        if image.ndim != 5:
            raise ValueError(
                f"2.5D _squeeze expects rank-5 image (B, C_res, D, H, W); "
                f"got shape={tuple(image.shape)}")
        if label.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"image / label batch+C_res mismatch: image="
                f"{tuple(image.shape)}, label={tuple(label.shape)}")
        B, C_res, D, H, W = image.shape
        # (B, C_res, D, H, W) → (B, C_res*D, H, W). ``contiguous`` keeps
        # the memory layout linear so downstream chunking (MultiStemProj
        # uses torch.split) is a zero-copy view.
        image = image.reshape(B, C_res * D, H, W).contiguous()
        # Supervision uses ONLY the 1× FOV view (channel 0). Wider-FOV
        # views feed the encoder but their resampled labels would not
        # be voxel-aligned with the model output's slab.
        label = label[:, 0]
        if wmap is not None:
            wmap = wmap[:, 0]
        return image, label, wmap

    def _center_crop(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        wmap: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Center-crop oversized tensors to target patch_size after
        augmentation (used when `aug_oversample_ratio > 1.0` in cubic mode).
        """
        tD, tH, tW = self.target_patch_size
        _, _, D, H, W = image.shape
        d0, h0, w0 = (D - tD) // 2, (H - tH) // 2, (W - tW) // 2
        image = image[:, :, d0:d0 + tD, h0:h0 + tH, w0:w0 + tW]
        label = label[:, :, d0:d0 + tD, h0:h0 + tH, w0:w0 + tW]
        if wmap is not None:
            wmap = wmap[:, :, d0:d0 + tD, h0:h0 + tH, w0:w0 + tW]
        return image, label, wmap

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _build_state_dict(self, ema_as_primary: bool) -> Dict:
        """Assemble a complete training state.

        When `ema_as_primary` is True the saved `model_state_dict` holds EMA
        weights (deployment-friendly) and online weights are preserved in
        `model_online_state_dict` for correct resuming. Otherwise
        `model_state_dict` is online and EMA lives in `ema_state_dict`.
        """
        bare = _unwrap_compile(self.model)
        online_sd = bare.state_dict()

        # Snapshot RNG state for bit-exact resume. Covers torch CPU / CUDA,
        # numpy, and Python's random — the three sources seed_everything sets.
        rng_state = {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None),
            "numpy": __import__("numpy").random.get_state(),
            "python": __import__("random").getstate(),
        }

        state: Dict = {
            "epoch": 0,  # filled by caller
            "model_state_dict": online_sd,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "has_best": self.has_best,
            "patience_counter": self.patience_counter,
            "rng_state": rng_state,
            "config": self.cfg,
        }

        if self.ema is not None:
            state["ema_state_dict"] = self.ema.state_dict()
            if ema_as_primary:
                # Capture EMA weights as the primary state_dict. try/finally
                # ensures the model is never left with EMA weights bound.
                self.ema.apply_shadow(self.model)
                try:
                    state["model_state_dict"] = _unwrap_compile(
                        self.model).state_dict()
                finally:
                    self.ema.restore(self.model)
                state["model_online_state_dict"] = online_sd

        return state

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        state = self._build_state_dict(ema_as_primary=is_best)
        state["epoch"] = epoch

        if is_best:
            path = self.output_dir / "best_model.pth"
            torch.save(state, path)
            logger.info("Best model saved: %s", path)
        else:
            path = self.output_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save(state, path)
            logger.debug("Checkpoint saved: %s", path)

    def _load_checkpoint(self, path: str) -> None:
        logger.info("Loading checkpoint: %s", path)
        # PyTorch 2.6+ flipped ``weights_only`` default to True, which
        # rejects our checkpoints because they contain numpy RNG state
        # (``numpy._core.multiarray._reconstruct``) and the ``Config``
        # dataclass — neither is on the safe-globals allowlist. These
        # checkpoints are written by this trainer itself (trusted source),
        # so we opt back into full unpickling explicitly.
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # Prefer the online copy if present (best-model checkpoints store
        # EMA as the primary state_dict, online as a sibling).
        model_sd = ckpt.get("model_online_state_dict",
                            ckpt["model_state_dict"])
        _unwrap_compile(self.model).load_state_dict(model_sd)

        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if self.ema is not None and "ema_state_dict" in ckpt:
            self.ema.load_state_dict(ckpt["ema_state_dict"])

        self.start_epoch = ckpt.get("epoch", -1) + 1
        default_best = -math.inf if self._best_mode == "max" else math.inf
        self.best_metric = ckpt.get("best_metric", default_best)
        self.best_epoch = ckpt.get("best_epoch", 0)
        self.has_best = ckpt.get(
            "has_best", math.isfinite(self.best_metric))
        self.patience_counter = ckpt.get("patience_counter", 0)

        # Restore RNG state when present. Missing keys (older checkpoints)
        # are silently skipped — training still works, just not bit-exact.
        rng = ckpt.get("rng_state")
        if rng:
            try:
                if rng.get("torch_cpu") is not None:
                    torch.set_rng_state(rng["torch_cpu"])
                if rng.get("torch_cuda") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(rng["torch_cuda"])
                if rng.get("numpy") is not None:
                    import numpy as _np
                    _np.random.set_state(rng["numpy"])
                if rng.get("python") is not None:
                    import random as _rnd
                    _rnd.setstate(rng["python"])
                logger.info("Restored RNG state from checkpoint.")
            except Exception as e:  # pragma: no cover
                logger.warning("Failed to restore RNG state: %s", e)

        logger.info(
            "Resumed from epoch %d, best=%s=%s (patience=%d)",
            self.start_epoch, self.cfg.train.save_best_metric,
            f"{self.best_metric:.4f}" if self.has_best else "n/a",
            self.patience_counter)

    # ------------------------------------------------------------------
    # Pretrain (weights-only initialisation)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_model_state_dict(ckpt, prefer_ema: bool):
        """Locate the model state_dict inside a checkpoint, format-tolerant.

        Supports three layouts:
          1. Trainer's own checkpoints (dict with ``model_state_dict`` /
             optionally ``model_online_state_dict`` / ``ema_state_dict``).
          2. Common third-party convention: ``{"state_dict": OrderedDict}``.
          3. Raw state_dict pickled directly (an OrderedDict of tensors).

        Returns:
            (state_dict, source_label) where ``source_label`` is a short string
            describing which slot was picked — useful for logs.
        """
        # Case 3: raw state_dict
        if not isinstance(ckpt, dict) or all(
                isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt, "raw_state_dict"

        # Case 1a: prefer EMA shadow when requested and present
        if prefer_ema and "ema_state_dict" in ckpt:
            ema_state = ckpt["ema_state_dict"]
            if isinstance(ema_state, dict) and "shadow" in ema_state:
                return ema_state["shadow"], "ema_shadow"

        # Case 1b: trainer-format online weights (best-model ckpts keep online
        # in a sibling slot when EMA is the primary).
        if "model_online_state_dict" in ckpt:
            return ckpt["model_online_state_dict"], "model_online_state_dict"
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"], "model_state_dict"

        # Case 2: third-party "state_dict" wrapper
        if "state_dict" in ckpt:
            return ckpt["state_dict"], "state_dict"

        raise KeyError(
            "Pretrain checkpoint does not contain a recognisable model "
            "state_dict. Expected one of: 'model_state_dict', "
            "'model_online_state_dict', 'state_dict', or a raw OrderedDict.")

    @staticmethod
    def _strip_common_prefixes(sd):
        """Drop ``module.`` (DDP) and ``_orig_mod.`` (torch.compile) prefixes.

        Pretrain sources are often produced by DDP / compiled training; our
        target model here is the bare unwrapped module, so unifying the
        namespace before loading avoids spurious "missing key" floods.
        """
        if not isinstance(sd, dict):
            return sd
        prefixes = ("module.", "_orig_mod.")
        out = {}
        changed = False
        for k, v in sd.items():
            new_k = k
            # Strip iteratively in case prefixes are nested (compile(DDP(...))).
            while new_k.startswith(prefixes):
                for p in prefixes:
                    if new_k.startswith(p):
                        new_k = new_k[len(p):]
                        changed = True
                        break
            out[new_k] = v
        return out if changed else sd

    def _load_pretrain(self, path: str, strict: bool, load_ema: bool) -> None:
        """Load model weights only — used for transfer-learning init.

        Crucially distinct from ``_load_checkpoint``:
            * Does NOT touch optimizer / scheduler / scaler / RNG.
            * Does NOT advance ``start_epoch`` or restore best-metric tracking.
            * Re-aligns EMA shadow to the freshly loaded weights so EMA does
              not silently drift back toward the random init.
        """
        logger.info(
            "Loading pretrain weights: %s (strict=%s, load_ema=%s)",
            path, strict, load_ema)
        # weights_only=False mirrors `_load_checkpoint`: trainer-format ckpts
        # contain numpy RNG state and the Config dataclass which the safe
        # unpickler rejects. Pretrain sources are explicitly user-provided.
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        sd, source = self._extract_model_state_dict(ckpt, prefer_ema=load_ema)
        sd = self._strip_common_prefixes(sd)

        bare = _unwrap_compile(self.model)
        result = bare.load_state_dict(sd, strict=strict)
        missing = list(getattr(result, "missing_keys", []) or [])
        unexpected = list(getattr(result, "unexpected_keys", []) or [])

        # Compact, actionable diagnostics. Showing the first few keys is far
        # more useful than dumping hundreds when the head doesn't match.
        def _preview(keys, n=8):
            head = ", ".join(keys[:n])
            return head + (f", ... (+{len(keys) - n} more)" if len(keys) > n else "")

        if missing:
            logger.warning(
                "Pretrain: %d missing key(s) [%s]. These params keep their "
                "random init.", len(missing), _preview(missing))
        if unexpected:
            logger.warning(
                "Pretrain: %d unexpected key(s) [%s]. These ckpt params are "
                "discarded.", len(unexpected), _preview(unexpected))
        if not missing and not unexpected:
            logger.info("Pretrain: all keys matched cleanly.")

        # Re-align EMA shadow with the loaded weights. Without this, the
        # shadow would still hold the model's *random* init from
        # ``ModelEMA.__init__`` and EMA-based validation/checkpoints would
        # regress toward noise for the first ~1/(1-decay) steps.
        if self.ema is not None:
            with torch.no_grad():
                live_sd = bare.state_dict()
                for k, v in live_sd.items():
                    if k in self.ema.shadow:
                        self.ema.shadow[k].copy_(v)
            logger.info("Pretrain: EMA shadow re-aligned to loaded weights.")

        logger.info(
            "Pretrain loaded from `%s` slot. Training will start from "
            "epoch 0 with fresh optimizer / scheduler / scaler / best / RNG.",
            source)