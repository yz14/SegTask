"""3D 分割训练 pipeline：AMP/EMA/warmup/累积/compile/DS/checkpoint。"""

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
from einops import rearrange
# PyTorch ≥2.3：torch.amp.GradScaler 接受 device 首参；旧版需走 torch.cuda.amp。
import inspect as _inspect
try:
    from torch.amp import GradScaler as _GradScaler  # type: ignore
    from torch.amp import autocast  # type: ignore
except ImportError:  # pragma: no cover
    from torch.cuda.amp import GradScaler as _GradScaler  # type: ignore
    from torch.amp import autocast  # type: ignore


def GradScaler(device: str = "cuda", **kwargs):  # noqa: N802
    """版本无关 GradScaler：新 API 才传 device。"""
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
    compute_dice_per_class, dice_batch_stats, surface_dice_batch_stats,
)

logger = logging.getLogger(__name__)


_AMP_DTYPES = {
    "float16" : torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _unwrap_compile(m: nn.Module) -> nn.Module:
    """剥 torch.compile 的 _orig_mod 包装。"""
    return getattr(m, "_orig_mod", m)


def _cuda_supports_bf16() -> bool:
    """当前 CUDA 设备是否原生支持 bf16（Ampere+/ROCm）。"""
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
    if   tc.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=tc.lr, weight_decay=tc.weight_decay)
    elif tc.optimizer == "adam":
        return torch.optim.Adam(params, lr=tc.lr, weight_decay=tc.weight_decay)
    elif tc.optimizer == "sgd":
        return torch.optim.SGD(
            params, lr=tc.lr, weight_decay=tc.weight_decay,
            momentum=tc.momentum, nesterov=tc.nesterov)
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
    """构造 warmup 之后的 base scheduler；horizon 按 post_warmup_steps 对齐。"""
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
        # mode 跟随 save_best_mode。
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
        # 自带 warmup（pct_start）；不可与外层 WarmupScheduler 叠加。
        total_steps = tc.epochs * steps_per_epoch
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=tc.lr, total_steps=total_steps,
            pct_start=max(tc.warmup_epochs, 1) / max(tc.epochs, 1))
    raise ValueError(f"Unknown scheduler: {tc.scheduler}")


# ---------------------------------------------------------------------------
# Warmup wrapper
# ---------------------------------------------------------------------------
class WarmupScheduler:
    """线性 warmup → base scheduler。Plateau 逐 epoch step，其余逐 step。"""

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
        # 一并存 warmup 参数，便于 load 时检出配置漂移。
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
        # warmup 参数漂移会改变 schedule 形状；不致命但告警。
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
    """3D 分割完整训练 pipeline。"""

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

        # 顺序：先 to(device)（optimizer/EMA 绑已迁移参数），最后 torch.compile。
        self.model = model.to(device)

        # --- Loss ------------------------------------------------------
        # base_loss 验证时用；训练时外层再包 DeepSupervisionLoss / MultiResolutionLoss。
        self.base_loss = build_loss(cfg.loss)
        self.is_2_5d   = cfg.data.patch_mode == "2_5d"
        # Plan A lift：2.5D 走真 3D (B, num_fg, D, H, W)。与 aux_keep_native_d 互斥。
        self.lift_2_5d_to_3d = bool(
            getattr(cfg.model, "lift_2_5d_to_3d", False) and self.is_2_5d)

        # 2.5D时aux为原尺寸
        self.aux_keep_native_d = bool(
            getattr(cfg.data, "aux_keep_native_d", False)
            and self.is_2_5d
            and len(cfg.data.multi_res_scales) > 1)
        self.aux_view_depths: List[int] = (
            list(cfg.aux_view_depths) if self.aux_keep_native_d else [])

        # 3D多分辨率输入
        self.keep_native_multi_res = bool(
            getattr(cfg.data, "keep_native_multi_res", False)
            and not self.is_2_5d
            and cfg.data.patch_mode in ("z_axis", "cubic")
            and len(cfg.data.multi_res_scales) > 1)
        if self.keep_native_multi_res:
            # 每个视图的原始尺寸：z_axis 仅缩 D；cubic 缩 3 轴。
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
            # view 0 强对齐 patch_size，防浮点漂移。
            self._mr_native_sizes[0] = (pD, pH, pW)
        else:
            self._mr_native_sizes = []

        # 损失复合：INNER=MultiResolutionLoss/SliceChannelLoss；OUTER=DeepSupervisionLoss。
        # 3D pred (B, num_fg*C_res, ...)；2.5D pred (B, num_fg*D, H, W) + label (B, D, H, W)。
        if self.is_2_5d and not self.lift_2_5d_to_3d:
            num_slices = int(cfg.data.patch_size[0])
            inner      = SliceChannelLoss(  # 2D切片损失
                base_loss      = self.base_loss,
                num_fg_classes = cfg.num_fg_classes,
                num_slices     = num_slices,
                label_values   = cfg.data.label_values,
                reduction      = cfg.loss.slice_loss_reduction)
            num_res = 1   # 仅日志用；SliceChannelLoss 内部 C_res==1
            logger.info(
                "Loss: %s [2.5D, reduction=%s], num_slices=%d, fg_classes=%d",
                cfg.loss.name, cfg.loss.slice_loss_reduction,
                num_slices, cfg.num_fg_classes)
        else:
            # 3D / lifted-2.5D：shape (B, num_fg*C_res, D, H, W)。
            # lift 强制 num_res=1（aux 视图不作主监督）；3D 按 multi_res_scales 长度。
            if self.lift_2_5d_to_3d:
                num_res = 1
            else:
                num_res = len(cfg.data.multi_res_scales)
            inner = MultiResolutionLoss(  # 3D体积损失
                base_loss      = self.base_loss,
                num_fg_classes = cfg.num_fg_classes,
                num_res        = num_res,
                label_values   = cfg.data.label_values)
            logger.info(
                "Loss: %s, scales=%d, fg_classes=%d%s",
                cfg.loss.name, num_res, cfg.num_fg_classes,
                " [2.5D LIFTED to 3D]" if self.lift_2_5d_to_3d else "")

        if cfg.model.deep_supervision and cfg.loss.deep_supervision_weights:
            self.criterion = DeepSupervisionLoss(
                inner, cfg.loss.deep_supervision_weights)
        else:
            self.criterion = inner
        # 保留 INNER 句柄供指标 reshape。
        self._inner_loss = inner

        # ---- 2.5D 多 FOV aux 分割监督：仅 2.5D + n_views>1 + 开关 ----
        n_views_data = len(cfg.data.multi_res_scales)
        self.aux_seg_supervision = bool(
            getattr(cfg.model, "aux_seg_supervision", False)
            and self.is_2_5d
            and n_views_data > 1)
        if self.aux_seg_supervision:
            n_aux  = n_views_data - 1
            user_w = list(getattr(cfg.loss, "aux_supervision_weights", []))
            if not user_w:
                # 几何衰减：越宽 FOV 对齐越差，权重越小。
                user_w = [0.5 ** (k + 1) for k in range(n_aux)]
            elif len(user_w) != n_aux:
                raise ValueError(
                    f"loss.aux_supervision_weights length ({len(user_w)}) "
                    f"must equal n_views-1 ({n_aux}); got {user_w}.")
            self.aux_weights = [float(w) for w in user_w]
            # Aux 均为单分辨率逐视图损失（不走 DS）：
            #   aux_keep_native_d=False -> 共享单 SliceChannelLoss(num_slices=D)
            #   aux_keep_native_d=True  -> 逐视图 SliceChannelLoss(num_slices=D_k)
            if self.lift_2_5d_to_3d:
                # Lift+aux：aux 头 (B, num_fg, D, H, W) → MultiResolutionLoss(num_res=1)。
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
                aux_depths = self.aux_view_depths[1:]  # 跳过 view 0
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

        # OneCycleLR 自带 warmup，不可与 WarmupScheduler 叠加。
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
        # "auto"：设备支持 bf16 选 bf16，否则 fp16。
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
        # GradScaler 仅 fp16 需要。
        self._scaler_active = self.use_amp and self.amp_dtype == torch.float16
        self.scaler = GradScaler("cuda", enabled=self._scaler_active)

        # --- EMA (bind to placed, not-yet-compiled model) -------------
        self.ema = ModelEMA(self.model, tc.ema_decay) if tc.use_ema else None

        # --- torch.compile (最后) -------------------------------------
        # Inductor CUDA 后端需 Triton（Windows 无官方轮子），提前探测。
        self._compile_enabled = False
        # 首次完整 step 后记一次 GPU 峰值。
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
        # image/label/wmap 同步空间变换；label 近邻，wmap 按 cfg.augment.wmap_interp_mode。
        _scales = cfg.data.multi_res_scales or [1.0]
        self.augmentor = GPUAugmentor(cfg.augment, max_scale=max(_scales))

        # --- 裁剪（过采样 patch）---------------------------
        # z_axis/cubic：dataset 发超尺寸 patch，增强后中心裁回 patch_size。
        # aux_keep_native_d (2.5D)：dataset 发 max-FOV cube，保留全尺寸供 aux 视图。
        if self.aux_keep_native_d:
            max_scale              = max(cfg.data.multi_res_scales)
            target_d_native        = int(round(int(cfg.data.patch_size[0]) * max_scale))
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
            # 3D 懒抽取：保留全 max-FOV，后面逐视图裁+resize。z_axis 仅缩 z，cubic 缩 3 轴。
            max_scale  = max(cfg.data.multi_res_scales)
            pD, pH, pW = (int(x) for x in cfg.data.patch_size)
            if cfg.data.patch_mode == "z_axis":
                self.target_patch_size = (int(round(pD * max_scale)), pH, pW)
            else:  # cubic
                self.target_patch_size = (
                    int(round(pD * max_scale)),
                    int(round(pH * max_scale)),
                    int(round(pW * max_scale)))
            # 交叉检查：逐视图原生尺寸不得超 max-FOV 目标。
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
        self.num_fg           = cfg.num_fg_classes
        self._best_mode       = tc.save_best_mode  # "max" or "min"
        self.best_metric      = (-math.inf if self._best_mode == "max" else math.inf)
        self.has_best         = False
        self.best_epoch       = 0
        self.start_epoch      = 0
        self.patience_counter = 0

        # --- Output directory -----------------------------------------
        self.output_dir = Path(tc.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- Resume / Pretrain ----------------------------------------
        # resume：全状态恢复；pretrain：仅加载权重。同设优先 resume。
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
            # 路径传了但不存在时提前警告。
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
        """amp_dtype='auto' 解析：CUDA 支持 bf16 返 'bfloat16'，否则 'float16'。"""
        if device.type == "cuda" and _cuda_supports_bf16():
            return "bfloat16"
        return "float16"

    # ------------------------------------------------------------------
    # Memory accounting
    # ------------------------------------------------------------------
    def _estimate_train_memory(self) -> Dict[str, float]:
        """静态估计持久 GPU 内存（params/grads/optim/EMA, MiB）；不含激活/workspace。"""
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
        optim_bytes = optim_mult * n_train * 4

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
        """运行完整训练循环，返回最佳验证指标。"""
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

            # 仅 plateau 逐 epoch 驱动。
            plateau_metric = val_metrics.get(tc.save_best_metric, None)
            self.scheduler.step_epoch(metric=plateau_metric)

            # --- Best-checkpoint 决策 -----------------------------------
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
            # 汇总 aux 分量（与 per-step 同渲染器）；无 aux 时为空。
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
            # 单独一行记录本 epoch 的 GPU 峰值，随后重置计数器。
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
        """临时将 EMA 权重换入 model；try/finally 保证异常时也能还原在线权重。"""
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
        """单 epoch 训练，支持梯度累积。"""
        self.model.train()
        loss_meter = AverageMeter()
        dice_meter = AverageMeter()
        # 逐分量 meter（L_main / L_aux_k），首 batch 延初始化。
        component_meters = {}
        tc               = self.cfg.train
        accum            = self.grad_accum_steps

        total_steps   = len(self.train_loader)
        remainder     = total_steps % accum if accum > 1 else 0
        partial_start = total_steps - remainder

        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(self.train_loader):
            image = batch["image"].to(self.device, non_blocking=True)
            label = batch["label"].to(self.device, non_blocking=True).float()
            wmap  = batch.get("weight_map")
            if wmap is not None:
                wmap = wmap.to(self.device, non_blocking=True)
                if wmap.numel() == 0 or wmap.shape[1] == 0:
                    wmap = None  # 视为缺失
            
            image, label, wmap = self.augmentor(image, label, wmap)  # 数据增强

            # oversample时中心裁回 patch_size*max_scale
            if self.needs_crop:
                image, label, wmap = self._center_crop(image, label, wmap)

            # 3D 懒多分辨率：从 max-FOV cube 重建逐视图，resize到patch_size后通道拼接
            if self.keep_native_multi_res:
                image, label, wmap = self._split_views_native_3d(image, label, wmap)

            # 2.5D 懒多分辨率
            label_all_views: Optional[torch.Tensor] = None
            wmap_all_views : Optional[torch.Tensor] = None
            # aux_keep_native_d 时逐视图 D_k 不同，必须用 list。
            aux_view_labels: Optional[List[torch.Tensor]] = None
            aux_view_wmaps : Optional[List[Optional[torch.Tensor]]] = None
            if self.is_2_5d:
                if self.lift_2_5d_to_3d and self.aux_seg_supervision:
                    # Lift+aux：image 保 rank-5；label/wmap 逐视图取 [:, k:k+1] 维持 C_res 轴。
                    label_all_views = label
                    wmap_all_views = wmap
                    label = label_all_views[:, :1].contiguous()
                    wmap = (wmap_all_views[:, :1].contiguous()
                            if wmap_all_views is not None else None)
                elif self.lift_2_5d_to_3d:
                    # Lift：image 不变；仅以 view 0 作监督，保留 C_res=1 以合 num_res=1。
                    label = label[:, :1].contiguous()
                    if wmap is not None:
                        wmap = wmap[:, :1].contiguous()
                elif self.aux_seg_supervision and self.aux_keep_native_d:
                    # 原生深度路径：forward 前逐视图中心裁，view 0 = 主监督，view k = D_k aux。
                    (image, label, wmap, aux_view_labels, aux_view_wmaps) = (
                        self._split_views_native_d(image, label, wmap))
                elif self.aux_seg_supervision:
                    image, label_all_views, wmap_all_views = (
                        self._squeeze_2_5d_keep_views(image, label, wmap))
                    # label/wmap 取 view 0，使下面 dice 指标与无 aux 路径一致。
                    label = label_all_views[:, 0]
                    wmap = (wmap_all_views[:, 0]
                            if wmap_all_views is not None else None)
                else:
                    image, label, wmap = self._squeeze_2_5d(image, label, wmap)

            import SimpleITK as sitk
            debug_path = './debug1'
            os.makedirs(debug_path, exist_ok=True)
            imgs = torch.chunk(image, [12, 16, 24], dim=1)
            for jj, (a,b,c) in enumerate(zip(imgs, [label]+aux_view_labels, [wmap]+aux_view_wmaps)):
                aa = a.detach().cpu().numpy()
                aa = sitk.GetImageFromArray(aa)
                sitk.WriteImage(aa, f"{debug_path}/{jj}a.nii.gz")
                aa = b.detach().cpu().numpy()
                aa = sitk.GetImageFromArray(aa)
                sitk.WriteImage(aa, f"{debug_path}/{jj}b.nii.gz")
                aa = c.detach().cpu().numpy()
                aa = sitk.GetImageFromArray(aa)
                sitk.WriteImage(aa, f"{debug_path}/{jj}c.nii.gz")
            raise
            
            # 有效累积分母（尾巴 step 用真尾长）
            if remainder > 0 and step >= partial_start:  # TODO 不太懂
                effective_accum = remainder
            else:
                effective_accum = accum

            # Forward 走 AMP，损失下 fp32：Dice/BCE 在 fp16 下汇总易溢出→NaN。
            with autocast(device_type="cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                pred = self.model(image)
            breakdown: Dict[str, float] = {}
            # aux监督 + aux为原尺寸
            if self.aux_seg_supervision and self.aux_keep_native_d:
                loss = self._compute_loss_aux_native_d_fp32(
                    pred, label, wmap, aux_view_labels, aux_view_wmaps, breakdown=breakdown)
            elif self.aux_seg_supervision:
                # aux 路径：pred 为 dict，逐视图路由。主路仍走 DS-wrapped criterion。
                # breakdown 以 detach 标量填 L_main/L_aux_k/w_aux_k/L_total 供诊断。
                loss = self._compute_loss_aux_fp32(
                    pred, label_all_views, wmap_all_views, breakdown=breakdown)
            else:
                loss = self._compute_loss_fp32(
                    self.criterion, pred, label, weight_map=wmap)
            if effective_accum > 1:
                loss = loss / effective_accum

            # Backward
            self.scaler.scale(loss).backward()

            # 参数更新
            is_step_boundary = ((step + 1) % accum == 0 or (step + 1) == total_steps)
            if is_step_boundary:
                if tc.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                self.scheduler.step()
                if self.ema is not None:
                    self.ema.update(self.model)

                # 首个完整 optimizer-step 周期后记一次真实 GPU 峰值（fit 调用中仅一次）
                if (not self._first_step_mem_logged and self.device.type == "cuda"):
                    one_step_peak = (
                        torch.cuda.max_memory_allocated(self.device) / (1 << 20))
                    logger.info(
                        "Actual one-step GPU peak: %.1f MiB "
                        "(forward + backward + optimizer.step + EMA "
                        "update; accum=%d micro-batches). Steady-state "
                        "training peak should stay close to this; the "
                        "full-epoch peak is reported separately at end "
                        "of each epoch as 'GPU peak (epoch N)'.",
                        one_step_peak, accum)
                    self._first_step_mem_logged = True

            # 记录未缩放损失，丢弃非有限值避免污染均值（GradScaler 会跳该 step）
            loss_val = (loss.item() * effective_accum if effective_accum > 1 else loss.item())
            if math.isfinite(loss_val):
                loss_meter.update(loss_val, image.shape[0])
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
                    # dict/list/tensor 统一提取主输出；label 上面已对齐为 view 0。
                    p = self._extract_main_pred(pred)
                    # 与模式无关：3D 返 (B,num_fg,*spatial)；2.5D 返 (B*D,num_fg,H,W)。
                    p_1x, lbl_1x = self._inner_loss.split_for_metrics(
                        p.detach(), label)
                    dice      = compute_dice_per_class(p_1x, lbl_1x)
                    mean_dice = dice.mean().item()
                    dice_meter.update(mean_dice, image.shape[0])
                aux_msg = self._format_breakdown(breakdown)
                logger.debug(
                    "  [%d/%d] loss=%.4f dice=%.4f lr=%.2e%s",
                    step + 1, total_steps,
                    loss_val, mean_dice, self.scheduler.get_lr(),
                    aux_msg)

        # 输出 epoch 均值，aux 分量随后拼接到 epoch 总结行。
        out: Dict[str, float] = {"loss": loss_meter.avg, "dice": dice_meter.avg}
        for name, meter in component_meters.items():
            out[name] = meter.avg
        return out

    @staticmethod
    def _format_breakdown(breakdown: Dict[str, float]) -> str:
        """渲染 " | L_main=... L_aux_k=...(w=...)"；空 breakdown 返 ""。"""
        if not breakdown:
            return ""
        parts: List[str] = []
        # L_main 优先，随后按 k 升序输出 L_aux_k。
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
        """验证集评估（启用 EMA 时以 EMA 权重），汇总逐类 dice。

        Pooled dice：dice[c] = 2*Σ inter[c] / Σ denom[c]（nnU-Net 约定），
        避免某些 batch 空类时 per-batch 均值的负偏。
        """
        self.model.eval()
        loss_meter = AverageMeter()
        inter_sum: Optional[torch.Tensor] = None  # (C,)
        denom_sum: Optional[torch.Tensor] = None  # (C,)
        cov_sum:   Optional[torch.Tensor] = None  # (C,) 逐类有 GT 的样本数

        # Surface Dice 仅在 criterion 需要时启用，避免不必要的 maxpool 开销。
        tc = self.cfg.train
        crit = str(tc.save_best_criterion).lower().strip()
        compute_sd = (crit == "dice+surface_dice")
        sd_tol = int(tc.surface_dice_tolerance)
        sd_w   = float(tc.surface_dice_weight)
        sd_num_sum:   Optional[torch.Tensor] = None  # (C,)
        sd_denom_sum: Optional[torch.Tensor] = None  # (C,)

        n_samples = 0

        with self._ema_swapped():
            for batch in self.val_loader:
                image = batch["image"].to(self.device, non_blocking=True)
                # 与训练一致：label 在 GPU 上转 fp32。
                label = batch["label"].to(self.device, non_blocking=True).float()

                # 2.5D 验证（无增强）：折叠/逐视图裁与训练一致，仅以 view 0 评估。
                if self.is_2_5d:
                    if self.lift_2_5d_to_3d:
                        label = label[:, :1].contiguous()
                    elif self.aux_keep_native_d:
                        image, label, _, _, _ = (
                            self._split_views_native_d(image, label, None))
                    else:
                        image, label, _ = self._squeeze_2_5d(image, label, None)
                elif self.keep_native_multi_res:
                    # 3D 懒多分辨率：val 无增强，cube 已到 max-FOV target，直接拆。
                    image, label, _ = self._split_views_native_3d(
                        image, label, None)

                with autocast(device_type="cuda", enabled=self.use_amp,
                              dtype=self.amp_dtype):
                    pred = self.model(image)
                    # dict/list/tensor 均兼容；eval 下 UNet3D 返 tensor，但包接以提高鲁棒。
                    pred = self._extract_main_pred(pred)
                    pred_1x, target_1x = self._inner_loss.split_for_metrics(
                        pred, label)
                # 损失 fp32，同训练路径原因。
                loss = self._compute_loss_fp32(
                    self.base_loss, pred_1x, target_1x)

                loss_val = loss.item()
                if math.isfinite(loss_val):
                    loss_meter.update(loss_val, image.shape[0])
                else:
                    logger.warning(
                        "Non-finite val loss (%s) at epoch %d; skipping "
                        "meter update.", loss_val, epoch + 1)
                pred_1x_f = pred_1x.float()
                stats = dice_batch_stats(pred_1x_f, target_1x)
                if inter_sum is None:
                    inter_sum = stats["inter"].clone()
                    denom_sum = stats["denom"].clone()
                    cov_sum   = stats["n_with_gt"].clone()
                else:
                    inter_sum += stats["inter"]
                    denom_sum += stats["denom"]
                    cov_sum   += stats["n_with_gt"]
                if compute_sd:
                    sd_stats = surface_dice_batch_stats(
                        pred_1x_f, target_1x, tolerance=sd_tol)
                    if sd_num_sum is None:
                        sd_num_sum   = sd_stats["sd_num"].clone()
                        sd_denom_sum = sd_stats["sd_denom"].clone()
                    else:
                        sd_num_sum   += sd_stats["sd_num"]
                        sd_denom_sum += sd_stats["sd_denom"]
                n_samples += image.shape[0]

        if inter_sum is None:
            logger.warning("Validation loader yielded no batches.")
            return {"val_loss": float("nan"), "mean_dice": 0.0}

        # Pooled dice，平滑係数 1e-5 与训练损失一致。
        smooth = 1e-5
        dice_per_class = (2.0 * inter_sum + smooth) / (denom_sum + smooth)
        dice_per_class = dice_per_class.cpu()

        metrics: Dict[str, float] = {"val_loss": loss_meter.avg}
        for c in range(len(dice_per_class)):
            metrics[f"dice_class_{c}"] = dice_per_class[c].item()
        metrics["mean_dice"] = dice_per_class.mean().item()

        # Surface Dice（pooled，逐类）。仅当 criterion 启用时计算并写入指标。
        sd_msg = ""
        if compute_sd and sd_num_sum is not None:
            sd_per_class = (sd_num_sum + smooth) / (sd_denom_sum + smooth)
            sd_per_class = sd_per_class.cpu()
            for c in range(len(sd_per_class)):
                metrics[f"surface_dice_class_{c}"] = sd_per_class[c].item()
            metrics["mean_surface_dice"] = sd_per_class.mean().item()
            metrics["mean_combined"] = (
                (1.0 - sd_w) * metrics["mean_dice"]
                + sd_w * metrics["mean_surface_dice"])
            sd_msg = (
                f", pooled_mean_surface_dice@{sd_tol}px="
                f"{metrics['mean_surface_dice']:.4f}, "
                f"per_class_sd={[f'{d:.4f}' for d in sd_per_class.tolist()]}, "
                f"combined(w={sd_w:.2f})={metrics['mean_combined']:.4f}")

        # 逐类覆盖数助诊断“某类几乎不在 val 出现 vs 真实错误”。
        cov = cov_sum.cpu().tolist()
        logger.info(
            "  Val: loss=%.4f, pooled_mean_dice=%.4f, per_class=%s, "
            "coverage=%s/%d samples%s",
            metrics["val_loss"], metrics["mean_dice"],
            [f"{d:.4f}" for d in dice_per_class.tolist()],
            [int(c) for c in cov], n_samples, sd_msg)
        return metrics

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    # logit 夹匯上限：fp16 可能产生 ±inf—导致 BCEWithLogits 出 NaN。
    # sigmoid(50) ≈ 1.0，不影响正常训练（|x|≪20）行为。
    _LOGIT_CLAMP: float = 50.0

    @staticmethod
    def _compute_loss_fp32(
        loss_fn: nn.Module, pred, target: torch.Tensor, weight_map: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """在 autocast 外以 fp32 调用 loss_fn；pred 裁裁防±inf，整型 label 保持。"""
        c = Trainer._LOGIT_CLAMP
        if isinstance(pred, list):
            pred_fp32 = [p.float().clamp(-c, c) for p in pred]
        else:
            pred_fp32 = pred.float().clamp(-c, c)
        target_fp32 = target.float() if target.is_floating_point() else target
        wmap_fp32   = weight_map.float() if weight_map is not None else None
        with autocast(device_type="cuda", enabled=False):
            if wmap_fp32 is None:
                return loss_fn(pred_fp32, target_fp32)
            return loss_fn(pred_fp32, target_fp32, weight_map=wmap_fp32)

    # ------------------------------------------------------------------
    # Aux-aware helpers (multi-FOV deep supervision in 2.5D mode)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_main_pred(pred):
        """提取主路输出：dict→main→list[0]；list→[0]；tensor 原返。"""
        if isinstance(pred, dict):
            pred = pred["main"]
        if isinstance(pred, list):
            pred = pred[0]
        return pred

    def _split_views_native_3d(
        self, image: torch.Tensor, label: torch.Tensor, wmap : Optional[torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """3D 懒多分辨率：从 (B,1,eD_max,eH_max,eW_max) max-FOV cube 逐视图裁+resize
        为 (B,C_res,pD,pH,pW)，与旧 False-path 素阶等价（label 近邻，img/wmap 三线性）。"""
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

        def _center_crop_3d(t: torch.Tensor, sizes: Tuple[int, int, int]) -> torch.Tensor:
            """(B,1,D,H,W) 中心裁出 (d_k,h_k,w_k)。"""
            d_k, h_k, w_k = sizes
            d0 = (tD - d_k) // 2
            h0 = (tH - h_k) // 2
            w0 = (tW - w_k) // 2
            return t[:, :, d0:d0 + d_k, h0:h0 + h_k, w0:w0 + w_k]

        img_views : List[torch.Tensor] = []
        lbl_views : List[torch.Tensor] = []
        wmap_views: List[torch.Tensor] = []
        for k, sizes in enumerate(self._mr_native_sizes):
            img_k  = _center_crop_3d(image, sizes)
            lbl_k  = _center_crop_3d(label, sizes)
            wmap_k = (_center_crop_3d(wmap, sizes) if wmap is not None else None)

            # 记与原生尺寸不同时 resize 回 patch_size；view 0 / 重合轴跳过 interpolate。
            if sizes != (pD, pH, pW):
                img_k = F.interpolate(
                    img_k, size=(pD, pH, pW), mode="trilinear", align_corners=False)
                lbl_k = F.interpolate(lbl_k, size=(pD, pH, pW), mode="nearest")
                if wmap_k is not None:
                    wmap_k = F.interpolate(wmap_k, size=(pD, pH, pW), mode="nearest")

            # 每 view 贡献 1 个通道：squeeze(1) 后 stack(dim=1) → (B,C_res,pD,pH,pW)。
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
        self, image: torch.Tensor, label: torch.Tensor, wmap: Optional[torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], List[torch.Tensor], List[Optional[torch.Tensor]]]:
        """2.5D 懒多分辨率：(B,1,eD_max,H,W) 逐视图中心抽 D_k 切片。
        输出 image (B,ΣD_k,H,W)、view0 作主监督、aux 以 list 返回（D_k 可变）。"""
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
        depths             = self.aux_view_depths  # 各个视图的D
        D                  = depths[0]
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
            """(B,1,D,H,W) 中心取 d_k 切片。"""
            d0 = (eD_max - d_k) // 2
            return t[:, 0, d0:d0 + d_k].contiguous()  # (B, d_k, H, W)

        # ---- View 0：主监督目标 (B,D,H,W) -----------------------------
        view0_img  = _center_slab(image, D)
        label_main = _center_slab(label, D)
        wmap_main  = _center_slab(wmap, D) if wmap is not None else None

        # ---- Aux views ---------------------------------------------------
        aux_imgs  : List[torch.Tensor]           = []
        aux_labels: List[torch.Tensor]           = []
        aux_wmaps : List[Optional[torch.Tensor]] = []
        for d_k in depths[1:]:
            aux_imgs.append(_center_slab(image, d_k))
            aux_labels.append(_center_slab(label, d_k))
            aux_wmaps.append(_center_slab(wmap, d_k) if wmap is not None else None)

        # ---- 拼接输入通道 ----------------------------------------------
        if aux_imgs:
            image_2d = torch.cat([view0_img] + aux_imgs, dim=1).contiguous()
        else:
            # 退化为单视图。
            image_2d = view0_img.contiguous()
        expected_in = sum(depths)
        if image_2d.shape[1] != expected_in:
            raise RuntimeError(
                f"native-d split produced {image_2d.shape[1]} input "
                f"channels; expected sum(depths)={expected_in}.")
        return image_2d, label_main, wmap_main, aux_labels, aux_wmaps

    def _compute_loss_aux_native_d_fp32(
        self, pred, label_main: torch.Tensor, wmap_main : Optional[torch.Tensor],
        aux_labels: Optional[List[torch.Tensor]],
        aux_wmaps : Optional[List[Optional[torch.Tensor]]],
        breakdown : Optional[Dict[str, float]] = None) -> torch.Tensor:
        """逐视图走自己的 SliceChannelLoss。
        公式与 breakdown schema 与 _compute_loss_aux_fp32 一致。"""
        if isinstance(pred, dict):  # aux监督时是字典
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
            wm_k   = (aux_wmaps[k_idx] if aux_wmaps is not None else None)
            aux_l  = self._compute_loss_fp32(loss_k, ap, lbl_k, weight_map=wm_k)
            total  = total + w_k * aux_l
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
        """_squeeze_2_5d 的 aux 变体：image 同样折叠但 label/wmap 保 rank-5 供逐视图索引。"""
        if image.ndim != 5:
            raise ValueError(
                f"2.5D _squeeze_keep_views expects rank-5 image "
                f"(B, C_res, D, H, W); got shape={tuple(image.shape)}")
        if label.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"image / label batch+C_res mismatch: image="
                f"{tuple(image.shape)}, label={tuple(label.shape)}")
        image_2d = rearrange(image, 'b c d h w -> b (c d) h w').contiguous()
        return image_2d, label, wmap

    def _compute_loss_aux_fp32(
        self,
        pred,
        label_all: torch.Tensor,
        wmap_all: Optional[torch.Tensor],
        breakdown: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """2.5D 多 FOV aux 损失聚合：L = L_main(view0) + Σ w_k * L_aux(view_k)。

        pred 为 dict({main, aux:[...]})；fallback 为 tensor/list 时仅计主损失。
        Lift 下逐视图 label 取 [:, k:k+1] 保 C_res 轴；折叠下取 [:, k]。
        """
        # 逐视图切片：folded 2.5D 需 rank-4；lift 需 rank-5。
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
            # 防御：模型未产出 aux（如 eval 泄露或模型侧禁用），仅计主损失。
            main_pred, aux_preds = pred, []

        # 主路走完整 criterion（DS 包装后）；_compute_loss_fp32 负责 fp32 + 裁裁。
        main_l = self._compute_loss_fp32(
            self.criterion, main_pred, label_main, weight_map=wmap_main)
        total = main_l
        if breakdown is not None:
            # detach 避免额外 autograd 引用；.item 同步可接受（log_every 节奏）。
            breakdown["L_main"] = float(main_l.detach().item())

        # Aux 路径：逐视图 w_k * L_aux(view_k)。
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
            view_k = k_idx + 1   # label 通道空间中 k=1..n_views-1
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
                # 记未加权 aux 损失供诊断；对优化器的实际贡献为 w_k * L_aux。
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
        """2.5D 折叠 C_res 轴。image (B,C_res,D,H,W)→(B,C_res*D,H,W)；
        label/wmap 仅取 view 0 (1× FOV) 作为监督目标。C_res==1 时等价于 squeeze(1)。"""
        if image.ndim != 5:
            raise ValueError(
                f"2.5D _squeeze expects rank-5 image (B, C_res, D, H, W); "
                f"got shape={tuple(image.shape)}")
        if label.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"image / label batch+C_res mismatch: image="
                f"{tuple(image.shape)}, label={tuple(label.shape)}")
        # contiguous 使下游 MultiStemProj.split 为零拷贝 view。
        image = rearrange(image, 'b c d h w -> b (c d) h w').contiguous()
        # 仅以 1× FOV view 0 作监督；宽 FOV 重采样后的 label 与输出不对齐。
        label = label[:, 0]
        if wmap is not None:
            wmap = wmap[:, 0]
        return image, label, wmap

    def _center_crop(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        wmap : Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """增强后将超尺寸 tensor 中心裁回 target_patch_size。"""
        tD, tH, tW    = self.target_patch_size
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
        """打包训练状态。ema_as_primary=True 时 model_state_dict 为 EMA，
        在线权重放到 model_online_state_dict；反之方向。"""
        bare = _unwrap_compile(self.model)
        online_sd = bare.state_dict()

        # 快照 RNG 状态以支持位精确 resume（torch CPU/CUDA + numpy + python）。
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
                # try/finally 保证不会把 model 留在 EMA 权重状态。
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
        # PyTorch 2.6+ 默认 weights_only=True 会拒 numpy RNG / Config；ckpt 为本 trainer 写，显式关闭。
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # best-model 将 EMA 存为主 state_dict，在线权重放 sibling；load 时优先取在线。
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

        # 恢复 RNG；旧版 ckpt 无该键时静默跳过（训练仍正常但非位精确）。
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
        """定位 ckpt 里的 model state_dict，兼容 3 种布局：
        本 trainer ckpt / 第三方 {"state_dict":...} / 裸 OrderedDict。
        返回 (state_dict, source_label)。"""
        # 裸 state_dict
        if not isinstance(ckpt, dict) or all(
                isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt, "raw_state_dict"

        # 优先 EMA shadow
        if prefer_ema and "ema_state_dict" in ckpt:
            ema_state = ckpt["ema_state_dict"]
            if isinstance(ema_state, dict) and "shadow" in ema_state:
                return ema_state["shadow"], "ema_shadow"

        # trainer-format 在线权重
        if "model_online_state_dict" in ckpt:
            return ckpt["model_online_state_dict"], "model_online_state_dict"
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"], "model_state_dict"

        # 第三方 state_dict
        if "state_dict" in ckpt:
            return ckpt["state_dict"], "state_dict"

        raise KeyError(
            "Pretrain checkpoint does not contain a recognisable model "
            "state_dict. Expected one of: 'model_state_dict', "
            "'model_online_state_dict', 'state_dict', or a raw OrderedDict.")

    @staticmethod
    def _strip_common_prefixes(sd):
        """剥去 module.（DDP）与 _orig_mod.（torch.compile）前缀，避免 missing key 洪水。"""
        if not isinstance(sd, dict):
            return sd
        prefixes = ("module.", "_orig_mod.")
        out = {}
        changed = False
        for k, v in sd.items():
            new_k = k
            # 反复剥防嵌套包装。
            while new_k.startswith(prefixes):
                for p in prefixes:
                    if new_k.startswith(p):
                        new_k = new_k[len(p):]
                        changed = True
                        break
            out[new_k] = v
        return out if changed else sd

    def _load_pretrain(self, path: str, strict: bool, load_ema: bool) -> None:
        """仅加载权重作迁移初始化：不动 optimizer/scheduler/scaler/RNG，不推进 epoch，
        重对齐 EMA shadow 以免带着随机初始泄露。"""
        logger.info(
            "Loading pretrain weights: %s (strict=%s, load_ema=%s)",
            path, strict, load_ema)
        # weights_only=False：供手动指定的可信质类 ckpt。
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        sd, source = self._extract_model_state_dict(ckpt, prefer_ema=load_ema)
        sd = self._strip_common_prefixes(sd)

        bare = _unwrap_compile(self.model)
        result = bare.load_state_dict(sd, strict=strict)
        missing = list(getattr(result, "missing_keys", []) or [])
        unexpected = list(getattr(result, "unexpected_keys", []) or [])

        # 紧凑诊断：只示前几个 key。
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

        # 对齐 EMA shadow 到加载权重，避免 EMA 验证初期回反选随机初始。
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