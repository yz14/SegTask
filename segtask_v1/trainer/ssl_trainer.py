"""自监督预训练（SSL）训练循环。

精简版 Trainer：无分割 label / Dice / 验证集滑窗，仅"破坏→重建"的重建损失。
复用分割侧的 optimizer / scheduler / AMP / EMA / ckpt 工具，配置全部读 ``train.*``
（epochs/lr/output_dir/...）+ ``ssl.*``（破坏参数 / 重建损失）。

产出的 ckpt 含 ``model_state_dict``（``encoder.*`` / ``decoder.*`` / ``recon_head.*``），
可经分割侧 ``train.pretrain`` 非严格加载衔接（enc+dec 命中、seg head 随机）。
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import Config
from ..data.ssl_transforms import GenesisCorruptor
from ..data.vesselness import frangi_vesselness
from ..utils import AverageMeter, ModelEMA, Timer
from .amp import _AMP_DTYPES, GradScaler, autocast, resolve_auto_amp_dtype
from .checkpoint import unwrap_compile
from .optim import WarmupScheduler, build_optimizer, build_scheduler

logger = logging.getLogger(__name__)

_RECON_LOSSES = {
    "l1": F.l1_loss,
    "smooth_l1": F.smooth_l1_loss,
    "mse": F.mse_loss,
}


def _center_crop_spatial(x: torch.Tensor, target: Tuple[int, ...]) -> torch.Tensor:
    """沿末尾 spatial 维中心裁到 target（仅当当前更大）。"""
    spatial = x.shape[2:]
    slices = [slice(None), slice(None)]
    for cur, tgt in zip(spatial, target):
        if cur > tgt:
            start = (cur - tgt) // 2
            slices.append(slice(start, start + tgt))
        else:
            slices.append(slice(None))
    return x[tuple(slices)]


class SSLTrainer:
    """Models Genesis 式自监督预训练 pipeline。"""

    def __init__(
        self,
        model: nn.Module,
        cfg: Config,
        train_loader: DataLoader,
        device: torch.device):
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        tc = cfg.train

        self.method = cfg.ssl.method
        self.spatial_dims = int(cfg.model.spatial_dims)
        self.corruptor = GenesisCorruptor(cfg.ssl, self.spatial_dims)
        self.recon_loss_fn = _RECON_LOSSES[cfg.ssl.recon_loss]
        self.target_spatial = tuple(
            int(s) for s in cfg.data.patch_size[-int(cfg.model.spatial_dims):])
        self.needs_crop = cfg.data.aug_oversample_ratio > 1.0

        # --- Optimizer + scheduler (复用分割侧 train.*) ---
        self.optimizer = build_optimizer(self.model, cfg)
        steps_per_epoch = len(train_loader)
        warmup_steps = tc.warmup_epochs * steps_per_epoch
        total_steps = tc.epochs * steps_per_epoch
        post_warmup = total_steps - warmup_steps
        if tc.scheduler == "one_cycle" and warmup_steps > 0:
            raise ValueError(
                "OneCycleLR has built-in warmup; set train.warmup_epochs=0.")
        base_scheduler = build_scheduler(
            self.optimizer, cfg, steps_per_epoch, post_warmup_steps=post_warmup)
        self.scheduler = WarmupScheduler(
            self.optimizer, base_scheduler, warmup_steps=warmup_steps,
            warmup_lr=tc.warmup_lr, base_lr=tc.lr)

        # --- AMP ---
        amp_dtype_cfg = tc.amp_dtype
        if amp_dtype_cfg == "auto":
            amp_dtype_cfg = resolve_auto_amp_dtype(device)
        if amp_dtype_cfg not in _AMP_DTYPES:
            raise ValueError(f"Unknown amp_dtype: {tc.amp_dtype!r}.")
        self.amp_dtype = _AMP_DTYPES[amp_dtype_cfg]
        self.use_amp = tc.use_amp and device.type == "cuda"
        self._scaler_active = self.use_amp and self.amp_dtype == torch.float16
        self.scaler = GradScaler("cuda", enabled=self._scaler_active)

        # --- EMA ---
        self.ema = ModelEMA(self.model, tc.ema_decay) if tc.use_ema else None

        self.grad_accum_steps = max(tc.grad_accum_steps, 1)
        self.output_dir = Path(tc.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = math.inf

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _make_io(self, clean: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """按 method 构造 (model_input, target)。

        - genesis: input = 破坏图, target = 干净图（重建）。
        - prior  : input = 干净图(可选破坏), target = 干净图的 Frangi vesselness（回归）。
        """
        s = self.cfg.ssl
        if self.method == "genesis":
            return self.corruptor(clean), clean
        target = frangi_vesselness(
            clean, scales=s.prior_scales, spatial_dims=self.spatial_dims,
            alpha=s.prior_alpha, beta=s.prior_beta,
            black_vessels=s.prior_black_vessels)
        model_input = self.corruptor(clean) if s.prior_corrupt_input else clean
        return model_input, target

    # ------------------------------------------------------------------
    def _state_dict(self) -> Dict:
        """EMA 优先的权重快照（与分割侧 extract_model_state_dict 兼容）。"""
        bare = unwrap_compile(self.model)
        if self.ema is not None:
            self.ema.apply_shadow(self.model)
            try:
                sd = {k: v.detach().cpu().clone()
                      for k, v in unwrap_compile(self.model).state_dict().items()}
            finally:
                self.ema.restore(self.model)
            return sd
        return {k: v.detach().cpu().clone() for k, v in bare.state_dict().items()}

    def _save(self, epoch: int, tag: str) -> Path:
        path = self.output_dir / f"ssl_{tag}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self._state_dict(),
            "ssl_method": self.cfg.ssl.method,
            "best_loss": self.best_loss,
        }, path)
        return path

    # ------------------------------------------------------------------
    def fit(self) -> Dict[str, float]:
        tc = self.cfg.train
        timer = Timer()
        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info("=" * 60)
        logger.info("SSL pretrain (%s): %d epochs, device=%s",
                    self.cfg.ssl.method, tc.epochs, self.device)
        logger.info("Model params: %.2fM | recon_loss=%s | AMP=%s EMA=%s",
                    total_params, self.cfg.ssl.recon_loss,
                    self.use_amp, self.ema is not None)
        logger.info("Train batches: %d, output_dir=%s",
                    len(self.train_loader), self.output_dir)
        logger.info("=" * 60)

        for epoch in range(tc.epochs):
            train_loss = self._train_epoch(epoch)
            elapsed = timer.elapsed()
            logger.info("[SSL epoch %d/%d] recon_loss=%.5f lr=%.2e (%.1fs)",
                        epoch + 1, tc.epochs, train_loss,
                        self.scheduler.get_lr(), elapsed)

            if train_loss < self.best_loss:
                self.best_loss = train_loss
                p = self._save(epoch, "best")
                logger.info("★ New best SSL recon_loss=%.5f → %s",
                            train_loss, p)
            if (epoch + 1) % self.cfg.ssl.save_every == 0 or epoch == tc.epochs - 1:
                self._save(epoch, "last")

        logger.info("SSL pretrain done. best_loss=%.5f. "
                    "Use ssl_best.pt via train.pretrain for segmentation.",
                    self.best_loss)
        return {"best_recon_loss": self.best_loss}

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        loss_meter = AverageMeter()
        tc = self.cfg.train
        accum = self.grad_accum_steps
        total_steps = len(self.train_loader)

        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(self.train_loader):
            clean = batch["image"].to(self.device, non_blocking=True).float()
            if self.needs_crop:
                clean = _center_crop_spatial(clean, self.target_spatial)
            model_input, target = self._make_io(clean)

            with autocast(device_type="cuda", enabled=self.use_amp,
                          dtype=self.amp_dtype):
                pred = self.model(model_input)
            # 重建/回归损失在 fp32 计算，避免 fp16 汇总误差。
            loss = self.recon_loss_fn(pred.float(), target.float())
            if accum > 1:
                loss = loss / accum
            self.scaler.scale(loss).backward()

            is_boundary = ((step + 1) % accum == 0 or (step + 1) == total_steps)
            if is_boundary:
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

            step_loss = loss.item() * (accum if accum > 1 else 1)
            if math.isfinite(step_loss):
                loss_meter.update(step_loss, clean.shape[0])
            else:
                logger.warning(
                    "Non-finite SSL loss at epoch %d step %d/%d; skipping.",
                    epoch + 1, step + 1, total_steps)

            if (step + 1) % tc.log_every == 0 or step == 0:
                logger.debug("  [%d/%d] recon_loss=%.5f lr=%.2e",
                             step + 1, total_steps, step_loss,
                             self.scheduler.get_lr())
        return loss_meter.avg


__all__ = ["SSLTrainer"]
