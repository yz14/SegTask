"""生成（超分）训练器。

分割 ``Trainer`` / 评估器与 Dice 深度耦合，不适合直接承载生成任务；本模块提供
独立但精简的 ``GenerationTrainer``，复用既有训练基建（优化器 / 调度器 / warmup /
AMP / EMA），针对图像复原实现训练与验证：

* 训练：从干净图 ``hr`` 在线退化得低分条件图，前向 → 回归 / 扩散损失 → 反传。
* 验证：``model.restore(lr)`` 复原后算 PSNR / SSIM，并报告 LR 基线 PSNR 作参照。

模型由 ``models.generation.build_generation_model`` 构造，对外暴露
``forward(hr) / restore(lr) / degrade(hr)`` 统一接口（见该模块）。
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from ..data.augment import GPUAugmentor
from ..losses.recon import DiffusionLoss, build_recon_loss, psnr, ssim
from ..utils import AverageMeter, ModelEMA
from .amp import GradScaler, resolve_auto_amp_dtype
from .checkpoint import unwrap_compile
from .optim import WarmupScheduler, build_optimizer, build_scheduler
from .pipelines import build_pipeline

logger = logging.getLogger(__name__)

_AMP_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16,
               "float32": torch.float32}


class GenerationTrainer:
    """超分生成训练器（回归 / 扩散）。"""

    def __init__(
        self,
        model: nn.Module,
        cfg: Config,
        train_loader,
        val_loader,
        device: torch.device):
        self.cfg = cfg
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(device)
        tc = cfg.train

        self.is_diffusion = str(cfg.task.algorithm).lower() == "diffusion"
        # 单一真相源：topology 派生量（sync() 写入），2.5D+lift 时为 3。
        self.spatial_dims = int(cfg.model.spatial_dims)
        # 验证指标的像素动态范围与损失侧一致：minmax≈[0,1]，zscore≈±1。
        self._minmax = str(cfg.data.normalize).lower() == "minmax"
        self.val_data_range = 1.0 if self._minmax else 2.0
        if self.is_diffusion:
            self.loss_fn: nn.Module = DiffusionLoss()
        else:
            self.loss_fn = build_recon_loss(cfg)

        # batch 几何准备管线（过采样余量裁剪 / 多视图拆分打包）与 GPU 增强。
        self.pipeline = build_pipeline(cfg)
        scales = cfg.data.multi_res_scales or [1.0]
        self.augmentor = (GPUAugmentor(cfg.augment,
                                       max_scale=float(max(scales)))
                          if cfg.augment.enabled else None)

        self.optimizer = build_optimizer(self.model, cfg)
        self.grad_accum_steps = max(tc.grad_accum_steps, 1)
        # scheduler.step() 每个优化器步触发一次，horizon 按优化器步计（而非
        # 批次数），否则 grad_accum_steps>1 时 warmup/退火时长被放大 accum 倍。
        steps_per_epoch = max(len(train_loader), 1)
        updates_per_epoch = max(math.ceil(steps_per_epoch / self.grad_accum_steps), 1)
        warmup_steps = tc.warmup_epochs * updates_per_epoch
        if tc.scheduler == "one_cycle" and warmup_steps > 0:
            # OneCycleLR 自带 warmup（pct_start），不叠加外层线性 warmup。
            logger.info("scheduler='one_cycle' has built-in warmup; "
                        "outer linear warmup disabled.")
            warmup_steps = 0
        total_steps = tc.epochs * updates_per_epoch
        base_scheduler = build_scheduler(
            self.optimizer, cfg, updates_per_epoch,
            post_warmup_steps=total_steps - warmup_steps)
        self.scheduler = WarmupScheduler(
            self.optimizer, base_scheduler, warmup_steps=warmup_steps,
            warmup_lr=tc.warmup_lr, base_lr=tc.lr)

        amp_name = tc.amp_dtype
        if amp_name == "auto":
            amp_name = resolve_auto_amp_dtype(device)
        self.amp_dtype = _AMP_DTYPES.get(amp_name, torch.float32)
        self.use_amp = tc.use_amp and device.type == "cuda"
        self.scaler = GradScaler(
            "cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)

        self.ema = ModelEMA(self.model, tc.ema_decay) if tc.use_ema else None

        self.output_dir = Path(tc.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # 选模指标：PSNR 越大越好。
        self.best_metric = -math.inf
        self.best_epoch = 0

    # ------------------------------------------------------------------
    @staticmethod
    def _check_supported_data_options(cfg: Config) -> None:
        """拒绝本训练器尚无法消费的数据配置（fit 入口检查）。

        多视图 / 过采样已由 trainer/pipelines 消费；仅剩扩散算法的多视图
        （DiffusionModel 无多视图目标对齐逻辑）显式报错而非静默错训。
        """
        if (str(cfg.task.algorithm).lower() == "diffusion"
                and len(cfg.data.multi_res_scales) > 1):
            raise NotImplementedError(
                "DiffusionModel does not support multi-view inputs "
                f"(multi_res_scales={cfg.data.multi_res_scales}); use "
                "multi_res_scales=[1.0] or task.algorithm='regression'.")

    def _hr_batch(self, batch) -> torch.Tensor:
        """取干净高分图（忽略分割标签）。"""
        return batch["image"].to(self.device, non_blocking=True).float()

    def _cond_batch(self, batch) -> Optional[torch.Tensor]:
        cond = batch.get("cond")
        if cond is None:
            return None
        return cond.to(self.device, non_blocking=True).float()

    def _align_weight_map(
        self,
        weight_map: Optional[torch.Tensor],
        ref: torch.Tensor) -> Optional[torch.Tensor]:
        if weight_map is None:
            return None
        w = weight_map.to(self.device, non_blocking=True).float()
        if ref.ndim == 4 and w.ndim == 5:
            w = w[:, 0] if w.shape[1] == 1 else w.flatten(1, 2)
        elif ref.ndim == 5 and w.ndim == 4:
            w = w.unsqueeze(1)

        if w.shape[-self.spatial_dims:] != ref.shape[-self.spatial_dims:]:
            try:
                w = F.interpolate(
                    w, size=tuple(ref.shape[-self.spatial_dims:]), mode="area")
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"weight_map shape {tuple(weight_map.shape)} cannot be "
                    f"aligned to reference shape {tuple(ref.shape)}.") from exc

        try:
            torch.broadcast_shapes(w.shape, ref.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"weight_map shape {tuple(weight_map.shape)} cannot be "
                f"broadcast to reference shape {tuple(ref.shape)} after "
                "alignment.") from exc
        return w

    def _step_loss(
        self,
        out: Dict[str, torch.Tensor],
        breakdown,
        weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.is_diffusion:
            return self.loss_fn(out, breakdown=breakdown)
        preds = out.get("ds_preds")
        total = None
        if preds is not None and len(preds) > 1:
            total = self._ds_recon_loss(preds, out["target"], breakdown, weight_map)
        else:
            weight = self._align_weight_map(weight_map, out["target"])
            total = self.loss_fn(out["pred"], out["target"], weight=weight,
                                 breakdown=breakdown)
        aux_preds = out.get("aux_preds")
        aux_targets = out.get("aux_targets")
        if aux_preds and aux_targets:
            aux_term = self._aux_recon_loss(aux_preds, aux_targets)
            total = total + aux_term
            if breakdown is not None:
                breakdown["L_aux"] = float(aux_term.detach().item())
        return total

    def _ds_recon_loss(
        self,
        preds,
        hr: torch.Tensor,
        breakdown,
        weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        """深监督重建损失：逐尺度与下采样后的 HR 算 recon，按归一化权重聚合。"""
        raw = list(self.cfg.loss.deep_supervision_weights[:len(preds)])
        if not raw:
            raw = [0.5 ** k for k in range(len(preds))]
        denom = float(sum(raw)) or 1.0
        weights = [w / denom for w in raw]
        total = hr.new_zeros(())
        for k, (w, p) in enumerate(zip(weights, preds)):
            if tuple(p.shape[-self.spatial_dims:]) == tuple(hr.shape[-self.spatial_dims:]):
                tgt = hr
            else:
                tgt = F.interpolate(
                    hr, size=tuple(p.shape[-self.spatial_dims:]), mode="area")
            wmap = self._align_weight_map(weight_map, tgt)
            bd = breakdown if k == 0 else None
            total = total + w * self.loss_fn(p, tgt, weight=wmap, breakdown=bd)
        return total

    def _aux_recon_loss(self, preds, targets) -> torch.Tensor:
        raw = list(self.cfg.loss.aux_recon_weights[:len(preds)])
        if not raw:
            raw = [0.5 ** k for k in range(len(preds))]
        denom = float(sum(raw)) or 1.0
        weights = [w / denom for w in raw]
        total = preds[0].new_zeros(())
        for w, pred, target in zip(weights, preds, targets):
            total = total + w * self.loss_fn(pred, target, breakdown=None)
        return total

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        loss_meter = AverageMeter()
        accum = self.grad_accum_steps
        total = len(self.train_loader)
        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(self.train_loader):
            hr = self._hr_batch(batch)
            cond = self._cond_batch(batch)
            weight_map = batch.get("weight_map")
            if weight_map is not None:
                weight_map = weight_map.to(self.device, non_blocking=True).float()
            if self.augmentor is not None:
                hr, weight_map, cond = self.augmentor(hr, weight_map, cond)
            hr, weight_map, cond = self.pipeline.prepare_batch(
                hr, weight_map, cond)
            with torch.autocast(device_type=self.device.type,
                                enabled=self.use_amp, dtype=self.amp_dtype):
                out = self.model(hr, cond=cond)
            bd: Dict[str, float] = {}
            loss = self._step_loss(out, bd, weight_map=weight_map)
            loss_scaled = loss / accum if accum > 1 else loss
            self.scaler.scale(loss_scaled).backward()

            if (step + 1) % accum == 0 or (step + 1) == total:
                if self.cfg.train.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.train.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                if self.ema is not None:
                    self.ema.update(self.model)

            if math.isfinite(loss.item()):
                loss_meter.update(loss.item(), hr.shape[0])
            if (step + 1) % max(self.cfg.train.log_every, 1) == 0 or step == 0:
                logger.debug("  [%d/%d] loss=%.4f lr=%.2e",
                             step + 1, total, loss.item(), self.scheduler.get_lr())
        return {"loss": loss_meter.avg}

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        if self.ema is not None:
            self.ema.apply_shadow(self.model)
        psnr_m, ssim_m, base_m = AverageMeter(), AverageMeter(), AverageMeter()
        try:
            for batch in self.val_loader:
                hr = self._hr_batch(batch)
                cond = self._cond_batch(batch)
                hr, _, cond = self.pipeline.prepare_batch(hr, None, cond)
                bare = unwrap_compile(self.model)
                lr = bare.degrade(hr)
                rec = bare.restore(lr, cond=cond)
                # 多视图时模型只复原主视图；hr / lr 裁对齐到 rec 通道布局。
                rec, hr_m, lr_m = self.pipeline.metric_views(rec, hr, lr)
                if self._minmax:  # zscore 无固定值域，不钳位。
                    rec = rec.clamp(0.0, 1.0)
                    lr_m = lr_m.clamp(0.0, 1.0)
                dr = self.val_data_range
                psnr_m.update(float(psnr(rec, hr_m, data_range=dr)), hr_m.shape[0])
                ssim_m.update(
                    float(ssim(rec, hr_m, self.spatial_dims, data_range=dr)),
                    hr_m.shape[0])
                base_m.update(float(psnr(lr_m, hr_m, data_range=dr)), hr_m.shape[0])
        finally:
            if self.ema is not None:
                self.ema.restore(self.model)
        return {"psnr": psnr_m.avg, "ssim": ssim_m.avg, "psnr_lr": base_m.avg}

    def _save_best(self, epoch: int) -> None:
        bare = unwrap_compile(self.model)
        state = {
            "epoch": epoch,
            "model_state_dict": bare.state_dict(),
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "config": self.cfg,
        }
        if self.ema is not None:
            state["ema_state_dict"] = self.ema.state_dict()
        torch.save(state, self.output_dir / "best_model.pth")
        logger.info("Best generation model saved (PSNR=%.3f) @ epoch %d",
                    self.best_metric, epoch + 1)

    def fit(self) -> Dict[str, float]:
        self._check_supported_data_options(self.cfg)
        last: Dict[str, float] = {}
        for epoch in range(self.cfg.train.epochs):
            tr = self._train_epoch(epoch)
            val = self._validate(epoch)
            self.scheduler.step_epoch(val.get("psnr"))
            logger.info(
                "Epoch %d/%d: train_loss=%.4f val_PSNR=%.3f (LR=%.3f) val_SSIM=%.4f",
                epoch + 1, self.cfg.train.epochs, tr["loss"],
                val["psnr"], val["psnr_lr"], val["ssim"])
            if val["psnr"] > self.best_metric:
                self.best_metric = val["psnr"]
                self.best_epoch = epoch
                self._save_best(epoch)
            last = {**tr, **val}
        return {"best_psnr": self.best_metric, "best_epoch": self.best_epoch,
                **last}


__all__ = ["GenerationTrainer"]
