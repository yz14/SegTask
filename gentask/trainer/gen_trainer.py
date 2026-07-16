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

import copy
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from ..data.augment import GPUAugmentor
from ..losses.recon import DiffusionLoss, build_recon_loss, psnr, ssim
from ..models.generation import DiffusionModel
from ..utils import AverageMeter, ModelEMA
from .amp import GradScaler, resolve_auto_amp_dtype
from .checkpoint import (
    extract_model_state_dict, restore_rng_state, snapshot_rng_state,
    strip_common_prefixes, unwrap_compile,
)
from .optim import WarmupScheduler, build_optimizer, build_scheduler
from .pipelines import build_pipeline
from .prefetch import CudaPrefetcher

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

        # 可选 channels_last 内存格式（数值等价；输入由 conv 自动适配）。
        if tc.channels_last:
            fmt = (torch.channels_last_3d
                   if int(cfg.model.spatial_dims) == 3
                   else torch.channels_last)
            self.model = self.model.to(memory_format=fmt)

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

        # --- torch.compile（最后：optimizer / EMA 已绑裸参数）----------
        self._compile_enabled = False
        if tc.compile_mode != "none" and hasattr(torch, "compile"):
            triton_ok = True
            if device.type == "cuda":
                import importlib.util
                if importlib.util.find_spec("triton") is None:
                    triton_ok = False
                    logger.warning(
                        "torch.compile (mode='%s') requested but Triton not "
                        "installed; falling back to eager. Install Triton or "
                        "set compile_mode='none'.", tc.compile_mode)
            if triton_ok:
                logger.info("Compiling model with mode='%s'", tc.compile_mode)
                self.model = torch.compile(self.model, mode=tc.compile_mode)
                self._compile_enabled = True

        self.output_dir = Path(tc.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # 选模指标：PSNR 越大越好；启用整卷验证时改用整卷 PSNR（M13）。
        self.best_metric = -math.inf
        self.best_epoch = 0
        self.val_full_volume = bool(tc.val_full_volume)
        self._vol_predictor = None
        self.history: list = []

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
        bd_meters: Dict[str, AverageMeter] = {}
        accum = self.grad_accum_steps
        total = len(self.train_loader)
        self.optimizer.zero_grad(set_to_none=True)
        group_bad = False   # 当前 accum 组内出现非有限 loss，整组丢弃
        skipped = 0
        # prefetch_to_gpu：独立 copy stream 提前一个 batch 上卡，H2D 与计算
        # 重叠。交付的张量已在 device 上，下方 .to(device) 退化为 no-op。
        batch_iter = self.train_loader
        if self.cfg.train.prefetch_to_gpu and self.device.type == "cuda":
            batch_iter = CudaPrefetcher(self.train_loader, self.device)
        for step, batch in enumerate(batch_iter):
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
            loss_val = float(loss.item())
            if not math.isfinite(loss_val):
                # 非有限 loss：不反传，标记整个 accum 组丢弃，保护权重/EMA。
                group_bad = True
                skipped += 1
                logger.warning("Non-finite loss (%.4g) at step %d/%d; "
                               "dropping current accum group.",
                               loss_val, step + 1, total)
            else:
                loss_scaled = loss / accum if accum > 1 else loss
                self.scaler.scale(loss_scaled).backward()

            if (step + 1) % accum == 0 or (step + 1) == total:
                stepped = False
                if group_bad:
                    group_bad = False
                else:
                    self.scaler.unscale_(self.optimizer)
                    clip = self.cfg.train.grad_clip_norm
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        clip if clip > 0 else float("inf"))
                    if torch.isfinite(grad_norm):
                        self.scaler.step(self.optimizer)
                        stepped = True
                    else:
                        skipped += 1
                        logger.warning(
                            "Non-finite grad norm at step %d/%d; skipping "
                            "optimizer step.", step + 1, total)
                    self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                # scheduler 无条件计时，保持 horizon 与总优化步对齐。
                self.scheduler.step()
                if stepped and self.ema is not None:
                    self.ema.update(unwrap_compile(self.model))

            if math.isfinite(loss_val):
                loss_meter.update(loss_val, hr.shape[0])
                for k, v in bd.items():
                    bd_meters.setdefault(k, AverageMeter()).update(
                        v, hr.shape[0])
            if (step + 1) % max(self.cfg.train.log_every, 1) == 0 or step == 0:
                logger.debug("  [%d/%d] loss=%.4f lr=%.2e",
                             step + 1, total, loss_val, self.scheduler.get_lr())
        if skipped:
            logger.warning("Epoch %d: %d non-finite loss/grad event(s) skipped.",
                           epoch + 1, skipped)
        out = {"loss": loss_meter.avg}
        # 分项损失均值进 history/日志（单项时与总 loss 重复，省略）。
        if len(bd_meters) > 1:
            out.update({k: m.avg for k, m in sorted(bd_meters.items())})
            logger.info("  loss breakdown: %s",
                        ", ".join(f"{k}={m.avg:.4f}"
                                  for k, m in sorted(bd_meters.items())))
        return out

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        if self.ema is not None:
            self.ema.apply_shadow(unwrap_compile(self.model))
        psnr_m, ssim_m, base_m = AverageMeter(), AverageMeter(), AverageMeter()
        bare0 = unwrap_compile(self.model)
        if isinstance(bare0, DiffusionModel):
            # 扩散采样用固定 seed 的 RNG：逐 epoch 指标确定可比，
            # 选模/早停/plateau 不被采样噪声干扰。
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(self.cfg.train.seed))
            bare0.sample_generator = gen
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
                if lr_m.shape[2:] != hr_m.shape[2:]:
                    # post-upsampling SISR：真 LR 尺寸的基线先线性插回 HR 网格。
                    mode = "bilinear" if self.spatial_dims == 2 else "trilinear"
                    lr_m = F.interpolate(
                        lr_m, size=hr_m.shape[2:], mode=mode,
                        align_corners=False)
                if self._minmax:  # zscore 无固定值域，不钳位。
                    rec = rec.clamp(0.0, 1.0)
                    lr_m = lr_m.clamp(0.0, 1.0)
                dr = self.val_data_range
                psnr_m.update(float(psnr(rec, hr_m, data_range=dr)), hr_m.shape[0])
                ssim_m.update(
                    float(ssim(rec, hr_m, self.spatial_dims, data_range=dr)),
                    hr_m.shape[0])
                base_m.update(float(psnr(lr_m, hr_m, data_range=dr)), hr_m.shape[0])
            metrics = {"psnr": psnr_m.avg, "ssim": ssim_m.avg,
                       "psnr_lr": base_m.avg}
            if self.val_full_volume:
                metrics.update(self._validate_volumes())
        finally:
            if isinstance(bare0, DiffusionModel):
                bare0.sample_generator = None
            if self.ema is not None:
                self.ema.restore(unwrap_compile(self.model))
        return metrics

    # ------------------------------------------------------------------
    # 整卷验证（M13）：与部署同口径，复用推理器滑窗路径
    # ------------------------------------------------------------------
    def _get_vol_predictor(self):
        """懒构造整卷验证用推理器（与 self.model 共享权重，EMA 影子生效）。"""
        if self._vol_predictor is None:
            from ..predictor.gen_predictor import GenerationPredictor
            cfg = copy.deepcopy(self.cfg)
            # 验证输入由在线退化产生：SISR（post-upsampling）吃真 LR 网格，
            # 其余已在 HR 网格；不走逐体 spacing 覆盖。
            cfg.predict.input_grid = (
                "lr" if str(cfg.model.arch).lower() in ("edsr", "rcan")
                else "hr")
            cfg.predict.target_z_spacing = 0.0
            self._vol_predictor = GenerationPredictor(
                self.model, cfg, self.device)
        return self._vol_predictor

    def _validate_volumes(self) -> Dict[str, float]:
        """整卷验证：退化整卷 → 滑窗复原 → 逐卷 PSNR/SSIM 平均。

        退化在 no_grad 下固定用基础核/噪声（随机池不生效），指标可比。"""
        ds = self.val_loader.dataset
        pred = self._get_vol_predictor()
        bare = unwrap_compile(self.model)
        n = len(ds._npz_paths)
        limit = int(self.cfg.train.val_full_volume_max)
        if limit > 0:
            n = min(n, limit)
        psnr_m, ssim_m, base_m = AverageMeter(), AverageMeter(), AverageMeter()
        dr = self.val_data_range
        for i in range(n):
            hr_np = ds._load_image(i)
            cond_np = ds._load_cond(i)
            hr = torch.from_numpy(
                np.ascontiguousarray(hr_np)).float().to(self.device)
            if pred.net_upsamples:
                # post-upsampling SISR：裁到倍率整除，复原网格与 HR 对齐。
                sc = pred.vol_axis_scales
                crop = [(s // k) * k for s, k in zip(hr.shape, sc)]
                hr = hr[:crop[0], :crop[1], :crop[2]]
                if cond_np is not None:
                    cond_np = cond_np[:, :crop[0], :crop[1], :crop[2]]
            # 2.5D（未 lift）退化作用在 (H,W)，z 当通道；3D/lift 作用在 (D,H,W)。
            x = hr[None, None] if self.spatial_dims == 3 else hr[None]
            lr = bare.degrade(x)
            lr_vol = lr[0, 0] if lr.ndim == 5 else lr[0]  # (d,h,w)
            rec_np = pred.restore_volume(
                lr_vol.float().cpu().numpy(), cond_vol=cond_np)
            rec = torch.from_numpy(rec_np).to(self.device)[None, None]
            hr_t = hr[None, None]
            lr_up = lr_vol[None, None]
            if lr_up.shape[2:] != hr_t.shape[2:]:
                # 真 LR 尺寸基线先线性插回 HR 网格（与 patch 级口径一致）。
                lr_up = F.interpolate(
                    lr_up, size=hr_t.shape[2:], mode="trilinear",
                    align_corners=False)
            if self._minmax:
                rec = rec.clamp(0.0, 1.0)
                lr_up = lr_up.clamp(0.0, 1.0)
            psnr_m.update(float(psnr(rec, hr_t, data_range=dr)))
            ssim_m.update(float(ssim(rec, hr_t, 3, data_range=dr)))
            base_m.update(float(psnr(lr_up, hr_t, data_range=dr)))
        return {"vol_psnr": psnr_m.avg, "vol_ssim": ssim_m.avg,
                "vol_psnr_lr": base_m.avg}

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

    # ------------------------------------------------------------------
    # 断点续训（M14）：last_checkpoint 保存完整训练状态
    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int) -> None:
        """周期保存可续训 checkpoint（模型 + optimizer/scheduler/scaler/EMA）。"""
        bare = unwrap_compile(self.model)
        state = {
            "epoch": epoch,
            "model_state_dict": bare.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            # RNG 快照：torch CPU/CUDA + numpy + python，支持位精确 resume。
            "rng_state": snapshot_rng_state(),
            "config": self.cfg,
        }
        if self.ema is not None:
            state["ema_state_dict"] = self.ema.state_dict()
        path = self.output_dir / "last_checkpoint.pth"
        tmp = path.with_suffix(".pth.tmp")  # 原子替换，防中断写坏
        torch.save(state, tmp)
        tmp.replace(path)
        logger.info("Checkpoint saved @ epoch %d -> %s", epoch + 1, path)

    def _save_history(self) -> None:
        """逐 epoch 指标落盘 history.json（原子替换）。"""
        path = self.output_dir / "history.json"
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        tmp.replace(path)

    def _load_resume(self, path: str) -> int:
        """从 checkpoint 恢复完整训练状态；返回续训起始 epoch。"""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        bare = unwrap_compile(self.model)
        bare.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if self.ema is not None and "ema_state_dict" in ckpt:
            self.ema.load_state_dict(ckpt["ema_state_dict"])
        self.best_metric = float(ckpt.get("best_metric", -math.inf))
        self.best_epoch = int(ckpt.get("best_epoch", 0))
        # 恢复 RNG；旧版 ckpt 无该键时静默跳过（训练仍正常但非位精确）。
        rng = ckpt.get("rng_state")
        if rng:
            try:
                restore_rng_state(rng)
                logger.info("Restored RNG state from checkpoint.")
            except Exception as e:  # pragma: no cover
                logger.warning("Failed to restore RNG state: %s", e)
        start = int(ckpt.get("epoch", -1)) + 1
        hist_path = self.output_dir / "history.json"
        if hist_path.exists():
            with open(hist_path, "r", encoding="utf-8") as f:
                self.history = [h for h in json.load(f)
                                if int(h.get("epoch", 0)) <= start]
        logger.info(
            "Resumed from %s: start_epoch=%d, best_PSNR=%.3f @ epoch %d",
            path, start + 1, self.best_metric, self.best_epoch + 1)
        return start

    def _load_pretrain(self, path: str, strict: bool, load_ema: bool) -> None:
        """仅加载权重作迁移初始化：不动 optimizer/scheduler/scaler/RNG，
        不推进 epoch，重对齐 EMA shadow 以免带着随机初始泄露。"""
        logger.info(
            "Loading pretrain weights: %s (strict=%s, load_ema=%s)",
            path, strict, load_ema)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        sd, source = extract_model_state_dict(ckpt, prefer_ema=load_ema)
        sd = strip_common_prefixes(sd)

        bare = unwrap_compile(self.model)
        result = bare.load_state_dict(sd, strict=strict)
        missing = list(getattr(result, "missing_keys", []) or [])
        unexpected = list(getattr(result, "unexpected_keys", []) or [])

        def _preview(keys, n=8):
            head = ", ".join(keys[:n])
            return head + (f", ... (+{len(keys) - n} more)"
                           if len(keys) > n else "")

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

    def fit(self) -> Dict[str, float]:
        self._check_supported_data_options(self.cfg)
        tc = self.cfg.train
        # resume：全状态恢复；pretrain：仅加载权重。同设优先 resume。
        resume_active = bool(tc.resume) and os.path.isfile(tc.resume)
        pretrain_active = bool(tc.pretrain) and os.path.isfile(tc.pretrain)
        start_epoch = 0
        if resume_active:
            if tc.pretrain:
                logger.warning(
                    "Both `train.resume` and `train.pretrain` are set; "
                    "using resume (%s). Pretrain weights from %s are ignored.",
                    tc.resume, tc.pretrain)
            start_epoch = self._load_resume(tc.resume)
        elif pretrain_active:
            self._load_pretrain(
                tc.pretrain,
                strict=tc.pretrain_strict,
                load_ema=tc.pretrain_load_ema)
        else:
            if tc.resume and not os.path.isfile(tc.resume):
                logger.warning(
                    "`train.resume` is set but file not found: %s. "
                    "Training will start from scratch.", tc.resume)
            if tc.pretrain and not os.path.isfile(tc.pretrain):
                logger.warning(
                    "`train.pretrain` is set but file not found: %s. "
                    "Training will start from scratch.", tc.pretrain)
        val_every = max(int(tc.val_every), 1)
        save_every = max(int(tc.save_every), 1)
        last: Dict[str, float] = {}
        for epoch in range(start_epoch, tc.epochs):
            tr = self._train_epoch(epoch)
            is_last = (epoch + 1) == tc.epochs
            do_val = ((epoch + 1) % val_every == 0) or is_last
            if do_val:
                val = self._validate(epoch)
                # 启用整卷验证时，退火/选模以整卷 PSNR 为准（与部署一致）。
                select = val.get("vol_psnr", val["psnr"])
                self.scheduler.step_epoch(select)
                logger.info(
                    "Epoch %d/%d: train_loss=%.4f val_PSNR=%.3f (LR=%.3f) val_SSIM=%.4f",
                    epoch + 1, tc.epochs, tr["loss"],
                    val["psnr"], val["psnr_lr"], val["ssim"])
                if "vol_psnr" in val:
                    logger.info(
                        "  full-volume: PSNR=%.3f (LR=%.3f) SSIM=%.4f",
                        val["vol_psnr"], val["vol_psnr_lr"], val["vol_ssim"])
                if select > self.best_metric:
                    self.best_metric = select
                    self.best_epoch = epoch
                    self._save_best(epoch)
                last = {**tr, **val}
            else:
                logger.info("Epoch %d/%d: train_loss=%.4f (val skipped, "
                            "val_every=%d)", epoch + 1, tc.epochs,
                            tr["loss"], val_every)
                last = {**tr}
            self.history.append(
                {"epoch": epoch + 1, "lr": self.scheduler.get_lr(), **last})
            self._save_history()
            if ((epoch + 1) % save_every == 0) or is_last:
                self._save_checkpoint(epoch)
            # 早停：连续 early_stopping 个 epoch 无提升即止（按 epoch 计）。
            if (tc.early_stopping > 0 and do_val
                    and epoch - self.best_epoch >= int(tc.early_stopping)):
                logger.info(
                    "Early stopping @ epoch %d (no improvement for %d "
                    "epochs; best @ epoch %d).",
                    epoch + 1, epoch - self.best_epoch, self.best_epoch + 1)
                self._save_checkpoint(epoch)
                break
        return {"best_psnr": self.best_metric, "best_epoch": self.best_epoch,
                **last}


__all__ = ["GenerationTrainer"]
