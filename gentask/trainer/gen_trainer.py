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
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from ..data.augment import GPUAugmentor
from ..losses.recon import DiffusionLoss, build_recon_loss, psnr, ssim
from ..models.generation import DiffusionModel
from taskcore.utils.common import AverageMeter
from taskcore.engine.checkpoint import (
    AsyncCheckpointSaver, atomic_torch_save, snapshot_rng_state,
    state_to_cpu, unwrap_compile,
)
from .pipelines import build_pipeline
from taskcore.engine.dist_utils import (
    all_reduce_meters_, get_rank, shard_for_rank,
)
from taskcore.engine.prefetch import CudaPrefetcher
from taskcore.engine.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class GenerationTrainer(BaseTrainer):
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
        self._setup_channels_last()

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
        # 逐 rank 分流的独立增强 RNG（与 seg trainer 同构）；训练循环传入的
        # hr/weight_map/cond 均是 H2D 私有拷贝且增强后不再以原值复用，
        # 满足 inplace 所有权契约，省一次入口 clone。
        _aug_seed = (int(tc.seed) + 7919 * (get_rank() + 1)) & 0x7FFFFFFF
        self.augmentor = (GPUAugmentor(cfg.augment,
                                       max_scale=float(max(scales)),
                                       seed=_aug_seed, inplace=True)
                          if cfg.augment.enabled else None)

        # --- Optimizer + scheduler / AMP / EMA（共用工程件，见 BaseTrainer）---
        self._setup_optim_sched()
        self._setup_amp()
        self._setup_ema()

        # --- torch.compile（最后：optimizer / EMA 已绑裸参数）----------
        self._maybe_compile()

        # --- DDP 装配与训练采样器识别（公用工程件，见 BaseTrainer）-------
        # 单卡时 fwd_model 即裸模块，路径零变化；多卡时前向走 DDP 包装。
        self._setup_ddp()
        self._setup_train_sampler()

        self._setup_output_dir()
        # save_async=True 时权重先深拷到 CPU，后台线程 torch.save，主循环
        # 不再被写盘阻塞；fit 收尾 wait+close 保证全部落盘（同 seg）。
        self._ckpt_saver = (AsyncCheckpointSaver()
                            if tc.save_async and self._is_main else None)
        # 选模指标：PSNR 越大越好；启用整卷验证时改用整卷 PSNR（M13）。
        self._setup_best_tracking(mode="max")
        self.best_key = "psnr"
        self._ckpt_task_label = "generation"
        # SWA 尾段权重平均（opt-in，公用工程件，见 BaseTrainer）。
        self._setup_swa()
        self.val_full_volume = bool(tc.val_full_volume)
        self._vol_predictor = None
        self.history: list = []

        # --- 训练监测仪表盘（公用工程件，见 BaseTrainer / taskcore.monitor；
        #     cfg.monitor 守卫，失败隔离不阻断训练）-----------------------
        # 选模指标口径同 fit：PSNR 越大越好（整卷验证时 fit 内改用 vol_psnr，
        # is_best 标记以实际选模为准）。
        self._setup_monitor(
            resume_active=bool(tc.resume),
            run_name_default="gen_run",
            save_best_metric="psnr",
            save_best_mode="max",
            save_best_criterion="",
            num_classes=0,
            config_meta={
                "algorithm": cfg.task.algorithm,
                "loss": cfg.loss.name,
                "batch_size": cfg.data.batch_size,
            })

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
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(epoch)   # 多卡重洗（单卡 no-op）
        self.model.train()
        loss_meter = AverageMeter()
        bd_meters: Dict[str, AverageMeter] = {}
        tc = self.cfg.train
        accum = self.grad_accum_steps
        total = len(self.train_loader)
        self.optimizer.zero_grad(set_to_none=True)
        group_has_nonfinite = False
        skipped = 0
        grad_norm_meter = AverageMeter()
        grad_norm_max = 0.0
        nonfinite_steps = 0
        clipped_steps = 0
        opt_steps = 0

        # 未缩放损失先以 GPU 张量缓存，延迟到日志步 / bf16 边界单次同步
        # （与 seg trainer pending 同源），避免每 micro-step loss.item()。
        pending: "list[tuple[int, torch.Tensor, int]]" = []

        def _flush_pending() -> "float | None":
            nonlocal group_has_nonfinite, nonfinite_steps, skipped
            if not pending:
                return None
            vals = torch.stack([t for _, t, _ in pending]).tolist()
            last: "float | None" = None
            for (s, _, bs), v in zip(pending, vals):
                last = v
                if math.isfinite(v):
                    loss_meter.update(v, bs)
                else:
                    group_has_nonfinite = True
                    nonfinite_steps += 1
                    skipped += 1
                    logger.warning(
                        "Non-finite train loss (%s) at epoch %d step %d/%d; "
                        "surrounding optimizer step will be skipped (%s).",
                        v, epoch + 1, s + 1, total,
                        "by GradScaler" if self._scaler_active
                        else "non-finite loss guard")
            pending.clear()
            return last

        # prefetch_to_gpu：独立 copy stream 提前一个 batch 上卡，H2D 与计算
        # 重叠。交付的张量已在 device 上，下方 .to(device) 退化为 no-op。
        batch_iter = self.train_loader
        if tc.prefetch_to_gpu and self.device.type == "cuda":
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

            eff_accum = self._effective_accum(step, total, accum)
            is_step_boundary = ((step + 1) % accum == 0 or (step + 1) == total)
            # 非边界步免 all-reduce；forward 也必须放进 no_sync（与 seg 同）。
            sync_ctx = self._ddp_no_sync(is_step_boundary)
            is_log_step = ((step + 1) % max(tc.log_every, 1) == 0 or step == 0)
            bd: Dict[str, float] = {}

            # 恒反传：非有限 loss 也走 backward，保证 DDP 各 rank 集体计数一致；
            # 跳步决策在边界用 all_reduce_flag_any 统一（见下）。
            with sync_ctx:
                with torch.autocast(device_type=self.device.type,
                                    enabled=self.use_amp, dtype=self.amp_dtype):
                    out = self.fwd_model(hr, cond=cond)
                loss = self._step_loss(
                    out, bd if is_log_step else None, weight_map=weight_map)
                loss_scaled = loss / eff_accum if eff_accum > 1 else loss
                self.scaler.scale(loss_scaled).backward()

            pending.append((step, loss.detach(), hr.shape[0]))

            step_loss: "float | None" = None
            if is_log_step or (is_step_boundary and not self._scaler_active):
                step_loss = _flush_pending()
            if step_loss is not None and math.isfinite(step_loss):
                for k, v in bd.items():
                    if math.isfinite(v):
                        bd_meters.setdefault(k, AverageMeter()).update(
                            v, hr.shape[0])

            if is_step_boundary:
                result = self._optimizer_step_boundary(
                    group_has_nonfinite=group_has_nonfinite,
                    epoch=epoch, step=step, total_steps=total)
                group_has_nonfinite = False
                grad_norm_val = result.grad_norm
                skipped_nf = result.skipped_nonfinite
                result.acknowledge()
                if skipped_nf:
                    if self._health_monitor:
                        opt_steps += 1
                    continue
                opt_steps += 1
                if grad_norm_val is not None and math.isfinite(grad_norm_val):
                    grad_norm_meter.update(grad_norm_val)
                    grad_norm_max = max(grad_norm_max, grad_norm_val)
                    if (tc.grad_clip_norm > 0
                            and grad_norm_val > tc.grad_clip_norm):
                        clipped_steps += 1

            if is_log_step:
                logger.debug(
                    "  [%d/%d] loss=%.4f lr=%.2e",
                    step + 1, total,
                    step_loss if step_loss is not None else float("nan"),
                    self.scheduler.get_lr())

        _flush_pending()
        if skipped:
            logger.warning("Epoch %d: %d non-finite loss/grad event(s) skipped.",
                           epoch + 1, skipped)
        out = {"loss": loss_meter.avg}
        self._collect_health_metrics(
            out,
            grad_norm_meter=grad_norm_meter,
            grad_norm_max=grad_norm_max,
            nonfinite_steps=nonfinite_steps,
            clipped_steps=clipped_steps,
            opt_steps=opt_steps,
            grad_clip_norm=float(self.cfg.train.grad_clip_norm))
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
        with self._ema_swapped():
            return self._validate_inner(epoch)

    @torch.no_grad()
    def _swa_bn_forward(self) -> None:
        """SWA BN 重估用前向：未增强训练数据，与推理分布同构。"""
        steps = int(self.cfg.train.swa_bn_update_steps)
        for step, batch in enumerate(self.train_loader):
            if step >= steps:
                break
            hr = self._hr_batch(batch)
            cond = self._cond_batch(batch)
            hr, _, cond = self.pipeline.prepare_batch(hr, None, cond)
            with torch.autocast(device_type=self.device.type,
                                enabled=self.use_amp, dtype=self.amp_dtype):
                self.model(hr, cond=cond)

    def _validate_inner(self, epoch: int) -> Dict[str, float]:
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
            # DDP：val 集按 batch 块分片到各 rank，(sum, count) 跨 rank 求和
            # 后 avg 即全集加权均值，选模/早停决策各 rank 天然一致。
            all_reduce_meters_([psnr_m, ssim_m, base_m], self.device)
            metrics = {"psnr": psnr_m.avg, "ssim": ssim_m.avg,
                       "psnr_lr": base_m.avg}
            if self.val_full_volume:
                metrics.update(self._validate_volumes())
        finally:
            if isinstance(bare0, DiffusionModel):
                bare0.sample_generator = None
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
        # DDP：整卷列表不相交切给各 rank，逐卷指标经 all-reduce 汇总。
        for i in shard_for_rank(list(range(n))):
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
        all_reduce_meters_([psnr_m, ssim_m, base_m], self.device)
        return {"vol_psnr": psnr_m.avg, "vol_ssim": ssim_m.avg,
                "vol_psnr_lr": base_m.avg}

    # ------------------------------------------------------------------
    # 断点续训（M14）：last_checkpoint 保存完整训练状态
    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int) -> None:
        """周期保存可续训 checkpoint（模型 + optimizer/scheduler/scaler/EMA）。"""
        if not self._is_main:   # DDP：落盘仅 rank0
            return
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
        if self._ckpt_saver is not None:
            self._ckpt_saver.submit(
                state_to_cpu(state), path,
                on_done=lambda e=epoch, p=path: logger.info(
                    "Checkpoint saved @ epoch %d -> %s", e + 1, p))
        else:
            atomic_torch_save(state, path)  # 原子替换，防中断写坏
            logger.info("Checkpoint saved @ epoch %d -> %s", epoch + 1, path)

    def _save_history(self) -> None:
        """逐 epoch 指标落盘 history.json（原子替换）。"""
        if not self._is_main:   # DDP：落盘仅 rank0
            return
        path = self.output_dir / "history.json"
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        tmp.replace(path)

    def _load_resume(self, path: str) -> int:
        """从 checkpoint 恢复完整训练状态；返回续训起始 epoch。"""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        # 公共段（model/EMA/optim/sched/scaler/best/RNG）见 BaseTrainer。
        start = self._restore_train_state(ckpt)
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
        """仅加载权重作迁移初始化（公共策略见 BaseTrainer）。"""
        self._load_pretrain_weights(path, strict=strict, load_ema=load_ema)

    def fit(self) -> Dict[str, float]:
        self._check_supported_data_options(self.cfg)
        tc = self.cfg.train
        # resume：全状态恢复；pretrain：仅加载权重。同设优先 resume。
        # 显式路径不存在即报错（fail-fast，防静默从头训；口径同 cls/det）。
        start_epoch = 0
        if tc.resume:
            if tc.pretrain:
                logger.warning(
                    "Both `train.resume` and `train.pretrain` are set; "
                    "using resume (%s). Pretrain weights from %s are ignored.",
                    tc.resume, tc.pretrain)
            if not os.path.isfile(tc.resume):
                raise FileNotFoundError(
                    f"train.resume checkpoint not found: {tc.resume!r}")
            start_epoch = self._load_resume(tc.resume)
        elif tc.pretrain:
            if not os.path.isfile(tc.pretrain):
                raise FileNotFoundError(
                    f"train.pretrain checkpoint not found: {tc.pretrain!r}")
            self._load_pretrain(
                tc.pretrain,
                strict=tc.pretrain_strict,
                load_ema=tc.pretrain_load_ema)
        val_every = max(int(tc.val_every), 1)
        save_every = max(int(tc.save_every), 1)
        last: Dict[str, float] = {}
        final_status = "completed"
        for epoch in range(start_epoch, tc.epochs):
            epoch_t0 = time.time()
            tr = self._train_epoch(epoch)
            is_last = (epoch + 1) == tc.epochs
            do_val = ((epoch + 1) % val_every == 0) or is_last
            is_best = False
            val: "Dict[str, float] | None" = None
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
                    is_best = True
                    self.best_metric = select
                    self.best_epoch = epoch
                    # 与 BaseTrainer 对齐：metrics[best_key] 用于日志；选模值注入 psnr。
                    self._save_best(epoch, {**val, "psnr": float(select)})
                last = {**tr, **val}
            else:
                logger.info("Epoch %d/%d: train_loss=%.4f (val skipped, "
                            "val_every=%d)", epoch + 1, tc.epochs,
                            tr["loss"], val_every)
                last = {**tr}
            self.history.append(
                {"epoch": epoch + 1, "lr": self.scheduler.get_lr(), **last})
            self._swa_update(epoch)
            self._save_history()
            gpu_peak_mib = None
            if self.device.type == "cuda":
                gpu_peak_mib = (torch.cuda.max_memory_allocated(self.device)
                                / (1 << 20))
                torch.cuda.reset_peak_memory_stats(self.device)
            self._monitor_log_epoch(
                epoch, tr, val,
                lr=self.scheduler.get_lr(), gpu_peak_mib=gpu_peak_mib,
                wall_time_s=time.time() - epoch_t0, is_best=is_best,
                last_epoch=is_last)
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
                final_status = "early_stopped"
                break
        try:
            self._finalize_swa(
                validate_fn=lambda: self._validate_inner(tc.epochs - 1),
                bn_forward_fn=self._swa_bn_forward)
        except Exception:  # SWA 收尾失败不影响已完成的训练/best 产物。
            logger.exception("SWA finalization failed; online/best "
                             "checkpoints are unaffected.")
        self._monitor_finalize(final_status)
        if self._ckpt_saver is not None:
            # 收尾前排空异步写盘队列；写盘异常在此抛出。
            self._ckpt_saver.close()
            self._ckpt_saver = None
        return {"best_psnr": self.best_metric, "best_epoch": self.best_epoch,
                **last}


__all__ = ["GenerationTrainer"]
