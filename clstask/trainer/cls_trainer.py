"""分类训练器（复用 segtask 训练基建：optim / warmup / AMP / EMA / GPU 增强）。

与 ``gentask.GenerationTrainer`` 同构的独立精简训练器：

* 训练：patch → （可选 GPUAugmentor 联合增强 image+label 后派生 target）
  → forward → BCE/CE/Focal（可选 mixup/cutmix）→ 反传；损失在 autocast 外
  以 fp32 计算（logit clamp 防溢出，沿用 segtask 惯例）。bf16/fp32（无
  GradScaler）路径下 loss/梯度非有限时丢弃本 accum 组，不推 scheduler/EMA
  （口径同 segtask Trainer）。
* 验证：收集全量 logits/targets → patch 级 AUC / F1 / acc，另按卷分组经
  ``cls.agg_mode`` MIL 聚合出卷级 vol_auc / vol_f1 / vol_acc（与推理同口
  径）；按 ``cls.save_best_metric`` 选模保存 ``best_model.pth``（与 seg 同，
  best 的 ``model_state_dict`` 以 EMA 为主，在线权重存
  ``model_online_state_dict``）。
* encoder 差分学习率：``cls.encoder_lr_mult`` 对 encoder 参数组缩放 lr
  （微调预训练权重的惯用手段）；头部参数保持 ``train.lr``。warmup 段亦
  保持各组倍率（见 :class:`_GroupWarmupScheduler`）。
* 工程能力（口径同 seg Trainer）：每 epoch 落盘 ``latest_model.pth``（含
  model/EMA/optimizer/scheduler/scaler/epoch/best 状态）与 ``history.json``；
  ``train.resume`` 指向 checkpoint 时完整恢复续训；``train.early_stopping``
  连续 N 次验证无提升时提前停止（分类每 epoch 验证一次）。
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from taskcore.config.core import Config as SegConfig
from taskcore.data.augment import GPUAugmentor
from taskcore.engine.amp import _LOGIT_CLAMP, autocast
from taskcore.engine.checkpoint import (
    AsyncCheckpointSaver, atomic_torch_save, snapshot_rng_state,
    state_to_cpu, unwrap_compile,
)
from taskcore.engine.dist_utils import all_gather_objects
from taskcore.engine.prefetch import CudaPrefetcher
from taskcore.engine.optim import (
    WarmupScheduler,
    build_optimizer,
    build_scheduler,
)
from taskcore.utils.common import AverageMeter
from taskcore.engine.base_trainer import BaseTrainer

from ..config import ClsConfig, resolve_num_classes
from ..data.cls_dataset import derive_volume_targets
from ..losses.cls_loss import build_cls_loss
from ..metrics import multilabel_metrics, singlelabel_metrics
from ..predictor.cls_predictor import aggregate_probs
from .mixup import apply_mixup_cutmix

logger = logging.getLogger(__name__)


def _build_optimizer_with_lr_mult(model: nn.Module, cfg: SegConfig,
                                  encoder_lr_mult: float):
    """AdamW/Adam/SGD + weight-decay 分组 + encoder 学习率倍率。

    分组口径与 ``segtask_v1.trainer.optim._param_groups`` 一致（ndim<=1 免
    decay），再按参数属于 encoder 与否二分（2×2 组）。AdamW 的 fused 开关
    口径同 ``segtask_v1.trainer.optim.build_optimizer``。
    """
    tc = cfg.train
    enc_ids = {id(p) for p in model.encoder.parameters()}
    groups = []
    for is_enc in (True, False):
        lr = tc.lr * (encoder_lr_mult if is_enc else 1.0)
        for wd, pred in ((tc.weight_decay, lambda p: p.ndim >= 2),
                         (0.0, lambda p: p.ndim <= 1)):
            params = [p for p in model.parameters()
                      if p.requires_grad and (id(p) in enc_ids) == is_enc
                      and pred(p)]
            if params:
                groups.append(
                    {"params": params, "lr": lr, "weight_decay": wd})
    if tc.optimizer == "adamw":
        first = next((p for p in model.parameters()), None)
        on_cuda = first is not None and first.is_cuda
        use_fused = tc.adamw_fused and torch.cuda.is_available()
        return torch.optim.AdamW(groups, lr=tc.lr,
                                 fused=(use_fused and on_cuda))
    if tc.optimizer == "adam":
        return torch.optim.Adam(groups, lr=tc.lr)
    if tc.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=tc.lr, momentum=tc.momentum,
                               nesterov=tc.nesterov)
    raise ValueError(f"Unknown optimizer: {tc.optimizer}")


class _GroupWarmupScheduler(WarmupScheduler):
    """保留各参数组 lr 倍率的线性 warmup。

    seg 的 ``WarmupScheduler`` 假定所有组同一 base_lr，warmup 段对全部组写
    统一 lr —— 与 encoder 差分学习率（``cls.encoder_lr_mult``）矛盾。这里
    在构造时记下各组 base lr，warmup 段按 ``组 base / 全局 base`` 的比例
    缩放整段 ramp，各组倍率全程不变；全组同 lr 时退化为父类行为。
    """

    def __init__(self, optimizer, scheduler, warmup_steps: int,
                 warmup_lr: float, base_lr: float,
                 group_base_lrs: "list[float]"):
        self._ratios = [
            (g / base_lr) if base_lr > 0 else 1.0 for g in group_base_lrs]
        super().__init__(optimizer, scheduler, warmup_steps=warmup_steps,
                         warmup_lr=warmup_lr, base_lr=base_lr)
        if warmup_steps > 0:
            for pg, r in zip(optimizer.param_groups, self._ratios):
                pg["lr"] = warmup_lr * r

    def step(self) -> None:
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            alpha = self.current_step / max(self.warmup_steps, 1)
            lr = self.warmup_lr + alpha * (self.base_lr - self.warmup_lr)
            for pg, r in zip(self.optimizer.param_groups, self._ratios):
                pg["lr"] = lr * r
        elif self.scheduler is not None and not self._is_plateau:
            self.scheduler.step()


class ClsTrainer(BaseTrainer):
    """分类训练器；``fit()`` 返回 best 指标摘要。"""

    def __init__(self, model: nn.Module, cfg: SegConfig, cls: ClsConfig,
                 train_loader, val_loader, device: torch.device):
        self.cfg = cfg
        self.cls = cls
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(device)
        tc = cfg.train

        # 可选 channels_last 内存格式（数值等价；共用工程件，见 BaseTrainer）。
        self._setup_channels_last()

        self.num_classes = resolve_num_classes(cls, cfg)
        self.loss_fn = build_cls_loss(cfg, cls).to(device)
        self.single_label = not cls.multi_label

        if abs(cls.encoder_lr_mult - 1.0) > 1e-9:
            self.optimizer = _build_optimizer_with_lr_mult(
                self.model, cfg, cls.encoder_lr_mult)
        else:
            self.optimizer = build_optimizer(self.model, cfg)
        # 调度时钟按优化步（非 micro-step）计，与 scheduler.step() 的调用频率
        # 一致（accum 尾组也触发一步，口径同 segtask Trainer）。
        self.grad_accum_steps = max(tc.grad_accum_steps, 1)
        steps_per_epoch = max(
            math.ceil(len(train_loader) / self.grad_accum_steps), 1)
        warmup_steps = tc.warmup_epochs * steps_per_epoch
        total_steps = tc.epochs * steps_per_epoch
        base_scheduler = build_scheduler(
            self.optimizer, cfg, steps_per_epoch,
            post_warmup_steps=total_steps - warmup_steps)
        # one_cycle 自带 warmup（pct_start），外层不再叠加线性 warmup。
        warmup_steps = 0 if tc.scheduler == "one_cycle" else warmup_steps
        # 差分学习率下 warmup 保留各组倍率（全组同 lr 时退化为父类行为）。
        self.scheduler = _GroupWarmupScheduler(
            self.optimizer, base_scheduler, warmup_steps=warmup_steps,
            warmup_lr=tc.warmup_lr, base_lr=tc.lr,
            group_base_lrs=[float(pg["lr"])
                            for pg in self.optimizer.param_groups])

        # --- AMP / EMA（共用工程件，见 BaseTrainer）------------------------
        self._setup_amp()
        self._setup_ema()

        # --- torch.compile（最后：optimizer / EMA 已绑裸参数）----------
        self._maybe_compile()

        # --- DDP 装配与训练采样器识别（公用工程件，见 BaseTrainer）-------
        # 单卡时 fwd_model 即裸模块，路径零变化；多卡时前向走 DDP 包装。
        self._setup_ddp()
        self._setup_train_sampler()

        # GPU 增强（复用 segtask GPUAugmentor）：在未折叠的 (B,1,D,H,W) 上
        # 对 image+label 联合施加，增强后派生分类 target 再折叠（见
        # ClsPatchDataset 的 gpu_augment 模式）。
        self.augmentor = (
            GPUAugmentor(cfg.augment, max_scale=1.0,
                         label_fill=float(cfg.data.label_values[0]))
            if cfg.augment.enabled else None)
        self.fg_values = [float(v) for v in (
            cfg.data.label_values[1:]
            if len(cfg.data.label_values) > 1 else [1.0])]
        self.fold_2_5d = int(cfg.model.spatial_dims) == 2

        self._setup_output_dir()
        # save_async=True 时权重先深拷到 CPU，后台线程 torch.save，主循环
        # 不再被写盘阻塞；fit 收尾 wait+close 保证全部落盘（同 seg）。
        self._ckpt_saver = (AsyncCheckpointSaver()
                            if self.cfg.train.save_async and self._is_main
                            else None)
        # SWA 尾段权重平均（opt-in，公用工程件，见 BaseTrainer）。
        self._setup_swa()
        self.best_key = cls.save_best_metric        # auc|f1|acc|loss|vol_*
        self.best_sign = -1.0 if self.best_key == "loss" else 1.0
        self._setup_best_tracking(mode="max")
        self.history: "list[Dict[str, float]]" = []
        # mask 源卷级 MIL 真值：整卷 label 一次性派生（与 patch 抽样解耦，
        # 避免抽样未覆盖病灶时阳性卷被当阴性）；惰性计算一次。
        self._val_vol_targets: "torch.Tensor | None" = None

        # --- 训练监测仪表盘（公用工程件，见 BaseTrainer / taskcore.monitor；
        #     cfg.monitor 守卫，失败隔离不阻断训练）-----------------------
        self._setup_monitor(
            resume_active=bool(str(tc.resume or "").strip()),
            run_name_default="cls_run",
            save_best_metric=self.best_key,
            save_best_mode="min" if self.best_key == "loss" else "max",
            save_best_criterion="",
            num_classes=0,
            config_meta={
                "loss": cfg.loss.name,
                "batch_size": cfg.data.batch_size,
                "label_granularity": cls.label_granularity,
            })

    # ------------------------------------------------------------------
    def _loss_fp32(self, logits: torch.Tensor,
                   target: torch.Tensor) -> torch.Tensor:
        """autocast 外 fp32 损失（logit clamp 防 fp16 溢出）。"""
        logits = logits.float().clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)
        with torch.autocast(device_type=self.device.type, enabled=False):
            return self.loss_fn(logits, target)

    def _targets_from_label(self, lbl: torch.Tensor) -> torch.Tensor:
        """GPU 上由增强后 label (B,1,D,H,W) 派生分类 target
        （口径同 ``ClsPatchDataset._target_from_mask``）。"""
        if self.cls.label_granularity == "volume":
            t = [(lbl == v).flatten(1).any(dim=1) for v in self.fg_values]
            return torch.stack(t, dim=1).float()               # (B, K)
        t = [(lbl == v).any(dim=-1).any(dim=-1).squeeze(1)
             for v in self.fg_values]
        return torch.stack(t, dim=1).float()                   # (B, K, D)

    def _prepare_batch(self, batch: Dict) -> "tuple[torch.Tensor, torch.Tensor]":
        """batch → (image, target)；GPU 增强模式下先增强再派生/折叠。"""
        img = batch["image"].to(self.device, non_blocking=True).float()
        if self.augmentor is None:
            target = batch["target"].to(self.device, non_blocking=True)
            return img, target
        if "label" in batch:                                   # mask 源
            lbl = batch["label"].to(self.device, non_blocking=True).float()
            img, lbl, _ = self.augmentor(img, lbl)
            target = self._targets_from_label(lbl)
        else:                                                  # table 源
            dummy = torch.zeros_like(img)
            img, _, _ = self.augmentor(img, dummy)
            target = batch["target"].to(self.device, non_blocking=True)
        if self.fold_2_5d:
            b, _, d, h, w = img.shape
            img = img.reshape(b, d, h, w)                      # 深度折进通道
        return img, target

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(epoch)   # 多卡重洗（单卡 no-op）
        self.model.train()
        loss_meter = AverageMeter()
        accum = self.grad_accum_steps
        total = len(self.train_loader)
        grad_norm_meter = AverageMeter()
        grad_norm_max = 0.0
        nonfinite_steps = 0
        clipped_steps = 0
        opt_steps = 0
        # GPU 预取（口径同 seg Trainer）：独立 copy stream 提前一个 batch 上卡。
        batch_iter = self.train_loader
        if self.cfg.train.prefetch_to_gpu and self.device.type == "cuda":
            batch_iter = CudaPrefetcher(self.train_loader, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        use_mix = ((self.cls.mixup_alpha > 0 or self.cls.cutmix_alpha > 0)
                   and self.cls.label_granularity == "volume")
        group_has_nonfinite = False

        # 未缩放损失先以 GPU 张量缓存，延迟到真正需要标量的时刻（日志步；无
        # GradScaler 时还有 accum 边界的非有限 guard）单次 stack+tolist 取回，
        # 避免每 micro-step 一次 loss.item() 的 device→host 同步（同 seg）。
        pending: "list[tuple[int, torch.Tensor, int]]" = []

        def _flush_pending() -> "float | None":
            """单次同步取回缓存损失并更新 meter；返回最后一步的损失值。"""
            nonlocal group_has_nonfinite, nonfinite_steps
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
                    logger.warning(
                        "Non-finite train loss (%s) at epoch %d step %d/%d; "
                        "skipping meter update.", v, epoch + 1, s + 1, total)
            pending.clear()
            return last

        for step, batch in enumerate(batch_iter):
            img, target = self._prepare_batch(batch)
            if use_mix:
                img, target = apply_mixup_cutmix(
                    img, target, self.num_classes,
                    self.cls.mixup_alpha, self.cls.cutmix_alpha,
                    self.cls.mixup_prob)
            with torch.autocast(device_type=self.device.type,
                                enabled=self.use_amp, dtype=self.amp_dtype):
                logits = self.fwd_model(img)
            loss = self._loss_fp32(logits, target)
            # 尾组 micro-batch 不满 accum 时用真实尾长作分母（同 seg）。
            eff_accum = self._effective_accum(step, total, accum)
            loss_scaled = loss / eff_accum if eff_accum > 1 else loss
            self.scaler.scale(loss_scaled).backward()

            pending.append((step, loss.detach(), img.shape[0]))
            is_boundary = (step + 1) % accum == 0 or (step + 1) == total
            is_log_step = ((step + 1) % max(self.cfg.train.log_every, 1) == 0
                           or step == 0)
            loss_val: "float | None" = None
            if is_log_step or (is_boundary and not self._scaler_active):
                loss_val = _flush_pending()

            if is_boundary:
                # 梯度范数：clip 开时复用其范数；bf16/fp32 无 GradScaler 保护，
                # clip 关时也算一次全局范数作非有限守护（同 seg）。
                grad_norm_val = None
                if self.cfg.train.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    gn = nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.train.grad_clip_norm)
                    if not self._scaler_active:
                        grad_norm_val = float(gn)
                elif not self._scaler_active:
                    grad_norm_val = self._global_grad_norm()
                grad_nonfinite = (grad_norm_val is not None
                                  and not math.isfinite(grad_norm_val))
                if grad_norm_val is not None and not grad_nonfinite:
                    grad_norm_meter.update(grad_norm_val)
                    grad_norm_max = max(grad_norm_max, grad_norm_val)
                    if (self.cfg.train.grad_clip_norm > 0
                            and grad_norm_val > self.cfg.train.grad_clip_norm):
                        clipped_steps += 1

                # bf16/fp32：loss/梯度非有限则丢弃本 accum 组，不推
                # scheduler/EMA，避免 NaN 永久污染权重；fp16 由 GradScaler
                # 内部跳过含 inf/NaN 梯度的优化步。
                if not self._scaler_active and (group_has_nonfinite
                                                or grad_nonfinite):
                    logger.warning(
                        "Skipping optimizer step at epoch %d step %d/%d: "
                        "non-finite %s in this accumulation group.",
                        epoch + 1, step + 1, total,
                        "loss" if group_has_nonfinite else "gradient")
                    self.optimizer.zero_grad(set_to_none=True)
                    group_has_nonfinite = False
                    continue
                group_has_nonfinite = False

                scale_before = (self.scaler.get_scale()
                                if self._scaler_active else None)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scaler_skipped = (scale_before is not None
                                  and self.scaler.get_scale() < scale_before)
                self.optimizer.zero_grad(set_to_none=True)
                opt_steps += 1
                # scheduler/EMA 仅在 optimizer 真正更新后推进（同 seg）。
                if not scaler_skipped:
                    self.scheduler.step()
                    if self.ema is not None:
                        self.ema.update(unwrap_compile(self.model))

            if is_log_step and loss_val is not None:
                logger.debug("  [%d/%d] loss=%.4f lr=%.2e", step + 1, total,
                             loss_val, self.scheduler.get_lr())
        _flush_pending()
        out = {"loss": loss_meter.avg}
        self._collect_health_metrics(
            out,
            grad_norm_meter=grad_norm_meter,
            grad_norm_max=grad_norm_max,
            nonfinite_steps=nonfinite_steps,
            clipped_steps=clipped_steps,
            opt_steps=opt_steps,
            grad_clip_norm=float(self.cfg.train.grad_clip_norm))
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
            img = batch["image"].to(self.device, non_blocking=True).float()
            if self.fold_2_5d:
                b, _, d, h, w = img.shape
                img = img.reshape(b, d, h, w)
            with autocast(device_type=self.device.type,
                          enabled=self.use_amp, dtype=self.amp_dtype):
                self.model(img)

    @torch.no_grad()
    def _validate_inner(self, epoch: int) -> Dict[str, float]:
        """在**当前权重**上评测（不换 EMA；供 _validate / SWA 收尾共用）。"""
        all_logits, all_targets, all_vols = [], [], []
        loss_meter = AverageMeter()
        for batch in self.val_loader:
            img = batch["image"].to(self.device).float()
            target = batch["target"].to(self.device)
            # 验证前向 autocast 口径同训练/推理（seg validation 同）；
            # 损失仍在 fp32 计算。
            with autocast(device_type=self.device.type,
                          enabled=self.use_amp, dtype=self.amp_dtype):
                logits = self.model(img)
            loss_meter.update(
                float(self._loss_fp32(logits, target).item()),
                img.shape[0])
            all_logits.append(logits.float().cpu())
            all_targets.append(target.cpu())
            all_vols.append(batch["vol_idx"].cpu())
        if not all_logits and not self._is_dist:
            raise RuntimeError("validation loader yielded no batches.")
        # DDP：val 集按 batch 块分片到各 rank；把 (logits, targets, vols,
        # loss 累计) 聚齐后在每个 rank 上算全集指标（AUC/F1 不可分解，
        # 不能逐 rank 算后平均），选模/早停决策各 rank 天然一致。
        parts = all_gather_objects({
            "logits": (torch.cat(all_logits) if all_logits
                       else torch.empty(0)),
            "targets": (torch.cat(all_targets) if all_targets
                        else torch.empty(0, dtype=torch.long)),
            "vols": (torch.cat(all_vols) if all_vols
                     else torch.empty(0, dtype=torch.long)),
            "loss_sum": float(loss_meter.sum),
            "loss_count": int(loss_meter.count),
        })
        parts = [p for p in parts if p["loss_count"] > 0 or len(p["logits"])]
        if not parts:
            raise RuntimeError("validation loader yielded no batches.")
        logits = torch.cat([p["logits"] for p in parts])
        targets = torch.cat([p["targets"] for p in parts])
        vols = torch.cat([p["vols"] for p in parts])
        loss_meter.sum = sum(p["loss_sum"] for p in parts)
        loss_meter.count = sum(p["loss_count"] for p in parts)
        if self.single_label:
            probs = torch.softmax(logits, dim=1)
            m = singlelabel_metrics(probs, targets)
        else:
            probs = torch.sigmoid(logits)
            m = multilabel_metrics(probs, targets)
        m["loss"] = loss_meter.avg
        m.update(self._volume_metrics(probs, targets, vols))
        if m.get("auc_defined_classes", 1.0) == 0.0:
            logger.warning(
                "val AUC undefined for all classes (single-class val split); "
                "reported auc=0.5. Consider more val volumes or stratified "
                "split.")
        return m

    def _volume_metrics(self, probs: torch.Tensor, targets: torch.Tensor,
                        vols: torch.Tensor) -> Dict[str, float]:
        """卷级 MIL 指标：按卷分组，用与推理同口径的 ``aggregate_probs``
        （cls.agg_mode/topk/lse_r）聚合 patch 概率；slice 粒度先把各 patch
        的切片展平为实例。卷级 target：mask 源用整卷 label 派生的精确多热
        真值（:func:`derive_volume_targets`，与 patch 抽样解耦）；table 源为
        卷内常量，取首个。"""
        if (self.cls.label_source == "mask"
                and self._val_vol_targets is None):
            self._val_vol_targets = derive_volume_targets(
                self.val_loader.dataset.paths, self.fg_values)
        vol_probs, vol_targets = [], []
        for u in torch.unique(vols):
            sel = vols == u
            p = probs[sel]                                     # (n,K)/(n,K,D)
            t = targets[sel]
            if p.ndim == 3:                                    # slice 粒度
                p = p.permute(0, 2, 1).reshape(-1, p.shape[1])
            vol_probs.append(aggregate_probs(
                p, self.cls.agg_mode, self.cls.agg_topk, self.cls.agg_lse_r))
            if self.single_label:
                vol_targets.append(t[0])
            elif self.cls.label_source == "mask":
                vol_targets.append(self._val_vol_targets[int(u)])
            else:
                tt = t
                if tt.ndim == 3:
                    tt = tt.amax(dim=2)
                vol_targets.append(tt.amax(dim=0))
        vp = torch.stack(vol_probs)
        vt = torch.stack(vol_targets)
        if self.single_label:
            vm = singlelabel_metrics(vp, vt)
        else:
            vm = multilabel_metrics(vp, vt)
        return {f"vol_{k}": v for k, v in vm.items()
                if k in ("auc", "f1", "acc")}

    def _save_best(self, epoch: int, metrics: Dict[str, float]) -> None:
        """best checkpoint（口径同 seg：EMA 为主）：启用 EMA 时
        ``model_state_dict`` 存 EMA 权重（与选模/部署一致），在线权重另存
        ``model_online_state_dict``。"""
        if not self._is_main:   # DDP：落盘仅 rank0
            return
        bare = unwrap_compile(self.model)
        if self.ema is not None:
            online_sd = {k: v.detach().cpu().clone()
                         for k, v in bare.state_dict().items()}
            self.ema.apply_shadow(bare)
            try:
                primary_sd = {k: v.detach().cpu().clone()
                              for k, v in bare.state_dict().items()}
            finally:
                self.ema.restore(bare)
        else:
            online_sd = None
            primary_sd = bare.state_dict()
        state = {
            "epoch": epoch,
            "model_state_dict": primary_sd,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "metrics": metrics,
            "config": self.cfg,
            "cls_config": self.cls,
        }
        if online_sd is not None:
            state["model_online_state_dict"] = online_sd
        if self.ema is not None:
            state["ema_state_dict"] = self.ema.state_dict()
        path = self.output_dir / "best_model.pth"
        if self._ckpt_saver is not None:
            self._ckpt_saver.submit(
                state_to_cpu(state), path,
                on_done=lambda e=epoch, m=metrics[self.best_key]: logger.info(
                    "Best cls model saved (%s=%.4f) @ epoch %d",
                    self.best_key, m, e + 1))
        else:
            atomic_torch_save(state, path)
            logger.info("Best cls model saved (%s=%.4f) @ epoch %d",
                        self.best_key, metrics[self.best_key], epoch + 1)

    def _save_latest(self, epoch: int, metrics: Dict[str, float]) -> None:
        """latest checkpoint（续训用，口径同 seg）：在线权重 + EMA +
        optimizer/scheduler/scaler/epoch/best 状态/history 全量落盘；先写
        临时文件再原子替换，防中断留半个 checkpoint。"""
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
            "metrics": metrics,
            "history": self.history,
            "config": self.cfg,
            "cls_config": self.cls,
            "rng_state": snapshot_rng_state(),
        }
        if self.ema is not None:
            state["ema_state_dict"] = self.ema.state_dict()
        path = self.output_dir / "latest_model.pth"
        if self._ckpt_saver is not None:
            self._ckpt_saver.submit(state_to_cpu(state), path)
        else:
            atomic_torch_save(state, path)  # 原子替换，防中断写坏

    def _try_resume(self) -> None:
        """``train.resume`` 指向 checkpoint 时完整恢复（model/EMA/optimizer/
        scheduler/scaler/epoch/best 状态/history）；路径不存在则报错，
        避免静默从头训。"""
        path = str(self.cfg.train.resume or "").strip()
        if not path:
            return
        if not Path(path).is_file():
            raise FileNotFoundError(f"train.resume checkpoint not found: "
                                    f"{path!r}")
        ckpt = torch.load(path, map_location=self.device,
                          weights_only=False)
        # 公共段（model/EMA/optim/sched/scaler/best/RNG）见 BaseTrainer。
        self.start_epoch = self._restore_train_state(ckpt)
        self.history = list(ckpt.get("history", []))
        logger.info(
            "Resumed from %s: next epoch %d, best %s=%.4f @ epoch %d",
            path, self.start_epoch + 1, self.best_key,
            self.best_sign * self.best_metric, self.best_epoch + 1)

    def _write_history(self) -> None:
        if not self._is_main:   # DDP：落盘仅 rank0
            return
        (self.output_dir / "history.json").write_text(
            json.dumps(self.history, indent=2), encoding="utf-8")

    def fit(self) -> Dict[str, float]:
        # 公共加载策略：resume 全状态恢复；pretrain 仅权重；同设优先 resume。
        self._try_pretrain()
        self._try_resume()
        last: Dict[str, float] = dict(self.history[-1]) if self.history else {}
        patience = max(int(self.cfg.train.early_stopping), 0)
        epochs_no_improve = max(
            self.start_epoch - 1 - self.best_epoch, 0)
        final_status = "completed"
        for epoch in range(self.start_epoch, self.cfg.train.epochs):
            epoch_t0 = time.time()
            tr = self._train_epoch(epoch)
            val = self._validate(epoch)
            self.scheduler.step_epoch(val.get(self.best_key))
            logger.info(
                "Epoch %d/%d: train_loss=%.4f val_loss=%.4f val_auc=%.4f "
                "val_f1=%.4f val_acc=%.4f vol_auc=%.4f vol_f1=%.4f "
                "vol_acc=%.4f",
                epoch + 1, self.cfg.train.epochs, tr["loss"], val["loss"],
                val["auc"], val["f1"], val["acc"], val["vol_auc"],
                val["vol_f1"], val["vol_acc"])
            score = self.best_sign * val[self.best_key]
            is_best = score > self.best_metric
            if is_best:
                self.best_metric = score
                self.best_epoch = epoch
                self._save_best(epoch, val)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            last = {"epoch": float(epoch), **tr,
                    **{f"val_{k}": v for k, v in val.items()}}
            self.history.append(last)
            self._swa_update(epoch)
            self._save_latest(epoch, val)
            self._write_history()
            gpu_peak_mib = None
            if self.device.type == "cuda":
                gpu_peak_mib = (torch.cuda.max_memory_allocated(self.device)
                                / (1 << 20))
                torch.cuda.reset_peak_memory_stats(self.device)
            self._monitor_log_epoch(
                epoch, tr, val,
                lr=self.scheduler.get_lr(), gpu_peak_mib=gpu_peak_mib,
                wall_time_s=time.time() - epoch_t0, is_best=is_best,
                last_epoch=(epoch + 1) == self.cfg.train.epochs)
            if patience > 0 and epochs_no_improve >= patience:
                logger.info(
                    "Early stopping at epoch %d: no %s improvement for %d "
                    "validation(s) (best @ epoch %d).", epoch + 1,
                    self.best_key, patience, self.best_epoch + 1)
                final_status = "early_stopped"
                break
        try:
            self._finalize_swa(
                validate_fn=lambda: self._validate_inner(
                    self.cfg.train.epochs - 1),
                bn_forward_fn=self._swa_bn_forward)
        except Exception:  # SWA 收尾失败不影响已完成的训练/best 产物。
            logger.exception("SWA finalization failed; online/best "
                             "checkpoints are unaffected.")
        self._monitor_finalize(final_status)
        if self._ckpt_saver is not None:
            # 收尾前排空异步写盘队列；写盘异常在此抛出。
            self._ckpt_saver.close()
            self._ckpt_saver = None
        return {f"best_{self.best_key}": self.best_sign * self.best_metric,
                "best_epoch": float(self.best_epoch), **last}


__all__ = ["ClsTrainer"]
