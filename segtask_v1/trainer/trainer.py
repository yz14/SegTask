"""3D / 2.5D 分割训练 pipeline：AMP / EMA / warmup / 累积 / compile / DS / checkpoint。

Round 2 重构后，``Trainer`` 不再判断训练模式 —— 所有"如何把 batch 重塑为
``(model_input, supervision)``、如何把 ``model_output`` 折成 loss"的逻辑都
归口到 ``self.pipeline: ViewPipeline``（见 ``trainer.pipelines``）。

``Trainer`` 仅协调：模型 / 优化器 / 调度器 / scaler / EMA / 增强 / 训练循环 /
checkpoint I/O。视图拆分 / 中心裁剪等纯张量操作位于 ``trainer.views``，损失
fp32 计算与 breakdown 格式化位于 ``trainer.amp`` / ``trainer.breakdown``，
``Trainer`` 直接调用这些纯函数，不再保留供测试用的 thin shim。
"""

from __future__ import annotations

import logging
import math
import os
import random
import re
import time
from typing import Dict, Iterable, List, Tuple

from colorama import Fore, Style

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from taskcore.config.core import Config
from taskcore.data.augment import GPUAugmentor
from ..losses.losses import build_loss
from taskcore.models.unet import UNet3D
from taskcore.models.mednext import upkern_remap_state_dict
from taskcore.utils.common import (
    AverageMeter, Timer, compute_dice_per_class,
)
from . import views
from taskcore.engine.amp import autocast
from .breakdown import collect_multi_res_breakdown, format_breakdown
from taskcore.engine.prefetch import CudaPrefetcher
from taskcore.engine.checkpoint import (
    AsyncCheckpointSaver,
    atomic_torch_save,
    state_to_cpu,
    unwrap_compile,
)
from taskcore.engine.dist_utils import (
    barrier,
)
from taskcore.engine.memory import estimate_train_memory
from .pipelines import (
    Patch3DNativeMultiResPipeline,
    Slab2_5DNativeDPipeline,
    ViewPipeline,
    build_pipeline,
)
from .validation import build_val_evaluator
from taskcore.engine.base_trainer import (  # noqa: F401  (_reseed_rank_rng re-export供旧路径)
    BaseTrainer,
    reseed_rank_rng,
    _reseed_rank_rng,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer(BaseTrainer):
    """3D / 2.5D 分割完整训练 pipeline。"""

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
        self._setup_channels_last()

        # --- Pipeline (criterion + view ops) -------------------------
        self.base_loss = build_loss(cfg.loss)  # 预设dice/bce等等这些
        self.pipeline: ViewPipeline = build_pipeline(cfg, self.base_loss)

        self.n_views        = self.pipeline.n_views         # 多分辨率输入
        self.n_aux_views    = self.pipeline.n_aux_views     # 2.5D的多分辨率辅助输入
        self.num_res_groups = self.pipeline.num_res_groups  # 3D的多分辨率辅助输入
        self.slab_depth     = self.pipeline.slab_depth      # 2.5D | z-axis的depth

        self.criterion   = self.pipeline.criterion
        self.aux_loss_fn = self.pipeline.aux_loss_fn  # 2.5D多分辨率用aux损失

        self.keep_native_multi_res  = isinstance(
            self.pipeline, Patch3DNativeMultiResPipeline)
        self.keep_native_view_depth = isinstance(
            self.pipeline, Slab2_5DNativeDPipeline)

        self._mr_native_sizes: List[Tuple[int, int, int]] = list(
            getattr(self.pipeline, "mr_native_sizes", []))
        self.per_view_depths: List[int] = list(
            getattr(self.pipeline, "per_view_depths", []))
        # target_patch_size / needs_crop（增强后中心裁回）
        self.target_patch_size = self.pipeline.target_patch_size  # 模型输入尺寸
        self.needs_crop         = cfg.data.aug_oversample_ratio > 1.0

        # --- Optimizer + scheduler / AMP / EMA（共用工程件，见 BaseTrainer）---
        self._setup_optim_sched()
        self._setup_amp()
        self._setup_ema()

        # --- SWA 尾段权重平均（opt-in，公用工程件，见 BaseTrainer）---------
        self._setup_swa()

        # --- torch.compile (最后) -------------------------------------
        self._first_step_mem_logged = False
        self._maybe_compile()

        # --- DDP (多卡) ------------------------------------------------
        # ``self.model`` 始终保持"裸 / 已 compile"模块：optimizer / EMA / checkpoint /
        # predictor 全部继续作用其上，参数张量与单卡完全一致（DDP 不复制参数，仅挂
        # 反传 all-reduce 钩子）。前向走 ``self.fwd_model``（DDP 包装），梯度即在各卡
        # 间同步。``world_size<=1`` 时 fwd_model 即裸模块，单卡路径零变化。
        self._setup_ddp()
        self._setup_train_sampler()

        # --- 增强 ------------------------------------------------------
        _scales = cfg.data.multi_res_scales or [1.0]
        # label_fill：affine/elastic 越界区域的背景 label 值（label_values[0]，
        # loader 构建后必已填充）。
        _bg = float(cfg.data.label_values[0]) if cfg.data.label_values else 0.0
        # 独立增强随机流：与全局 RNG（模型初始化/dropout/DataLoader）解耦并逐
        # rank 分流；固定 seed 下增强序列可复现（augment 合流等价性验证前置）。
        # inplace=True：本循环的 image/label/wmap 是 H2D 私有拷贝（trainer
        # _train_epoch），增强后不再以原值复用，满足 inplace 契约，省一份
        # 过采样 cube 的瞬时显存。
        _aug_seed = (int(cfg.train.seed) + 7919 * (self._rank + 1)) & 0x7FFFFFFF
        self.augmentor = GPUAugmentor(
            cfg.augment, max_scale=max(_scales), label_fill=_bg,
            seed=_aug_seed, inplace=True)

        # --- Tracking --------------------------------------------------
        self.num_fg = cfg.num_fg_classes
        self._setup_best_tracking(mode=tc.save_best_mode)  # "max" or "min"
        self.best_key = tc.save_best_metric
        self._ckpt_task_label = "seg"

        # --- Validation / model-selection evaluator -------------------
        # medium（随机 patch 指标）/ high（整卷滑窗指标）由 val_metric_mode 决定；
        # 两者产出同结构 metrics dict，选模/调度/ckpt 逻辑无需分支。
        self.evaluator = build_val_evaluator(self)
        logger.info("Validation metric mode: %s (evaluator=%s)",
                    tc.val_metric_mode, type(self.evaluator).__name__)

        # --- Output directory -----------------------------------------
        self._setup_output_dir()

        # --- Async checkpoint saver（opt-in，仅 rank0）-------------------
        # save_async=True 时权重先深拷到 CPU，后台线程 torch.save，主循环
        # 不再被写盘阻塞；fit 收尾 wait+close 保证全部落盘。
        self._ckpt_saver = (AsyncCheckpointSaver()
                            if tc.save_async and self._is_main else None)

        # --- Resume / Pretrain ----------------------------------------
        # resume：全状态恢复；pretrain：仅加载权重。同设优先 resume。
        # 显式路径不存在即报错（fail-fast，防静默从头训；口径同 cls/det）。
        resume_active = bool(tc.resume)
        if tc.resume:
            if tc.pretrain:
                logger.warning(
                    "Both `train.resume` and `train.pretrain` are set; "
                    "using resume (%s). Pretrain weights from %s are ignored.",
                    tc.resume, tc.pretrain)
            if not os.path.isfile(tc.resume):
                raise FileNotFoundError(
                    f"train.resume checkpoint not found: {tc.resume!r}")
            self._load_checkpoint(tc.resume)
        elif tc.pretrain:
            if not os.path.isfile(tc.pretrain):
                raise FileNotFoundError(
                    f"train.pretrain checkpoint not found: {tc.pretrain!r}")
            self._load_pretrain(
                tc.pretrain,
                strict=tc.pretrain_strict,
                load_ema=tc.pretrain_load_ema)

        # --- Training monitor (optional; fully isolated) --------------
        # 逐 epoch 把指标落盘并周期性重渲染自包含 HTML 仪表盘（公用工程件，
        # 见 BaseTrainer / ``taskcore.monitor``）。由 ``cfg.monitor.enabled``
        # 守卫，落盘 / 渲染等副作用仅在 rank0 进行，任何失败仅告警。
        self._setup_monitor(
            resume_active,
            num_classes=self.num_fg,
            config_meta={
                "loss": self.cfg.loss.name,
                "batch_size": self.cfg.data.batch_size,
                "val_metric_mode": tc.val_metric_mode,
            })

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
        mem = estimate_train_memory(self.model, self.optimizer, self.ema)
        ema_part = (f" + ema={mem['ema_mib']:.1f}" if mem["ema_mib"] > 0 else "")
        logger.info(
            "Static GPU mem (persistent, excl. activations): "
            "param=%.1f + grad=%.1f + optim(%s,%dx)=%.1f%s "
            "= %.1f MiB (real peak reported per-epoch as 'GPU peak')",
            mem["param_mib"], mem["grad_mib"],
            mem["optim_name"], mem["optim_mult"], mem["optim_mib"],
            ema_part, mem["persistent_mib"])
        if self.cfg.model.grad_checkpointing:
            logger.info(
                "Gradient checkpointing: ON — encoder/decoder activations "
                "recomputed in backward (~+20-33%% compute, much lower "
                "activation memory; numerics unchanged vs OFF).")
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
        logger.info(
            "Pipeline=%s | n_views=%d, n_aux_views=%d, num_res_groups=%d, "
            "slab_depth=%d | fg_classes=%d, Loss=%s",
            type(self.pipeline).__name__,
            self.n_views, self.n_aux_views, self.num_res_groups,
            self.slab_depth, self.num_fg, self.cfg.loss.name)
        if tc.compile_mode != "none":
            logger.info(
                "torch.compile mode: %s (active=%s)",
                tc.compile_mode, self._compile_enabled)
        logger.info("=" * 60)

        best_metrics: Dict[str, float] = {}
        final_status = "completed"
        for epoch in range(self.start_epoch, tc.epochs):
            epoch_t0 = time.time()
            train_metrics = self._train_epoch(epoch)
            train_time_s = time.time() - epoch_t0

            self._swa_update(epoch)

            val_metrics: Dict[str, float] = {}
            val_time_s = 0.0
            val_selects = True
            if (epoch + 1) % tc.val_every == 0 or epoch == tc.epochs - 1:
                val_t0 = time.time()
                val_metrics = self._validate(epoch)
                val_time_s = time.time() - val_t0
                # 混合调度的 medium 监控轮次不参与选模/早停/plateau（口径
                # 与 high 不同）；非混合 evaluator 恒 True，行为不变。
                val_selects = self.evaluator.selects_model()

            # 仅 plateau 逐 epoch 驱动。
            plateau_metric = (val_metrics.get(tc.save_best_metric, None)
                              if val_selects else None)
            self.scheduler.step_epoch(metric=plateau_metric)

            # --- Best-checkpoint 决策 -----------------------------------
            is_best = False
            if val_selects and tc.save_best_metric in val_metrics:
                tracked = val_metrics[tc.save_best_metric]
                if not self.has_best:
                    is_best = True
                elif self._best_mode == "max":
                    is_best = tracked > self.best_metric
                else:
                    is_best = tracked < self.best_metric

                if is_best:
                    self.best_metric = tracked
                    self.best_epoch  = epoch
                    self.has_best    = True
                    self.patience_counter = 0
                    # best 走 BaseTrainer（EMA-primary 槽位与 cls/det/gen 对齐）；
                    # 周期 checkpoint 仍用本类 _save_checkpoint（keep-last-k）。
                    self._save_best(epoch, val_metrics)
                    best_metrics = val_metrics
                    logger.info(
                        "★ New best: %s=%.4f at epoch %d",
                        tc.save_best_metric, tracked, epoch + 1,
                        extra={"msg_color": Fore.YELLOW + Style.BRIGHT})
                else:
                    self.patience_counter += 1

            # --- Epoch summary ----------------------------------------
            best_str = (f"{self.best_metric:.4f} (ep{self.best_epoch + 1})"
                        if self.has_best else "n/a")
            aux_summary_dict = {
                k: v for k, v in train_metrics.items()
                if k.startswith("L_main") or k.startswith("L_aux_")
                or k.startswith("w_aux_")
                or k.startswith("L_res_") or k.startswith("L_aux_res_")}
            aux_msg = format_breakdown(aux_summary_dict)
            logger.info(
                "Epoch %d/%d | LR=%.2e | loss=%.4f | val_dice=%.4f | "
                "best=%s | %s%s",
                epoch + 1, tc.epochs, self.scheduler.get_lr(),
                train_metrics.get("loss", 0.0),
                val_metrics.get("mean_dice", 0.0),
                best_str,
                timer.elapsed_str(),
                aux_msg)
            # 训练 / 验证分段计时：定位整 epoch 时间到底花在哪一阶段。
            tot_tv = train_time_s + val_time_s
            logger.info(
                "  Phase time (epoch %d): train=%s | val=%s | "
                "val=%.1f%% of (train+val)",
                epoch + 1,
                time.strftime("%H:%M:%S", time.gmtime(train_time_s)),
                time.strftime("%H:%M:%S", time.gmtime(val_time_s)),
                100.0 * val_time_s / tot_tv if tot_tv > 0 else 0.0)

            gpu_peak_mib = None
            if self.device.type == "cuda":
                gpu_peak_mib = torch.cuda.max_memory_allocated(self.device) / (1 << 20)
                logger.info("  GPU peak (epoch %d): %.1f MiB", epoch + 1, gpu_peak_mib)
                torch.cuda.reset_peak_memory_stats(self.device)

            # --- Training monitor (isolated) --------------------------
            self._monitor_log_epoch(
                epoch, train_metrics, val_metrics,
                lr=self.scheduler.get_lr(), gpu_peak_mib=gpu_peak_mib,
                wall_time_s=time.time() - epoch_t0, is_best=is_best,
                last_epoch=(epoch == tc.epochs - 1))

            # --- Periodic checkpoint ----------------------------------
            if (epoch + 1) % tc.save_every == 0:
                self._save_checkpoint(epoch, is_best=False)

            # --- Early stopping ---------------------------------------
            if tc.early_stopping > 0 and self.patience_counter >= tc.early_stopping:
                logger.info("Early stopping at epoch %d (patience=%d)",
                            epoch + 1, tc.early_stopping)
                final_status = "early_stopped"
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
        try:
            # SWA 收尾走 base 版（换入平均权重 → 重估 BN → 验证 → rank0 另存）；
            # validate_fn 在 SWA 权重上直调 evaluator（不再换入 EMA）。
            self._finalize_swa(
                validate_fn=lambda: self.evaluator.evaluate(tc.epochs - 1),
                bn_forward_fn=self._swa_bn_forward,
                is_main=self._is_main)
        except Exception:  # SWA 收尾失败不影响已完成的训练/best 产物。
            logger.exception("SWA finalization failed; online/best "
                             "checkpoints are unaffected.")
        self._monitor_finalize(final_status)
        if self._ckpt_saver is not None:
            # 收尾前排空异步写盘队列；写盘异常在此抛出。
            self._ckpt_saver.close()
            self._ckpt_saver = None
        # DDP：收尾屏障，确保 rank0 完成 best/checkpoint 落盘后各进程再退出。
        barrier()
        return best_metrics

    # ------------------------------------------------------------------
    # Training / validation loops
    # （EMA 换入 / 梯度累积尾批 / 模型健康监测等共用件见 BaseTrainer）
    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """单 epoch 训练，支持梯度累积。模式判断完全交给 ``self.pipeline``。"""
        self.model.train()
        loss_meter = AverageMeter()
        dice_meter = AverageMeter()
        component_meters: Dict[str, AverageMeter] = {}
        tc          = self.cfg.train
        accum       = self.grad_accum_steps
        total_steps = len(self.train_loader)

        # 模型健康监测累计器（仅 rank0 监测启用时填充；否则恒为空，零开销）。
        grad_norm_meter = AverageMeter()
        grad_norm_max = 0.0
        nonfinite_steps = 0
        clipped_steps = 0
        opt_steps = 0
        update_ratio_val: "float | None" = None  # 每 epoch 仅首个优化步测一次

        # DDP：每 epoch 重置 DistributedSampler 的洗牌种子，保证各 epoch 切分不同。
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(epoch)

        self.optimizer.zero_grad(set_to_none=True)
        group_has_nonfinite = False

        # 未缩放损失先以 GPU 张量缓存，延迟到真正需要标量的时刻（日志步；无
        # GradScaler 时还有 accum 边界的非有限 guard）单次 stack+tolist 取回，
        # 避免每 micro-step 一次 loss.item() 的 device→host 同步。
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
                        "skipping meter update. The surrounding optimizer "
                        "step will be skipped (%s).",
                        v, epoch + 1, s + 1, total_steps,
                        "by GradScaler" if self._scaler_active
                        else "non-finite loss guard")
            pending.clear()
            return last

        # prefetch_to_gpu：独立 copy stream 提前一个 batch 上卡，H2D 与计算重叠。
        # 交付的张量已在 device 上，下方 .to(device) 退化为 no-op，循环体不变。
        batch_iter: "Iterable" = self.train_loader
        if tc.prefetch_to_gpu and self.device.type == "cuda":
            batch_iter = CudaPrefetcher(self.train_loader, self.device)

        for step, batch in enumerate(batch_iter):
            image = batch["image"].to(self.device, non_blocking=True)
            label = batch["label"].to(self.device, non_blocking=True).float()
            wmap = batch.get("weight_map")
            if wmap is not None:
                wmap = wmap.to(self.device, non_blocking=True)
                if wmap.numel() == 0 or wmap.shape[1] == 0:
                    wmap = None

            # 增强 + oversample 中心裁
            image, label, wmap = self.augmentor(image, label, wmap)
            if self.needs_crop:
                image, label, wmap = views.center_crop(
                    image, label, wmap, self.target_patch_size)

            # 视图重塑（pipeline 内部决定）
            image, sup = self.pipeline.prepare_batch(image, label, wmap)
            if self._memory_format is not None:
                image = image.to(memory_format=self._memory_format)

            effective_accum = self._effective_accum(step, total_steps, accum)
            is_step_boundary = ((step + 1) % accum == 0 or (step + 1) == total_steps)
            # 非边界步免 all-reduce；forward 也必须放进 no_sync。
            sync_ctx = self._ddp_no_sync(is_step_boundary)

            # 分量诊断仅在日志步抽取：breakdown 的逐分量 .item() 会强制
            # CUDA→CPU 同步，非日志步传 None 让 pipeline 跳过标量抽取。
            is_log_step = (step + 1) % tc.log_every == 0 or step == 0
            breakdown: Dict[str, float] = {}

            # Forward AMP / Loss fp32（Dice/BCE 在 fp16 下汇总易溢出 → NaN）
            with sync_ctx:
                with autocast(device_type="cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                    pred = self.fwd_model(image)
                loss = self.pipeline.compute_loss(
                    pred, sup, breakdown=breakdown if is_log_step else None)
                if effective_accum > 1:
                    loss = loss / effective_accum

                self.scaler.scale(loss).backward()
            if is_log_step:
                # per-res history 在非日志步间累积，pop 时取窗口均值。
                collect_multi_res_breakdown(
                    self.criterion, self.aux_loss_fn, breakdown)

            # 未缩放损失留在 GPU 缓存；仅在日志步（需标量打印）或无 scaler
            # 路径的 accum 边界（非有限 guard 需在 skip 判定前知晓）单次同步。
            detached = loss.detach()
            if effective_accum > 1:
                detached = detached * effective_accum
            pending.append((step, detached, image.shape[0]))

            step_loss: "float | None" = None
            if is_log_step or (is_step_boundary and not self._scaler_active):
                step_loss = _flush_pending()
            if step_loss is not None and math.isfinite(step_loss):
                for name, val in breakdown.items():
                    if not math.isfinite(val):
                        continue
                    if name not in component_meters:
                        component_meters[name] = AverageMeter()
                    component_meters[name].update(val, image.shape[0])

            # 参数更新
            if is_step_boundary:
                # update/weight 比值：每 epoch 仅首个优化步测一次（step 前快照）。
                # 快照放在 before_step，仅当未因非有限跳步时才会执行。
                pre_step_snapshot = None

                def _before_step() -> None:
                    nonlocal pre_step_snapshot, update_ratio_val
                    if self._health_update_ratio and update_ratio_val is None:
                        try:
                            pre_step_snapshot = self._param_snapshot()
                        except Exception:  # 监测失败绝不打断训练
                            logger.warning(
                                "Health param snapshot failed; skipping "
                                "update/weight ratio this epoch.",
                                exc_info=True)
                            pre_step_snapshot = None

                result = self._optimizer_step_boundary(
                    group_has_nonfinite=group_has_nonfinite,
                    epoch=epoch, step=step, total_steps=total_steps,
                    before_step=_before_step)
                group_has_nonfinite = False
                grad_norm_val = result.grad_norm
                skipped_nf = result.skipped_nonfinite
                result.acknowledge()

                if skipped_nf:
                    if self._health_monitor:
                        opt_steps += 1
                    continue

                if pre_step_snapshot is not None:
                    try:
                        update_ratio_val = self._update_ratio_from_snapshot(
                            pre_step_snapshot)
                    except Exception:  # 监测失败绝不打断训练
                        logger.warning(
                            "Health update/weight ratio computation failed; "
                            "skipping.", exc_info=True)
                    finally:
                        pre_step_snapshot = None

                if self._health_monitor:
                    opt_steps += 1
                    if grad_norm_val is not None and math.isfinite(grad_norm_val):
                        grad_norm_meter.update(grad_norm_val)
                        if grad_norm_val > grad_norm_max:
                            grad_norm_max = grad_norm_val
                        if (tc.grad_clip_norm > 0
                                and grad_norm_val > tc.grad_clip_norm):
                            clipped_steps += 1

                if (not self._first_step_mem_logged
                        and self.device.type == "cuda"):
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

            if is_log_step:
                with torch.no_grad():
                    p = self.pipeline.extract_main_pred(pred)        # 主路
                    p_1x, lbl_1x = self.pipeline.split_for_metrics(  # 主路
                        p.detach(), sup.label_main)
                    dice = compute_dice_per_class(p_1x, lbl_1x)
                    mean_dice = dice.mean().item()
                    dice_meter.update(mean_dice, image.shape[0])
                aux_msg = format_breakdown(breakdown)
                logger.debug(
                    "  [%d/%d] loss=%.4f dice=%.4f lr=%.2e%s",
                    step + 1, total_steps,
                    step_loss if step_loss is not None else float("nan"),
                    mean_dice, self.scheduler.get_lr(), aux_msg)

        # epoch 末把尚未取回的缓存损失一次性落入 meter。
        _flush_pending()

        # train dice / 分量损失均只在日志步采样，epoch 值为抽样均值（廉价监控），
        # 与 val 的全量指标口径不同。
        out = {"loss": loss_meter.avg, "dice": dice_meter.avg}
        for name, meter in component_meters.items():
            out[name] = meter.avg
        if self._health_monitor:
            try:
                self._collect_health_metrics(
                    out, grad_norm_meter=grad_norm_meter,
                    grad_norm_max=grad_norm_max, nonfinite_steps=nonfinite_steps,
                    clipped_steps=clipped_steps, opt_steps=opt_steps,
                    grad_clip_norm=tc.grad_clip_norm,
                    update_ratio=update_ratio_val)
            except Exception:  # 监测失败绝不打断训练
                logger.warning(
                    "Health metric collection failed; skipping.",
                    exc_info=True)
        return out

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """验证集评估（启用 EMA 时以 EMA 权重）。

        实际指标累加 / 导出与"指标在什么预测上算"的口径全部归口到
        ``self.evaluator``（见 ``trainer.validation``）：

        * ``medium`` → ``PatchValEvaluator``：遍历 ``val_loader`` 随机 patch。
        * ``high``   → ``VolumeValEvaluator``：每个 val 整卷滑窗推理后算指标。

        两者产出同结构 metrics dict，故下游选模 / 调度 / checkpoint 无需分支。
        EMA 换入在此统一处理，保证两种模式都以 EMA 权重评估。
        """
        self.model.eval()
        # val_empty_cache：验证前后各归还一次 cached blocks，给整卷滑窗大累加器
        # 腾出连续显存（只影响 allocator，不影响数值）。None=自动：high 模式开。
        _empty = self.cfg.train.val_empty_cache
        if _empty is None:
            _empty = (
                str(self.cfg.train.val_metric_mode).lower().strip() == "high")
        _flush = _empty and self.device.type == "cuda"
        if _flush:
            torch.cuda.empty_cache()
        try:
            with self._ema_swapped():
                return self.evaluator.evaluate(epoch)
        finally:
            if _flush:
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # SWA BN 重估前向（收尾流程走 base 版 ``_finalize_swa`` 回调）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _swa_bn_forward(self) -> None:
        """SWA BN 重估用前向：未增强训练数据，与推理分布同构。"""
        steps = int(self.cfg.train.swa_bn_update_steps)
        for step, batch in enumerate(self.train_loader):
            if step >= steps:
                break
            image = batch["image"].to(self.device, non_blocking=True)
            label = batch["label"].to(
                self.device, non_blocking=True).float()
            if self.needs_crop:
                image, label, _ = views.center_crop(
                    image, label, None, self.target_patch_size)
            image, _sup = self.pipeline.prepare_batch(image, label, None)
            if self._memory_format is not None:
                image = image.to(memory_format=self._memory_format)
            with autocast(device_type="cuda", enabled=self.use_amp,
                          dtype=self.amp_dtype):
                self.model(image)

    # ------------------------------------------------------------------
    # Checkpointing (kept on Trainer for inspect.getsource compatibility)
    # ------------------------------------------------------------------
    def _build_state_dict(self, ema_as_primary: bool) -> Dict:
        """打包训练状态。``ema_as_primary=True`` 时 ``model_state_dict`` 为 EMA，
        在线权重放到 ``model_online_state_dict``；反之方向。"""
        bare = unwrap_compile(self.model)
        online_sd = bare.state_dict()

        # 快照 RNG 状态以支持位精确 resume（torch CPU/CUDA + numpy + python）。
        rng_state = {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
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

        if self.swa is not None:
            state["swa_state_dict"] = self.swa.state_dict()

        if self.ema is not None:
            state["ema_state_dict"] = self.ema.state_dict()
            if ema_as_primary:
                self.ema.apply_shadow(unwrap_compile(self.model))
                try:
                    state["model_state_dict"] = unwrap_compile(
                        self.model).state_dict()
                finally:
                    self.ema.restore(unwrap_compile(self.model))
                state["model_online_state_dict"] = online_sd

        return state

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """周期 / 续训 checkpoint（非 best）。

        ``is_best=True`` 已弃用：选模落盘请走 ``BaseTrainer._save_best``。
        保留参数仅为兼容旧调用；真 best 路径不再进入本方法。
        """
        if is_best:
            # 防御：若仍有旧调用，转发到公共 best 路径。
            metrics = {self.best_key: float(self.best_metric)}
            self._save_best(epoch, metrics)
            return
        # ZeRO 优化器状态分片在各 rank：保存前需全 rank 集合式 consolidate 到
        # rank0（必须在 rank 早退之前调用，否则集合通信挂死）。
        if hasattr(self.optimizer, "consolidate_state_dict"):
            self.optimizer.consolidate_state_dict(to=0)
        # 多卡下仅 rank0 落盘，避免多进程写同一文件互相覆盖 / 损坏。
        if not self._is_main:
            return
        state = self._build_state_dict(ema_as_primary=False)
        state["epoch"] = epoch

        path = self.output_dir / f"checkpoint_epoch_{epoch + 1}.pth"
        if self._ckpt_saver is not None:
            # 清理在写完后的后台回调里做，保证 keep-last-k 计数含本次。
            def _on_done(p=path):
                logger.debug("Checkpoint saved: %s", p)
                self._prune_old_checkpoints()
            self._ckpt_saver.submit(state_to_cpu(state), path, _on_done)
        else:
            atomic_torch_save(state, path)
            logger.debug("Checkpoint saved: %s", path)
            self._prune_old_checkpoints()

    def _prune_old_checkpoints(self) -> None:
        """周期 checkpoint 的 keep-last-k 保留：仅留最近 ``save_keep_last`` 个
        ``checkpoint_epoch_*.pth``，更早的删除；``best_model.pth`` 不受影响。
        ``save_keep_last <= 0`` 时不清理。仅 rank0 调用（由 ``_save_checkpoint`` 保证）。"""
        keep = int(self.cfg.train.save_keep_last)
        if keep <= 0:
            return
        ckpts = []
        for p in self.output_dir.glob("checkpoint_epoch_*.pth"):
            m = re.fullmatch(r"checkpoint_epoch_(\d+)\.pth", p.name)
            if m:
                ckpts.append((int(m.group(1)), p))
        ckpts.sort(key=lambda t: t[0])
        for _, p in ckpts[:-keep]:
            try:
                p.unlink()
                logger.debug("Pruned old checkpoint: %s", p)
            except OSError as e:  # 清理失败不影响训练
                logger.warning("Failed to prune old checkpoint %s: %s", p, e)

    def _load_checkpoint(self, path: str) -> None:
        logger.info("Loading checkpoint: %s", path)
        # PyTorch 2.6+ 默认 weights_only=True 会拒 numpy RNG / Config；ckpt 为本 trainer 写，显式关闭。
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        # 公共段（model/EMA/optim/sched/scaler/SWA/best/RNG）见 BaseTrainer。
        self.start_epoch = self._restore_train_state(ckpt)
        if self._is_dist and self._rank > 0:
            reseed_rank_rng(
                self.cfg.train.seed, self._rank, self.start_epoch,
                self.cfg.train.deterministic)

        logger.info(
            "Resumed from epoch %d, best=%s=%s (patience=%d)",
            self.start_epoch, self.cfg.train.save_best_metric,
            f"{self.best_metric:.4f}" if self.has_best else "n/a",
            self.patience_counter)

    # ------------------------------------------------------------------
    # Pretrain (weights-only initialisation)
    # ------------------------------------------------------------------
    def _pretrain_transform_state_dict(self, sd, bare):
        """seg 专有：``train.pretrain_upkern`` 时对 depthwise 卷积核做 UpKern
        升核重采样（trilinear 插值到目标 kernel 尺寸），其余走公共策略。"""
        if not self.cfg.train.pretrain_upkern:
            return sd
        target_sd = bare.state_dict()
        n_upkern = 0
        for key, src_tensor in sd.items():
            tgt_tensor = target_sd.get(key)
            if (tgt_tensor is None or not torch.is_tensor(src_tensor)
                    or not torch.is_tensor(tgt_tensor)):
                continue
            if (src_tensor.shape != tgt_tensor.shape
                    and src_tensor.ndim in (4, 5)
                    and tgt_tensor.ndim == src_tensor.ndim
                    and src_tensor.shape[:2] == tgt_tensor.shape[:2]
                    and src_tensor.shape[2:] != tgt_tensor.shape[2:]):
                n_upkern += 1
        sd = upkern_remap_state_dict(
            sd, bare,
            normalize_spatial=bool(self.cfg.train.pretrain_upkern_normalize))
        logger.info(
            "Pretrain: applied UpKern remap to %d depthwise tensor(s).",
            n_upkern)
        return sd

    def _load_pretrain(self, path: str, strict: bool, load_ema: bool) -> None:
        """仅加载权重作迁移初始化（公共策略见 BaseTrainer；UpKern 经
        ``_pretrain_transform_state_dict`` 钩子接入）。"""
        self._load_pretrain_weights(path, strict=strict, load_ema=load_ema)


__all__ = ["Trainer"]
