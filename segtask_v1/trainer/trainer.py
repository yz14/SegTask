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

import contextlib
import logging
import math
import os
import random
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from colorama import Fore, Style

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from ..config import Config
from ..data.augment import GPUAugmentor
from ..losses.losses import build_loss
from ..models.unet import UNet3D
from ..models.mednext import upkern_remap_state_dict
from ..predictor.adabn import collect_bn_modules, estimate_bn_stats
from ..utils import (
    AverageMeter, ModelEMA, ModelSWA, Timer, compute_dice_per_class,
    seed_everything,
)
from . import views
from .amp import (
    _AMP_DTYPES,
    GradScaler,
    autocast,
    resolve_auto_amp_dtype,
)
from .breakdown import collect_multi_res_breakdown, format_breakdown
from .checkpoint import (
    AsyncCheckpointSaver,
    atomic_torch_save,
    extract_model_state_dict,
    relocate_optimizer_state,
    restore_rng_state,
    state_to_cpu,
    strip_common_prefixes,
    unwrap_compile,
)
from .dist_utils import (
    all_reduce_bn_running_stats_,
    all_reduce_flag_any,
    barrier,
    get_rank,
    get_world_size,
    is_dist_avail_and_initialized,
    is_main_process,
)
from .memory import estimate_train_memory
from .optim import WarmupScheduler, build_optimizer, build_scheduler
from .pipelines import (
    Patch3DNativeMultiResPipeline,
    Slab2_5DNativeDPipeline,
    ViewPipeline,
    build_pipeline,
)
from .validation import build_val_evaluator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RNG helper
# ---------------------------------------------------------------------------
def _reseed_rank_rng(seed: int, rank: int, epoch: int, deterministic: bool) -> None:
    """按 rank/epoch 重新分流 RNG。rank0 保持原流不动。"""
    if int(rank) <= 0:
        return
    # resume 后 rank>0 重新分流，避免所有 DDP rank 退化成 rank0 的随机流。
    mix = int(seed) + int(epoch) * 100003 + int(rank)
    seed_everything(mix, deterministic)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
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
        self._memory_format = None
        if tc.channels_last:
            self._memory_format = (
                torch.channels_last_3d
                if int(cfg.model.spatial_dims) == 3
                else torch.channels_last)
            self.model = self.model.to(memory_format=self._memory_format)

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

        # --- Optimizer + scheduler ------------------------------------
        self.optimizer = build_optimizer(self.model, cfg)
        # 调度器以优化步（optimizer.step 次数）为单位推进：梯度累积下每
        # epoch 的优化步数为 ceil(micro-batch 数 / accum)（epoch 尾部不满
        # accum 的尾组也触发一步，见 _train_epoch 的 is_step_boundary）。
        self.grad_accum_steps = max(tc.grad_accum_steps, 1)
        steps_per_epoch = math.ceil(len(train_loader) / self.grad_accum_steps)
        warmup_steps    = tc.warmup_epochs * steps_per_epoch
        total_steps     = tc.epochs * steps_per_epoch
        post_warmup     = total_steps - warmup_steps

        base_scheduler = build_scheduler(
            self.optimizer, cfg, steps_per_epoch, post_warmup_steps=post_warmup)
        # one_cycle 用 warmup_epochs 映射 pct_start，外层不再做线性 warmup，
        # 避免 warmup 双重叠加。
        warmup_steps = 0 if tc.scheduler == "one_cycle" else warmup_steps
        self.scheduler = WarmupScheduler(
            self.optimizer, base_scheduler, warmup_steps=warmup_steps,
            warmup_lr=tc.warmup_lr, base_lr=tc.lr)

        # --- AMP -------------------------------------------------------
        amp_dtype_cfg = tc.amp_dtype
        if amp_dtype_cfg == "auto":
            amp_dtype_cfg = resolve_auto_amp_dtype(device)
            logger.info("amp_dtype='auto' resolved to %r (device=%s).",
                        amp_dtype_cfg, device)
        if amp_dtype_cfg not in _AMP_DTYPES:
            raise ValueError(
                f"Unknown amp_dtype: {tc.amp_dtype!r}. "
                f"Expected one of {sorted(_AMP_DTYPES) + ['auto']}.")
        self.amp_dtype = _AMP_DTYPES[amp_dtype_cfg]
        self._amp_dtype_name = amp_dtype_cfg
        self.use_amp = tc.use_amp and device.type == "cuda"
        self._scaler_active = self.use_amp and self.amp_dtype == torch.float16
        self.scaler = GradScaler("cuda", enabled=self._scaler_active)

        # --- EMA -------------------------------------------------------
        # ema_device="cpu" 时 shadow/backup 常驻 CPU（省 1× 参数量 GPU 显存，
        # 数学等价）；默认 "" 跟随模型设备，行为不变。
        self.ema = (ModelEMA(self.model, tc.ema_decay, warmup=tc.ema_warmup,
                             offload_device=(tc.ema_device or None))
                    if tc.use_ema else None)

        # --- SWA 尾段权重平均（opt-in，见 TrainConfig.swa_enabled）---------
        self.swa = ModelSWA(self.model) if tc.swa_enabled else None
        self._swa_start_epoch = (
            int(math.floor(tc.swa_start_ratio * tc.epochs))
            if tc.swa_enabled else 0)

        # --- torch.compile (最后) -------------------------------------
        self._compile_enabled = False
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
                        tc.compile_mode)
            if triton_ok:
                logger.info("Compiling model with mode='%s'", tc.compile_mode)
                self.model = torch.compile(self.model, mode=tc.compile_mode)
                self._compile_enabled = True

        # --- DDP (多卡) ------------------------------------------------
        # ``self.model`` 始终保持"裸 / 已 compile"模块：optimizer / EMA / checkpoint /
        # predictor 全部继续作用其上，参数张量与单卡完全一致（DDP 不复制参数，仅挂
        # 反传 all-reduce 钩子）。前向走 ``self.fwd_model``（DDP 包装），梯度即在各卡
        # 间同步。``world_size<=1`` 时 fwd_model 即裸模块，单卡路径零变化。
        self._rank       = get_rank()
        self._world_size = get_world_size()
        self._is_main    = is_main_process()
        self._is_dist    = is_dist_avail_and_initialized() and self._world_size > 1
        if self._is_dist:
            ddp_kwargs: Dict[str, object] = {}
            if device.type == "cuda" and device.index is not None:
                ddp_kwargs = {"device_ids": [device.index],
                              "output_device": device.index}
            self.fwd_model = nn.parallel.DistributedDataParallel(
                self.model,
                find_unused_parameters=bool(tc.ddp_find_unused_parameters),
                gradient_as_bucket_view=bool(tc.ddp_gradient_as_bucket_view),
                static_graph=bool(tc.ddp_static_graph),
                **ddp_kwargs)
            if tc.ddp_static_graph and tc.ddp_find_unused_parameters:
                logger.warning(
                    "ddp_static_graph=True 与 ddp_find_unused_parameters=True "
                    "同时开启：static_graph 首步后会接管 unused-parameter 处理，"
                    "建议将 ddp_find_unused_parameters 设为 False 以免首步额外开销。")
            logger.info(
                "DDP enabled: rank=%d/%d, device=%s, "
                "find_unused_parameters=%s, gradient_as_bucket_view=%s, "
                "static_graph=%s. "
                "Training grads all-reduce per backward. Note: math-equivalence "
                "to single-GPU under grad-accum holds for per-sample separable "
                "losses (BCE/Focal/per-sample Dice); batch-pooled ratio losses "
                "(batch_dice/Tversky/GDL) pool over the per-rank micro-batch, "
                "so their effective statistics window shrinks with accum/ranks "
                "(approximate, not strictly equivalent).",
                self._rank, self._world_size, device,
                tc.ddp_find_unused_parameters,
                tc.ddp_gradient_as_bucket_view,
                tc.ddp_static_graph)
        else:
            self.fwd_model = self.model

        # 训练采样器（每 epoch set_epoch 重洗）：单源 DDP 为 DistributedSampler
        # （loader.sampler），双源混合为 MixedBatchSampler（loader.batch_sampler，
        # DDP 下各 rank 依赖 set_epoch 对齐全局排列）；均按 set_epoch 协议鸭子识别。
        self._train_sampler = None
        for _s in (getattr(self.train_loader, "sampler", None),
                   getattr(self.train_loader, "batch_sampler", None)):
            if _s is not None and callable(getattr(_s, "set_epoch", None)):
                self._train_sampler = _s
                break

        # --- 增强 ------------------------------------------------------
        _scales = cfg.data.multi_res_scales or [1.0]
        self.augmentor = GPUAugmentor(cfg.augment, max_scale=max(_scales))

        # --- Tracking --------------------------------------------------
        self.num_fg           = cfg.num_fg_classes
        self._best_mode       = tc.save_best_mode  # "max" or "min"
        self.best_metric      = (-math.inf if self._best_mode == "max" else math.inf)
        self.has_best         = False
        self.best_epoch       = 0
        self.start_epoch      = 0
        self.patience_counter = 0

        # --- Validation / model-selection evaluator -------------------
        # medium（随机 patch 指标）/ high（整卷滑窗指标）由 val_metric_mode 决定；
        # 两者产出同结构 metrics dict，选模/调度/ckpt 逻辑无需分支。
        self.evaluator = build_val_evaluator(self)
        logger.info("Validation metric mode: %s (evaluator=%s)",
                    tc.val_metric_mode, type(self.evaluator).__name__)

        # --- Output directory -----------------------------------------
        self.output_dir = Path(tc.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- Async checkpoint saver（opt-in，仅 rank0）-------------------
        # save_async=True 时权重先深拷到 CPU，后台线程 torch.save，主循环
        # 不再被写盘阻塞；fit 收尾 wait+close 保证全部落盘。
        self._ckpt_saver = (AsyncCheckpointSaver()
                            if tc.save_async and self._is_main else None)

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
            if tc.resume and not os.path.isfile(tc.resume):
                logger.warning(
                    "`train.resume` is set but file not found: %s. "
                    "Training will start from scratch.", tc.resume)
            if tc.pretrain and not os.path.isfile(tc.pretrain):
                logger.warning(
                    "`train.pretrain` is set but file not found: %s. "
                    "Training will start from scratch.", tc.pretrain)

        # --- Training monitor (optional; fully isolated) --------------
        # 逐 epoch 把指标落盘并周期性重渲染自包含 HTML 仪表盘。整套逻辑封装在
        # ``segtask_v1.monitor``，由 ``cfg.monitor.enabled`` 守卫，任何失败都被
        # 隔离（仅告警），绝不影响训练本身。
        self._monitor = None
        self._monitor_html = None
        self._monitor_cfg = getattr(self.cfg, "monitor", None)
        # 落盘 / 渲染等副作用仅在 rank0 进行，避免多进程争抢同一文件。
        if (self._monitor_cfg is not None and self._monitor_cfg.enabled
                and self._is_main):
            self._init_monitor(resume_active)
        # 模型健康监测：仅当监测启用、配置开启且在 rank0 时采集（成本极低，
        # 失败被隔离）。非有限步计数 / 梯度范数 / 裁剪比例 / 权重范数 / AMP 标度。
        self._health_monitor = bool(
            self._monitor is not None
            and getattr(self._monitor_cfg, "health_monitor", False))
        self._health_grad_norm_when_no_clip = bool(
            getattr(self._monitor_cfg, "health_grad_norm_when_no_clip", True))
        self._health_update_ratio = bool(
            self._health_monitor
            and getattr(self._monitor_cfg, "health_update_ratio", False))

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

            if self.swa is not None and epoch >= self._swa_start_epoch:
                self.swa.update(unwrap_compile(self.model))

            val_metrics: Dict[str, float] = {}
            val_time_s = 0.0
            if (epoch + 1) % tc.val_every == 0 or epoch == tc.epochs - 1:
                val_t0 = time.time()
                val_metrics = self._validate(epoch)
                val_time_s = time.time() - val_t0

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
                    self.best_epoch  = epoch
                    self.has_best    = True
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, is_best=True)
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
            self._finalize_swa()
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
    # Training monitor (optional, isolated) —— 见 segtask_v1.monitor
    # ------------------------------------------------------------------
    def _init_monitor(self, resume_active: bool) -> None:
        """实例化 ``MetricsLogger`` 并确定 HTML 落点。失败仅告警、不影响训练。"""
        mc = self._monitor_cfg
        tc = self.cfg.train
        try:
            from ..monitor import MetricsLogger

            root = Path(mc.output_dir) if mc.output_dir else self.output_dir
            mon_dir = root / "monitor"
            self._monitor_html = root / (mc.filename or "training_monitor.html")
            run_name = mc.run_name or self.output_dir.name or "run"
            self._monitor = MetricsLogger(
                mon_dir,
                run_name=run_name,
                save_best_metric=tc.save_best_metric,
                save_best_mode=tc.save_best_mode,
                save_best_criterion=tc.save_best_criterion,
                num_classes=self.num_fg,
                total_epochs=tc.epochs,
                config_meta={
                    "loss": self.cfg.loss.name,
                    "batch_size": self.cfg.data.batch_size,
                    "val_metric_mode": tc.val_metric_mode,
                },
                resume=resume_active,
            )
            logger.info("Training monitor enabled → metrics: %s | dashboard: %s",
                        mon_dir, self._monitor_html)
        except Exception as e:  # 隔离：监测初始化失败绝不阻断训练
            self._monitor = None
            logger.warning("Training monitor disabled (init failed): %s", e)

    def _monitor_log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        *,
        lr: float,
        gpu_peak_mib,
        wall_time_s: float,
        is_best: bool,
        last_epoch: bool,
    ) -> None:
        """落盘一个 epoch 并按 ``update_every`` 节奏重渲染 HTML（全程异常隔离）。"""
        if self._monitor is None:
            return
        mc = self._monitor_cfg
        try:
            self._monitor.log_epoch(
                epoch, train=train_metrics, val=val_metrics, lr=lr,
                gpu_peak_mib=gpu_peak_mib, wall_time_s=wall_time_s,
                is_best=is_best)
        except Exception as e:
            logger.warning("Training monitor: log_epoch failed at epoch %d: %s",
                           epoch + 1, e)
            return
        every = max(int(mc.update_every), 1)
        if is_best or last_epoch or ((epoch + 1) % every == 0):
            self._monitor_render(auto_reload_seconds=int(mc.auto_reload_seconds))

    def _monitor_render(self, *, auto_reload_seconds: int) -> None:
        """从已落盘历史重渲染单 run 仪表盘 HTML（异常隔离）。"""
        if self._monitor is None or self._monitor_html is None:
            return
        try:
            from ..monitor import MetricsHistory, write_dashboard

            hist = MetricsHistory.from_dir(self._monitor.dir)
            write_dashboard(hist, self._monitor_html,
                            auto_reload_seconds=auto_reload_seconds)
        except Exception as e:
            logger.warning("Training monitor: dashboard render failed: %s", e)

    def _monitor_finalize(self, status: str) -> None:
        """训练收尾：更新 run 状态并做一次静态（无自动刷新）终渲染。"""
        if self._monitor is None:
            return
        try:
            self._monitor.finalize(status)
        except Exception as e:
            logger.warning("Training monitor: finalize failed: %s", e)
        self._monitor_render(auto_reload_seconds=0)

    # ------------------------------------------------------------------
    # EMA swap helper (exception-safe)
    # ------------------------------------------------------------------
    @contextmanager
    def _ema_swapped(self) -> Iterator[None]:
        """临时将 EMA 权重换入 model；try/finally 保证异常时也能还原在线权重。"""
        if self.ema is None:
            yield
            return
        self.ema.apply_shadow(unwrap_compile(self.model))
        try:
            yield
        finally:
            self.ema.restore(unwrap_compile(self.model))

    # ------------------------------------------------------------------
    # Effective grad-accum denominator (尾批不满 accum 时取真实尾长)
    # ------------------------------------------------------------------
    @staticmethod
    def _effective_accum(step: int, total_steps: int, accum: int) -> int:
        """尾批 micro-batch 数不满 ``accum`` 时，用真实尾长作分母，
        以免最后一组 micro-batch 因被除以 ``accum`` 而权重偏小。"""
        if accum <= 1:
            return 1
        remainder = total_steps % accum
        partial_start = total_steps - remainder
        return remainder if (remainder > 0 and step >= partial_start) else accum

    # ------------------------------------------------------------------
    # Model-health helpers (轻量；仅 rank0 监测启用时调用)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _global_grad_norm(self) -> "float | None":
        """当前已 unscale 的全局梯度 L2 范数（遍历一次有梯度的参数）。

        仅在未开启 grad_clip（无现成范数可复用）且监测需要时手动调用；调用方
        负责在 AMP fp16 下先 ``scaler.unscale_``，以免量纲被 loss scale 污染。
        """
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        if not grads:
            return None
        # foreach 批量算逐张量范数后聚合，全程仅末尾一次 .item() 同步（逐参数
        # .item() 会打断 CUDA 流水）。任一梯度含 inf/NaN 时结果同样非有限。
        norms = torch._foreach_norm(grads, 2)
        return float(torch.linalg.vector_norm(
            torch.stack([n.float() for n in norms])).item())

    @torch.no_grad()
    def _global_weight_norm(self) -> "float | None":
        """全部参数的全局 L2 范数（每 epoch 仅算一次，开销可忽略）。"""
        params = [p.detach() for p in self.model.parameters()]
        if not params:
            return None
        norms = torch._foreach_norm(params, 2)
        return float(torch.linalg.vector_norm(
            torch.stack([n.float() for n in norms])).item())

    @torch.no_grad()
    def _param_snapshot(self) -> "list":
        """对全部参数做一次瞬时 clone（用于 update/weight 比值，每 epoch 仅一次）。"""
        return [p.detach().clone() for p in self.model.parameters()]

    @torch.no_grad()
    def _update_ratio_from_snapshot(self, snapshot: "list") -> "float | None":
        """由 ``optimizer.step`` 前的快照算全局 ‖Δw‖/‖w‖（用完释放快照）。

        参数遍历顺序与 ``_param_snapshot`` 一致，逐张量累加更新量与原权重的平方和。
        若原权重范数为 0（理论上不会发生）则返回 ``None`` 以免除零。
        """
        params = [p.detach() for p in self.model.parameters()]
        if not params:
            return None
        diffs = torch._foreach_sub(params, snapshot)
        upd = torch.linalg.vector_norm(torch.stack(
            [n.float() for n in torch._foreach_norm(diffs, 2)]))
        w = torch.linalg.vector_norm(torch.stack(
            [n.float() for n in torch._foreach_norm(snapshot, 2)]))
        w_norm = float(w.item())
        if w_norm <= 0.0:
            return None
        return float(upd.item()) / w_norm

    def _collect_health_metrics(
        self,
        out: Dict[str, float],
        *,
        grad_norm_meter: AverageMeter,
        grad_norm_max: float,
        nonfinite_steps: int,
        clipped_steps: int,
        opt_steps: int,
        grad_clip_norm: float,
        update_ratio: "float | None" = None,
    ) -> None:
        """把本 epoch 聚合的健康指标并入 ``out``（仅写入有意义的键）。

        - ``grad_norm`` / ``grad_norm_max``：仅在采集到范数时写入。
        - ``nonfinite_steps``：非有限 loss 的 micro-batch 计数。
        - ``grad_clip_frac``：开启 grad_clip 时，范数超阈值的优化步占比。
        - ``weight_norm``：每 epoch 末一次全参数范数。
        - ``amp_scale``：AMP fp16 scaler 标度（仅 scaler 实际启用时）。
        - ``update_ratio``：全局 ‖Δw‖/‖w‖（仅开启且本 epoch 成功测得时）。
        """
        if grad_norm_meter.count > 0:
            out["grad_norm"] = grad_norm_meter.avg
            out["grad_norm_max"] = grad_norm_max
        out["nonfinite_steps"] = float(nonfinite_steps)
        if grad_clip_norm > 0 and opt_steps > 0:
            out["grad_clip_frac"] = clipped_steps / opt_steps
        wn = self._global_weight_norm()
        if wn is not None:
            out["weight_norm"] = wn
        if self._scaler_active:
            out["amp_scale"] = float(self.scaler.get_scale())
        if update_ratio is not None and math.isfinite(update_ratio):
            out["update_ratio"] = update_ratio

    # ------------------------------------------------------------------
    # Training / validation loops
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

        for step, batch in enumerate(self.train_loader):
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
            sync_ctx = (self.fwd_model.no_sync()
                        if (self._is_dist and not is_step_boundary)
                        else contextlib.nullcontext())

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
                # 梯度范数：clip 开时复用其范数；bf16/fp32 无 scaler 保护，clip
                # 关时也在所有 rank 上算一次全局范数作非有限守护（DDP
                # all-reduce 后各 rank 梯度一致，范数天然跨 rank 一致），
                # 与仅 rank0 的健康监测采集解耦。
                grad_norm_val = None
                if tc.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    gn = nn.utils.clip_grad_norm_(
                        self.model.parameters(), tc.grad_clip_norm)
                    # 懒同步（grad_norm_lazy_sync）：fp16+GradScaler 且健康监控
                    # 关闭时，该范数无任何消费者（非有限守护由 scaler 完成），
                    # 跳过 float(gn) 的 D2H 同步；裁剪本身已照常执行。
                    if (tc.grad_norm_lazy_sync and self._scaler_active
                            and not self._health_monitor):
                        grad_norm_val = None
                    else:
                        grad_norm_val = float(gn)
                elif not self._scaler_active:
                    grad_norm_val = self._global_grad_norm()
                elif self._health_monitor and self._health_grad_norm_when_no_clip:
                    try:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm_val = self._global_grad_norm()
                    except Exception:  # 监测失败绝不打断训练
                        logger.warning(
                            "Health grad-norm computation failed; skipping.",
                            exc_info=True)
                        grad_norm_val = None
                grad_nonfinite = (
                    grad_norm_val is not None and not math.isfinite(grad_norm_val))
                loss_nonfinite = group_has_nonfinite

                # fp16 由 GradScaler 内部跳过含 inf/NaN 梯度的优化步（基于
                # all-reduce 后一致的梯度，各 rank 天然同步）；bf16/fp32 在
                # loss 或梯度非有限时丢弃本 accum 组梯度，避免 NaN 永久污染
                # 权重与 EMA（scheduler/EMA 均只在 optimizer 真正更新后推进，
                # LR schedule 按“参数更新次数”而非“尝试步数”前进）。loss 非有限
                # 是各 rank 本地信息，跳步决策需 all-reduce(any) 统一，维持
                # DDP 各副本施加相同更新的不变量。
                skip_optim_step = False
                if not self._scaler_active:
                    skip_optim_step = all_reduce_flag_any(
                        loss_nonfinite or grad_nonfinite, self.device)
                group_has_nonfinite = False
                if skip_optim_step:
                    causes = []
                    if loss_nonfinite:
                        causes.append("non-finite loss")
                    if grad_nonfinite:
                        causes.append("non-finite gradient")
                    logger.warning(
                        "Skipping optimizer step at epoch %d step %d/%d: "
                        "%s in this accumulation group "
                        "(amp_dtype=%s has no GradScaler protection).",
                        epoch + 1, step + 1, total_steps,
                        " and ".join(causes) if causes
                        else "non-finite on a peer rank",
                        self._amp_dtype_name)
                    self.optimizer.zero_grad(set_to_none=True)
                    # 不推 scheduler/EMA：权重未变，推进只会消耗 LR schedule /
                    # 白增 num_updates（跳步决策已 all-reduce，各 rank 一致）。
                    if self._health_monitor:
                        opt_steps += 1
                    continue

                # update/weight 比值：每 epoch 仅首个优化步测一次（step 前快照）。
                pre_step_snapshot = None
                if self._health_update_ratio and update_ratio_val is None:
                    try:
                        pre_step_snapshot = self._param_snapshot()
                    except Exception:  # 监测失败绝不打断训练
                        logger.warning(
                            "Health param snapshot failed; skipping "
                            "update/weight ratio this epoch.", exc_info=True)
                        pre_step_snapshot = None

                scale_before = (self.scaler.get_scale()
                                if self._scaler_active else None)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # GradScaler 跳步（梯度含 inf/NaN）时会把 scale 回退减半；
                # 据此识别未生效的优化步，不推 scheduler/EMA。
                scaler_skipped = (
                    scale_before is not None
                    and self.scaler.get_scale() < scale_before)
                self.optimizer.zero_grad(set_to_none=True)
                # scheduler/EMA 仅在 optimizer 真正更新后推进（GradScaler 跳步
                # 基于 all-reduce 后一致的梯度，各 rank 判定天然同步）。
                if not scaler_skipped:
                    self.scheduler.step()
                    if self.ema is not None:
                        self.ema.update(unwrap_compile(self.model))

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
    # SWA finalization（见 TrainConfig.swa_enabled）
    # ------------------------------------------------------------------
    def _finalize_swa(self) -> None:
        """SWA 收尾：换入平均权重 → 重估 BN → 验证 → rank0 另存 swa_model.pth。

        所有 rank 均参与（验证/BN 重估可能含集合通信），仅 rank0 落盘。
        早停等原因未收集到任何快照时跳过。结束后恢复在线权重。"""
        if self.swa is None:
            return
        tc = self.cfg.train
        if self.swa.n_averaged == 0:
            logger.warning(
                "SWA enabled but no snapshots collected (training ended "
                "before start epoch %d = swa_start_ratio %.2f x %d epochs); "
                "skipping SWA finalization.",
                self._swa_start_epoch + 1, tc.swa_start_ratio, tc.epochs)
            return
        bare = unwrap_compile(self.model)
        self.swa.apply_shadow(bare)
        try:
            self.model.eval()
            self._swa_recalibrate_bn()
            metrics: Dict[str, float] = {}
            try:
                metrics = self.evaluator.evaluate(tc.epochs - 1)
            except Exception:
                logger.warning(
                    "SWA validation failed; saving SWA weights anyway.",
                    exc_info=True)
            metric_str = ", ".join(
                f"{k}={v:.4f}" for k, v in metrics.items()
                if isinstance(v, (int, float))) or "n/a"
            logger.info(
                "SWA (avg of %d epoch snapshots, from epoch %d): %s",
                self.swa.n_averaged, self._swa_start_epoch + 1, metric_str)
            if self._is_main:
                path = self.output_dir / "swa_model.pth"
                atomic_torch_save({
                    "model_state_dict": bare.state_dict(),
                    "swa_n_averaged": self.swa.n_averaged,
                    "swa_val_metrics": metrics,
                    "config": self.cfg,
                }, path)
                logger.info("SWA model saved: %s", path)
        finally:
            self.swa.restore(bare)

    def _swa_recalibrate_bn(self) -> None:
        """在若干 train batch 上重估 BN running stats（AdaBN 同款累积平均）。

        平均权重下各层激活分布改变，BN 的 running stats 不再匹配；用未增强
        的训练数据前向重估（与推理分布同构）。模型无 BatchNorm（instance/
        group norm）时为 no-op。"""
        steps = int(self.cfg.train.swa_bn_update_steps)
        if steps <= 0:
            return
        bn_modules = collect_bn_modules(unwrap_compile(self.model))
        if not bn_modules:
            return
        n_batches = min(steps, len(self.train_loader))
        logger.info(
            "SWA: re-estimating %d BatchNorm module(s) running stats on "
            "%d train batch(es).", len(bn_modules), n_batches)

        @torch.no_grad()
        def _run_forward() -> None:
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

        estimate_bn_stats(bn_modules, _run_forward)
        # DDP：各 rank 只看到自己的 train shard，聚合后 running stats 才代表
        # 全训练集（rank0 落盘的 swa_model.pth 否则只含 rank0 shard 的统计）。
        all_reduce_bn_running_stats_(bn_modules)

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
        # ZeRO 优化器状态分片在各 rank：保存前需全 rank 集合式 consolidate 到
        # rank0（必须在 rank 早退之前调用，否则集合通信挂死）。
        if hasattr(self.optimizer, "consolidate_state_dict"):
            self.optimizer.consolidate_state_dict(to=0)
        # 多卡下仅 rank0 落盘，避免多进程写同一文件互相覆盖 / 损坏。
        if not self._is_main:
            return
        state = self._build_state_dict(ema_as_primary=is_best)
        state["epoch"] = epoch

        if is_best:
            path = self.output_dir / "best_model.pth"
            if self._ckpt_saver is not None:
                self._ckpt_saver.submit(
                    state_to_cpu(state), path,
                    on_done=lambda p=path: logger.info(
                        "Best model saved: %s", p))
            else:
                atomic_torch_save(state, path)
                logger.info("Best model saved: %s", path)
        else:
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

        model_sd = ckpt.get("model_online_state_dict",
                            ckpt["model_state_dict"])
        unwrap_compile(self.model).load_state_dict(model_sd)

        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        relocate_optimizer_state(self.optimizer)
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if self.ema is not None and "ema_state_dict" in ckpt:
            self.ema.load_state_dict(ckpt["ema_state_dict"])
        if self.swa is not None and "swa_state_dict" in ckpt:
            self.swa.load_state_dict(ckpt["swa_state_dict"])

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
                restore_rng_state(rng)
                logger.info("Restored RNG state from checkpoint.")
            except Exception as e:  # pragma: no cover
                logger.warning("Failed to restore RNG state: %s", e)
        if self._is_dist and self._rank > 0:
            _reseed_rank_rng(
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
    def _load_pretrain(self, path: str, strict: bool, load_ema: bool) -> None:
        """仅加载权重作迁移初始化：不动 optimizer/scheduler/scaler/RNG，不推进 epoch，
        重对齐 EMA shadow 以免带着随机初始泄露。"""
        logger.info(
            "Loading pretrain weights: %s (strict=%s, load_ema=%s)",
            path, strict, load_ema)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        sd, source = extract_model_state_dict(ckpt, prefer_ema=load_ema)
        sd = strip_common_prefixes(sd)

        bare = unwrap_compile(self.model)
        if self.cfg.train.pretrain_upkern:
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
            sd = upkern_remap_state_dict(sd, bare)
            logger.info(
                "Pretrain: applied UpKern remap to %d depthwise tensor(s).",
                n_upkern)
        result = bare.load_state_dict(sd, strict=strict)
        missing = list(getattr(result, "missing_keys", []) or [])
        unexpected = list(getattr(result, "unexpected_keys", []) or [])

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


__all__ = ["Trainer"]
