"""3D / 2.5D 分割训练 pipeline：AMP / EMA / warmup / 累积 / compile / DS / checkpoint。

Round 2 重构后，``Trainer`` 不再判断训练模式 —— 所有"如何把 batch 重塑为
``(model_input, supervision)``、如何把 ``model_output`` 折成 loss"的逻辑都
归口到 ``self.pipeline: ViewPipeline``（见 ``trainer.pipelines``）。

``Trainer`` 仅协调：模型 / 优化器 / 调度器 / scaler / EMA / 增强 / 训练循环 /
checkpoint I/O。

为兼容现有测试与外部代码，部分历史属性（如 ``is_2_5d`` / ``keep_native_multi_res``
/ ``per_view_depths``）与方法（``_split_views_*`` / ``_squeeze_*`` / ``_center_crop``
/ ``_compute_loss_*``）以 thin shim 形式保留，全部从 ``self.pipeline`` 或
``trainer.views`` 模块取值/委托。
"""

from __future__ import annotations

import logging
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from colorama import Fore, Style

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import Config
from ..data.augment import GPUAugmentor
from ..losses.losses import build_loss
from ..models.unet import UNet3D
from ..utils import (
    AverageMeter, ModelEMA, Timer, compute_dice_per_class,
)
from . import views
from .amp import (
    _AMP_DTYPES,
    GradScaler,
    autocast,
    compute_loss_fp32,
    resolve_auto_amp_dtype,
)
from .breakdown import collect_multi_res_breakdown, format_breakdown
from .checkpoint import (
    extract_model_state_dict,
    strip_common_prefixes,
    unwrap_compile,
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
            self.optimizer, cfg, steps_per_epoch, post_warmup_steps=post_warmup)
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
        self.ema = ModelEMA(self.model, tc.ema_decay) if tc.use_ema else None

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

        # --- 增强 ------------------------------------------------------
        _scales = cfg.data.multi_res_scales or [1.0]
        self.augmentor = GPUAugmentor(cfg.augment, max_scale=max(_scales))

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

        # --- Validation / model-selection evaluator -------------------
        # medium（随机 patch 指标）/ high（整卷滑窗指标）由 val_metric_mode 决定；
        # 两者产出同结构 metrics dict，选模/调度/ckpt 逻辑无需分支。
        self.evaluator = build_val_evaluator(self)
        logger.info("Validation metric mode: %s (evaluator=%s)",
                    tc.val_metric_mode, type(self.evaluator).__name__)

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
            if tc.resume and not os.path.isfile(tc.resume):
                logger.warning(
                    "`train.resume` is set but file not found: %s. "
                    "Training will start from scratch.", tc.resume)
            if tc.pretrain and not os.path.isfile(tc.pretrain):
                logger.warning(
                    "`train.pretrain` is set but file not found: %s. "
                    "Training will start from scratch.", tc.pretrain)

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

        self.optimizer.zero_grad(set_to_none=True)
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
            
            # import SimpleITK as sitk
            # debug_dir = "./debug0603-1"
            # os.makedirs(debug_dir, exist_ok=True)
            # imgs = torch.split(image, [12, 18, 24], dim=1)
            # lbls = [sup.label_main] + sup.aux_labels
            # maps = [sup.wmap_main]  + sup.aux_wmaps
            # for jj, (aa,bb,cc) in enumerate(zip(imgs, lbls, maps)): # n views
            #     for j, (a,b,c) in enumerate(zip(aa, bb, cc)):
            #         a = a.detach().cpu().numpy()
            #         a = sitk.GetImageFromArray(a)
            #         sitk.WriteImage(a, f'{debug_dir}/{j}-{jj}-a.nii.gz')
            #         a = b.detach().cpu().numpy()
            #         a = sitk.GetImageFromArray(a)
            #         sitk.WriteImage(a, f'{debug_dir}/{j}-{jj}-b.nii.gz')
            #         a = c.detach().cpu().numpy()
            #         a = sitk.GetImageFromArray(a)
            #         sitk.WriteImage(a, f'{debug_dir}/{j}-{jj}-c.nii.gz')
            # raise

            effective_accum = self._effective_accum(step, total_steps, accum)

            # Forward AMP / Loss fp32（Dice/BCE 在 fp16 下汇总易溢出 → NaN）
            with autocast(device_type="cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                pred = self.model(image)
            breakdown: Dict[str, float] = {}
            loss = self.pipeline.compute_loss(pred, sup, breakdown=breakdown)
            collect_multi_res_breakdown(self.criterion, self.aux_loss_fn, breakdown)
            if effective_accum > 1:
                loss = loss / effective_accum

            self.scaler.scale(loss).backward()

            # 参数更新
            is_step_boundary = ((step + 1) % accum == 0 or (step + 1) == total_steps)
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

            # 记录未缩放损失，丢弃非有限值避免污染均值（GradScaler 会跳该 step）
            step_loss = (loss.item() * effective_accum
                         if effective_accum > 1 else loss.item())
            if math.isfinite(step_loss):
                loss_meter.update(step_loss, image.shape[0])
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
                    step_loss, epoch + 1, step + 1, total_steps)

            if (step + 1) % tc.log_every == 0 or step == 0:
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
                    step + 1, total_steps, step_loss, mean_dice,
                    self.scheduler.get_lr(), aux_msg)

        out = {"loss": loss_meter.avg, "dice": dice_meter.avg}
        for name, meter in component_meters.items():
            out[name] = meter.avg
        return out

    # 类级别静态别名：保持 ``self._compute_loss_fp32`` / ``self._format_breakdown`` 旧调用约定不破坏。
    _compute_loss_fp32 = staticmethod(compute_loss_fp32)
    _format_breakdown = staticmethod(format_breakdown)

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
        with self._ema_swapped():
            return self.evaluator.evaluate(epoch)

    # ==================================================================
    # Backward-compatibility shims for unit tests
    # 仅保留三个仍被现有测试直接调用的方法（其余 shim 已在 Round 3 移除）：
    #   * test_keep_native_multi_res_trainer.py → _split_views_native_3d
    #   * test_keep_native_view_depth.py            → _split_views_native_d
    #   * test_segtask_v1.py::TestTrainerCenterCrop → _center_crop
    # 所有逻辑委托至 ``segtask_v1.trainer.views``；新代码请直接用纯函数 / pipeline。
    # ==================================================================
    def _split_views_native_3d(
        self, image: torch.Tensor, label: torch.Tensor,
        wmap: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if not self.keep_native_multi_res:
            raise RuntimeError(
                "_split_views_native_3d called but "
                "keep_native_multi_res=False")
        return views.split_views_native_3d(
            image, label, wmap,
            target_patch_size=self.target_patch_size,
            mr_native_sizes=self._mr_native_sizes,
            patch_size=tuple(int(x) for x in self.cfg.data.patch_size),
        )

    def _split_views_native_d(
        self, image: torch.Tensor, label: torch.Tensor,
        wmap: Optional[torch.Tensor],
    ):
        if not self.keep_native_view_depth:
            raise RuntimeError(
                "_split_views_native_d called but keep_native_view_depth=False")
        return views.split_views_native_d(
            image, label, wmap,
            per_view_depths=self.per_view_depths,
            target_patch_size=self.target_patch_size,
        )

    def _center_crop(
        self, image: torch.Tensor, label: torch.Tensor,
        wmap: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return views.center_crop(image, label, wmap, self.target_patch_size)

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
                self.ema.apply_shadow(self.model)
                try:
                    state["model_state_dict"] = unwrap_compile(
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

        model_sd = ckpt.get("model_online_state_dict",
                            ckpt["model_state_dict"])
        unwrap_compile(self.model).load_state_dict(model_sd)

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
