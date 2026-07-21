"""检测训练器（复用 segtask 训练基建：optim / warmup / AMP / EMA）。

* 训练：patch + 变长框 → ``DetectorModel`` 损失 dict → 求和反传；
  head 内部损失已在 fp32 计算（focal / GIoU 数值敏感）。空间增强（flip）
  在 dataset 内与框联动施加；强度增强复用 seg ``GPUAugmentor``（仅强度
  分支，不改变几何，故不动框）。bf16/fp32（无 GradScaler）路径下 loss/
  梯度非有限时丢弃本 accum 组，不推 scheduler/EMA（口径同 segtask）。
* 验证：patch 级 predict → mAP@``det.eval_iou_thresh``（体级 FROC 由
  predictor 在拼接后的 3D 框上给出）；按 ``det.save_best_metric`` 选模。
* encoder 差分学习率复用 taskcore 的分组实现；warmup 段保留各组倍率
  （:class:`taskcore.engine.optim.GroupWarmupScheduler`）。
* 工程能力（口径同 seg / cls Trainer）：每 epoch 落盘 ``latest_model.pth``
  （含 model/EMA/optimizer/scheduler/scaler/epoch/best 状态）与
  ``history.json``；``train.resume`` 完整恢复续训；``train.early_stopping``
  连续 N 次验证无提升时提前停止。
"""

from __future__ import annotations

import dataclasses
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
from taskcore.engine.checkpoint import AsyncCheckpointSaver
from taskcore.engine.dist_utils import all_gather_objects
from taskcore.engine.optim import (
    GroupWarmupScheduler,
    build_optimizer,
    build_optimizer_with_lr_mult,
    build_scheduler,
)
from taskcore.engine.prefetch import CudaPrefetcher
from taskcore.utils.common import AverageMeter
from taskcore.engine.base_trainer import BaseTrainer

from ..config import DetConfig, resolve_num_classes
from ..metrics import detection_map

logger = logging.getLogger(__name__)


def _intensity_only_augcfg(aug):
    """AugConfig → 仅保留强度分支（空间/遮挡概率清零）的副本。

    检测的空间增强必须与框联动（dataset 内 flip 已联动）；affine/elastic/
    grid_dropout 无框变换实现，这里显式关闭，避免图像几何与框错位。
    """
    return dataclasses.replace(
        aug,
        random_flip_prob=0.0,
        random_affine_prob=0.0,
        elastic_deform_prob=0.0,
        grid_dropout_prob=0.0,
    )


class DetTrainer(BaseTrainer):
    """检测训练器；``fit()`` 返回 best 指标摘要。"""

    def __init__(self, model: nn.Module, cfg: SegConfig, det: DetConfig,
                 train_loader, val_loader, device: torch.device):
        self.cfg = cfg
        self.det = det
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(device)
        tc = cfg.train

        # 可选 channels_last 内存格式（数值等价；共用工程件，见 BaseTrainer）。
        self._setup_channels_last()

        self.num_classes = resolve_num_classes(det, cfg)
        if abs(det.encoder_lr_mult - 1.0) > 1e-9:
            self.optimizer = build_optimizer_with_lr_mult(
                self.model, cfg, det.encoder_lr_mult)
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
        self.scheduler = GroupWarmupScheduler(
            self.optimizer, base_scheduler, warmup_steps=warmup_steps,
            warmup_lr=tc.warmup_lr, base_lr=tc.lr,
            group_base_lrs=[float(pg["lr"])
                            for pg in self.optimizer.param_groups])

        # --- AMP / EMA（共用工程件，见 BaseTrainer）------------------------
        self._setup_amp()
        self._setup_ema()
        self.grad_clip_norm = float(tc.grad_clip_norm)

        # --- torch.compile（最后：optimizer / EMA 已绑裸参数）----------
        self._maybe_compile()

        # --- DDP 装配与训练采样器识别（公用工程件，见 BaseTrainer）-------
        # 单卡时 fwd_model 即裸模块，路径零变化；多卡时前向走 DDP 包装。
        self._setup_ddp()
        self._setup_train_sampler()

        # 强度增强（复用 seg GPUAugmentor，仅强度分支；框不受影响）。
        # 逐 rank 分流的独立增强 RNG（与 seg trainer 同构）；_to_device 传入的
        # x 是 H2D 私有拷贝（2.5D 时为其 unsqueeze 视图）且增强后即覆写 img，
        # 满足 inplace 所有权契约，省一次入口 clone。
        _aug_seed = (int(tc.seed) + 7919 * (self._rank + 1)) & 0x7FFFFFFF
        self.augmentor = (
            GPUAugmentor(_intensity_only_augcfg(cfg.augment), max_scale=1.0,
                         seed=_aug_seed, inplace=True)
            if cfg.augment.enabled else None)
        self.fold_2_5d = int(cfg.model.spatial_dims) == 2

        self._setup_output_dir()
        # save_async=True 时权重先深拷到 CPU，后台线程 torch.save，主循环
        # 不再被写盘阻塞；fit 收尾 wait+close 保证全部落盘（同 seg）。
        self._ckpt_saver = (AsyncCheckpointSaver()
                            if self.cfg.train.save_async and self._is_main
                            else None)
        # SWA 尾段权重平均（opt-in，公用工程件，见 BaseTrainer）。
        self._setup_swa()
        self.best_key = det.save_best_metric        # map | loss
        self.best_sign = -1.0 if self.best_key == "loss" else 1.0
        self._setup_best_tracking(mode="max")
        self.history: "list[Dict[str, float]]" = []

        # --- 训练监测仪表盘（公用工程件，见 BaseTrainer / taskcore.monitor；
        #     cfg.monitor 守卫，失败隔离不阻断训练）-----------------------
        self._setup_monitor(
            resume_active=bool(str(tc.resume or "").strip()),
            run_name_default="det_run",
            save_best_metric=self.best_key,
            save_best_mode="min" if self.best_key == "loss" else "max",
            save_best_criterion="",
            num_classes=0,
            config_meta={
                "batch_size": cfg.data.batch_size,
                "num_classes": self.num_classes,
            })

    # ------------------------------------------------------------------
    def _to_device(self, batch, augment: bool = False):
        img = batch["image"].to(self.device, non_blocking=True).float()
        boxes = [b.to(self.device) for b in batch["boxes"]]
        labels = [l.to(self.device) for l in batch["labels"]]
        if augment and self.augmentor is not None:
            x = img if not self.fold_2_5d else img.unsqueeze(1)
            # 强度分支不使用 label（几何分支已清零）；zeros 仅占位。
            x, _, _ = self.augmentor(x, torch.zeros_like(x))
            img = x if not self.fold_2_5d else x.squeeze(1)
        return img, boxes, labels

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(epoch)   # 多卡重洗（单卡 no-op）
        self.model.train()
        loss_meter = AverageMeter()
        part_meters: Dict[str, AverageMeter] = {}
        accum = self.grad_accum_steps
        total = len(self.train_loader)
        grad_norm_meter = AverageMeter()
        grad_norm_max = 0.0
        nonfinite_steps = 0
        clipped_steps = 0
        opt_steps = 0
        # GPU 预取（口径同 seg/cls Trainer）：独立 copy stream 提前一个
        # batch 上卡；boxes/labels 是 list 字段由预取器原样透传。
        batch_iter = self.train_loader
        if self.cfg.train.prefetch_to_gpu and self.device.type == "cuda":
            batch_iter = CudaPrefetcher(self.train_loader, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        group_has_nonfinite = False
        # 未缩放损失 GPU 缓存，日志步 / bf16 边界单次同步（同 seg/cls）。
        pending: "list[tuple[int, torch.Tensor, int]]" = []

        def _flush_pending() -> "float | None":
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
            img, boxes, labels = self._to_device(batch, augment=True)
            # 尾组 micro-batch 不满 accum 时用真实尾长作分母。
            eff_accum = self._effective_accum(step, total, accum)
            is_boundary = (step + 1) % accum == 0 or (step + 1) == total
            # 非边界步免 all-reduce；forward 也必须放进 no_sync（同 seg）。
            sync_ctx = self._ddp_no_sync(is_boundary)
            with sync_ctx:
                with torch.autocast(device_type=self.device.type,
                                    enabled=self.use_amp, dtype=self.amp_dtype):
                    losses = self.fwd_model(img, boxes, labels)
                loss = sum(losses.values())
                loss_scaled = loss / eff_accum if eff_accum > 1 else loss
                self.scaler.scale(loss_scaled).backward()

            pending.append((step, loss.detach(), img.shape[0]))
            is_log_step = ((step + 1) % max(self.cfg.train.log_every, 1) == 0
                           or step == 0)
            loss_val: "float | None" = None
            if is_log_step or (is_boundary and not self._scaler_active):
                loss_val = _flush_pending()
            # 分项损失仅在日志步取标量，避免每步多次 D2H。
            if is_log_step and loss_val is not None and math.isfinite(loss_val):
                for k, v in losses.items():
                    part_meters.setdefault(k, AverageMeter()).update(
                        float(v.detach().item()), img.shape[0])

            if is_boundary:
                result = self._optimizer_step_boundary(
                    group_has_nonfinite=group_has_nonfinite,
                    epoch=epoch, step=step, total_steps=total,
                    grad_clip_norm=self.grad_clip_norm)
                group_has_nonfinite = False
                grad_norm_val = result.grad_norm
                if result.skipped_nonfinite:
                    continue
                opt_steps += 1
                if grad_norm_val is not None and math.isfinite(grad_norm_val):
                    grad_norm_meter.update(grad_norm_val)
                    grad_norm_max = max(grad_norm_max, grad_norm_val)
                    if (self.grad_clip_norm > 0
                            and grad_norm_val > self.grad_clip_norm):
                        clipped_steps += 1

            if is_log_step:
                logger.debug(
                    "  [%d/%d] loss=%.4f (%s) lr=%.2e",
                    step + 1, total,
                    loss_val if loss_val is not None else float("nan"),
                    ", ".join(f"{k}={float(v.detach().item()):.3f}"
                              for k, v in losses.items()),
                    self.scheduler.get_lr())
        _flush_pending()
        out = {"loss": loss_meter.avg}
        out.update({k: m.avg for k, m in part_meters.items()})
        self._collect_health_metrics(
            out,
            grad_norm_meter=grad_norm_meter,
            grad_norm_max=grad_norm_max,
            nonfinite_steps=nonfinite_steps,
            clipped_steps=clipped_steps,
            opt_steps=opt_steps,
            grad_clip_norm=self.grad_clip_norm)
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
            img, _, _ = self._to_device(batch)
            with torch.autocast(device_type=self.device.type,
                                enabled=self.use_amp, dtype=self.amp_dtype):
                self.model.extract_pyramid(img)

    @torch.no_grad()
    def _validate_inner(self, epoch: int) -> Dict[str, float]:
        """在**当前权重**上评测（不换 EMA；供 _validate / SWA 收尾共用）。"""
        preds, gts = [], []
        loss_meter = AverageMeter()
        for batch in self.val_loader:
            img, boxes, labels = self._to_device(batch)
            # 金字塔只前向一次，predict 与 loss 共用；autocast 口径同
            # 训练/推理（head 内损失仍 fp32）。
            img_size = list(img.shape[2:])
            with torch.autocast(device_type=self.device.type,
                                enabled=self.use_amp,
                                dtype=self.amp_dtype):
                feats = self.model.extract_pyramid(img)
                dets = self.model.det_head.predict(feats, img_size)
                losses = self.model.det_head.compute_loss(
                    feats, boxes, labels, img_size)
            loss_val = float(sum(losses.values()).item())
            if math.isfinite(loss_val):
                loss_meter.update(loss_val, img.shape[0])
            else:
                logger.warning("Non-finite val loss (%s) at epoch %d; "
                               "excluded from val_loss.", loss_val,
                               epoch + 1)
            preds.extend([{k: v.cpu() for k, v in d.items()}
                          for d in dets])
            gts.extend([(b.cpu(), l.cpu())
                        for b, l in zip(boxes, labels)])
        # DDP：val 集按 batch 块分片到各 rank；把预测/真值聚齐后在每个
        # rank 上算全集 mAP（AP 不可分解，不能逐 rank 算后平均），选模/
        # 早停决策各 rank 天然一致。
        parts = all_gather_objects({
            "preds": preds, "gts": gts,
            "loss_sum": float(loss_meter.sum),
            "loss_count": int(loss_meter.count),
        })
        preds = [d for p in parts for d in p["preds"]]
        gts = [g for p in parts for g in p["gts"]]
        loss_meter.sum = sum(p["loss_sum"] for p in parts)
        loss_meter.count = sum(p["loss_count"] for p in parts)
        m = detection_map(preds, gts, self.num_classes,
                          self.det.eval_iou_thresh)
        m["loss"] = loss_meter.avg
        return m

    # best/latest 落盘走 BaseTrainer 模板（EMA 为主 + 全量续训状态）。
    _ckpt_task_label = "det"

    def _ckpt_extra_state(self) -> Dict:
        return {"det_config": self.det}

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
        epochs_no_improve = max(self.start_epoch - 1 - self.best_epoch, 0)
        final_status = "completed"
        for epoch in range(self.start_epoch, self.cfg.train.epochs):
            epoch_t0 = time.time()
            tr = self._train_epoch(epoch)
            val = self._validate(epoch)
            self.scheduler.step_epoch(val.get(self.best_key))
            logger.info(
                "Epoch %d/%d: train_loss=%.4f val_loss=%.4f val_map=%.4f",
                epoch + 1, self.cfg.train.epochs, tr["loss"], val["loss"],
                val["map"])
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


__all__ = ["DetTrainer"]
