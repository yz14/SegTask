"""自监督预训练通用训练循环（方法无关）。

只负责 optimizer / scheduler / AMP / EMA / ckpt / 日志；"破坏→损失"全部交给
:class:`ssltask.methods.base.SSLMethod`。优化器/调度器/AMP/EMA/输出目录/epochs/lr
复用 segtask ``train.*`` 配置与工具（不另造轮子）。

产出 ckpt 的 ``model_state_dict`` 由 ``method.export_backbone_state_dict()`` 给出，键与
``segtask_v1.models.factory.build_model`` 同名 → 下游 ``train.pretrain`` 非严格加载衔接。
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from taskcore.data.augment import GPUAugmentor
from taskcore.engine.amp import autocast
from taskcore.engine.checkpoint import (
    AsyncCheckpointSaver, relocate_optimizer_state, restore_rng_state,
    state_to_cpu, unwrap_compile)
from taskcore.engine.dist_utils import (
    get_rank, get_world_size,
    is_dist_avail_and_initialized, is_main_process)
from taskcore.engine.optim import (
    WarmupScheduler, build_optimizer, build_scheduler)
from taskcore.engine.prefetch import CudaPrefetcher
from taskcore.engine.views import fold_depth_to_channels
from taskcore.utils.common import AverageMeter, Timer
from taskcore.engine.base_trainer import BaseTrainer, _reseed_rank_rng

from ..methods.base import SSLMethod

logger = logging.getLogger(__name__)


class SSLTrainer(BaseTrainer):
    """方法无关的 SSL 预训练 pipeline。"""

    def __init__(self, method: SSLMethod, cfg, ssl, train_loader, device):
        self.method = method
        self.cfg = cfg
        self.ssl = ssl
        self.device = device
        self.train_loader = train_loader
        tc = cfg.train
        model = method.module
        self.model = model

        # --- DDP（多卡）---------------------------------------------------
        # SSL 方法插件在 ``compute_loss`` 内直调子模块（dino 系 ``module.student``
        # 等），不经过单一 forward 入口，无法套 ``DistributedDataParallel``。改用
        # “初始广播 + accum 边界手动梯度 all-reduce 均值”：对任意调用模式都
        # 正确，代价是无反传-通信重叠（SSL 步内计算占比高，可接受）。
        self._rank = get_rank()
        self._world_size = get_world_size()
        self._is_main = is_main_process()
        self._is_dist = is_dist_avail_and_initialized() and self._world_size > 1
        if self._is_dist:
            with torch.no_grad():
                for t in model.state_dict().values():
                    if torch.is_tensor(t):
                        dist.broadcast(t, src=0)
            logger.info(
                "SSL DDP enabled: rank=%d/%d (manual grad all-reduce at "
                "accumulation boundaries; params/buffers broadcast from "
                "rank0).", self._rank, self._world_size)
        # 训练采样器识别（共用工程件，见 BaseTrainer；按 set_epoch 协议鸭子
        # 识别 sampler / batch_sampler）。
        self._setup_train_sampler()

        # --- Optimizer + scheduler (复用 segtask train.*) ---
        # 关键：scheduler / warmup 只在“梯度累积边界”(= 一次 optimizer.step)推进，
        # 故其时钟必须用 **optimizer-step** 计数，而非 micro-batch 计数。否则
        # grad_accum_steps=k 时整段 schedule 只会走完约 1/k，warmup 被拉长 ~k 倍。
        # 每 epoch 的边界数 = ceil(len(loader)/accum)，与方法内 EMA/温度 schedule
        # (configure_schedule) 保持同一时钟。
        self.grad_accum_steps = max(tc.grad_accum_steps, 1)
        self.optimizer = build_optimizer(model, cfg)
        micro_steps_per_epoch = len(train_loader)
        opt_steps_per_epoch = max(
            math.ceil(micro_steps_per_epoch / self.grad_accum_steps), 1)
        warmup_steps = tc.warmup_epochs * opt_steps_per_epoch
        total_steps = tc.epochs * opt_steps_per_epoch
        post_warmup = total_steps - warmup_steps
        base_scheduler = build_scheduler(
            self.optimizer, cfg, opt_steps_per_epoch,
            post_warmup_steps=post_warmup)
        # one_cycle 自带 warmup（warmup_epochs 已映射 pct_start，见
        # build_scheduler），外层不再叠加线性 warmup（口径同 seg/cls/det）。
        warmup_steps = 0 if tc.scheduler == "one_cycle" else warmup_steps
        self.scheduler = WarmupScheduler(
            self.optimizer, base_scheduler, warmup_steps=warmup_steps,
            warmup_lr=tc.warmup_lr, base_lr=tc.lr)
        self._opt_steps_per_epoch = opt_steps_per_epoch
        self._total_opt_steps = total_steps

        # --- AMP（共用工程件，见 BaseTrainer）---
        self._setup_amp()

        # --- channels_last 内存格式（共用工程件，见 BaseTrainer）-----------
        # 数值等价；Ampere+ 上 3D conv 可能提速。模型与 batch 同时转排布。
        self._setup_channels_last()

        # --- EMA (over the method's module; orthogonal to any method-internal teacher) ---
        # 绑裸模型（compile 包装前），shadow key 无 ``_orig_mod.`` 前缀。
        # 与 segtask trainer 一致接入 warmup（早期低 decay，避免随机初值长期拖累
        # shadow）与 ema_device（可 offload 到 CPU 省显存，配置 train.ema_device）。
        self._setup_ema()

        # --- 通用增强（仅重建类方法，见 SSLMethod.trainer_augment）-----------
        # 复用 segtask GPUAugmentor（cfg.augment 控制）：在 corruption/mask 之前
        # 对 batch 图像做空间/强度增强，增强后图即新的自洽重建样本。SSL 为
        # 单 FOV（multi_res_scales==[1.0]），max_scale=1。
        # dataset 现统一输出 3D (B,1,D,H,W)（含 2.5D），因此 GPUAugmentor 的 3D
        # 空间变换对 2.5D / 3D 均适用；2.5D 的深度折叠推迟到增强之后、
        # 送模型之前（见 ``_fold_batch``），与 segtask 送模型前口径一致。
        self._fold_2_5d = int(cfg.model.spatial_dims) == 2
        # z 轴过采样目标深度：dataset 抽 (1,eD,H,W)（eD=round(pD*ratio)），
        # trainer 在增强后沿 z 中心裁回 pD（与 segtask aug_oversample_ratio 一致）。
        self._patch_d = int(cfg.data.patch_size[0])
        self.augmentor = None
        if method.trainer_augment and bool(cfg.augment.enabled):
            # 逐 rank 分流的独立增强 RNG（与 seg trainer 同构）；训练循环传入的
            # image 是 H2D 私有拷贝且增强后即覆写 batch["image"]，满足
            # inplace 所有权契约，省一次入口 clone。
            _aug_seed = (int(tc.seed)
                         + 7919 * (self._rank + 1)) & 0x7FFFFFFF
            self.augmentor = GPUAugmentor(
                cfg.augment, max_scale=1.0, seed=_aug_seed, inplace=True)
            logger.info(
                "Trainer-level augmentation ENABLED for SSL method %r "
                "(GPUAugmentor, cfg.augment); spatial_dims=%d (2.5D folded "
                "after augment).", ssl.method, int(cfg.model.spatial_dims))

        self._setup_output_dir()

        # --- Async checkpoint saver（opt-in，仅 rank0；与 seg trainer 同款）--
        # save_async=True 时 state 先深拷到 CPU，后台线程 torch.save，主循环
        # 不再被写盘阻塞；fit 收尾 close 排空队列保证全部落盘。
        self._ckpt_saver = (AsyncCheckpointSaver()
                            if tc.save_async and self._is_main else None)

        self.best_loss = math.inf
        self.best_probe = -math.inf
        self._best_saved = False
        self._global_step = 0
        self.start_epoch = 0

        # --- 训练监控面板（公用工程件，见 BaseTrainer / taskcore.monitor；
        #     cfg.monitor 守卫，仅 rank0，失败隔离不阻断训练）--------------
        # SSL 口径：无验证集，val 位置留空；选模指标为 train loss / probe dice。
        _select_by = ("probe_dice"
                      if bool(getattr(ssl, "probe_enabled", False))
                      and bool(ssl.probe_select_best) else "loss")
        self._setup_monitor(
            resume_active=bool(tc.resume),
            run_name_default="ssl_run",
            save_best_metric=_select_by,
            save_best_mode="max" if _select_by == "probe_dice" else "min",
            save_best_criterion="",
            num_classes=0,
            config_meta={
                "ssl_method": ssl.method,
                "recon_loss": ssl.recon_loss,
                "batch_size": cfg.data.batch_size,
            })

        # 告知方法总优化步数（= boundary 数：每 grad_accum 个 micro-step 一次）。
        # 与上方 scheduler/warmup 完全同一时钟（optimizer-step）。供自蒸馏等方法
        # 预计算 EMA 动量 / teacher 温度的 cosine 调度（默认 no-op）。
        self.method.configure_schedule(self._total_opt_steps)

        # --- Online seg probe (§0.5): drives best selection by representation
        #     quality instead of the SSL proxy loss. Optional, isolated.
        #     DDP 下仅 rank0 跑探针（选模/落盘也仅 rank0）。 ---
        self.probe = None
        if bool(getattr(ssl, "probe_enabled", False)) and self._is_main:
            from ..eval.probe import SegProbe
            self.probe = SegProbe(cfg, ssl, device)
            logger.info(
                "Online seg probe ENABLED: every %d epoch(s), %d iters, "
                "select_best_by=%s.", ssl.probe_every, ssl.probe_iters,
                "probe_dice" if ssl.probe_select_best else "train_loss")

        # --- Resume (全状态：method/optimizer/scheduler/scaler/EMA/RNG) ---
        # 显式路径不存在即报错（fail-fast，防静默从头训；口径同 cls/det）。
        if tc.resume:
            if not os.path.isfile(tc.resume):
                raise FileNotFoundError(
                    f"train.resume checkpoint not found: {tc.resume!r}")
            self._load_resume(tc.resume)

        # --- torch.compile（最后：optimizer/EMA/resume 已绑裸模型参数）-----
        # 共用工程件（见 BaseTrainer._maybe_compile）。替换 ``method.module``：
        # 直接前向的重建类方法（genesis/simmim/spark 等 ``self.module(x)``）被
        # 编译；子模块直调的方法（dino 系 ``module.student``）经 OptimizedModule
        # 属性代理回到裸子模块，行为不变。各方法的
        # ``export_backbone_state_dict`` 已统一走 ``unwrap_compile``。
        self.method.module = self._maybe_compile(self.method.module)

    # ------------------------------------------------------------------
    # Model-health helpers（同母项目；仅 rank0 监测启用时调用）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _global_grad_norm(self) -> "float | None":
        """当前已 unscale 的全局梯度 L2 范数（末尾仅一次 .item() 同步）。"""
        grads = [p.grad for p in self.method.parameters()
                 if p.grad is not None]
        if not grads:
            return None
        norms = torch._foreach_norm(grads, 2)
        return float(torch.linalg.vector_norm(
            torch.stack([n.float() for n in norms])).item())

    @torch.no_grad()
    def _global_weight_norm(self) -> "float | None":
        """全部参数的全局 L2 范数（每 epoch 仅算一次）。"""
        params = [p.detach() for p in self.method.parameters()]
        if not params:
            return None
        norms = torch._foreach_norm(params, 2)
        return float(torch.linalg.vector_norm(
            torch.stack([n.float() for n in norms])).item())

    def _collect_health_metrics(
        self, out: Dict[str, float], *,
        grad_norm_meter: AverageMeter, grad_norm_max: float,
        nonfinite_steps: int, clipped_steps: int, opt_steps: int,
        grad_clip_norm: float,
    ) -> None:
        """把本 epoch 聚合的健康指标并入 ``out``（仅写入有意义的键）。"""
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

    # ------------------------------------------------------------------
    def _prepare(self, batch: Dict) -> Dict:
        """把 batch 内张量搬到 device 并转 fp32（非张量原样透传）。"""
        out: Dict = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device, non_blocking=True).float()
            else:
                out[k] = v
        # channels_last 仅在不需后续折叠（3D）时在此施加；2.5D 待 ``_fold_batch``
        # 把 (B,1,D,H,W)→(B,D,H,W) 后再转成 4D channels_last（避免对 5D 张量
        # 施加 4D 内存格式导致报错/无效）。
        if (self._memory_format is not None and "image" in out
                and not self._fold_2_5d):
            out["image"] = out["image"].to(memory_format=self._memory_format)
        return out

    def _center_crop_z(self, batch: Dict) -> Dict:
        """沿 z 轴把 (B,1,eD,H,W) 中心裁回 (B,1,pD,H,W)（eD=round(pD*ratio)）。
        在数据增强之后、``_fold_batch`` 之前调用；ratio==1.0（eD==pD）时为 no-op。"""
        img = batch.get("image")
        if (torch.is_tensor(img) and img.dim() == 5
                and img.shape[2] > self._patch_d):
            start = (img.shape[2] - self._patch_d) // 2
            batch["image"] = img[:, :, start:start + self._patch_d]
        return batch

    def _fold_batch(self, batch: Dict) -> Dict:
        """2.5D：把 (B,1,D,H,W) 折成 (B,D,H,W)（D→通道），与 segtask 送模型前
        ``squeeze_2_5d`` 口径一致；3D 原样返回。在数据增强之后、compute_loss
        之前调用，使 3D GPUAugmentor 可作用于 2.5D 样本。"""
        if not self._fold_2_5d:
            return batch
        img = batch.get("image")
        if torch.is_tensor(img) and img.dim() == 5:
            img = fold_depth_to_channels(img)
            if self._memory_format is not None:
                img = img.to(memory_format=self._memory_format)
            batch["image"] = img
        return batch

    def _export_state_dict(self) -> Dict:
        """EMA 优先的可迁移骨干权重快照（键与 build_model 同名）。"""
        if self.ema is not None:
            bare = unwrap_compile(self.method.module)
            self.ema.apply_shadow(bare)
            try:
                return self.method.export_backbone_state_dict()
            finally:
                self.ema.restore(bare)
        return self.method.export_backbone_state_dict()

    @staticmethod
    def _atomic_save(obj: Dict, path: Path) -> None:
        """原子落盘：先写同目录临时文件 + fsync，再 ``os.replace`` 覆盖目标。
        避免训练/断电中断留下截断的半份 ckpt（``torch.load`` 直接崩）。"""
        tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
        with open(tmp, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)                 # 同文件系统内原子替换

    def _write_ckpt(self, state: Dict, path: Path) -> None:
        """落盘一份 ckpt：save_async 时深拷 CPU 后交后台线程（指纹已在主线程
        对在线张量算好，深拷后字节一致），否则同步原子写。"""
        if self._ckpt_saver is not None:
            self._ckpt_saver.submit(
                state_to_cpu(state), path,
                on_done=lambda p=path: logger.debug(
                    "Async checkpoint saved: %s", p))
        else:
            self._atomic_save(state, path)

    @staticmethod
    def _state_fingerprint(state: Dict[str, torch.Tensor]) -> str:
        """权重快照内容指纹（sha256）：写入 ckpt，加载时校验完整性/一致性。
        对键排序后按 (name, dtype, shape, bytes) 累积，跨进程稳定。"""
        h = hashlib.sha256()
        for k in sorted(state):
            v = state[k]
            h.update(k.encode("utf-8"))
            if torch.is_tensor(v):
                h.update(str(v.dtype).encode("utf-8"))
                h.update(str(tuple(v.shape)).encode("utf-8"))
                h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def _save(self, epoch: int, tag: str) -> Path:
        path = self.output_dir / f"ssl_{tag}.pt"
        if not self._is_main:          # DDP：仅 rank0 落盘
            if tag == "best":
                self._best_saved = True
            return path
        model_state = self._export_state_dict()
        self._write_ckpt({
            "epoch": epoch,
            "model_state_dict": model_state,
            "ssl_method": self.ssl.method,
            "best_loss": self.best_loss,
            "best_probe": self.best_probe,
            "fingerprint": self._state_fingerprint(model_state),
        }, path)
        if tag == "best":
            self._best_saved = True
        return path

    def _save_resume(self, epoch: int) -> Path:
        """全状态 resume checkpoint（与 ssl_best/last 的导出快照正交）：
        method 完整 state_dict（含方法内 teacher/queue/center 等 buffer）+
        optimizer/scheduler/scaler/EMA + 进度/最优指标 + RNG。"""
        # ZeRO 优化器状态分片在各 rank：保存前需全 rank 集合式 consolidate 到
        # rank0（必须在非主 rank 早退之前调用，否则集合通信挂死）；consolidate
        # 后仅 rank0 持有全局状态，故 state 仅在 rank0 组装/落盘。
        if hasattr(self.optimizer, "consolidate_state_dict"):
            self.optimizer.consolidate_state_dict(to=0)
        path = self.output_dir / "ssl_resume.pt"
        if not self._is_main:          # DDP：仅 rank0 落盘
            return path
        rng_state = {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        state: Dict = {
            "epoch": epoch,
            "ssl_method": self.ssl.method,
            "method_state_dict": unwrap_compile(self.method.module).state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "global_step": self._global_step,
            "best_loss": self.best_loss,
            "best_probe": self.best_probe,
            "best_saved": self._best_saved,
            "rng_state": rng_state,
        }
        if self.ema is not None:
            state["ema_state_dict"] = self.ema.state_dict()
        state["fingerprint"] = self._state_fingerprint(
            state["method_state_dict"])
        self._write_ckpt(state, path)
        return path

    def _load_resume(self, path: str) -> None:
        logger.info("Resuming SSL pretrain from: %s", path)
        # ckpt 含 numpy/python RNG 对象，为本 trainer 自写；显式关 weights_only。
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        method = ckpt.get("ssl_method")
        if method is not None and method != self.ssl.method:
            raise ValueError(
                f"Resume ckpt was written by ssl.method={method!r}, but current "
                f"config uses {self.ssl.method!r}.")
        fp = ckpt.get("fingerprint")
        if fp is not None:
            actual = self._state_fingerprint(ckpt["method_state_dict"])
            if actual != fp:
                raise ValueError(
                    f"Resume ckpt fingerprint mismatch (expected {fp[:16]}…, "
                    f"got {actual[:16]}…): file corrupted or tampered.")
        unwrap_compile(self.method.module).load_state_dict(
            ckpt["method_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        relocate_optimizer_state(self.optimizer)
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if self.ema is not None and "ema_state_dict" in ckpt:
            self.ema.load_state_dict(ckpt["ema_state_dict"])
        self._global_step = int(ckpt.get("global_step", 0))
        self.method.on_resume(self._global_step)
        self.start_epoch = int(ckpt.get("epoch", -1)) + 1
        self.best_loss = float(ckpt.get("best_loss", math.inf))
        self.best_probe = float(ckpt.get("best_probe", -math.inf))
        self._best_saved = bool(ckpt.get("best_saved", False))
        rng = ckpt.get("rng_state")
        if rng:
            try:
                restore_rng_state(rng)
                logger.info("Restored RNG state from resume checkpoint.")
            except Exception as e:  # pragma: no cover
                logger.warning("Failed to restore RNG state: %s", e)
        # ckpt 的 RNG 快照来自 rank0；rank>0 重新分流，避免各 rank 退化同流。
        if self._is_dist and self._rank > 0:
            _reseed_rank_rng(
                self.cfg.train.seed, self._rank, self.start_epoch,
                self.cfg.train.deterministic)
        logger.info(
            "Resume: start_epoch=%d, global_step=%d, best_loss=%.5f, "
            "best_probe=%.4f", self.start_epoch, self._global_step,
            self.best_loss, self.best_probe)

    # ------------------------------------------------------------------
    def fit(self) -> Dict[str, float]:
        tc = self.cfg.train
        timer = Timer()
        total_params = sum(p.numel() for p in self.method.parameters()) / 1e6
        logger.info("=" * 60)
        logger.info("SSL pretrain (%s): %d epochs, device=%s",
                    self.ssl.method, tc.epochs, self.device)
        logger.info("Model params: %.2fM | recon_loss=%s | AMP=%s EMA=%s",
                    total_params, self.ssl.recon_loss,
                    self.use_amp, self.ema is not None)
        logger.info("Train batches: %d, output_dir=%s",
                    len(self.train_loader), self.output_dir)
        logger.info("=" * 60)

        use_probe_select = self.probe is not None and bool(self.ssl.probe_select_best)
        for epoch in range(self.start_epoch, tc.epochs):
            if (self._monitor is not None and self.device.type == "cuda"):
                torch.cuda.reset_peak_memory_stats(self.device)
            epoch_t0 = timer.elapsed()
            train_loss, train_metrics = self._train_epoch(epoch)
            elapsed = timer.elapsed()
            logger.info("[SSL epoch %d/%d] loss=%.5f lr=%.2e (%.1fs)",
                        epoch + 1, tc.epochs, train_loss,
                        self.scheduler.get_lr(), elapsed)

            improved_loss = train_loss < self.best_loss
            if improved_loss:
                self.best_loss = train_loss

            # --- online probe (§0.5): periodic representation-quality eval ---
            is_last = epoch == tc.epochs - 1
            probe_dice = None
            if self.probe is not None and (
                    (epoch + 1) % self.ssl.probe_every == 0 or is_last):
                try:
                    probe_dice = self.probe.evaluate(
                        self._export_state_dict())["probe_dice"]
                    logger.info("[SSL epoch %d] online probe Dice=%.4f "
                                "(best=%.4f)", epoch + 1, probe_dice,
                                max(self.best_probe, probe_dice))
                except Exception:  # 探针失败绝不打断预训练
                    logger.warning("Online probe failed at epoch %d; skipping.",
                                   epoch + 1, exc_info=True)
                    probe_dice = None
            improved_probe = probe_dice is not None and probe_dice > self.best_probe
            if improved_probe:
                self.best_probe = probe_dice

            # --- best ckpt selection: probe Dice (if enabled) else train loss ---
            if use_probe_select:
                if improved_probe:
                    p = self._save(epoch, "best")
                    logger.info("New best probe Dice=%.4f -> %s", probe_dice, p)
            elif improved_loss:
                p = self._save(epoch, "best")
                logger.info("New best SSL loss=%.5f -> %s", train_loss, p)

            if (epoch + 1) % self.ssl.save_every == 0 or is_last:
                self._save(epoch, "last")
                self._save_resume(epoch)

            # --- 监控面板：逐 epoch 落盘 + 节奏化重渲染（异常隔离）---
            if self._monitor is not None:
                # 选模指标镜像进 val，供 MetricsLogger._compute_best 与 charts
                # Validation 概览使用（SSL 无独立验证集）。
                val_metrics: Dict[str, float] = {}
                if probe_dice is not None:
                    train_metrics["probe_dice"] = float(probe_dice)
                    val_metrics["probe_dice"] = float(probe_dice)
                if "loss" in train_metrics:
                    val_metrics["loss"] = float(train_metrics["loss"])
                gpu_peak = (
                    torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
                    if self.device.type == "cuda" else None)
                is_best = (improved_probe if use_probe_select
                           else improved_loss)
                self._monitor_log_epoch(
                    epoch, train_metrics, val_metrics,
                    lr=self.scheduler.get_lr(),
                    gpu_peak_mib=gpu_peak, wall_time_s=elapsed - epoch_t0,
                    is_best=is_best, last_epoch=is_last)

        # 保底：若选模策略从未保存过 best（如探针全程失败），最后兜底存一次。
        if not self._best_saved:
            self._save(tc.epochs - 1, "best")
            logger.info("No best ckpt selected during training; saved final "
                        "state as ssl_best.pt (fallback).")

        self._monitor_finalize("finished")
        if self._ckpt_saver is not None:
            # 收尾前排空异步写盘队列；写盘异常在此抛出。
            self._ckpt_saver.close()
            self._ckpt_saver = None
        logger.info("SSL pretrain done. best_loss=%.5f, best_probe=%.4f. Use "
                    "ssl_best.pt via train.pretrain for downstream.",
                    self.best_loss, self.best_probe)
        out = {"best_loss": self.best_loss}
        if self.probe is not None:
            out["best_probe"] = self.best_probe
        return out

    @staticmethod
    def _effective_accum(step: int, total_steps: int, accum: int) -> int:
        """尾批 micro-batch 数不满 ``accum`` 时，用真实尾长作分母，
        以免最后一组 micro-batch 因被除以 ``accum`` 而权重偏小。"""
        if accum <= 1:
            return 1
        remainder = total_steps % accum
        partial_start = total_steps - remainder
        return remainder if (remainder > 0 and step >= partial_start) else accum

    def _sync_grads(self) -> None:
        """DDP：accum 边界对全部梯度做均值 all-reduce。

        方法插件直调子模块、无单一 forward 入口，故不用 DDP wrapper 而手动
        同步；与 wrapper 的 bucket all-reduce 数学等价（无重叠优化）。fp16 下
        梯度带 scale，均值后 scale 不变，GradScaler 语义不受影响。"""
        if not self._is_dist:
            return
        grads = [p.grad for p in self.method.parameters()
                 if p.grad is not None]
        if not grads:
            return
        torch._foreach_div_(grads, float(self._world_size))
        for g in grads:
            dist.all_reduce(g, op=dist.ReduceOp.SUM)

    def _reduce_meter_avg(self, meter: AverageMeter) -> float:
        """DDP：按样本数加权 all-reduce 各 rank 的 meter 均值，使 best_loss 判定与
        日志在所有副本上一致（各 rank 经 DistributedSampler 看不同分片，本地均值
        不同 → rank0 存 best 的决策否则只代表其自身分片）。非 DDP 直接返回均值。"""
        if not self._is_dist:
            return meter.avg
        t = torch.tensor(
            [meter.avg * meter.count, float(meter.count)],
            dtype=torch.float64, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total, count = float(t[0]), float(t[1])
        return total / count if count > 0 else meter.avg

    def _train_epoch(self, epoch: int) -> "tuple[float, Dict[str, float]]":
        """单 epoch 训练；返回 (平均 loss, 监控指标 dict（含方法 logs 与健康
        指标，仅监控启用时采集）)。"""
        self.method.train()
        loss_meter = AverageMeter()
        log_meters: Dict[str, AverageMeter] = {}
        tc = self.cfg.train
        accum = self.grad_accum_steps
        total_steps = len(self.train_loader)
        health_on = self._monitor is not None and self._health_monitor
        grad_norm_meter = AverageMeter()
        grad_norm_max = 0.0
        nonfinite_steps = 0
        clipped_steps = 0
        opt_steps = 0

        # DDP：每 epoch 重置 DistributedSampler 的洗牌种子。
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(epoch)

        self.optimizer.zero_grad(set_to_none=True)
        group_has_nonfinite = False
        # 待同步损失/日志缓存：只缓 device 标量（0-dim tensor），仅在日志步 /
        # bf16 累积边界处批量 ``.tolist()`` 一次性取回，从而免去每 micro-step 一次
        # host/device 同步（与 segtask trainer 的 pending 机制同源）。每项：
        # (step, 未缩放 loss 张量, batch_size, logs)。
        pending: "list[tuple[int, torch.Tensor, int, Dict]]" = []

        def _flush_pending() -> "float | None":
            """把 pending 中所有 device 标量（loss + 张量型 log 值）stack 后一次
            ``.tolist()`` 取回，更新 loss/log meter 与非有限计数；返回最后一个
            micro-step 的 loss（供日志打印）。单次 D2H 同步。"""
            nonlocal group_has_nonfinite, nonfinite_steps
            if not pending:
                return None
            flat: "list[torch.Tensor]" = []
            for _, lt, _, lg in pending:
                flat.append(lt.reshape(()).float())
                for v in lg.values():
                    if torch.is_tensor(v):
                        flat.append(v.reshape(()).float())
            vals = torch.stack(flat).tolist()
            vi = 0
            last: "float | None" = None
            for (s, _lt, bs_i, lg) in pending:
                lv = vals[vi]; vi += 1
                last = lv
                mlogs: Dict[str, float] = {}
                for k, v in lg.items():
                    if torch.is_tensor(v):
                        mlogs[k] = vals[vi]; vi += 1
                    else:
                        mlogs[k] = float(v)
                if math.isfinite(lv):
                    loss_meter.update(lv, bs_i)
                    if self._monitor is not None:
                        for k, fv in mlogs.items():
                            if math.isfinite(fv):
                                log_meters.setdefault(
                                    k, AverageMeter()).update(fv, bs_i)
                else:
                    group_has_nonfinite = True
                    nonfinite_steps += 1
                    logger.warning(
                        "Non-finite SSL loss (%s) at epoch %d step %d/%d; "
                        "excluded from loss meter (surrounding optimizer step "
                        "skipped).", lv, epoch + 1, s + 1, total_steps)
            pending.clear()
            return last

        # prefetch_to_gpu：独立 copy stream 提前一个 batch 上卡；顶层 Tensor
        # 已在 device 时下方 _prepare 的 .to(device) 退化为 no-op。
        batch_iter = self.train_loader
        if tc.prefetch_to_gpu and self.device.type == "cuda":
            batch_iter = CudaPrefetcher(self.train_loader, self.device)

        for step, batch in enumerate(batch_iter):
            batch = self._prepare(batch)
            bs = batch["image"].shape[0] if "image" in batch else tc.batch_size
            if self.augmentor is not None and "image" in batch:
                img = batch["image"]
                # image-only：传入 dummy 单通道 label 满足管道接口，结果丢弃。
                img, _, _ = self.augmentor(
                    img, torch.zeros_like(img[:, :1]))
                batch["image"] = img
            # 增强（在 3D 体上）完成后：先沿 z 中心裁掉过采样余量（eD→pD），
            # 再（2.5D）把深度折进通道，最后送方法。
            batch = self._center_crop_z(batch)
            batch = self._fold_batch(batch)

            with autocast(device_type="cuda", enabled=self.use_amp,
                          dtype=self.amp_dtype):
                loss, logs = self.method.compute_loss(batch)
            # 尾批 micro-batch 数不满 accum 时用真实尾长作分母，避免尾组梯度
            # 因除以 accum 而权重偏小。
            effective_accum = self._effective_accum(step, total_steps, accum)
            loss_unscaled = loss.detach()          # 缓存未缩放全量 loss，暂不同步
            if effective_accum > 1:
                loss = loss / effective_accum
            self.scaler.scale(loss).backward()
            pending.append((step, loss_unscaled, bs, logs))

            is_boundary = ((step + 1) % accum == 0 or (step + 1) == total_steps)
            is_log_step = (step + 1) % tc.log_every == 0 or step == 0
            # bf16/fp32 无 GradScaler 保护：累积边界须先取回本组 loss 有限性以决定
            # 跳步 → 边界（非 scaler）必 flush；日志步也 flush 以打印。
            step_loss = None
            if is_log_step or (is_boundary and not self._scaler_active):
                step_loss = _flush_pending()

            if is_boundary:
                # DDP：先均值 all-reduce 各 rank 梯度（与 DDP wrapper 数学等价）。
                self._sync_grads()
                result = self._optimizer_step_boundary(
                    group_has_nonfinite=group_has_nonfinite,
                    epoch=epoch, step=step, total_steps=total_steps,
                    parameters=self.method.parameters(),
                    ema_module=unwrap_compile(self.method.module),
                    before_step=self.method.on_before_optimizer_step,
                    warn_scaler_skip=True,
                    always_step_scheduler=True)
                group_has_nonfinite = False
                stepped = result.stepped
                grad_norm_val = result.grad_norm
                if result.skipped_nonfinite:
                    if health_on:
                        opt_steps += 1
                elif result.scaler_skipped:
                    nonfinite_steps += 1
                else:
                    opt_steps += 1
                    if health_on and grad_norm_val is not None and \
                            math.isfinite(grad_norm_val):
                        grad_norm_meter.update(grad_norm_val)
                        grad_norm_max = max(grad_norm_max, grad_norm_val)
                        if (tc.grad_clip_norm > 0
                                and grad_norm_val > tc.grad_clip_norm):
                            clipped_steps += 1
                self._global_step += 1
                self.method.on_after_step(self._global_step, stepped=stepped)

            if is_log_step and step_loss is not None:
                logger.debug("  [%d/%d] loss=%.5f lr=%.2e",
                             step + 1, total_steps, step_loss,
                             self.scheduler.get_lr())

        # 取回尾部残留（fp16 仅在日志步 flush，末组可能未取回）。
        _flush_pending()
        epoch_loss = self._reduce_meter_avg(loss_meter)
        metrics: Dict[str, float] = {"loss": epoch_loss}
        if self._monitor is not None:
            for k, m in log_meters.items():
                metrics[k] = m.avg
            if health_on:
                try:
                    self._collect_health_metrics(
                        metrics, grad_norm_meter=grad_norm_meter,
                        grad_norm_max=grad_norm_max,
                        nonfinite_steps=nonfinite_steps,
                        clipped_steps=clipped_steps, opt_steps=opt_steps,
                        grad_clip_norm=tc.grad_clip_norm)
                except Exception:  # 监测失败绝不打断训练
                    logger.warning(
                        "SSL health metric collection failed; skipping.",
                        exc_info=True)
        return epoch_loss, metrics


__all__ = ["SSLTrainer"]
