"""验证期指标累加与选模评估策略（TODO #5）。

把 ``Trainer._validate`` 中"累加 pooled 混淆量 → 闭式导出全部指标"的逻辑收敛到
``MetricAccumulator``（medium / high 共用），并以 ``ValEvaluator`` 策略对象区分
两种"指标在什么预测上算"的口径：

* ``PatchValEvaluator``  (medium) — 遍历 ``val_loader`` 的随机 patch/切片，逐 batch
  前向取指标。快，但非整卷、z 向上下文被切断，与既有行为一致。
* ``VolumeValEvaluator`` (high)   — 对每个 val 整卷复用 ``Predictor`` 做与部署一致的
  滑窗推理后再取指标。最可靠但更慢。

两个 evaluator 唯一的差别是 ``(pred, target)`` 的来源；累加与导出完全共用
``MetricAccumulator``，故 medium / high 产出的 metrics dict 结构严格一致，
``Trainer`` 的选模 / 调度 / checkpoint 逻辑无需任何分支。工厂
``build_val_evaluator`` 按 ``cfg.train.val_metric_mode`` 选择实现。
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Union

import torch

from ..utils import (
    AverageMeter,
    derive_overlap_metrics,
    dice_batch_stats,
    harmonic_mean_metrics,
    surface_dice_batch_stats,
)
from .amp import autocast, compute_loss_fp32
from .dist_utils import (
    all_reduce_sum_,
    get_world_size,
    is_main_process,
    shard_for_rank,
)

if TYPE_CHECKING:  # pragma: no cover
    from .trainer import Trainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pooled metric accumulator (medium / high 共用)
# ---------------------------------------------------------------------------
class MetricAccumulator:
    """nnU-Net 风格 pooled 指标累加器。

    逐 batch / 逐卷累加混淆量（inter / pred_sum / target_sum / voxels / coverage）
    与可选 surface-dice 分子分母，``compute()`` 末尾闭式导出 dice / iou / recall /
    precision / vol_sim / mcc / surface_dice / combined / balanced 全部指标。

    所有累加张量常驻喂入张量的 device（GPU），仅在 ``compute()`` 一次性搬到 CPU。
    """

    def __init__(
        self,
        criterion: str,
        surface_dice_tolerance: int,
        surface_dice_weight: float,
        threshold: Union[float, Sequence[float]] = 0.5,
        surface_dice_tolerance_mm: float = 0.0,
        spacing: Optional[Sequence[float]] = None,
    ):
        crit = str(criterion).lower().strip()
        self._crit = crit
        # 二值化阈值：标量或逐前景类列表（与 predict.threshold 同源，使 medium 选模
        # 与部署阈值口径一致）。
        self._threshold = (
            float(threshold) if isinstance(threshold, (int, float))
            else [float(t) for t in threshold])
        # balanced 也需要 surface dice 参与调和均值。
        self.compute_sd = crit in ("dice+surface_dice", "balanced")
        self.sd_tol = int(surface_dice_tolerance)
        self.sd_w = float(surface_dice_weight)
        # 物理空间 (mm) 各向异性欧氏 NSD：tolerance_mm>0 且 spacing 非空时启用，
        # 否则沿用 voxel-Chebyshev@sd_tol。spacing 为每轴 mm（numpy 轴序）。
        self.sd_tol_mm = float(surface_dice_tolerance_mm)
        self.sd_spacing = (
            [float(s) for s in spacing] if spacing is not None else None)
        self.sd_physical = self.sd_tol_mm > 0.0 and self.sd_spacing is not None

        self.loss_meter = AverageMeter()
        self._inter = None
        self._pred_sum = None
        self._target_sum = None
        self._voxels = None
        self._cov = None
        self._sd_num = None
        self._sd_denom = None
        self.n_samples = 0

    @torch.no_grad()
    def update(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
        loss_value: Optional[float] = None,
        voxels_override: Optional[float] = None,
        pred_is_binary: bool = False,
    ) -> None:
        """累加单 batch / 单卷。

        ``pred_logits`` 形如 ``(B, num_fg, ...)`` 的 *logits*（指标算子内部自行做
        sigmoid + 阈值，阈值来自 ``predict.threshold``，默认 0.5）；``target`` 同形二值标签。``loss_value`` 为该 batch 的
        标量验证损失，``None`` 时不计入（high 模式无可逆 logits，故不产
        val_base_loss）。

        ``voxels_override``：喂入的张量若已按 pred∪GT bbox 裁剪，传入裁剪前的
        总体素数（B × 整卷空间体素），使 TN/MCC 仍按整卷口径严格等价。

        ``pred_is_binary=True`` 时 ``pred_logits`` 已是 {0,1} 二值体，指标算子
        跳过 sigmoid+阈值（high 模式已在上游按推理阈值二值化，结果逐位一致）。
        """
        if loss_value is not None and math.isfinite(loss_value):
            self.loss_meter.update(loss_value, pred_logits.shape[0])

        pred_f = pred_logits.float()
        thr = self._threshold
        if isinstance(thr, list):
            # 逐类阈值广播到 (C, 1, ..., 1)，右对齐后 C 对应通道轴。
            thr = torch.as_tensor(
                thr, dtype=pred_f.dtype, device=pred_f.device).view(
                    -1, *([1] * (pred_f.ndim - 2)))
        stats = dice_batch_stats(
            pred_f, target, threshold=thr, pred_is_binary=pred_is_binary)
        if voxels_override is not None:
            stats["voxels"] = torch.tensor(
                float(voxels_override), dtype=torch.float64,
                device=stats["voxels"].device)
        if self._inter is None:
            self._inter = stats["inter"].clone()
            self._pred_sum = stats["pred_sum"].clone()
            self._target_sum = stats["target_sum"].clone()
            self._voxels = stats["voxels"].clone()
            self._cov = stats["n_with_gt"].clone()
        else:
            self._inter += stats["inter"]
            self._pred_sum += stats["pred_sum"]
            self._target_sum += stats["target_sum"]
            self._voxels += stats["voxels"]
            self._cov += stats["n_with_gt"]

        if self.compute_sd:
            sd_stats = surface_dice_batch_stats(
                pred_f, target, tolerance=self.sd_tol,
                threshold=thr, pred_is_binary=pred_is_binary,
                tolerance_mm=self.sd_tol_mm if self.sd_physical else 0.0,
                spacing=self.sd_spacing if self.sd_physical else None)
            if self._sd_num is None:
                self._sd_num = sd_stats["sd_num"].clone()
                self._sd_denom = sd_stats["sd_denom"].clone()
            else:
                self._sd_num += sd_stats["sd_num"]
                self._sd_denom += sd_stats["sd_denom"]

        self.n_samples += int(pred_logits.shape[0])

    @torch.no_grad()
    def all_reduce(self, num_fg: int, device: torch.device) -> None:
        """多卡：将各 rank 累加的可加混淆量 all-reduce(SUM) 汇总。

        各 rank 处理不相交的样本子集，混淆量（inter / pred_sum / target_sum /
        voxels / cov / sd_num / sd_denom）均为跨样本求和，all-reduce 后与单进程在
        全集上累加**严格相等**（非近似），故 ``compute()`` 导出的 dice/sd/balanced
        等与单卡一致。某 rank 未分到任何样本时累加器为空，以零参与求和。

        每个可加量按其原生 (shape, dtype) 参与 all-reduce；空 rank 据此零初始化。
        单进程（world_size<=1）时为 no-op。
        """
        if get_world_size() <= 1:
            return

        # 各可加量的 (shape, dtype)，与 dice_batch_stats / surface_dice_batch_stats
        # 的返回一致：逐类量为 (num_fg,) float32，唯 voxels 为标量 () float64。
        # 未分到样本的 rank 据此以正确形状/类型零初始化，保各 rank all-reduce 形状对齐
        # （否则 collective 形状不一致会死锁）。保留各自原生 dtype，不强转 float32 以免
        # voxel 大计数精度损失。
        per_class = (int(num_fg),)
        specs = {
            "_inter":      (per_class, torch.float32),
            "_pred_sum":   (per_class, torch.float32),
            "_target_sum": (per_class, torch.float32),
            "_voxels":     ((),        torch.float64),
            "_cov":        (per_class, torch.float32),
        }
        if self.compute_sd:
            specs["_sd_num"]   = (per_class, torch.float32)
            specs["_sd_denom"] = (per_class, torch.float32)
        for name, (shape, dtype) in specs.items():
            t = getattr(self, name)
            if t is None:
                t = torch.zeros(shape, dtype=dtype, device=device)
            else:
                t = t.detach().to(device=device)
            all_reduce_sum_(t)
            setattr(self, name, t)

        # 样本计数与（medium 模式的）loss 累加量同样可加。
        agg = torch.tensor(
            [float(self.n_samples), float(self.loss_meter.sum),
             float(self.loss_meter.count)],
            dtype=torch.float64, device=device)
        all_reduce_sum_(agg)
        self.n_samples = int(round(agg[0].item()))
        self.loss_meter.sum = float(agg[1].item())
        self.loss_meter.count = int(round(agg[2].item()))

    @torch.no_grad()
    def compute(self, log_prefix: str = "Val", *, log: bool = True) -> Dict[str, float]:
        """闭式导出全部指标并打印一行概要。无样本时返回退化 dict。

        ``log=False`` 时不打印概要行（多卡下仅 rank0 打印，避免 N 倍重复）。
        """
        if self._inter is None:
            if log:
                logger.warning("%s: accumulator received no samples.", log_prefix)
            return {"val_base_loss": float("nan"), "mean_dice": 0.0}

        derived = derive_overlap_metrics(
            self._inter, self._pred_sum, self._target_sum, self._voxels)
        derived_cpu = {k: v.cpu() for k, v in derived.items()}
        dice_per_class = derived_cpu["dice"]
        iou_per_class = derived_cpu["iou"]
        rec_per_class = derived_cpu["recall"]
        pre_per_class = derived_cpu["precision"]
        vs_per_class = derived_cpu["vol_sim"]
        mcc_per_class = derived_cpu["mcc"]

        # 命名为 val_base_loss：只含裸 base_loss（无 DS/aux/topo 加权），与训练
        # 组合损失口径不同，避免被当作可与 train loss 直接对比的同量。
        val_base_loss = (self.loss_meter.avg if self.loss_meter.count > 0
                         else float("nan"))
        metrics: Dict[str, float] = {"val_base_loss": val_base_loss}
        for c in range(len(dice_per_class)):
            metrics[f"dice_class_{c}"] = dice_per_class[c].item()
            metrics[f"iou_class_{c}"] = iou_per_class[c].item()
            metrics[f"recall_class_{c}"] = rec_per_class[c].item()
            metrics[f"precision_class_{c}"] = pre_per_class[c].item()
            metrics[f"vol_sim_class_{c}"] = vs_per_class[c].item()
            metrics[f"mcc_class_{c}"] = mcc_per_class[c].item()

        # nnU-Net ignore_empty：整个 val 集都无 GT 的类（cov==0）从 mean_*/min_*
        # 中剔除，避免其退化值污染选模指标；全空时退回全类。
        gt_mask = (self._cov.cpu() > 0)
        if not bool(gt_mask.any()):
            gt_mask = torch.ones_like(gt_mask)

        def _masked_mean(v: torch.Tensor) -> float:
            return v[gt_mask].mean().item()

        def _masked_min(v: torch.Tensor) -> float:
            return v[gt_mask].min().item()

        metrics["mean_dice"] = _masked_mean(dice_per_class)
        metrics["mean_iou"] = _masked_mean(iou_per_class)
        metrics["mean_recall"] = _masked_mean(rec_per_class)
        metrics["mean_precision"] = _masked_mean(pre_per_class)
        metrics["mean_vol_sim"] = _masked_mean(vs_per_class)
        metrics["mean_mcc"] = _masked_mean(mcc_per_class)
        metrics["min_class_dice"] = _masked_min(dice_per_class)
        metrics["min_class_iou"] = _masked_min(iou_per_class)

        smooth = 1e-5
        sd_msg = ""
        if self.compute_sd and self._sd_num is not None:
            sd_per_class = (self._sd_num + smooth) / (self._sd_denom + smooth)
            sd_per_class = sd_per_class.cpu()
            for c in range(len(sd_per_class)):
                metrics[f"surface_dice_class_{c}"] = sd_per_class[c].item()
            metrics["mean_surface_dice"] = _masked_mean(sd_per_class)
            metrics["mean_combined"] = (
                (1.0 - self.sd_w) * metrics["mean_dice"]
                + self.sd_w * metrics["mean_surface_dice"])
            sd_tag = (f"{self.sd_tol_mm:g}mm" if self.sd_physical
                      else f"{self.sd_tol}px")
            sd_msg = (
                f", pooled_mean_surface_dice@{sd_tag}="
                f"{metrics['mean_surface_dice']:.4f}, "
                f"per_class_sd={[f'{d:.4f}' for d in sd_per_class.tolist()]}, "
                f"combined(w={self.sd_w:.2f})={metrics['mean_combined']:.4f}")

        # Balanced：四指标调和均值；MCC∈[−1,1] 重映射到 [0,1]：(mcc+1)/2。
        if self._crit == "balanced" and "mean_surface_dice" in metrics:
            mcc01 = max(0.0, (metrics["mean_mcc"] + 1.0) * 0.5)
            hm = harmonic_mean_metrics([
                torch.tensor(metrics["mean_dice"]),
                torch.tensor(metrics["mean_surface_dice"]),
                torch.tensor(metrics["mean_iou"]),
                torch.tensor(mcc01)])
            metrics["mean_balanced"] = float(hm.item())

        cov = self._cov.cpu().tolist()
        if log:
            loss_str = (
                "N/A" if self.loss_meter.count == 0
                else f"{metrics['val_base_loss']:.4f}")
            logger.info(
                "  %s: loss=%s, pooled_mean_dice=%.4f, per_class=%s, "
                "iou=%.4f, recall=%.4f, precision=%.4f, vol_sim=%.4f, "
                "mcc=%.4f, min_class_dice=%.4f, coverage=%s/%d samples%s%s",
                log_prefix, loss_str, metrics["mean_dice"],
                [f"{d:.4f}" for d in dice_per_class.tolist()],
                metrics["mean_iou"], metrics["mean_recall"],
                metrics["mean_precision"], metrics["mean_vol_sim"],
                metrics["mean_mcc"], metrics["min_class_dice"],
                [int(c) for c in cov], self.n_samples, sd_msg,
                (f", balanced={metrics['mean_balanced']:.4f}"
                 if "mean_balanced" in metrics else ""))
        return metrics


# ---------------------------------------------------------------------------
# Validation evaluators (strategy objects)
# ---------------------------------------------------------------------------
class ValEvaluator(ABC):
    """选模评估策略基类。持有 ``Trainer`` 引用以读取模型 / pipeline / 配置 / 数据。

    EMA 权重换入由 ``Trainer._validate`` 在外层统一处理，evaluator 只负责
    产出 ``(pred, target)`` 并喂给 ``MetricAccumulator``。
    """

    log_prefix: str = "Val"

    def __init__(self, trainer: "Trainer"):
        self.trainer = trainer

    def _resolve_sd_spacing(self) -> Optional[list]:
        """物理 NSD 所需的每轴 mm spacing：仅当 surface_dice_tolerance_mm>0 时解析。
        取 data.target_spacing（未显式配置则从 npz_dir/_manifest.json 回读）；
        无法解析（如未开启 spacing_normalization）时返回 None 并告警一次，
        由 MetricAccumulator 回退 voxel-Chebyshev 容差。"""
        tc = self.trainer.cfg.train
        if float(tc.surface_dice_tolerance_mm) <= 0.0:
            return None
        dc = self.trainer.cfg.data
        ts = dc.target_spacing
        if ts is None:
            from ..predictor.predictor import _manifest_target_spacing
            ts = _manifest_target_spacing(dc.npz_dir)
        if ts is None:
            if not getattr(self, "_sd_spacing_warned", False):
                logger.warning(
                    "train.surface_dice_tolerance_mm=%.3f requires a resolved "
                    "voxel spacing, but data.target_spacing is unset and no "
                    "%s/_manifest.json target_spacing is available (enable "
                    "data.spacing_normalization when baking, or set "
                    "data.target_spacing). Falling back to voxel-Chebyshev "
                    "surface_dice_tolerance=%d.",
                    float(tc.surface_dice_tolerance_mm), dc.npz_dir,
                    int(tc.surface_dice_tolerance))
                self._sd_spacing_warned = True
            return None
        return [float(s) for s in ts]

    def _new_accumulator(self) -> MetricAccumulator:
        tc = self.trainer.cfg.train
        return MetricAccumulator(
            criterion=str(tc.save_best_criterion),
            surface_dice_tolerance=int(tc.surface_dice_tolerance),
            surface_dice_weight=float(tc.surface_dice_weight),
            threshold=self.trainer.cfg.predict.threshold,
            surface_dice_tolerance_mm=float(tc.surface_dice_tolerance_mm),
            spacing=self._resolve_sd_spacing())

    @abstractmethod
    def evaluate(self, epoch: int) -> Dict[str, float]:
        """运行一次完整验证，返回 metrics dict。"""


class PatchValEvaluator(ValEvaluator):
    """medium：遍历 ``val_loader`` 的随机 patch，逐 batch 前向取指标（既有行为）。"""

    log_prefix = "Val"

    @torch.no_grad()
    def evaluate(self, epoch: int) -> Dict[str, float]:
        t = self.trainer
        acc = self._new_accumulator()
        # 多卡切分在 DataLoader 采样器层完成（loader.ValBatchShardSampler 按
        # batch 块把 val 不相交切给各 rank，worker 只生产本 rank 的 batch），
        # 此处直接全量迭代本 rank 的 loader。
        for batch in t.val_loader:
            image = batch["image"].to(t.device, non_blocking=True)
            label = batch["label"].to(t.device, non_blocking=True).float()

            image, label = t.pipeline.prepare_val_batch(image, label)
            if t._memory_format is not None:
                image = image.to(memory_format=t._memory_format)
            with autocast(device_type="cuda", enabled=t.use_amp,
                          dtype=t.amp_dtype):
                pred = t.model(image)
                pred = t.pipeline.extract_main_pred(pred)
                pred_1x, target_1x = t.pipeline.split_for_metrics(pred, label)
            # val_base_loss 用裸 base_loss（无 DS/aux/topo 加权），与训练组合
            # 损失口径不同，两条曲线不可直接对比，仅作趋势监控。
            loss = compute_loss_fp32(t.base_loss, pred_1x, target_1x)
            loss_val = loss.item()
            if not math.isfinite(loss_val):
                logger.warning(
                    "Non-finite val loss (%s) at epoch %d; skipping "
                    "loss meter update.", loss_val, epoch + 1)
                loss_val = None
            acc.update(pred_1x, target_1x, loss_value=loss_val)
        acc.all_reduce(t.num_fg, t.device)
        return acc.compute(log_prefix=self.log_prefix, log=is_main_process())


class VolumeValEvaluator(ValEvaluator):
    """high：对每个 val 整卷复用 ``Predictor`` 做滑窗推理后取指标。

    整卷数据直接取自 npz 缓存（与 ``Predictor`` 同款预处理、bbox 已裁），无需磁盘
    NIfTI / bbox 处理。整卷 blended 概率按推理阈值二值化后以 ``pred_is_binary``
    直接喂入与 medium 完全相同的 ``MetricAccumulator``。不产 val_base_loss（见 ``MetricAccumulator``）。
    """

    log_prefix = "Val[full-3D]"

    @staticmethod
    @torch.no_grad()
    def _union_bbox_slices(
        fg: torch.Tensor, margin: int,
    ) -> "Optional[tuple]":
        """前景并集掩膜 ``fg``（空间维 bool）的外接 bbox，逐轴外扩 ``margin``
        并夹到边界。全背景时返回 ``None``（调用方不裁剪）。"""
        if not bool(fg.any()):
            return None
        slices = []
        for ax in range(fg.ndim):
            line = fg
            for d in range(fg.ndim - 1, -1, -1):
                if d != ax:
                    line = line.any(dim=d)
            idx = torch.nonzero(line).flatten()
            lo = max(int(idx[0].item()) - margin, 0)
            hi = min(int(idx[-1].item()) + margin + 1, fg.shape[ax])
            slices.append(slice(lo, hi))
        return tuple(slices)

    def __init__(self, trainer: "Trainer"):
        super().__init__(trainer)
        self._predictor = None  # 懒构建，复用同一 Predictor（引用 trainer.model）

    def _get_predictor(self):
        if self._predictor is None:
            # 延迟导入：medium 模式下不触发 predictor 包（及 SimpleITK）加载。
            from ..predictor import Predictor
            self._predictor = Predictor(
                self.trainer.model, self.trainer.cfg, self.trainer.device)
            # 整卷验证每 epoch 对全部 val 卷滑窗推理，逐卷滑窗进度日志在此关闭，
            # 避免刷屏（CLI 推理仍默认开启）。
            self._predictor.log_progress = False
        return self._predictor

    @torch.no_grad()
    def evaluate(self, epoch: int) -> Dict[str, float]:
        from ..data.dataset import (
            load_npz_image, load_npz_label, load_npz_z_spacing,
            preprocess_label)

        t = self.trainer
        dc = t.cfg.data
        npz_paths = list(getattr(t.val_loader.dataset, "_npz_paths", []))
        if not npz_paths:
            logger.warning(
                "%s: val dataset exposes no `_npz_paths`; cannot run "
                "full-volume validation. Falling back to empty metrics.",
                self.log_prefix)
            return {"val_base_loss": float("nan"), "mean_dice": 0.0}

        predictor = self._get_predictor()
        predictor.model.eval()
        acc = self._new_accumulator()
        label_values = list(dc.label_values)

        # 多卡：把去重后的整卷列表按 rank 不相交切分（每卷恰好一次，无重复计数）。
        for path in shard_for_rank(npz_paths):
            vol = load_npz_image(
                path, dc.intensity_min, dc.intensity_max, dc.normalize,
                dc.global_mean, dc.global_std)
            label = load_npz_label(path)
            # 物理 z spacing（npz meta，make_data≥1.4/1.6）：使 2.5D z-interleave
            # 因子选择与部署路径一致；旧 npz 无记录时为 None（回退标准滑窗）。
            z_spacing = (load_npz_z_spacing(path)
                         if predictor.z_interleave_enabled else None)
            # (num_fg, D, H, W) 概率体（已 sigmoid，跨窗 blended）。
            prob = predictor.predict_preprocessed_array(
                vol, z_spacing=z_spacing)
            target_np = preprocess_label(label, label_values)

            prob_t = torch.from_numpy(prob).to(t.device)
            target_t = torch.from_numpy(target_np).to(t.device).float()
            if prob_t.shape != target_t.shape:
                raise RuntimeError(
                    f"{self.log_prefix}: pred shape {tuple(prob_t.shape)} != "
                    f"target shape {tuple(target_t.shape)} for {path}. "
                    "Predictor output geometry must match the npz label.")

            # 按推理阈值二值化，以 pred_is_binary 直接喂累加算子（免饱和 logits
            # 往返与二次 sigmoid）。threshold 可为标量或逐前景类列表（按通道广播）。
            thr_t = torch.as_tensor(
                predictor.threshold, dtype=prob_t.dtype, device=prob_t.device)
            if thr_t.ndim == 1:
                thr_t = thr_t.view(-1, 1, 1, 1)
            pred_bin = (prob_t > thr_t).float()

            # 可选：按 pred∪GT 并集 bbox 裁剪后再算指标（严格等价，见 config）。
            # 边距取 surface_dice 容差+1，保证腔蚀/膨胀在裁剪窗内结果不变；
            # 总体素数按整卷回传，使 TN/MCC 口径不变。
            voxels_override = None
            if bool(t.cfg.train.val_metric_bbox_crop):
                if acc.compute_sd and acc.sd_physical:
                    # mm 容差 → 各轴最保守（最小 spacing）的 voxel 半径。
                    margin = int(math.ceil(
                        acc.sd_tol_mm / min(acc.sd_spacing))) + 1
                elif acc.compute_sd:
                    margin = acc.sd_tol + 1
                else:
                    margin = 1
                fg = (pred_bin > 0.5).any(dim=0) | (target_t > 0.5).any(dim=0)
                sl = self._union_bbox_slices(fg, margin)
                if sl is not None:
                    voxels_override = float(target_t.shape[1:].numel())
                    pred_bin = pred_bin[(slice(None),) + sl]
                    target_t = target_t[(slice(None),) + sl]

            acc.update(
                pred_bin.unsqueeze(0), target_t.unsqueeze(0),
                loss_value=None, voxels_override=voxels_override,
                pred_is_binary=True)
        acc.all_reduce(t.num_fg, t.device)
        return acc.compute(log_prefix=self.log_prefix, log=is_main_process())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_val_evaluator(trainer: "Trainer") -> ValEvaluator:
    """``cfg.train.val_metric_mode`` → ``ValEvaluator``。"""
    mode = str(trainer.cfg.train.val_metric_mode).lower().strip()
    if mode == "high":
        return VolumeValEvaluator(trainer)
    return PatchValEvaluator(trainer)


__all__ = [
    "MetricAccumulator",
    "ValEvaluator",
    "PatchValEvaluator",
    "VolumeValEvaluator",
    "build_val_evaluator",
]
