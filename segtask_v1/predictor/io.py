"""Predictor 包 I/O 子模块（R6）：checkpoint 加载、precision 选择、``run_inference`` 顶层入口。

从 ``segtask_v1.predictor.predictor`` 抽出，原 1412 行 God Module 的 ~140 行 ckpt + entry-point
代码搬到此处；``segtask_v1.predictor.__init__`` 仍 re-export 全部符号，外部 API 不变。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..config import Config
from ..models.mednext import reparameterize_model
from .predictor import Predictor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def _strip_compile_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """剥去 torch.compile 添加的 ``_orig_mod.`` 前缀。"""
    prefix = "_orig_mod."
    if any(k.startswith(prefix) for k in sd):
        return {(k[len(prefix):] if k.startswith(prefix) else k): v
                for k, v in sd.items()}
    return sd


def _unwrap_ema_state(ema_sd: Dict) -> Dict[str, torch.Tensor]:
    """将 ``{shadow, decay}`` 拆为普通 state_dict；已是拆过的旧格式原返。"""
    if isinstance(ema_sd, dict) and "shadow" in ema_sd and isinstance(
            ema_sd["shadow"], dict):
        return ema_sd["shadow"]
    return ema_sd


def _select_state_dict(
    ckpt: Dict, variant: str,
) -> Tuple[Dict[str, torch.Tensor], str]:
    """从 ckpt 选权重。``variant``: ``'auto'`` (优 EMA) / ``'ema'`` / ``'online'``。

    返 ``(state_dict, label)``，``label`` 用于日志。
    """
    has_online = "model_online_state_dict" in ckpt
    has_ema = "ema_state_dict" in ckpt
    primary = ckpt["model_state_dict"]

    if variant == "online":
        return (ckpt["model_online_state_dict"] if has_online else primary,
                "online")
    if variant == "ema":
        if has_ema:
            return _unwrap_ema_state(ckpt["ema_state_dict"]), "ema"
        logger.warning("EMA requested but not found in checkpoint; "
                       "using online weights.")
        return (ckpt["model_online_state_dict"] if has_online else primary,
                "online")
    # auto
    if has_ema:
        return _unwrap_ema_state(ckpt["ema_state_dict"]), "ema"
    return primary, "online"


# ---------------------------------------------------------------------------
# Precision resolution
# ---------------------------------------------------------------------------
_PRECISION_CHOICES = ("auto", "fp32", "bf16", "fp16")


def _resolve_inference_precision(precision: str, cfg: Config) -> str:
    """选推理 dtype。

    * ``auto`` 跟随 ``cfg.train.amp_dtype``：``{bf16}→bf16``, ``{fp16}→fp16``, 其余退 ``bf16``。
    * ``fp16`` 下 ConvNeXt LayerNorm 可能 NaN，仅 opt-in。
    """
    p = precision.lower()
    if p not in _PRECISION_CHOICES:
        raise ValueError(
            f"precision={precision!r} not in {_PRECISION_CHOICES}")
    if p != "auto":
        return p
    amp = (cfg.train.amp_dtype or "bfloat16").lower()
    if amp in ("float16", "fp16"):
        return "fp16"
    return "bf16"


# ---------------------------------------------------------------------------
# Top-level entry: build model + load ckpt + iterate images
# ---------------------------------------------------------------------------
def run_inference(
    cfg: Config,
    checkpoint_path: str,
    image_paths: List[str],
    weight_variant: str = "auto",
    bbox_paths: Optional[List[str]] = None,
    precision: str = "auto",
) -> None:
    """对一组图像运行推理。

    * ``weight_variant``：``'auto'`` (优 EMA) | ``'ema'`` | ``'online'``
    * ``bbox_paths``：与 ``image_paths`` 1∶1 的 ROI 掩膜；``None`` 走全卷
    * ``precision``：``auto`` / ``fp32`` / ``bf16`` / ``fp16``（``fp16`` 仅兼容，ConvNeXt LayerNorm 可能 NaN）
    """
    if bbox_paths is not None and len(bbox_paths) != len(image_paths):
        raise ValueError(
            f"bbox_paths length {len(bbox_paths)} != image_paths "
            f"length {len(image_paths)}")
    from ..models.factory import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 独立推理入口不经过 utils.seed_everything，默认跑在 cudnn.benchmark=False。
    # 滑窗窗口形状固定，opt-in 开启 autotune（见 config.PredictConfig.cudnn_benchmark）。
    if cfg.predict.cudnn_benchmark and device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        logger.info("cudnn.benchmark enabled for inference "
                    "(predict.cudnn_benchmark=True).")

    model = build_model(cfg)
    # weights_only=False：本 trainer ckpt 含 Config / numpy RNG，PyTorch 2.6+ 默认安全模式会拒。
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd, label = _select_state_dict(ckpt, weight_variant)
    sd = _strip_compile_prefix(sd)

    # 形状预校验：共有键形状不一致（典型：num_classes/label_values 与训练时
    # 不同导致输出 head 尺寸变化）时给出明确错误，而非底层 size-mismatch 堆栈。
    model_sd = model.state_dict()
    shape_mismatch = [
        f"{k}: ckpt{tuple(v.shape)} vs model{tuple(model_sd[k].shape)}"
        for k, v in sd.items()
        if k in model_sd and tuple(model_sd[k].shape) != tuple(v.shape)]
    if shape_mismatch:
        raise RuntimeError(
            f"Checkpoint/model shape mismatch for {len(shape_mismatch)} "
            f"key(s) (first 8): {shape_mismatch[:8]}. If the mismatch is in "
            "seg/aux heads, cfg.data.label_values / num_classes likely "
            "differ from the training config.")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning("Missing keys when loading checkpoint: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading checkpoint: %s", unexpected)
    # 加载 < 半参数时硬错：避免随机初始权重静默推理。
    n_total = len(model.state_dict())
    n_loaded = n_total - len(missing)
    if n_total > 0 and n_loaded < max(1, n_total // 2):
        raise RuntimeError(
            f"Only {n_loaded}/{n_total} parameters loaded from "
            f"{checkpoint_path} (variant={label}). The checkpoint key "
            f"layout does not match the model — refusing to predict with "
            f"random weights. Unexpected keys: {unexpected[:8]}")

    if cfg.model.reparam_deploy:
        logger.info(
            "Applying MedNeXt deploy reparameterization before device transfer.")
        model = reparameterize_model(model)

    model = model.to(device).eval()
    # 推理精度：auto 跟随 train.amp_dtype（默认 bf16 autocast，fp32 权重）；
    # fp16 仅 opt-in（LayerNorm 会 NaN，见 _resolve_inference_precision）。
    resolved_precision = _resolve_inference_precision(precision, cfg)
    if device.type == "cuda" and resolved_precision == "fp16":
        model = model.half()
        logger.info(
            "Inference precision: fp16 (model cast to float16; device=%s). "
            "WARNING: fp16 + LayerNorm backbones can overflow on near-"
            "constant patches and produce NaN — prefer 'bf16' or 'fp32' "
            "if you see NaN-driven 'all foreground' predictions.",
            device)
    else:
        logger.info(
            "Inference precision: %s (model in fp32; autocast=%s; device=%s).",
            resolved_precision,
            resolved_precision if resolved_precision in ("bf16", "fp16") else "off",
            device)
    logger.info("Model loaded from %s (variant=%s)", checkpoint_path, label)

    predictor = Predictor(model, cfg, device)

    # 测试时自适应 BatchNorm — global 模式：推理前用少量目标域整卷重估 BN running
    # stats，全程复用。per_volume 模式不在此处理（见 Predictor.predict_volume）。
    if cfg.predict.adabn_enabled and \
            cfg.predict.adabn_mode == "global":
        from .adabn import collect_bn_modules, estimate_bn_stats

        bn_modules = collect_bn_modules(model)
        if not bn_modules:
            logger.warning(
                "[AdaBN] global enabled but model has no BatchNorm layers "
                "(norm_type != 'batch'); skipping — predictions unchanged.")
        else:
            n_warm = min(int(cfg.predict.adabn_num_volumes), len(image_paths))
            warm_paths = image_paths[:n_warm]
            if resolved_precision == "fp16":
                logger.warning(
                    "[AdaBN] running BN re-estimation under fp16 weights; "
                    "stat accumulation precision is reduced — prefer "
                    "'bf16'/'fp32' if results look unstable.")
            logger.info(
                "[AdaBN] global: re-estimating BN stats over %d BatchNorm "
                "layer(s) from %d target volume(s) before inference ...",
                len(bn_modules), n_warm)

            def _warmup() -> None:
                # 估计期强制 TTA 串行（同 per_volume 路径，见 Predictor._adabn_estimating
                # 注释）：BN 处于 train+累积平均，flip 变体拼大 batch 会让 running
                # stats 依赖 tta_batch_size。
                predictor._adabn_estimating = True
                try:
                    for j, wp in enumerate(warm_paths, 1):
                        logger.info("[AdaBN] warmup [%d/%d]: %s", j, n_warm, wp)
                        try:
                            # 整卷预热（不裁 bbox / 不落盘），仅为驱动前向更新 BN 统计。
                            predictor.predict_volume(wp, output_dir=None,
                                                      bbox_path=None)
                        except Exception as e:  # 单卷失败不应中断整体预热。
                            logger.warning(
                                "[AdaBN] warmup failed on %s: %s", wp, e)
                finally:
                    predictor._adabn_estimating = False

            estimate_bn_stats(bn_modules, _warmup)
            logger.info(
                "[AdaBN] global: BN running stats updated from target domain.")

    n = len(image_paths)
    for i, path in enumerate(image_paths, 1):
        bbox_path = bbox_paths[i - 1] if bbox_paths is not None else None
        logger.info("[%d/%d] Processing: %s%s", i, n, path,
                    f" (bbox={bbox_path})" if bbox_path else "")
        try:
            result = predictor.predict_volume(
                path, output_dir=cfg.predict.output_dir,
                bbox_path=bbox_path)
            logger.info("  Label map shape: %s, unique labels: %s",
                        result["label_map"].shape,
                        np.unique(result["label_map"]).tolist())
        except Exception as e:
            logger.exception("Failed to process %s: %s", path, e)
            continue


__all__ = [
    "run_inference",
    "_strip_compile_prefix",
    "_unwrap_ema_state",
    "_select_state_dict",
    "_resolve_inference_precision",
    "_PRECISION_CHOICES",
]
