"""Predictor 包 I/O 子模块（R6）：checkpoint 加载、precision 选择、``run_inference`` 顶层入口。

从 ``segtask_v1.predictor.predictor`` 抽出，原 1412 行 God Module 的 ~140 行 ckpt + entry-point
代码搬到此处；``segtask_v1.predictor.__init__`` 仍 re-export 全部符号，外部 API 不变。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from taskcore.config.core import Config
from taskcore.models.mednext import reparameterize_model
from taskcore.models.topology import arch_fingerprint
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


# 结构键：不一致则模型的下采样计划/拓扑与训练时不同，strict load 也拦不住。
_FINGERPRINT_STRUCT_KEYS = (
    "spatial_dims", "n_levels", "stem_mode",
    "downsample_strides", "decoder_type")


def _check_arch_fingerprint(ckpt: Dict, cfg: Config, path: str) -> None:
    """比对 ckpt 内的结构指纹与当前配置，结构键不一致直接拒绝推理。

    旧 ckpt 无指纹时跳过（兼容）。``patch_size``/``patch_mode`` 仅诊断用，
    不参与硬比对（推理换 patch 尺寸在 stride 计划一致时是合法的）。"""
    fp_ckpt = ckpt.get("arch_fingerprint")
    if not isinstance(fp_ckpt, dict):
        return
    fp_now = arch_fingerprint(cfg)
    diffs = [
        f"{k}: ckpt={fp_ckpt[k]!r} vs current={fp_now[k]!r}"
        for k in _FINGERPRINT_STRUCT_KEYS
        if k in fp_ckpt and fp_ckpt[k] != fp_now[k]]
    if diffs:
        raise RuntimeError(
            f"Architecture fingerprint mismatch between checkpoint {path!r} "
            f"and current config: {'; '.join(diffs)}. "
            f"Training-time patch_size={fp_ckpt.get('patch_size')!r} "
            f"patch_mode={fp_ckpt.get('patch_mode')!r}; align "
            "data.patch_size / model.unet.downsample_strides with the "
            "training config (weights would load but produce wrong "
            "predictions under a drifted downsampling plan).")


# 预处理/几何镜像键（data.*）：训练与推理必须逐一相同，否则模型看到的输入
# 分布/几何与训练时不同 → 静默错误输出。patch_size 单列为软键（换 patch
# 尺寸在 stride 计划一致时合法，仅告警）。
_MIRROR_DATA_KEYS = (
    "normalize", "global_mean", "global_std",
    "intensity_min", "intensity_max",
    "spacing_normalization", "target_spacing",
    "patch_mode", "multi_res_scales",
    "keep_native_multi_res", "keep_native_view_depth",
    "z_boundary_mode", "label_values", "resize_antialias",
)
_MIRROR_SOFT_DATA_KEYS = ("patch_size",)


def _ckpt_data_dict(ckpt: Dict) -> Optional[Dict]:
    """从 ckpt 取训练时 data 段字段字典；无 config 时返 None。"""
    ckpt_cfg = ckpt.get("config")
    if isinstance(ckpt_cfg, Config):
        return dict(vars(ckpt_cfg.data))
    if isinstance(ckpt_cfg, dict) and isinstance(ckpt_cfg.get("data"), dict):
        return dict(ckpt_cfg["data"])
    return None


def _adopt_fingerprint_normalization(
    ckpt: Dict, cfg: Config, path: str) -> None:
    """normalize='ct_fingerprint'（2-3）推理侧参数采纳：训练时 loader 已把
    数据集指纹解析进 ckpt 内 config 的 intensity_min/max + global_mean/std，
    推理 YAML 无需（也不应）手抄这 4 个数值 —— 两侧都为 ct_fingerprint 时
    直接采纳 ckpt 解析值（随后的镜像硬比对自然通过）。"""
    if cfg.data.normalize != "ct_fingerprint":
        return
    ckpt_data = _ckpt_data_dict(ckpt)
    if ckpt_data is None:
        raise RuntimeError(
            f"data.normalize='ct_fingerprint' but checkpoint {path!r} "
            "carries no training config to adopt the resolved "
            "normalization parameters from. Use a checkpoint trained with "
            "this repo (>= fingerprint support), or switch to an explicit "
            "normalize mode with hand-set parameters.")
    if ckpt_data.get("normalize") != "ct_fingerprint":
        # 镜像硬比对会报 normalize 不一致，此处不采纳。
        return
    if "intensity_min" in ckpt_data:
        cfg.data.intensity_min = float(ckpt_data["intensity_min"])
    if "intensity_max" in ckpt_data:
        cfg.data.intensity_max = float(ckpt_data["intensity_max"])
    if "global_mean" in ckpt_data:
        cfg.data.global_mean = float(ckpt_data["global_mean"])
    if "global_std" in ckpt_data:
        cfg.data.global_std = float(ckpt_data["global_std"])
    logger.info(
        "normalize='ct_fingerprint': adopted resolved normalization from "
        "checkpoint training config -> clip=[%.2f, %.2f], mean=%.3f, "
        "std=%.3f.",
        cfg.data.intensity_min, cfg.data.intensity_max,
        cfg.data.global_mean, cfg.data.global_std)


def _check_preprocess_mirror(ckpt: Dict, cfg: Config, path: str) -> None:
    """比对 ckpt 内保存的训练配置与当前推理配置的预处理/几何镜像键。

    硬键不一致默认直接拒绝推理（``predict.allow_preprocess_mismatch=True``
    降级为告警）；软键（patch_size）仅告警。旧 ckpt 无 ``config`` 时告警
    跳过（兼容）。"""
    ckpt_cfg = ckpt.get("config")
    # 兼容两种存法：pickled Config 对象（本仓）或普通 dict（外部导出）。
    # vars() 取实例字段字典：旧版 pickled Config 自然缺新字段，下方按缺键跳过。
    if isinstance(ckpt_cfg, Config):
        ckpt_data = dict(vars(ckpt_cfg.data))
    elif isinstance(ckpt_cfg, dict) and isinstance(
            ckpt_cfg.get("data"), dict):
        ckpt_data = ckpt_cfg["data"]
    else:
        logger.warning(
            "Checkpoint %r carries no training config; preprocessing "
            "mirror check skipped (ensure the inference YAML matches the "
            "training preprocessing manually).", path)
        return
    cur_data = dict(vars(cfg.data))

    def _norm(val):
        return list(val) if isinstance(val, (tuple, list)) else val

    def _diffs(keys) -> List[str]:
        # 旧 ckpt 的 Config 版本可能缺新字段：缺键跳过（无从比对）。
        return [
            f"data.{k}: ckpt={_norm(ckpt_data[k])!r} vs "
            f"current={_norm(cur_data[k])!r}"
            for k in keys
            if k in ckpt_data and k in cur_data
            and _norm(ckpt_data[k]) != _norm(cur_data[k])]

    hard_diffs = _diffs(_MIRROR_DATA_KEYS)
    soft_diffs = _diffs(_MIRROR_SOFT_DATA_KEYS)

    if soft_diffs:
        logger.warning(
            "Inference config differs from training config on soft mirror "
            "key(s): %s. Legal when the downsampling plan is unchanged, "
            "but verify this is intentional.", "; ".join(soft_diffs))
    if hard_diffs:
        msg = (
            f"Preprocessing mirror mismatch between checkpoint {path!r} "
            f"training config and current inference config: "
            f"{'; '.join(hard_diffs)}. The model would receive inputs "
            "with a different distribution/geometry than it was trained "
            "on (silently wrong predictions). Align the inference YAML "
            "with the training config, or set "
            "predict.allow_preprocess_mismatch=true to downgrade this "
            "error to a warning.")
        if cfg.predict.allow_preprocess_mismatch:
            logger.warning("%s (allowed by allow_preprocess_mismatch)", msg)
        else:
            raise RuntimeError(msg)


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


def _unique_output_stems(image_paths: List[str]) -> Dict[str, str]:
    """逐图像输出 stem：basename 唯一时直接用；递归输入下同名文件自动
    前缀父目录名（仍冲突则逐级上溯），避免 ``*_pred.nii.gz`` 互覆。"""
    def _base(p: str) -> str:
        return Path(p).name.replace(".nii.gz", "").replace(".nii", "")

    stems: Dict[str, str] = {}
    depth = 0
    remaining = list(image_paths)
    while remaining and depth < 16:
        candidates: Dict[str, str] = {}
        for p in remaining:
            parts = Path(p).parent.parts[::-1][:depth]
            candidates[p] = "__".join(list(parts[::-1]) + [_base(p)])
        counts: Dict[str, int] = {}
        for s in candidates.values():
            counts[s] = counts.get(s, 0) + 1
        next_remaining: List[str] = []
        for p, s in candidates.items():
            if counts[s] == 1:
                stems[p] = s
            else:
                next_remaining.append(p)
        remaining = next_remaining
        depth += 1
    for i, p in enumerate(remaining):  # 完全同路径重复（理论上不会）兜底
        stems[p] = f"{_base(p)}__{i}"
    renamed = {p: s for p, s in stems.items() if s != _base(p)}
    if renamed:
        logger.warning(
            "Duplicate output basenames detected under recursive input; "
            "auto-prefixing %d output(s) with their sub-directory to avoid "
            "overwrites: %s", len(renamed),
            list(renamed.values())[:8])
    return stems


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
) -> int:
    """对一组图像运行推理。返回失败卷数（CLI 据此返非零退出码）。

    * ``weight_variant``：``'auto'`` (优 EMA) | ``'ema'`` | ``'online'``
    * ``bbox_paths``：与 ``image_paths`` 1∶1 的 ROI 掩膜；``None`` 走全卷
    * ``precision``：``auto`` / ``fp32`` / ``bf16`` / ``fp16``（``fp16`` 仅兼容，ConvNeXt LayerNorm 可能 NaN）
    """
    if bbox_paths is not None and len(bbox_paths) != len(image_paths):
        raise ValueError(
            f"bbox_paths length {len(bbox_paths)} != image_paths "
            f"length {len(image_paths)}")
    from taskcore.models.factory import build_model

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
    _check_arch_fingerprint(ckpt, cfg, checkpoint_path)
    _adopt_fingerprint_normalization(ckpt, cfg, checkpoint_path)
    _check_preprocess_mirror(ckpt, cfg, checkpoint_path)
    # 打包口径标识（1-3）：训练时 loader 从 npz manifest 回写，供事后追溯。
    ckpt_data = _ckpt_data_dict(ckpt)
    if ckpt_data and ckpt_data.get("data_identifier"):
        logger.info("Training-time npz data_identifier: %s",
                    ckpt_data["data_identifier"])
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

    if cfg.train.reparam_deploy:
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
    # 阈值标定（3-7）：ckpt 含训练期标定的逐类阈值时默认消费（可配置关闭）。
    predictor.apply_calibrated_thresholds(ckpt)

    # 测试时自适应 BatchNorm — global 模式：推理前用少量目标域整卷重估 BN running
    # stats，全程复用。per_volume 模式不在此处理（见 Predictor.predict_volume）。
    if cfg.predict.adabn_enabled and \
            cfg.predict.adabn_mode == "global":
        from taskcore.engine.bn_stats import collect_bn_modules, estimate_bn_stats

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
    output_stems = _unique_output_stems(image_paths)
    n_failed = 0
    for i, path in enumerate(image_paths, 1):
        bbox_path = bbox_paths[i - 1] if bbox_paths is not None else None
        logger.info("[%d/%d] Processing: %s%s", i, n, path,
                    f" (bbox={bbox_path})" if bbox_path else "")
        try:
            result = predictor.predict_volume(
                path, output_dir=cfg.predict.output_dir,
                bbox_path=bbox_path,
                output_stem=output_stems[path])
            logger.info("  Label map shape: %s, unique labels: %s",
                        result["label_map"].shape,
                        np.unique(result["label_map"]).tolist())
        except Exception as e:
            n_failed += 1
            logger.exception("Failed to process %s: %s", path, e)
            continue
    if n_failed:
        logger.error("Inference finished with %d/%d failed volume(s).",
                     n_failed, n)
    return n_failed


__all__ = [
    "run_inference",
    "_strip_compile_prefix",
    "_unwrap_ema_state",
    "_select_state_dict",
    "_resolve_inference_precision",
    "_PRECISION_CHOICES",
]
