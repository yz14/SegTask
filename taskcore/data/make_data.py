"""逐样本 npz 预烘包（image+label+可选 rw+fg 索引）。将 bbox 裁剪后的体积输出为 <out_dir>/<pid>.npz，训练时 mmap 多 worker 共享 OS page cache。默认 np.savez 不压缩（供共享）；--compress 使用 savez_compressed。CLI：`python -m segtask_v1.data.make_data --config <yaml> --out <dir> [--workers N]`。已存在不覆盖（除非 --overwrite）；失败写 <out_dir>/_failures.txt（与 data.exclude_list 兼容）。"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import Config, load_config
from .dataset import (
    BBox,
    compute_bbox_from_volume,
    load_nifti,
    load_nifti_cropped,
    load_region_weight_volume,
    read_nifti_geometry,
    read_nifti_spacing,
    resample_to_spacing,
)
from .loader import (
    _filter_by_exclude,
    _load_exclude_pids,
    discover_samples,
    match_bbox_paths,
    match_region_weight_paths,
    detect_label_values,
)

logger = logging.getLogger(__name__)


_TOOL_VERSION = "make_data/1.8"  # 1.8: spacing_zyx=stored spacing; skip checks label_values/fg_subsample

# skip 幂等时要求 on-disk meta 至少具备这些键（旧包缺键则触发重生成）。
_REQUIRED_SKIP_META_KEYS = (
    "label_counts",
    "image_shape",
    "fg_per_class",
)

# 同 _GEOM_SPACING_ATOL：target_spacing 比对容差 (mm)。
_META_SPACING_ATOL = 1e-3

def _read_npz_meta(path: Path) -> Optional[dict]:
    """只读 npz 内 ``meta`` 对象（不解码 image/label），供 skip 前快速比对。"""
    try:
        with np.load(path, allow_pickle=True) as zf:
            if "meta" not in zf.files:
                return None
            raw = zf["meta"]
            return dict(raw.item()) if hasattr(raw, "item") else dict(raw)
    except Exception as exc:
        logger.warning("Cannot read meta from %s: %s", path, exc)
        return None


def _npz_meta_allows_skip(
    meta: Optional[dict],
    *,
    spacing_normalization: bool,
    target_spacing: Optional[List[float]],
    label_values: Optional[List[int]] = None,
    fg_subsample: Optional[int] = None,
    expect_rw: bool = False,
    expect_bbox: bool = False,
) -> tuple[bool, str]:
    """判断既有 npz 是否可与当前打包参数幂等 skip。"""
    if meta is None:
        return False, "missing or unreadable meta"
    for key in _REQUIRED_SKIP_META_KEYS:
        if key not in meta:
            return False, f"missing meta key {key!r} (stale package)"
    on_disk_norm = bool(meta.get("spacing_normalized", False))
    if on_disk_norm != bool(spacing_normalization):
        return False, (
            f"spacing_normalized mismatch: on-disk={on_disk_norm} "
            f"!= requested={spacing_normalization}")
    if spacing_normalization:
        on_disk_ts = meta.get("target_spacing")
        req_ts = ([float(s) for s in target_spacing]
                  if target_spacing is not None else None)
        if on_disk_ts is None and req_ts is None:
            pass
        elif on_disk_ts is None or req_ts is None:
            return False, (
                f"target_spacing mismatch: on-disk={on_disk_ts} "
                f"!= requested={req_ts}")
        elif (len(on_disk_ts) != len(req_ts)
              or not np.allclose(on_disk_ts, req_ts,
                                 rtol=0.0, atol=_META_SPACING_ATOL)):
            return False, (
                f"target_spacing mismatch: on-disk={on_disk_ts} "
                f"!= requested={req_ts}")
    if label_values is not None:
        on_disk_lv = meta.get("label_values")
        req_lv = [int(v) for v in label_values]
        if on_disk_lv is None or [int(v) for v in on_disk_lv] != req_lv:
            return False, (
                f"label_values mismatch: on-disk={on_disk_lv} "
                f"!= requested={req_lv}")
    if fg_subsample is not None and "fg_subsample" in meta:
        on_disk_fg = int(meta["fg_subsample"])
        if on_disk_fg != int(fg_subsample):
            return False, (
                f"fg_subsample mismatch: on-disk={on_disk_fg} "
                f"!= requested={int(fg_subsample)}")
    # 与 has_cond 同构：事后打开 region_weight_dir / bbox_dir 时不得静默
    # 复用无 rw / 未裁剪的旧包。
    if expect_rw and not bool(meta.get("has_rw")):
        return False, "missing rw (stale package; region_weight_dir now set)"
    if expect_bbox and not bool(meta.get("src_bbox")):
        return False, "missing bbox (stale package; bbox_dir now set)"
    return True, ""


_DEFAULT_FG_SUBSAMPLE = 50_000


def _stem(path: str, suffix) -> str:
    """返回去后缀的文件名；suffix 可为 str 或候选列表。"""
    name = Path(path).name
    suffixes = [suffix] if isinstance(suffix, str) else list(suffix)
    for sfx in suffixes:
        if sfx and name.endswith(sfx):
            return name[: -len(sfx)]
    return Path(name).stem


def _compute_fg_indices(
    label: np.ndarray,
    label_values: List[int],
    fg_subsample: int,
    seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """逐前景类计算 fg 索引（类均衡采样用，seed=42）。

    返 (fg_slices, fg_coords, fg_coords_cls, fg_slices_cls_z, fg_slices_cls)：
    - fg_slices    (M,)  全前景 z 切片索引（各类并集，向后兼容）；
    - fg_coords    (N,3) 逐类 argwhere 后拼接；每类独立 cap 到 fg_subsample，
      避免稀有小结构被大器官淹没；
    - fg_coords_cls   (N,) 与 fg_coords 逐行对齐的类值；
    - fg_slices_cls_z (K,) 逐类 z 切片索引拼接；
    - fg_slices_cls   (K,) 与之逐行对齐的类值。"""
    empty = (np.zeros((0,), dtype=np.int32),
             np.zeros((0, 3), dtype=np.int32),
             np.zeros((0,), dtype=np.int16),
             np.zeros((0,), dtype=np.int32),
             np.zeros((0,), dtype=np.int16))
    if label.size == 0:
        return empty
    label_int = label.astype(np.int32, copy=False)
    rng = np.random.RandomState(seed)

    coords_parts    : List[np.ndarray] = []
    coords_cls_parts: List[np.ndarray] = []
    slices_parts    : List[np.ndarray] = []
    slices_cls_parts: List[np.ndarray] = []
    for v in label_values[1:]:
        cls_mask = label_int == int(v)
        if not cls_mask.any():
            continue
        z = np.where(np.any(cls_mask, axis=(1, 2)))[0].astype(np.int32)
        c = np.argwhere(cls_mask).astype(np.int32)
        if fg_subsample > 0 and len(c) > fg_subsample:
            idx = rng.choice(len(c), fg_subsample, replace=False)
            c = c[idx]
        slices_parts.append(z)
        slices_cls_parts.append(np.full(len(z), int(v), dtype=np.int16))
        coords_parts.append(c)
        coords_cls_parts.append(np.full(len(c), int(v), dtype=np.int16))

    if not coords_parts:
        return empty
    fg_coords       = np.concatenate(coords_parts, axis=0)
    fg_coords_cls   = np.concatenate(coords_cls_parts, axis=0)
    fg_slices_cls_z = np.concatenate(slices_parts, axis=0)
    fg_slices_cls   = np.concatenate(slices_cls_parts, axis=0)
    fg_slices       = np.unique(fg_slices_cls_z).astype(np.int32)
    return fg_slices, fg_coords, fg_coords_cls, fg_slices_cls_z, fg_slices_cls


# 物理几何一致性容差：spacing/origin 为 mm（头存 float32，允许亚 mm 量化误差），
# direction 为方向余弦。
_GEOM_SPACING_ATOL   = 1e-3
_GEOM_ORIGIN_ATOL    = 1e-2
_GEOM_DIRECTION_ATOL = 1e-3


def check_physical_geometry(
    pid: str,
    image_path: str,
    others: List[Tuple[str, Optional[str]]]) -> None:
    """校验 label/bbox/rw 与 image 共物理坐标系（spacing/origin/direction，
    只读头不解码像素）；不一致说明未严格共注册，voxel 索引对齐无意义，
    直接 fail-fast（与 shape 校验同级）。"""
    ref = read_nifti_geometry(image_path)
    names = ("spacing", "origin", "direction")
    atols = (_GEOM_SPACING_ATOL, _GEOM_ORIGIN_ATOL, _GEOM_DIRECTION_ATOL)
    for role, path in others:
        if not path:
            continue
        got = read_nifti_geometry(path)
        for name, atol, a, b in zip(names, atols, ref, got):
            if len(a) != len(b) or not np.allclose(a, b, rtol=0.0, atol=atol):
                raise ValueError(
                    f"pid={pid}: {role} volume is not in the same physical "
                    f"space as the image ({name} mismatch: image={a} vs "
                    f"{role}={b}, atol={atol}). Voxel-wise pairing would be "
                    f"geometrically wrong; resample/co-register {role} onto "
                    f"the image grid first (paths: image={image_path}, "
                    f"{role}={path}).")


_check_physical_geometry = check_physical_geometry  # 旧名别名，兼容存量引用


def _bbox_from_mask_path(bbox_path: Optional[str]) -> Optional[BBox]:
    """读 mask→compute_bbox_from_volume；路径为空或 mask 空返 None。"""
    if not bbox_path:
        return None
    mask = load_nifti(bbox_path, dtype=np.int16)
    return compute_bbox_from_volume(mask)


def prepare_one(
    pid: str,
    image_path: str,
    label_path: str,
    bbox_path: Optional[str],
    rw_path: Optional[str],
    out_path: str,
    label_values: List[int],
    fg_subsample: int = _DEFAULT_FG_SUBSAMPLE,
    compress: bool = False,
    overwrite: bool = False,
    spacing_normalization: bool = False,
    target_spacing: Optional[List[float]] = None,
    cond_paths: Optional[List[str]] = None) -> Dict[str, object]:
    """为单样本生成 npz 包；默认幂等（除非 overwrite）。返状态 dict (pid/status/size/耗时/...)。
    spacing_normalization=True 且 target_spacing 非空时把 image/label/rw/cond 从原生 spacing
    重采样到 target_spacing（numpy 轴序 (D,H,W) mm）后再落盘。
    ``cond_paths`` 为可选条件体路径列表（生成任务）；与 image 同 bbox 裁剪后按 channel stack。
    """
    out_p = Path(out_path)
    expect_cond = bool(cond_paths)
    expect_rw = bool(rw_path)
    expect_bbox = bool(bbox_path)
    if out_p.is_file() and not overwrite:
        meta = _read_npz_meta(out_p)
        ok, reason = _npz_meta_allows_skip(
            meta,
            spacing_normalization=spacing_normalization,
            target_spacing=target_spacing,
            label_values=label_values,
            fg_subsample=fg_subsample,
            expect_rw=expect_rw,
            expect_bbox=expect_bbox)
        if ok and expect_cond and not bool((meta or {}).get("has_cond")):
            ok, reason = False, "missing cond (stale package)"
        if ok:
            return {"pid": pid, "status": "skipped",
                    "size_bytes": out_p.stat().st_size, "elapsed_s": 0.0}
        logger.warning(
            "pid=%s: existing npz at %s does not match current packing "
            "parameters (%s); regenerating.",
            pid, out_p, reason)

    t0 = time.perf_counter()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # 0. 物理几何校验：label/bbox/rw/cond 必须与 image 共 spacing/origin/direction
    #    （shape 相等不蕴含共坐标系；只读头，成本可忽略）。
    _check_physical_geometry(
        pid, image_path,
        [("label", label_path), ("bbox", bbox_path), ("rw", rw_path)]
        + [(f"cond{i}", c) for i, c in enumerate(cond_paths or [])])

    # 1. mask → bbox（无/空为 None）。
    bbox = _bbox_from_mask_path(bbox_path)

    # 2-3. 读 image (raw HU int16) 与 label (int16)，按 bbox 裁剪。
    image = load_nifti_cropped(image_path, bbox=bbox, dtype=np.int16)
    label = load_nifti_cropped(label_path, bbox=bbox, dtype=np.int16)

    if image.shape != label.shape:
        raise ValueError(
            f"image shape {image.shape} != label shape {label.shape} for "
            f"pid={pid} (image={image_path}, label={label_path})")

    # 4. 区域权重（+1 偏移）：整数且 int16 范围内存 int16，否则 fp32（runtime 始终返 fp32）。
    rw: Optional[np.ndarray] = None
    rw_dtype_stored = None
    if rw_path:
        rw = load_region_weight_volume(rw_path, bbox=bbox)
        if rw.shape != image.shape:
            raise ValueError(
                f"region_weight shape {rw.shape} != image shape {image.shape} "
                f"for pid={pid} (rw={rw_path})")
        rw_min = float(rw.min())
        rw_max = float(rw.max())
        is_integer_valued = np.all(rw == np.round(rw))
        fits_int16 = (rw_min >= np.iinfo(np.int16).min
                      and rw_max <= np.iinfo(np.int16).max)
        if is_integer_valued and fits_int16:
            rw = rw.astype(np.int16, copy=False)
            rw_dtype_stored = "int16"
        else:
            rw_dtype_stored = "float32"
            logger.warning(
                "pid=%s rw has non-integer or out-of-int16 values "
                "(min=%.3f, max=%.3f, integer_valued=%s) — storing as float32.",
                pid, rw_min, rw_max, is_integer_valued)

    # 4b. 条件体：与 image 1:1 对齐、同 bbox 裁剪；按 channel stack。
    cond: Optional[np.ndarray] = None
    if cond_paths:
        cond_vols: List[np.ndarray] = []
        for cpath in cond_paths:
            cvol = load_nifti_cropped(cpath, bbox=bbox, dtype=np.float32)
            if cvol.shape != image.shape:
                raise ValueError(
                    f"cond shape {cvol.shape} != image shape {image.shape} "
                    f"for pid={pid} (cond={cpath})")
            cond_vols.append(cvol)
        cond = np.stack(cond_vols, axis=0).astype(np.float32, copy=False)

    # 4.5 物理 spacing 归一化（可选）：把 bbox-裁剪后的体积重采样到 target_spacing。
    #     在 fg 索引/shape 记录之前做，保证下游全部落在归一化坐标系。
    #     orig_spacing 无论归一化与否都记录（make_data≥1.6），使未归一化 npz 也
    #     携带物理 z spacing（整卷验证的 z-interleave 因子选择与部署一致）。
    src = read_nifti_spacing(image_path)  # (sz, sy, sx) mm
    _, origin, direction = read_nifti_geometry(image_path)
    pre_resample_shape = [int(s) for s in image.shape]
    orig_spacing: Optional[List[float]] = [float(s) for s in src]
    spacing_normalized = False
    if spacing_normalization and target_spacing is not None:
        tgt = tuple(float(s) for s in target_spacing)
        image = resample_to_spacing(image, src, tgt, is_label=False)
        label = resample_to_spacing(label, src, tgt, is_label=True)
        if rw is not None:
            # rw 为离散权重，用近邻保值。
            rw = resample_to_spacing(rw, src, tgt, is_label=True)
        if cond is not None:
            # cond 按通道独立重采样（连续值，用线性插值）。
            cond = np.stack(
                [resample_to_spacing(cond[c], src, tgt, is_label=False)
                 for c in range(cond.shape[0])],
                axis=0).astype(np.float32, copy=False)
        spacing_normalized = True
    achieved_spacing = [
        float(src[i]) * float(pre_resample_shape[i]) / float(image.shape[i])
        for i in range(3)]

    # 5. 裁剪坐标系下的前景索引（逐类，供类均衡前景采样）。
    (fg_slices, fg_coords, fg_coords_cls,
     fg_slices_cls_z, fg_slices_cls) = _compute_fg_indices(
        label, label_values, fg_subsample)

    # 5.5 逐值精确体素计数（落盘 label 同坐标系）：供 loader 直接从 meta 读取
    #     label_values / 分层划分统计，免去启动期全量解码 label。
    uniq_vals, uniq_counts = np.unique(
        label.astype(np.int32, copy=False), return_counts=True)
    label_counts = {int(v): int(c) for v, c in zip(uniq_vals, uniq_counts)}

    # 6. 谱系 meta（自描述）。
    meta = {
        "pid"         : pid,
        "src_image"   : str(image_path),
        "src_label"   : str(label_path),
        "src_bbox"    : str(bbox_path) if bbox_path else "",
        "src_rw"      : str(rw_path) if rw_path else "",
        "src_cond"    : list(map(str, cond_paths)) if cond_paths else [],
        "bbox"        : (list(map(list, bbox)) if bbox is not None else None),
        "label_values": list(map(int, label_values)),
        "has_rw"      : rw is not None,
        "has_cond"    : cond is not None,
        "rw_shift"    : 1.0,
        "rw_dtype"    : rw_dtype_stored,    # int16 / float32 / None
        "image_dtype" : str(image.dtype),
        "image_shape" : [int(s) for s in image.shape],  # (D,H,W)，供 dataset 免解码读形状（make_data≥1.4）
        "fg_per_class": True,   # fg_coords 逐类 cap；含 *_cls 类对齐数组
        "label_counts": label_counts,  # {label_value: voxel_count}，精确不抑采（make_data≥1.3）
        "spacing_normalized": spacing_normalized,
        "orig_spacing" : orig_spacing,   # 原生头 spacing [sz,sy,sx] mm
        "pre_resample_shape": pre_resample_shape,
        "achieved_spacing": achieved_spacing,
        "origin"      : [float(v) for v in origin],
        "direction"   : [float(v) for v in direction],
        # 落盘体素物理 spacing：归一化后为 target，否则等于 orig。
        # gen 历史键 spacing_zyx 取此值（勿再恒等于 orig，避免归一化后错位）。
        "spacing_zyx"  : ([float(s) for s in target_spacing]
                          if spacing_normalized and target_spacing is not None
                          else orig_spacing),
        "target_spacing": ([float(s) for s in target_spacing]
                           if spacing_normalized else None),
        "fg_subsample": int(fg_subsample),
        "made_at"     : datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool_version": _TOOL_VERSION}
    meta_arr = np.array(meta, dtype=object)

    # 7. 原子写（tmp + os.replace）；传文件句柄避免 np.savez 自动追加 .npz。
    # tmp 名带 pid，避免 DDP 多 rank / 多 worker 并发写同一 *.npz.tmp 交错损坏。
    tmp_path = out_p.with_name(f"{out_p.name}.{os.getpid()}.tmp")
    save_fn  = np.savez_compressed if compress else np.savez
    payload  = {
        "image"    : image,
        "label"    : label,
        "fg_slices": fg_slices,
        "fg_coords": fg_coords,
        "fg_coords_cls"  : fg_coords_cls,
        "fg_slices_cls_z": fg_slices_cls_z,
        "fg_slices_cls"  : fg_slices_cls,
        "meta"     : meta_arr}
    if rw is not None:
        payload["rw"] = rw
    if cond is not None:
        payload["cond"] = cond
    try:
        with open(tmp_path, "wb") as fh:
            save_fn(fh, **payload)
        # 原子覆盖：os.replace 在 Windows 上走 MoveFileEx(REPLACE_EXISTING)，
        # 无需先 unlink（unlink→rename 窗口内崩溃会丢目标文件）。
        os.replace(tmp_path, out_p)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise

    elapsed = time.perf_counter() - t0
    return {
        "pid"         : pid,
        "status"      : "written",
        "size_bytes"  : out_p.stat().st_size,
        "elapsed_s"   : elapsed,
        "shape"       : tuple(image.shape),
        "n_fg_slices" : int(fg_slices.size),
        "n_fg_coords" : int(fg_coords.shape[0])}


def _build_sample_table(cfg: Config) -> List[Dict[str, Optional[str]]]:
    """通过 loader helpers 发现/配对 image/label/bbox/rw 路径；遵守 exclude_list。"""
    dc = cfg.data

    image_paths, label_paths = discover_samples(  # 配对
        dc.image_dir, dc.label_dir, dc.image_suffix, dc.label_suffix,
        allow_unpaired=bool(dc.allow_unpaired))

    exclude_pids = _load_exclude_pids(dc.exclude_list)
    image_paths, label_paths, _ = _filter_by_exclude(  # 过滤
        image_paths, label_paths, dc.image_suffix, exclude_pids)

    bbox_paths_all: Optional[List[str]] = None
    if dc.bbox_dir:
        bbox_paths_all = match_bbox_paths(  # 匹配bbox(严格1:1)
            image_paths, dc.bbox_dir, dc.image_suffix, dc.bbox_suffix)

    rw_paths_all: Optional[List[str]] = None
    if dc.region_weight_dir:
        rw_paths_all = match_region_weight_paths(  # 匹配region weight(严格1:1)
            image_paths, dc.region_weight_dir, dc.image_suffix,
            dc.region_weight_suffix)

    samples: List[Dict[str, Optional[str]]] = []
    for i, (img, lbl) in enumerate(zip(image_paths, label_paths)):
        samples.append({
            "pid"  : _stem(img, dc.image_suffix),
            "image": img,
            "label": lbl,
            "bbox" : bbox_paths_all[i] if bbox_paths_all else None,
            "rw"   : rw_paths_all[i] if rw_paths_all else None})
    return samples


def _resolve_label_values(
    cfg: Config, samples: List[Dict[str, Optional[str]]]) -> List[int]:
    """未配置时自动探测 label values（同 loader）。"""
    dc = cfg.data
    if dc.label_values:
        return list(map(int, dc.label_values))
    label_paths = [s["label"] for s in samples]
    detected    = detect_label_values(label_paths)
    return list(map(int, detected))


def _resolve_target_spacing(
    cfg: Config, samples: List[Dict[str, Optional[str]]]) -> List[float]:
    """解析 target_spacing（numpy 轴序 (D,H,W) mm）：显式配置优先；
    否则扫描所有 image 头信息取逐轴中位数（nnU-Net 式指纹，只读头不解码像素）。"""
    ts = cfg.data.target_spacing
    if ts is not None:
        return [float(s) for s in ts]
    spacings = []
    for s in samples:
        img = s["image"]
        if img:
            spacings.append(read_nifti_spacing(img))
    if not spacings:
        raise ValueError(
            "spacing_normalization=True but data.target_spacing is None and no "
            "image headers could be read to compute a median spacing.")
    arr = np.asarray(spacings, dtype=np.float64)  # (N, 3) = (sz, sy, sx)
    median = [float(np.median(arr[:, i])) for i in range(3)]
    logger.info(
        "Computed dataset median spacing (D,H,W)=%s mm from %d volumes.",
        median, len(spacings))
    return median


def prepare_dataset(
    cfg         : Config,
    out_dir     : str,
    workers     : int = 4,
    fg_subsample: int = _DEFAULT_FG_SUBSAMPLE,
    compress    : bool = False,
    overwrite   : bool = False,
    limit       : int = 0) -> Dict[str, int]:
    """为 cfg.data 下的所有样本生成 npz。
    compress 使用 savez_compressed；limit>0 仅处理前 N 个。返 counters {written, skipped, failed, total}。"""
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    samples = _build_sample_table(cfg)  # 配对路径
    if limit and limit > 0:
        samples = samples[:limit]
        logger.info("--limit %d: processing only the first %d samples.",
                    limit, len(samples))

    label_values = _resolve_label_values(cfg, samples)
    logger.info("Using label_values=%s (bg=%d)", label_values, label_values[0])

    # spacing 归一化：解析 target_spacing（显式配置优先，否则扫描头信息取逐轴中位数）。
    spacing_norm = bool(cfg.data.spacing_normalization)
    target_spacing = _resolve_target_spacing(cfg, samples) if spacing_norm else None
    if spacing_norm:
        logger.info(
            "spacing_normalization=True: resampling every volume to "
            "target_spacing=%s mm (numpy axis order (D,H,W)).", target_spacing)

    tasks: List[Tuple[Dict[str, Optional[str]], str]] = []
    for s in samples:
        out_path = str(out_p / f"{s['pid']}.npz")
        tasks.append((s, out_path))

    n_total  = len(tasks)
    counters = {"written": 0, "skipped": 0, "failed": 0, "total": n_total}
    failures : List[Tuple[str, str]] = []   # (pid, err)
    timings  : List[float] = []
    sizes    : List[int] = []

    logger.info(
        "Preparing %d samples → %s (workers=%d, compress=%s, "
        "overwrite=%s, fg_subsample=%d)",
        n_total, out_p, workers, compress, overwrite, fg_subsample)

    def _kwargs(sample: Dict[str, Optional[str]], out_path: str) -> dict:
        return dict(
            pid          = sample["pid"],
            image_path   = sample["image"],
            label_path   = sample["label"],
            bbox_path    = sample["bbox"],
            rw_path      = sample["rw"],
            out_path     = out_path,
            label_values = label_values,
            fg_subsample = fg_subsample,
            compress     = compress,
            overwrite    = overwrite,
            spacing_normalization = spacing_norm,
            target_spacing = target_spacing)

    t0 = time.perf_counter()

    if workers <= 0:
        # 内联：全 traceback、无 pickle，便于调试
        for i, (s, out_path) in enumerate(tasks):
            try:
                res = prepare_one(**_kwargs(s, out_path))
                _record(res, counters, timings, sizes)
                _log_progress(i + 1, n_total, res, t0)
            except Exception as exc:
                counters["failed"] += 1
                failures.append((s["pid"], _short_exc(exc)))
                logger.exception("FAILED pid=%s: %s", s["pid"], exc)
    else:
        # 进程池（Windows spawn；SimpleITK 逐 worker 导入一次）
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_pid = {
                pool.submit(prepare_one, **_kwargs(s, out_path)): s["pid"]
                for s, out_path in tasks}
            for i, fut in enumerate(as_completed(future_to_pid)):
                pid = future_to_pid[fut]
                try:
                    res = fut.result()
                    _record(res, counters, timings, sizes)
                    _log_progress(i + 1, n_total, res, t0)
                except Exception as exc:
                    counters["failed"] += 1
                    failures.append((pid, _short_exc(exc)))
                    logger.error("FAILED pid=%s: %s", pid, exc)

    # 汇总报告
    elapsed     = time.perf_counter() - t0
    total_bytes = sum(sizes)
    total_gb    = total_bytes / (1024 ** 3)
    mean_s      = (sum(timings) / max(len(timings), 1)) if timings else 0.0
    logger.info(
        "Done in %.1fs: written=%d, skipped=%d, failed=%d / total=%d. "
        "Total npz size: %.2f GiB (mean per sample: %.1f MiB, "
        "mean compute: %.2fs).",
        elapsed, counters["written"], counters["skipped"],
        counters["failed"], counters["total"], total_gb,
        (total_bytes / max(len(sizes), 1)) / (1024 ** 2) if sizes else 0.0, mean_s)

    # _failures.txt 与 data.exclude_list 兼容；成功时清除陈旧文件。
    fail_path = out_p / "_failures.txt"
    if not failures and fail_path.is_file():
        fail_path.unlink()
    if failures:
        with open(fail_path, "w", encoding="utf-8") as f:
            f.write("# make_data failures — generated %s\n" %
                    datetime.now(timezone.utc).isoformat(timespec="seconds"))
            f.write("# Format: <pid>\\t<error>\n")
            for pid, err in failures:
                f.write(f"{pid}\t{err}\n")
        logger.warning(
            "Wrote %d failed pid(s) to %s — review before training, then "
            "either re-run with --overwrite for the affected files OR add "
            "them to data.exclude_list.", len(failures), fail_path)

    # 供下游追溯的 manifest。
    manifest = {
        "tool_version": _TOOL_VERSION,
        "made_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_paths": {
            "image_dir": cfg.data.image_dir,
            "label_dir": cfg.data.label_dir,
            "bbox_dir": cfg.data.bbox_dir,
            "region_weight_dir": cfg.data.region_weight_dir,
        },
        "label_values": label_values,
        # 推理复现契约：Predictor 在 cfg.data.target_spacing 未显式配置时从此处
        # 回读（自动中位数不再只存在于日志里）。
        "spacing_normalization": spacing_norm,
        "target_spacing": ([float(s) for s in target_spacing]
                           if target_spacing is not None else None),
        "n_total": counters["total"],
        "n_written": counters["written"],
        "n_skipped": counters["skipped"],
        "n_failed": counters["failed"],
        "compress": compress,
        "fg_subsample": fg_subsample,
    }
    with open(out_p / "_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return counters


def _record(
    res: Dict[str, object],
    counters: Dict[str, int],
    timings: List[float],
    sizes: List[int]) -> None:
    """内联/池两路共用的计数。"""
    status = res.get("status", "written")
    counters[status] = counters.get(status, 0) + 1
    timings.append(float(res.get("elapsed_s", 0.0)))
    sizes.append(int(res.get("size_bytes", 0)))


def _log_progress(
    done: int, total: int, res: Dict[str, object], t0: float) -> None:
    """逐样本/批次进度日志。"""
    if done == 1 or done == total or done % 10 == 0:
        elapsed = time.perf_counter() - t0
        rate = done / max(elapsed, 1e-6)
        eta = (total - done) / max(rate, 1e-6)
        size_mib = float(res.get("size_bytes", 0)) / (1024 ** 2)
        shape = res.get("shape", "-")
        logger.info(
            "[%d/%d] %s pid=%s  shape=%s  %.1f MiB  (%.2fs)  "
            "rate=%.2f sample/s  ETA=%.0fs",
            done, total, res.get("status", "?"),
            res.get("pid", "?"), shape, size_mib,
            float(res.get("elapsed_s", 0.0)),
            rate, eta)


def _short_exc(exc: BaseException, max_len: int = 200) -> str:
    """单行限长的异常概要，写入 failures 文件。"""
    msg = f"{type(exc).__name__}: {exc}".replace("\n", " | ").replace("\t", " ")
    return msg[:max_len]


def _setup_logging(level: str = "INFO") -> None:
    # 复用集中式日志配置；out_dir=None 表示只配控制台（彩色），不写文件。
    from ..utils.logging_utils import setup_logging
    setup_logging(output_dir=None, level=level)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-compute per-sample npz packages "
                    "(image+label+rw+fg-index, bbox-cropped).")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config (same one used by train.py).")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Config overrides (key=value, dot notation), "
                             "e.g. --override data.image_dir=F:/x data.label_values=[0,1].")
    parser.add_argument("--out", type=str, required=True,
                        help="Output directory for the npz packages.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel worker processes (0 = inline). "
                             "Each worker peaks at ~1 cropped sample's "
                             "RAM; tune to host memory.")
    parser.add_argument("--fg-subsample", type=int,
                        default=_DEFAULT_FG_SUBSAMPLE,
                        help="Max stored 3D fg coords per sample per "
                             "foreground class.")
    parser.add_argument("--compress", action="store_true",
                        help="Use np.savez_compressed (smaller disk, "
                             "but no shared OS page cache and slower load).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-write existing npz files.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N samples "
                             "(smoke-test; 0=all).")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    _setup_logging(args.log_level)

    cfg = load_config(args.config)
    logger.info("Config loaded from %s", args.config)
    if args.override:
        # 复用任务侧点记法 override 语义（sections=None 即单段 Config 路由）；
        # 懒导入保持 make_data 轻量。
        from ..config.task_io import apply_dotted_overrides
        apply_dotted_overrides(cfg, args.override)
        cfg.sync()
        cfg.validate()

    counters = prepare_dataset(
        cfg=cfg,
        out_dir=args.out,
        workers=args.workers,
        fg_subsample=args.fg_subsample,
        compress=args.compress,
        overwrite=args.overwrite,
        limit=args.limit,
    )
    return 0 if counters["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
