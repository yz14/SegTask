#!/usr/bin/env python3
"""
CT Body Segmentation — 纯图像处理，无需深度学习
=================================================
从 CT 图像中自动分割患者 body，并鲁棒地去除 CT 床板假阳性。
支持"body 与床板紧贴/接触"的难处理场景。

核心思路：利用形状差异——"床板是薄片，body 是大体积"
──────────────────────────────────────────────────
CT 床板在患者垂直方向（前后轴 / A-P 方向）的厚度约 15~30 mm，
患者 body 在同一方向的厚度通常超过 150 mm。

方法：沿 A-P 轴做一维方向性腐蚀（kernel > 床板厚度）：
  • 床板（薄）→ 完全消失（太薄，腐蚀后不剩）
  • body（厚）→ 缩小但连通核心保留
  • 即使 body 与床板紧贴：腐蚀后两者自动断开

找到腐蚀体积中最大连通域 = body core（无床板污染）
再用相同 kernel 膨胀回来并与原始 filled mask 求交，
精确恢复 body 真实边界，且床板区域在膨胀到达范围之外。

            腐蚀前：|██████████████████ body ████|████ 床板 ████|
            腐蚀后：        |████ body core ████|  （床板消失）
            膨胀后：|██████████████████ body ████|  （精确恢复）
            ∩ filled：     body 保留，床板排除 ✓

Pipeline（共 7 步）：
  1. HU 阈值 (> -200) → 初始组织 mask
  2. 逐 axial slice 填充内部空洞（肺、肠道气体等）
  3. 沿 A-P 轴一维腐蚀，kernel 大于床板厚度 → 去除床板
  4. 取腐蚀结果最大连通域 → body core（如太小自动降级重试）
  5. 同尺寸一维膨胀 body core，与 filled mask 求交 → 恢复 body
  6. 多平面 + 3D 空洞填充（axial + coronal + 3D）
  7. 保留最大连通域 → 最终 body mask

GPU 加速：若安装了 CuPy，自动用 GPU 执行腐蚀/膨胀/3D 填充。

使用方法：
  python segment_body_ct.py --img_dir F:/airway_segment_with_img/imgs --out_dir F:/airway_segment_with_img/imgs/body_process
  python segment_body_ct.py --img_dir /path/to/nifti --out_dir /path/to/output \\
      --no_gpu --bed_mm 35 --hu_thresh -200 --overwrite
"""

import os
import sys
import glob
import time
import argparse
import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import nibabel as nib
from scipy import ndimage as cpu_ndi

# ── 可选 GPU 加速 (CuPy) ─────────────────────────────────────────────
try:
    import cupy as cp
    import cupyx.scipy.ndimage as gpu_ndi

    # 验证 binary_fill_holes 是否在当前 cupyx 版本中可用
    _GPU_FILL3D = hasattr(gpu_ndi, "binary_fill_holes")
    CUPY_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CUPY_AVAILABLE = False
    _GPU_FILL3D = False

# ── 日志 ─────────────────────────────────────────────────────────────
logging.basicConfig(
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger("body_seg")


# ══════════════════════════════════════════════════════════════════════
# 默认超参（可通过 CLI 覆盖）
# ══════════════════════════════════════════════════════════════════════
HU_THRESHOLD: float = -200.0   # 组织 / 空气 HU 阈值
BED_THICKNESS_MM: float = 30.0 # 预期床板最大厚度 (mm)，用于计算腐蚀尺寸
EROSION_MARGIN_MM: float = 10.0 # 安全余量：erosion_mm = bed_mm + margin
MIN_BODY_VOXELS: int = 50_000  # body core 最小体素数（用于异常检测）


# ══════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════

def to_numpy(x) -> np.ndarray:
    """CuPy 数组 → NumPy；NumPy 直接透传。"""
    if CUPY_AVAILABLE and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def find_ap_axis(affine: np.ndarray) -> int:
    """
    从 NIfTI affine 矩阵中找到与物理 Y 方向（前后轴 A-P）最对齐的图像轴。
    CT 床板沿该轴最薄（即该轴是需要做方向性腐蚀的轴）。
    对标准轴向 CT（RAS 方向），通常返回 1（图像 Y 轴）。
    """
    R = np.abs(affine[:3, :3])
    R = R / (np.linalg.norm(R, axis=0, keepdims=True) + 1e-9)
    # 第 1 行 = 物理 Y（RAS 坐标系的 A-P 方向）
    return int(np.argmax(R[1]))


def line_se(axis: int, half: int, ndim: int = 3) -> np.ndarray:
    """
    构造一维线形结构元素：
      沿 `axis` 方向长度 = 2*half+1，其余方向长度 = 1。
    用于只在单一方向做腐蚀/膨胀（不影响其他方向）。
    """
    shape = [1] * ndim
    shape[axis] = 2 * half + 1
    return np.ones(shape, dtype=bool)


def fill_holes_per_slice(mask: np.ndarray, axis: int = 2) -> np.ndarray:
    """
    沿指定轴逐层做 2D 空洞填充（始终在 CPU 执行）。
    用途：填充肺部、肠道气体等，使其被计入 body 内部。
    """
    out = np.empty_like(mask, dtype=bool)
    n = mask.shape[axis]
    for i in range(n):
        sl = tuple(i if d == axis else slice(None) for d in range(3))
        out[sl] = cpu_ndi.binary_fill_holes(mask[sl])
    return out


def largest_n_cc(mask: np.ndarray, n: int = 1) -> np.ndarray:
    """
    保留 mask 中体素数最多的 n 个 3D 连通域，返回布尔 mask。
    始终在 CPU 执行（scipy.ndimage.label 更稳定）。
    """
    lbl, k = cpu_ndi.label(mask.astype(bool))
    if k == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.array(cpu_ndi.sum(mask, lbl, range(1, k + 1)))
    top_labels = np.argsort(sizes)[-n:] + 1  # 1-indexed
    return np.isin(lbl, top_labels)


# ══════════════════════════════════════════════════════════════════════
# 核心分割函数
# ══════════════════════════════════════════════════════════════════════

def segment_body(
    hu: np.ndarray,
    vox_mm: np.ndarray,
    affine: np.ndarray,
    hu_thresh: float = HU_THRESHOLD,
    bed_mm: float = BED_THICKNESS_MM,
    use_gpu: bool = False,
) -> np.ndarray:
    """
    从 CT HU 体积中分割 body，返回 uint8 mask（1=body, 0=背景/床板）。

    Parameters
    ----------
    hu       : 3D HU 数组，shape (X, Y, Z)
    vox_mm   : 体素大小，(dx, dy, dz) mm
    affine   : NIfTI affine 矩阵（用于确定 A-P 轴）
    hu_thresh: HU 阈值，默认 -200
    bed_mm   : 预期床板最大厚度 mm，腐蚀尺寸 = bed_mm + EROSION_MARGIN_MM
    use_gpu  : 是否使用 CuPy GPU 加速

    Returns
    -------
    uint8 mask, shape 同 hu
    """
    t0 = time.time()
    use_gpu = use_gpu and CUPY_AVAILABLE

    if use_gpu:
        xp  = cp
        ndi = gpu_ndi
        log.debug("  device: GPU (CuPy)")
    else:
        xp  = np
        ndi = cpu_ndi
        log.debug("  device: CPU (NumPy / SciPy)")

    # ── 步骤 1：HU 阈值 ──────────────────────────────────────────────
    log.info("  1/7  HU 阈值 (> %.0f)", hu_thresh)
    vol = xp.asarray(hu.astype(np.float32))
    raw = (vol > hu_thresh)

    # ── 步骤 2：逐 axial slice 填充 2D 空洞 ──────────────────────────
    # 目的：将肺部、胃肠道气体等内部空腔纳入 body mask
    log.info("  2/7  逐 axial slice 填充空洞（肺/肠道气体）")
    filled = fill_holes_per_slice(to_numpy(raw), axis=2)

    # ── 步骤 3：沿 A-P 轴一维方向性腐蚀 ─────────────────────────────
    #
    # 关键原理（以 A-P 轴为 Y，以下数值为示例）：
    #
    #   体积（Y 方向）：
    #     body = Y[50, 200]，厚度 150 mm
    #     床板 = Y[200, 225]，厚度  25 mm（与 body 紧贴）
    #
    #   erosion kernel = 40 mm（bed_mm 30 + margin 10）
    #
    #   腐蚀后：
    #     body core = Y[70, 180]  （150 - 2×20 = 110 mm，存活）
    #     床板      = 完全消失     （25 < 40，全部腐蚀）
    #
    #   → 即使 body 与床板紧贴，腐蚀后两者自动断开！

    ap_axis = find_ap_axis(affine)
    vox_ap  = float(vox_mm[ap_axis])
    total_erosion_mm = bed_mm + EROSION_MARGIN_MM
    half_k = max(2, int(np.ceil(total_erosion_mm / vox_ap)))

    log.info(
        "  3/7  沿 axis=%d 做一维腐蚀  "
        "half_k=%d vox (= %.0f mm；床板 ≤ %.0f mm)",
        ap_axis, half_k, half_k * vox_ap, bed_mm,
    )

    se_np = line_se(ap_axis, half_k)

    if use_gpu:
        se_g     = xp.asarray(se_np)
        filled_g = xp.asarray(filled)
        eroded_g = ndi.binary_erosion(filled_g, structure=se_g)
        eroded   = to_numpy(eroded_g)
    else:
        eroded = cpu_ndi.binary_erosion(filled, structure=se_np)

    # ── 步骤 4：取腐蚀后最大连通域 = body core ───────────────────────
    log.info("  4/7  提取 body core（腐蚀后最大连通域）")
    body_core = largest_n_cc(eroded)

    n_core = int(body_core.sum())
    log.info("       body core 体素数：%d", n_core)

    if n_core < MIN_BODY_VOXELS:
        # 降级处理：减小腐蚀尺寸重试（适用于极瘦小患者或奇特体位）
        warnings.warn(
            f"body core 体素数过少（{n_core} < {MIN_BODY_VOXELS}），"
            f"尝试以 50% 腐蚀尺寸重试。",
            RuntimeWarning,
            stacklevel=2,
        )
        half_k2   = max(1, half_k // 2)
        se2_np    = line_se(ap_axis, half_k2)
        eroded2   = cpu_ndi.binary_erosion(filled, structure=se2_np)
        body_core = largest_n_cc(eroded2)
        n_core2   = int(body_core.sum())
        log.warning("       降级后 body core：%d vox（half_k=%d → %d）",
                    n_core2, half_k, half_k2)
        # 更新腐蚀参数（确保膨胀量与腐蚀量一致）
        half_k = half_k2
        se_np  = se2_np
        if use_gpu:
            se_g = xp.asarray(se_np)

    # ── 步骤 5：膨胀恢复 + 与 filled mask 求交 ───────────────────────
    #
    # 几何原理（body 紧贴床板时的恢复过程）：
    #
    #   body core 位于 Y[70, 180]，half_k = 20 vox
    #   膨胀后     Y[50, 200]   ← 精确恢复到原始 body 边界
    #   filled 中：body = Y[50,200]，床板 = Y[200,225]
    #   膨胀后范围 Y[50,200] ∩ filled → 包含 body，不包含床板 ✓
    #
    # 因为膨胀只能从 body core 的边界向外扩展 half_k 个体素，
    # 恰好到达 body 的真实边界；床板在 body 边界之外，无法到达。

    log.info("  5/7  膨胀 body core 并与 filled mask 求交（恢复真实边界）")

    if use_gpu:
        core_g   = xp.asarray(body_core)
        dil_g    = ndi.binary_dilation(core_g, structure=se_g)
        body_g   = dil_g & filled_g
        body     = to_numpy(body_g)
    else:
        body = cpu_ndi.binary_dilation(body_core, structure=se_np) & filled

    # ── 步骤 6：多平面 + 3D 空洞填充 ─────────────────────────────────
    # axial（处理肺）+ coronal（处理手臂与躯干的连接）+ 3D（全局）
    log.info("  6/7  多平面空洞填充（axial + coronal + 3D）")

    body = fill_holes_per_slice(body, axis=2)   # axial
    body = fill_holes_per_slice(body, axis=1)   # coronal

    # 3D 整体填充
    if use_gpu and _GPU_FILL3D:
        body_g = xp.asarray(body)
        body   = to_numpy(ndi.binary_fill_holes(body_g))
    else:
        body = cpu_ndi.binary_fill_holes(body)

    # ── 步骤 7：保留最大连通域（最终清理）────────────────────────────
    log.info("  7/7  保留最大连通域")
    body = largest_n_cc(body)

    # ── 统计输出 ──────────────────────────────────────────────────────
    n_vox = int(body.sum())
    vol_cc = n_vox * float(np.prod(vox_mm)) / 1000.0
    log.info(
        "  ✓ 完成  %.1fs  |  body ≈ %.0f cm³  (%d 体素)",
        time.time() - t0, vol_cc, n_vox,
    )

    return body.astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════
# 单文件处理
# ══════════════════════════════════════════════════════════════════════

def process_file(
    src: str,
    out_dir: str,
    use_gpu: bool = False,
    overwrite: bool = False,
    hu_thresh: float = HU_THRESHOLD,
    bed_mm: float = BED_THICKNESS_MM,
) -> bool:
    """加载一个 NIfTI CT 文件，分割 body，保存结果。"""

    src_path = Path(src)
    # 处理 .nii.gz / .nii 后缀
    stem = src_path.name
    for suf in (".nii.gz", ".nii"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    dst = Path(out_dir) / f"{stem}_body.nii.gz"

    if dst.exists() and not overwrite:
        log.info("  [跳过] %s（已存在，使用 --overwrite 强制覆盖）", dst.name)
        return True

    log.info("─" * 65)
    log.info("  文件：%s", src_path.name)

    # ── 加载 ──────────────────────────────────────────────────────
    try:
        nii = nib.load(str(src_path))
        hu  = np.asarray(nii.dataobj, dtype=np.float32)
        vox = np.abs(np.array(nii.header.get_zooms()[:3], dtype=float))
    except Exception as exc:
        log.error("  加载失败：%s", exc)
        return False

    log.info(
        "  shape=%-15s  vox=%s mm  HU=[%.0f, %.0f]",
        str(hu.shape), vox, hu.min(), hu.max(),
    )

    # ── 分割 ──────────────────────────────────────────────────────
    try:
        mask = segment_body(
            hu, vox, nii.affine,
            hu_thresh=hu_thresh,
            bed_mm=bed_mm,
            use_gpu=use_gpu,
        )
    except Exception as exc:
        log.error("  分割失败：%s", exc)
        import traceback
        traceback.print_exc()
        return False

    # ── 保存（保持原始几何信息）────────────────────────────────────
    try:
        out_nii = nib.Nifti1Image(mask, nii.affine, nii.header)
        out_nii.header.set_data_dtype(np.uint8)
        out_nii.header["cal_min"] = 0
        out_nii.header["cal_max"] = 1
        nib.save(out_nii, str(dst))
        log.info("  已保存 → %s", dst)
    except Exception as exc:
        log.error("  保存失败：%s", exc)
        return False

    return True


# ══════════════════════════════════════════════════════════════════════
# 命令行入口
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="segment_body_ct",
        description="CT body 分割（纯图像处理，无深度学习，可选 CuPy GPU 加速）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--img_dir",   required=True,
                    help="输入目录，包含 .nii / .nii.gz 文件")
    ap.add_argument("--out_dir",   required=True,
                    help="输出目录，存放 body 分割结果")
    ap.add_argument("--no_gpu",    action="store_true",
                    help="禁用 CuPy GPU 加速，仅用 CPU")
    ap.add_argument("--overwrite", action="store_true",
                    help="覆盖已有输出文件")
    ap.add_argument(
        "--hu_thresh", type=float, default=HU_THRESHOLD,
        help="HU 阈值（组织 vs 空气）",
    )
    ap.add_argument(
        "--bed_mm", type=float, default=BED_THICKNESS_MM,
        help="预期 CT 床板最大厚度（mm），腐蚀 kernel = bed_mm + 10mm 余量",
    )
    ap.add_argument("--verbose", action="store_true",
                    help="开启 DEBUG 日志")
    args = ap.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    os.makedirs(args.out_dir, exist_ok=True)
    use_gpu = (not args.no_gpu) and CUPY_AVAILABLE

    log.info("═" * 65)
    log.info("CT Body Segmentation（纯图像处理）")
    log.info("  img_dir  : %s", args.img_dir)
    log.info("  out_dir  : %s", args.out_dir)
    log.info("  HU 阈值  : %.0f", args.hu_thresh)
    log.info("  床板厚度 : %.0f mm", args.bed_mm)
    log.info(
        "  GPU 加速 : %s  (CuPy 可用: %s, binary_fill_holes: %s)",
        use_gpu, CUPY_AVAILABLE, _GPU_FILL3D,
    )
    log.info("═" * 65)

    files = sorted(
        glob.glob(os.path.join(args.img_dir, "*.nii.gz")) +
        glob.glob(os.path.join(args.img_dir, "*.nii"))
    )
    if not files:
        log.error("在 %s 中未找到 .nii / .nii.gz 文件", args.img_dir)
        sys.exit(1)

    log.info("共找到 %d 个文件", len(files))
    ok = 0
    for i, f in enumerate(files, 1):
        log.info("\n[%d / %d]", i, len(files))
        if process_file(
            f,
            args.out_dir,
            use_gpu=use_gpu,
            overwrite=args.overwrite,
            hu_thresh=args.hu_thresh,
            bed_mm=args.bed_mm,
        ):
            ok += 1

    log.info("\n%s", "═" * 65)
    log.info("完成：%d / %d 成功", ok, len(files))


if __name__ == "__main__":
    main()