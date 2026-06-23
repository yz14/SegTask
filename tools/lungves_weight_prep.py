# lungves_weight_prep.py
# 为肺血管（标签1、标签2，如动/静脉）生成区域权重 nii 文件
# 逻辑：
#   1) 骨架提取 -> 分叉点检测 -> 膨胀 = 血管分叉点 (W=19)
#   2) mask - open(mask) = 细血管 (W=14)
#   3) mask - 细血管 = 主干; 主干 - erode(主干) = 主干边缘 (W=9)
#   4) erode(主干) = 主干内部 (W=4)
# 对标签1、2分别处理后合并（空间不重叠，直接叠加）。
#
# 此外，对血管外的背景按“离血管表面的距离”再分两层（难负样本加权，
# 抑制血管周围假阳），仅赋给真正的背景体素、不覆盖任何前景：
#   5) 紧贴血管表面 1~INNER_SHELL_R voxel 的背景 = 内壳 (W=3)
#   6) 再往外 INNER_SHELL_R~OUTER_SHELL_R voxel 的背景 = 外壳 (W=1)
# 更远的背景保持 W=0。（注：加载时整体 +1 偏移，故实际进损失的权重为此处值+1。）

import os
import sys
from glob import glob

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from skimage.morphology import skeletonize
from scipy import ndimage as ndi

# opt_cupy 依赖同目录的 debug_utils，必须把目录加入 sys.path 后再导入
_OPT_CUPY_DIR = r'D:\codes\A0_dev\MedImgProc\code'
if _OPT_CUPY_DIR not in sys.path:
    sys.path.insert(0, _OPT_CUPY_DIR)
from opt_cupy import morph_dilate_gpu, morph_erode_gpu  # noqa: E402


# ==== 路径 ====
IMG_DIR    = r'F:\lung_vessel\imgs'
LBL_DIR    = r'F:\lung_vessel\lbls'
OUT_DIR    = r'F:\lung_vessel\lungves_weight'
os.makedirs(OUT_DIR, exist_ok=True)

# ==== 参数（体素为单位） ====
OPEN_RADIUS      = 4      # 开运算核半径：决定“细血管”阈值
ERODE_RADIUS     = 4      # 主干腐蚀半径：决定边缘厚度
SKEL_DILATE_R    = 3      # 分叉点膨胀半径：覆盖分叉周围区域
INNER_SHELL_R    = 2      # 内壳半径：紧贴血管表面的背景带（1~INNER_SHELL_R voxel）
OUTER_SHELL_R    = 6      # 外壳半径：内壳之外、本半径之内的背景带

# ==== 权重 ====
W_JUNCTION       = 19     # 血管分叉点
W_THIN_VESSEL    = 14     # 细血管
W_TRUNK_EDGE     = 9      # 主干边缘
W_TRUNK_INTERIOR = 4      # 主干内部
W_INNER_SHELL    = 3      # 血管外·内壳（紧贴血管表面的背景）
W_OUTER_SHELL    = 1      # 血管外·外壳（内壳之外的背景）
W_DEFAULT        = 0      # 背景/其它

# ==== 标签值 ====
LABELS = [1, 2]  # 分别处理（如动脉/静脉）


def find_junctions(skel_bool: np.ndarray) -> np.ndarray:
    """
    在 3D 骨架二值图中检测分叉点（junctions）。
    对骨架的每个前景体素，统计其 26-邻域内其它前景体素数量。
    邻居数 >= 3 认为是分叉点。
    """
    if not skel_bool.any():
        return np.zeros_like(skel_bool)

    # 26-邻域卷积核
    kernel = np.ones((3, 3, 3), dtype=np.int32)
    kernel[1, 1, 1] = 0

    # 用 int32 做卷积，避免溢出
    skel_int = skel_bool.astype(np.int32)
    neighbor_count = ndi.convolve(skel_int, kernel, mode='constant', cval=0)

    # 邻居数 >= 3 且本身是骨架点
    junctions = skel_bool & (neighbor_count >= 3)
    return junctions


def process_single_label(mask_bool: np.ndarray, label_value: int) -> np.ndarray:
    """
    对单个标签的二值 mask 计算区域权重图（uint8）。
    mask_bool: 3D bool array
    label_value: 仅用于打印信息
    """
    if not mask_bool.any():
        return np.zeros(mask_bool.shape, dtype=np.uint8)

    # ---- 1) 分叉点 ----
    # 骨架提取（3D）
    skel = skeletonize(mask_bool.astype(np.uint8)).astype(bool)
    # 检测分叉点
    junctions = find_junctions(skel)
    # 膨胀分叉点，限制在 mask 内
    if junctions.any():
        junctions_dilated = morph_dilate_gpu(
            junctions.astype(np.uint8), kernel_size=2 * SKEL_DILATE_R + 1
        ).astype(bool)
        junctions_dilated = junctions_dilated & mask_bool
    else:
        junctions_dilated = junctions

    # ---- 2) 细血管 = mask - open(mask) ----
    # 开运算 = 腐蚀后膨胀
    opened = morph_erode_gpu(mask_bool.astype(np.uint8), kernel_size=2 * OPEN_RADIUS + 1)
    opened = morph_dilate_gpu(opened, kernel_size=2 * OPEN_RADIUS + 1)
    opened = (opened > 0).astype(bool)

    thin_vessel = mask_bool & (~opened)  # 细血管区域

    # ---- 3) 主干 = mask - 细血管 = open(mask) ----
    trunk = opened  # 等价于 mask_bool & (~thin_vessel)

    # ---- 4) 主干边缘 & 主干内部 ----
    trunk_eroded = morph_erode_gpu(trunk.astype(np.uint8), kernel_size=2 * ERODE_RADIUS + 1)
    trunk_eroded = (trunk_eroded > 0).astype(bool)

    trunk_edge = trunk & (~trunk_eroded)
    trunk_interior = trunk_eroded

    # ---- 5) 组合权重（按优先级从低到高赋值，高权重覆盖低权重） ----
    weight = np.full(mask_bool.shape, W_DEFAULT, dtype=np.uint8)

    weight[trunk_interior] = W_TRUNK_INTERIOR   # 4
    weight[trunk_edge]     = W_TRUNK_EDGE       # 9（覆盖边缘处的内部）
    weight[thin_vessel]    = W_THIN_VESSEL      # 14
    weight[junctions_dilated] = W_JUNCTION      # 19（最高优先级）

    return weight


def compute_bg_shells(fg_union: np.ndarray) -> tuple:
    """由全部前景的并集，按离血管表面的距离生成两层“背景壳”掩码。
    返回 (inner_shell, outer_shell)，均为 bool 且仅落在背景（fg 之外）；二者互斥。
    inner_shell: fg 之外、距表面 1~INNER_SHELL_R voxel。
    outer_shell: fg 之外、距表面 (INNER_SHELL_R, OUTER_SHELL_R] voxel。
    """
    bg = ~fg_union
    fg_u8 = fg_union.astype(np.uint8)

    dil_inner = morph_dilate_gpu(
        fg_u8, kernel_size=2 * INNER_SHELL_R + 1
    ).astype(bool)
    dil_outer = morph_dilate_gpu(
        fg_u8, kernel_size=2 * OUTER_SHELL_R + 1
    ).astype(bool)

    inner_shell = dil_inner & bg                 # 紧贴表面的背景环
    outer_shell = dil_outer & (~dil_inner) & bg  # 再往外的背景环
    return inner_shell, outer_shell


def main():
    lbl_files = sorted(glob(os.path.join(LBL_DIR, '*.nii.gz')))
    print(f'Found {len(lbl_files)} label files')

    for lbl_f in tqdm(lbl_files):
        name = os.path.basename(lbl_f)
        out_f = os.path.join(OUT_DIR, name)

        if os.path.exists(out_f):
            continue

        # 读取 label（多标签：背景=0, 标签1, 标签2）
        lbl_img = sitk.ReadImage(lbl_f)
        lbl_arr = sitk.GetArrayFromImage(lbl_img)

        # 检查对应 image 是否存在（仅做存在性校验，图像数据本身未使用）
        # 标签名含 _mask 后缀，图像名没有，例如：
        #   lbl: CE021001-013007818-65860-7_mask.nii.gz
        #   img: CE021001-013007818-65860-7.nii.gz
        img_name = name.replace('_mask.nii.gz', '.nii.gz')
        img_f = os.path.join(IMG_DIR, img_name)
        if not os.path.exists(img_f):
            print(f'[skip] missing image: {img_f}')
            continue

        final_weight = np.zeros(lbl_arr.shape, dtype=np.uint8)

        for lab in LABELS:
            mask_bool = (lbl_arr == lab)
            if not mask_bool.any():
                continue

            w = process_single_label(mask_bool, lab)

            # 两标签空间互斥，直接叠加（取最大，因同位置不应有冲突）
            final_weight = np.maximum(final_weight, w)

        # ---- 背景壳（难负样本加权）：仅赋给前景之外的背景，不覆盖任何前景 ----
        fg_union = np.zeros(lbl_arr.shape, dtype=bool)
        for lab in LABELS:
            fg_union |= (lbl_arr == lab)

        if fg_union.any():
            inner_shell, outer_shell = compute_bg_shells(fg_union)
            # 先外后内（二者已互斥，顺序仅为稳妥）；只写入背景体素。
            final_weight[outer_shell] = W_OUTER_SHELL
            final_weight[inner_shell] = W_INNER_SHELL

        # 保存
        weight_img = sitk.GetImageFromArray(final_weight)
        weight_img.CopyInformation(lbl_img)
        sitk.WriteImage(weight_img, out_f, useCompression=True)

    print('Done.')


if __name__ == '__main__':
    main()
