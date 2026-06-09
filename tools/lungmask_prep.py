# region_weight_path生成区域权重nii文件
# 对目标区域进行膨胀、腐蚀，得到：外圈，内圈，内部
# 对相似像素膨胀+腐蚀：去掉空洞

import os
import sys
from glob import glob

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# opt_cupy 依赖同目录的 debug_utils，必须把目录加入 sys.path 后再导入
_OPT_CUPY_DIR = r'D:\codes\A0_dev\MedImgProc\code'
if _OPT_CUPY_DIR not in sys.path:
    sys.path.insert(0, _OPT_CUPY_DIR)
from opt_cupy import morph_dilate_gpu, morph_erode_gpu  # noqa: E402


# bone_path          = r'F:\BaiduNetdiskDownload\bone_bbox'
nii_path           = r'F:\lung\imgs'
lung_path          = r'F:\lung\lung'
region_weight_path = r'F:\lung\lung_weight'
# lung_bone_path     = r'F:\BaiduNetdiskDownload\lung_bone'
body_path          = r'F:\lung\body_base_pred'
os.makedirs(region_weight_path, exist_ok=True)
# os.makedirs(lung_bone_path, exist_ok=True)


# ==== 参数 ====
MORPH_RADIUS    = 15       # 腐蚀/膨胀半径（体素）
LUNG_HU_LOW     = -2000    # 像素类似下界
LUNG_HU_HIGH    = -600     # 像素类似上界
W_INNER_EDGE    = 9        # 内圈
W_LUNG_INTERIOR = 2        # 内部
W_OUTER         = 8        # 外圈
W_HU_LIKE       = 19       # 像素类似
# W_BONE          = 4        # 骨头
W_DEFAULT       = 0        # 其他区域


# def _bone_file(case_name: str) -> str:
#     """bone_bbox 中的文件命名约定：<case>_pred.nii.gz"""
#     return os.path.join(bone_path, case_name.replace('.nii.gz', '.nii.gz'))

def _bbox_file(case_name: str) -> str:
    """bone_bbox 中的文件命名约定：<case>_pred.nii.gz"""
    return os.path.join(body_path, case_name.replace('.nii.gz', '_pred.nii.gz'))


lung_files = sorted(glob(os.path.join(lung_path, '*.nii.gz')))
print(f'Found {len(lung_files)} lung files')
for lung_f in tqdm(lung_files):
    name     = os.path.basename(lung_f)
    # bone_f   = _bone_file(name)
    ct_f     = os.path.join(nii_path, name)
    body_f   = _bbox_file(name)
    weight_f = os.path.join(region_weight_path, name)
    # merge_f  = os.path.join(lung_bone_path, name)

    # if not os.path.exists(bone_f):
    #     print(f'[skip] missing bone: {bone_f}')
    #     continue
    if not os.path.exists(ct_f):
        print(f'[skip] missing ct: {ct_f}')
        continue
    if not os.path.exists(body_f):
        print(f'[skip] missing body: {body_f}')
        continue

    # ---- 读入 ----
    # 标签1
    lung_img = sitk.ReadImage(lung_f)
    lung_arr = (sitk.GetArrayFromImage(lung_img) > 0).astype(np.uint8)

    # # 标签2
    # bone_img = sitk.ReadImage(bone_f)
    # bone_arr = (sitk.GetArrayFromImage(bone_img) > 0).astype(np.uint8)

    # # bone 与 lung 来自不同流水线，形状若不一致直接跳过避免错位
    # if bone_arr.shape != lung_arr.shape:
    #     print(f'[skip] shape mismatch lung{lung_arr.shape} vs bone{bone_arr.shape}: {name}')
    #     continue

    # # ---- 1) 合并 bone + lung -> lung_bone ----
    # if not os.path.exists(merge_f):
    #     merged = np.where(lung_arr > 0, 1, 0).astype(np.uint8)
    #     merged[bone_arr > 0] = 2  # 骨头覆盖肺
    #     merged_img = sitk.GetImageFromArray(merged)
    #     merged_img.CopyInformation(lung_img)
    #     sitk.WriteImage(merged_img, merge_f, useCompression=True)

    # ---- 2) 区域权重 ----
    if os.path.exists(weight_f):
        continue

    # 图像
    ct_img = sitk.ReadImage(ct_f)
    ct_arr = sitk.GetArrayFromImage(ct_img)
    if ct_arr.shape != lung_arr.shape:
        print(f'[skip-weight] shape mismatch lung{lung_arr.shape} vs ct{ct_arr.shape}: {name}')
        continue

    # bbox
    body_img = sitk.ReadImage(body_f)
    body_arr = (sitk.GetArrayFromImage(body_img) > 0).astype(np.uint8)
    if body_arr.shape != lung_arr.shape:
        print(f'[skip-weight] shape mismatch lung{lung_arr.shape} vs body{body_arr.shape}: {name}')
        continue
    body_arr = morph_dilate_gpu(body_arr, kernel_size=5)  # 膨胀去空洞
    body_arr = morph_erode_gpu(body_arr, kernel_size=5)   # 腐蚀
    body_arr = (body_arr > 0).astype(bool) 

    # 标签1 内、外带
    eroded    = morph_erode_gpu(lung_arr, kernel_size=MORPH_RADIUS).astype(bool)   # 腐蚀
    dilated   = morph_dilate_gpu(lung_arr, kernel_size=MORPH_RADIUS).astype(bool)  # 膨胀
    lung_bool = lung_arr.astype(bool)

    inner_edge    = lung_bool & (~eroded)   # 标签1 内部边缘（带状）
    lung_interior = eroded                  # 标签1 内部
    outer_edge    = dilated & (~lung_bool)  # 标签1 外部边缘（带状）
    
    # 与标签1像素相似
    hu_like = ((ct_arr >= LUNG_HU_LOW) &
               (ct_arr <= LUNG_HU_HIGH) &
               (~lung_bool) & body_arr).astype(np.uint8)  # 限制在 bbox 内
    hu_like = morph_dilate_gpu(hu_like, kernel_size=5)    # 膨胀去空洞
    hu_like = morph_erode_gpu(hu_like, kernel_size=5)     # 腐蚀
    hu_like = (hu_like > 0).astype(bool)

    weight = np.full(lung_arr.shape, W_DEFAULT, dtype=np.uint8)
    weight[outer_edge]    = W_OUTER
    weight[lung_interior] = W_LUNG_INTERIOR
    weight[inner_edge]    = W_INNER_EDGE
    # weight[bone_arr > 0]  = W_BONE
    weight[hu_like]       = W_HU_LIKE

    weight_img = sitk.GetImageFromArray(weight)
    weight_img.CopyInformation(lung_img)
    sitk.WriteImage(weight_img, weight_f, useCompression=True)

print('Done.')