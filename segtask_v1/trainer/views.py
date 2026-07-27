"""视图重塑纯函数：center_crop / split_views_native_3d / split_views_native_d /
squeeze_2_5d / squeeze_2_5d_keep_views。

无状态、不依赖 Trainer / Config —— 直接用入参传所需尺寸；既给 Round 2 的
``ViewPipeline`` 子类调用，也供 ``Trainer`` 旧方法做 thin shim 复用，避免重复实现。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from taskcore.engine.views import (  # noqa: F401  (re-export)
    squeeze_2_5d,
    squeeze_2_5d_keep_views,
)


# ---------------------------------------------------------------------------
# Center crop (post-augment oversample 回切)
# ---------------------------------------------------------------------------
def center_crop(
    image: torch.Tensor,
    label: torch.Tensor,
    wmap: Optional[torch.Tensor],
    target_patch_size: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """``(B,C,D,H,W)`` 中心裁回 ``target_patch_size``。"""
    tD, tH, tW    = target_patch_size
    _, _, D, H, W = image.shape
    d0, h0, w0 = (D - tD) // 2, (H - tH) // 2, (W - tW) // 2
    image = image[:, :, d0:d0 + tD, h0:h0 + tH, w0:w0 + tW]
    label = label[:, :, d0:d0 + tD, h0:h0 + tH, w0:w0 + tW]
    if wmap is not None:
        wmap = wmap[:, :, d0:d0 + tD, h0:h0 + tH, w0:w0 + tW]
    return image, label, wmap


# ---------------------------------------------------------------------------
# 3D lazy multi-resolution split (z_axis / cubic)
# ---------------------------------------------------------------------------
def split_views_native_3d(
    image: torch.Tensor,
    label: torch.Tensor,
    wmap: Optional[torch.Tensor],
    target_patch_size: Tuple[int, int, int],
    mr_native_sizes: List[Tuple[int, int, int]],
    patch_size: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """从 ``(B,1,eD_max,eH_max,eW_max)`` max-FOV cube 逐视图中心裁 + resize 到
    ``patch_size``，结果 stack 成 ``(B,n_views,pD,pH,pW)``。
    label 走 nearest，img/wmap 走 trilinear（与旧 False-path 等价）。
    """
    if image.ndim != 5 or image.shape[1] != 1:
        raise ValueError(
            "native-3d split expects (B, 1, eD_max, eH_max, eW_max); "
            f"got image.shape={tuple(image.shape)}")
    if (label.shape[:2] != image.shape[:2]
            or label.shape[2:] != image.shape[2:]):
        raise ValueError(
            "image / label shape mismatch: "
            f"image={tuple(image.shape)}, label={tuple(label.shape)}")
    _, _, tD, tH, tW = image.shape
    if (tD, tH, tW) != tuple(target_patch_size):
        raise ValueError(
            f"native-3d split expects spatial dims == target_patch_size"
            f"={target_patch_size}; got {(tD, tH, tW)}. The post-augment "
            "center crop should already have removed the augment "
            "oversample margin.")

    pD, pH, pW = (int(x) for x in patch_size)

    def _ccrop(t: torch.Tensor, sizes: Tuple[int, int, int]) -> torch.Tensor:
        d_k, h_k, w_k = sizes
        d0 = (tD - d_k) // 2
        h0 = (tH - h_k) // 2
        w0 = (tW - w_k) // 2
        return t[:, :, d0:d0 + d_k, h0:h0 + h_k, w0:w0 + w_k]

    img_views : List[torch.Tensor] = []
    lbl_views : List[torch.Tensor] = []
    wmap_views: List[torch.Tensor] = []
    for sizes in mr_native_sizes:
        img_k  = _ccrop(image, sizes)
        lbl_k  = _ccrop(label, sizes)
        wmap_k = (_ccrop(wmap, sizes) if wmap is not None else None)
        # view 0 / 重合轴跳过 interpolate。
        if sizes != (pD, pH, pW):
            img_k = F.interpolate(
                img_k, size=(pD, pH, pW), mode="trilinear", align_corners=False)
            lbl_k = F.interpolate(lbl_k, size=(pD, pH, pW), mode="nearest")
            if wmap_k is not None:
                wmap_k = F.interpolate(wmap_k, size=(pD, pH, pW), mode="nearest")
        # 每 view 贡献 1 个通道。
        img_views.append(img_k.squeeze(1))
        lbl_views.append(lbl_k.squeeze(1))
        if wmap_k is not None:
            wmap_views.append(wmap_k.squeeze(1))

    image_out = torch.stack(img_views, dim=1).contiguous()
    label_out = torch.stack(lbl_views, dim=1).contiguous()
    wmap_out: Optional[torch.Tensor] = None
    if wmap_views:
        wmap_out = torch.stack(wmap_views, dim=1).contiguous()
    return image_out, label_out, wmap_out


# ---------------------------------------------------------------------------
# 2.5D lazy multi-resolution split (keep_native_view_depth)
# ---------------------------------------------------------------------------
def split_views_native_d(
    image            : torch.Tensor,
    label            : torch.Tensor,
    wmap             : Optional[torch.Tensor],
    per_view_depths  : List[int],
    target_patch_size: Tuple[int, int, int]) -> Tuple[
    torch.Tensor, torch.Tensor, Optional[torch.Tensor],
    List[torch.Tensor], List[Optional[torch.Tensor]]]:
    """``(B,1,eD_max,H,W)`` 逐视图中心抽 ``D_k`` 切片。

    返回 ``(image_2d, label_main, wmap_main, aux_labels, aux_wmaps)``：
    ``image_2d`` 通道维 = ``Σ D_k``，view 0 居前；aux 用 list 带回（深度可异）。
    """
    if image.ndim != 5 or image.shape[1] != 1:
        raise ValueError(
            "native-d split expects (B, 1, eD_max, H, W); got "
            f"image.shape={tuple(image.shape)}")
    if label.shape[:2] != image.shape[:2] or label.shape[2:] != image.shape[2:]:
        raise ValueError(
            "image / label shape mismatch: "
            f"image={tuple(image.shape)}, label={tuple(label.shape)}")

    _, _, eD_max, _, _ = image.shape
    depths             = per_view_depths
    D                  = depths[0]
    if eD_max != int(target_patch_size[0]):
        raise ValueError(
            f"native-d split expects depth axis == target_patch_size[0]"
            f"={target_patch_size[0]}; got {eD_max}. The post-augment "
            "center crop should already have removed the augment "
            "oversample margin.")
    if max(depths) > eD_max:
        raise ValueError(
            f"max(per_view_depths)={max(depths)} exceeds eD_max={eD_max}; "
            "this indicates a multi_res_scales / patch_size mismatch.")

    # 注意：切片保持 view 不做 contiguous —— 图像路随后的 cat 本身就产出
    # 连续新张量，逐视图先拷一次纯属多余；label/wmap 路生命周期与源 cube
    # 相同（同一 step 内消费），非连续 view 对损失计算无影响。
    def _slab(t: torch.Tensor, d_k: int) -> torch.Tensor:
        d0 = (eD_max - d_k) // 2
        return t[:, 0, d0:d0 + d_k]  # (B, d_k, H, W) view

    image_main = _slab(image, D)
    label_main = _slab(label, D)
    wmap_main  = _slab(wmap, D) if wmap is not None else None

    aux_imgs  : List[torch.Tensor]           = []
    aux_labels: List[torch.Tensor]           = []
    aux_wmaps : List[Optional[torch.Tensor]] = []
    for d_k in depths[1:]:
        aux_imgs.append(_slab(image, d_k))
        aux_labels.append(_slab(label, d_k))
        aux_wmaps.append(_slab(wmap, d_k) if wmap is not None else None)

    if aux_imgs:
        image_2d = torch.cat([image_main] + aux_imgs, dim=1).contiguous()
    else:
        image_2d = image_main.contiguous()
    expected_in = sum(depths)
    if image_2d.shape[1] != expected_in:
        raise RuntimeError(
            f"native-d split produced {image_2d.shape[1]} input "
            f"channels; expected sum(depths)={expected_in}.")
    return image_2d, label_main, wmap_main, aux_labels, aux_wmaps


# ---------------------------------------------------------------------------
# 2.5D lazy multi-resolution split (folded: per-view z-resize back to D)
# ---------------------------------------------------------------------------
def split_views_2_5d_folded(
    image            : torch.Tensor,
    label            : torch.Tensor,
    wmap             : Optional[torch.Tensor],
    per_view_depths  : List[int],
    target_patch_size: Tuple[int, int, int]) -> Tuple[
    torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """``(B,1,eD_max,H,W)`` 逐视图中心裁 ``D_k`` 切片后 D 轴 resize 回 ``D``，
    stack 成 ``(B,n_views,D,H,W)``（folded 布局，供 ``squeeze_2_5d`` 折叠）。

    与推理侧 ``build_z_window_cpu_multi_res`` / GPU builder 同一几何约定：
    img/wmap 走 trilinear，label 走 nearest。
    """
    if image.ndim != 5 or image.shape[1] != 1:
        raise ValueError(
            "2.5D folded split expects (B, 1, eD_max, H, W); got "
            f"image.shape={tuple(image.shape)}")
    if label.shape[:2] != image.shape[:2] or label.shape[2:] != image.shape[2:]:
        raise ValueError(
            "image / label shape mismatch: "
            f"image={tuple(image.shape)}, label={tuple(label.shape)}")

    _, _, eD_max, H, W = image.shape
    depths = per_view_depths
    D      = depths[0]
    if eD_max != int(target_patch_size[0]):
        raise ValueError(
            f"2.5D folded split expects depth axis == target_patch_size[0]"
            f"={target_patch_size[0]}; got {eD_max}. The post-augment "
            "center crop should already have removed the augment "
            "oversample margin.")
    if max(depths) > eD_max:
        raise ValueError(
            f"max(per_view_depths)={max(depths)} exceeds eD_max={eD_max}; "
            "this indicates a multi_res_scales / patch_size mismatch.")

    def _view(t: torch.Tensor, d_k: int, is_label: bool) -> torch.Tensor:
        d0 = (eD_max - d_k) // 2
        v  = t[:, :, d0:d0 + d_k]                    # (B, 1, d_k, H, W)
        if d_k != D:
            v = F.interpolate(
                v, size=(D, H, W),
                **({"mode": "nearest"} if is_label else
                   {"mode": "trilinear", "align_corners": False}))
        return v.squeeze(1)                          # (B, D, H, W)

    img_views  = [_view(image, d_k, False) for d_k in depths]
    lbl_views  = [_view(label, d_k, True) for d_k in depths]
    image_out  = torch.stack(img_views, dim=1).contiguous()
    label_out  = torch.stack(lbl_views, dim=1).contiguous()
    wmap_out: Optional[torch.Tensor] = None
    if wmap is not None:
        wmap_out = torch.stack(
            [_view(wmap, d_k, True) for d_k in depths], dim=1).contiguous()
    return image_out, label_out, wmap_out


# ---------------------------------------------------------------------------
# 2.5D fold ops —— 已上提 taskcore.engine.views（折叠契约见该模块 docstring），
# 此处 re-export 保留旧路径。
# ---------------------------------------------------------------------------

__all__ = [
    "center_crop",
    "split_views_native_3d",
    "split_views_native_d",
    "split_views_2_5d_folded",
    "squeeze_2_5d",
    "squeeze_2_5d_keep_views"]
