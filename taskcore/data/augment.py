"""GPU 3D 数据增强（分割 / 生成 / 分类共用）。

空间变换（flip/affine/elastic/grid-dropout）逐样本独立，并同步作用于
**伴随张量**（:class:`Companion`）：label / weight_map / cond 等声明插值模式
与越界填充后，由同一份 warp 消化（MONAI / TorchIO 式）。

强度变换仅作用于 image。``weight_map`` 插值由 ``AugConfig.wmap_interp_mode`` 控制。

同步点约束：选样 Bernoulli 掩码与逐样本标量参数均在 CPU 上采样
（``_bernoulli_mask``），再异步搬到设备；避免对 CUDA RNG 结果的隐式
device→host 同步打断流水。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange

from ..config import AugConfig


def _bernoulli_mask(
    n: int, prob: float,
    gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """CPU 上采样 (n,) bool 选样掩码；后续 any/sum/nonzero 均零同步。"""
    return torch.rand(n, generator=gen) < prob


@dataclass
class Companion:
    """随 image 同空间变换的伴随张量。

    * ``mode``：``grid_sample`` 插值（label=nearest，image/cond=bilinear）
    * ``oob_fill``：仿射/弹性越界体素填充；``None`` 表示不覆盖（保留 border）
    """

    tensor: torch.Tensor
    mode: str = "bilinear"
    oob_fill: Optional[float] = None


class GPUAugmentor:
    """GPU 3D 增强管道。

    ``max_scale`` 缩小 elastic_deform_alpha，使最大物理通道位移 ≤ alpha 体素。
    ``label_fill`` 为 label 越界填充（通常 = ``label_values[0]``）。
    """

    def __init__(self, cfg: AugConfig, max_scale: float = 1.0,
                 label_fill: float = 0.0,
                 seed: Optional[int] = None,
                 inplace: Optional[bool] = None):
        self.cfg       = cfg
        self.enabled   = cfg.enabled
        self.max_scale = max(float(max_scale), 1.0)
        self.label_fill = float(label_fill)
        # inplace 覆写：调用方在自身拥有输入张量所有权时（如训练循环的 H2D
        # 私有拷贝）可显式传 True 跳过入口 clone；None 时沿用 cfg.inplace。
        self.inplace = bool(cfg.inplace if inplace is None else inplace)
        # 独立随机流：seed 非 None 时创建专属 CPU/设备 Generator，增强采样与
        # 训练循环的全局 RNG 解耦（固定 seed 等价性验证的前置）；None 时
        # 沿用全局 RNG，行为与历史一致。设备端 Generator 惰性按首次输入
        # 设备创建（弹性位移/grid-dropout 在设备上采样）。
        self._seed: Optional[int] = None if seed is None else int(seed)
        self._gen_cpu: Optional[torch.Generator] = None
        self._gen_dev: Optional[torch.Generator] = None
        # resume 时设备端 Generator 尚未惰性创建，状态先挂起。
        self._gen_dev_pending_state: Optional[torch.Tensor] = None
        if self._seed is not None:
            self._gen_cpu = torch.Generator().manual_seed(self._seed)
        # wmap interp：'nearest' 保留离散权重（默认）；'bilinear' 适连续。
        wmode = cfg.wmap_interp_mode
        if wmode not in ("nearest", "bilinear"):
            raise ValueError(
                f"AugConfig.wmap_interp_mode={wmode!r}; expected "
                "'nearest' or 'bilinear'.")
        self.wmap_interp_mode = wmode

    def _device_generator(
        self, device: torch.device) -> Optional[torch.Generator]:
        """返回与输入设备匹配的专属 Generator；未启用独立流时返 None。"""
        if self._seed is None:
            return None
        if device.type == "cpu":
            return self._gen_cpu
        if self._gen_dev is None or self._gen_dev.device != device:
            self._gen_dev = torch.Generator(device=device)
            self._gen_dev.manual_seed(self._seed + 1)
            if self._gen_dev_pending_state is not None:
                self._gen_dev.set_state(self._gen_dev_pending_state)
                self._gen_dev_pending_state = None
        return self._gen_dev

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """私有增强 RNG 状态快照（入 checkpoint 支持位精确 resume）。

        ``seed=None``（沿用全局 RNG）时返回空 dict：全局 RNG 状态由训练循环
        自己快照。"""
        if self._seed is None or self._gen_cpu is None:
            return {}
        state: Dict[str, torch.Tensor] = {
            "gen_cpu": self._gen_cpu.get_state()}
        if self._gen_dev is not None:
            state["gen_dev"] = self._gen_dev.get_state()
        elif self._gen_dev_pending_state is not None:
            state["gen_dev"] = self._gen_dev_pending_state
        return state

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        """恢复私有增强 RNG 状态；设备端 Generator 惰性创建，状态先挂起，
        首次拿到实际设备时再灌入。空 dict（旧 ckpt / 全局 RNG 模式）为 no-op。"""
        if not state or self._seed is None or self._gen_cpu is None:
            return
        self._gen_cpu.set_state(state["gen_cpu"])
        if "gen_dev" in state:
            # get_state/set_state 均以 CPU ByteTensor 交互，无需跨设备搬运。
            if self._gen_dev is not None:
                self._gen_dev.set_state(state["gen_dev"])
            else:
                self._gen_dev_pending_state = state["gen_dev"]

    def apply(
        self,
        image: torch.Tensor,
        companions: Optional[Sequence[Companion]] = None,
    ) -> Tuple[torch.Tensor, List[Companion]]:
        """对 ``image`` + companions 施加完整增强管线。"""
        comps: List[Companion] = list(companions or [])
        if not self.enabled:
            return image, comps

        if not self.inplace:
            image = image.clone()
            comps = [
                Companion(c.tensor.clone(), c.mode, c.oob_fill) for c in comps]

        c = self.cfg
        gen_cpu = self._gen_cpu
        gen_dev = self._device_generator(image.device)

        # 开 intensity_clamp 时在任何增强前记录逐样本逐通道 min/max，
        # 确保基准是真正的"增强前"范围（不受 border 复制/dropout 置零影响）。
        if c.intensity_clamp:
            reduce_dims = tuple(range(2, image.ndim))
            clamp_lo = image.amin(dim=reduce_dims, keepdim=True)
            clamp_hi = image.amax(dim=reduce_dims, keepdim=True)

        image, comps = _random_flip_companions(
            image, c.random_flip_prob, c.random_flip_axes,
            companions=comps, gen_cpu=gen_cpu)
        effective_alpha = c.elastic_deform_alpha / self.max_scale
        image, comps = _random_affine_elastic_companions(
            image,
            affine_prob=c.random_affine_prob,
            rotate_range=c.random_rotate_range,
            scale_range=c.random_scale_range,
            elastic_prob=c.elastic_deform_prob,
            sigma=c.elastic_deform_sigma,
            alpha=effective_alpha,
            companions=comps,
            translate_range=c.random_translate_range,
            rotate_range_per_axis=c.random_rotate_range_per_axis,
            aspect_correct=c.random_affine_aspect_correct,
            elastic_field_mode=c.elastic_field_mode,
            elastic_normalize_displacement=c.elastic_normalize_displacement,
            gen_cpu=gen_cpu, gen_dev=gen_dev)

        # Intensity (image only)。全部强度增强后夹回增强前范围。
        image = _random_brightness(
            image, c.random_brightness_prob, c.random_brightness_range,
            gen_cpu=gen_cpu)
        image = _random_contrast(
            image, c.random_contrast_prob, c.random_contrast_range,
            gen_cpu=gen_cpu)
        image = _random_gamma(
            image, c.random_gamma_prob, c.random_gamma_range, gen_cpu=gen_cpu)
        image = _gaussian_noise(
            image, c.gaussian_noise_prob, c.gaussian_noise_std,
            gen_cpu=gen_cpu, gen_dev=gen_dev)
        image = _gaussian_blur_3d(
            image, c.gaussian_blur_prob, c.gaussian_blur_sigma,
            gen_cpu=gen_cpu)
        image = _simulate_lowres(
            image, c.simulate_lowres_prob, c.simulate_lowres_zoom,
            gen_cpu=gen_cpu)
        if c.intensity_clamp:
            image = torch.maximum(torch.minimum(image, clamp_hi), clamp_lo)

        # grid_dropout 必须在 clamp 之后：否则洞被置 0 后会被 clamp 抬回
        # clamp_lo（软组织窗 / zscore 下常见），dropout 静默失效。
        image, comps = _grid_dropout_companions(
            image, c.grid_dropout_prob, c.grid_dropout_ratio,
            c.grid_dropout_holes, companions=comps,
            gen_cpu=gen_cpu, gen_dev=gen_dev)

        return image, comps

    def __call__(
        self, image: torch.Tensor, label: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """分割/分类入口：``(image, label[, weight_map])`` → 同结构。

        提供 weight_map 时越界语义为「label 保 border 复制 + wmap 置 0 精确
        排除」——把"不知道这里是什么"编码进损失权重而非伪造背景标签；无
        weight_map 时（cls 等）沿用 label 填 ``label_fill`` 的旧语义。"""
        if weight_map is not None:
            comps: List[Companion] = [
                Companion(label, mode="nearest", oob_fill=None),
                Companion(weight_map, mode=self.wmap_interp_mode,
                          oob_fill=0.0)]
        else:
            comps = [
                Companion(label, mode="nearest", oob_fill=self.label_fill)]
        image, comps = self.apply(image, comps)
        label_out = comps[0].tensor
        wmap_out = comps[1].tensor if len(comps) > 1 else None
        return image, label_out, wmap_out


# 空间增强（逐样本独立）。
def _random_flip_companions(
    image: torch.Tensor,
    prob: float, axes: list,
    companions: Optional[Sequence[Companion]] = None,
    gen_cpu: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, List[Companion]]:
    """逐样本随机翻转；image 与全部 companions 同步。"""
    comps = list(companions or [])
    if prob <= 0 or not axes:
        return image, comps
    B = image.shape[0]
    for axis in axes:
        mask = _bernoulli_mask(B, prob, gen_cpu)
        if mask.any():
            idx = mask.nonzero(as_tuple=True)[0].to(
                image.device, non_blocking=True)
            image[idx] = torch.flip(image[idx], [axis])
            for c in comps:
                c.tensor[idx] = torch.flip(c.tensor[idx], [axis])
    return image, comps


def _random_flip(
    image: torch.Tensor, label: torch.Tensor,
    prob: float, axes: list,
    weight_map: Optional[torch.Tensor] = None,
    gen_cpu: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """逐样本随机翻转（分割旧签名包装；仅测试消费，生产路径走 *_companions）。"""
    comps: List[Companion] = [Companion(label, "nearest", None)]
    if weight_map is not None:
        comps.append(Companion(weight_map, "nearest", None))
    image, comps = _random_flip_companions(
        image, prob, axes, companions=comps, gen_cpu=gen_cpu)
    wmap = comps[1].tensor if len(comps) > 1 else None
    return image, comps[0].tensor, wmap


def _build_rotation_matrices(
    angles: torch.Tensor, scales: torch.Tensor,
    translations: Optional[torch.Tensor] = None,
    aspect: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """从欧拉角（N×3 rad）与 scales（N×1）构建 (N,3,4) 仿射矩阵。

    translations：(N,3) 归一化平移；None=无平移。
    aspect：(3,) 各轴尺度比例（轴序 (x,y,z)=(W,H,D)）；非 None 时对旋转
    做共轭校正 R←A⁻¹RA，使旋转在 voxel-count 各向同性坐标里进行（各同性 scale
    与对角阵可交换，不受影响）。这不代替真实 spacing 校正。
    """
    N = angles.shape[0]
    device = angles.device

    cx, cy, cz = angles[:, 0].cos(), angles[:, 1].cos(), angles[:, 2].cos()
    sx, sy, sz = angles[:, 0].sin(), angles[:, 1].sin(), angles[:, 2].sin()

    # R = Rz @ Ry @ Rx。
    r00 = cy * cz
    r01 = sx * sy * cz - cx * sz
    r02 = cx * sy * cz + sx * sz
    r10 = cy * sz
    r11 = sx * sy * sz + cx * cz
    r12 = cx * sy * sz - sx * cz
    r20 = -sy
    r21 = sx * cy
    r22 = cx * cy

    rot = torch.stack([
        r00, r01, r02,
        r10, r11, r12,
        r20, r21, r22,
    ], dim=-1)
    rot = rearrange(rot, 'n (r c) -> n r c', r=3, c=3)

    if aspect is not None:
        a = aspect.to(device=device, dtype=rot.dtype)
        rot = torch.diag(1.0 / a) @ rot @ torch.diag(a)

    m = scales.view(N, 1, 1) * rot  # (N,3,3)

    if translations is None:
        t = torch.zeros(N, 3, 1, device=device, dtype=m.dtype)
    else:
        t = translations.view(N, 3, 1).to(m.dtype)

    return torch.cat([m, t], dim=-1)  # (N,3,4)


def _elastic_grid_disp(
    n: int, D: int, H: int, W: int,
    sigma: float, alpha: float, device: torch.device,
    gen_dev: Optional[torch.Generator] = None,
    field_mode: str = "legacy",
    normalize_displacement: bool = False) -> torch.Tensor:
    """采样 n 个弹性位移场，返 (n,D,H,W,3) 归一化 grid 坐标位移（轴序 W,H,D）。"""
    cD = max(int(round(D / sigma)), 4)
    cH = max(int(round(H / sigma)), 4)
    cW = max(int(round(W / sigma)), 4)
    disp = torch.randn(n, 3, cD, cH, cW, device=device, generator=gen_dev)
    disp = F.interpolate(disp, size=(D, H, W), mode="trilinear", align_corners=False)
    if field_mode == "gaussian":
        radius = max(1, int(round(sigma)))
        coords = torch.arange(
            -radius, radius + 1, device=device, dtype=disp.dtype)
        kernel = torch.exp(-0.5 * (coords / max(float(sigma), 1e-6)) ** 2)
        kernel = kernel / kernel.sum()
        kd = kernel.view(1, 1, -1, 1, 1)
        kh = kernel.view(1, 1, 1, -1, 1)
        kw = kernel.view(1, 1, 1, 1, -1)
        disp = F.conv3d(disp, kd.expand(3, 1, -1, 1, 1),
                        padding=(radius, 0, 0), groups=3)
        disp = F.conv3d(disp, kw.expand(3, 1, 1, 1, -1),
                        padding=(0, 0, radius), groups=3)
        disp = F.conv3d(disp, kh.expand(3, 1, 1, -1, 1),
                        padding=(0, radius, 0), groups=3)
    if normalize_displacement:
        rms = disp.square().mean(dim=(1, 2, 3, 4), keepdim=True).sqrt()
        disp = disp / rms.clamp_min(torch.finfo(disp.dtype).eps)

    # 体素位移→归一化 grid 坐标（1 voxel = 2/N）；permute 后通道 (0,1,2) 对应 grid 轴 (W,H,D)。
    voxel_to_grid = rearrange(
        torch.tensor([2.0 / W, 2.0 / H, 2.0 / D],
                     dtype=disp.dtype, device=device),
        'c -> 1 c 1 1 1')
    disp = disp * alpha * voxel_to_grid
    return rearrange(disp, 'b c d h w -> b d h w c')


def _random_affine_elastic_companions(
    image: torch.Tensor,
    affine_prob: float, rotate_range: list, scale_range: list,
    elastic_prob: float, sigma: float, alpha: float,
    companions: Optional[Sequence[Companion]] = None,
    translate_range: Optional[list] = None,
    rotate_range_per_axis: Optional[list] = None,
    aspect_correct: bool = False,
    gen_cpu: Optional[torch.Generator] = None,
    gen_dev: Optional[torch.Generator] = None,
    elastic_field_mode: str = "legacy",
    elastic_normalize_displacement: bool = False,
) -> Tuple[torch.Tensor, List[Companion]]:
    """仿射与弹性形变融合为单次 grid_sample；companions 共享同一 warp。

    两路选中掩码仍逐样本独立采样；对同时选中的样本，合成采样网格
    G(x) = Θ(x + d(x)) = affine_grid + M·d（M 为 Θ 的 3×3 线性部分）。

    image 用 padding_mode='border'；companion 的 ``oob_fill`` 非 None 时，
    越界体素覆写为该常数（label→背景、seg wmap→1.0；gen wmap/cond 为 None）。
    """
    comps = list(companions or [])
    B, _, D, H, W = image.shape
    device = image.device

    mask_a = _bernoulli_mask(B, affine_prob, gen_cpu)
    mask_e = _bernoulli_mask(B, elastic_prob, gen_cpu)
    mask = mask_a | mask_e
    if not mask.any():
        return image, comps

    idx_cpu = mask.nonzero(as_tuple=True)[0]
    idx = idx_cpu.to(device, non_blocking=True)
    n = idx_cpu.shape[0]
    sel_a = mask_a[idx_cpu]
    sel_e = mask_e[idx_cpu]

    theta = torch.eye(3, 4, device=device).unsqueeze(0).repeat(n, 1, 1)
    na = int(sel_a.sum())
    if na:
        if rotate_range_per_axis is not None:
            angles = torch.empty(na, 3)
            for ax in range(3):
                lo = math.radians(rotate_range_per_axis[ax][0])
                hi = math.radians(rotate_range_per_axis[ax][1])
                angles[:, ax].uniform_(lo, hi, generator=gen_cpu)
        else:
            lo, hi = math.radians(rotate_range[0]), math.radians(rotate_range[1])
            angles = torch.empty(na, 3).uniform_(lo, hi, generator=gen_cpu)
        angles = angles.to(device, non_blocking=True)
        scales = torch.empty(na, 1).uniform_(
            scale_range[0], scale_range[1],
            generator=gen_cpu).to(device, non_blocking=True)

        translations = None
        if translate_range is not None and (
                translate_range[0] != 0.0 or translate_range[1] != 0.0):
            translations = torch.empty(na, 3).uniform_(
                translate_range[0], translate_range[1],
                generator=gen_cpu).to(device, non_blocking=True)

        aspect = None
        if aspect_correct:
            aspect = torch.tensor(
                [float(W), float(H), float(D)],
                device=device, dtype=torch.float32)
        theta[sel_a.to(device, non_blocking=True)] = _build_rotation_matrices(
            angles, scales, translations, aspect)

    grid = F.affine_grid(theta, [n, 1, D, H, W], align_corners=False)

    ne = int(sel_e.sum())
    if ne:
        sel_e_dev = sel_e.to(device, non_blocking=True)
        disp = _elastic_grid_disp(
            ne, D, H, W, sigma, alpha, device, gen_dev,
            field_mode=elastic_field_mode,
            normalize_displacement=elastic_normalize_displacement)
        m = theta[sel_e_dev][:, :, :3]
        grid[sel_e_dev] = grid[sel_e_dev] + torch.einsum(
            'n r c, n d h w c -> n d h w r', m, disp)

    oob = (grid.abs() > 1.0).any(dim=-1)  # (n, D, H, W)

    image[idx] = F.grid_sample(
        image[idx], grid, mode="bilinear", padding_mode="border",
        align_corners=False)
    for c in comps:
        warped = F.grid_sample(
            c.tensor[idx], grid, mode=c.mode,
            padding_mode="border", align_corners=False)
        if c.oob_fill is not None:
            warped[oob.unsqueeze(1).expand_as(warped)] = c.oob_fill
        c.tensor[idx] = warped

    return image, comps


def _random_affine_elastic(
    image: torch.Tensor, label: torch.Tensor,
    affine_prob: float, rotate_range: list, scale_range: list,
    elastic_prob: float, sigma: float, alpha: float,
    weight_map: Optional[torch.Tensor] = None,
    wmap_mode: str = "nearest",
    translate_range: Optional[list] = None,
    rotate_range_per_axis: Optional[list] = None,
    aspect_correct: bool = False,
    label_fill: float = 0.0,
    gen_cpu: Optional[torch.Generator] = None,
    gen_dev: Optional[torch.Generator] = None,
    elastic_field_mode: str = "legacy",
    elastic_normalize_displacement: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """仿射+弹性（分割旧签名包装；仅测试消费，生产路径走 *_companions）：
    label nearest+oob_fill，wmap 填 1.0。"""
    comps: List[Companion] = [
        Companion(label, mode="nearest", oob_fill=float(label_fill))]
    if weight_map is not None:
        comps.append(Companion(
            weight_map, mode=wmap_mode, oob_fill=1.0))
    image, comps = _random_affine_elastic_companions(
        image,
        affine_prob=affine_prob, rotate_range=rotate_range,
        scale_range=scale_range, elastic_prob=elastic_prob,
        sigma=sigma, alpha=alpha, companions=comps,
        translate_range=translate_range,
        rotate_range_per_axis=rotate_range_per_axis,
        aspect_correct=aspect_correct,
        elastic_field_mode=elastic_field_mode,
        elastic_normalize_displacement=elastic_normalize_displacement,
        gen_cpu=gen_cpu, gen_dev=gen_dev)
    wmap = comps[1].tensor if len(comps) > 1 else None
    return image, comps[0].tensor, wmap


def _grid_dropout_companions(
    image: torch.Tensor,
    prob: float, ratio: float, num_holes: int,
    companions: Optional[Sequence[Companion]] = None,
    gen_cpu: Optional[torch.Generator] = None,
    gen_dev: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, List[Companion]]:
    """随机零掩 num_holes 个矩形区域；仅 image，companions 原样返回。"""
    comps = list(companions or [])
    if prob <= 0 or ratio <= 0:
        return image, comps

    B, _, D, H, W = image.shape
    device = image.device

    selected = _bernoulli_mask(B, prob, gen_cpu)
    if not selected.any():
        return image, comps

    frac = (ratio / max(num_holes, 1)) ** (1.0 / 3.0)
    hd = min(D, max(1, int(D * frac)))
    hh = min(H, max(1, int(H * frac)))
    hw = min(W, max(1, int(W * frac)))

    d0 = torch.randint(0, D - hd + 1, (B, num_holes), device=device,
                       generator=gen_dev)
    h0 = torch.randint(0, H - hh + 1, (B, num_holes), device=device,
                       generator=gen_dev)
    w0 = torch.randint(0, W - hw + 1, (B, num_holes), device=device,
                       generator=gen_dev)

    out = image.clone()
    d_off = torch.arange(hd, device=device)
    h_off = torch.arange(hh, device=device)
    w_off = torch.arange(hw, device=device)
    # nonzero 在 CPU 掌握的选样掩码上完成，避免设备端 nonzero 的 D2H 同步。
    b_selected = selected.nonzero(as_tuple=True)[0].to(
        device, non_blocking=True)
    for k in range(num_holes):
        ds = d0[b_selected, k, None] + d_off[None, :]
        hs = h0[b_selected, k, None] + h_off[None, :]
        ws = w0[b_selected, k, None] + w_off[None, :]
        out[
            b_selected[:, None, None, None], :,
            ds[:, :, None, None],
            hs[:, None, :, None],
            ws[:, None, None, :],
        ] = 0
    return out, comps


def _grid_dropout(
    image: torch.Tensor, label: torch.Tensor,
    prob: float, ratio: float, num_holes: int,
    weight_map: Optional[torch.Tensor] = None,
    gen_cpu: Optional[torch.Generator] = None,
    gen_dev: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """随机零掩（分割旧签名包装；仅测试消费，生产路径走 *_companions）；
    label/weight_map 不被掩。"""
    comps: List[Companion] = [Companion(label, "nearest", None)]
    if weight_map is not None:
        comps.append(Companion(weight_map, "nearest", None))
    image, comps = _grid_dropout_companions(
        image, prob, ratio, num_holes, companions=comps,
        gen_cpu=gen_cpu, gen_dev=gen_dev)
    wmap = comps[1].tensor if len(comps) > 1 else None
    return image, comps[0].tensor, wmap


# 强度增强（逐样本独立）。
def _random_brightness(
    image: torch.Tensor, prob: float, brange: list,
    gen_cpu: Optional[torch.Generator] = None) -> torch.Tensor:
    """逐样本随机加性亮度偏移。"""
    if prob <= 0:
        return image
    B = image.shape[0]
    mask = _bernoulli_mask(B, prob, gen_cpu)
    if not mask.any():
        return image
    shift = torch.empty(B, 1, 1, 1, 1).uniform_(
        brange[0], brange[1], generator=gen_cpu)
    shift[~mask] = 0
    return image + shift.to(image.device, non_blocking=True)


def _random_contrast(
    image: torch.Tensor, prob: float, crange: list,
    gen_cpu: Optional[torch.Generator] = None) -> torch.Tensor:
    """逐样本随机对比度，以逐通道均值为轴。"""
    if prob <= 0:
        return image
    B = image.shape[0]
    mask = _bernoulli_mask(B, prob, gen_cpu)
    if not mask.any():
        return image
    spatial_dims = tuple(range(2, image.ndim))
    mean = image.mean(dim=spatial_dims, keepdim=True)
    factor = torch.ones(B, 1, 1, 1, 1)
    factor[mask] = torch.empty(
        int(mask.sum()), 1, 1, 1, 1).uniform_(
            crange[0], crange[1], generator=gen_cpu)
    return (image - mean) * factor.to(image.device, non_blocking=True) + mean


def _random_gamma(
    image: torch.Tensor, prob: float, grange: list,
    gen_cpu: Optional[torch.Generator] = None) -> torch.Tensor:
    """逐样本随机 gamma：逐通道 minmax 归一→pow(gamma)→反归一。"""
    if prob <= 0:
        return image
    B = image.shape[0]
    mask = _bernoulli_mask(B, prob, gen_cpu)  # (B,) bool，CPU
    if not mask.any():
        return image

    # 仅在空间轴 reduce，通道独立（多分辨率安全）。
    reduce_dims = tuple(range(2, image.ndim))
    mn = image.amin(dim=reduce_dims, keepdim=True)  # (B,C,1,1,1)
    mx = image.amax(dim=reduce_dims, keepdim=True)
    rng = (mx - mn).clamp(min=1e-7)
    normed = ((image - mn) / rng).clamp(0.0, 1.0)

    # 未选中样本 gamma=1（CPU 采样后异步搬设备）。
    gamma = torch.empty(B).uniform_(grange[0], grange[1], generator=gen_cpu)
    gamma = torch.where(mask, gamma, torch.ones_like(gamma))
    gamma = gamma.to(image.device, non_blocking=True)
    # 动态阐 (B,) → (B, 1, 1, ..., 1) 以适应 image.ndim。
    gpattern = 'b -> b' + ' 1' * (image.ndim - 1)
    gamma = rearrange(gamma, gpattern).to(image.dtype)

    return normed.pow(gamma) * rng + mn


def _gaussian_noise(
    image: torch.Tensor, prob: float, std: float,
    gen_cpu: Optional[torch.Generator] = None,
    gen_dev: Optional[torch.Generator] = None) -> torch.Tensor:
    """逐样本加性高斯噪声。"""
    if prob <= 0:
        return image
    B = image.shape[0]
    mask = _bernoulli_mask(B, prob, gen_cpu)
    if not mask.any():
        return image
    idx = mask.nonzero(as_tuple=True)[0].to(image.device, non_blocking=True)
    sub = image[idx]
    noise = torch.randn(sub.shape, dtype=sub.dtype, device=sub.device,
                        generator=gen_dev)
    image[idx] = sub + noise * std
    return image


def _gaussian_blur_3d(
    image: torch.Tensor, prob: float, sigma_range: list,
    gen_cpu: Optional[torch.Generator] = None) -> torch.Tensor:
    """可分离 3D 高斯模糊；逐样本独立采样 sigma。

    全部选中样本一次性向量化处理：核长统一取 sigma 上界对应的 ks（小 sigma
    样本仅多出近零尾部，归一化后等价），逐样本核通过 grouped conv 并行。"""
    if prob <= 0:
        return image
    B, C = image.shape[:2]
    device = image.device
    mask = _bernoulli_mask(B, prob, gen_cpu)
    if not mask.any():
        return image

    idx = mask.nonzero(as_tuple=True)[0].to(device, non_blocking=True)
    n = idx.numel()
    sigmas = torch.empty(n, dtype=image.dtype).uniform_(
        sigma_range[0], sigma_range[1],
        generator=gen_cpu).to(device, non_blocking=True)  # (n,)
    ks = max(int(2 * round(3 * float(sigma_range[1])) + 1), 3)
    pad = ks // 2
    x = torch.arange(ks, dtype=image.dtype, device=device) - pad  # (ks,)
    k1d = torch.exp(-0.5 * (x[None, :] / sigmas[:, None]) ** 2)   # (n, ks)
    k1d = k1d / k1d.sum(dim=1, keepdim=True)
    kc = k1d.repeat_interleave(C, dim=0)  # (n*C, ks)，同样本各通道共用同核

    # 将 (n,C) 折入通道轴，groups=n*C 使每通道用自己的核。
    sub = rearrange(image[idx], 'n c d h w -> 1 (n c) d h w')
    nc = n * C
    for axis_pat, pad_arg in (
        ('g k -> g 1 k 1 1', [0, 0, 0, 0, pad, pad]),
        ('g k -> g 1 1 k 1', [0, 0, pad, pad, 0, 0]),
        ('g k -> g 1 1 1 k', [pad, pad, 0, 0, 0, 0]),
    ):
        k = rearrange(kc, axis_pat)
        sub = F.pad(sub, pad_arg, mode="replicate")
        sub = F.conv3d(sub, k, groups=nc)

    image[idx] = rearrange(sub, '1 (n c) d h w -> n c d h w', n=n)
    return image


def _simulate_lowres(
    image: torch.Tensor, prob: float, zoom_range: list,
    gen_cpu: Optional[torch.Generator] = None) -> torch.Tensor:
    """trilinear 下采→上采模拟低分辨率；逐样本独立采样 zoom。

    zoom 一次性批量采样（单次同步）；目标尺寸相同的样本分组后批量
    interpolate（不同尺寸无法单次处理，只能逐组）。"""
    if prob <= 0:
        return image
    B = image.shape[0]
    mask = _bernoulli_mask(B, prob, gen_cpu)
    if not mask.any():
        return image
    _, _, D, H, W = image.shape
    idxs = mask.nonzero(as_tuple=True)[0].tolist()
    zooms = torch.empty(len(idxs)).uniform_(
        zoom_range[0], zoom_range[1], generator=gen_cpu).tolist()
    groups: dict = {}
    for i, z in zip(idxs, zooms):
        if z >= 0.99:
            continue
        size = (max(1, int(D * z)), max(1, int(H * z)), max(1, int(W * z)))
        groups.setdefault(size, []).append(i)
    for size, members in groups.items():
        small = F.interpolate(
            image[members], size=size, mode="trilinear", align_corners=False)
        image[members] = F.interpolate(
            small, size=(D, H, W), mode="trilinear", align_corners=False)
    return image
