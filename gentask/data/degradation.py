"""生成任务的退化算子（GPU）。

当前仅实现 **超分（super-resolution）** 退化：把干净的高分图（HR）下采样到
``1/scale`` 再上采样回原尺寸，得到与 HR 同尺寸的低分图（LR），作为模型输入；
HR 本身作为重建目标（pre-upsampling SISR，类 SRCNN / VDSR 设定，输入输出同尺寸，
可直接复用编解码同尺寸的 U-Net）。下采样核（``sr_kernel``，默认 area ≈ 层内均值，
对应部分容积效应）与上采样核（``sr_kernel_up``，默认 trilinear，对应临床重建后
线性插值到目标层厚）分开配置。可选高斯噪声模拟采集噪声，噪声施加在 **LR 域**
（下采样后、上采样前），与真实采集链路一致（噪声经插值后呈空间相关）。

退化倍率支持 **各向同性**（标量 ``sr_scale`` 各轴同倍）或 **各向异性**
（``sr_scale_per_axis`` 逐空间轴单独配置）。后者用于 CT 「厚层→薄层」：仅 z 轴是
低分辨率，故只对 z 轴下采样/上采样（z-SISR），其余轴保持原分辨率（部分容积效应
下，厚层切片≈层厚范围内体素的平均，对应 ``area`` 核沿 z 的区域均值降采样）。

退化在 trainer 增强（augment + 中心裁）之后、于 GPU 上对一个 batch 施加，
故 ``degrade`` 接收 ``(B, C, *spatial)`` 的张量：

* 3D（``spatial_dims==3``）：tensor 形如 ``(B, 1, D, H, W)``，逐空间轴 ``(D, H, W)``
  按各自倍率下采样（``[s_z, 1, 1]`` 即只对 z 轴超分，``[1, s, s]`` 即面内超分）。
* 2.5D（``spatial_dims==2``）：tensor 形如 ``(B, D, H, W)``（D 折叠进通道轴），
  仅在 ``(H, W)`` 两个空间轴下采样（逐切片 2D 面内超分），与把 D 视作通道一致。

设计为策略对象，未来可在此扩展 denoise / inpaint 等其它退化而不改调用方。
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..config import TaskConfig

# F.interpolate 的可选插值模式（按空间维度区分）。
_SCALABLE_MODES = {
    2: {"trilinear": "bilinear", "area": "area", "nearest": "nearest"},
    3: {"trilinear": "trilinear", "area": "area", "nearest": "nearest"},
}

# CT 层敏感度剖面（SSP）核：层内响应非理想 box，近高斯 / 三角（M4）。
_SSP_KINDS = ("gauss", "tri")
_VALID_DOWN_KERNELS = tuple(sorted(
    set(_SCALABLE_MODES[3]) | set(_SSP_KINDS)))


def _interp_mode(sr_kernel: str, spatial_dims: int) -> str:
    """把配置里的 ``sr_kernel`` 映射到 ``F.interpolate`` 的 mode（按空间维度）。"""
    table = _SCALABLE_MODES[spatial_dims]
    if sr_kernel not in table:
        raise ValueError(
            f"Unknown sr_kernel {sr_kernel!r}; valid: {sorted(table)}.")
    return table[sr_kernel]


def _ssp_kernel_1d(
    kind: str, scale: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """1D SSP 核（归一化，奇长）。尺度以 HR 体素计，层厚 ≈ ``scale``。

    * ``gauss``：FWHM = 层厚 → σ = scale/2.3548，支撑 ±⌈ 3σ ⌉。
    * ``tri``：三角剖面，半宽 = 层厚，支撑 (-scale, scale)。
    """
    if kind == "gauss":
        sigma = scale / 2.3548200450309493  # FWHM → σ
        r = max(1, int(math.ceil(3.0 * sigma)))
        x = torch.arange(-r, r + 1, device=device, dtype=dtype)
        k = torch.exp(-0.5 * (x / sigma) ** 2)
    else:  # "tri"
        r = max(1, int(scale) - 1)
        x = torch.arange(-r, r + 1, device=device, dtype=dtype)
        k = 1.0 - x.abs() / float(scale)
    return k / k.sum()


def _ssp_downsample_axis(
    x: torch.Tensor, dim: int, scale: int, k: torch.Tensor) -> torch.Tensor:
    """沿 ``dim`` 用 1D SSP 核平滑 + 按 ``scale`` 抽样（边界 replicate）。

    输出长度 = round(n/scale)，输出 j 的窗口中心 ≈ ``j*scale + (scale-1)//2``
    （与 area 降采样的窗口中心 ``(j+0.5)*scale-0.5`` 对齐，偶倍率差 0.5 体素，
    已被平滑核支撑覆盖）。"""
    n = x.shape[dim]
    low = max(int(round(n / scale)), 1)
    length = k.numel()
    center = (length - 1) // 2
    pad_l = center - (scale - 1) // 2
    pad_r = (low - 1) * scale + length - n - pad_l
    xt = x.movedim(dim, -1)
    lead = xt.shape[:-1]
    xt = xt.reshape(-1, 1, n)
    xt = F.pad(xt, (max(pad_l, 0), max(pad_r, 0)), mode="replicate")
    if pad_l < 0 or pad_r < 0:  # 核支撑小于步长时裁掉多余输入
        xt = xt[..., max(-pad_l, 0): xt.shape[-1] - max(-pad_r, 0)]
    y = F.conv1d(xt, k.view(1, 1, -1).to(xt.dtype), stride=int(scale))
    return y.reshape(*lead, low).movedim(-1, dim)


class SuperResDegradation:
    """超分退化：HR → LR（同尺寸，已上采样回原大小）。

    倍率可按轴配置：``axis_scales`` 为长度 ``spatial_dims`` 的逐轴倍率（为
    ``None`` 时退化为各轴同倍的标量 ``scale``）。某轴倍率为 1 表示该轴不退化。

    ``sampling`` 选退化方式：

    * ``"blur"``（SISR）：下采样到 ``1/scale``（``kernel``，默认 area）再用
      ``kernel_up``（默认 trilinear）上采样回原尺寸，造成同尺寸模糊。
      下采样尺寸由原尺寸除以倍率取整，上采样显式还原原尺寸避免 ±1 偏差。
    * ``"decimate"``（VFI 插帧）：沿退化轴按 ``scale`` 抽稀保留帧（``[::sc]``），再用
      相位对齐的逐轴线性插值填回原尺寸：保留帧在 LR 中逐体素精确保留
      （``lr[k*sc] == hr[k*sc]``），仅在保留帧之间插出中间片，末尾越界复制最后
      一帧。对应「取稀疏厚层切片、模型补足中间薄层」的帧插值设定
      （线性插值作为天真 baseline，模型学残差）。

    ``keep_lr_size=True``（post-upsampling SISR，如 EDSR/RCAN）：省去「上采回
    HR 尺寸」步骤，直接返回真 LR 尺寸的张量（blur 为降采样结果，decimate 为
    保留帧），由网络的上采头把 LR 放大回 HR 网格。

    下采核除 F.interpolate 模式外支持 CT 层敏感度剖面（SSP）核
    ``'gauss'`` / ``'tri'``（M4，仅 blur 模式）：沿退化轴分离地用 1D SSP 核
    平滑后抽样，比 box 均值更贴近真实厚层重建。

    随机退化池（M6，Real-ESRGAN 风格的轻量版）：``kernel_pool`` 非空时每次
    ``degrade`` 随机抽一个下采核；``noise_std_range=(lo,hi)`` 非空时噪声 std
    每次均匀采样（覆盖 ``noise_std``）。仅在梯度开启时（训练前向）随机；
    验证/推理（``torch.no_grad``）固定用基础 ``kernel`` / ``noise_std``，
    保证指标可比。
    """

    def __init__(
        self,
        scale: int,
        spatial_dims: int,
        kernel: str = "area",
        kernel_up: str = "trilinear",
        noise_std: float = 0.0,
        axis_scales: Optional[Sequence[int]] = None,
        sampling: str = "blur",
        keep_lr_size: bool = False,
        kernel_pool: Optional[Sequence[str]] = None,
        noise_std_range: Optional[Sequence[float]] = None):
        if spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3; got {spatial_dims}.")
        sampling = str(sampling).lower()
        if sampling not in ("blur", "decimate"):
            raise ValueError(
                f"sampling must be 'blur' | 'decimate'; got {sampling!r}.")
        if axis_scales is None:
            if scale < 1:
                raise ValueError(f"sr_scale must be >= 1; got {scale}.")
            axes = (int(scale),) * spatial_dims
        else:
            axes = tuple(int(s) for s in axis_scales)
            if len(axes) != spatial_dims:
                raise ValueError(
                    f"axis_scales length ({len(axes)}) must equal "
                    f"spatial_dims ({spatial_dims}); got {axes}.")
            if any(s < 1 for s in axes):
                raise ValueError(f"each axis scale must be >= 1; got {axes}.")
        self.axis_scales = axes
        self.scale = max(axes)
        self.spatial_dims = int(spatial_dims)
        self.sampling = sampling
        self.kernel = str(kernel).lower()
        self._check_down_kernel(self.kernel)
        self.mode_up = _interp_mode(kernel_up, spatial_dims)
        self.noise_std = float(noise_std)
        self.kernel_pool: List[str] = [
            str(k).lower() for k in (kernel_pool or [])]
        for k in self.kernel_pool:
            self._check_down_kernel(k)
        if noise_std_range:
            lo, hi = (float(noise_std_range[0]), float(noise_std_range[1]))
            if not (0.0 <= lo <= hi):
                raise ValueError(
                    f"noise_std_range must satisfy 0 <= lo <= hi; got "
                    f"({lo}, {hi}).")
            self.noise_std_range: Optional[Tuple[float, float]] = (lo, hi)
        else:
            self.noise_std_range = None
        self._align_up = None if self.mode_up in ("area", "nearest") else False
        self._is_identity = all(s == 1 for s in axes)
        self.keep_lr_size = bool(keep_lr_size)
        self._ssp_cache: Dict[Tuple[str, int, torch.device, torch.dtype],
                              torch.Tensor] = {}

    @staticmethod
    def _check_down_kernel(kernel: str) -> None:
        if kernel not in _VALID_DOWN_KERNELS:
            raise ValueError(
                f"Unknown sr_kernel {kernel!r}; valid: "
                f"{list(_VALID_DOWN_KERNELS)}.")

    def _sample_params(self) -> Tuple[str, float]:
        """本次 degrade 的 (下采核, 噪声 std)。随机池仅在梯度开启时生效。"""
        kernel, std = self.kernel, self.noise_std
        if torch.is_grad_enabled():
            if self.kernel_pool:
                kernel = random.choice(self.kernel_pool)
            if self.noise_std_range is not None:
                std = random.uniform(*self.noise_std_range)
        return kernel, std

    def _ssp_k(
        self, kind: str, scale: int, device: torch.device,
        dtype: torch.dtype) -> torch.Tensor:
        key = (kind, int(scale), device, dtype)
        k = self._ssp_cache.get(key)
        if k is None:
            k = _ssp_kernel_1d(kind, int(scale), device, dtype)
            self._ssp_cache[key] = k
        return k

    def degrade(self, hr: torch.Tensor) -> torch.Tensor:
        """从干净 HR 生成 LR。``hr`` 形如 ``(B, C, *spatial)``。"""
        kernel, noise_std = self._sample_params()
        if self._is_identity and noise_std == 0.0:
            return hr.clone()

        spatial = tuple(int(s) for s in hr.shape[-self.spatial_dims:])
        if self._is_identity:
            return hr + torch.randn_like(hr) * noise_std

        if self.sampling == "decimate":
            return self._decimate_interp(hr, spatial, noise_std)

        if kernel in _SSP_KINDS:  # SSP 平滑 + 抽样（逐退化轴分离）
            down = hr
            first = hr.ndim - self.spatial_dims
            for i, sc in enumerate(self.axis_scales):
                if sc > 1:
                    down = _ssp_downsample_axis(
                        down, first + i, int(sc),
                        self._ssp_k(kernel, sc, hr.device, hr.dtype))
        else:
            mode = _interp_mode(kernel, self.spatial_dims)
            align = None if mode in ("area", "nearest") else False
            low = [max(int(round(s / sc)), 1)
                   for s, sc in zip(spatial, self.axis_scales)]
            down = F.interpolate(
                hr, size=tuple(low), mode=mode, align_corners=align)
        if noise_std > 0.0:  # 噪声在 LR 域施加，再随插值上采。
            down = down + torch.randn_like(down) * noise_std
        if self.keep_lr_size:
            return down
        return F.interpolate(
            down, size=spatial, mode=self.mode_up, align_corners=self._align_up)

    def _decimate_interp(
        self, hr: torch.Tensor, spatial: Tuple[int, ...],
        noise_std: float) -> torch.Tensor:
        """沿倍率>1 的轴抽稀保留帧，再逐轴相位对齐线性插值填回原尺寸（VFI 输入）。

        保留帧位于原始网格索引 ``0, sc, 2sc, ...``，输出在这些位置逐体素等于 HR
        （无相位偏移）；末尾超出最后一个保留帧的位置复制最后一帧。多轴退化时
        逐轴串行施加（线性插值可分离）。"""
        first = hr.ndim - self.spatial_dims
        idx = [slice(None)] * hr.ndim
        for i, sc in enumerate(self.axis_scales):
            if sc > 1:
                idx[first + i] = slice(0, None, sc)
        kept = hr[tuple(idx)]
        if noise_std > 0.0:  # 噪声施加在保留帧（LR 域）。
            kept = kept + torch.randn_like(kept) * noise_std
        if self.keep_lr_size:
            return kept
        lr = kept
        for i, sc in enumerate(self.axis_scales):
            if sc > 1:
                lr = _phase_aligned_linear_upsample(
                    lr, dim=first + i, scale=int(sc), out_size=int(spatial[i]))
        return lr


def _phase_aligned_linear_upsample(
    x: torch.Tensor, dim: int, scale: int, out_size: int) -> torch.Tensor:
    """沿 ``dim`` 把抽稀序列线性插值回 ``out_size``，保留帧相位精确对齐。

    ``x`` 沿 ``dim`` 的第 ``k`` 个样本对应原始网格索引 ``k*scale``；输出索引 ``j``
    由相邻保留帧 ``k=j//scale`` 与 ``k+1`` 线性混合，末尾 ``k+1`` 越界时钳位到
    最后一帧（等价 edge 复制）。"""
    n_kept = x.shape[dim]
    j = torch.arange(out_size, device=x.device)
    k = torch.div(j, scale, rounding_mode="floor").clamp(max=n_kept - 1)
    k_next = (k + 1).clamp(max=n_kept - 1)
    frac = ((j - k * scale).to(x.dtype) / scale)
    a = x.index_select(dim, k)
    b = x.index_select(dim, k_next)
    shape = [1] * x.ndim
    shape[dim] = out_size
    return a + (b - a) * frac.view(shape)


def build_degradation(
    cfg_task: TaskConfig, spatial_dims: int,
    keep_lr_size: bool = False) -> SuperResDegradation:
    """按 ``task`` 配置构造退化算子。当前仅 'superres'。

    ``task.sr_scale_per_axis`` 非空时走各向异性（逐空间轴倍率），否则用
    标量 ``task.sr_scale`` 各轴同倍。
    """
    deg = str(cfg_task.degradation).lower()
    if deg != "superres":
        raise ValueError(f"Unsupported degradation {deg!r}; only 'superres'.")
    per_axis = list(cfg_task.sr_scale_per_axis)
    return SuperResDegradation(
        scale=int(cfg_task.sr_scale),
        spatial_dims=spatial_dims,
        kernel=str(cfg_task.sr_kernel).lower(),
        kernel_up=str(cfg_task.sr_kernel_up).lower(),
        noise_std=float(cfg_task.sr_noise_std),
        axis_scales=per_axis if per_axis else None,
        sampling=str(cfg_task.sr_sampling).lower(),
        keep_lr_size=keep_lr_size,
        kernel_pool=list(cfg_task.sr_kernel_pool),
        noise_std_range=list(cfg_task.sr_noise_std_range))



__all__ = ["SuperResDegradation", "build_degradation"]
