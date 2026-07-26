"""Dataset selection strategy（data 侧 ``ViewPipeline`` 类比）。

R4：把 ``loader.py::build_dataloaders`` 中"按 ``patch_mode`` 选 dataset 类
+ 准备各模式专属 kwargs"的 6 处重复构造收敛到 3 个 ``DatasetSpec`` 子类与
1 个 ``build_data_spec`` 工厂。

* ``DatasetCommonCfg``  —— 14 个公共构造参数（与模式无关）的不可变快照
* ``SplitPaths``        —— 单 split 的路径三元组
* ``DatasetSpec``       —— 策略基类，``make_split(paths, is_train, common)``
* ``WholeSpec`` / ``ZCubeSpec`` / ``CubicSpec`` —— 3 个具体策略
* ``build_data_spec``   —— **整个 codebase 唯一允许 patch_mode if/elif 的地方
  （data 侧）**

注意：本文件不修改 ``dataset.py`` 的 ``__init__`` 签名，仅在 spec 内部封装
"split-dependent kwargs（``aug_oversample_ratio`` / ``samples_per_volume`` /
``foreground_oversample_ratio``）随 ``is_train`` 切换"的逻辑，使
``loader.py`` 不再涉及训练/验证差异。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import inspect
from typing import Any, List, Optional, Tuple

from torch.utils.data import Dataset

from .dataset import SegDataset3D, SegDataset3DCubic, SegDataset3DWhole

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Common config snapshot
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetCommonCfg:
    """与 patch_mode 无关的 dataset 公共构造参数。

    所有 3 个 dataset 子类（``SegDataset3D`` / ``SegDataset3DCubic`` /
    ``SegDataset3DWhole``）共享这 12 个字段。原 ``loader.py:522-532`` 的
    ``common_kwargs`` dict 已被这个 dataclass 替代。
    """

    label_values: List[int]
    patch_size: Tuple[int, int, int]
    intensity_min: float
    intensity_max: float
    normalize: str
    global_mean: float
    global_std: float
    cache_enabled: bool
    cache_max_volumes: int
    cache_int16: bool
    region_weights: Optional[List[float]]
    resize_antialias: bool = False

    @classmethod
    def from_cfg(cls, cfg: Any) -> "DatasetCommonCfg":
        """从运行期 cfg 构建（``SegBundle`` / 带 ``.loss`` 的视图）。

        ``region_weights`` 取自 ``cfg.loss.region_weights``（seg 任务段）；
        纯 core ``Config`` 无 ``loss`` 时为 ``None``。生成任务请用
        ``gentask.data.specs.DatasetCommonCfg.from_cfg``（权重挂在 ``data``）。
        """
        dc = cfg.data
        loss = getattr(cfg, "loss", None)
        rw = getattr(loss, "region_weights", None) if loss is not None else None
        return cls(
            label_values      = list(dc.label_values),
            patch_size        = tuple(int(x) for x in dc.patch_size),
            intensity_min     = float(dc.intensity_min),
            intensity_max     = float(dc.intensity_max),
            normalize         = str(dc.normalize),
            global_mean       = float(dc.global_mean),
            global_std        = float(dc.global_std),
            cache_enabled     = (str(dc.cache_mode) == "memory"),
            cache_max_volumes = int(dc.cache_max_volumes),
            cache_int16       = (str(dc.cache_dtype) == "int16"),
            region_weights    = list(rw) if rw else None,
            resize_antialias  = bool(dc.resize_antialias))

    def to_kwargs(self) -> dict:
        """直接展开为 dataset ``__init__`` 的 kwargs（含子类扩展字段）。"""
        values = asdict(self)
        values.pop("resize_antialias", None)
        return values


@dataclass(frozen=True)
class SplitPaths:
    """单个 split（train 或 val）的样本路径三元组。"""

    image_paths: List[str]
    label_paths: List[str]
    npz_paths  : List[str]

    def to_kwargs(self) -> dict:
        return dict(
            image_paths = self.image_paths,
            label_paths = self.label_paths,
            npz_paths   = self.npz_paths)


# ---------------------------------------------------------------------------
# Strategy base + concrete specs
# ---------------------------------------------------------------------------
class DatasetSpec(ABC):
    """Data 侧策略对象：把 ``cfg`` + (paths, is_train) 翻译成具体 ``Dataset``。

    子项目（如 gentask）可通过覆盖各具体 spec 的 ``dataset_cls`` 类属性
    把同一套选择/参数逻辑接到自己的 dataset 实现上。"""

    name: str = "abstract"
    dataset_cls: "Optional[type]" = None

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    @abstractmethod
    def make_split(
        self, paths: SplitPaths, is_train: bool, common: DatasetCommonCfg) -> Dataset:
        """返回某一 split 对应的 dataset 实例。"""

    # ------------------------------------------------------------------
    # 共用辅助：随 is_train 切换的 4 个动态参数。
    # ------------------------------------------------------------------
    def _aug_oversample(self, is_train: bool) -> float:
        """train 取 ``max(aug_oversample_ratio, 1.0)``；val 始终 1.0。"""
        if is_train:
            return max(float(self.cfg.data.aug_oversample_ratio), 1.0)
        return 1.0

    def _samples_per_volume(self, is_train: bool) -> int:
        """train 完整 ``samples_per_volume``；val 减半（不少于 1）。"""
        spv = int(self.cfg.data.samples_per_volume)
        return spv if is_train else max(spv // 2, 1)

    def _fg_ratio(self, is_train: bool) -> float:
        """val 不做前景过采样以避免污染验证分布。"""
        return float(self.cfg.data.foreground_oversample_ratio) if is_train else 0.0

    def _resize_kwargs(self) -> dict:
        """仅向支持该参数的 dataset 传递 CPU resize 抗混叠开关。"""
        if "resize_antialias" in inspect.signature(
                type(self).dataset_cls.__init__).parameters:
            return {"resize_antialias": bool(self.cfg.data.resize_antialias)}
        return {}

    def log_summary(self) -> None:
        """供 ``build_dataloaders`` 在构造 split 前打印模式概要（默认无操作）。"""

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__}>"


class WholeSpec(DatasetSpec):
    """整体模式，无 multi_res / 无 fg 过采样"""

    name = "whole"
    dataset_cls = SegDataset3DWhole

    def log_summary(self) -> None:
        logger.info(
            "Using WHOLE-VOLUME patch mode (oversample=%.2f)",
            self._aug_oversample(is_train=True))

    def make_split(
        self, paths: SplitPaths, is_train: bool, common: DatasetCommonCfg
        ) -> Dataset:
        # whole 在 Config.validate 中已强制 multi_res_scales=[1.0]、忽略 fg 过采样。
        # val 无增强、整卷输入确定：spv>1 只是等比重复同一样本（pooled Dice
        # 数值不变，纯白算），固定 1。
        extra: dict = {}
        if "oversample_mode" in inspect.signature(
                type(self).dataset_cls.__init__).parameters:
            extra["oversample_mode"] = str(
                self.cfg.data.whole_oversample_mode)
        return type(self).dataset_cls(
            **paths.to_kwargs(),
            **common.to_kwargs(),
            aug_oversample_ratio = self._aug_oversample(is_train),
            samples_per_volume   = (self._samples_per_volume(True)
                                    if is_train else 1),
            **self._resize_kwargs(),
            **extra,
            is_train=is_train)


class ZCubeSpec(DatasetSpec):
    """z 轴 single max-FOV cube 模式（``patch_mode in {z_axis, 2_5d}``）。

    两种 patch_mode 在 dataset 侧抽取逻辑完全一致 —— 都发 max-FOV z-cube；
    多分辨率拆视图全部交给 trainer/predictor 完成。仅日志区分。
    """

    name = "z_axis|2_5d"
    dataset_cls = SegDataset3D

    def log_summary(self) -> None:
        dc        = self.cfg.data
        n_views   = max(len(dc.multi_res_scales), 1)
        max_scale = max(dc.multi_res_scales) if dc.multi_res_scales else 1.0
        logger.info(
            "Using %s patch mode (oversample=%.2f, scales=%s, n_views=%d, "
            "max_scale=%.2f, z_boundary=%s) — SINGLE max-FOV z-cube extraction; "
            "trainer crops+resizes per view before forward.",
            dc.patch_mode.upper(), self._aug_oversample(is_train=True),
            dc.multi_res_scales, n_views, max_scale,
            dc.z_boundary_mode)

    def make_split(
        self, paths: SplitPaths, is_train: bool, common: DatasetCommonCfg
        ) -> Dataset:
        dc = self.cfg.data
        return type(self).dataset_cls(
            **paths.to_kwargs(),
            **common.to_kwargs(),
            aug_oversample_ratio        = self._aug_oversample(is_train),
            multi_res_scales            = list(dc.multi_res_scales),
            foreground_oversample_ratio = self._fg_ratio(is_train),
            samples_per_volume          = self._samples_per_volume(is_train),
            is_train                    = is_train,
            z_boundary_mode             = dc.z_boundary_mode,
            z_sampling_mode             = dc.z_sampling_mode,
            **self._resize_kwargs(),
            val_grid_coverage           = dc.val_grid_coverage)


class CubicSpec(DatasetSpec):
    """3 轴 cubic max-FOV 模式（``patch_mode='cubic'``）。"""

    name = "cubic"
    dataset_cls = SegDataset3DCubic

    def log_summary(self) -> None:
        dc = self.cfg.data
        max_scale = max(dc.multi_res_scales) if dc.multi_res_scales else 1.0
        logger.info(
            "Using CUBIC patch mode (oversample=%.2f, scales=%s, "
            "max_scale=%.2f) — SINGLE max-FOV cube extraction; trainer "
            "crops+resizes per view before the 3D forward.",
            self._aug_oversample(is_train=True), dc.multi_res_scales, max_scale)

    def make_split(
        self, paths: SplitPaths, is_train: bool, common: DatasetCommonCfg
        ) -> Dataset:
        dc = self.cfg.data
        return type(self).dataset_cls(
            **paths.to_kwargs(),
            **common.to_kwargs(),
            aug_oversample_ratio        = self._aug_oversample(is_train),
            multi_res_scales            = list(dc.multi_res_scales),
            foreground_oversample_ratio = self._fg_ratio(is_train),
            samples_per_volume          = self._samples_per_volume(is_train),
            is_train                    = is_train,
            **self._resize_kwargs(),
            val_grid_coverage           = dc.val_grid_coverage)


# ---------------------------------------------------------------------------
# Factory — 整个 data 子包中唯一允许 patch_mode if/elif 的地方
# ---------------------------------------------------------------------------
def build_data_spec(cfg: Config) -> DatasetSpec:
    """``cfg.data.patch_mode`` → ``DatasetSpec``。新增 patch_mode 仅需在此追加一行。"""
    pm = str(cfg.data.patch_mode).lower()
    if pm in ("2_5d", "z_axis"):
        return ZCubeSpec(cfg)
    if pm == "whole":
        return WholeSpec(cfg)
    if pm == "cubic":
        return CubicSpec(cfg)
    raise ValueError(
        f"Unknown patch_mode: {pm!r}. Valid: 'z_axis' | '2_5d' | 'whole' | 'cubic'.")


__all__ = [
    "DatasetCommonCfg",
    "SplitPaths",
    "DatasetSpec",
    "WholeSpec",
    "ZCubeSpec",
    "CubicSpec",
    "build_data_spec"]
