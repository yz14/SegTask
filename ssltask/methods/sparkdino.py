"""方案⑧ SparK + DINO：像素掩码主干（SparK①）+ 全局蒸馏（DINO④）的朴素多任务组合。

以最简单的"双损失相加"把像素 MIM（SparK，密集强）与图像级自蒸馏（DINO，全局强）结合，
力图在不引入 ⑥ iBOT 复杂度的前提下得到分割+分类双强的单一 CNN。两分支**共享同一 encoder**：
SparK 分支走稀疏掩码输入→层次解码器→像素重建（同①）；DINO 分支走多视图完整输入→全局
池化→投影头→对 EMA 教师蒸馏（同④）。由掩码-稠密权重等价，编码器在两分支间无缝复用。

``L = L_SparK(像素重建) + μ·L_DINO(全局蒸馏)``，μ=``sparkdino_dino_weight``。实现上子类化
``DINOMethod`` 复用其多裁剪 / EMA 教师 / center-sharpen / 温度·动量调度 / 教师 encoder 导出；
``compute_loss`` 先调父类得 ``L_DINO``（行为与 ④ 完全一致），再在原始整图上加 SparK 重建项
（行为与 ① 完全一致）。SparK 分支直接复用 DINO **学生**的 encoder，故单一 encoder 同时吃两路
梯度、教师照常 EMA 跟随。下游交接同 ④/①：只导出 **教师** ``encoder.*``，SparK 解码器 / DINO
头用完即弃。
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from ..data.masking import make_unit_mask, masked_recon_loss, per_unit_normalize
from ..models.spark_modules import build_spark_decoder, spark_encode
from .base import SSLMethod  # noqa: F401  (保持与同级方法一致的导入面)
from .dino import DINOMethod, _DINOModule


class _SparkDINOModule(_DINOModule):
    """在 ④ 的 student/teacher/center 之上挂一个作用于 **学生 encoder** 的 SparK 解码器。"""

    def __init__(self, student: nn.Module, teacher: nn.Module, out_dim: int,
                 spark_decoder: nn.Module):
        super().__init__(student, teacher, out_dim)
        self.spark_decoder = spark_decoder


class SparkDINOMethod(DINOMethod):
    name = "sparkdino"

    def __init__(self, cfg, ssl, device: torch.device):
        super().__init__(cfg, ssl, device)
        self.dino_weight = float(ssl.sparkdino_dino_weight)
        self.spark_mask_ratio = float(ssl.spark_mask_ratio)
        self.spark_unit = int(ssl.spark_mask_unit)
        self.recon_loss = str(ssl.recon_loss)
        self.spark_norm_pix = bool(ssl.spark_norm_pix)

    # ---- modules ----------------------------------------------------------
    def build_modules(self) -> nn.Module:
        base = super().build_modules()                # _DINOModule(student, teacher)
        decoder = build_spark_decoder(               # 作用于学生 encoder（共享骨干）
            self.cfg, base.student.encoder,
            dim_div=int(self.ssl.spark_decoder_dim_div),
            min_dim=int(self.ssl.spark_decoder_min_dim))
        return _SparkDINOModule(
            base.student, base.teacher, int(self.ssl.dino_out_dim), decoder)

    # ---- SparK pixel-reconstruction branch -------------------------------
    def _spark_loss(self, image: torch.Tensor) -> torch.Tensor:
        """单一被遮视图：稀疏掩码输入→共享 encoder→层次解码器→仅被遮位点重建（同①）。"""
        spatial = image.shape[2:]
        mask_full = make_unit_mask(
            image.shape[0], spatial, self.spark_unit, self.spark_mask_ratio,
            image.device)
        features, visibles = spark_encode(self.module.student.encoder, image, mask_full)
        pred = self.module.spark_decoder(features, visibles, spatial)
        target = (per_unit_normalize(image, self.spark_unit)
                  if self.spark_norm_pix else image)
        return masked_recon_loss(pred, target, mask_full, self.recon_loss)

    # ---- loss -------------------------------------------------------------
    def compute_loss(self, batch: Dict[str, torch.Tensor]
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
        dino_loss, logs = super().compute_loss(batch)
        spark_loss = self._spark_loss(batch["image"])
        loss = spark_loss + self.dino_weight * dino_loss
        logs["spark_loss"] = float(spark_loss.detach())
        logs["dino_weight"] = self.dino_weight
        return loss, logs


__all__ = ["SparkDINOMethod"]
