"""DetectorModel：共享 Encoder + Decoder 金字塔 + FPN 适配 + 检测头。

参数命名 ``encoder.* / decoder.* / fpn.* / det_head.*``——encoder/decoder
与分割/SSL 同名同形，预训练权重 strict=False 直接迁移（Plan §3.2 / §3.7）。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

__all__ = ["DetectorModel"]


class DetectorModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 fpn: nn.Module, det_head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fpn = fpn
        self.det_head = det_head

    def extract_pyramid(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.fpn(self.decoder(self.encoder(x)))

    def forward(
        self,
        images   : torch.Tensor,
        gt_boxes : Optional[List[torch.Tensor]] = None,
        gt_labels: Optional[List[torch.Tensor]] = None,
    ):
        """训练（gt 给定）→ 损失 dict；推理 → 逐样本检测 dict 列表。"""
        img_size = list(images.shape[2:])
        feats = self.extract_pyramid(images)
        if gt_boxes is not None:
            assert gt_labels is not None
            return self.det_head.compute_loss(feats, gt_boxes, gt_labels,
                                              img_size)
        return self.det_head.predict(feats, img_size)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def state_summary(self) -> Dict[str, int]:
        return {name: sum(p.numel() for p in mod.parameters())
                for name, mod in (("encoder", self.encoder),
                                  ("decoder", self.decoder),
                                  ("fpn", self.fpn),
                                  ("det_head", self.det_head))}
