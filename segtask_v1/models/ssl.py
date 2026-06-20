"""自监督预训练（SSL）重建模型。

`SSLReconModel` = 与分割同构的 UNet 编/解码器（由 `build_model` 构建后复用其
`encoder` / `decoder` 子模块）+ 一个**独立命名**的重建头 `recon_head`
（out_channels = 模型输入通道数）。

关键设计：重建头名为 `recon_head`（而非分割的 `seg_head`），因此 SSL ckpt 与
分割模型之间不会发生 head 权重的 shape 冲突——分割侧 `_load_pretrain`
（`strict=False`）会让 `encoder.*` / `decoder.*` 全部命中、`recon_head.*` 作为
unexpected 被丢弃、`seg_head.*` 作为 missing 保持随机初始化。这是 SSL→分割
干净交接的核心。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .unet import SegmentationHead


class SSLReconModel(nn.Module):
    """编码器-解码器重建模型（Models Genesis 式自监督）。

    forward(x): (B, C, *spatial) → (B, C, *spatial) 重建。"""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        out_channels: int,
        spatial_dims: int = 3):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.out_channels = int(out_channels)
        self.recon_head = SegmentationHead(
            decoder.out_channels[-1], self.out_channels,
            spatial_dims=spatial_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_features = self.encoder(x)
        dec_features = self.decoder(enc_features)
        out = self.recon_head(dec_features[-1])
        if out.shape[2:] != x.shape[2:]:
            raise RuntimeError(
                f"SSL recon output size mismatch: got {tuple(out.shape[2:])}, "
                f"expected {tuple(x.shape[2:])}. Check stem_stride / encoder "
                f"downsampling vs input spatial dims.")
        return out

    def param_count(self) -> dict:
        enc  = sum(p.numel() for p in self.encoder.parameters())
        dec  = sum(p.numel() for p in self.decoder.parameters())
        head = sum(p.numel() for p in self.recon_head.parameters())
        return {"encoder": enc, "decoder": dec, "recon_head": head,
                "total": enc + dec + head}


__all__ = ["SSLReconModel"]
