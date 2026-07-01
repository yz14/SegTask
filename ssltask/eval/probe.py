"""在线分割线性探针（SSL.md §0.5）。

动机：自监督代理损失（尤其 DINO/JEPA/对比族）与下游表征质量并非单调，按 SSL loss
选 best ckpt 会误选。本探针在一小份**标注** npz 上，冻结/微调当前 encoder、仅训
练一个**多尺度 1×1 线性头**若干步，回报前景 Dice / HD95，作为可比的表征质量信号。

设计要点：
* **线性探针**：encoder 冻结（``requires_grad=False`` + ``eval()``）时只训练逐尺度
  1×1 卷积头（对每级 encoder 特征预测前景 logits，上采样求和）；微调时 encoder 也
  参与训练，但学习率更低。
* **可比性**：每次评估都从固定随机种子重置头、训练固定步数，跨 epoch 可比。
* **零侵入下游**：encoder 经 ``build_model(cfg).encoder`` 构造（与 SSL 同构），探
针仅读 SSL 导出 state_dict 的 ``encoder.*`` 子集（``strict=True`` 校验同名同形）。
* **范围**：3D（``spatial_dims==3``）与 2.5D（``spatial_dims==2``，深度 D 折进通
道）均支持。
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from segtask_v1.models.blocks import _CONV, INTERP_SMOOTH
from segtask_v1.models.factory import build_model
from segtask_v1.trainer.checkpoint import strip_common_prefixes
from segtask_v1.utils import compute_dice_per_class

from ..data.ssl_dataset import LabeledPatchDataset, discover_image_npz
from .metrics import hd95

logger = logging.getLogger(__name__)


def build_probe_loaders(cfg, ssl) -> Tuple[DataLoader, DataLoader]:
    """从 ``ssl.probe_data_dir`` 的标注 npz 划分 train/val 探针 loader（num_workers=0）。"""
    paths = discover_image_npz(ssl.probe_data_dir, cfg.data.npz_suffix)
    rng = random.Random(int(ssl.probe_seed))
    paths = list(paths)
    rng.shuffle(paths)
    n_val = max(1, int(round(len(paths) * float(ssl.probe_val_ratio))))
    if len(paths) > 1:
        n_val = min(n_val, len(paths) - 1)              # 保证 train 非空
    val_paths = paths[:n_val]
    train_paths = paths[n_val:] or list(paths)          # 单卷时 train==val
    dc = cfg.data
    spatial_dims = int(cfg.model.spatial_dims)

    def _mk(p, spv):
        return LabeledPatchDataset(
            npz_paths=p,
            patch_size=dc.patch_size,
            intensity_min=dc.intensity_min,
            intensity_max=dc.intensity_max,
            normalize=dc.normalize,
            samples_per_volume=spv,
            spatial_dims=spatial_dims)

    train_ds = _mk(train_paths, int(ssl.probe_samples_per_volume))
    val_ds = _mk(val_paths, max(int(ssl.probe_samples_per_volume) // 2, 1))
    bs = max(int(dc.batch_size), 1)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=0, drop_last=False)
    logger.info(
        "Probe loaders: %d train / %d val volume(s) from %s",
        len(train_paths), len(val_paths), ssl.probe_data_dir)
    return train_loader, val_loader


class _MultiScaleLinearHead(nn.Module):
    """逐 encoder 特征级一个 1×1 卷积 → 前景 logits，上采样到输入分辨率后求和。"""

    def __init__(self, enc_channels: List[int], out_channels: int,
                 spatial_dims: int):
        super().__init__()
        self.spatial_dims = int(spatial_dims)
        self.mode = INTERP_SMOOTH[self.spatial_dims]
        self.heads = nn.ModuleList(
            [_CONV[self.spatial_dims](int(c), int(out_channels), kernel_size=1)
             for c in enc_channels])

    def forward(self, feats: List[torch.Tensor], target_spatial) -> torch.Tensor:
        target_spatial = tuple(int(s) for s in target_spatial)
        out: Optional[torch.Tensor] = None
        for head, f in zip(self.heads, feats):
            o = head(f)
            if o.shape[2:] != target_spatial:
                o = F.interpolate(o, size=target_spatial, mode=self.mode,
                                  align_corners=False)
            out = o if out is None else out + o
        return out


class SegProbe:
    """冻结/微调 encoder 的多尺度线性分割探针。``evaluate`` 输入 SSL 导出的 state_dict。"""

    def __init__(self, cfg, ssl, device: torch.device, finetune: Optional[bool] = None):
        self.cfg = cfg
        self.ssl = ssl
        self.device = device
        self.spatial_dims = int(cfg.model.spatial_dims)
        if self.spatial_dims not in (2, 3):
            raise ValueError(
                f"SegProbe supports model.spatial_dims in (2, 3); got {self.spatial_dims}.")
        self.num_fg = int(cfg.num_fg_classes)
        self.enc_channels = list(cfg.model.encoder_channels)
        self.slab_depth = int(cfg.data.patch_size[0]) if self.spatial_dims == 2 else 1
        self.head_out = self.num_fg * self.slab_depth
        lv = list(cfg.data.label_values)
        self.fg_values = [float(v) for v in lv[1:]] if len(lv) > 1 else [1.0]
        self.iters = int(ssl.probe_iters)
        self.lr = float(ssl.probe_lr)
        self.finetune_lr = float(ssl.probe_finetune_lr)
        self.seed = int(ssl.probe_seed)
        self.finetune = bool(ssl.probe_finetune if finetune is None else finetune)

        self.encoder = build_model(cfg).encoder.to(device)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.train_loader, self.val_loader = build_probe_loaders(cfg, ssl)

    def _set_encoder_trainable(self, trainable: bool) -> None:
        self.encoder.train(mode=bool(trainable))
        for p in self.encoder.parameters():
            p.requires_grad_(bool(trainable))

    def _build_head(self) -> nn.Module:
        return _MultiScaleLinearHead(self.enc_channels, self.head_out,
                                     self.spatial_dims).to(self.device)

    def _binary_target(self, label: torch.Tensor) -> torch.Tensor:
        if self.spatial_dims == 2:
            B, D, H, W = label.shape
            chans = [(label == v).float() for v in self.fg_values]
            stacked = torch.stack(chans, dim=1)
            return stacked.reshape(B, self.num_fg * D, H, W)
        chans = [(label[:, 0] == v).float() for v in self.fg_values]
        return torch.stack(chans, dim=1)

    @staticmethod
    def _probe_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        dims = tuple(range(2, logits.ndim))
        inter = (prob * target).sum(dims)
        denom = prob.sum(dims) + target.sum(dims)
        dice = (2.0 * inter + 1e-5) / (denom + 1e-5)
        return bce + (1.0 - dice.mean())

    @torch.no_grad()
    def _load_encoder(self, full_sd: Dict[str, torch.Tensor]) -> None:
        sd = strip_common_prefixes(full_sd)
        enc_sd = {k[len("encoder."):]: v for k, v in sd.items()
                  if k.startswith("encoder.")}
        if not enc_sd:
            raise KeyError(
                "Probe got a state_dict with no 'encoder.*' keys; cannot evaluate representation quality.")
        self.encoder.load_state_dict(enc_sd, strict=True)

    def _save_rng_state(self):
        return (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        )

    def _restore_rng_state(self, state) -> None:
        py_state, np_state, torch_state, cuda_state = state
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    def _forward_logits(self, head: nn.Module, img: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(img)
        return head(feats, img.shape[2:])

    def _train_step(self, batch: Dict[str, torch.Tensor], head: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    loss_fn: nn.Module) -> torch.Tensor:
        img = batch["image"].to(self.device).float()
        target = self._binary_target(batch["label"].to(self.device).float())
        if self.finetune:
            self._set_encoder_trainable(True)
            logits = self._forward_logits(head, img)
        else:
            self._set_encoder_trainable(False)
            with torch.no_grad():
                feats = self.encoder(img)
            logits = head(feats, img.shape[2:])
        loss = loss_fn(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return loss

    def _collect_scores(self, head: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, float]:
        dices: List[torch.Tensor] = []
        hd_vals: List[float] = []
        self.encoder.eval()
        head.eval()
        with torch.no_grad():
            for batch in loader:
                img = batch["image"].to(self.device).float()
                tgt = self._binary_target(batch["label"].to(self.device).float())
                logits = self._forward_logits(head, img)
                prob = torch.sigmoid(logits)
                dices.append(compute_dice_per_class(logits, tgt).cpu())
                hd_vals.append(hd95(prob.cpu().numpy(), tgt.cpu().numpy()))
        probe_dice = float(torch.stack(dices).mean()) if dices else 0.0
        finite = [v for v in hd_vals if np.isfinite(v)]
        probe_hd95 = float(np.mean(finite)) if finite else 0.0
        return probe_dice, probe_hd95

    def evaluate(self, full_sd: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
        """加载 encoder → 训练线性头 ``probe_iters`` 步 → 返回 ``{'probe_dice', 'probe_hd95'}``。"""
        if full_sd is not None:
            self._load_encoder(full_sd)
        rng_state = self._save_rng_state()
        try:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

            head = self._build_head()
            if self.finetune:
                params = [
                    {"params": self.encoder.parameters(), "lr": self.finetune_lr},
                    {"params": head.parameters(), "lr": self.lr},
                ]
            else:
                params = [{"params": head.parameters(), "lr": self.lr}]
            optimizer = torch.optim.Adam(params)
            loss_fn = nn.BCEWithLogitsLoss()

            data_iter = iter(self.train_loader)
            for _ in range(self.iters):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)
                self._train_step(batch, head, optimizer, loss_fn)

            probe_dice, probe_hd95 = self._collect_scores(head, self.val_loader)
            return {"probe_dice": probe_dice, "probe_hd95": probe_hd95}
        finally:
            self._restore_rng_state(rng_state)


__all__ = ["SegProbe", "build_probe_loaders"]
