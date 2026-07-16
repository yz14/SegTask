"""在线分类探针（SSL.md §0.4）。

动机：为后续 P6 评测管线提供一个与分割探针同构的表征读数：
``encoder → 全局平均池化 → 小 MLP 头``，并支持两种模式：

* ``frozen``：encoder 冻结，仅训练头；
* ``finetune``：encoder 与头一起训练，但 encoder 用更小学习率。

目标定义为「每类是否出现」的多标签二分类，默认从 ``LabeledPatchDataset`` 的
``label`` patch 按 ``cfg.data.label_values[1:]`` 派生；若 npz 中提供了
``ssl.cls_label_key`` 对应键，则优先使用该显式类别标签。
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from segtask_v1.models.factory import build_model
from segtask_v1.trainer.checkpoint import strip_common_prefixes

from ..data.ssl_dataset import LabeledPatchDataset, discover_image_npz
from .metrics import _binary_f1, _rank_auc, macro_cls_metrics
from .split import group_split

logger = logging.getLogger(__name__)


def build_cls_probe_loaders(cfg, ssl) -> Tuple[DataLoader, DataLoader]:
    """从 ``ssl.probe_data_dir`` 的标注 npz 划分 train/val loader。

    划分为组级（患者级，:func:`ssltask.eval.split.group_split`，与分割探针
    同口径）：同组文件不跨 train/val，避免同患者多序列泄漏。"""
    paths = discover_image_npz(ssl.probe_data_dir, cfg.data.npz_suffix)
    train_paths, val_paths = group_split(
        paths, float(ssl.probe_val_ratio), int(ssl.probe_seed),
        group_regex=str(ssl.probe_group_regex),
        allow_single_group=bool(ssl.probe_allow_single_group))
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
            global_mean=dc.global_mean,
            global_std=dc.global_std,
            spatial_dims=spatial_dims,
            patch_mode=dc.patch_mode,
            cls_label_key=ssl.cls_label_key,
            cache_enabled=dc.cache_mode == "memory",
            cache_max_volumes=dc.cache_max_volumes,
        )

    train_ds = _mk(train_paths, int(ssl.probe_samples_per_volume))
    val_ds = _mk(val_paths, max(int(ssl.probe_samples_per_volume) // 2, 1))
    bs = max(int(dc.batch_size), 1)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=0, drop_last=False)
    logger.info(
        "ClsProbe loaders: %d train / %d val volume(s) from %s",
        len(train_paths), len(val_paths), ssl.probe_data_dir)
    return train_loader, val_loader


class _ClsHead(nn.Module):
    """全局池化后的两层 MLP 头。"""

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClsProbe:
    """在线分类探针：encoder + GAP + MLP 头。"""

    def __init__(self, cfg, ssl, device: torch.device,
                 finetune: Optional[bool] = None):
        self.cfg = cfg
        self.ssl = ssl
        self.device = device
        self.spatial_dims = int(cfg.model.spatial_dims)
        if self.spatial_dims not in (2, 3):
            raise ValueError(
                f"ClsProbe supports model.spatial_dims in (2, 3); got "
                f"{self.spatial_dims}.")
        self.num_fg = int(cfg.num_fg_classes)
        self.enc_out_dim = int(cfg.model.encoder_channels[-1])
        self.hidden_dim = int(ssl.cls_probe_hidden_dim)
        self.iters = int(ssl.cls_probe_iters)
        self.lr = float(ssl.cls_probe_lr)
        self.finetune_lr = float(ssl.cls_probe_finetune_lr)
        self.seed = int(ssl.probe_seed)
        self.finetune = bool(ssl.cls_probe_finetune if finetune is None else finetune)
        lv = list(cfg.data.label_values)
        self.fg_values = [float(v) for v in lv[1:]] if len(lv) > 1 else [1.0]
        self.cls_label_key = str(ssl.cls_label_key)

        self.encoder = build_model(cfg).encoder.to(device)
        self.train_loader, self.val_loader = build_cls_probe_loaders(cfg, ssl)

    def _set_encoder_trainable(self, trainable: bool) -> None:
        self.encoder.train(mode=bool(trainable))
        for p in self.encoder.parameters():
            p.requires_grad_(bool(trainable))

    def _build_head(self) -> nn.Module:
        return _ClsHead(self.enc_out_dim, self.hidden_dim, self.num_fg).to(self.device)

    def _load_encoder(self, full_sd: Dict[str, torch.Tensor]) -> None:
        sd = strip_common_prefixes(full_sd)
        enc_sd = {k[len("encoder."):]: v for k, v in sd.items()
                  if k.startswith("encoder.")}
        if not enc_sd:
            raise KeyError(
                "Probe got a state_dict with no 'encoder.*' keys; cannot "
                "evaluate representation quality.")
        self.encoder.load_state_dict(enc_sd, strict=True)

    def _target_from_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "cls_label" in batch:
            cls = batch["cls_label"].to(self.device).float()
            if cls.ndim == 0:
                cls = cls.view(1, 1)
            elif cls.ndim == 1:
                if cls.shape[0] == batch["image"].shape[0] and self.num_fg == 1:
                    cls = cls.unsqueeze(1)
                elif cls.numel() == self.num_fg:
                    cls = cls.view(1, self.num_fg)
                else:
                    cls = cls.view(batch["image"].shape[0], -1)
            else:
                cls = cls.view(batch["image"].shape[0], -1)
            if cls.shape[1] != self.num_fg:
                raise ValueError(
                    f"cls_label shape {tuple(cls.shape)} does not match "
                    f"num_fg={self.num_fg}.")
            return cls
        label = batch["label"].to(self.device)
        targets = []
        for v in self.fg_values:
            targets.append((label == v).flatten(1).any(dim=1).float())
        return torch.stack(targets, dim=1)

    def _fold_2_5d_t(self, x: torch.Tensor) -> torch.Tensor:
        """2.5D：(B,1,D,H,W)→(B,D,H,W)（D 折进通道）；3D 原样返回。
        LabeledPatchDataset 现统一输出 3D，折叠在此消费方完成。"""
        if self.spatial_dims == 2 and x.dim() == 5:
            b, c, d, h, w = x.shape
            return x.reshape(b, c * d, h, w)
        return x

    def _forward_logits(self, head: nn.Module, img: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(img)
        pooled = feats[-1].mean(dim=tuple(range(2, feats[-1].ndim)))
        return head(pooled)

    def _train_step(self, batch: Dict[str, torch.Tensor], head: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    loss_fn: nn.Module) -> torch.Tensor:
        img = self._fold_2_5d_t(batch["image"].to(self.device).float())
        target = self._target_from_batch(batch)
        if self.finetune:
            self._set_encoder_trainable(True)
            logits = self._forward_logits(head, img)
        else:
            self._set_encoder_trainable(False)
            with torch.no_grad():
                feats = self.encoder(img)
            pooled = feats[-1].mean(dim=tuple(range(2, feats[-1].ndim)))
            logits = head(pooled)
        loss = loss_fn(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return loss

    def _collect_scores(self, head: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        ys: List[np.ndarray] = []
        ps: List[np.ndarray] = []
        self.encoder.eval()
        head.eval()
        with torch.no_grad():
            for batch in loader:
                img = self._fold_2_5d_t(batch["image"].to(self.device).float())
                target = self._target_from_batch(batch)
                logits = self._forward_logits(head, img)
                ys.append(target.detach().cpu().numpy())
                ps.append(torch.sigmoid(logits).detach().cpu().numpy())
        y_true = np.concatenate(ys, axis=0) if ys else np.zeros((0, self.num_fg))
        y_score = np.concatenate(ps, axis=0) if ps else np.zeros((0, self.num_fg))
        return y_true, y_score

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

    def evaluate(self, full_sd: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
        """加载 encoder → 训练分类头 → 返回 ``{'cls_auc', 'cls_f1'}``。"""
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
            params = [{"params": head.parameters(), "lr": self.lr}]
            if self.finetune:
                params = [
                    {"params": self.encoder.parameters(), "lr": self.finetune_lr},
                    *params,
                ]
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

            y_true, y_score = self._collect_scores(head, self.val_loader)
            return macro_cls_metrics(y_true, y_score)
        finally:
            self._restore_rng_state(rng_state)


__all__ = [
    "ClsProbe",
    "build_cls_probe_loaders",
    "macro_cls_metrics",
    "_binary_f1",
    "_rank_auc",
]
