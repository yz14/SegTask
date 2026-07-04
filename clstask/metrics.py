"""分类评估指标：AUC / F1 / Accuracy（纯 torch，无 sklearn 依赖）。

多标签：逐类算再宏平均；slice 粒度把 (N, K, D) 摊平为 (N*D, K) 后同口径。
单标签：AUC 用 one-vs-rest 宏平均，acc/f1 用 argmax。

AUC 用 Mann-Whitney U 统计（rank 实现，含并列校正）；某类全正/全负时该类
AUC 无定义，跳过不计入宏平均（全部无定义时返回 0.5 并由调用方打警告）。
"""

from __future__ import annotations

from typing import Dict

import torch


def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """单类 AUC（Mann-Whitney U / rank 法，并列取平均秩）。

    ``scores`` (N,) 任意实数分数，``labels`` (N,) ∈ {0, 1}。
    正/负样本缺失时返回 ``nan``。
    """
    labels = labels.float()
    n_pos = int(labels.sum().item())
    n_neg = labels.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(
        1, scores.numel() + 1, dtype=scores.dtype, device=scores.device)
    # 并列校正：同分数取平均秩。
    sorted_scores = scores[order]
    uniq, inv, counts = torch.unique(
        sorted_scores, return_inverse=True, return_counts=True)
    if uniq.numel() != scores.numel():
        cum = counts.cumsum(0)
        start = cum - counts
        avg_rank = (start + cum + 1).float() / 2.0    # 平均秩（1-based）
        ranks[order] = avg_rank[inv].to(scores.dtype)
    rank_pos = ranks[labels > 0.5].sum().item()
    u = rank_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _flatten_slice(t: torch.Tensor) -> torch.Tensor:
    """(N, K, D) → (N*D, K)；(N, K) 原样返回。"""
    if t.ndim == 3:
        return t.permute(0, 2, 1).reshape(-1, t.shape[1])
    return t


def multilabel_metrics(probs: torch.Tensor, targets: torch.Tensor,
                       threshold: float = 0.5) -> Dict[str, float]:
    """多标签宏平均 AUC / F1 / acc。``probs``/``targets`` (N, K) 或 (N, K, D)。"""
    probs = _flatten_slice(probs.detach().float().cpu())
    targets = _flatten_slice(targets.detach().float().cpu())
    k = probs.shape[1]
    aucs, f1s, accs = [], [], []
    pred = (probs >= threshold).float()
    hard = (targets >= 0.5).float()
    for c in range(k):
        auc = binary_auc(probs[:, c], hard[:, c])
        if auc == auc:  # not nan
            aucs.append(auc)
        tp = float((pred[:, c] * hard[:, c]).sum())
        fp = float((pred[:, c] * (1 - hard[:, c])).sum())
        fn = float(((1 - pred[:, c]) * hard[:, c]).sum())
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 1.0)
        accs.append(float((pred[:, c] == hard[:, c]).float().mean()))
    mean_auc = sum(aucs) / len(aucs) if aucs else 0.5
    return {
        "auc": mean_auc,
        "f1": sum(f1s) / max(len(f1s), 1),
        "acc": sum(accs) / max(len(accs), 1),
        "auc_defined_classes": float(len(aucs)),
    }


def singlelabel_metrics(probs: torch.Tensor,
                        targets: torch.Tensor) -> Dict[str, float]:
    """单标签指标：one-vs-rest 宏 AUC + argmax acc / 宏 F1。

    ``probs`` (N, K) softmax 概率，``targets`` (N,) long。
    """
    probs = probs.detach().float().cpu()
    targets = targets.detach().long().cpu()
    k = probs.shape[1]
    onehot = torch.zeros_like(probs)
    onehot[torch.arange(targets.numel()), targets] = 1.0
    m = multilabel_metrics(probs, onehot)
    pred = probs.argmax(dim=1)
    m["acc"] = float((pred == targets).float().mean())
    return m


__all__ = ["binary_auc", "multilabel_metrics", "singlelabel_metrics"]
