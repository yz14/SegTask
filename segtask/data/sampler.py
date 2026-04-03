"""Custom samplers for class-balanced and foreground-aware sampling.

Provides:
- ClassBalancedSampler: oversample minority classes based on pixel frequency
- ForegroundSampler: oversample slices/patches containing foreground structures
"""

from __future__ import annotations

import logging
from typing import Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class ClassBalancedSampler(Sampler[int]):
    """Weighted random sampler based on per-sample class distribution.

    Each sample is assigned a weight proportional to the inverse frequency
    of the rarest class it contains. This ensures minority-class samples
    are drawn more frequently.
    """

    def __init__(
        self,
        sample_weights: List[float],
        num_samples: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Args:
            sample_weights: Per-sample sampling weight.
            num_samples: Number of samples per epoch (default: len(sample_weights)).
            seed: Random seed for reproducibility.
        """
        self.weights = torch.as_tensor(sample_weights, dtype=torch.float64)
        self.num_samples = num_samples or len(sample_weights)
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch)
        indices = torch.multinomial(
            self.weights, self.num_samples, replacement=True, generator=g
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


def compute_sample_weights_from_labels(
    dataset,
    num_classes: int,
    class_weights: Optional[List[float]] = None,
) -> List[float]:
    """Compute per-sample weights from label class distribution.

    For each sample, weight = max over classes of (class_weight[c] * has_class[c]).
    This biases sampling toward rare-class samples.

    Args:
        dataset: A dataset with __len__ and __getitem__ returning dict with 'label'.
        num_classes: Number of classes.
        class_weights: Optional per-class weights. If None, uses inverse frequency.

    Returns:
        List of per-sample weights.
    """
    logger.info("Computing sample weights for %d samples...", len(dataset))

    # Count class pixel frequencies across all samples
    class_pixel_counts = np.zeros(num_classes, dtype=np.float64)
    sample_class_presence = []

    for i in range(len(dataset)):
        sample = dataset[i]
        label = sample["label"]  # (num_classes, ...) tensor
        if isinstance(label, torch.Tensor):
            label = label.numpy()

        # Per-class presence: fraction of pixels belonging to each class
        presence = np.zeros(num_classes)
        for c in range(min(num_classes, label.shape[0])):
            n_pixels = label[c].sum()
            class_pixel_counts[c] += n_pixels
            presence[c] = n_pixels
        sample_class_presence.append(presence)

    # Compute class weights from inverse frequency if not provided
    if class_weights is None or len(class_weights) == 0:
        total = class_pixel_counts.sum()
        if total > 0:
            freq = class_pixel_counts / total
            # Inverse frequency, clamped to avoid extreme weights
            cw = np.where(freq > 0, 1.0 / (freq * num_classes), 1.0)
            cw = np.clip(cw, 0.1, 10.0)
        else:
            cw = np.ones(num_classes)
    else:
        cw = np.array(class_weights[:num_classes])

    # Per-sample weight = max class weight among present classes
    weights = []
    for presence in sample_class_presence:
        # Weight by rarest present class
        mask = presence > 0
        if mask.any():
            w = (cw * mask).max()
        else:
            w = 1.0
        weights.append(float(w))

    logger.info(
        "Class pixel counts: %s, Class weights: %s",
        [f"{c:.0f}" for c in class_pixel_counts],
        [f"{w:.3f}" for w in cw],
    )

    return weights
