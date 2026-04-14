"""Training visualization: save sample predictions as images.

Generates side-by-side comparisons of input, ground truth, and
model predictions during training for visual quality monitoring.
Uses matplotlib for rendering — saved as PNG files, no display needed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def save_prediction_grid(
    image: np.ndarray,
    label: np.ndarray,
    pred: np.ndarray,
    save_path: str,
    class_names: Optional[List[str]] = None,
    title: str = "",
) -> None:
    """Save a grid of input / ground truth / prediction for visual comparison.

    Args:
        image: Single 2D slice (H, W), normalized to [0, 1].
        label: Integer label map (H, W) with class indices.
        pred: Integer prediction map (H, W) with class indices.
        save_path: Output PNG path.
        class_names: Optional list of class names for legend.
        title: Optional title string.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        logger.warning("matplotlib not available — skipping visualization.")
        return

    num_classes = max(int(label.max()), int(pred.max())) + 1

    # Create a colormap for segmentation overlays
    base_colors = [
        [0, 0, 0, 0],       # background: transparent
        [1, 0, 0, 0.5],     # class 1: red
        [0, 1, 0, 0.5],     # class 2: green
        [0, 0, 1, 0.5],     # class 3: blue
        [1, 1, 0, 0.5],     # class 4: yellow
        [1, 0, 1, 0.5],     # class 5: magenta
        [0, 1, 1, 0.5],     # class 6: cyan
    ]
    colors = base_colors[:num_classes] + base_colors[1:] * ((num_classes // 6) + 1)
    colors = colors[:num_classes]
    cmap = ListedColormap(colors)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Input image
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Input")
    axes[0].axis("off")

    # Ground truth overlay
    axes[1].imshow(image, cmap="gray", vmin=0, vmax=1)
    mask = np.ma.masked_where(label == 0, label)
    axes[1].imshow(mask, cmap=cmap, vmin=0, vmax=num_classes - 1, interpolation="nearest")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Prediction overlay
    axes[2].imshow(image, cmap="gray", vmin=0, vmax=1)
    mask_pred = np.ma.masked_where(pred == 0, pred)
    axes[2].imshow(mask_pred, cmap=cmap, vmin=0, vmax=num_classes - 1, interpolation="nearest")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _prob_to_label(logits: np.ndarray, is_per_class: bool, threshold: float = 0.5) -> np.ndarray:
    """Convert (C, H, W) logits to (H, W) integer label map.

    For per_class (sigmoid): threshold each channel, assign fg class+1, bg=0.
    For softmax: softmax then argmax.
    """
    if is_per_class:
        prob = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
        fg_max = prob.max(axis=0)
        fg_argmax = prob.argmax(axis=0)
        return np.where(fg_max > threshold, fg_argmax + 1, 0)
    else:
        p_exp = np.exp(logits - logits.max(axis=0, keepdims=True))
        p_prob = p_exp / p_exp.sum(axis=0, keepdims=True)
        return p_prob.argmax(axis=0)


@torch.no_grad()
def visualize_batch(
    model: torch.nn.Module,
    batch: dict,
    epoch: int,
    output_dir: str,
    device: torch.device,
    semantic_classes: int,
    total_slices: int = 1,
    max_samples: int = 4,
    output_mode: str = "softmax",
) -> None:
    """Generate and save visualization for a batch of samples.

    Picks the center slice for 2.5D, generates prediction, and saves
    side-by-side comparison images.

    Args:
        model: Trained model (in eval mode).
        batch: Data batch dict with 'image' and 'label' tensors.
        epoch: Current epoch number (for filename).
        output_dir: Directory to save images.
        device: Computation device.
        semantic_classes: Number of semantic classes (including bg).
        total_slices: Total slices in 2.5D (1 for 2D/3D).
        max_samples: Maximum number of samples to visualize.
        output_mode: "softmax" or "per_class".
    """
    is_per_class = (output_mode == "per_class")
    model.eval()
    image = batch["image"].to(device)
    label = batch["label"].to(device)

    # 2.5D: squeeze channel dim before model input
    if total_slices > 1:
        image = image.squeeze(1)  # (B, 1, D, H, W) → (B, D, H, W)

    pred = model(image)
    if isinstance(pred, list):
        pred = pred[0]

    B = min(image.shape[0], max_samples)
    vis_dir = Path(output_dir) / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for b in range(B):
        # Get input image as 2D slice
        img = image[b].cpu().numpy()
        if total_slices > 1:
            # 2.5D: (D, H, W) — take center slice
            img_2d = img[img.shape[0] // 2]
        elif img.ndim == 4:
            # 3D: (1, D, H, W) — take center D slice
            img_2d = img[0, img.shape[1] // 2]
        elif img.shape[0] > 1:
            img_2d = img[img.shape[0] // 2]
        else:
            # 2D: (1, H, W)
            img_2d = img[0]

        # Get label as 2D class-index map
        lbl = label[b].cpu().numpy()  # (C, D, H, W) or (C, H, W)
        if lbl.ndim == 4:
            center_d = lbl.shape[1] // 2
            lbl_2d = lbl[:, center_d]  # (C, H, W)
        else:
            lbl_2d = lbl  # (C, H, W)
        if is_per_class:
            # Fg-only channels: argmax gives 0-based fg index, +1 for display
            fg_max = lbl_2d.max(axis=0)  # any fg present?
            lbl_idx = np.where(fg_max > 0.5, lbl_2d.argmax(axis=0) + 1, 0)
        else:
            lbl_idx = lbl_2d.argmax(axis=0)  # (H, W)

        # Get prediction as 2D class-index map
        p = pred[b].cpu().numpy()
        if total_slices > 1:
            # 2.5D: (N*S, H, W) → (N, S, H, W) → softmax over S → center slice
            N = semantic_classes
            S = total_slices
            p = p.reshape(N, S, p.shape[-2], p.shape[-1])
            p_exp = np.exp(p - p.max(axis=1, keepdims=True))
            p_prob = p_exp / p_exp.sum(axis=1, keepdims=True)
            center_idx = S // 2
            p_prob = p_prob[:, center_idx]  # (N, H, W)
            pred_idx = p_prob.argmax(axis=0)  # (H, W)
        elif p.ndim == 4:
            # 3D: (C, D, H, W) → center D slice
            center_d = p.shape[1] // 2
            p_slice = p[:, center_d]  # (C, H, W)
            pred_idx = _prob_to_label(p_slice, is_per_class)
        else:
            # 2D: (C, H, W)
            pred_idx = _prob_to_label(p, is_per_class)

        save_path = vis_dir / f"epoch{epoch + 1:03d}_sample{b}.png"
        save_prediction_grid(
            image=img_2d,
            label=lbl_idx,
            pred=pred_idx,
            save_path=str(save_path),
            title=f"Epoch {epoch + 1}, Sample {b}",
        )

    logger.info("Saved %d visualization(s) to %s", B, vis_dir)
