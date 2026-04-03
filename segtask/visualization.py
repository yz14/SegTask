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
        semantic_classes: Number of semantic classes.
        total_slices: Total slices in 2.5D (1 for 2D/3D).
        max_samples: Maximum number of samples to visualize.
    """
    model.eval()
    image = batch["image"].to(device)
    label = batch["label"].to(device)

    pred = model(image)
    if isinstance(pred, list):
        pred = pred[0]

    B = min(image.shape[0], max_samples)
    vis_dir = Path(output_dir) / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for b in range(B):
        # Get input image (center slice for 2.5D)
        img = image[b].cpu().numpy()
        if img.shape[0] > 1:
            # Multi-channel: take center slice
            center = img.shape[0] // 2
            img_2d = img[center]
        else:
            img_2d = img[0]

        # Get label (center slice for 2.5D multi-slice)
        lbl = label[b].cpu().numpy()  # (S*C, H, W) or (C, H, W)
        if total_slices > 1:
            # Reshape: (S*C, H, W) → (S, C, H, W), take center slice
            C = semantic_classes
            S = total_slices
            lbl_r = lbl.reshape(S, C, lbl.shape[-2], lbl.shape[-1])
            center_lbl = lbl_r[S // 2]  # (C, H, W)
        else:
            center_lbl = lbl  # (C, H, W)
        lbl_idx = center_lbl.argmax(axis=0)  # (H, W)

        # Get prediction (center slice for 2.5D multi-slice)
        p = pred[b].cpu().numpy()  # (S*C, H, W) or (C, H, W)
        if total_slices > 1:
            p_r = p.reshape(S, C, p.shape[-2], p.shape[-1])
            center_p = p_r[S // 2]  # (C, H, W)
        else:
            center_p = p
        # Apply softmax (on numpy)
        center_p_exp = np.exp(center_p - center_p.max(axis=0, keepdims=True))
        center_p_prob = center_p_exp / center_p_exp.sum(axis=0, keepdims=True)
        pred_idx = center_p_prob.argmax(axis=0)  # (H, W)

        save_path = vis_dir / f"epoch{epoch + 1:03d}_sample{b}.png"
        save_prediction_grid(
            image=img_2d,
            label=lbl_idx,
            pred=pred_idx,
            save_path=str(save_path),
            title=f"Epoch {epoch + 1}, Sample {b}",
        )

    logger.info("Saved %d visualization(s) to %s", B, vis_dir)
