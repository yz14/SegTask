"""Smoke test for the new bbox-aware inference path.

Validates that `Predictor.predict_volume(image_path, bbox_path=...)`:
  1. Crops to the bbox before inference,
  2. Splices the prediction back into the original (D_orig, H_orig, W_orig)
     coordinate system,
  3. Outside-bbox voxels stay at probability 0 → label = background.

Uses a tiny *stub* model (returns random logits of the requested shape)
so we don't need a trained checkpoint. Patch mode = "whole" for speed.

Run:
    conda activate torch27_env
    python smoke_test_bbox_predict.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from segtask_v1.config import Config  # noqa: E402
from segtask_v1.predictor import Predictor  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("smoke_predict")

IMG = r"F:/med_data/Totalsegmentator_dataset_v201/small_data/nii/s0000.nii.gz"
BBX = r"F:/med_data/Totalsegmentator_dataset_v201/small_data/bbox/s0000.nii.gz"


class StubModel(nn.Module):
    """Returns sigmoid-friendly logits of shape (B, num_fg, D, H, W).

    Logits are deterministic from a fixed seed so the test is reproducible.
    """

    def __init__(self, num_fg: int):
        super().__init__()
        self.num_fg = num_fg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, C, D, H, W) for 3D path. Output channels = num_fg.
        B, _, D, H, W = x.shape
        g = torch.Generator(device=x.device).manual_seed(0)
        return torch.randn(
            (B, self.num_fg, D, H, W),
            generator=g, device=x.device, dtype=x.dtype)


def make_cfg() -> Config:
    cfg = Config()
    cfg.data.image_dir = ""  # unused here
    cfg.data.label_dir = ""
    cfg.data.label_values = [0, 1, 2]
    cfg.data.num_classes = 3
    cfg.data.patch_mode = "whole"
    cfg.data.patch_size = [16, 32, 32]
    cfg.data.multi_res_scales = [1.0]
    cfg.data.intensity_min = -1024.0
    cfg.data.intensity_max = 3071.0
    cfg.data.normalize = "minmax"
    cfg.predict.batch_size = 1
    cfg.predict.tta_flip = False
    cfg.predict.threshold = 0.5
    cfg.predict.blend_mode = "gaussian"
    cfg.train.use_amp = False  # CPU smoke
    cfg.sync()
    return cfg


def main() -> None:
    cfg = make_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    model = StubModel(num_fg=cfg.num_fg_classes).to(device).eval()
    pred = Predictor(model, cfg, device)

    # 1. Without bbox — sanity baseline.
    out = pred.predict_volume(IMG)
    raw_shape = out["label_map"].shape
    log.info("[no-bbox] label_map=%s prob=%s",
             raw_shape, out["probabilities"].shape)
    assert out["probabilities"].shape == (cfg.num_fg_classes,) + raw_shape

    # 2. With bbox — output must keep full-volume shape, outside zero.
    out_b = pred.predict_volume(IMG, bbox_path=BBX)
    log.info("[with-bbox] label_map=%s prob=%s",
             out_b["label_map"].shape, out_b["probabilities"].shape)
    assert out_b["label_map"].shape == raw_shape, (
        f"bbox mode shape {out_b['label_map'].shape} != raw {raw_shape}")
    assert out_b["probabilities"].shape == out["probabilities"].shape

    # Compute the bbox we expect the predictor to have used and verify
    # that everything OUTSIDE that bbox is exactly 0 in the prob map.
    from segtask_v1.data.dataset import compute_bbox_from_volume, load_nifti
    bbox = compute_bbox_from_volume(load_nifti(BBX))
    assert bbox is not None
    (d0, d1), (h0, h1), (w0, w1) = bbox
    prob = out_b["probabilities"]
    outside = prob.copy()
    outside[:, d0:d1, h0:h1, w0:w1] = 0
    assert outside.max() == 0, (
        f"Probabilities outside bbox should be 0 but got max={outside.max()}")
    inside_max = prob[:, d0:d1, h0:h1, w0:w1].max()
    log.info("Inside-bbox prob max=%.4f, outside max=%.4f",
             inside_max, outside.max())

    # And label_map outside bbox must equal the background label.
    bg = cfg.data.label_values[0]
    lbl_outside = out_b["label_map"].copy()
    lbl_outside[d0:d1, h0:h1, w0:w1] = bg
    uniq = np.unique(lbl_outside).tolist()
    assert uniq == [bg], (
        f"Outside-bbox label_map should be all bg={bg}, got {uniq}")

    log.info("ALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
