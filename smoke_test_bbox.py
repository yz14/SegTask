"""Smoke test for the new bbox ROI cropping in the data pipeline.

Run:
    conda activate torch27_env
    python smoke_test_bbox.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make the in-tree package importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from segtask_v1.data.dataset import (  # noqa: E402
    SegDataset3D, SegDataset3DCubic, SegDataset3DWhole,
    load_nifti, precompute_bboxes)
from segtask_v1.data.loader import (  # noqa: E402
    detect_label_values, discover_samples, match_bbox_paths)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("test_bbox")


IMG_DIR  = r"F:/med_data/Totalsegmentator_dataset_v201/small_data/nii"
LBL_DIR  = r"F:/med_data/Totalsegmentator_dataset_v201/small_data/mask"
BBOX_DIR = r"F:/med_data/Totalsegmentator_dataset_v201/small_data/bbox"


def main() -> None:
    img_paths, lbl_paths = discover_samples(IMG_DIR, LBL_DIR)
    log.info("Discovered %d image-label pairs.", len(img_paths))

    bbox_paths = match_bbox_paths(img_paths, BBOX_DIR, ".nii.gz", ".nii.gz")
    assert len(bbox_paths) == len(img_paths)

    bboxes = precompute_bboxes(bbox_paths)
    assert any(b is not None for b in bboxes), "All bboxes are empty!"

    raw_shape = load_nifti(img_paths[0]).shape
    bb        = bboxes[0]
    assert bb is not None, "First sample's bbox is empty"
    (d0, d1), (h0, h1), (w0, w1) = bb
    crop_shape = (d1 - d0, h1 - h0, w1 - w0)
    log.info("Sample 0 raw shape=%s, bbox crop shape=%s", raw_shape, crop_shape)
    assert all(c <= r for c, r in zip(crop_shape, raw_shape))
    assert any(c <  r for c, r in zip(crop_shape, raw_shape)), (
        "BBox crop did not shrink any axis — bbox mask might be wrong.")

    label_values = detect_label_values(lbl_paths)
    log.info("Detected label values: %s", label_values)

    common = dict(
        image_paths=img_paths,
        label_paths=lbl_paths,
        bbox_paths=bbox_paths,
        label_values=label_values,
        patch_size=(32, 64, 64),
        samples_per_volume=1,
        cache_enabled=False)

    log.info("---- SegDataset3D (z_axis) ----")
    ds_z = SegDataset3D(**common, multi_res_scales=[1.0])
    s = ds_z[0]
    log.info("z_axis image=%s label=%s",
             tuple(s["image"].shape), tuple(s["label"].shape))
    assert s["image"].shape == (1, 32, 64, 64)

    log.info("---- SegDataset3DCubic ----")
    ds_c = SegDataset3DCubic(**common, multi_res_scales=[1.0])
    s = ds_c[0]
    log.info("cubic image=%s label=%s",
             tuple(s["image"].shape), tuple(s["label"].shape))
    assert s["image"].shape == (1, 32, 64, 64)

    log.info("---- SegDataset3DWhole ----")
    ds_w = SegDataset3DWhole(**common)
    s = ds_w[0]
    log.info("whole image=%s label=%s",
             tuple(s["image"].shape), tuple(s["label"].shape))
    assert s["image"].shape == (1, 32, 64, 64)

    log.info("ALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
