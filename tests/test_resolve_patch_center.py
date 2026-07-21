"""resolve_patch_center 辅助。"""

from taskcore.data.patch_extract import resolve_patch_center


def test_resolve_whole():
    assert resolve_patch_center(
        "whole", sample_z=lambda: 99, sample_center=lambda: (1, 2, 3)) == (0, 0, 0)


def test_resolve_z():
    assert resolve_patch_center(
        "z_axis", sample_z=lambda: 7, sample_center=lambda: (1, 2, 3)) == (7, 0, 0)


def test_resolve_cubic():
    assert resolve_patch_center(
        "cubic", sample_z=lambda: 0, sample_center=lambda: (1, 2, 3)) == (1, 2, 3)
