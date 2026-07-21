"""gen Volume* 继承 seg patch dataset + cond mixin。"""

from gentask.data.dataset.core import Volume3D, Volume3DCubic, Volume3DWhole, VolumeNpzDatasetBase
from gentask.data.dataset.cond_mixin import CondVolumeMixin
from taskcore.data.dataset import SegDataset3D, SegDataset3DCubic, SegDataset3DWhole, SegDatasetNpzBase


def test_volume_npz_base_inherits_seg_base():
    assert issubclass(VolumeNpzDatasetBase, SegDatasetNpzBase)
    assert issubclass(VolumeNpzDatasetBase, CondVolumeMixin)


def test_volume3d_inherits_seg_dataset3d():
    assert issubclass(Volume3D, SegDataset3D)
    assert issubclass(Volume3DCubic, SegDataset3DCubic)
    assert issubclass(Volume3DWhole, SegDataset3DWhole)
