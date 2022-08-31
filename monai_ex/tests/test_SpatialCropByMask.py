import pytest

import numpy as np
from monai_ex.transforms.croppad.array import SpatialCropByMask
from monai_ex.transforms.croppad.dictionary import SpatialCropByMaskD
from monai_ex.transforms import GenerateSyntheticData, GenerateSyntheticDataD


@pytest.mark.parametrize("dim", [2,3])
def test_spatialCropByMask(dim):
    spatial_size = (100,) * dim

    generator = GenerateSyntheticData(
        *spatial_size,
        num_objs=1,
        rad_max=5,
        rad_min=4,
        noise_max=0,
        num_seg_classes=1,
        channel_dim=0,
    )

    crop_size = (32,)*dim
    image, label = generator(None)
    cropper = SpatialCropByMask(roi_size=crop_size)
    img_slice = cropper(image, label)

    assert img_slice.shape == (1, *crop_size)
    assert np.count_nonzero(img_slice) > 0


@pytest.mark.parametrize("dim", [2, 3])
def test_selectslicesbymaskdict(dim):
    spatial_size = (100,) * dim
    crop_size = (32,) * dim

    generator = GenerateSyntheticDataD(
        ["image", "label"],
        *spatial_size,
        num_objs=1,
        rad_max=5,
        rad_min=4,
        noise_max=0,
        num_seg_classes=1,
        channel_dim=0,
    )

    outputs = generator({"image": "1", "label": "1"})
    cropper = SpatialCropByMaskD(keys=["image", "label"], roi_size=crop_size, mask_key="label")

    img_slice = cropper(outputs)

    assert img_slice["image"].shape == (1, *crop_size)
    assert img_slice["label"].shape == (1, *crop_size)
    assert np.count_nonzero(img_slice["image"]) > 0
    assert np.count_nonzero(img_slice["label"]) > 0
