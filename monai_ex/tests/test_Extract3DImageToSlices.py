import pytest

import numpy as np
from monai_ex.transforms.croppad.array import Extract3DImageToSlices
from monai_ex.transforms.croppad.dictionary import Extract3DImageToSlicesd
from monai_ex.transforms import GenerateSyntheticData, GenerateSyntheticDataD

def test_selectslicesbymask():
    dim = 3
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

    image, label = generator(None)
    cropper = Extract3DImageToSlices(axis=2)
    img_slice = cropper(image)[0]

    assert img_slice.shape == (1, 100, 100)


def test_selectslicesbymaskdict():
    dim = 3
    spatial_size = (100,) * dim

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
    cropper = Extract3DImageToSlicesd(keys=["image", "label"], axis=2)

    img_slice = cropper(outputs)

    assert len(img_slice) == 100
    assert img_slice[1]["image"].shape == (1, 100, 100)
    assert img_slice[1]["label"].shape == (1, 100, 100)
    assert "image_meta_dict" in img_slice[0] and isinstance(img_slice[0]["image_meta_dict"], dict)
