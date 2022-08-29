import pytest

import numpy as np
from monai_ex.transforms.croppad.array import SelectSlicesByMask
from monai_ex.transforms.croppad.dictionary import SelectSlicesByMaskD
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
    cropper = SelectSlicesByMask(z_axis=2, slice_select_mode='center', mask_data=label)
    img_slice = cropper(image)[0]

    assert img_slice.shape == (1, 100, 100)
    assert np.count_nonzero(img_slice) > 0


@pytest.mark.parametrize("mode", ["center", "all", "maximum"])
def test_selectslicesbymaskdict(mode):
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
    cropper = SelectSlicesByMaskD(keys=["image", "label"], mask_key="label", z_axis=2, slice_select_mode=mode)

    img_slice = cropper(outputs)

    if mode == "center" or mode == "maximum":
        assert img_slice[0]["image"].shape == (1, 100, 100)
        assert img_slice[0]["label"].shape == (1, 100, 100)
        assert np.count_nonzero(img_slice[0]["image"]) > 0
        assert np.count_nonzero(img_slice[0]["label"]) > 0
    elif mode == "all":
        assert len(img_slice) > 1
        assert img_slice[1]["image"].shape == (1, 100, 100)
        assert img_slice[1]["label"].shape == (1, 100, 100)
        assert np.count_nonzero(img_slice[1]["image"]) > 0
        assert np.count_nonzero(img_slice[1]["label"]) > 0
