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
    cropper = SelectSlicesByMask(axis=2, slice_select_mode='center', mask_data=label)
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
    cropper = SelectSlicesByMaskD(keys=["image", "label"], mask_key="label", axis=2, slice_select_mode=mode)

    img_slice = cropper(outputs)

    if mode == "center" or mode == "maximum":
        assert img_slice[0]["image"].shape == (1, 100, 100)
        assert img_slice[0]["label"].shape == (1, 100, 100)
        assert np.count_nonzero(img_slice[0]["image"]) > 0
        assert np.count_nonzero(img_slice[0]["label"]) > 0
        assert "image_meta_dict" in img_slice[0] and isinstance(img_slice[0]["image_meta_dict"], dict)
    elif mode == "all":
        assert len(img_slice) > 1
        assert img_slice[1]["image"].shape == (1, 100, 100)
        assert img_slice[1]["label"].shape == (1, 100, 100)
        assert np.count_nonzero(img_slice[1]["image"]) > 0
        assert np.count_nonzero(img_slice[1]["label"]) > 0
        assert "image_meta_dict" in img_slice[0] and isinstance(img_slice[0]["image_meta_dict"], dict)


def test_selectslicesbymaskdict_discontinuous():
    dim = 3
    spatial_size = (100,) * dim

    shape = (1, 20, 20, 20)
    label = np.zeros(shape)
    image = np.zeros_like(label)
    label[..., 5] = 1
    label[..., 8] = 1
    label[..., 10] = 1

    cropper = SelectSlicesByMask(axis=2, slice_select_mode="all", mask_data=label)

    img_slice = cropper(image)

    assert img_slice[1].shape == shape[:-1]
    assert len(img_slice) == 3
