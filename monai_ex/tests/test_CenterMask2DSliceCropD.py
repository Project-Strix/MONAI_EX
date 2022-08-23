import pytest

from monai_ex.transforms.croppad.dictionary import CenterMask2DSliceCropD
from monai.data import Dataset
from monai_ex.transforms import GenerateSyntheticDataD, Compose


@pytest.mark.parametrize("crop_size,crop_mode,expected", [((50,50), "single", (1,50,50)), ((50,50), "parallel", (3,50,50))])
def test_fullimiage2dslicecropd(crop_size, crop_mode, expected):
    dim = 3
    spatial_size = (100,) * dim

    generator = GenerateSyntheticDataD(
        ["image", "label"],
        *spatial_size,
        num_objs=1,
        rad_max=5,
        rad_min=4,
        noise_max=0.5,
        num_seg_classes=1,
        channel_dim=0,
    )

    source_dataset = Dataset(
        [{"image": 'dummy.nii', "label": 'dummy_label.nii'} for i in range(2)],
        transform=Compose([
            generator, 
            CenterMask2DSliceCropD(
                keys="image",
                mask_key="label",
                roi_size=crop_size,
                crop_mode=crop_mode,
                center_mode="center",
                z_axis=2,
                n_slices=3
            )
        ])
    )

    output_item = source_dataset[0]

    assert output_item['image'].shape == expected
