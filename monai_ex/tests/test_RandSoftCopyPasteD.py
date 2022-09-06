import pytest

from pathlib import Path
import nibabel as nib

import numpy as np
from monai_ex.transforms.utility.dictionary import RandSoftCopyPasteD
from monai.data.synthetic import create_test_image_3d
from monai.data import Dataset
from monai_ex.transforms import MapTransform, GenerateSyntheticData, Compose, adaptor



class GenerateSyntheticDataD(MapTransform):
    def __init__(
        self,
        keys,
        label_key,
        height: int,
        width: int,
        depth: int = None,
        num_objs: int = 12,
        rad_max: int = 30,
        rad_min: int = 5,
        noise_max: float = 0.0,
        num_seg_classes: int = 5,
        channel_dim: int = None,
        random_state: np.random.RandomState = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)

        self.label_key = label_key
        self.loader = GenerateSyntheticData(
            height,
            width,
            depth,
            num_objs,
            rad_max,
            rad_min,
            noise_max,
            num_seg_classes,
            channel_dim,
            random_state,
        )

    def __call__(self, filename: dict):
        test_data = self.loader(None)

        data = {}
        for key in self.keys:
            data[key] = test_data[0]
        data[self.label_key] = test_data[1]
        return data


@pytest.mark.parametrize("dim", [2, 3])
def test_randsoftcopypaste(dim):
    data_num = 2
    spatial_size = (100,) * dim
    generator = GenerateSyntheticDataD(
        "image",
        "label",
        *spatial_size,
        num_objs=1,
        rad_max=5,
        rad_min=4,
        noise_max=0.5,
        num_seg_classes=1,
        channel_dim=0,
    )

    dummy_fpath = [{"image": "d.nii", "label": "l.nii"} for i in range(data_num)]

    output = generator(dummy_fpath[0])
    volume_size = np.count_nonzero(output["label"])

    source_dataset = Dataset(
        [{"image": 'dummy.nii', "label": 'dummy_label.nii'} for i in range(data_num)],
        transform=generator
    )

    dataset = Dataset(
        dummy_fpath, transform=Compose([
            generator,
            RandSoftCopyPasteD(
                keys="image", mask_key="label",
                source_dataset=source_dataset,  # will generate image & mask
                source_fg_key="label",
                source_fg_value=1,
                k_erode=2,
                k_dilate=5,
                alpha=0.8,
                prob=1,
                mask_select_fn=lambda x: x == 0,
            )
        ])
    )

    for i, item in enumerate(dataset):
        image, label = item["image"], item["label"]

        # save_fpath = Path.home() / f"sythetic_{dim}Dimg_{i}.nii.gz"
        # nib.save(nib.Nifti1Image(image.squeeze(), np.eye(4)), save_fpath)
        # save_fpath = Path.home() / f"sythetic_{dim}Dlabel_{i}.nii.gz"
        # nib.save(nib.Nifti1Image(label.squeeze(), np.eye(4)), save_fpath)

        assert volume_size < np.count_nonzero(label) <= 2 * volume_size



@pytest.mark.parametrize("dim", [2, 3])
def test_randsoftcopypaste_multiimage(dim):
    data_num = 2
    spatial_size = (100,) * dim
    generator = GenerateSyntheticDataD(
        ["image1", "image2"],
        "label",
        *spatial_size,
        num_objs=1,
        rad_max=5,
        rad_min=4,
        noise_max=0,
        num_seg_classes=1,
        channel_dim=0,
    )

    dummy_fpath = [{"image1": "d.nii", "image2": "d.nii", "label": "l.nii"} for i in range(data_num)]
    source_dataset = Dataset(
        [{"image1": '1', "image2": "2", "label": '1'} for i in range(data_num)],
        transform=generator
    )

    outputs = generator({"image1": '1', "image2": '2', "label": '1'})

    generator = RandSoftCopyPasteD(
        keys=["image1", "image2"], mask_key="label",
        source_dataset=source_dataset,
        source_fg_key="label",
        source_fg_value=1,
        k_erode=2,
        k_dilate=5,
        alpha=0.8,
        prob=1,
        mask_select_fn=lambda x: x == 0,
    )

    generated_item = generator(outputs)
    assert generated_item["image1"].shape == (1, *spatial_size)
    assert generated_item["image2"].shape == (1, *spatial_size)
    assert np.all(generated_item["image1"] == generated_item["image2"])
