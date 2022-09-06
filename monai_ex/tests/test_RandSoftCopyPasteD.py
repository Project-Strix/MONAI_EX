import pytest

from pathlib import Path
import nibabel as nib

import numpy as np
from monai_ex.transforms.utility.dictionary import RandSoftCopyPasteD
from monai.data.synthetic import create_test_image_3d
from monai.data import Dataset
from monai_ex.transforms import GenerateSyntheticData, Compose, adaptor


@pytest.mark.parametrize("dim", [2, 3])
def test_randsoftcopypaste(dim):
    data_num = 2
    spatial_size = (100,) * dim
    generator = GenerateSyntheticData(
        *spatial_size,
        num_objs=1,
        rad_max=5,
        rad_min=4,
        noise_max=0.5,
        num_seg_classes=1,
        channel_dim=0,
    )

    img, msk = generator()
    volume_size = np.count_nonzero(msk)

    dummy_fpath = [{"image": "d.nii", "label": "l.nii"} for i in range(data_num)]

    source_dataset = Dataset(['dummy.nii' for i in range(data_num)], transform=generator)

    dataset = Dataset(
        dummy_fpath, transform=Compose([
            adaptor(generator, ["image", "label"]),
            RandSoftCopyPasteD(
                keys="image", mask_key="label",
                source_dataset=source_dataset,  # will generate image & mask
                k_erode=2,
                k_dilate=5,
                alpha=0.8,
                prob=1,
                source_label_value=1,
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