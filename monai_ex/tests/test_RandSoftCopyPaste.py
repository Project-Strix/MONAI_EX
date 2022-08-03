import pytest

from pathlib import Path
import nibabel as nib

import numpy as np
from monai_ex.transforms.utility.array import RandSoftCopyPaste
from monai.data.synthetic import create_test_image_3d
from monai_ex.transforms.io.array import GenerateSyntheticData


@pytest.mark.parametrize("dim", [2, 3])
def test_randsoftcopypaste(dim):
    spatial_size = (100,) * dim
    generator = GenerateSyntheticData(
        *spatial_size,
        num_objs=1,
        rad_max=10,
        rad_min=9,
        noise_max=0.2,
        num_seg_classes=1,
        channel_dim=0,
    )

    src_image, src_mask = generator(None)
    tar_image, tar_mask = generator(None)

    print("dummy data, mask shape:", src_image.shape, src_image.shape)
    print("mask label: ", np.unique(src_mask))
    sythetic_img = RandSoftCopyPaste(2, 4, label_idx=1)(src_image, src_mask, tar_image, tar_mask == 0)
    assert sythetic_img.shape == (1, *spatial_size)

    # save_fpath = Path.home() / f"sythetic_img_{dim}.nii.gz"
    # nib.save(nib.Nifti1Image(sythetic_img.squeeze(), np.eye(4)), save_fpath)

    sythetic_img = RandSoftCopyPaste(2, 4, label_idx=1)(src_image, src_mask, tar_image, None)
    assert sythetic_img.shape == (1, *spatial_size)
