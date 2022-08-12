import pytest

from pathlib import Path
import nibabel as nib

import numpy as np
from monai_ex.transforms.utility.array import RandSoftCopyPaste
from monai.data.synthetic import create_test_image_3d
from monai_ex.transforms.io.array import GenerateSyntheticData


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("prob", [0, 1])
def test_randsoftcopypaste(dim, prob):
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
    volume_size = np.count_nonzero(src_mask) + np.count_nonzero(tar_mask)

    print("dummy data, mask shape:", src_image.shape, src_image.shape)
    print("mask label: ", np.unique(src_mask))
    sythetic_img, sythetic_msk = RandSoftCopyPaste(
        2, 4, prob=prob, mask_select_fn=lambda x: x==0, source_label_value=1
    )(tar_image, tar_mask, src_image, src_mask)
    if prob == 0:
        assert np.all(sythetic_img == tar_image)
        assert np.all(sythetic_msk == tar_mask)
    else:
        assert sythetic_img.shape == (1, *spatial_size)
        assert volume_size/2 <= np.count_nonzero(sythetic_msk) <= volume_size

    # save_fpath = Path.home() / f"sythetic_img_{dim}.nii.gz"
    # nib.save(nib.Nifti1Image(sythetic_img.squeeze(), np.eye(4)), save_fpath)

    sythetic_img, sythetic_msk = RandSoftCopyPaste(
        2, 4, prob=prob, source_label_value=1
    )(tar_image, None, src_image, src_mask)
    if prob == 0:
        assert np.all(sythetic_img == tar_image)
        assert sythetic_msk is None
    else:
        assert sythetic_img.shape == (1, *spatial_size)
        assert volume_size/2 <= np.count_nonzero(sythetic_msk) <= volume_size
