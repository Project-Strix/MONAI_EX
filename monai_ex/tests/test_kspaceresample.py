import pytest
import nibabel as nib
from ignite.engine import Engine

import numpy as np 
from monai_ex.transforms.spatial.array import KSpaceResample
from monai.transforms import Spacing, ResizeWithPadOrCrop
from monai.transforms.utils import Fourier


@pytest.mark.parametrize('dim', [2, 3])
def test_kspace_resample(dim):
    pixdim = [0.5,] * dim
    resampler = KSpaceResample(pixdim, diagonal=False, device='cpu', tolerance=1e-3)

    resampler2 = Spacing(pixdim, align_corners=True, image_only=False)
    affine = np.eye(4)

    input_data = np.zeros([1,] + [10,] * dim)

    output, affine, new_affine = resampler(input_data, affine)
    output2, affine2, new_affine2 = resampler2(input_data, affine)

    assert output.shape == output2.shape
