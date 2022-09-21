import pytest
import nibabel as nib
from ignite.engine import Engine

import numpy as np 
from monai_ex.transforms.spatial.array import KSpaceResample, KSpaceResampleEx
from monai.transforms import Spacing, ResizeWithPadOrCrop
from monai.transforms.utils import Fourier


class FourierEx(Fourier):
    def __call__(self, img, output_shape=None):
        k = self.shift_fourier(img, 3)
        print(type(k), k.dtype, np.min(k), np.max(k))
        if output_shape:
            k = ResizeWithPadOrCrop(spatial_size=output_shape)(k)
            print(np.min(k), np.max(k))
        output_data = self.inv_shift_fourier(k, 3)
        return output_data

@pytest.mark.parametrize('dim', [3])
def test_kspace_resample(dim):
    pixdim = (0.4, 0.4, 1)
    resampler = KSpaceResample(
        pixdim, diagonal=False, device='cpu', tolerance=1e-3,
    )

    resampler2 = Spacing(pixdim, align_corners=True, image_only=False)
    resampler3 = KSpaceResampleEx(
        pixdim, diagonal=False, device='cpu', tolerance=1e-3,
    )

    affine = np.eye(4)

    size = (1,) + (10,)*dim
    file_path = "/homes/clwang/Data/RJH/Moji/RJH_000151__PREPROCESSED/_STAGE_1ST_VERSION/STAGE/05_TSWI/8005_STAGE_trueSWI.nii.gz"

    nii = nib.load(file_path)
    print("original shape:", nii.shape)
    input_data = nii.get_fdata()[np.newaxis, ...]
    print("Original:", np.min(input_data), np.max(input_data))

    output, affine, new_affine = resampler(input_data, nii.affine)
    # output2, affine2, new_affine2 = resampler2(input_data, nii.affine)
    output3, affine3, new_affine3 = resampler3(input_data, nii.affine)


    # print(output.shape, output2.shape, output3.shape)
    # print(np.min(output), np.max(output))
    # print(np.min(output2), np.max(output2))
    # print(np.min(output3), np.max(output3))

    nib.save(nib.Nifti1Image(output.squeeze()-output3.squeeze(), new_affine), '/homes/clwang/Data/kresampled-diff.nii.gz')

    
