import pytest
from monai_ex.transforms.croppad.array import RandSelectSlicesFromImage
import numpy as np 
import torch 

@pytest.mark.parametrize('dim', [0, 1, 2])
@pytest.mark.parametrize('use_np', [True, False])
@pytest.mark.parametrize('num', [1, 2])
def test_kspace_resample(dim, use_np, num):
    cropper = RandSelectSlicesFromImage(dim, num_samples=num)
    if use_np:
        image = np.random.rand(1, 4, 5, 6)
    else:
        image = torch.rand(1, 4, 5, 6)
    ret = cropper(image)

    if dim == 0:
        assert ret[0].shape == (1,5,6)
    elif dim == 1:
        assert ret[0].shape == (1,4,6)
    elif dim == 2:
        assert ret[0].shape == (1,4,5)    
        if num>1:
            assert ret[1].shape == (1,4,5)
