import pytest
from monai_ex.transforms.croppad.dictionary import RandSelectSliceFromImageD
import numpy as np 
import torch 

@pytest.mark.parametrize('dim', [0, 1, 2])
@pytest.mark.parametrize('use_np', [True, False])
@pytest.mark.parametrize('num', [1,2])
def test_kspace_resample(dim, use_np, num):
    cropper = RandSelectSliceFromImageD(keys=["image1", "image2"], dim=dim, num_samples=num)
    if use_np:
        image = np.random.rand(1, 4, 5, 6)
    else:
        image = torch.rand(1, 4, 5, 6)
    data = {"image1": image, "image2": image}
    ret = cropper(data)

    assert isinstance(ret, list)
    assert len(ret) == num
    # print(type(ret), ret[0]["image1"].shape)
    if dim == 0:
        assert ret[0]["image1"].shape == (1,5,6)
        assert ret[0]["image2"].shape == (1,5,6)
    elif dim == 1:
        assert ret[0]["image1"].shape == (1,4,6)
        assert ret[0]["image2"].shape == (1,4,6)
    elif dim == 2:
        assert ret[0]["image1"].shape == (1,4,5)
        assert ret[0]["image2"].shape == (1,4,5)
        if num > 2:
            assert ret[1]["image1"].shape == (1,4,5)
            assert ret[1]["image2"].shape == (1,4,5)