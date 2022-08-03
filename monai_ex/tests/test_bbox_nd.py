import pytest
from monai_ex.utils.misc import bbox_ND
import numpy as np

dummy_data_3d = np.zeros([10, 10, 10])
dummy_data_3d[4:7, 4:8, 4:9] = 1

dummy_data_2d = np.zeros([10, 10])
dummy_data_2d[4:7, 4:8] = 1

@pytest.mark.parametrize('data', [dummy_data_2d, dummy_data_3d])
def test_bbox_nd(data):
    bounding = bbox_ND(data, False)
    if len(data) == 3:
        assert bounding == (4, 6, 4, 7, 4, 8)
    elif len(data) == 2:
        assert bounding == (4, 6, 4, 7)

    bbox_range = bbox_ND(data, True)
    if len(data) == 3:
        assert bbox_range == (2, 3, 4)
    elif len(data) == 2:
        assert bbox_range == (2, 3)
