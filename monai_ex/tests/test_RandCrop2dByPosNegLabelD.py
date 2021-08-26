import unittest

import numpy as np
from parameterized import parameterized

# from monai.transforms import RandCropByPosNegLabel
from monai_ex.transforms.croppad.dictionary import RandCrop2dByPosNegLabelD


a = np.zeros([1,8,8])
a[0,4,4] = 1
TEST_CASE_1 = [
    {
        "label_key": None,
        "spatial_size": [5, 5],
        "pos": 1,
        "neg": 0,
        "offset": 1,
        "num_samples": 1,
        "image": None,
        "image_threshold": 0,
    },
    {   
        "img": np.array(
            [[[0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0],
              [0, 1, 2, 1, 2, 1, 0, 0],
              [0, 1, 3, 0, 1, 0, 0, 0],
              [0, 0, 0, 4, 5, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]]]
        ),
        'label': a
    },
    list,
    (1, 5, 5),
]


class TestRandCropByPosNegLabel(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_type_shape(self, input_param, input_data, expected_type, expected_shape):
        result = RandCrop2dByPosNegLabelD(**input_param)(**input_data)
        self.assertIsInstance(result, expected_type)
        self.assertTupleEqual(result[0].shape, expected_shape)
        print("Crop results:")
        [print(r) for r in result]

if __name__ == "__main__":
    unittest.main()