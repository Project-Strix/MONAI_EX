import unittest

import numpy as np
from parameterized import parameterized

from monai_ex.transforms.croppad.array import KSpaceResample


TEST_CASE_1 = [
    {
        "pixdim": (0.5, 0.5, 0.5),
        "diagonal": False,
        "device": "CPU",
        "tolerance": 0.0001,
    },
    {
        "img": np.random.rand(1, 20, 20, 20)
    },
    np.ndarray,
    (1, 10, 10, 10),
]

TEST_CASE_2 = [
    {
        "pixdim": (1, 1, 1),
        "diagonal": False,
        "device": "CPU",
        "tolerance": 0.0001,
    },
    {
        "img": np.random.rand(1, 15, 15, 15)
    },
    np.ndarray,
    (1, 20, 20, 20),
]


class TestKSpaceResample(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_type_shape(self, input_param, input_data, expected_type, expected_shape):
        result = KSpaceResample(**input_param)(**input_data)
        self.assertIsInstance(result, expected_type)
        self.assertTupleEqual(result.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
