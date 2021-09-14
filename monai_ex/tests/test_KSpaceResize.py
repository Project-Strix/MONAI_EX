import unittest

import numpy as np
from parameterized import parameterized

from monai_ex.transforms.croppad.array import KSpaceResize


TEST_CASE_1 = [
    {
        "roi_size": (10, 10, 10),
        "as_tensor_output": False,
        "device": None,
    },
    {
        "img": np.random.rand(1, 20, 20, 20)
    },
    np.ndarray,
    (1, 10, 10, 10),
]

TEST_CASE_2 = [
    {
        "roi_size": (20, 20, 20),
        "as_tensor_output": False,
        "device": None,
    },
    {
        "img": np.random.rand(1, 15, 15, 15)
    },
    np.ndarray,
    (1, 20, 20, 20),
]


class TestKSpaceResize(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_type_shape(self, input_param, input_data, expected_type, expected_shape):
        result = KSpaceResize(**input_param)(**input_data)
        self.assertIsInstance(result, expected_type)
        self.assertTupleEqual(result.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
