# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
from parameterized import parameterized

# from monai.transforms import RandCropByPosNegLabel
from monai_ex.transforms.croppad.array import RandCropByPosNegLabelEx

a = np.zeros([1,8,8])
a[0,4,4] = 1
TEST_CASE_1 = [
    {
        "label": None,
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

TEST_CASE_2 = [
    {
        "label": None,
        "spatial_size": [2, 2, 2],
        "pos": 1,
        "neg": 1,
        "offset": 0,
        "num_samples": 1,
        "image": np.random.randint(0, 2, size=[3, 3, 3, 3]),
        "image_threshold": 0,
    },
    {
        "img": np.random.randint(0, 2, size=[3, 3, 3, 3]),
        "label": np.random.randint(0, 2, size=[3, 3, 3, 3]),
        "image": np.random.randint(0, 2, size=[3, 3, 3, 3]),
    },
    list,
    (3, 2, 2, 2),
]


class TestRandCropByPosNegLabel(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_type_shape(self, input_param, input_data, expected_type, expected_shape):
        result = RandCropByPosNegLabelEx(**input_param)(**input_data)
        self.assertIsInstance(result, expected_type)
        self.assertTupleEqual(result[0].shape, expected_shape)
        print("Crop results:")
        [print(r) for r in result]



if __name__ == "__main__":
    unittest.main()
