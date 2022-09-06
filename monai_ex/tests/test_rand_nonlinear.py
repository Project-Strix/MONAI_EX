import pytest

import numpy as np 

from monai_ex.transforms.intensity.array import RandNonlinear 
from monai_ex.transforms.intensity.dictionary import RandNonlineard


class TestRandNonlinear:

    input = np.random.rand(3, 3, 3, 4)
    vanilla_converter = RandNonlinear(prob=1)
    dict_converter = RandNonlineard(keys=['image', 'image2'], prob=1, allow_missing_keys=True)

    def test_randnonlinear(self):
        output = self.vanilla_converter(self.input)
        assert output.shape == self.input.shape
        assert not np.all(output == self.input)

    def test_randomlinear_dict(self): 
        output = self.dict_converter({'image': self.input})
        assert output['image'].shape == self.input.shape
        assert not np.all(output['image'] == self.input)

    def test_randomlinear_dict_2items(self):
        output = self.dict_converter({'image': self.input, 'image2': self.input})
        assert output['image'].shape == self.input.shape
        assert np.all(output['image'] == output['image2'])
