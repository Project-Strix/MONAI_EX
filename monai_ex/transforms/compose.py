"""
A collection of generic interfaces for MONAI transforms.
"""

from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from monai.transforms.compose import Randomizable
from monai.transforms.utils import apply_transform
from monai.utils import ensure_tuple, get_seed

class RandomSelect(Randomizable):
    def __init__(self, transforms: Optional[Union[Sequence[Callable], Callable]] = None, prob: float = 0.5) -> None:
        if transforms is None:
            transforms = []
        self.transforms = ensure_tuple(transforms)
        self.set_random_state(seed=get_seed())
        self.prob = prob
    
    def set_random_state(self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None) -> "Compose":
        super().set_random_state(seed=seed, state=state)
        for _transform in self.transforms:
            if not isinstance(_transform, Randomizable):
                continue
            _transform.set_random_state(seed=self.R.randint(low=0, high=np.iinfo(np.uint32).max, dtype="uint32"))
        return self

    def randomize(self, data: Optional[Any] = None) -> None:
        self._do_transform = self.R.random() < self.prob
        self.selected_trans = self.R.choice(self.transforms)
    
    def __call__(self, input_):
        self.randomize()
        if not self._do_transform:
            return input_

        return apply_transform(self.selected_trans, input_)