"""
A collection of generic interfaces for MONAI transforms.
"""

from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from monai.transforms.compose import Randomizable, Compose
from monai.transforms.transform import apply_transform
from monai.utils import ensure_tuple, get_seed


class RandomSelect(Randomizable):
    def __init__(
        self,
        transforms: Optional[Union[Sequence[Callable], Callable]] = None,
        prob: float = 0.5,
    ) -> None:
        if transforms is None:
            transforms = []
        self.transforms = ensure_tuple(transforms)
        self.set_random_state(seed=get_seed())
        self.prob = prob

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "Compose":
        super().set_random_state(seed=seed, state=state)
        for _transform in self.transforms:
            if not isinstance(_transform, Randomizable):
                continue
            _transform.set_random_state(
                seed=self.R.randint(low=0, high=np.iinfo(np.uint32).max, dtype="uint32")
            )
        return self

    def randomize(self, data: Optional[Any] = None) -> None:
        self._do_transform = self.R.random() < self.prob
        self.selected_trans = self.R.choice(self.transforms)

    def __call__(self, input_):
        self.randomize()
        if not self._do_transform:
            return input_

        return apply_transform(self.selected_trans, input_)


class ComposeEx(Compose):
    """Extension of MONAI's Compose transform.
    Extented: `first`

    Args:
        transforms (Optional[Union[Sequence[Callable], Callable]], optional): transforms. Defaults to None.
        map_items (bool, optional): If some transform takes a data item dictionary as input, and returns a
            sequence of data items in the transform chain, all following transforms
            will be applied to each item of this list if `map_items` is `True`.
            If `map_items` is `False`, the returned sequence is passed whole
            to the next callable in the chain. Defaults to True.
        unpack_items (bool, optional): _description_. Defaults to False.
        first (bool, optional): whether only extract specified keys from the first item 
            if input data is a list of dictionaries, it's used to extract the scalar data
            which doesn't have batch dim and was replicated into every dictionary 
            when decollating, like `loss`, etc.. Defaults to False.
    """
    def __init__(
        self,
        transforms: Optional[Union[Sequence[Callable], Callable]] = None,
        map_items: bool = True,
        unpack_items: bool = False,
        first: bool = False
    ) -> None:
        super(ComposeEx, self).__init__(
            transforms=transforms,
            map_items=map_items,
            unpack_items=unpack_items,
        )
        self.first = first

    def add_transforms(
        self, transforms: Optional[Union[Sequence[Callable], Callable]]
    ) -> None:
        self.transforms += ensure_tuple(transforms)

    def __call__(self, input_):
        for _transform in self.transforms:
            input_ = apply_transform(_transform, input_, self.map_items, self.unpack_items)

        if self.first:
            return input_[0]
        return input_
