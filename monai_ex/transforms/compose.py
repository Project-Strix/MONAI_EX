"""
A collection of generic interfaces for MONAI transforms.
"""

from typing import Any, Callable, Optional, Sequence, Union, TypeVar, List
import logging

import torch
import numpy as np

from monai.transforms.compose import Randomizable, Compose
from monai.transforms.transform import apply_transform
from monai.transforms.utility.array import DataStats
from monai.utils.misc import ensure_tuple, get_seed
from monai_ex.utils.exceptions import TransformException, trycatch

ReturnType = TypeVar("ReturnType")

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


ReturnType = TypeVar("ReturnType")

def _apply_transform(
    transform: Callable[..., ReturnType], parameters: Any, unpack_parameters: bool = False
) -> ReturnType:
    """
    Perform transformation `transform` with the provided parameters `parameters`.

    If `parameters` is a tuple and `unpack_items` is True, each parameter of `parameters` is unpacked
    as arguments to `transform`.
    Otherwise `parameters` is considered as single argument to `transform`.

    Args:
        transform: a callable to be used to transform `data`.
        parameters: parameters for the `transform`.
        unpack_parameters: whether to unpack parameters for `transform`. Defaults to False.

    Returns:
        ReturnType: The return type of `transform`.
    """
    if isinstance(parameters, tuple) and unpack_parameters:
        return transform(*parameters)

    return transform(parameters)

def apply_transform_ex(
    transform: Callable[..., ReturnType],
    data: Any,
    map_items: bool = True,
    unpack_items: bool = False,
    debug_info=True,
) -> Union[List[ReturnType], ReturnType]:
    """ Extension of MONAI's apply_transform function. Able to turnoff debug info.
    """

    try:
        if isinstance(data, (list, tuple)) and map_items:
            return [_apply_transform(transform, item, unpack_items) for item in data]
        return _apply_transform(transform, data, unpack_items)
    except Exception as e:

        if not isinstance(transform, Compose) and debug_info:
            # log the input data information of exact transform in the transform chain
            datastats = DataStats(data_shape=False, value_range=False)
            logger = logging.getLogger(datastats._logger_name)
            logger.info(f"\n=== Transform input info -- {type(transform).__name__} ===")
            if isinstance(data, (list, tuple)):
                data = data[0]

            def _log_stats(data, prefix: Optional[str] = "Data"):
                if isinstance(data, (np.ndarray, torch.Tensor)):
                    # log data type, shape, range for array
                    datastats(img=data, data_shape=True, value_range=True, prefix=prefix)  # type: ignore
                else:
                    # log data type and value for other meta data
                    datastats(img=data, data_value=True, prefix=prefix)

            if isinstance(data, dict):
                for k, v in data.items():
                    _log_stats(data=v, prefix=k)
            else:
                _log_stats(data=data)
        raise TransformException(f"applying transform {transform}") from e


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

    @trycatch()
    def __call__(self, input_):
        for _transform in self.transforms:
            input_ = apply_transform_ex(_transform, input_, self.map_items, self.unpack_items, debug_info=False)

        if self.first:
            return input_[0]
        return input_
