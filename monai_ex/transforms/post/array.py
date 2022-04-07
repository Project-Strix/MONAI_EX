import warnings
from typing import Callable, Iterable, Optional, Sequence, Union

import numpy as np
import torch

from monai.config.type_definitions import NdarrayOrTensor
from monai.networks import one_hot
from monai.networks.layers import GaussianFilter, apply_filter
from monai.transforms.transform import Transform
from monai.transforms.utils import fill_holes, get_largest_connected_component_mask
from monai.transforms.utils_pytorch_numpy_unification import unravel_index
from monai.utils import TransformBackends, convert_data_type, deprecated_arg, ensure_tuple, look_up_option
from monai.utils.type_conversion import convert_to_dst_type


class AsDiscreteEx(Transform):
    """
    Convert the input tensor/array into discrete values, possible operations are:
        -  execute `argmax`.
        -  threshold input value to binary values.
        -  convert input value to One-Hot format.
        -  round the value to the closest integer.
    Args:
        argmax: whether to execute argmax function on input data before transform.
            Defaults to ``False``.
        to_onehot: if not None, convert input data into the one-hot format with specified number of classes.
            Defaults to ``None``.
        threshold: if not None, threshold the float values to int number 0 or 1 with specified theashold.
            Defaults to ``None``.
        rounding: if not None, round the data according to the specified option,
            available options: ["torchrounding"].
        kwargs: additional parameters to `torch.argmax`, `monai.networks.one_hot`.
            currently ``dim``, ``keepdim``, ``dtype`` are supported, unrecognized parameters will be ignored.
            These default to ``0``, ``True``, ``torch.float`` respectively.
    Example:
        >>> transform = AsDiscrete(argmax=True)
        >>> print(transform(np.array([[[0.0, 1.0]], [[2.0, 3.0]]])))
        # [[[1.0, 1.0]]]
        >>> transform = AsDiscrete(threshold=0.6)
        >>> print(transform(np.array([[[0.0, 0.5], [0.8, 3.0]]])))
        # [[[0.0, 0.0], [1.0, 1.0]]]
        >>> transform = AsDiscrete(argmax=True, to_onehot=2, threshold=0.5)
        >>> print(transform(np.array([[[0.0, 1.0]], [[2.0, 3.0]]])))
        # [[[0.0, 0.0]], [[1.0, 1.0]]]
    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        argmax: bool = False,
        to_onehot: Optional[int] = None,
        threshold: Optional[float] = None,
        rounding: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.argmax = argmax
        self.to_onehot = to_onehot
        self.threshold = threshold

        self.rounding = rounding
        self.kwargs = kwargs
        
    def __call__(
        self,
        img: NdarrayOrTensor,
        argmax: Optional[bool] = None,
        to_onehot: Optional[int] = None,
        threshold: Optional[float] = None,
        rounding: Optional[str] = None
    ) -> NdarrayOrTensor:
        """
        Args:
            img: the input tensor data to convert, if no channel dimension when converting to `One-Hot`,
                will automatically add it.
            argmax: whether to execute argmax function on input data before transform.
                Defaults to ``self.argmax``.
            to_onehot: if not None, convert input data into the one-hot format with specified number of classes.
                Defaults to ``self.to_onehot``.
            threshold: if not None, threshold the float values to int number 0 or 1 with specified threshold value.
                Defaults to ``self.threshold``.
            rounding: if not None, round the data according to the specified option,
                available options: ["torchrounding"].
        """

        img_t: torch.Tensor
        img_t, *_ = convert_data_type(img, torch.Tensor)  # type: ignore
        if argmax or self.argmax:
            img_t = torch.argmax(img_t, dim=self.kwargs.get("dim", 0), keepdim=self.kwargs.get("keepdim", True))

        to_onehot = self.to_onehot if to_onehot is None else to_onehot
        if to_onehot is not None:
            if not isinstance(to_onehot, int):
                raise ValueError("the number of classes for One-Hot must be an integer.")
            img_t = one_hot(
                img_t, num_classes=to_onehot, dim=self.kwargs.get("dim", 0), dtype=self.kwargs.get("dtype", torch.float)
            )

        threshold = self.threshold if threshold is None else threshold
        if threshold is not None:
            img_t = img_t >= threshold

        rounding = self.rounding if rounding is None else rounding
        if rounding is not None:
            look_up_option(rounding, ["torchrounding"])
            img_t = torch.round(img_t)

        img, *_ = convert_to_dst_type(img_t, img, dtype=self.kwargs.get("dtype", torch.float))
        return img