from typing import Optional, Sequence, Union

import numpy as np
import torch

from monai.config.type_definitions import NdarrayOrTensor
from monai.networks import one_hot
from monai.transforms.transform import Transform
from monai.utils import TransformBackends, convert_data_type, ensure_tuple, look_up_option
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


class EnsembleEx:
    @staticmethod
    def get_stacked_torch(
        item_index: int,
        img: Union[Sequence[NdarrayOrTensor], NdarrayOrTensor]
    ) -> torch.Tensor:
        """Get either a sequence or single instance of np.ndarray/torch.Tensor. Return single torch.Tensor."""
        if isinstance(img, Sequence) and isinstance(img[0], Sequence):
            img = [torch.as_tensor(i[item_index]) for i in img]
        elif isinstance(img, Sequence) and isinstance(img[0], np.ndarray):
            img = [torch.as_tensor(i) for i in img]
        elif isinstance(img, np.ndarray):  #! not tested
            img = torch.as_tensor(img)
        out: torch.Tensor = torch.stack(img) if isinstance(img, Sequence) else img  # type: ignore
        return out

    @staticmethod
    def post_convert(
        img: torch.Tensor,
        orig_img: Union[Sequence[NdarrayOrTensor], NdarrayOrTensor],
        item_index: int,
    ) -> NdarrayOrTensor: 
        if isinstance(orig_img, Sequence):
            if isinstance(orig_img[0], Sequence):
                orig_img_ = orig_img[0][item_index]
            elif isinstance(orig_img[0], NdarrayOrTensor):
                orig_img_ = orig_img[0]
            else:
                orig_img_ = orig_img

        out, *_ = convert_to_dst_type(img, orig_img_)
        return out



class MultitaskMeanEnsemble(EnsembleEx, Transform):
    """
    Execute mean ensemble on the data generated by multitask engine.
    Different from normal single task engine, multitask generate multiple predictions.
    The input data can be a list or tuple of PyTorch Tensor with shape: [C[, H, W, D]],
    Or a single PyTorch Tensor with shape: [E, C[, H, W, D]], the `E` dimension represents
    the output data from different models.
    Typically, the input data is model output of segmentation task or classification task.
    And it also can support to add `weights` for the input data.

    Args:
        weights: can be a list or tuple of numbers for input data with shape: [E, C, H, W[, D]].
            or a Numpy ndarray or a PyTorch Tensor data.
            the `weights` will be added to input data from highest dimension, for example:
            1. if the `weights` only has 1 dimension, it will be added to the `E` dimension of input data.
            2. if the `weights` has 2 dimensions, it will be added to `E` and `C` dimensions.
            it's a typical practice to add weights for different classes:
            to ensemble 3 segmentation model outputs, every output has 4 channels(classes),
            so the input data shape can be: [3, 4, H, W, D].
            and add different `weights` for different classes, so the `weights` shape can be: [3, 4].
            for example: `weights = [[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 1, 1]]`.

    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        task_num: int,
        weights: Optional[Union[Sequence[float], NdarrayOrTensor]] = None
        ) -> None:
        self.task_num = task_num
        self.weights = torch.as_tensor(weights, dtype=torch.float) if weights is not None else None

    def __call__(self, img: Union[Sequence[NdarrayOrTensor], NdarrayOrTensor]) -> NdarrayOrTensor:
        final_tensors = []
        for task_id in range(self.task_num):
            img_ = self.get_stacked_torch(task_id, img)
            if self.weights is not None:
                self.weights = self.weights.to(img_.device)
                shape = tuple(self.weights.shape)
                for _ in range(img_.ndimension() - self.weights.ndimension()):
                    shape += (1,)
                weights = self.weights.reshape(*shape)

                img_ = img_ * weights / weights.mean(dim=0, keepdim=True)

            out_pt = torch.mean(img_, dim=0)
            final_tensors.append(self.post_convert(out_pt, img, task_id))
        return tuple(final_tensors)

