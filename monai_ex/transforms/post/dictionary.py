import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Union

import torch

from monai.config.type_definitions import KeysCollection, NdarrayOrTensor, PathLike
from monai.data.csv_saver import CSVSaver
from monai.transforms.inverse import InvertibleTransform
from monai_ex.transforms.post.array import AsDiscreteEx, MultitaskMeanEnsemble
from monai.transforms.post.dictionary import Ensembled
from monai.transforms.transform import MapTransform
from monai.transforms.utility.array import ToTensor
from monai.transforms.utils import allow_missing_keys_mode, convert_inverse_interp_mode
from monai.utils import deprecated_arg, ensure_tuple, ensure_tuple_rep

class AsDiscreteExd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AsDiscrete`.
    """

    backend = AsDiscreteEx.backend

    @deprecated_arg(name="n_classes", new_name="num_classes", since="0.6", msg_suffix="please use `to_onehot` instead.")
    @deprecated_arg("num_classes", since="0.7", msg_suffix="please use `to_onehot` instead.")
    @deprecated_arg("logit_thresh", since="0.7", msg_suffix="please use `threshold` instead.")
    @deprecated_arg(
        name="threshold_values", new_name="threshold", since="0.7", msg_suffix="please use `threshold` instead."
    )
    def __init__(
        self,
        keys: KeysCollection,
        argmax: Union[Sequence[bool], bool] = False,
        to_onehot: Union[Sequence[Optional[int]], Optional[int]] = None,
        threshold: Union[Sequence[Optional[float]], Optional[float]] = None,
        rounding: Union[Sequence[Optional[str]], Optional[str]] = None,
        allow_missing_keys: bool = False,
        n_classes: Optional[Union[Sequence[int], int]] = None,  # deprecated
        num_classes: Optional[Union[Sequence[int], int]] = None,  # deprecated
        logit_thresh: Union[Sequence[float], float] = 0.5,  # deprecated
        threshold_values: Union[Sequence[bool], bool] = False,  # deprecated
        **kwargs,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to model output and label.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            argmax: whether to execute argmax function on input data before transform.
                it also can be a sequence of bool, each element corresponds to a key in ``keys``.
            to_onehot: if not None, convert input data into the one-hot format with specified number of classes.
                defaults to ``None``. it also can be a sequence, each element corresponds to a key in ``keys``.
            threshold: if not None, threshold the float values to int number 0 or 1 with specified theashold value.
                defaults to ``None``. it also can be a sequence, each element corresponds to a key in ``keys``.
            rounding: if not None, round the data according to the specified option,
                available options: ["torchrounding"]. it also can be a sequence of str or None,
                each element corresponds to a key in ``keys``.
            allow_missing_keys: don't raise exception if key is missing.
            kwargs: additional parameters to ``AsDiscrete``.
                ``dim``, ``keepdim``, ``dtype`` are supported, unrecognized parameters will be ignored.
                These default to ``0``, ``True``, ``torch.float`` respectively.
        .. deprecated:: 0.6.0
            ``n_classes`` is deprecated, use ``to_onehot`` instead.
        .. deprecated:: 0.7.0
            ``num_classes`` is deprecated, use ``to_onehot`` instead.
            ``logit_thresh`` is deprecated, use ``threshold`` instead.
            ``threshold_values`` is deprecated, use ``threshold`` instead.
        """
        super().__init__(keys, allow_missing_keys)
        self.argmax = ensure_tuple_rep(argmax, len(self.keys))
        to_onehot_ = ensure_tuple_rep(to_onehot, len(self.keys))
        num_classes = ensure_tuple_rep(num_classes, len(self.keys))
        self.to_onehot = []
        for flag, val in zip(to_onehot_, num_classes):
            if isinstance(flag, bool):
                warnings.warn("`to_onehot=True/False` is deprecated, please use `to_onehot=num_classes` instead.")
                self.to_onehot.append(val if flag else None)
            else:
                self.to_onehot.append(flag)

        threshold_ = ensure_tuple_rep(threshold, len(self.keys))
        logit_thresh = ensure_tuple_rep(logit_thresh, len(self.keys))
        self.threshold = []
        for flag, val in zip(threshold_, logit_thresh):
            if isinstance(flag, bool):
                warnings.warn("`threshold_values=True/False` is deprecated, please use `threshold=value` instead.")
                self.threshold.append(val if flag else None)
            else:
                self.threshold.append(flag)

        self.rounding = ensure_tuple_rep(rounding, len(self.keys))
        self.converter = AsDiscreteEx()
        self.converter.kwargs = kwargs


    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, argmax, to_onehot, threshold, rounding in self.key_iterator(
            d, self.argmax, self.to_onehot, self.threshold, self.rounding
        ):
            d[key] = self.converter(d[key], argmax, to_onehot, threshold, rounding)
        return d


class MultitaskMeanEnsembled(Ensembled):
    """
    Dictionary-based wrapper of :py:class:`monai_ex.transforms.MultitaskMeanEnsemble`.
    """
    backend = MultitaskMeanEnsemble.backend

    def __init__(
        self,
        keys: KeysCollection,
        task_num: int,
        output_key: Optional[str] = None,
        weights: Optional[Union[Sequence[float], NdarrayOrTensor]] = None
    ) -> None:
        ensemble = MultitaskMeanEnsemble(task_num, weights=weights)
        super().__init__(keys, ensemble, output_key)
            


AsDiscreteExD = AsDiscreteExDict = AsDiscreteExd
MultitaskMeanEnsembleD = MultitaskMeanEnsembleDict = MultitaskMeanEnsembled