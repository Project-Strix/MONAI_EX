import logging
from typing import Callable, Dict, Hashable, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from monai.config import KeysCollection, NdarrayTensor, NdarrayOrTensor
from monai.transforms.compose import MapTransform
from monai.utils import ensure_tuple_rep

from monai.transforms import SplitChannel
from monai_ex.transforms.utility.array import (
    CastToTypeEx,
    ToTensorEx,
    DataStatsEx
)

class CastToTypeExd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai_ex.transforms.CastToTypeEx`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        dtype: Union[Sequence[Union[np.dtype, torch.dtype, str]], np.dtype, torch.dtype, str] = np.float32,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            dtype: convert image to this data type, default is `np.float32`.
                it also can be a sequence of np.dtype or torch.dtype,
                each element corresponds to a key in ``keys``.

        """
        MapTransform.__init__(self, keys)
        self.dtype = ensure_tuple_rep(dtype, len(self.keys))
        self.converter = CastToTypeEx()

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for idx, key in enumerate(self.keys):
            d[key] = self.converter(d[key], dtype=self.dtype[idx])

        return d


class ToTensorExd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai_ex.transforms.ToTensorEx`.
    """

    def __init__(self, keys: KeysCollection) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        super().__init__(keys)
        self.converter = ToTensorEx()

    def __call__(self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.converter(d[key])
        return d


class DataStatsExd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai_ex.transforms.DataStatsEx`.
    """
    def __init__(
        self,
        keys: KeysCollection,
        meta_key_postfix: str = "meta_dict",
        prefix: Union[Sequence[str], str] = "Data",
        data_type: Union[Sequence[bool], bool] = True,
        data_shape: Union[Sequence[bool], bool] = True,
        value_range: Union[Sequence[bool], bool] = True,
        data_value: Union[Sequence[bool], bool] = False,
        additional_info: Optional[Union[Sequence[Callable], Callable]] = None,
        logger_handler: Optional[logging.Handler] = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.meta_key_postfix = meta_key_postfix
        self.prefix = ensure_tuple_rep(prefix, len(self.keys))
        self.data_type = ensure_tuple_rep(data_type, len(self.keys))
        self.data_shape = ensure_tuple_rep(data_shape, len(self.keys))
        self.value_range = ensure_tuple_rep(value_range, len(self.keys))
        self.data_value = ensure_tuple_rep(data_value, len(self.keys))
        self.additional_info = ensure_tuple_rep(additional_info, len(self.keys))
        self.logger_handler = logger_handler
        self.printer = DataStatsEx(logger_handler=logger_handler)

    def __call__(self, data: Mapping[Hashable, NdarrayTensor]) -> Dict[Hashable, NdarrayTensor]:
        d = dict(data)
        for key, prefix, data_type, data_shape, value_range, data_value, additional_info in self.key_iterator(
            d, self.prefix, self.data_type, self.data_shape, self.value_range, self.data_value, self.additional_info
        ):
            d[key] = self.printer(
                d[key],
                d[f'{key}_{self.meta_key_postfix}'],
                prefix,
                data_type,
                data_shape,
                value_range,
                data_value,
                additional_info,
            )
        return d


class SplitChannelExd(MapTransform):
    """
    Extension of `monai.transforms.SplitChanneld`.
    Extended: `output_names`: the names to construct keys to store split data if 
              you don't want postfixes.
              `remove_origin`: delete original data of given keys

    """

    backend = SplitChannel.backend

    def __init__(
        self,
        keys: KeysCollection,
        output_postfixes: Optional[Sequence[str]] = None,
        output_names: Optional[Sequence[str]] = None,
        channel_dim: int = 0,
        remove_origin: bool = False,
        allow_missing_keys: bool = False,
        meta_key_postfix='meta_dict',
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            output_postfixes: the postfixes to construct keys to store split data.
                for example: if the key of input data is `pred` and split 2 classes, the output
                data keys will be: pred_(output_postfixes[0]), pred_(output_postfixes[1])
                if None, using the index number: `pred_0`, `pred_1`, ... `pred_N`.
            channel_dim: which dimension of input image is the channel, default to 0.
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.output_postfixes = output_postfixes
        self.output_names = output_names
        self.remove_origin = remove_origin
        self.meta_key_postfix = meta_key_postfix
        self.splitter = SplitChannel(channel_dim=channel_dim)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            rets = self.splitter(d[key])
            postfixes: Sequence = list(range(len(rets))) if self.output_postfixes is None else self.output_postfixes
            if len(postfixes) != len(rets):
                raise AssertionError("count of split results must match output_postfixes.")
            for i, r in enumerate(rets):
                split_key = f"{key}_{postfixes[i]}" if self.output_names is None else self.output_names[i]
                if split_key in d:
                    raise RuntimeError(f"input data already contains key {split_key}.")
                d[split_key] = r
                if self.remove_origin:
                    d[f"{split_key}_{self.meta_key_postfix}"] = d[f"{key}_{self.meta_key_postfix}"]
            if self.remove_origin:
                d.pop(key)
                d.pop(f"{key}_{self.meta_key_postfix}")
        return d


ToTensorExD = ToTensorExDict = ToTensorExd
CastToTypeExD = CastToTypeExDict = CastToTypeExd
DataStatsExD = DataStatsExDict = DataStatsExd
SplitChannelExD = SplitChannelExDict = SplitChannelExd
