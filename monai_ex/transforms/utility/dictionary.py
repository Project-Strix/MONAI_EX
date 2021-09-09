import logging
from typing import Callable, Dict, Hashable, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from monai.config import KeysCollection, NdarrayTensor
from monai.transforms.compose import MapTransform
from monai.utils import ensure_tuple_rep

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


ToTensorExD = ToTensorExDict = ToTensorExd
CastToTypeExD = CastToTypeExDict = CastToTypeExd
DataStatsExD = DataStatsExDict = DataStatsExd
