import logging
from typing import Callable, Dict, Hashable, Mapping, Optional, Sequence, Union, List

import numpy as np
import torch

from monai.config import KeysCollection, NdarrayTensor, NdarrayOrTensor
from monai.transforms.compose import MapTransform, Randomizable
from monai.utils import ensure_tuple_rep
from monai_ex.utils import ensure_list

from monai.transforms import SplitChannel
from monai_ex.transforms.utility.array import (
    CastToTypeEx,
    ToTensorEx,
    DataStatsEx,
    DataLabelling,
    Clahe,
)

from monai_ex.transforms import (
    generate_pos_neg_label_crop_centers,
    map_binary_to_indices,
    SpatialCrop,
)


class CastToTypeExd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai_ex.transforms.CastToTypeEx`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        dtype: Union[
            Sequence[Union[np.dtype, torch.dtype, str]], np.dtype, torch.dtype, str
        ] = np.float32,
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

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, torch.Tensor]:
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

    def __call__(
        self, data: Mapping[Hashable, NdarrayTensor]
    ) -> Dict[Hashable, NdarrayTensor]:
        d = dict(data)
        for (
            key,
            prefix,
            data_type,
            data_shape,
            value_range,
            data_value,
            additional_info,
        ) in self.key_iterator(
            d,
            self.prefix,
            self.data_type,
            self.data_shape,
            self.value_range,
            self.data_value,
            self.additional_info,
        ):
            d[key] = self.printer(
                d[key],
                d[f"{key}_{self.meta_key_postfix}"],
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

class DataLabellingd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
    ) -> None:
        super().__init__(keys)
        self.converter = DataLabelling()

    def __call__(
        self, img: Mapping[Hashable, torch.Tensor]
    ) -> Dict[Hashable, torch.Tensor]:
        d = dict(img)
        for idx, key in enumerate(self.keys):
            d[key] = self.converter(d[key])
        return d


class Clahed(MapTransform):
    def __init__(
        self, keys: KeysCollection, kernel_size=None, clip_limit=0.01, nbins=256
    ) -> None:
        super().__init__(keys)
        self.converter = Clahe()
        self.kernel_size = kernel_size
        self.clip_limit = clip_limit
        self.nbins = nbins

    def __call__(
        self, img: Mapping[Hashable, torch.Tensor]
    ) -> Dict[Hashable, torch.Tensor]:
        d = dict(img)
        for idx, key in enumerate(self.keys):
            d[key] = self.converter(d[key])
        return d


class ConcatModalityd(MapTransform):
    """Concat multi-modality data by given keys."""

    def __init__(self, keys, output_key, axis):
        super().__init__(keys)
        self.output_key = output_key
        self.axis = axis

    def __call__(self, data):
        d = dict(data)
        concat_data = np.concatenate([d[key] for key in self.keys], axis=self.axis)
        d[self.output_key] = concat_data

        return d


class RandCrop2dByPosNegLabeld(Randomizable, MapTransform):
    def __init__(
        self,
        n_layer: int,
        keys: KeysCollection,
        label_key: str,
        spatial_size: Union[Sequence[int], int],
        crop_mode: str,
        z_axis: int,
        pos: float = 1.0,
        neg: float = 0.0,
        num_samples: int = 1,
        image_key: Optional[str] = None,
        image_threshold: float = 0.0,
        fg_indices_key: Optional[str] = None,
        bg_indices_key: Optional[str] = None,
    ) -> None:
        super().__init__(keys)
        self.spatial_size = ensure_tuple_rep(spatial_size, 2)
        self.label_key = label_key
        self.num_samples = num_samples
        self.image_key = image_key
        self.image_threshold = image_threshold
        self.centers: Optional[List[List[np.ndarray]]] = None
        self.fg_indices_key = fg_indices_key
        self.bg_indices_key = bg_indices_key
        self.n_layer = n_layer

        if pos < 0 or neg < 0:
            raise ValueError(
                f"pos and neg must be nonnegative, got pos={pos} neg={neg}."
            )
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.pos_ratio = pos / (pos + neg)
        if crop_mode not in ["single", "cross", "parallel"]:
            raise ValueError("Cropping mode must be one of 'single, cross, parallel'")
        self.crop_mode = crop_mode
        self.z_axis = z_axis

    def get_new_spatial_size(self):
        spatial_size_ = ensure_list(self.spatial_size)
        if self.crop_mode == "single":
            spatial_size_.insert(self.z_axis, 1)
        elif self.crop_mode == "parallel":
            spatial_size_.insert(self.z_axis, self.n_layer)
        else:
            spatial_size_ = [
                max(spatial_size_),
            ] * 3

        return spatial_size_

    def randomize(
        self,
        label: np.ndarray,
        fg_indices: Optional[np.ndarray] = None,
        bg_indices: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
    ) -> None:
        if fg_indices is None or bg_indices is None:
            fg_indices_, bg_indices_ = map_binary_to_indices(
                label, image, self.image_threshold
            )
        else:
            fg_indices_ = fg_indices
            bg_indices_ = bg_indices

        self.centers = generate_pos_neg_label_crop_centers(
            self.get_new_spatial_size(),
            self.num_samples,
            self.pos_ratio,
            label.shape[1:],
            fg_indices_,
            bg_indices_,
            self.R,
        )

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> List[Dict[Hashable, np.ndarray]]:
        d = dict(data)
        label = d[self.label_key]
        image = d[self.image_key] if self.image_key else None
        fg_indices = (
            d.get(self.fg_indices_key, None)
            if self.fg_indices_key is not None
            else None
        )
        bg_indices = (
            d.get(self.bg_indices_key, None)
            if self.bg_indices_key is not None
            else None
        )

        self.randomize(label, fg_indices, bg_indices, image)
        assert isinstance(self.spatial_size, tuple)
        assert self.centers is not None
        results: List[Dict[Hashable, np.ndarray]] = [
            dict() for _ in range(self.num_samples)
        ]
        for key in data.keys():
            if key in self.keys:
                img = d[key]
                for i, center in enumerate(self.centers):
                    if self.crop_mode in ["single", "parallel"]:
                        size_ = self.get_new_spatial_size()
                        slice_ = SpatialCrop(roi_center=tuple(center), roi_size=size_)(
                            img
                        )

                        seg_sum = slice_.squeeze().sum()
                        results[i][key] = np.moveaxis(slice_.squeeze(0), self.z_axis, 0)
                    else:
                        cross_slices = np.zeros(shape=(3,) + self.spatial_size)
                        for k in range(3):
                            size_ = np.insert(self.spatial_size, k, 1)
                            slice_ = SpatialCrop(
                                roi_center=tuple(center), roi_size=size_
                            )(img)
                            cross_slices[k] = slice_.squeeze()
                        results[i][key] = cross_slices
            else:
                for i in range(self.num_samples):
                    results[i][key] = data[key]

        return results


ToTensorExD = ToTensorExDict = ToTensorExd
CastToTypeExD = CastToTypeExDict = CastToTypeExd
DataStatsExD = DataStatsExDict = DataStatsExd
SplitChannelExD = SplitChannelExDict = SplitChannelExd
DataLabellinD = DataLabellingDict = DataLabellingd
ClaheD = ClaheDict = Clahed
ConcatModalityD = ConcatModalityDict = ConcatModalityd
RandCrop2dByPosNegLabelD = RandCrop2dByPosNegLabelDict = RandCrop2dByPosNegLabeld
