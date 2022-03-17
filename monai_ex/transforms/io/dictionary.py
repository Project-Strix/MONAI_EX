# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A collection of dictionary-based wrappers around the "vanilla" transforms for IO functions
defined in :py:class:`monai.transforms.io.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""
from typing import Optional, Sequence, Union, Hashable, Mapping, Dict, Any

import numpy as np

from monai.config import KeysCollection
from monai.utils import ensure_tuple, ImageMetaKey as Key
from monai.data.image_reader import ImageReader
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.transform import MapTransform
from monai.config.type_definitions import NdarrayOrTensor
from monai_ex.transforms import GenerateSyntheticData, GenerateRandomData


class LoadImageExd(LoadImaged):
    """
    Extension of MONAI's LoadImaged
    Extended: drop_meta_keys

    Dictionary-based wrapper of :py:class:`monai.transforms.LoadImage`,
    must load image and metadata together. If loading a list of files in one key,
    stack them together and add a new dimension as the first dimension, and use the
    meta data of the first image to represent the stacked result. Note that the affine
    transform of all the stacked images should be same. The output metadata field will
    be created as ``key_{meta_key_postfix}``.
    """

    def __init__(
        self,
        keys: KeysCollection,
        reader: Optional[ImageReader] = None,
        dtype: Optional[np.dtype] = np.float32,
        meta_key_postfix: str = "meta_dict",
        drop_meta_keys: Optional[Union[Sequence[str], str]] = None,
        overwriting: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            reader: register reader to load image file and meta data, if None, still can register readers
                at runtime or use the default ITK reader.
            dtype: if not None convert the loaded image data to this data type.
            meta_key_postfix: use `key_{postfix}` to store the metadata of the nifti image,
                default is `meta_dict`. The meta data is a dictionary object.
                For example, load nifti file for `image`, store the metadata into `image_meta_dict`.
            drop_meta_keys: specified keys to drop. This will help to fix the collate error.
            overwriting: whether allow to overwrite existing meta data of same key.
                default is False, which will raise exception if encountering existing key.
        """
        super(LoadImageExd, self).__init__(
            keys=keys,
            reader=reader,
            dtype=dtype,
            meta_key_postfix=meta_key_postfix,
            overwriting=overwriting,
        )
        self.drop_meta_keys = drop_meta_keys

    def __call__(self, data, reader: Optional[ImageReader] = None):
        """
        Raises:
            KeyError: When not ``self.overwriting`` and key already exists in ``data``.

        """
        d = dict(data)
        for key in self.keys:
            data = self._loader(d[key], reader)
            assert isinstance(
                data, (tuple, list)
            ), "loader must return a tuple or list."
            d[key] = data[0]
            assert isinstance(data[1], dict), "metadata must be a dict."
            key_to_add = f"{key}_{self.meta_key_postfix}"
            if key_to_add in d and not self.overwriting:
                raise KeyError(
                    f"Meta data with key {key_to_add} already exists and overwriting=False."
                )
            if self.drop_meta_keys is not None:
                for k in ensure_tuple(self.drop_meta_keys):
                    data[1].pop(k, None)
            d[key_to_add] = data[1]
        return d


class GenerateSyntheticDatad(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        height: int,
        width: int,
        depth: Optional[int] = None,
        num_objs: int = 12,
        rad_max: int = 30,
        rad_min: int = 5,
        noise_max: float = 0.0,
        num_seg_classes: int = 5,
        channel_dim: Optional[int] = None,
        random_state: Optional[np.random.RandomState] = None,
        image_only: bool = False,
        meta_key_postfix="meta_dict",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        if image_only:
            assert len(self.keys) == 1, f"Need only one key, but got {self.keys}"
        else:
            assert len(self.keys) == 2, f"Need two keys, but got {self.keys}"

        self.image_only = image_only
        self.meta_postfix = meta_key_postfix
        self.loader = GenerateSyntheticData(
            height,
            width,
            depth,
            num_objs,
            rad_max,
            rad_min,
            noise_max,
            num_seg_classes,
            channel_dim,
            random_state,
        )

    def __call__(self, data: Any) -> Dict[Hashable, NdarrayOrTensor]:
        test_data = self.loader(None)
        if self.image_only:
            test_data = [test_data[0]]

        data = {}
        for key, d in zip(self.keys, test_data):
            data[key] = d
            data[f"{key}_{self.meta_postfix}"] = {
                "affine": np.eye(4),
                Key.FILENAME_OR_OBJ: "./dummy_file",
                "original_channel_dim": "no_channel",
            }

        return data


class GenerateRandomDataD(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        height: int,
        width: int,
        depth: Optional[int] = None,
        num_classes: int = 5,
        channel_dim: Optional[int] = None,
        random_state: Optional[np.random.RandomState] = None,
        meta_key_postfix="meta_dict",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        assert len(self.keys) == 2, f"Need two keys, but got {self.keys}"
        self.meta_postfix = meta_key_postfix
        self.generator = GenerateRandomData(
            height, width, depth, num_classes, channel_dim, random_state
        )

    def __call__(self, data):
        random_data = self.generator(data)

        data = {}
        for key, d in zip(self.keys, random_data):
            data[key] = d
            data[f"{key}_{self.meta_postfix}"] = {
                "affine": np.eye(4),
                Key.FILENAME_OR_OBJ: "./dummy_file",
                "original_channel_dim": "no_channel",
            }
        return data


LoadImageExD = LoadImageExDict = LoadImageExd
GenerateSyntheticDataD = GenerateSyntheticDataDict = GenerateSyntheticDatad
