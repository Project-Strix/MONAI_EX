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
from typing import Optional, Sequence, Union

import numpy as np
import h5py

from monai.config import KeysCollection
from monai.utils import ensure_tuple
from monai.data.image_reader import ImageReader
from monai.transforms.io.dictionary import LoadImaged


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
        drop_meta_keys: Optional[Union[Sequence[str],str]] = None,
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
            overwriting=overwriting
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
            assert isinstance(data, (tuple, list)), "loader must return a tuple or list."
            d[key] = data[0]
            assert isinstance(data[1], dict), "metadata must be a dict."
            key_to_add = f"{key}_{self.meta_key_postfix}"
            if key_to_add in d and not self.overwriting:
                raise KeyError(f"Meta data with key {key_to_add} already exists and overwriting=False.")
            if self.drop_meta_keys is not None:
                for k in ensure_tuple(self.drop_meta_keys):
                    data[1].pop(k, None)
            d[key_to_add] = data[1]
        return d


# class LoadHdf5d(LoadDatad):
#     def __init__(
#         self,
#         keys: KeysCollection,
#         h5_keys: Optional[KeysCollection] = None,
#         affine_keys: Optional[KeysCollection] = None,
#         dtype: Optional[Sequence[np.dtype]] = None,
#         has_keys: Optional[bool] = True,
#     ) -> None:
#         self.keys = keys
#         self.h5_keys = self.keys if h5_keys is None else h5_keys
#         self.has_keys = has_keys
#         self.dtype = dtype
#         self.meta_key_postfix = 'meta_dict'
#         assert len(self.keys) == len(self.h5_keys), f'Dict keys {self.keys} len must match hdf5 keys {self.h5_keys}'
#         if self.dtype is not None:
#             assert len(self.keys) == len(self.dtype), f'Dict keys {self.keys} len must match dtypes {self.dtype}'
#         if affine_keys is not None:
#             if isinstance(affine_keys, str):
#                 self.affine_keys = [affine_keys]*len(self.h5_keys)
#             elif isinstance(affine_keys, list):
#                 self.affine_keys = affine_keys
#                 assert len(self.affine_keys) == len(self.h5_keys), f'Affine keys {self.affine_keys} len must match h5 keys {self.h5_keys}'
#             else:
#                 raise ValueError


#     def __call__(self, data):
#         hf = h5py.File(data, 'r')
#         if self.has_keys:
#             assert np.all([ k in list(hf.keys()) for k in self.h5_keys]), f'Keys are not found in {hf.keys()}'
#         if self.dtype is not None:
#             dataset = { self.keys[i]:np.copy(hf.get(key)).astype(self.dtype[i]) if
#                         key in hf.keys() else None for i, key in enumerate(self.h5_keys) }
#         else:
#             dataset = { self.keys[i]:np.copy(hf.get(key)) if
#                         key in hf.keys() else None for i, key in enumerate(self.h5_keys) }
        
#         key_to_add = [f"{key}_{self.meta_key_postfix}" for key in self.keys]
#         for k, affine in zip(key_to_add, self.affine_keys):
#             if isinstance(affine, str):
#                 meta_data = {"filename_or_obj":data, 'affine': np.copy(hf.get(affine)).astype(np.float32)}
#             else:
#                 meta_data = {"filename_or_obj":data, 'affine': np.eye(4)}
#             dataset[k] = meta_data
#         hf.close()
#         return dataset


LoadImageExD = LoadImageExDict = LoadImageExd
