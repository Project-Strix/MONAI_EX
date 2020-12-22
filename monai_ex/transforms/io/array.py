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
A collection of "vanilla" transforms for IO functions
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from monai.utils import ensure_tuple, optional_import
from monai.transforms.io.array import LoadPNG

Image, _ = optional_import("PIL.Image")


class LoadPNGEx(LoadPNG):
    """
    Extension of MONAI's LoadPNG
    Extended: grayscale

    Load common 2D image format (PNG, JPG, etc. using PIL) file or files from provided path.
    If loading a list of files, stack them together and add a new dimension as first dimension,
    and use the meta data of the first image to represent the stacked result.
    It's based on the Image module in PIL library:
    https://pillow.readthedocs.io/en/stable/reference/Image.html
    """

    def __init__(self, image_only: bool = False, dtype: Optional[np.dtype] = np.float32, grayscale: bool = False) -> None:
        """
        Args:
            image_only: if True return only the image volume, otherwise return image data array and metadata.
            dtype: if not None convert the loaded image to this data type.
            grayscale: convert image to grayscale.
        """
        super(LoadPNGEx, self).__init__(
            image_only=image_only,
            dtype=dtype
        )
        self.grayscale = grayscale

    def __call__(self, filename: Union[Sequence[Union[Path, str]], Path, str]):
        """
        Args:
            filename: path file or file-like object or a list of files.
        """
        filename = ensure_tuple(filename)
        img_array = list()
        compatible_meta = None
        for name in filename:
            img = Image.open(name).convert('L') if self.grayscale else Image.open(name)
            data = np.asarray(img)
            if self.dtype:
                data = data.astype(self.dtype)
            img_array.append(data)

            if self.image_only:
                continue

            meta = dict()
            meta["filename_or_obj"] = name
            meta["spatial_shape"] = data.shape[:2]
            meta["format"] = img.format if img.format is not None else name.split('.')[-1]
            meta["mode"] = img.mode
            meta["width"] = img.width
            meta["height"] = img.height
            #meta["info"] = img.info
            if not compatible_meta:
                compatible_meta = meta
            else:
                assert np.allclose(
                    meta["spatial_shape"], compatible_meta["spatial_shape"]
                ), "all the images in the list should have same spatial shape."

        img_array = np.stack(img_array, axis=0) if len(img_array) > 1 else img_array[0]
        return img_array if self.image_only else (img_array, compatible_meta)

