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
A collection of dictionary-based wrappers around the "vanilla" transforms for spatial operations
defined in :py:class:`monai.transforms.spatial.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

from typing import Dict, Hashable, Mapping, Optional, Sequence, Union, Tuple, Any

import torch
import numpy as np

from monai.config import KeysCollection, NdarrayOrTensor
from monai.transforms.compose import MapTransform, Randomizable
from monai_ex.transforms.spatial.array import (
    FixedResize,
    LabelMorphology,
    RandLabelMorphology,
    Rotate90Ex,
    KSpaceResample,
    RandDrop
)

from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    NumpyPadMode,
    ensure_tuple_rep,
)

GridSampleModeSequence = Union[Sequence[Union[GridSampleMode, str]], GridSampleMode, str]
GridSamplePadModeSequence = Union[Sequence[Union[GridSamplePadMode, str]], GridSamplePadMode, str]
InterpolateModeSequence = Union[Sequence[Union[InterpolateMode, str]], InterpolateMode, str]
NumpyPadModeSequence = Union[Sequence[Union[NumpyPadMode, str]], NumpyPadMode, str]


class FixedResized(MapTransform):
    """
    Dict-based version :py:class:`monai.transforms.FixedResize`.

    Args:
        keys: Keys to pick data for transformation.
    """
    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Union[Sequence[int], int],
        mode: InterpolateModeSequence = InterpolateMode.AREA,
        align_corners: Union[Sequence[Optional[bool]], Optional[bool]] = None,
    ) -> None:
        super().__init__(keys)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.resizer = FixedResize(spatial_size=spatial_size)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for idx, key in enumerate(self.keys):
            d[key] = self.resizer(d[key], mode=self.mode[idx], align_corners=self.align_corners[idx])
        return d


class LabelMorphologyd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`DataMorphology`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        mode: str,
        radius: int,
        binary: bool,
    ) -> None:
        super().__init__(keys)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.radius = ensure_tuple_rep(radius, len(self.keys))
        self.binary = ensure_tuple_rep(binary, len(self.keys))
        self.converter = LabelMorphology('dilation', 0, True)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for idx, key in enumerate(self.keys):
            if self.radius[idx] <= 0:
                continue
            d[key] = self.converter(d[key], mode=self.mode[idx], radius=self.radius[idx], binary=self.binary[idx])
        return d


class RandLabelMorphologyd(Randomizable, MapTransform):
    """Dictionary-based version :py:class:`monai_ex.transforms.RandLabelMorphology`.
    """
    def __init__(
        self,
        keys: KeysCollection,
        prob: float,
        mode: str,
        radius: int,
        binary: bool
    ):
        super().__init__(keys)
        self.prob = prob
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.radius = ensure_tuple_rep(radius, len(self.keys))
        self.binary = ensure_tuple_rep(binary, len(self.keys))
        self.converter = LabelMorphology('dilation', 0, True)

    def randomize(self):
        self._do_transform = self.R.random() < self.prob

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Mapping[Hashable, np.ndarray]:
        self.randomize()
        if not self._do_transform:
            return data

        d = dict(data)
        for idx, key in enumerate(self.keys):
            if self.radius[idx] <= 0:
                continue
            d[key] = self.converter(d[key], mode=self.mode[idx], radius=self.radius[idx], binary=self.binary[idx])
        return d


class RandRotate90Exd(Randomizable, MapTransform):
    """
    Dictionary-based version :py:class:`monai_ex.transforms.RandRotate90ex`.
    With probability `prob`, input arrays are rotated by 90 degrees
    in the plane specified by `spatial_axes`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        max_k: int = 3,
        spatial_axes: Tuple[int, int] = (0, 1),
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            prob: probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array.)
            max_k: number of rotations will be sampled from `np.random.randint(max_k) + 1`.
                (Default 3)
            spatial_axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
        """
        super().__init__(keys)

        self.prob = min(max(prob, 0.0), 1.0)
        self.max_k = max_k
        self.spatial_axes = spatial_axes

        self._do_transform = False
        self._rand_k = 0

    def randomize(self, data: Optional[Any] = None) -> None:
        self._rand_k = self.R.randint(self.max_k) + 1
        self._do_transform = self.R.random() < self.prob

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Mapping[Hashable, np.ndarray]:
        self.randomize()
        if not self._do_transform:
            return data

        rotator = Rotate90Ex(self._rand_k, self.spatial_axes)
        d = dict(data)
        for key in self.keys:
            d[key] = rotator(d[key])
        return d


class KSpaceResampled(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai_ex.transforms.KSpaceResample`.

    This transform assumes the ``data`` dictionary has a key for the input
    data's metadata and contains `affine` field.  The key is formed by ``key_{meta_key_postfix}``.

    After resampling the input array, this transform will write the new affine
    to the `affine` field of metadata which is formed by ``key_{meta_key_postfix}``.

    see also:
        :py:class:`monai_ex.transforms.KSpaceResample`
    """

    def __init__(
        self,
        keys: KeysCollection,
        pixdim: Union[Sequence[float], float],
        diagonal: bool = False,
        device: Optional[torch.device] = None,
        tolerance: float = 0.0001,
        meta_key_postfix: str = "meta_dict",
    ) -> None:
        super().__init__(keys)
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.resizer = KSpaceResample(
            pixdim=pixdim,
            diagonal=diagonal,
            device=device,
            tolerance=tolerance,
        )

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key, metakey_postfix in zip(self.keys, self.meta_key_postfix):
            meta_key = f"{key}_{metakey_postfix}"
            # create metadata if necessary
            if meta_key not in d:
                d[meta_key] = {"affine": None}
            meta_data = d[meta_key]

            d[key], old_affine, new_affine = self.resizer(
                d[key],
                affine=meta_data["affine"],
            )
            meta_data["affine"] = new_affine
        return d


class RandDropd(MapTransform):
    """Dictionary-based version :py:class:`monai_ex.transforms.RandomDrop`."""

    def __init__(
        self,
        keys: KeysCollection,
        cline_key: str,
        roi_size: int,
        roi_number: int,
        random_seed: int = 20221201,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.cline_key = cline_key
        self.transformer = RandDrop(
            roi_number,
            roi_size,
            random_seed
        )

    def __call__(
        self,
        data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.transformer(d[key], d[self.cline_key])
        return d


FixedResizeD = FixedResizeDict = FixedResized
LabelMorphologyD = LabelMorphologyDict = LabelMorphologyd
RandRotate90ExD = RandRotate90ExDict = RandRotate90Exd
RandLabelMorphologyD = RandLabelMorphologyDict = RandLabelMorphologyd
KSpaceResampleD = KSpaceResampleDict = KSpaceResampled
RandDropD = RandDropDict = RandDropd
