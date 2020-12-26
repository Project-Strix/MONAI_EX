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

from typing import Dict, Hashable, Mapping, Optional, Sequence, Union

import numpy as np

from monai.config import KeysCollection
from monai.transforms.compose import MapTransform
from monai_ex.transforms.spatial.array import FixedResize, LabelMorphology

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

FixedResizeD = FixedResizeDict = FixedResized
LabelMorphologyD = LabelMorphologyDict = LabelMorphologyd
