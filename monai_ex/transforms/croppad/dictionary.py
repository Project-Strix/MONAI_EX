from typing import Dict, Hashable, Mapping, Optional, Sequence, Union

import numpy as np

from monai.config import KeysCollection
from monai.transforms.compose import MapTransform

from monai_ex.transforms.croppad.array import CenterMask2DSliceCrop


class CenterMask2DSliceCropd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        mask_key: KeysCollection,
        roi_size: Union[Sequence[int], int],
        crop_mode: str,
        center_mode: str,
        z_axis: int,
    ) -> None:
        super().__init__(keys)
        self.mask_key = mask_key
        self.cropper = CenterMask2DSliceCrop(
            roi_size=roi_size,
            crop_mode=crop_mode,
            center_mode=center_mode,
            z_axis=z_axis,
            mask_data=None
        )
    
    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.cropper(d[key], d[self.mask_key])
        return d


CenterMask2DSliceCropD = CenterMask2DSliceCropDict = CenterMask2DSliceCropd