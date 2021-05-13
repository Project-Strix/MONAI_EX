from typing import Dict, Hashable, Mapping, Optional, Sequence, Union, List

import numpy as np

from monai.config import KeysCollection
from monai.transforms.compose import MapTransform
from monai.transforms.utils import (
    map_binary_to_indices,
    generate_pos_neg_label_crop_centers
)
from monai.transforms import RandCropByPosNegLabeld
from monai_ex.utils import fall_back_tuple
from monai_ex.transforms.croppad.array import CenterMask2DSliceCrop, FullMask2DSliceCrop, GetMaxSlices3direcCrop, FullImage2DSliceCrop


class CenterMask2DSliceCropd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        mask_key: KeysCollection,
        roi_size: Union[Sequence[int], int],
        crop_mode: str,
        center_mode: str,
        z_axis: int,
        n_slices: int = 3
    ) -> None:
        super().__init__(keys)
        self.mask_key = mask_key
        self.cropper = CenterMask2DSliceCrop(
            roi_size=roi_size,
            crop_mode=crop_mode,
            center_mode=center_mode,
            z_axis=z_axis,
            mask_data=None,
            n_slices=n_slices
        )
    
    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.cropper(d[key], d[self.mask_key])
        return d


class FullMask2DSliceCropd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        mask_key: KeysCollection,
        roi_size: Union[Sequence[int], int],
        crop_mode: str,
        z_axis: int,
        n_slices: int = 3
    ) -> None:
        super().__init__(keys)
        self.mask_key = mask_key
        self.cropper = FullMask2DSliceCrop(
            roi_size=roi_size,
            crop_mode=crop_mode,
            z_axis=z_axis,
            mask_data=None,
            n_slices=n_slices
        )
    
    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        mask = d[self.mask_key]
        centers = self.cropper.get_center_pos_(mask)
        results: List[Dict[Hashable, np.ndarray]] = [dict() for _ in centers]
        for key in data.keys():
            if key in self.keys:
                img = d[key]
                for i, crop in enumerate(self.cropper(img, msk=mask)):
                    results[i][key] = crop
            else:
                for i in range(len(centers)):
                    results[i][key] = data[key]
        
        return results


class GetMaxSlices3direcCropd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        mask_key: KeysCollection,
        roi_size: Union[Sequence[int], int],
        crop_mode: str,
        center_mode: str,
        n_slices: int,
    ) -> None:
        super().__init__(keys)
        self.mask_key = mask_key
        self.cropper = GetMaxSlices3direcCrop(
            roi_size=roi_size,
            crop_mode=crop_mode,
            center_mode=center_mode,
            mask_data=None,
            n_slices=n_slices,
        )
    
    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.cropper(d[key], d[self.mask_key])
        return d

class FullImage2DSliceCropd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        mask_key: KeysCollection,
        roi_size: Union[Sequence[int], int],
        crop_mode: str,
        z_axis: int,
        n_slices: int = 3
    ) -> None:
        super().__init__(keys)
        self.mask_key = mask_key
        self.cropper = FullImage2DSliceCrop(
            roi_size=roi_size,
            crop_mode=crop_mode,
            z_axis=z_axis,
            n_slices=n_slices
        )
    
    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        mask = d[self.mask_key]
        centers = self.cropper.get_center_pos_(mask)
        results: List[Dict[Hashable, np.ndarray]] = [dict() for _ in centers]
        for key in data.keys():
            if key in self.keys:
                img = d[key]
                for i, crop in enumerate(self.cropper(img, msk=mask)):
                    results[i][key] = crop
            else:
                for i in range(len(centers)):
                    results[i][key] = data[key]
        
        return results

class RandCropByPosNegLabelExd(RandCropByPosNegLabeld):
    """Dictionary-based version :py:class:`monai_ex.transforms.RandCropByPosNegLabelEx`.
    """
    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        spatial_size: Union[Sequence[int], int],
        pos: float = 1.0,
        neg: float = 1.0,
        offset: float = 0.0,
        num_samples: int = 1,
        image_key: Optional[str] = None,
        image_threshold: float = 0.0,
        fg_indices_key: Optional[str] = None,
        bg_indices_key: Optional[str] = None,
    ) -> None:
        super().__init__(
            keys=keys,
            label_key=label_key,
            spatial_size=spatial_size,
            pos=pos,
            neg=neg,
            num_samples=num_samples,
            image_key=image_key,
            image_threshold=image_threshold,
            fg_indices_key=fg_indices_key,
            bg_indices_key=bg_indices_key,
        )
        self.offset = offset
        if self.offset < 0:
            raise ValueError(f'Offset value must greater than 0, but got {offset}')



    def randomize(
        self,
        label: np.ndarray,
        fg_indices: Optional[np.ndarray] = None,
        bg_indices: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
    ) -> None:
        self.spatial_size = fall_back_tuple(self.spatial_size, default=label.shape[1:])

        # Select subregion to assure valid roi
        valid_start = np.floor_divide(self.spatial_size, 2)
        # add 1 for random
        valid_end = np.subtract(label.shape[1:] + np.array(1), self.spatial_size / np.array(2)).astype(np.uint16)
        # int generation to have full range on upper side, but subtract unfloored size/2 to prevent rounded range
        # from being too high
        for i in range(len(valid_start)):  # need this because np.random.randint does not work with same start and end
            if valid_start[i] == valid_end[i]:
                valid_end[i] += 1

        def _correct_centers(
            center_ori: List[np.ndarray], valid_start: np.ndarray, valid_end: np.ndarray
        ) -> List[np.ndarray]:
            for i, c in enumerate(center_ori):
                center_i = c
                if c < valid_start[i]:
                    center_i = valid_start[i]
                if c >= valid_end[i]:
                    center_i = valid_end[i] - 1
                center_ori[i] = center_i
            return center_ori

        if fg_indices is None or bg_indices is None:
            fg_indices_, bg_indices_ = map_binary_to_indices(label, image, self.image_threshold)
        else:
            fg_indices_ = fg_indices
            bg_indices_ = bg_indices
        self.centers = generate_pos_neg_label_crop_centers(
            self.spatial_size, self.num_samples, self.pos_ratio, label.shape[1:], fg_indices_, bg_indices_, self.R
        )
        self.offset_centers = []
        for center in self.centers:
            if 0 < self.offset <= 1:
                offset = [self.R.randint(self.offset*sz//2)*self.R.choice([1, -1]) for sz in self.spatial_size]
            elif self.offset > 1:
                offset = [self.R.randint(self.offset)*self.R.choice([1, -1]) for sz in self.spatial_size]
            else:
                offset = [0, ] * len(self.spatial_size)
            # print('Offset: ', offset, "Center: ", center)
            offset_centers = [int(c+b) for c, b in zip(center, offset)]
            self.offset_centers.append(_correct_centers(offset_centers, valid_start, valid_end))

        self.centers = self.offset_centers


CenterMask2DSliceCropD = CenterMask2DSliceCropDict = CenterMask2DSliceCropd
FullMask2DSliceCropD = FullMask2DSliceCropDict = FullMask2DSliceCropd
FullImage2DSliceCropD = FullImage2DSliceCropDict = FullImage2DSliceCropd
GetMaxSlices3direcCropD = GetMaxSlices3direcCropDict = GetMaxSlices3direcCropd
RandCropByPosNegLabelExD = RandCropByPosNegLabelExDict = RandCropByPosNegLabelExd
