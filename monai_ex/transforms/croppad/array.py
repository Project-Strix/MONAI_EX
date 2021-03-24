from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from monai.config import IndexSelection
from monai_ex.utils import ensure_list, ensure_tuple_rep
from monai.transforms.utils import generate_spatial_bounding_box
from monai.transforms import (
    Transform,
    SpatialCrop,
    ResizeWithPadOrCrop
)


class CenterMask2DSliceCrop(Transform):
    """
    Extract 2D slices from the image at the 
    center of mask with specified ROI size.

    Args:
        roi_size: the spatial size of the crop region e.g. [224,224,128]
            If its components have non-positive values, the corresponding size of input image will be used.
    """

    def __init__(
        self,
        roi_size: Union[Sequence[int], int],
        crop_mode: str,
        z_axis: int,
        center_mode: Optional[str]='center',
        mask_data: Optional[np.ndarray]=None,
    ) -> None:
        super().__init__()
        self.roi_size = ensure_tuple_rep(roi_size, 2)
        self.mask_data = mask_data
        self.crop_mode = crop_mode
        self.z_axis = z_axis
        self.center_mode = center_mode
        if crop_mode not in ['single', 'cross', 'parallel']:
            raise ValueError("Cropping mode must be one of 'single, cross, parallel'")
        if center_mode not in ['center', 'maximum']:
            raise ValueError("Centering mode must be one of 'center, maximum'")

    def get_new_spatial_size(self):
        spatial_size_ = ensure_list(self.roi_size)
        if self.crop_mode == 'single':
            spatial_size_.insert(self.z_axis, 1)
        elif self.crop_mode == 'parallel':
            spatial_size_.insert(self.z_axis, 3)
        else:
            spatial_size_ = [max(spatial_size_),]*3

        return spatial_size_

    def get_center_pos(self, mask_data):
        if self.center_mode == 'center':
            starts, ends = generate_spatial_bounding_box(mask_data, lambda x:x>0)
            return [(st+ed)//2 for st, ed in zip(starts, ends)]
        elif self.center_mode == 'maximum':
            axes = list(range(3))
            axes.remove(self.z_axis)
            mask_data_ = mask_data.squeeze()
            z_index = np.argmax(np.count_nonzero(mask_data_, axis=tuple(axes)))
            if z_index == 0 and self.crop_mode == 'parallel':
                z_index += 1
            elif z_index == mask_data_.shape[self.z_axis]-1 and self.crop_mode == 'parallel':
                z_index -= 1 
            
            starts, ends = generate_spatial_bounding_box(np.take(mask_data_, z_index, self.z_axis)[np.newaxis,...], lambda x:x>0)
            centers = [(st+ed)//2 for st, ed in zip(starts, ends)]
            centers.insert(self.z_axis, z_index)
            return centers

    def __call__(self, img: np.ndarray, msk: Optional[np.ndarray]=None):
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        if self.mask_data is None and msk is None:
            raise ValueError("Unknown mask_data.")
        mask_data_ = np.array([[1]])
        if self.mask_data is not None and msk is None:
            mask_data_ = self.mask_data > 0
        if msk is not None:
            mask_data_ = msk > 0
        mask_data_ = np.asarray(mask_data_)

        if mask_data_.shape[0] != 1 and mask_data_.shape[0] != img.shape[0]:
            raise ValueError(
                "When mask_data is not single channel, mask_data channels must match img, "
                f"got img={img.shape[0]} mask_data={mask_data_.shape[0]}."
            )
    
        center = self.get_center_pos(mask_data_)

        if self.crop_mode in ['single', 'parallel']:
            size_ = self.get_new_spatial_size()
            slice_ = SpatialCrop(roi_center=center, roi_size=size_)(img)
            if np.any(slice_.shape[1:] != size_):
                slice_ = ResizeWithPadOrCrop(spatial_size=size_)(slice_)

            return np.moveaxis(slice_.squeeze(0), self.z_axis, 0)
        else:
            cross_slices = np.zeros(shape=(3,)+self.roi_size)
            for k in range(3):
                size_ = np.insert(self.roi_size, k, 1)
                slice_ = SpatialCrop(roi_center=center, roi_size=size_)(img)
                if np.any(slice_.shape[1:] != size_):
                    slice_ = ResizeWithPadOrCrop(spatial_size=size_)(slice_)
                
                cross_slices[k] = slice_.squeeze()
            return cross_slices        
