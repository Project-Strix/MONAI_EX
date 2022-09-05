from typing import List, Optional, Sequence, Union, Any, Callable

import numpy as np
import torch

from monai_ex.utils import (
    ensure_list,
    ensure_tuple_rep,
    fall_back_tuple,
)
from monai.transforms.utils import (
    is_positive,
    map_binary_to_indices,
    generate_spatial_bounding_box,
    generate_pos_neg_label_crop_centers,
)
from monai.transforms import (
    Transform,
    Randomizable,
    SpatialCrop,
    ResizeWithPadOrCrop,
    RandCropByPosNegLabel,
)

from monai.config.type_definitions import NdarrayOrTensor


class CenterMask2DSliceCrop(Transform):
    """Extract 2D slices from the image at the
        center of mask with specified ROI size.

    Args:
        roi_size (Union[Sequence[int], int]): the 2D spatial size of the crop region e.g. [224,224]
        crop_mode (str): 2D slice crop mode: "single", "cross", "parallel"
        z_axis (int): the index of z axis (channel dim not counted)
        center_mode (Optional[str], optional): center point calculation mode: "center", "maximum". Defaults to "center".
        mask_data (Optional[np.ndarray], optional): mask data. Defaults to None.
        n_slices (int, optional): the slice# will be croped, if crop_mode is "cross" or "parallel". Defaults to 3.

    Raises:
        ValueError: _description_
        ValueError: _description_
    """
    def __init__(
        self,
        roi_size: Union[Sequence[int], int],
        crop_mode: str,
        z_axis: int,
        center_mode: Optional[str] = "center",
        mask_data: Optional[np.ndarray] = None,
        n_slices: int = 3,
    ) -> None:
        super().__init__()
        self.roi_size = ensure_tuple_rep(roi_size, 2)
        self.mask_data = mask_data
        self.crop_mode = crop_mode
        self.z_axis = z_axis
        self.center_mode = center_mode
        self.n_slices = 1 if crop_mode == "single" else n_slices

        if crop_mode not in ["single", "cross", "parallel"]:
            raise ValueError("Cropping mode must be one of 'single, cross, parallel'")
        if center_mode not in ["center", "maximum"]:
            raise ValueError("Centering mode must be one of 'center, maximum'")

    def get_new_spatial_size(self, z_axis):
        spatial_size_ = ensure_list(self.roi_size)
        if self.crop_mode in ["single", "parallel"]:
            spatial_size_.insert(z_axis, self.n_slices)
        else:
            spatial_size_ = [
                max(spatial_size_),
            ] * 3

        return spatial_size_

    def get_center_pos(self, mask_data, z_axis):
        if self.center_mode == "center":
            starts, ends = generate_spatial_bounding_box(mask_data, lambda x: x > 0)
            return tuple((st + ed) // 2 for st, ed in zip(starts, ends))
        elif self.center_mode == "maximum":
            axes = np.delete(np.arange(3), z_axis)
            mask_data_ = mask_data.squeeze()
            z_index = np.argmax(np.count_nonzero(mask_data_, axis=tuple(axes)))
            if z_index == 0 and self.crop_mode == "parallel":
                z_index = (self.n_slices - 1) // 2
            elif (
                z_index == mask_data_.shape[z_axis] - 1 and self.crop_mode == "parallel"
            ):
                z_index -= (self.n_slices - 1) // 2

            starts, ends = generate_spatial_bounding_box(
                np.take(mask_data_, z_index, z_axis)[np.newaxis, ...], lambda x: x > 0
            )
            centers = [(st + ed) // 2 for st, ed in zip(starts, ends)]
            centers.insert(z_axis, z_index)
            return tuple(centers)

    def __call__(
        self,
        img: np.ndarray,
        msk: Optional[np.ndarray] = None,
        center: Optional[tuple] = None,
        z_axis: Optional[int] = None,
    ):
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

        z_axis_ = z_axis if z_axis is not None else self.z_axis

        if center is None:
            center = self.get_center_pos(mask_data_, z_axis_)

        if self.crop_mode in ["single", "parallel"]:
            size_ = self.get_new_spatial_size(z_axis_)
            size_ = list(map(int, size_))
            slice_ = SpatialCrop(roi_center=center, roi_size=size_)(img)
            if np.any(slice_.shape[1:] != size_):
                slice_ = ResizeWithPadOrCrop(spatial_size=size_)(slice_)

            return np.moveaxis(slice_.squeeze(0), z_axis_, 0)
        else:
            cross_slices = np.zeros(shape=(3,) + self.roi_size)
            for k in range(3):
                size_ = np.insert(self.roi_size, k, 1)
                slice_ = SpatialCrop(roi_center=center, roi_size=size_)(img)
                if np.any(slice_.shape[1:] != size_):
                    slice_ = ResizeWithPadOrCrop(spatial_size=size_)(slice_)

                cross_slices[k] = slice_.squeeze()
            return cross_slices


class FullMask2DSliceCrop(CenterMask2DSliceCrop):
    def __init__(
        self,
        roi_size: Union[Sequence[int], int],
        crop_mode: str,
        z_axis: int,
        mask_data: Optional[np.ndarray] = None,
        n_slices: int = 3,
    ):
        super().__init__(
            roi_size=roi_size,
            crop_mode=crop_mode,
            z_axis=z_axis,
            mask_data=mask_data,
            n_slices=n_slices,
        )

    def get_center_pos_(self, mask_data):
        axes = np.delete(np.arange(3), self.z_axis)
        starts, ends = generate_spatial_bounding_box(mask_data, lambda x: x > 0)
        z_start, z_end = (
            starts[self.z_axis] + (self.n_slices - 1) // 2,
            ends[self.z_axis] - (self.n_slices - 1) // 2,
        )
        centers = []
        for z in np.arange(z_start, z_end):
            center = [
                (st + ed) // 2
                for st, ed in zip(np.array(starts)[axes], np.array(ends)[axes])
            ]
            center.insert(self.z_axis, z)
            centers.append(tuple(center))

        return centers

    def __call__(self, img: np.ndarray, msk: Optional[np.ndarray] = None):
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

        centers = self.get_center_pos_(mask_data_)
        slices = [
            super(FullMask2DSliceCrop, self).__call__(img, msk, center)
            for center in centers
        ]
        return slices


class GetMaxSlices3direcCrop(CenterMask2DSliceCrop):
    def __init__(
        self,
        roi_size: Union[Sequence[int], int],
        crop_mode: str,
        center_mode: str,
        mask_data: Optional[np.ndarray] = None,
        n_slices: int = 3,
    ):
        super().__init__(
            roi_size=roi_size,
            crop_mode=crop_mode,
            center_mode=center_mode,
            z_axis=None,
            mask_data=mask_data,
            n_slices=n_slices,
        )

    def __call__(self, img: np.ndarray, msk: Optional[np.ndarray] = None):
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

        final_slices = np.empty(shape=(0, self.roi_size[0], self.roi_size[1]))
        for z_axis in range(3):
            slices = super(GetMaxSlices3direcCrop, self).__call__(
                img, msk, z_axis=z_axis
            )
            final_slices = np.concatenate([final_slices, slices])
        return final_slices


class FullImage2DSliceCrop(CenterMask2DSliceCrop):
    def __init__(
        self,
        roi_size: Union[Sequence[int], int],
        crop_mode: str,
        z_axis: int,
        n_slices: int = 3
    ):
        super().__init__(
            roi_size=roi_size,
            crop_mode=crop_mode,
            z_axis=z_axis,
            mask_data=None,
            n_slices=n_slices
        )
    
    def get_center_pos_(self, img):
        axes = np.delete(np.arange(3), self.z_axis)
        starts = [0, 0, 0]
        ends = [img.shape[1], img.shape[2], img.shape[3]]
        centers = []
        for z in np.arange(0, img.shape[3]):
            center = [(st+ed)//2 for st, ed in zip(np.array(starts)[axes], np.array(ends)[axes])]
            center.insert(self.z_axis, z)
            centers.append(tuple(center))
        return centers
    
    def __call__(self, img: np.ndarray, msk: Optional[np.ndarray]=None):
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

        centers = self.get_center_pos_(mask_data_)
        slices = [super(FullImage2DSliceCrop, self).__call__(img, msk, center) for center in centers]
        return slices


class RandCropByPosNegLabelEx(RandCropByPosNegLabel):
    """Extension of RandCropByPosNegLabel.
    Extended: offset.
    Extended: use ResizeWithPadOrCrop to support excceed spatial_size.
    Extended: add target_label to specify one label in label image.

    Crop random fixed sized regions with the center being a foreground or background voxel
    based on the Pos Neg Ratio.
    And will return a list of arrays for all the cropped images.
    For example, crop two (3 x 3) arrays from (5 x 5) array with pos/neg=1::

        [[[0, 0, 0, 0, 0],
          [0, 1, 2, 1, 0],            [[0, 1, 2],     [[2, 1, 0],
          [0, 1, 3, 0, 0],     -->     [0, 1, 3],      [3, 0, 0],
          [0, 0, 0, 0, 0],             [0, 0, 0]]      [0, 0, 0]]
          [0, 0, 0, 0, 0]]]

    Args:
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            If its components have non-positive values, the corresponding size of `label` will be used.
        label: the label image that is used for finding foreground/background, if None, must set at
            `self.__call__`.  Non-zero indicates foreground, zero indicates background.
        pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        offset: add randonm offset to the center, if 0<offset<1, then offset_range = [0, offset*spatial_size/2),
            if offset > 1, then offset_range = [0, offset)
        num_samples: number of samples (crop regions) to take in each list.
        image: optional image data to help select valid area, can be same as `img` or another image array.
            if not None, use ``label == 0 & image > image_threshold`` to select the negative
            sample (background) center. So the crop center will only come from the valid image areas.
        image_threshold: if enabled `image`, use ``image > image_threshold`` to determine
            the valid image content areas.
        fg_indices: if provided pre-computed foreground indices of `label`, will ignore above `image` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices`
            and `bg_indices` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndices` transform first and cache the results.
        bg_indices: if provided pre-computed background indices of `label`, will ignore above `image` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices`
            and `bg_indices` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndices` transform first and cache the results.

    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.
    """

    def __init__(
        self,
        spatial_size: Union[Sequence[int], int],
        label: Optional[np.ndarray] = None,
        pos: float = 1.0,
        neg: float = 1.0,
        offset: float = 0.0,
        num_samples: int = 1,
        image: Optional[np.ndarray] = None,
        image_threshold: float = 0.0,
        fg_indices: Optional[np.ndarray] = None,
        bg_indices: Optional[np.ndarray] = None,
        target_label: Optional[int] = None,
    ):
        super().__init__(
            spatial_size=spatial_size,
            label=label,
            pos=pos,
            neg=neg,
            num_samples=num_samples,
            image=image,
            image_threshold=image_threshold,
            fg_indices=fg_indices,
            bg_indices=bg_indices,
        )
        self.offset = offset
        self.target_label = target_label
        if self.offset < 0:
            raise ValueError(f"Offset value must greater than 0, but got {offset}")

    def randomize(
        self,
        label: np.ndarray,
        fg_indices: Optional[np.ndarray] = None,
        bg_indices: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
    ) -> None:
        self.spatial_size = fall_back_tuple(self.spatial_size, default=label.shape[1:])
        if np.greater(self.spatial_size, label.shape[1:]).any():
            self.centers = [
                None,
            ] * self.num_samples
            return

        if fg_indices is None or bg_indices is None:
            fg_indices_, bg_indices_ = map_binary_to_indices(
                label, image, self.image_threshold
            )
        else:
            fg_indices_ = fg_indices
            bg_indices_ = bg_indices
        self.centers = generate_pos_neg_label_crop_centers(
            self.spatial_size,
            self.num_samples,
            self.pos_ratio,
            label.shape[1:],
            fg_indices_,
            bg_indices_,
            self.R,
        )

        self.offset_centers = []
        for center in self.centers:
            if 0 < self.offset <= 1:
                offset = [
                    self.R.randint(self.offset * sz // 2) * self.R.choice([1, -1])
                    for sz in self.spatial_size
                ]
            elif self.offset > 1:
                offset = [
                    self.R.randint(self.offset) * self.R.choice([1, -1])
                    for sz in self.spatial_size
                ]
            else:
                offset = [
                    0,
                ] * len(self.spatial_size)
            self.offset_centers.append([int(c + b) for c, b in zip(center, offset)])
        self.centers = self.offset_centers

    def __call__(
        self,
        img: np.ndarray,
        label: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
        fg_indices: Optional[np.ndarray] = None,
        bg_indices: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """
        Args:
            img: input data to crop samples from based on the pos/neg ratio of `label` and `image`.
                Assumes `img` is a channel-first array.
            label: the label image that is used for finding foreground/background, if None, use `self.label`.
            image: optional image data to help select valid area, can be same as `img` or another image array.
                use ``label == 0 & image > image_threshold`` to select the negative sample(background) center.
                so the crop center will only exist on valid image area. if None, use `self.image`.
            fg_indices: foreground indices to randomly select crop centers,
                need to provide `fg_indices` and `bg_indices` together.
            bg_indices: background indices to randomly select crop centers,
                need to provide `fg_indices` and `bg_indices` together.

        """
        if label is None:
            label = self.label
        if label is None:
            raise ValueError("label should be provided.")
        if image is None:
            image = self.image
        if fg_indices is None or bg_indices is None:
            if self.fg_indices is not None and self.bg_indices is not None:
                fg_indices = self.fg_indices
                bg_indices = self.bg_indices
            else:
                fg_indices, bg_indices = map_binary_to_indices(
                    label, image, self.image_threshold
                )

        if self.target_label is not None:
            label = (label == self.target_label).astype(np.uint8)

        self.randomize(label, fg_indices, bg_indices, image)
        results: List[np.ndarray] = []
        if self.centers is not None:
            for center in self.centers:
                if np.any(np.greater(self.spatial_size, img.shape[1:])):
                    cropper = ResizeWithPadOrCrop(spatial_size=self.spatial_size)
                else:
                    cropper = SpatialCrop(roi_center=tuple(center), spatial_size=self.spatial_size)  # type: ignore
                results.append(cropper(img))

        return results


class Extract3DImageToSlices(Transform):
    backend: SpatialCrop.backend
    """Extract 3D image to slices along given axis.

    Args:
        axis (int): the index of axis to extract slices.
    """

    def __init__(self, axis: int) -> None:
        super().__init__()
        self.axis = axis

    def get_slice_indices(self, img: NdarrayOrTensor):
        return list(range(0, img.shape[self.axis + 1]))

    def __call__(self, img: NdarrayOrTensor, slice_indices: Optional[List] = None):
        if slice_indices is None:
            slice_indices = self.get_slice_indices(img)

        if isinstance(img, np.ndarray):
            return [np.take(img, slice_idx, axis=self.axis + 1) for slice_idx in slice_indices]
        elif isinstance(img, torch.Tensor):
            return [torch.index_select(img, dim=self.axis + 1, index=torch.tensor(slice_idx)).squeeze(self.axis + 1) for slice_idx in slice_indices]
        else:
            raise NotImplementedError(f"Only support np.array and torch.Tensor, but got {type(img)}")


class SelectSlicesByMask(Extract3DImageToSlices):
    backend = SpatialCrop.backend
    """Select specific slices from 3D image based on mask data.

    Args:
        axis (int): the index of z axis (channel dim not counted)
        slice_select_mode (Optional[str], optional): select slices by different mask calculation mode: "center", "maximum", "all". Defaults to "center".
           if set to "all", all available masked slices will be selected.
        mask_data (Optional[np.ndarray], optional): mask data. Defaults to None.
        mask_select_fn (Callable): condition to get available mask region. Defaults to "is_positive",
    """

    def __init__(
        self,
        axis: int,
        slice_select_mode: Optional[str] = "center",
        mask_data: Optional[np.ndarray] = None,
        mask_select_fn: Callable = is_positive,
    ) -> None:
        super().__init__(axis=axis)
        self.mask_data = mask_data
        self.mask_select_fn = mask_select_fn
        self.slice_select_mode = slice_select_mode

    def get_slice_indices(self, mask_data, axis):
        if self.slice_select_mode == "center":
            starts, ends = generate_spatial_bounding_box(mask_data, self.mask_select_fn)
            return [(starts[axis] + ends[axis]) // 2]
        elif self.slice_select_mode == "all":
            starts, ends = generate_spatial_bounding_box(mask_data, self.mask_select_fn)
            return list(range(starts[axis], ends[axis] + 1))
        elif self.slice_select_mode == "maximum":
            axes = np.delete(np.arange(3), axis)
            mask_data_ = mask_data.squeeze()
            z_index = np.argmax(np.count_nonzero(mask_data_, axis=tuple(axes)))
            return [z_index]

    def __call__(
        self,
        img: NdarrayOrTensor,
        msk: Optional[np.ndarray] = None,
    ) -> Any:
        if self.mask_data is None and msk is None:
            raise ValueError("Unknown mask_data.")

        mask_data_ = msk if msk is not None else self.mask_data
        mask_data_ = self.mask_select_fn(mask_data_)

        if mask_data_.shape[0] != 1 and mask_data_.shape[0] != img.shape[0]:
            raise ValueError(
                "When mask_data is not single channel, mask_data channels must match img, "
                f"got img={img.shape[0]} mask_data={mask_data_.shape[0]}."
            )

        slice_indices = self.get_slice_indices(mask_data_, self.axis)

        return super().__call__(img, slice_indices)


class RandSelectSlicesFromImage(Randomizable):
    backend = SpatialCrop.backend
    """Randomly select one slice from 3D volume along given axis.

    Args:
        dim (int, optional): The axis to randomly select, ignoring the channel dim and must be postive. Defaults to 0.
        num_samples (int, optional): The slice number selected once. Defaults to 1.
    """

    def __init__(self, dim: int = 0, num_samples: int = 1) -> None:
        self.dim = dim
        self.num_samples = num_samples

    def randomize(self, low, high) -> None:
        return self.R.randint(low, high, size=self.num_samples)

    def __call__(self, img: NdarrayOrTensor, slice_idx: Optional[Union[int, np.ndarray]] = None) -> Any:
        spatial_shape = img.shape[1:]
        slice_num = spatial_shape[self.dim]
        if slice_idx is None:
            slice_indices = self.randomize(0, slice_num)
        else:
            slice_indices = slice_idx

        results: List[NdarrayOrTensor] = []
        if slice_indices is not None:                
            for index in slice_indices:
                if isinstance(img, np.ndarray):
                    results.append(np.take(img, index, axis=self.dim+1))
                elif isinstance(img, torch.Tensor):
                    results.append(torch.index_select(img, dim=self.dim+1, index=torch.tensor(index)).squeeze(self.dim+1))
                else:
                    raise NotImplementedError(f"Only support np.array and torch.Tensor, but got {type(img)}")
        return results


class SpatialCropByMask(Transform):
    """Spatial crop images centered at the mask center.

    Args:
        roi_size (Union[Sequence[int], NdarrayOrTensor, None]): size of the crop ROI.
        mask_select_fn (Callable, optional): function to select expected foreground, default is to select values > 0.
    """
    def __init__(
        self,
        roi_size: Union[Sequence[int], NdarrayOrTensor, None],
        mask_select_fn: Callable = is_positive
    ) -> None:
        self.roi_size = roi_size
        self.mask_select_fn = mask_select_fn

    def __call__(self, img: NdarrayOrTensor, msk: Optional[NdarrayOrTensor]) -> NdarrayOrTensor:
        cropped_data = []
        if msk is None:
            msk = img

        for channel in range(img.shape[0]):
            start_, end_ = generate_spatial_bounding_box(msk, select_fn=self.mask_select_fn, channel_indices=channel)
            cropped = SpatialCrop(roi_center=np.add(start_, end_) // 2, roi_size=self.roi_size)(img)
            cropped_data.append(cropped)

        return np.concatenate(cropped_data, axis=0)
