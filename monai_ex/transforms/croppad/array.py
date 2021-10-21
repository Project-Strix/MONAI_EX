from typing import List, Optional, Sequence, Union

import numpy as np
import torch

from monai_ex.utils import (
    ensure_list,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
    dtype_numpy_to_torch,
)
from monai.transforms.utils import (
    map_binary_to_indices,
    generate_spatial_bounding_box,
    generate_pos_neg_label_crop_centers,
)
from monai.transforms import (
    Transform,
    SpatialCrop,
    ResizeWithPadOrCrop,
    RandCropByPosNegLabel,
)
from monai.data.utils import compute_shape_offset, to_affine_nd, zoom_affine

try:
    # PT_BEFORE_1_8 = torch.__version__ != "1.8.0" and version_leq(torch.__version__, "1.8.0")
    PT_BEFORE_1_8 = float(torch.__version__[:3]) < 1.8
except (AttributeError, TypeError):
    PT_BEFORE_1_8 = True

if PT_BEFORE_1_8:
    import numpy.fft as fft
    from numpy import absolute
else:
    import torch.fft as fft
    from torch import absolute


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


class KSpaceResample(Transform):
    """Resample input image into the specified `pixdim` in K-space domain."""

    def __init__(
        self,
        pixdim: Union[Sequence[float], float],
        diagonal: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            pixdim (Union[Sequence[float], float]): output voxel spacing. if providing a single number,
                will use it for the first dimension. items of the pixdim sequence map to the spatial
                dimensions of input image, if length of pixdim sequence is longer than image spatial dimensions,
                will ignore the longer part, if shorter, will pad with `1.0`.
            diagonal (bool, optional): whether to resample the input to have a diagonal affine matrix.
                If True, the input data is resampled to the following affine::
                    np.diag((pixdim_0, pixdim_1, ..., pixdim_n, 1))
                This effectively resets the volume to the world coordinate system (RAS+ in nibabel).
                The original orientation, rotation, shearing are not preserved.
                If False, this transform preserves the axes orientation, orthogonal rotation and
                translation components from the original affine. This option will not flip/swap axes
                of the original data. Defaults to False.
            device (Optional[torch.device], optional): device to store the output grid data. Defaults to None.
        """
        super().__init__()
        self.pixdim = np.array(ensure_tuple(pixdim), dtype=np.float64)
        self.diagonal = diagonal
        self.device = device

    def image2kspace(self, tensor: Union[np.ndarray, torch.Tensor]):
        return fft.fftshift(fft.fftn(fft.ifftshift(tensor)))

    def kspace2image(self, kspace: Union[np.ndarray, torch.Tensor]):
        return fft.fftshift(fft.ifftn(fft.ifftshift(kspace))).real

    def __call__(
        self,
        img: Union[np.ndarray, torch.Tensor],
        affine: Optional[np.ndarray] = None,
    ):
        sr = int(img.ndim - 1)
        if sr <= 0:
            raise ValueError("data_array must have at least one spatial dimension.")
        if affine is None:
            # default to identity
            affine = np.eye(sr + 1, dtype=np.float64)
            affine_ = np.eye(sr + 1, dtype=np.float64)
        else:
            affine_ = to_affine_nd(sr, affine)

        out_d = self.pixdim[:sr]
        if out_d.size < sr:
            out_d = np.append(out_d, [1.0] * (out_d.size - sr))
        if np.any(out_d <= 0):
            raise ValueError(f"pixdim must be positive, got {out_d}.")

        # compute output affine, shape and offset
        new_affine = zoom_affine(affine_, out_d, diagonal=self.diagonal)
        output_shape, offset = compute_shape_offset(img.shape[1:], affine_, new_affine)
        new_affine[:sr, -1] = offset[:sr]

        original_dtype = img.dtype
        convert_to_np = PT_BEFORE_1_8 and isinstance(img, torch.Tensor)
        convert_to_torch = not PT_BEFORE_1_8 and isinstance(img, np.ndarray)

        if convert_to_np:
            img = img.cpu().numpy()
        elif convert_to_torch:
            img = torch.as_tensor(img, dtype=dtype_numpy_to_torch(original_dtype))

        if isinstance(img, torch.Tensor):
            img = img.to(self.device)

        kdata = self.image2kspace(img)
        kdata = ResizeWithPadOrCrop(spatial_size=output_shape)(kdata)
        resampled_data = self.kspace2image(kdata)

        if convert_to_np:
            return (
                torch.as_tensor(resampled_data, dtype=original_dtype),
                affine,
                new_affine,
            )
        elif convert_to_torch:
            return (
                resampled_data.cpu().numpy().astype(original_dtype),
                affine,
                new_affine,
            )
        else:
            return resampled_data.astype(original_dtype), affine, new_affine
