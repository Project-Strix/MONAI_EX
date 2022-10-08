"""
A collection of "vanilla" transforms for spatial operations
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

from typing import Optional, Sequence, Union, Tuple

import numpy as np
import torch

from monai.transforms.compose import Transform, Randomizable
from monai.transforms.croppad.array import ResizeWithPadOrCrop
from monai.transforms.utils import Fourier
from monai.data.utils import compute_shape_offset, to_affine_nd, zoom_affine
from monai.utils import InterpolateMode, ensure_tuple, ensure_tuple_size
from monai.utils.type_conversion import NdarrayOrTensor, convert_data_type, convert_to_dst_type
from scipy import ndimage as ndi


class FixedResize(Transform):
    """
    Resize the input image to given spatial size with fixed aspect ratio.
    Implemented using :py:class:`torch.nn.functional.interpolate`.

    Args:
        spatial_size: expected shape of spatial dimensions after resize operation.
            if the components of the `spatial_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the original spatial dimension size is `(64, 128)`.
        mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
    """

    def __init__(
        self,
        spatial_size: Union[Sequence[int], int],
        mode: Union[InterpolateMode, str] = InterpolateMode.AREA,
        align_corners: Optional[bool] = None,
    ) -> None:
        self.spatial_size = spatial_size
        self.mode: InterpolateMode = InterpolateMode(mode)
        self.align_corners = align_corners

    def __call__(
        self,
        img: np.ndarray,
        mode: Optional[Union[InterpolateMode, str]] = None,
        align_corners: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]).
            mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
                The interpolation mode. Defaults to ``self.mode``.
            align_corners: This only has an effect when mode is
                'linear', 'bilinear', 'bicubic' or 'trilinear'. Defaults to ``self.align_corners``.

        Raises:
            ValueError: When ``self.spatial_size`` length is less than ``img`` spatial dimensions.

        """
        input_ndim = img.ndim - 1  # spatial ndim
        output_ndim = len(ensure_tuple(self.spatial_size))
        if output_ndim > input_ndim:
            input_shape = ensure_tuple_size(img.shape, output_ndim + 1, 1)
            img = img.reshape(input_shape)
        elif isinstance(self.spatial_size, tuple) and output_ndim < input_ndim:
            raise ValueError(
                "len(spatial_size) must be greater or equal to img spatial dimensions, "
                f"got spatial_size={output_ndim} img={input_ndim}."
            )
        assert np.count_nonzero(np.greater(self.spatial_size, 0)) == 1, \
                f"Spatial_size should have only one value > 0, but got {self.spatial_size}"

        if isinstance(self.spatial_size, int):
            spatial_size_ = (self.spatial_size, ) * (img.ndim - 1)
        else:
            spatial_size_ = self.spatial_size

        for idx in np.where(np.equal(spatial_size_, 0))[0]: #change 0 to -1
            spatial_size_[idx] = -1

        aspect_ratio = np.divide(img.squeeze().shape, spatial_size_)
        ratio = aspect_ratio[np.greater(aspect_ratio, 0)]
        if len(ratio) > 1:
            ratio = np.mean(ratio)
        spatial_size = np.divide(img.squeeze().shape, ratio).astype(np.int)

        resized = _torch_interp(
            input=torch.as_tensor(np.ascontiguousarray(img), dtype=torch.float).unsqueeze(0),
            size=tuple(spatial_size),
            mode=self.mode.value if mode is None else InterpolateMode(mode).value,
            align_corners=self.align_corners if align_corners is None else align_corners,
        )
        resized = resized.squeeze(0).detach().cpu().numpy()
        return resized


class LabelMorphology(Transform):
    def __init__(
        self,
        mode: str,
        radius: int,
        binary: bool
    ):
        """
        Args:
            mode: morphology mode, e.g. 'closing', 'dilation', 'erosion', 'opening'
            radius: radius of morphology operation.
            binary: whether using binary morphology (for binary data)

        """
        self.mode = mode
        self.radius = radius
        self.binary = binary
        assert self.mode in ['closing', 'dilation', 'erosion', 'opening'], \
            f"Mode must be one of 'closing', 'dilation', 'erosion', 'opening', but got {self.mode}"

    def __call__(
        self,
        img: np.ndarray,
        mode: Optional[str] = None,
        radius: Optional[int] = None,
        binary: Optional[bool] = None
    ) -> np.ndarray:
        """
        Apply the transform to `img`.

        """
        self.mode = self.mode if mode is None else mode
        self.radius = self.radius if radius is None else radius
        self.binary = self.binary if binary is None else binary

        input_ndim = img.squeeze().ndim # spatial ndim
        if input_ndim == 2:
            structure = ndi.generate_binary_structure(2, 1)
        elif input_ndim == 3:
            structure = ndi.generate_binary_structure(3, 1)
        else:
            raise ValueError(f'Currently only support 2D&3D data, but got image with shape of {img.shape}')

        channel_dim = None
        if input_ndim != img.ndim:
            channel_dim = img.shape.index(1)
            img = img.squeeze()

        if self.mode == 'closing':
            if self.binary:
                img = ndi.binary_closing(img, structure=structure, iterations=self.radius)
            else:
                for _ in range(self.radius):
                    img = ndi.grey_closing(img, footprint=structure)        
        elif self.mode == 'dilation':
            if self.binary:
                img = ndi.binary_dilation(img, structure=structure, iterations=self.radius)
            else:
                for _ in range(self.radius):
                    img = ndi.grey_dilation(img, footprint=structure)
        elif self.mode == 'erosion':
            if self.binary:
                img = ndi.binary_erosion(img, structure=structure, iterations=self.radius)
            else:
                for _ in range(self.radius):
                    img = ndi.grey_erosion(img, footprint=structure)
        elif self.mode == 'opening':
            if self.binary:
                img = ndi.binary_opening(img, structure=structure, iterations=self.radius)
            else:
                for _ in range(self.radius):
                    img = ndi.grey_opening(img, footprint=structure)
        else:
            raise ValueError(f'Unexpected keyword {self.mode}')

        if channel_dim is not None:
            return np.expand_dims(img, axis=channel_dim)
        else:
            return img


class RandLabelMorphology(Randomizable, Transform):
    def __init__(
        self,
        prob: float,
        mode: str,
        radius: int,
        binary: bool
    ):
        self.converter = LabelMorphology(mode, radius, binary)

    def randomize(self):
        self._do_transform = self.R.random() < self.prob

    def __call__(
        self,
        image,
        mode: Optional[str] = None,
        radius: Optional[int] = None,
        binary: Optional[bool] = None
    ):
        self.randomize()
        if not self._do_transform:
            return image

        return self.converter(image, mode, radius, binary)


class Rotate90Ex(Transform):
    """
    Extension of :py:class:`monai.transforms.Rotate90`.
    Add torch.tensor data support

    Rotate an array by 90 degrees in the plane specified by `axes`.
    """

    def __init__(self, k: int = 1, spatial_axes: Tuple[int, int] = (0, 1)) -> None:
        """
        Args:
            k: number of times to rotate by 90 degrees.
            spatial_axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
        """
        self.k = k
        self.spatial_axes = spatial_axes

    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        rotated = list()
        for channel in img:
            rotated.append(np.rot90(channel, self.k, self.spatial_axes))

        if isinstance(img, np.ndarray):
            return np.stack(rotated).astype(img.dtype)
        else:
            return np.stack(rotated)


class KSpaceResample(Transform, Fourier):
    """Resample input image into the specified `pixdim` in K-space domain."""

    def __init__(
        self,
        pixdim: Union[Sequence[float], float],
        diagonal: bool = False,
        device: Optional[torch.device] = None,
        tolerance: float = 1e-3,
        image_only: bool = False,
    ) -> None:
        """
        Args:
            pixdim (Union[Sequence[float], float]): output voxel spacing. if providing a single number,
                will use it for the first dimension. items of the pixdim sequence map to the spatial
                dimensions of input image, if length of pixdim sequence is longer than image spatial dimensions,
                will ignore the longer part, if shorter, will pad with the last value. 
                For example, for 3D image if pixdim is [2.0] will be padded to [2.0, 2.0, 2.0];
                    [1.0, 2.0] it will be padded to [1.0, 2.0, 2.0]
            diagonal (bool, optional): whether to resample the input to have a diagonal affine matrix.
                If True, the input data is resampled to the following affine::
                    np.diag((pixdim_0, pixdim_1, ..., pixdim_n, 1))
                This effectively resets the volume to the world coordinate system (RAS+ in nibabel).
                The original orientation, rotation, shearing are not preserved.
                If False, this transform preserves the axes orientation, orthogonal rotation and
                translation components from the original affine. This option will not flip/swap axes
                of the original data. Defaults to False.
            device (Optional[torch.device], optional): device to store the output grid data. Defaults to None.
            tolerance (float): Skip resample if spacing is same within given tolerance.
            image_only (bool): return just the image or image with old affine and new affine. Default is `False`.
        """
        super().__init__()
        self.pixdim = np.array(ensure_tuple(pixdim), dtype=np.float64)
        self.diagonal = diagonal
        self.device = device
        self.tolerance = tolerance
        self.image_only = image_only

    def __call__(self, img: NdarrayOrTensor, affine: Optional[np.ndarray] = None):
        sr = int(img.ndim - 1)
        if sr <= 0:
            raise ValueError("data_array must have at least one spatial dimension.")
        if affine is None:
            # default to identity
            affine = np.eye(sr + 1, dtype=np.float64)
            affine_ = np.eye(sr + 1, dtype=np.float64)
        else:
            affine_np, *_ = convert_data_type(affine, np.ndarray)  # type: ignore
            affine_ = to_affine_nd(sr, affine_np)

        out_d = self.pixdim[:sr]
        if out_d.size < sr:
            out_d = np.append(out_d, [out_d[-1]] * (sr - out_d.size))
        if np.any(out_d <= 0):
            raise ValueError(f"pixdim must be positive, got {out_d}.")

        # compute output affine, shape and offset
        new_affine = zoom_affine(affine_, out_d, diagonal=self.diagonal)
        output_shape, offset = compute_shape_offset(img.shape[1:], affine_, new_affine)
        new_affine[:sr, -1] = offset[:sr]
        transform = np.linalg.inv(affine_) @ new_affine
        # adapt to the actual rank
        transform = to_affine_nd(sr, transform)

        # no resampling if it's identity transform
        if np.allclose(transform, np.diag(np.ones(len(transform))), atol=self.tolerance):
            output_data = img
        else:
            k = self.shift_fourier(img, sr)
            k = ResizeWithPadOrCrop(spatial_size=output_shape)(k)
            output_data = self.inv_shift_fourier(k, sr)

        output_data, *_ = convert_to_dst_type(output_data, img, dtype=torch.float32)
        new_affine = to_affine_nd(affine_np, new_affine)  # type: ignore
        new_affine, *_ = convert_to_dst_type(src=new_affine, dst=affine, dtype=torch.float32)

        if self.image_only:
            return output_data
        return output_data, affine, new_affine
