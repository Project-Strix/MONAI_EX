"""
A collection of "vanilla" transforms for spatial operations
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

from typing import Optional, Sequence, Union

import numpy as np
import torch

from monai.transforms.compose import Transform
from monai.utils import InterpolateMode, ensure_tuple, ensure_tuple_size


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


class Transpose(Transform):
    def __init__(
            self,
            axes=None
    ) -> None:
        """
            Reverse or permute the axes of an array; returns the modified array.

        """
        self.axes = axes
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return np.transpose(img, axes=self.axes).astype(img.dtype)
