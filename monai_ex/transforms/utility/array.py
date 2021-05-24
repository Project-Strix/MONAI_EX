from typing import Callable, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
from numpy.lib.arraysetops import isin
import torch

from monai.transforms.compose import Transform


numpy_to_torch_dtype_dict = {
        np.bool       : torch.bool,
        np.uint8      : torch.uint8,
        np.int8       : torch.int8,
        np.int16      : torch.int16,
        np.int32      : torch.int32,
        np.int64      : torch.int64,
        np.float16    : torch.float16,
        np.float32    : torch.float32,
        np.float64    : torch.float64,
        np.complex64  : torch.complex64,
        np.complex128 : torch.complex128
    }

string_to_torch_dtype_dict = {
        'bool'       : torch.bool,
        'uint8'      : torch.uint8,
        'int8'       : torch.int8,
        'int16'      : torch.int16,
        'int32'      : torch.int32,
        'int64'      : torch.int64,
        'float16'    : torch.float16,
        'float32'    : torch.float32,
        'float64'    : torch.float64,
        'complex64'  : torch.complex64,
        'complex128' : torch.complex128
}

class CastToTypeEx(Transform):
    """
    Cast the Numpy data to specified numpy data type, or cast the PyTorch Tensor to
    specified PyTorch data type.
    """

    def __init__(self, dtype: Union[np.dtype, torch.dtype, str] = np.float32) -> None:
        """
        Args:
            dtype: convert image to this data type, default is `np.float32`.
        """
        self.dtype = dtype

    def __call__(
        self, img: Union[np.ndarray, torch.Tensor], dtype: Optional[Union[np.dtype, torch.dtype,]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply the transform to `img`, assuming `img` is a numpy array or PyTorch Tensor.

        Args:
            dtype: convert image to this data type, default is `self.dtype`.

        Raises:
            TypeError: When ``img`` type is not in ``Union[numpy.ndarray, torch.Tensor]``.

        """
        dtype_ = self.dtype if dtype is None else dtype

        if isinstance(img, np.ndarray):
            return img.astype(dtype_)
        elif torch.is_tensor(img):
            if isinstance(dtype_, str):
                return torch.as_tensor(img, dtype=string_to_torch_dtype_dict[dtype_])
            elif isinstance(dtype_, np.dtype):
                return torch.as_tensor(img, dtype=numpy_to_torch_dtype_dict[dtype_])
            else:
                raise TypeError(f'Error dtype {dtype_} for torch tensor')
        else:
            raise TypeError(f"img must be one of (numpy.ndarray, torch.Tensor) but is {type(img).__name__}.")

class ToTensorEx(Transform):
    """
    Converts the input image to a tensor without applying any other transformations.
    """

    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Apply the transform to `img` and make it contiguous.
        """
        if torch.is_tensor(img):
            return img.contiguous()
        return torch.as_tensor(np.ascontiguousarray(img))