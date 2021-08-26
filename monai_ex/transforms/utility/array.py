import logging
from typing import Callable, Optional, Union, Dict

import numpy as np
import torch

from monai.transforms.compose import Transform
from monai.transforms import DataStats, SaveImage
from monai.config import NdarrayTensor

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


class DataStatsEx(DataStats):
    """Extension of MONAI's DataStats
    Extended: save_data, save_data_dir

    """
    def __init__(
        self,
        prefix: str = "Data",
        data_type: bool = True,
        data_shape: bool = True,
        value_range: bool = True,
        data_value: bool = False,
        additional_info: Optional[Callable] = None,
        save_data: bool = False,
        save_data_dir: str = None,
        logger_handler: Optional[logging.Handler] = None,
    ) -> None:
        super().__init__(
            prefix,
            data_type,
            data_shape,
            value_range,
            data_value,
            additional_info,
            logger_handler
        )
        self.save_data = save_data
        self.save_data_dir = save_data_dir
        if self.save_data and self.save_data_dir is None:
            raise ValueError('save_data_dir is not given while save_data is True')

    def __call__(
        self,
        img: NdarrayTensor,
        meta_data: Optional[Dict] = None,
        prefix: Optional[str] = None,
        data_type: Optional[bool] = None,
        data_shape: Optional[bool] = None,
        value_range: Optional[bool] = None,
        data_value: Optional[bool] = None,
        additional_info: Optional[Callable] = None,
    ) -> NdarrayTensor:
        img = super().__init__(
            img,
            prefix,
            data_type,
            data_shape,
            value_range,
            data_value,
            additional_info
        )

        if self.save_data:
            saver = SaveImage(
                output_dir=self.save_data_dir,
                output_postfix=self.prefix,
                output_ext='.nii.gz',
                resample=False,
            )
            saver(img, meta_data)

        return img
