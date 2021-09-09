import logging
from typing import Callable, Optional, Union, Dict, Any

import numpy as np
import torch

from monai.transforms.compose import Transform
from monai.transforms import DataStats, SaveImage, CastToType
from monai.config import NdarrayTensor, DtypeLike
from monai_ex.utils import convert_data_type_ex


class CastToTypeEx(CastToType):
    """
    Cast the Numpy data to specified numpy data type, or cast the PyTorch Tensor to
    specified PyTorch data type.
    """

    def __init__(self, dtype=np.float32) -> None:
        """
        Args:
            dtype: convert image to this data type, default is `np.float32`.
        """
        self.dtype = dtype

    def __call__(self, img: Any, dtype: Optional[Union[DtypeLike, torch.dtype]] = None) -> Any:
        """
        Apply the transform to `img`, assuming `img` is a numpy array or PyTorch Tensor.

        Args:
            dtype: convert image to this data type, default is `self.dtype`.

        Raises:
            TypeError: When ``img`` type is not in ``Union[numpy.ndarray, torch.Tensor]``.

        """
        type_list = (torch.Tensor, np.ndarray, int, bool, float, list, tuple)
        if not isinstance(img, type_list):
            raise TypeError(f"img must be one of ({type_list}) but is {type(img).__name__}.")
        img_out, *_ = convert_data_type_ex(img, output_type=type(img), dtype=dtype or self.dtype)
        return img_out


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
