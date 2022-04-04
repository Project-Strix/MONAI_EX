import logging
from typing import Callable, Optional, Union, Dict, Any

import numpy as np
import torch
from scipy import ndimage as ndi

from monai.transforms.compose import Transform
from monai.transforms import DataStats, SaveImage, CastToType
from monai.config import NdarrayTensor, DtypeLike
from monai_ex.utils import convert_data_type_ex, optional_import

# Need NNI package
skimage, has_skimage = optional_import("skimage", "0.12")
if has_skimage:
    from skimage import exposure


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

    def __call__(
        self, img: Any, dtype: Optional[Union[DtypeLike, torch.dtype]] = None
    ) -> Any:
        """
        Apply the transform to `img`, assuming `img` is a numpy array or PyTorch Tensor.

        Args:
            dtype: convert image to this data type, default is `self.dtype`.

        Raises:
            TypeError: When ``img`` type is not in ``Union[numpy.ndarray, torch.Tensor]``.

        """
        type_list = (torch.Tensor, np.ndarray, int, bool, float, list, tuple)
        if not isinstance(img, type_list):
            raise TypeError(
                f"img must be one of ({type_list}) but is {type(img).__name__}."
            )
        img_out, *_ = convert_data_type_ex(
            img, output_type=type(img), dtype=dtype or self.dtype
        )
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
            logger_handler,
        )
        self.save_data = save_data
        self.save_data_dir = save_data_dir
        if self.save_data and self.save_data_dir is None:
            raise ValueError("save_data_dir is not given while save_data is True")

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
            img, prefix, data_type, data_shape, value_range, data_value, additional_info
        )

        if self.save_data:
            saver = SaveImage(
                output_dir=self.save_data_dir,
                output_postfix=self.prefix,
                output_ext=".nii.gz",
                resample=False,
            )
            saver(img, meta_data)

        return img


class DataLabelling(Transform):
    def __init__(self) -> None:
        """
        Args:
            to_onehot: whether convert labelling data to onehot format.

        """
        # self.to_onehot = to_onehot

    def __call__(self, img: np.ndarray) -> np.ndarray:
        input_ndim = img.squeeze().ndim  # spatial ndim
        if input_ndim == 2:
            structure = ndi.generate_binary_structure(2, 1)
        elif input_ndim == 3:
            structure = ndi.generate_binary_structure(3, 1)
        else:
            raise ValueError("Currently only support 2D&3D data")

        channel_dim = None
        if input_ndim != img.ndim:
            channel_dim = img.shape.index(1)
            img = img.squeeze()

        ccs, num_features = ndi.label(img, structure=structure)

        if channel_dim is not None:
            return np.expand_dims(ccs, axis=channel_dim)

        return ccs


class Clahe(Transform):
    def __init__(self, kernel_size=None, clip_limit=0.01, nbins=256) -> None:
        self.kernel_size = kernel_size
        self.clip_limit = clip_limit
        self.nbins = nbins
        if not has_skimage:
            raise ImportError("Please install scikit-image to use CLAHE.")

    def __call__(self, img: np.ndarray) -> np.ndarray:
        input_ndim = img.squeeze().ndim  # spatial ndim
        assert input_ndim in [2, 3], "Currently only support 2D&3D data"

        channel_dim = None
        if input_ndim != img.ndim:
            channel_dim = img.shape.index(1)
            img = img.squeeze()

        filter_img = exposure.equalize_adapthist(
            img,
            kernel_size=self.kernel_size,
            clip_limit=self.clip_limit,
            nbins=self.nbins,
        )

        if channel_dim is not None:
            return np.expand_dims(filter_img, axis=channel_dim)
        else:
            return filter_img
