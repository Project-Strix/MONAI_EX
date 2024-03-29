import logging
from typing import Callable, Optional, Union, Dict, Any, Sequence
from warnings import warn
import numpy as np
import torch
from scipy import ndimage as ndi
from monai.transforms.compose import Transform, Randomizable
from monai.transforms import DataStats, SaveImage, CastToType
from monai.config import NdarrayTensor, DtypeLike
from monai_ex.utils import convert_data_type_ex, ensure_tuple
from monai.utils import optional_import, min_version
from copy import deepcopy

skeletonize_3D, has_skeletonize_3D = optional_import(
    module="skimage.morphology._skeletonize",
    version="0.19.0",
    version_checker=min_version,
    name="skeletonize_3d"
)
skeletonize_2D, has_skeletonize_2D = optional_import(
    module="skimage.morphology._skeletonize",
    version="0.19.0",
    version_checker=min_version,
    name="skeletonize_2d"
)


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


class RandLabelToMask(Randomizable, Transform):
    """
    Convert labels to mask for other tasks. A typical usage is to convert segmentation labels
    to mask data to pre-process images and then feed the images into classification network.
    It can support single channel labels or One-Hot labels with specified `select_labels`.
    For example, users can select `label value = [2, 3]` to construct mask data, or select the
    second and the third channels of labels to construct mask data.
    The output mask data can be a multiple channels binary data or a single channel binary
    data that merges all the channels.

    Args:
        select_labels: labels to generate mask from. for 1 channel label, the `select_labels`
            is the expected label values, like: [1, 2, 3]. for One-Hot format label, the
            `select_labels` is the expected channel indices.
        merge_channels: whether to use `np.any()` to merge the result on channel dim. if yes,
            will return a single channel mask with binary data.

    """

    def __init__(  # pytype: disable=annotation-type-mismatch
        self,
        select_labels: Union[Sequence[int], int],
        merge_channels: bool = False,
    ) -> None:  # pytype: disable=annotation-type-mismatch
        self.select_labels = ensure_tuple(select_labels)
        self.merge_channels = merge_channels

    def randomize(self):
        self.select_label = self.R.choice(self.select_labels, 1)[0]

    def __call__(
        self,
        img: np.ndarray,
        select_label: Optional[Union[Sequence[int], int]] = None,
        merge_channels: bool = False,
    ) -> np.ndarray:
        """
        Args:
            select_labels: labels to generate mask from. for 1 channel label, the `select_labels`
                is the expected label values, like: [1, 2, 3]. for One-Hot format label, the
                `select_labels` is the expected channel indices.
            merge_channels: whether to use `np.any()` to merge the result on channel dim. if yes,
                will return a single channel mask with binary data.
        """
        if select_label is None:
            self.randomize()
        else:
            self.select_label = select_label

        if img.shape[0] > 1:
            data = img[[self.select_label]]
        else:
            data = np.where(np.in1d(img, self.select_label), True, False).reshape(
                img.shape
            )

        return (
            np.any(data, axis=0, keepdims=True)
            if (merge_channels or self.merge_channels)
            else data
        )


class ExtractCenterline(Transform):
    """Extract centerline of curvilinear structure."""
    def __init__(self, mode) -> None:
        super().__init__()
        self.mode = mode

    def _zeros_like(self, input):
        if isinstance(input, np.ndarray):
            return np.zeros_like(input)
        elif isinstance(input, torch.Tensor):
            return torch.zeros_like(input)
        else:
            raise ValueError(f'msk should be Ndarray or Tensor, but got {type(input)}')

    def __call__(self, msk):
        if self.mode == '2D':
            if has_skeletonize_2D:
                if len(msk.shape) == 2:
                    return skeletonize_2D(msk.squeeze())
                if len(msk.shape) == 3:
                    data = self._zeros_like(msk)
                    for i in range(msk.shape[0]):
                        data[i, ...] = skeletonize_2D(deepcopy(msk[i, ...]))
                    return data
            else:
                raise RuntimeError('Skimage.morphology.skeletonize_2d required.')
        elif self.mode == '3D':
            if has_skeletonize_3D:
                if len(msk.shape) == 3:
                    return skeletonize_3D(msk.squeeze())
                elif len(msk.shape) == 4:
                    data = self._zeros_like(msk)
                    for i in range(msk.shape[0]):
                        data[i, ...] = skeletonize_3D(deepcopy(msk[i, ...]))
                    return data
            else:
                raise RuntimeError('Skimage.morphology.skeletonize_3d required.')
