import logging
from typing import Callable, Optional, Union, Dict, Any, Sequence

import numpy as np
import torch
from scipy import ndimage as ndi

from monai.transforms.compose import Transform, Randomizable, RandomizableTransform
from monai.transforms import DataStats, SaveImage, CastToType
from monai.transforms.utils import generate_pos_neg_label_crop_centers, map_binary_to_indices, is_positive
from monai.config import NdarrayTensor, DtypeLike
from monai_ex.utils import convert_data_type_ex, bbox_ND, ensure_list


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

    def __call__(self, img: NdarrayTensor) -> torch.Tensor:
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
        img = super().__init__(img, prefix, data_type, data_shape, value_range, data_value, additional_info)

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
        self.select_labels = ensure_list(select_labels)
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
            data = np.where(np.in1d(img, self.select_label), True, False).reshape(img.shape)

        return np.any(data, axis=0, keepdims=True) if (merge_channels or self.merge_channels) else data


class RandSoftCopyPaste(RandomizableTransform):
    """
    Perform Soft Copy&Paste augmentation.
    Reference: `https://arxiv.org/ftp/arxiv/papers/2203/2203.10507.pdf`

    Args:
        k_erode (int | float): erosion iteration num. 
          Float value denotes the percentage from edge to center.
        k_dilate (int): dilation iteration num.
        alpha (float, optional): transparence ratio. Defaults to 0.8.
        prob (float, optional): Probability to perform this aug. Defaults to 0.1.
        mask_select_fn (Callable, optional): function to select expected foreground, default is to select values > 0.
        source_label_value (Optional[int], optional): source foregound value. Defaults to None.
        strict_paste (bool, optional): whether to strictly paste source mask inside of target mask region. Defaults to False.
        tolerance (int, optional): even in strict_paste mode, there is a tolerance to allow paste to the edge. Defaults to 10.
        log_name (Optional[str], optional): logger name. Defaults to None.
    """

    def __init__(
        self,
        k_erode: Union[int, float],
        k_dilate: int,
        alpha: float = 0.8,
        prob: float = 0.1,
        mask_select_fn: Callable = is_positive,
        source_label_value: Optional[int] = None,
        strict_paste: bool = False,
        tolerance: int = 100,
        shift_source_intensity: bool = False,
        log_name: Optional[str] = None,
    ) -> None:
        RandomizableTransform.__init__(self, prob)
        self.k_erode = k_erode
        self.k_dilate = k_dilate
        self.alpha = alpha
        self.mask_select_fn = mask_select_fn
        self.source_label_value = source_label_value
        self.strict_paste = strict_paste
        self.tolerance = tolerance
        self.shift_intensity = shift_source_intensity
        self.logger = logging.getLogger(log_name)

    def soften(self, src_mask):
        if src_mask.shape[0] > 1:
            if self.source_label_value is None:
                raise ValueError("Multi-channel label data need to specify label_idx")
            else:
                src_mask = src_mask[self.source_label_value, ...]
        elif src_mask.shape[0] == 1:
            if self.source_label_value is None:
                src_mask = (src_mask > 0).squeeze(0)
            else:
                src_mask = (src_mask == self.source_label_value).squeeze(0)

        def _minmax_norm(x):
            minValue, maxValue = np.min(x), np.max(x)
            return (x - minValue) / (maxValue - minValue)

        struct = ndi.generate_binary_structure(src_mask.ndim, src_mask.ndim - 1)
        if 0 < self.k_erode < 1:
            mask = _minmax_norm(ndi.distance_transform_edt(src_mask)) > self.k_erode
        else:
            mask = ndi.binary_erosion(src_mask, struct, iterations=self.k_erode).astype(src_mask.dtype)
        for j in range(self.k_dilate):
            mask_binary = np.where(mask > 0, 1, 0)
            mask_dilate = ndi.binary_dilation(mask_binary, struct, iterations=1).astype(mask.dtype)
            mask_alpha = mask_dilate * (self.alpha ** (j + 1))
            mask = (1 - mask) * mask_alpha + mask

        return mask

    def ensure_strict_center(self, softed_mask, target_mask):
        src_ranges = tuple(
            slice(self.boundingbox[2 * i], self.boundingbox[2 * i + 1]) for i in range(len(self.boundingbox) // 2)
        )
        src_slices = (slice(softed_mask.shape[0]), *src_ranges)
        shifted_src_mask = np.zeros_like(target_mask)
        for tar_slice in self.target_slices:
            shifted_src_mask[tar_slice] = softed_mask[src_slices]
            if np.count_nonzero(target_mask[shifted_src_mask] == 0) <= self.tolerance:  # full contain!
                return [tar_slice]  # only support sample num = 1
        return []

    def compute_target_position(self, src_mask, softed_mask, target_image, target_mask) -> None:
        n_ch = target_image.shape[0]
        self.boundingbox = bbox_ND(softed_mask[0, ...])
        bbox_size = tuple(
            self.boundingbox[2 * i + 1] - self.boundingbox[2 * i] for i in range(len(self.boundingbox) // 2)
        )
        selected_target_mask = self.mask_select_fn(target_mask)
        fg_indices_, bg_indices_ = map_binary_to_indices(selected_target_mask, None, None)
        centers = generate_pos_neg_label_crop_centers(
            bbox_size,
            10,  # pick 10 candidates
            1,
            softed_mask.shape[1:],
            fg_indices_,
            bg_indices_,
            self.R,
            False,
        )
        target_ranges = []
        for center in centers:
            target_ranges.append(
                tuple(slice(int(center - sz // 2), int(center - sz // 2 + sz)) for center, sz in zip(center, bbox_size))
            )
        self.target_slices = [(slice(n_ch), *ranges) for ranges in target_ranges]
        if self.strict_paste:
            self.logger.debug("Enter strict filtering mode.")
            self.target_slices = self.ensure_strict_center(
                src_mask, selected_target_mask
            )

    def paste(
        self,
        source_image: NdarrayTensor,
        origin_mask: NdarrayTensor,
        softed_mask: NdarrayTensor,
        target_image: NdarrayTensor,
        target_bg_mask: NdarrayTensor,
        randomize: True,
    ):
        n_ch = target_image.shape[0]
        if randomize:
            self.compute_target_position(origin_mask, softed_mask, target_image, target_bg_mask)
        src_ranges = tuple(
            slice(self.boundingbox[2 * i], self.boundingbox[2 * i + 1]) for i in range(len(self.boundingbox) // 2)
        )
        src_slices = (slice(n_ch), *src_ranges)

        if not self.target_slices:
            self.logger.debug("No position is found to paste strictly! Skip copy&paste")
            return None

        shifted_src_image = np.zeros_like(target_image)
        shifted_src_mask = np.zeros_like(target_image)

        offset = 0
        if self.shift_intensity:
            src_mean_intensity = np.mean(source_image[src_slices])
            tar_mean_intensity = np.mean(target_image[self.target_slices[0]])
            offset = tar_mean_intensity - src_mean_intensity

        # softed_image = source_image * softed_mask

        src_region = source_image[src_slices] + offset
        shifted_src_image[self.target_slices[0]] = src_region
        shifted_src_mask[self.target_slices[0]] = softed_mask[src_slices]
        shifted_src_image *= shifted_src_mask
        # shifted_src_image = np.clip(shifted_src_image, np.min(target_image), np.max(target_image))
        sythetic_image = shifted_src_image + (1 - shifted_src_mask) * target_image
        shifted_src_mask[self.target_slices[0]] = origin_mask[src_slices]
        return sythetic_image, shifted_src_mask

    def __call__(
        self,
        image: NdarrayTensor,
        fg_mask: Optional[NdarrayTensor],
        bg_mask: NdarrayTensor,
        source_image: NdarrayTensor,
        source_fg_mask: NdarrayTensor,
        softed_fg_mask: Optional[NdarrayTensor] = None,
        randomize: bool = True,
    ) -> NdarrayTensor:
        if randomize:
            self.randomize(None)

        if not self._do_transform:
            return image, fg_mask

        if self.strict_paste and np.count_nonzero(self.mask_select_fn(bg_mask)) < np.count_nonzero(source_fg_mask):
            self.logger.debug("Target mask area is smaller than source foreground area. Skip copy&paste")
            return image, fg_mask

        if softed_fg_mask is None:
            softed_fg_mask = self.soften(source_fg_mask)
            if np.count_nonzero(softed_fg_mask) == 0:
                self.logger.debug("Source foreground area is too small to be soften. Skip copy&paste")
                return image, fg_mask

            softed_fg_mask = softed_fg_mask[np.newaxis, ...]
            if source_image.shape[0] > 1:
                softed_fg_mask = np.repeat(softed_fg_mask, repeats=source_image.shape[0], axis=0)

        processed_data = self.paste(
            source_image=source_image,
            origin_mask=source_fg_mask,
            softed_mask=softed_fg_mask,
            target_image=image,
            target_bg_mask=bg_mask,
            randomize=randomize,
        )
        if processed_data is None:
            return image, fg_mask

        sythetic_image, shifted_src_mask = processed_data

        if fg_mask is None:
            shifted_src_mask[shifted_src_mask > 0] = self.source_label_value
            sythetic_mask = shifted_src_mask
        else:
            sythetic_mask = fg_mask.copy()
            sythetic_mask[shifted_src_mask > 0] = self.source_label_value

        return sythetic_image, sythetic_mask
