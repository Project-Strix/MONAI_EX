import logging
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Union
from grpc import metadata_call_credentials

import torch
import numpy as np

from monai.config import DtypeLike, IgniteInfo
from monai.data import decollate_batch
from monai.transforms import SaveImage
from monai.utils import GridSampleMode, GridSamplePadMode, InterpolateMode, deprecated, min_version, optional_import

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")


class ImageBatchSaver:
    """
    Event handler triggered on completing every iteration to save the input image into files.
    It can extract the input image, its meta data(filename, affine, original_shape, etc.) and resample if necessary
    The name of saved file will be `{input_image_name}_{output_postfix}{output_ext}`,
    where the input image name is extracted from the meta data dictionary.
    If no meta data provided, use index from 0 as the filename prefix.
    """

    def __init__(
        self,
        output_dir: str = "./",
        output_postfix: str = "image",
        output_ext: str = ".nii.gz",
        resample: bool = True,
        mode: Union[GridSampleMode, InterpolateMode, str] = "nearest",
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        scale: Optional[int] = None,
        dtype: DtypeLike = np.float64,
        output_dtype: DtypeLike = np.float32,
        squeeze_end_dims: bool = True,
        data_root_dir: str = "",
        separate_folder: bool = True,
        image_batch_transform: Callable = lambda x: x,
        name: Optional[str] = None,
    ) -> None:
        """
        Args:
            output_dir: output image directory.
            output_postfix: a string appended to all output file names, default to `seg`.
            output_ext: output file extension name, available extensions: `.nii.gz`, `.nii`, `.png`.
            resample: whether to resample before saving the data array.
                if saving PNG format image, based on the `spatial_shape` from metadata.
                if saving NIfTI format image, based on the `original_affine` from metadata.
            mode: This option is used when ``resample = True``. Defaults to ``"nearest"``.

                - NIfTI files {``"bilinear"``, ``"nearest"``}
                    Interpolation mode to calculate output values.
                    See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                - PNG files {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
                    The interpolation mode.
                    See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate

            padding_mode: This option is used when ``resample = True``. Defaults to ``"border"``.

                - NIfTI files {``"zeros"``, ``"border"``, ``"reflection"``}
                    Padding mode for outside grid values.
                    See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                - PNG files
                    This option is ignored.

            scale: {``255``, ``65535``} postprocess data by clipping to [0, 1] and scaling
                [0, 255] (uint8) or [0, 65535] (uint16). Default is None to disable scaling.
                It's used for PNG format only.
            dtype: data type for resampling computation. Defaults to ``np.float64`` for best precision.
                If None, use the data type of input data.
                It's used for Nifti format only.
            output_dtype: data type for saving data. Defaults to ``np.float32``, it's used for Nifti format only.
            squeeze_end_dims: if True, any trailing singleton dimensions will be removed (after the channel
                has been moved to the end). So if input is (C,H,W,D), this will be altered to (H,W,D,C), and
                then if C==1, it will be saved as (H,W,D). If D also ==1, it will be saved as (H,W). If false,
                image will always be saved as (H,W,D,C).
                it's used for NIfTI format only.
            data_root_dir: if not empty, it specifies the beginning parts of the input file's
                absolute path. it's used to compute `input_file_rel_path`, the relative path to the file from
                `data_root_dir` to preserve folder structure when saving in case there are files in different
                folders with the same file names. for example:
                input_file_name: /foo/bar/test1/image.nii,
                output_postfix: seg
                output_ext: nii.gz
                output_dir: /output,
                data_root_dir: /foo/bar,
                output will be: /output/test1/image/image_seg.nii.gz
            separate_folder: whether to save every file in a separate folder, for example: if input filename is
                `image.nii`, postfix is `seg` and folder_path is `output`, if `True`, save as:
                `output/image/image_seg.nii`, if `False`, save as `output/image_seg.nii`. default to `True`.
            image_batch_transform: a callable that is used to extract the input images & meta_dict from `ignite.engine.state.batch`.
                The purpose is to extract necessary information from the meta data:
                    filename, affine, original_shape, etc.
                `engine.state` and `batch_transform` inherit from the ignite concept:
                https://pytorch.org/ignite/concepts.html#state, explanation and usage example are in the tutorial:
                https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
            name: identifier of logging.logger to use, defaulting to `engine.logger`.

        """
        self._saver = SaveImage(
            output_dir=output_dir,
            output_postfix=output_postfix,
            output_ext=output_ext,
            resample=resample,
            mode=mode,
            padding_mode=padding_mode,
            scale=scale,
            dtype=dtype,
            output_dtype=output_dtype,
            squeeze_end_dims=squeeze_end_dims,
            data_root_dir=data_root_dir,
            separate_folder=separate_folder,
        )
        self.batch_transform = image_batch_transform

        self.logger = logging.getLogger(name)
        self._name = name

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self._name is None:
            self.logger = engine.logger
        if not engine.has_event_handler(self, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self)

    def __call__(self, engine: Engine) -> None:
        """
        This method assumes self.batch_transform will extract metadata from the input batch.
        Output file datatype is determined from ``engine.state.output.dtype``.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        try:
            image_data, meta_data = self.batch_transform(engine.state.batch)
        except TypeError as e:
            self.logger.error(f"Expect two items (image&meta). Detailed error msg: {e}")

        if isinstance(meta_data, dict):
            # decollate the `dictionary of list` to `list of dictionaries`
            meta_data = decollate_batch(meta_data)
            if torch.is_tensor(image_data):
                image_data = decollate_batch(image_data)

        for meta, image in zip(meta_data, image_data):
            print("file path:", meta["filename_or_obj"])
            self._saver(image, meta)
        self.logger.info("Image has been saved into files.")
