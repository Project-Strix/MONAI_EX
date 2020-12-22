import logging
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np

from monai.data import NiftiSaver, PNGSaver
from monai.utils import GridSampleMode, GridSamplePadMode, InterpolateMode, exact_version, optional_import
from monai.handlers import SegmentationSaver

Events, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Engine")


class SegmentationSaverEx(SegmentationSaver):
    """
    Event handler triggered on completing every iteration to save the segmentation predictions into files.
    """

    def __init__(
        self,
        output_dir: str = "./",
        output_postfix: str = "seg",
        output_ext: str = ".nii.gz",
        output_name_uplevel: int = 0,
        resample: bool = True,
        mode: Union[GridSampleMode, InterpolateMode, str] = "nearest",
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        scale: Optional[int] = None,
        dtype: Optional[np.dtype] = None,
        batch_transform: Callable = lambda x: x,
        output_transform: Callable = lambda x: x,
        name: Optional[str] = None,
    ) -> None:
        """
        Extension of MONAI's SegmentationSaver
        Extended: output_name_uplevel

        Args:
            output_dir: output image directory.
            output_postfix: a string appended to all output file names.
            output_ext: output file extension name.
            resample: whether to resample before saving the data array.
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
            dtype: convert the image data to save to this data type.
                If None, keep the original type of data. It's used for Nifti format only.
            batch_transform: a callable that is used to transform the
                ignite.engine.batch into expected format to extract the meta_data dictionary.
            output_transform: a callable that is used to transform the
                ignite.engine.output into the form expected image data.
                The first dimension of this transform's output will be treated as the
                batch dimension. Each item in the batch will be saved individually.
            name: identifier of logging.logger to use, defaulting to `engine.logger`.

        """
        self.saver: Union[NiftiSaver, PNGSaver]
        if output_ext in (".nii.gz", ".nii"):
            self.saver = NiftiSaver(
                output_dir=output_dir,
                output_postfix=output_postfix,
                output_ext=output_ext,
                output_name_uplevel=output_name_uplevel,
                resample=resample,
                mode=GridSampleMode(mode),
                padding_mode=padding_mode,
                dtype=dtype,
            )
        elif output_ext == ".png":
            self.saver = PNGSaver(
                output_dir=output_dir,
                output_postfix=output_postfix,
                output_ext=output_ext,
                resample=resample,
                mode=InterpolateMode(mode),
                scale=scale,
            )
        self.batch_transform = batch_transform
        self.output_transform = output_transform

        self.logger = logging.getLogger(name)
        self._name = name
