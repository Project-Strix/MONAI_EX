"""
A collection of dictionary-based wrappers around the "vanilla" transforms for intensity adjustment
defined in :py:class:`monai.transforms.intensity.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""
import torch
from typing import Dict, Hashable, Mapping, Optional

import numpy as np

from monai.config import KeysCollection
from monai.transforms.compose import MapTransform, Randomizable
from monai.transforms.intensity.array import ScaleIntensityRange, MaskIntensity
from monai_ex.transforms.intensity.array import ClipIntensity, MedianFilter, Clahe


class ScaleIntensityByDicomInfod(MapTransform):
    """
    Apply DICOM win WL to scaling to the whole numpy array.
    Scaling from [win_center-win_width/2, win_center+win_width/2] to [b_min, b_max] with clip option.

    Args:
        win_center_key: win center key in meta_dict
        win_width_key: win width key in meta_dict
        b_min: intensity target range min.
        b_max: intensity target range max.
        clip: whether to perform clip after scaling.
    """

    def __init__(
        self,
        keys: KeysCollection,
        win_center_key: str,
        win_width_key: str,
        b_min: float = 0,
        b_max: float = 1,
        clip: bool = False,
    ) -> None:
        super().__init__(keys)
        self.win_center_key = win_center_key
        self.win_width_key = win_width_key
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
            assert (
                self.win_center_key in d[key + "_meta_dict"]
            ), f"{key+'_meta_dict'} must contain key '{self.win_center_key}'"
            assert (
                self.win_width_key in d[key + "_meta_dict"]
            ), f"{key+'_meta_dict'} must contain key '{self.win_width_key}'"
            win_center = float(d[key + "_meta_dict"][self.win_center_key])
            win_width = float(d[key + "_meta_dict"][self.win_width_key])
            scaler = ScaleIntensityRange(
                win_center - win_width / 2,
                win_center + win_width / 2,
                self.b_min,
                self.b_max,
                self.clip,
            )
            d[key] = scaler(d[key])
        return d


class MaskIntensityExd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.MaskIntensity`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        mask_data: if mask data is single channel, apply to evey channel
            of input image. if multiple channels, the channel number must
            match input data. mask_data will be converted to `bool` values
            by `mask_data > 0` before applying transform to input image.

    """

    def __init__(
        self, keys: KeysCollection, mask_key: KeysCollection, fill_mode="zero"
    ) -> None:
        super().__init__(keys)
        self.mask_key = mask_key
        self.converter = MaskIntensity(mask_data=np.array([]))

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        mask_data = d[self.mask_key]
        for key in self.keys:
            d[key] = self.converter(d[key], mask_data=mask_data)
        return d


class ClipIntensityd(MapTransform):
    """Dictionary-based wrapper of :py:class:`monai_ex.transforms.ScaleIntensityRange`.

    Args:
        MapTransform ([type]): [description]
    """

    def __init__(self, keys, cmin: float, cmax: float):
        super().__init__(keys)
        self.clipper = ClipIntensity(cmin, cmax)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.clipper(d[key])
        return d


class MedianFilterd(MapTransform):
    """Dictionary-based wrapper of :py:class:`monai_ex.transforms.MedianFilter`.

    Args:
        MapTransform ([type]): [description]
    """

    def __init__(self, keys, size, mode="reflect", cval=0.0, origin=0):
        super(MedianFilterd, self).__init__(keys)
        self.converter = MedianFilter(size, mode=mode, cval=cval, origin=origin)

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for _, key in enumerate(self.keys):
            d[key] = self.converter(d[key])
        return d


class RandLocalPixelShuffled(MapTransform, Randomizable):
    def __init__(self, keys: KeysCollection):
        raise NotImplementedError


class RandImageInpaintingd(MapTransform, Randomizable):
    def __init__(self, keys: KeysCollection):
        raise NotImplementedError


class RandImageOutpaintingd(MapTransform, Randomizable):
    def __init__(self, keys: KeysCollection):
        raise NotImplementedError


class RandNonlineard(MapTransform, Randomizable):
    def __init__(self, keys: KeysCollection):
        raise NotImplementedError


class Clahed(MapTransform):
    """Dictionary-based wrapper of :py:class:`monai_ex.transforms.Clahe`."""

    def __init__(
        self,
        keys: KeysCollection,
        kernel_size: Optional[int] = None,
        clip_limit: float = 0.01,
        nbins: int = 256,
    ) -> None:
        super().__init__(keys)
        self.converter = Clahe()
        self.kernel_size = kernel_size
        self.clip_limit = clip_limit
        self.nbins = nbins

    def __call__(
        self, img: Mapping[Hashable, torch.Tensor]
    ) -> Dict[Hashable, torch.Tensor]:
        d = dict(img)
        for idx, key in enumerate(self.keys):
            d[key] = self.converter(d[key])
        return d


ScaleIntensityByDicomInfoD = ScaleIntensityByDicomInfoDict = ScaleIntensityByDicomInfod
MaskIntensityExD = MaskIntensityExDict = MaskIntensityExd
RandLocalPixelShuffleD = RandLocalPixelShuffleDict = RandLocalPixelShuffled
RandImageInpaintingD = RandImageInpaintingDict = RandImageInpaintingd
RandImageOutpaintingD = RandImageOutpaintingDict = RandImageOutpaintingd
RandNonlinearD = RandNonlinearDict = RandNonlineard
ClipIntensityD = ClipIntensityDict = ClipIntensityd
MedianFilterD = MedianFilterDict = MedianFilterd
ClaheD = ClaheDict = Clahed
