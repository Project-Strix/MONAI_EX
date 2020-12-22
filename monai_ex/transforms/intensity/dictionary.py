"""
A collection of dictionary-based wrappers around the "vanilla" transforms for intensity adjustment
defined in :py:class:`monai.transforms.intensity.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""
from typing import Dict, Hashable, Mapping

import numpy as np

from monai.config import KeysCollection
from monai.transforms.compose import MapTransform
from monai.transforms.intensity.array import ScaleIntensityRange


class ScaleIntensityViaDicomd(MapTransform):
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
        clip: bool = False
    ) -> None:
        super().__init__(keys)
        self.win_center_key = win_center_key
        self.win_width_key = win_width_key
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
            assert self.win_center_key in d[key+'_meta_dict'], f"{key+'_meta_dict'} must contain key '{self.win_center_key}'"
            assert self.win_width_key in d[key+'_meta_dict'], f"{key+'_meta_dict'} must contain key '{self.win_width_key}'"
            win_center = float(d[key+'_meta_dict'][self.win_center_key])
            win_width  = float(d[key+'_meta_dict'][self.win_width_key])
            scaler = ScaleIntensityRange(win_center-win_width/2, win_center+win_width/2, self.b_min, self.b_max, self.clip)
            d[key] = scaler(d[key])
        return d


ScaleIntensityViaDicomD = ScaleIntensityViaDicomDict = ScaleIntensityViaDicomd