from typing import Optional, Sequence, Union, Any

import numpy as np

from monai.transforms.transform import Transform
from monai.data.synthetic import create_test_image_2d, create_test_image_3d


class LoadTestData(Transform):
    def __init__(
        self,
        height: int,
        width: int,
        depth: Optional[int] = None,
        num_objs: int = 12,
        rad_max: int = 30,
        rad_min: int = 5,
        noise_max: float = 0.0,
        num_seg_classes: int = 5,
        channel_dim: Optional[int] = None,
        random_state: Optional[np.random.RandomState] = None,
    ):
        self.height = height
        self.width = width
        self.depth = depth
        self.num_objs = num_objs
        self.rad_max = rad_max
        self.rad_min = rad_min
        self.noise_max = noise_max
        self.num_seg_classes = num_seg_classes
        self.channel_dim = channel_dim
        self.random_state = random_state

    def __call__(self, data: Any):
        if self.depth:
            img, seg = create_test_image_3d(
                self.height,
                self.width,
                self.depth,
                self.num_objs,
                self.rad_max,
                self.rad_min,
                self.noise_max,
                self.num_seg_classes,
                self.channel_dim,
                self.random_state,
            )
        else:
            img, seg = create_test_image_2d(
                self.height,
                self.width,
                self.num_objs,
                self.rad_max,
                self.rad_min,
                self.noise_max,
                self.num_seg_classes,
                self.channel_dim,
                self.random_state,
            )

        return img, seg
