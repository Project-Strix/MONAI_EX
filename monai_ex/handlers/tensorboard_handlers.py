# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

from monai.utils import exact_version, optional_import
from monai.visualize import plot_2d_or_3d_image
from monai.handlers import TensorBoardImageHandler

if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Engine")

DEFAULT_TAG = "Loss"


class TensorBoardImageHandlerEx(TensorBoardImageHandler):
    def __init__(        
        self,
        summary_writer = None,
        log_dir: str = "./runs",
        interval: int = 1,
        epoch_level: bool = True,
        batch_transform: Callable = lambda x: x,
        output_transform: Callable = lambda x: x,
        global_iter_transform: Callable = lambda x: x,
        index: int = 0,
        max_channels: int = 1,
        max_frames: int = 64,
        prefix_name: str = ''):
        super().__init__(
            summary_writer=summary_writer,
            log_dir=log_dir,
            interval=interval,
            epoch_level=epoch_level,
            batch_transform=batch_transform,
            output_transform=output_transform,
            global_iter_transform=global_iter_transform,
            index=index,
            max_channels=max_channels,
            max_frames=max_frames
        )
        self.prefix_name = prefix_name
    
    def __call__(self, engine: Engine):
        step = self.global_iter_transform(engine.state.epoch if self.epoch_level else engine.state.iteration)
        show_images = self.batch_transform(engine.state.batch)[0]
        if torch.is_tensor(show_images):
            show_images = show_images.detach().cpu().numpy()
        if show_images is not None:
            if not isinstance(show_images, np.ndarray):
                raise ValueError("output_transform(engine.state.output)[0] must be an ndarray or tensor.")
            plot_2d_or_3d_image(
                show_images, step, self._writer, self.index, self.max_channels, self.max_frames, self.prefix_name+"/input_0"
            )

        show_labels = self.batch_transform(engine.state.batch)[1]
        if torch.is_tensor(show_labels):
            show_labels = show_labels.detach().cpu().numpy()
        if show_labels is not None:
            if not isinstance(show_labels, np.ndarray):
                raise ValueError("batch_transform(engine.state.batch)[1] must be an ndarray or tensor.")
            plot_2d_or_3d_image(
                show_labels, step, self._writer, self.index, self.max_channels, self.max_frames, self.prefix_name+"/input_1"
            )

        show_outputs = self.output_transform(engine.state.output)
        if torch.is_tensor(show_outputs):
            show_outputs = show_outputs.detach().cpu().numpy()
        if show_outputs is not None:
            if not isinstance(show_outputs, np.ndarray):
                raise ValueError("output_transform(engine.state.output) must be an ndarray or tensor.")
            plot_2d_or_3d_image(
                show_outputs, step, self._writer, self.index, self.max_channels, self.max_frames, self.prefix_name+"/output"
            )

        self._writer.flush()
