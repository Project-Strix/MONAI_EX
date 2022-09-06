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

import logging
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np
import torch
import pylab

from monai.utils import exact_version, optional_import
from monai.visualize import plot_2d_or_3d_image
from monai.handlers import TensorBoardImageHandler

if TYPE_CHECKING:
    from ignite.engine import Engine, Events
else:
    Events, _ = optional_import("ignite.engine", "0.4.7", exact_version, "Events")
    Engine, _ = optional_import("ignite.engine", "0.4.7", exact_version, "Engine")

DEFAULT_TAG = "Loss"

# def create_rgb_summary(label):
#     num_colors = label.shape[1]

#     cm = pylab.get_cmap('gist_rainbow')

#     new_label = np.zeros((label.shape[0], label.shape[1], label.shape[2], 3), dtype=np.float32)

#     for i in range(num_colors):
#         color = cm(1. * i / num_colors)  # color will now be an RGBA tuple
#         new_label[:, :, :, 0] += label[:, :, :, i] * color[0]
#         new_label[:, :, :, 1] += label[:, :, :, i] * color[1]
#         new_label[:, :, :, 2] += label[:, :, :, i] * color[2]

#     return new_label

# def add_3D_overlay_to_summary(
#     data: Union[torch.Tensor, np.ndarray],
#     mask: Union[torch.Tensor, np.ndarray],
#     writer,
#     index: int = 0,
#     tag: str = 'output',
#     centers=None
# ):
#     data_ = data[index].detach().cpu().numpy() if torch.is_tensor(data) else data[index]
#     mask_ = mask[index].detach().cpu().numpy() if torch.is_tensor(mask) else mask[index]

#     if mask_.shape[1] > 1:
#         # there are channels
#         mask_ = create_rgb_summary(mask_)
#         data_ = data_[..., np.newaxis]

#     else:
#         data_, mask_ = data_[..., np.newaxis], mask_[..., np.newaxis]

#     if centers is None:
#         center_x = np.argmax(np.sum(np.sum(np.sum(mask_, axis=3, keepdims=True), axis=2, keepdims=True), axis=1, keepdims=True), axis=0)
#         center_y = np.argmax(np.sum(np.sum(np.sum(mask_, axis=3, keepdims=True), axis=2, keepdims=True), axis=0, keepdims=True), axis=1)
#         center_z = np.argmax(np.sum(np.sum(np.sum(mask_, axis=3, keepdims=True), axis=1, keepdims=True), axis=0, keepdims=True), axis=2)
#     else:
#         center_x, center_y, center_z = centers

#     segmentation_overlay_x = \
#         np.squeeze(data_[center_x, :, :, :] + mask_[center_x, :, :, :])
#     segmentation_overlay_y = \
#         np.squeeze(data_[:, center_y, :, :] + mask_[:, center_y, :, :])
#     segmentation_overlay_z = \
#         np.squeeze(data_[:, :, center_z, :] + mask_[:, :, center_z, :])

#     if len(segmentation_overlay_x.shape) != 3:
#         segmentation_overlay_x, segmentation_overlay_y, segmentation_overlay_z = \
#             segmentation_overlay_x[..., np.newaxis], \
#             segmentation_overlay_y[..., np.newaxis], \
#             segmentation_overlay_z[..., np.newaxis]

#     writer.add_image(tag + '_x', segmentation_overlay_x)
#     writer.add_image(tag + '_y', segmentation_overlay_y)
#     writer.add_image(tag + '_z', segmentation_overlay_z)


class TensorBoardImageHandlerEx(TensorBoardImageHandler):
    def __init__(
        self,
        summary_writer=None,
        log_dir: str = "./runs",
        interval: int = 1,
        epoch_level: bool = True,
        batch_transform: Callable = lambda x: (None, None),
        output_transform: Callable = lambda x: None,
        global_iter_transform: Callable = lambda x: x,
        index: int = 0,
        max_channels: int = 1,
        frame_dim: int = -1,
        max_frames: int = 64,
        prefix_name: str = "",
        logger_name: Optional[str] = None,
    ):
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
            frame_dim=frame_dim,
            max_frames=max_frames,
        )
        self.prefix_name = prefix_name
        self.logger = logging.getLogger(logger_name)

    def __call__(self, engine: Engine):
        step = self.global_iter_transform(
            engine.state.epoch if self.epoch_level else engine.state.iteration
        )

        try:
            image_tensor = self.batch_transform(engine.state.batch)[0]
            label_tensor = self.batch_transform(engine.state.batch)[1]
        except KeyError as e:
            self.logger.warn(f"Cannot find specified data: {e}. Skip tensorboard logging.")
            return
        except Exception as e:
            raise ValueError(f"Error occurred in TensorBoardImageHandlerEx. {e}")

        if image_tensor is not None:
            show_images = image_tensor[self.index]
            if torch.is_tensor(show_images):
                show_images = show_images.detach().cpu().numpy()
            if show_images is not None:
                if not isinstance(show_images, (np.ndarray, torch.Tensor, list, tuple)):
                    raise TypeError(
                        "output_transform(engine.state.output)[0] must be None or one of "
                        f"(numpy.ndarray, torch.Tensor) but is {type(show_images).__name__}."
                    )
                plot_2d_or_3d_image(
                    # add batch dim and plot the first item
                    data=show_images[None],
                    step=step,
                    writer=self._writer,
                    index=0,
                    max_channels=self.max_channels,
                    frame_dim=self.frame_dim,
                    max_frames=self.max_frames,
                    tag=self.prefix_name + "/input_0",
                )

        if label_tensor is not None:
            show_labels = label_tensor[self.index]
            if isinstance(show_labels, torch.Tensor):
                show_labels = show_labels.detach().cpu().numpy()
            if show_labels is not None:
                if not isinstance(show_labels, np.ndarray):
                    raise TypeError(
                        "batch_transform(engine.state.batch)[1] must be None or one of "
                        f"(numpy.ndarray, torch.Tensor) but is {type(show_labels).__name__}."
                    )
                plot_2d_or_3d_image(
                    data=show_labels[None],
                    step=step,
                    writer=self._writer,
                    index=0,
                    max_channels=self.max_channels,
                    frame_dim=self.frame_dim,
                    max_frames=self.max_frames,
                    tag=self.prefix_name + "/input_1",
                )

        if self.output_transform(engine.state.output) is not None:
            show_outputs = self.output_transform(engine.state.output)[self.index]
            # ! tmp solution to handle multi-inputs
            if isinstance(show_outputs, (list, tuple)):
                show_outputs = show_outputs[0]

            if isinstance(show_outputs, torch.Tensor):
                show_outputs = show_outputs.detach().cpu().numpy()
            if show_outputs is not None:
                if not isinstance(show_outputs, np.ndarray):
                    raise TypeError(
                        "output_transform(engine.state.output) must be None or one of "
                        f"(numpy.ndarray, torch.Tensor) but is {type(show_outputs).__name__}."
                    )
                plot_2d_or_3d_image(
                    data=show_outputs[None],
                    step=step,
                    writer=self._writer,
                    index=0,
                    max_channels=self.max_channels,
                    frame_dim=self.frame_dim,
                    max_frames=self.max_frames,
                    tag=self.prefix_name + "/output",
                )

        self._writer.flush()


class TensorboardGraphHandler:
    """
    TensorboardGraph for visualize network architecture using tensorboard
    """

    def __init__(
        self,
        net,
        writer,
        batch_transform: Callable = lambda x: x,
        logger_name: Optional[str] = None,
    ) -> None:
        self.net = net
        self.writer = writer
        self.batch_transform = batch_transform
        self.logger_name = logger_name
        self.logger = logging.getLogger(logger_name)

    def attach(self, engine: Engine) -> None:
        if self.logger_name is None:
            self.logger = engine.logger

        engine.add_event_handler(Events.ITERATION_COMPLETED(once=1), self)

    def __call__(self, engine: Engine) -> None:
        input_images, input_labels = self.batch_transform(engine.state.batch, engine.state.device, engine.non_blocking)
        if input_images is not None:
            if isinstance(input_images, (tuple, list)):
                inputs = (input_images,)  #! temp solution to resolve `add_graph` compatiablity for multi-inputs
            elif torch.is_tensor(input_images):
                inputs = input_images[0:1, ...]
            
            try:
                self.writer.add_graph(self.net, inputs, False)
            except Exception as e:
                self.logger.error(
                    f"Error occurred when adding graph to tensorboard: {e}"
                )
        else:
            self.logger.warn("No inputs are found! Skip adding graph!")
