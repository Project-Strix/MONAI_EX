from typing import Callable

import torch.nn as nn
import numpy as np

from monai.visualize import GradCAM, default_upsampler, default_normalizer
from monai.utils import ensure_tuple


class GradCAMEx(GradCAM):
    """Extension of MONAI's GradCAM. Adapted to strix."""

    def __init__(
        self,
        nn_module: nn.Module,
        target_layers: str,
        upsampler: Callable = default_upsampler,
        postprocessing: Callable = default_normalizer,
        register_backward: bool = True,
        hierarchical: bool = False,
    ) -> None:
        super().__init__(
            nn_module=nn_module,
            target_layers=target_layers,
            upsampler=upsampler,
            postprocessing=postprocessing,
            register_backward=register_backward,
        )
        self.hierarchical = hierarchical

    def _upsample_and_post_process(self, acti_maps, x, spatial_size=None):
        # upsampling and postprocessing
        outputs = []
        acti_maps = ensure_tuple(acti_maps)
        for acti_map in acti_maps:
            if self.upsampler:
                img_spatial = spatial_size if spatial_size else x.shape[2:]
                acti_map = self.upsampler(img_spatial)(acti_map)
            if self.postprocessing:
                acti_map = self.postprocessing(acti_map)
            outputs.append(acti_map)

        return np.concatenate(outputs, axis=1)

    def __call__(self, x, class_idx=None, retain_graph=False, img_spatial_size=None):
        """
        Compute the activation map with upsampling and postprocessing.

        Args:
            x: input tensor, shape must be compatible with `nn_module`.
            class_idx: index of the class to be visualized. Default to argmax(logits)
            retain_graph: whether to retain_graph for torch module backward call.

        Returns:
            activation maps
        """
        acti_map = self.compute_map(
            x,
            class_idx=class_idx,
            retain_graph=retain_graph,
            # spatial_size=img_spatial_size
        )
        return self._upsample_and_post_process(acti_map, x, img_spatial_size)
