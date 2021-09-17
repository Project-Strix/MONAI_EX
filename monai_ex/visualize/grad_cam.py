from monai.visualize import GradCAM
import torch
import numpy as np

class GradCAMEx(GradCAM):
    """Extension of MONAI's GradCAM. Adapted to medlp.

    Args:
        GradCAM ([type]): [description]
    """
    def _upsample_and_post_process(self, acti_map, x, spatial_size=None):
        # upsampling and postprocessing
        if self.upsampler:
            img_spatial = spatial_size if spatial_size else x.shape[2:]
            acti_map = self.upsampler(img_spatial)(acti_map)
        if self.postprocessing:
            acti_map = self.postprocessing(acti_map)
        return acti_map

    def __call__(self, x, class_idx=None, layer_idx=-1, retain_graph=False, img_spatial_size=None):
        """
        Compute the activation map with upsampling and postprocessing.

        Args:
            x: input tensor, shape must be compatible with `nn_module`.
            class_idx: index of the class to be visualized. Default to argmax(logits)
            layer_idx: index of the target layer if there are multiple target layers. Defaults to -1.
            retain_graph: whether to retain_graph for torch module backward call.

        Returns:
            activation maps
        """
        acti_map = self.compute_map(x, class_idx=class_idx, retain_graph=retain_graph, layer_idx=layer_idx)
        return self._upsample_and_post_process(acti_map, x, img_spatial_size)
