import numpy as np

from monai.visualize import GradCAM


class GradCAMEx(GradCAM):
    """Extension of MONAI's GradCAM. Adapted to medlp.

    """
    def _upsample_and_post_process(self, acti_maps, x, spatial_size=None):
        # upsampling and postprocessing
        outputs = []
        for acti_map in acti_maps:
            if self.upsampler:
                img_spatial = spatial_size if spatial_size else x.shape[2:]
                acti_map = self.upsampler(img_spatial)(acti_map)
            if self.postprocessing:
                acti_map = self.postprocessing(acti_map)
            outputs.append(acti_map)

        return np.concatenate(outputs, axis=1)

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
        acti_maps = self.compute_map(x, class_idx=class_idx, retain_graph=retain_graph, layer_idx=layer_idx)
        return self._upsample_and_post_process(acti_maps, x, img_spatial_size)
