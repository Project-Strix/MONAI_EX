import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.visualize import CAMBase


class LayerCAM(CAMBase):
    """Implements a class activation map extractor as described in 
    `"LayerCAM: Exploring Hierarchical Class Activation
     Maps for Localization" <http://mmcheng.net/mftp/Papers/21TIP_LayerCAM.pdf>`_.
    The localization map is computed as follows:
    .. math::
        L^{(c)}_{Layer-CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)}(x, y) \\cdot A_k(x, y)\\Big)
    with the coefficient :math:`w_k^{(c)}(x, y)` being defined as:
    .. math::
        w_k^{(c)}(x, y) = ReLU\\Big(\\frac{\\partial Y^{(c)}}{\\partial A_k(i, j)}(x, y)\\Big)
    where :math:`A_k(x, y)` is the activation of node :math:`k` in the target layer of the model at
    position :math:`(x, y)`,
    and :math:`Y^{(c)}` is the model output score for class :math:`c` before softmax.
    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.cams import LayerCAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = LayerCAM(model, 'layer4')
        >>> scores = model(input_tensor)
        >>> cam(class_idx=100, scores=scores)
    Args:
        model: input model
        target_layer: name of the target layer
        input_shape: shape of the expected input tensor excluding the batch dimension
    """
    def compute_map(self, x, class_idx=None, retain_graph=False, layer_idx=-1):
        """Computes the weight coefficients of the hooked activation maps"""
        _, acti, grad = self.nn_module(x, class_idx=class_idx, retain_graph=retain_graph)
        acti, grad = acti[layer_idx], grad[layer_idx]
        b, c, *spatial = grad.shape
        weights = torch.relu(grad)
        acti_map = (weights * acti).sum(1, keepdim=True)
        return F.relu(acti_map)

    def __call__(self, x, class_idx=None, layer_idx=-1, retain_graph=False):
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
        return self._upsample_and_post_process(acti_map, x)
