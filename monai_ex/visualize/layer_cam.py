from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai_ex.visualize.grad_cam import GradCAMEx


class LayerCAM(GradCAMEx):
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
    position :math:`(x, y)`, and :math:`Y^{(c)}` is the model output score for class :math:`c` before softmax.
    """
    def get_hierarchical_map(self, last_map, weight, actimap):
        new_map = weight * actimap
        # last_map = self.upsampler(new_map.shape)(last_map)
        last_map = F.interpolate(last_map, size=new_map.shape[2:], mode="trilinear", align_corners=False)

        shape_ = last_map.shape
        min_value = last_map.view(shape_[0], -1).min(dim=1)\
            .values.view([shape_[0], ] + [1, ]*(len(shape_)-1))
        max_value = last_map.view(shape_[0], -1).max(dim=1)\
            .values.view([shape_[0], ] + [1, ]*(len(shape_)-1))
        norm_last_map = last_map.sub(min_value).div((max_value-min_value))

        return (new_map * norm_last_map).sum(1, keepdim=True)

    def compute_map(self, x, class_idx=None, retain_graph=False, spatial_size=None):
        """Computes the weight coefficients of the hooked activation maps"""
        _, acti, grad = self.nn_module(x, class_idx=class_idx, retain_graph=retain_graph)

        maps = []
        for i, layer_name in enumerate(self.nn_module.target_layers):
            acti_, grad_ = acti[i], grad[i]
            b, c, *spatial = grad_.shape
            weights = torch.relu(grad_)
            if self.hierarchical and len(maps) > 0:
                acti_map = self.get_hierarchical_map(maps[-1], weights, acti_)
            else:
                acti_map = (weights * acti_).sum(1, keepdim=True)
            # print('acti map shape:', acti_map.shape)
            maps.append(F.relu(acti_map))

        return maps
