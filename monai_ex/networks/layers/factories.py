from typing import Type, Union

from monai.networks.layers.factories import Conv
from monai_ex.networks.layers.prunable_conv import *


@Conv.factory_function("prunable_conv")
def prunableconv_factory(dim: int) -> Type[Union[PrunableConv1d, PrunableConv2d, PrunableConv3d]]:
    types = (PrunableConv1d, PrunableConv2d, PrunableConv3d)
    return types[dim - 1]

@Conv.factory_function("prunable_convtrans")
def prunableconvtrans_factory(dim: int) -> Type[Union[PrunableDeconv1d, PrunableDeconv2d, PrunableDeconv3d]]:
    types = (PrunableDeconv1d, PrunableDeconv2d, PrunableDeconv3d)
    return types[dim - 1]

