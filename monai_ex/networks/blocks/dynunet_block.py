from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import (
    UnetResBlock, 
    UnetBasicBlock, 
    UnetResBlock, 
    UnetOutBlock, 
    UnetUpBlock,
    get_padding,
    get_output_padding
)
from monai.networks.layers import get_act_layer, get_norm_layer
from monai_ex.networks.blocks.convolutions import ConvolutionEx as Convolution
from monai_ex.networks.layers import Act, Norm, Dropout, Norm, split_args


class UnetResBlockEx(UnetResBlock):
    """
    A skip-connection based module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: [``"batch"``, ``"instance"``, ``"group"``]
            feature normalization type and arguments. In this module, if using ``"group"``,
            `in_channels` should be divisible by 16 (default value for ``num_groups``).
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        num_groups: int,
        norm_name: str,
        is_prunable: bool = False
    ):
        super(UnetResBlockEx, self).__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm_name=norm_name,
        )
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            conv_only=True,
            num_groups=num_groups,
            is_prunable=is_prunable,
        )
        self.conv2 = get_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            conv_only=True,
            num_groups=num_groups,
            is_prunable=is_prunable,
        )
        self.conv3 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            conv_only=True,
            num_groups=num_groups,
            is_prunable=is_prunable,
        )
        self.lrelu = get_act_layer(("leakyrelu", {"inplace": True, "negative_slope": 0.01}))
        self.norm1 = get_norm_layer(spatial_dims, out_channels, norm_name)
        self.norm2 = get_norm_layer(spatial_dims, out_channels, norm_name)
        self.norm3 = get_norm_layer(spatial_dims, out_channels, norm_name)
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True


class UnetBasicBlockEx(UnetBasicBlock):
    """
    A CNN module module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: [``"batch"``, ``"instance"``, ``"group"``]
            feature normalization type and arguments. In this module, if using ``"group"``,
            `in_channels` should be divisible by 16 (default value for ``num_groups``).
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        num_groups: int,
        norm_name: str,
        is_prunable: bool = False,
    ):
        super(UnetBasicBlockEx, self).__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm_name=norm_name
        )
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            conv_only=True,
            num_groups=num_groups,
            is_prunable=is_prunable,
        )
        self.conv2 = get_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            conv_only=True,
            num_groups=num_groups,
            is_prunable=is_prunable,
        )
        self.lrelu = get_act_layer(("leakyrelu", {"inplace": True, "negative_slope": 0.01}))
        self.norm1 = get_norm_layer(spatial_dims, out_channels, norm_name)
        self.norm2 = get_norm_layer(spatial_dims, out_channels, norm_name)



class UnetUpBlockEx(UnetUpBlock):
    """
    An upsampling module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        upsample_kernel_size: convolution kernel size for transposed convolution layers.
        norm_name: [``"batch"``, ``"instance"``, ``"group"``]
            feature normalization type and arguments. In this module, if using ``"group"``,
            `in_channels` should be divisible by 16 (default value for ``num_groups``).
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        num_groups: int,
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: str,
        is_prunable: bool = False,
    ):
        super(UnetUpBlockEx, self).__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            upsample_kernel_size=upsample_kernel_size,
            norm_name=norm_name
        )
        self.num_groups = num_groups
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            num_groups=num_groups,
            is_transposed=True,
            is_prunable=is_prunable,
        )
        self.conv_block = UnetBasicBlockEx(
            spatial_dims,
            out_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            num_groups=num_groups,
            norm_name=norm_name,
            is_prunable=is_prunable,
        )
    
    def groupwise_concate(self, f1, f2):
        output      = torch.cat((f1.unsqueeze(2), f2.unsqueeze(2)), dim = 2)
        output      = output.reshape([f1.shape[0],-1]+f1.shape[2:])
        return output 

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        if self.num_groups == 1:
            out = torch.cat((out, skip), dim=1)
        else: #num_groups > 1
            out = self.groupwise_concate(out, skip)

        out = self.conv_block(out)
        return out


class UnetOutBlockEx(UnetOutBlock):
    def __init__(
        self, 
        spatial_dims: int, 
        in_channels: int, 
        out_channels: int,
        num_groups: int,
        **kwargs
    ):
        super(UnetOutBlockEx, self).__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels
        )
        kernel_size = kwargs.get('kernel_size', 1)
        stride = kwargs.get('stride',1)
        bias = kwargs.get('bias',True)
        conv_only = kwargs.get('conv_only',True)
        activation = kwargs.get('activation', None)
        is_prunable = kwargs.get('is_prunable', False)

        self.conv = get_conv_layer(
            spatial_dims, 
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            bias=bias, 
            conv_only=conv_only,
            num_groups=num_groups,
            is_prunable=is_prunable,
        )
        if activation is not None:
            act_name, act_args = split_args(activation)
            act_type = Act[act_name]
            self.conv.add_module("act", act_type(**act_args))

    def forward(self, inp):
        out = self.conv(inp)
        return out

def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Sequence[int], int] = 3,
    stride: Union[Sequence[int], int] = 1,
    act: Optional[Union[Tuple, str]] = Act.PRELU,
    norm: Union[Tuple, str] = Norm.INSTANCE,
    bias: bool = False,
    conv_only: bool = True,
    is_transposed: bool = False,
    num_groups: int= 1,
    is_prunable: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        groups=num_groups,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        is_prunable=is_prunable,
        padding=padding,
        output_padding=output_padding,
    )
