from typing import Sequence, Union, Optional, Any, Callable

import torch

from monai.inferers import Inferer
from monai.utils import BlendMode, PytorchPadMode
from monai_ex.inferers.utils import sliding_window_classification, sliding_window_2d_inference_3d


class UnifiedInferer(Inferer):
    """
    UnifiedInferer is an inference method that run model forward().
    Different from SimpleInferer, UnifiedInferer return losses, which means model should contain loss_fn.

    """
    def __init__(self) -> None:
        Inferer.__init__(self)

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor, network: torch.nn.Module):
        """Unified callable function API of Inferers.

        Args:
            inputs: model input data for inference.
            target: model target data for loss computation.
            network: target model to execute inference.

        """
        return network(inputs, targets)


class SlidingWindowClassify(Inferer):
    """
    Sliding window inferer for classification task.

    Args:
        Inferer ([type]): [description]
    """
    def __init__(
        self,
        roi_size: Union[Sequence[int], int],
        sw_batch_size: int = 1,
        overlap: float = 0.25,
        mode: Union[BlendMode, str] = BlendMode.CONSTANT,
        output_internel_results: Optional[bool] = False
    ) -> None:
        Inferer.__init__(self)
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mode: BlendMode = BlendMode(mode)
        self.output_internel_results = output_internel_results

    def __call__(self, inputs: torch.Tensor, network: torch.nn.Module) -> torch.Tensor:
        return sliding_window_classification(inputs, self.roi_size, self.sw_batch_size, network, self.overlap, self.mode)


class SlidingWindowInferer2Dfor3D(Inferer):
    """
    2D Sliding window inferer used for 3D image data.

    Args:
        Inferer ([type]): [description]
    """
    def __init__(
        self,
        roi_size: Union[Sequence[int], int],
        sw_batch_size: int = 1,
        overlap: float = 0.25,
        mode: Union[BlendMode, str] = BlendMode.CONSTANT,
        sigma_scale: Union[Sequence[float], float] = 0.125,
        padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
        cval: float = 0.0,
        z_axis: Optional[int] = 2,
        sw_device: Union[torch.device, str, None] = None,
        device: Union[torch.device, str, None] = None,
    ) -> None:
        Inferer.__init__(self)
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mode: BlendMode = BlendMode(mode)
        self.sigma_scale = sigma_scale
        self.padding_mode = padding_mode
        self.cval = cval
        self.z_axis = z_axis
        self.sw_device = sw_device
        self.device = device

    def __call__(
        self,
        inputs: torch.Tensor,
        network: Callable[..., torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        return sliding_window_2d_inference_3d(
            inputs,
            self.roi_size,
            self.sw_batch_size,
            network,
            self.overlap,
            self.mode,
            self.sigma_scale,
            self.padding_mode,
            self.cval,
            self.sw_device,
            self.device,
            self.z_axis,
            *args,
            **kwargs,
        )
