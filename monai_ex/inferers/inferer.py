from typing import Sequence, Union, Optional

import torch

from monai.inferers import Inferer
from monai.utils import BlendMode
from monai_ex.inferers.utils import sliding_window_classification


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
