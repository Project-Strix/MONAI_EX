import warnings
from typing import cast

import torch


def predict_segmentation(
    logits: torch.Tensor, mutually_exclusive: bool = False, threshold: float = 0.0
) -> torch.Tensor:
    """
    Given the logits from a network, computing the segmentation by thresholding all values above 0
    if multi-labels task, computing the `argmax` along the channel axis if multi-classes task,
    logits has shape `BCHW[D]`.

    Args:
        logits: raw data of model output.
        mutually_exclusive: if True, `logits` will be converted into a binary matrix using
            a combination of argmax, which is suitable for multi-classes task. Defaults to False.
        threshold: thresholding the prediction values if multi-labels task.
    """
    if not mutually_exclusive:
        return (cast(torch.Tensor, logits >= threshold)).int()
    else:
        if logits.shape[1] == 1:
            warnings.warn("single channel prediction, `mutually_exclusive=True` ignored, use threshold instead.")
            return (cast(torch.Tensor, logits >= threshold)).int()
        return logits.argmax(1, keepdim=False)
