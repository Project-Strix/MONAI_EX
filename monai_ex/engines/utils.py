from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch


class RcnnKeys:
    """
    A set of common keys for RCNN-style networks.
    """    

    IMAGE = "image"
    LABEL = "label"
    ROI_LABEL = "roi_label"
    ROI_BBOX = "roi_bbox"
    ROI_MASK = 'roi_mask'


def default_prepare_batch_ex(
    batchdata: Dict[str, torch.Tensor]
) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
    assert isinstance(batchdata, dict), "default prepare_batch expects dictionary input data."
    if RcnnKeys.ROI_BBOX in batchdata or RcnnKeys.ROI_LABEL in batchdata:
        return batchdata[RcnnKeys.IMAGE], batchdata[RcnnKeys.LABEL], \
               batchdata[RcnnKeys.ROI_BBOX], batchdata[RcnnKeys.ROI_LABEL]
    elif CommonKeys.LABEL in batchdata:
        return batchdata[CommonKeys.IMAGE], batchdata[CommonKeys.LABEL]
    elif GanKeys.REALS in batchdata:
        return batchdata[GanKeys.REALS]
    else:
        return batchdata[CommonKeys.IMAGE], None