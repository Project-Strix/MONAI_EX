from typing import Dict, List, Optional, Sequence, Tuple, Union
from monai.engines.utils import CommonKeys, GanKeys
import torch

class CustomKeys:
    """
    A set of customizable keys.
    `IMAGE` is the input image data.
    `MASK` is the input mask data.
    `LABEL` is the training or evaluation label of segmentation or classification task.
    `PRED` is the prediction data of model output.
    `LOSS` is the loss value of current iteration.
    `INFO` is some useful information during training or evaluation, like loss value, etc.
    """
    IMAGE: str = "image"
    MASK: str = "mask"
    LABEL: str = "label"
    PRED: str = "pred"
    LOSS: str = "loss"
    INFO: str = "info"
    LATENT: str = "latent"
    ATT: str = "attention"
    FORWARD: str = "acti"
    BACKWARD: str = "grad"


def get_keys_list(keytype):
    key_names = keytype.__annotations__.keys()
    return [keytype.__dict__[k] for k in key_names]

def get_keys_dict(keytype):
    key_names = keytype.__annotations__.keys()
    return {k:keytype.__dict__[k] for k in key_names}

def default_prepare_batch_ex(
    batchdata: Dict[str, torch.Tensor]
) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
    assert isinstance(batchdata, dict), "default prepare_batch expects dictionary input data."
    if CommonKeys.LABEL in batchdata:
        return batchdata[CommonKeys.IMAGE], batchdata[CommonKeys.LABEL]
    elif GanKeys.REALS in batchdata:
        return batchdata[GanKeys.REALS]
    else:
        return batchdata[CommonKeys.IMAGE], None