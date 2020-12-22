from typing import Any
from monai.utils.misc import issequenceiterable


def ensure_list(vals: Any):
    """
    Returns a list of `vals`.
    """
    if not issequenceiterable(vals):
        vals = [vals,]

    return list(vals)

