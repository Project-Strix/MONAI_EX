from typing import Any, List

import torch 

from monai.utils.misc import issequenceiterable
from functools import partial
from utils_cw import catch_exception
from .exceptions import GenericException


trycatch = partial(catch_exception, handled_exception_type=GenericException)


def ensure_same_dim(tensor1, tensor2):
    if not (isinstance(tensor1, torch.Tensor) and isinstance(tensor2, torch.Tensor)):
        raise TypeError(f"Only accept torch.Tensor type inputs, but got {type(tensor1)} and {type(tensor2)}")

    tensor1_dim, tensor2_dim = tensor1.dim(), tensor2.dim()

    if tensor1_dim > tensor2_dim:
        return tensor1.squeeze(), tensor2
    elif tensor1_dim < tensor2_dim:
        return tensor1, tensor2.squeeze()
    else:
        return tensor1, tensor2


def ensure_list(vals: Any):
    """
    Returns a list of `vals`.
    """
    if not issequenceiterable(vals) or isinstance(vals, dict):
        vals = [vals, ]

    return list(vals)


def ensure_list_rep(vals: Any, dim: int) -> List[Any]:
    """
    Returns a copy of `tup` with `dim` values by either shortened or duplicated input.

    Raises:
        ValueError: When ``tup`` is a sequence and ``tup`` length is not ``dim``.
    """
    if not issequenceiterable(vals):
        return [vals,] * dim
    elif len(vals) == dim:
        return list(vals)

    raise ValueError(f"Sequence must have length {dim}, got {len(vals)}.")


def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module


class Registry(dict):
    '''
    A helper class for managing registering modules, it extends a dictionary
    and provides a register functions.

    Eg. creeting a registry:
        some_registry = Registry({"default": default_module})

    There're two ways of registering new modules:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)
    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_modeul_nickname")
        def foo():
            ...

    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_modeul"]
    '''
    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module_name, module=None):
        # used as function call
        if module is not None:
            _register_generic(self, module_name, module)
            return

        # used as decorator
        def register_fn(fn):
            _register_generic(self, module_name, fn)
            return fn

        return register_fn