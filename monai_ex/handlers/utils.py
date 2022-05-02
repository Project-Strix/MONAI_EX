import numpy as np 
from typing import Sequence, Union, Callable
from monai.config import KeysCollection, NdarrayTensor
from monai.utils import ensure_tuple, ensure_tuple_rep
from monai.transforms import Compose


def EnsureDim(data: NdarrayTensor):
    if data.ndim == 0:
        if isinstance(data, np.ndarray):
            return np.expand_dims(data, 0)
        else:
            return data.unsqueeze(0)
    return data


def from_engine_ex(
    keys: KeysCollection,
    transforms: Union[Callable, Sequence[Callable]] = lambda x: x,
    first: bool = False,
    ensure_dim: bool = False,
):
    keys = ensure_tuple(keys)
    transforms = ensure_tuple_rep(transforms, dim=len(keys))
    if ensure_dim:
        transforms = tuple(Compose([EnsureDim, tsf]) for tsf in transforms)

    def _wrapper(data):
        if isinstance(data, dict):
            if len(keys) == 1:
                return transforms[0](data[keys[0]])  # avaid generate one-length tuple (var,) 
            return tuple(t(data[k]) for k, t in zip(keys, transforms))
        if isinstance(data, list) and isinstance(data[0], dict):
            # if data is a list of dictionaries, extract expected keys and construct lists,
            # if `first=True`, only extract keys from the first item of the list
            ret = [
                t(data[0][k]) if first else [t(i[k]) for i in data]
                for k, t in zip(keys, transforms)
            ]
            return tuple(ret) if len(ret) > 1 else ret[0]

    return _wrapper
