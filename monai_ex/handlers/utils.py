from typing import Sequence, Union, Callable
from monai.config import KeysCollection
from monai.utils import ensure_tuple, ensure_tuple_rep


def from_engine_ex(
    keys: KeysCollection,
    transforms: Union[Callable, Sequence[Callable]] = lambda x: x,
    first: bool = False
):
    keys = ensure_tuple(keys)
    transforms = ensure_tuple_rep(transforms, dim=len(keys))

    def _wrapper(data):
        if isinstance(data, dict):
            return tuple(t(data[k]) for k, t in zip(keys, transforms))
        if isinstance(data, list) and isinstance(data[0], dict):
            # if data is a list of dictionaries, extract expected keys and construct lists,
            # if `first=True`, only extract keys from the first item of the list
            ret = [t(data[0][k]) if first else [t(i[k]) for i in data] for k, t in zip(keys, transforms)]
            return tuple(ret) if len(ret) > 1 else ret[0]

    return _wrapper
