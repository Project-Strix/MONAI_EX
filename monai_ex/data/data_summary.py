from typing import Optional

from monai.data.dataset import Dataset
from monai.data.dataset_summary import DatasetSummary
from monai.transforms import Compose


class DatasetSummaryEx(DatasetSummary):
    """Extension of MONAI's DatasetSummaryEx.
    Extented: `select_transforms` select top N transforms to execute.

    Args:
        dataset (Dataset): dataset from which to load the data.
        image_key (Optional[str], optional): key name of images. Defaults to "image".
        label_key (Optional[str], optional): key name of labels. Defaults to "label".
        meta_key_postfix (str, optional): use `{image_key}_{meta_key_postfix}` to fetch the meta data from dict,
            the meta data is a dictionary object. Defaults to "meta_dict".
        num_workers (int, optional): how many subprocesses to use for data loading.
            ``0`` means that the data will be loaded in the main process. Defaults to 0.
        select_transforms (Optional[int], optional): select several transforms to be executed.
            ``0`` means no transforms were selected. Defaults to None.

    Raises:
        ValueError: if `select_transforms` > number of transforms.
        TypeError: Only accept `Compose` transform type.
    """

    def __init__(
        self,
        dataset: Dataset,
        image_key: Optional[str] = "image",
        label_key: Optional[str] = "label",
        meta_key_postfix: str = "meta_dict",
        num_workers: int = 0,
        select_transforms: Optional[int] = None,
        **kwargs,
    ):
        if select_transforms is not None:
            transform_fn = dataset.transform
            if isinstance(transform_fn, Compose):
                if len(transform_fn.transforms) < select_transforms:
                    raise ValueError(
                        f"Selected transforms num {select_transforms} "
                        f"> total transforms {len(transform_fn.transforms)}."
                    )
                elif select_transforms == 0:
                    dataset.transform = []
                else:
                    #? better to use type(transform_fn) instead of Compose?
                    dataset.transform = Compose(transform_fn.transforms[:select_transforms])
            else:
                raise TypeError(
                    f"Only handle 'Compose' transform, but got {type(transform_fn)}"
                )

        super().__init__(
            dataset=dataset,
            image_key=image_key,
            label_key=label_key,
            meta_key_postfix=meta_key_postfix,
            num_workers=num_workers,
            **kwargs,
        )
