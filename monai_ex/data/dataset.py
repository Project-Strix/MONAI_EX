import warnings
from typing import Any, Callable, Optional, Sequence

from monai.data import Dataset
from monai.transforms import Randomizable
from monai.transforms.transform import apply_transform


class SplitDataset(Randomizable, Dataset):
    """SplitDataset.

    Split input datalist into two sub-datalist. Mainly designed for siamese dataset.
    Ex: input dataset=[0,1,2,3,4,5], the SplitDataset will output [(0,3), (1,4), (2,5)] if shuffle is False.
    """
    def __init__(
        self,
        data: Sequence,
        transform: Optional[Callable] = None,
        shuffle: bool = True
    ) -> None:
        """
        Args:
            data (Sequence): input data to load and transform to generate dataset for model.
            transform (Optional[Callable], optional): a callable data transform on input data. Defaults to None.
            shuffle (bool, optional): whether shuffle the data during the process. Defaults to True.
        """
        super().__init__(data, transform)
        self.shuffle = shuffle

    def __len__(self) -> int:
        return len(self.data)//2

    def randomize(self, data: Optional[Any] = None) -> None:
        try:
            self.R.shuffle(data)
        except TypeError as e:
            warnings.warn(f"input data can't be shuffled in PairDataset with numpy.random.shuffle(): {e}.")

    def __getitem__(self, index: int):
        if self.shuffle:
            self.randomize(self.data)

        data1 = self.data[index]
        data2 = self.data[index+self.__len__()]

        if self.transform is not None:
            data1 = apply_transform(self.transform, data1)
            data2 = apply_transform(self.transform, data2)

        return (data1, data2)
