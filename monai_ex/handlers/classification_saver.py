import os
import logging
import torch
from typing import TYPE_CHECKING, Callable, Optional, Union, Dict
import numpy as np

from monai.data import CSVSaver, NiftiSaver
from monai.utils import exact_version, optional_import, ensure_tuple
from monai.handlers.classification_saver import ClassificationSaver

Events, _ = optional_import("ignite.engine", "0.4.7", exact_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", "0.4.7", exact_version, "Engine")


class CSVSaverEx(CSVSaver):
    def __init__(
        self,
        output_dir="./",
        filename="predictions.csv",
        overwrite=True,
        title: Optional[list] = None
    ):
        super().__init__(
            output_dir=output_dir,
            filename=filename,
            overwrite=overwrite
        )
        if title is not None:
            self._cache_dict[title[0]] = title[1:]

    def save(self, data: Union[torch.Tensor, np.ndarray], meta_data: Optional[Dict] = None) -> None:
        """Save data into the cache dictionary. The metadata should have the following key:
            - ``'filename_or_obj'`` -- save the data corresponding to file name or object.
        If meta_data is None, use the default index from 0 to save data instead.

        Args:
            data: target data content that save into cache.
            meta_data: the meta data information corresponding to the data.

        """
        save_key = meta_data["filename_or_obj"] if meta_data else str(self._data_index)
        self._data_index += 1
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()
        # assert isinstance(data, np.ndarray), f'Expect np.ndarray, but got {type(data)}'

        count = 0
        while save_key in self._cache_dict:
            count += 1
            save_key += f'_{count}'

        self._cache_dict[save_key] = data.astype(np.float32)

    def save_batch(
        self,
        batch_data: Union[torch.Tensor, np.ndarray],
        labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
        meta_data: Optional[Dict] = None
    ) -> None:
        """Save a batch of data into the cache dictionary.

        Args:
            batch_data: target batch data content that save into cache.
            labels: output ground-truth labels if exists.
            meta_data: every key-value in the meta_data is corresponding to 1 batch of data.

        """
        if labels is None:
            for i, data in enumerate(batch_data):  # save a batch of files
                self.save(data, {k: meta_data[k][i] for k in meta_data} if meta_data else None)
        else:
            for i, (data, label) in enumerate(zip(batch_data, labels)):
                if torch.is_tensor(data):
                    data = data.detach().cpu().numpy()
                if torch.is_tensor(label):
                    label = label.detach().cpu().numpy()

                self.save(
                    np.array((data, label)),
                    {k: meta_data[k][i] for k in meta_data} if meta_data else None
                )


class ClassificationSaverEx(ClassificationSaver):
    """
    Extension of MONAI's ClassificationSaver.
    Extended: save_labels.
    """

    def __init__(
        self,
        output_dir: str = "./",
        filename: str = "predictions.csv",
        overwrite: bool = True,
        save_labels: bool = False,
        batch_transform: Callable = lambda x: x,
        output_transform: Callable = lambda x: x,
        name: Optional[str] = None,
    ) -> None:
        """
        Args:
            output_dir: output CSV file directory.
            filename: name of the saved CSV file name.
            overwrite: whether to overwriting existing CSV file content. If we are not overwriting,
                then we check if the results have been previously saved, and load them to the prediction_dict.
            batch_transform: a callable that is used to transform the
                ignite.engine.batch into expected format to extract the meta_data dictionary.
            output_transform: a callable that is used to transform the
                ignite.engine.output into the form expected model prediction data.
                The first dimension of this transform's output will be treated as the
                batch dimension. Each item in the batch will be saved individually.
            name: identifier of logging.logger to use, defaulting to `engine.logger`.

        """
        super(ClassificationSaverEx, self).__init__(
            output_dir=output_dir,
            filename=filename,
            overwrite=overwrite,
            batch_transform=batch_transform,
            output_transform=output_transform,
            name=name,
        )
        self.save_labels = save_labels
        title = np.array(['Filename', 'Pred', 'Groudtruth']) if save_labels else None
        self.saver = CSVSaverEx(output_dir, filename, overwrite, title)

    def __call__(self, engine: Engine) -> None:
        """
        This method assumes self.batch_transform will extract metadata from the input batch.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self.save_labels:
            meta_data, labels = self.batch_transform(engine.state.batch)
        else:
            meta_data, labels = self.batch_transform(engine.state.batch), None

        engine_output = self.output_transform(engine.state.output)

        # if self.save_labels:
        #     raise NotImplementedError
        #     y = int(labels.detach().cpu().numpy()) if torch.is_tensor(labels) else int(y)
        #     o = int(engine_output.detach().cpu().numpy()) if torch.is_tensor(engine_output) else int(engine_output)
        #     if y > 0 and y != o:
        #         self.fn_saver.save_batch(image.detach().cpu().numpy(), meta_data)
        #     elif y == 0 and y != o:
        #         self.fp_saver.save_batch(image.detach().cpu().numpy(), meta_data)

        self.saver.save_batch(engine_output, labels, meta_data)
