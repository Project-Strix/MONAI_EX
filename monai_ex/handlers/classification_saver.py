import warnings
import torch
from typing import TYPE_CHECKING, Callable, Optional, Union, Dict
import numpy as np

from monai.config import IgniteInfo
from monai.data import CSVSaver, decollate_batch
from monai.utils import (
    exact_version,
    min_version,
    optional_import,
    evenly_divisible_all_gather,
    string_list_all_gather,
    ImageMetaKey as Key,
)
from monai.handlers.classification_saver import ClassificationSaver

idist, _ = optional_import("ignite", IgniteInfo.OPT_IMPORT_VERSION, min_version, "distributed")
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
        title: Optional[list] = None,
    ):
        super().__init__(output_dir=output_dir, filename=filename, overwrite=overwrite)
        if title is not None:
            self._cache_dict[title[0]] = title[1:]

    def save(
        self, data: Union[torch.Tensor, np.ndarray], meta_data: Optional[Dict] = None
    ) -> None:
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
            save_key += f"_{count}"

        self._cache_dict[save_key] = data.astype(np.float32)

    def save_batch(
        self,
        batch_data: Union[torch.Tensor, np.ndarray],
        labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
        meta_data: Optional[Dict] = None,
    ) -> None:
        """Save a batch of data into the cache dictionary.

        Args:
            batch_data: target batch data content that save into cache.
            labels: output ground-truth labels if exists.
            meta_data: every key-value in the meta_data is corresponding to 1 batch of data.

        """
        if labels:
            for i, (data, label) in enumerate(zip(batch_data, labels)):
                if torch.is_tensor(data):
                    data = data.detach().cpu().numpy()
                if torch.is_tensor(label):
                    label = label.detach().cpu().numpy()
                # print('Type:', type(data), type(label), type((data, label)))
                self.save(
                    np.array((data, label), dtype=object),
                    {k: meta_data[k][i] for k in meta_data} if meta_data else None,
                )
        else:
            for i, data in enumerate(batch_data):  # save a batch of files
                self.save(
                    data, {k: meta_data[k][i] for k in meta_data} if meta_data else None
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
        title = np.array(["Filename", "Pred", "Groudtruth"]) if save_labels else None
        self.saver = CSVSaverEx(output_dir, filename, overwrite, title)

    def _started(self, engine: Engine) -> None:
        self._outputs = []
        self._filenames = []
        self._labels = []

    def __call__(self, engine: Engine) -> None:
        """
        This method assumes self.batch_transform will extract metadata from the input batch.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        meta_data = self.batch_transform(engine.state.batch)
        if isinstance(meta_data, (list, tuple)):
            if len(meta_data) == 2:
                if isinstance(meta_data[1], (list, tuple)):
                    self._labels += list(meta_data[1])
                else:
                    self._labels.append(meta_data[1])
            meta_data = meta_data[0]

        if isinstance(meta_data, dict):
            # decollate the `dictionary of list` to `list of dictionaries`
            meta_data = decollate_batch(meta_data)

        engine_output = self.output_transform(engine.state.output)
        for m, o in zip(meta_data, engine_output):
            if isinstance(m, (list, tuple)):
                self._filenames.append(f"{m[0].get(Key.FILENAME_OR_OBJ)}")
            else:
                self._filenames.append(f"{m.get(Key.FILENAME_OR_OBJ)}")

            if isinstance(o, torch.Tensor):
                o = o.detach()
            elif isinstance(o, (list, tuple)):
                if len(o) == 1:
                    o = o[0].detach()
                else:
                    raise ValueError(
                        f"Something wrong. Expect tensor for saving but got {type(o)}: {o}"
                    )
            self._outputs.append(o)

    def _finalize(self, engine: Engine) -> None:
        """
        All gather classification results from ranks and save to CSV file.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        ws = idist.get_world_size()
        if self.save_rank >= ws:
            raise ValueError(
                "target save rank is greater than the distributed group size."
            )

        outputs = torch.stack(self._outputs, dim=0)
        filenames = self._filenames
        if ws > 1:
            outputs = evenly_divisible_all_gather(outputs, concat=True)
            filenames = string_list_all_gather(filenames)

        if len(filenames) == 0:
            meta_dict = None
        else:
            if len(filenames) != len(outputs):
                warnings.warn(
                    f"filenames length: {len(filenames)} doesn't match outputs length: {len(outputs)}."
                )
            meta_dict = {Key.FILENAME_OR_OBJ: filenames}

        # save to CSV file only in the expected rank
        if idist.get_rank() == self.save_rank:
            # print('Output:', type(outputs), len(outputs), type(outputs[0]), len(outputs[0]))
            # print('Labels:', type(self._labels), len(self._labels), type(self._labels[0]), len(self._labels[0]))
            # print('Meta:', type(meta_dict[Key.FILENAME_OR_OBJ]), len(meta_dict[Key.FILENAME_OR_OBJ]))
            self.saver.save_batch(outputs, self._labels, meta_dict)
            self.saver.finalize()
