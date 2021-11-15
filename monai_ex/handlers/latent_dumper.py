import os
from pathlib import Path
import logging
from typing import TYPE_CHECKING, Optional, Union, Callable, List

import torch

from monai.visualize.class_activation_maps import ModelWithHooks
from monai.data import decollate_batch
from monai.utils import ImageMetaKey as Key
from monai_ex.utils import exact_version, optional_import

if TYPE_CHECKING:
    from ignite.engine import Engine, Events
else:
    Engine, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Engine")
    Events, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Events")


class LatentCodeDumper:
    """
    Event handler triggered on completing every iteration/epoch to save the latent code to local
    """

    def __init__(
        self,
        net,
        target_layers,
        output_dir: str = "./",
        filename: str = "predictions.csv",
        overwrite: bool = True,
        batch_transform: Callable = lambda x: x,
        output_transform: Callable = lambda x: x,
        name: Optional[str] = None,
        saver: Optional[CSVSaver] = None,
    ) -> None:
        """
        Args:
            output_dir: if `saver=None`, output CSV file directory.
            filename: if `saver=None`, name of the saved CSV file name.
            overwrite: if `saver=None`, whether to overwriting existing file content, if True,
                will clear the file before saving. otherwise, will apend new content to the file.
            batch_transform: a callable that is used to extract the `meta_data` dictionary of
                the input images from `ignite.engine.state.batch`. the purpose is to get the input
                filenames from the `meta_data` and store with classification results together.
            output_transform: a callable that is used to extract the model prediction data from
                `ignite.engine.state.output`. the first dimension of its output will be treated as
                the batch dimension. each item in the batch will be saved individually.
            name: identifier of logging.logger to use, defaulting to `engine.logger`.
            save_rank: only the handler on specified rank will save to CSV file in multi-gpus validation,
                default to 0.
            saver: the saver instance to save classification results, if None, create a CSVSaver internally.
                the saver must provide `save_batch(batch_data, meta_data)` and `finalize()` APIs.

        """
        self.net = net
        self.target_layers = target_layers
        self.output_dir = output_dir
        self.filename = filename
        self.overwrite = overwrite
        self.batch_transform = batch_transform
        self.output_transform = output_transform
        self.saver = saver

        self.logger = logging.getLogger(name)
        self._name = name
        self._outputs: List[torch.Tensor] = []
        self._filenames: List[str] = []

        # Convert to model with hooks if necessary
        if not isinstance(self.net, ModelWithHooks):
            self.nn_module = ModelWithHooks(
                self.net,
                target_layers,
                register_forward=True,
                register_backward=False,
            )
        else:
            self.nn_module = self.net

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self._name is None:
            self.logger = engine.logger
        if not engine.has_event_handler(self._started, Events.EPOCH_STARTED):
            engine.add_event_handler(Events.EPOCH_STARTED, self._started)
        if not engine.has_event_handler(self, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self)
        if not engine.has_event_handler(self._finalize, Events.EPOCH_COMPLETED):
            engine.add_event_handler(Events.EPOCH_COMPLETED, self._finalize)

    def _started(self, engine: Engine) -> None:
        self._outputs = []
        self._filenames = []

    def __call__(self, engine: Engine) -> None:
        """
        This method assumes self.batch_transform will extract metadata from the input batch.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        meta_data = self.batch_transform(engine.state.batch)
        if isinstance(meta_data, dict):
            # decollate the `dictionary of list` to `list of dictionaries`
            meta_data = decollate_batch(meta_data)
        engine_output = self.output_transform(engine.state.output)
        for m, o in zip(meta_data, engine_output):
            self._filenames.append(f"{m.get(Key.FILENAME_OR_OBJ)}")
            if isinstance(o, torch.Tensor):
                o = o.detach()
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
            saver = self.saver or CSVSaver(
                self.output_dir, self.filename, self.overwrite
            )
            saver.save_batch(outputs, meta_dict)
            saver.finalize()
