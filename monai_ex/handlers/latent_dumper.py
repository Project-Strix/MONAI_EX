import os
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Callable, List
import numpy as np
from utils_cw import check_dir

import torch

from monai.data import decollate_batch
from monai.utils import ImageMetaKey as Key
from monai.utils import exact_version, optional_import

if TYPE_CHECKING:
    from ignite.engine import Engine, Events
else:
    Engine, _ = optional_import("ignite.engine", "0.4.7", exact_version, "Engine")
    Events, _ = optional_import("ignite.engine", "0.4.7", exact_version, "Events")


class LatentCodeSaver:
    """
    Event handler triggered on completing every iteration/epoch to save the latent code to local
    """

    def __init__(
        self,
        output_dir: str = "./",
        filename: str = "latents",
        data_root_dir: str = "",
        overwrite: bool = True,
        batch_transform: Callable = lambda x: x,
        output_transform: Callable = lambda x: x,
        name: Optional[str] = None,
        save_to_np: bool = False,
        save_as_onefile: bool = True,
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

        """
        self.output_dir = Path(output_dir)
        self.filename = filename
        self.data_root_dir = str(data_root_dir)
        self.overwrite = overwrite
        self.batch_transform = batch_transform
        self.output_transform = output_transform
        self.save_to_np = save_to_np
        self.save_as_onefile = save_as_onefile

        self.logger = logging.getLogger(name)
        self._name = name
        self._outputs: List[torch.Tensor] = []
        self._filenames: List[str] = []

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
            if isinstance(o, (list, tuple)):  #! hot fix for decollate
                o = o[0].detach()
            self._outputs.append(o)

    def _finalize(self, engine: Engine) -> None:
        """
        All gather classification results from ranks and save to CSV file.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self.save_as_onefile:
            print(len(self._outputs), len(self._outputs[0]))
            outputs = torch.stack(self._outputs, dim=0)

            if self.save_to_np:
                np.savez(self.output_dir / self.filename, outputs.cpu().numpy())
            else:
                torch.save(self.output_dir / self.filename, outputs.cpu())
        else:
            for output, filename in zip(self._outputs, self._filenames):
                try:
                    filename = Path(filename).relative_to(self.data_root_dir)
                except ValueError:  # todo: need a better way
                    filename = self.filename

                output_path = check_dir(self.output_dir, filename, isFile=True)
                if self.save_to_np:
                    np.savez(output_path, output.cpu().numpy())
                else:
                    torch.save(output_path, output.cpu())
