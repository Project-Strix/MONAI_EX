# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import TYPE_CHECKING, Dict, Optional

from monai.utils import exact_version, optional_import
from monai.handlers import CheckpointSaver

Events, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Events")
Checkpoint, _ = optional_import("ignite.handlers", "0.4.2", exact_version, "Checkpoint")
BaseSaveHandler, _ = optional_import("ignite.handlers.checkpoint", "0.4.2", exact_version, "BaseSaveHandler")

if TYPE_CHECKING:
    from ignite.engine import Engine
    from ignite.handlers import DiskSaver
else:
    Engine, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Engine")
    DiskSaver, _ = optional_import("ignite.handlers", "0.4.2", exact_version, "DiskSaver")


class CheckpointSaverEx(CheckpointSaver):
    """
    Extension verion of MONAI's CheckpointSaver .
    Extended: key_metric_mode, key_metric_save_after_epoch

    CheckpointSaver acts as an Ignite handler to save checkpoint data into files.
    It supports to save according to metrics result, epoch number, iteration number
    and last model or exception.

    Args:
        save_dir: the target directory to save the checkpoints.
        save_dict: source objects that save to the checkpoint. examples::

            {'network': net, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

        name: identifier of logging.logger to use, if None, defaulting to ``engine.logger``.
        file_prefix: prefix for the filenames to which objects will be saved.
        save_final: whether to save checkpoint or session at final iteration or exception.
            If checkpoints are to be saved when an exception is raised, put this handler before
            `StatsHandler` in the handler list, because the logic with Ignite can only trigger
            the first attached handler for `EXCEPTION_RAISED` event.
        final_filename: set a fixed filename to save the final model if `save_final=True`.
            If None, default to `checkpoint_final_iteration=N.pt`.
        save_key_metric: whether to save checkpoint or session when the value of key_metric is
            higher than all the previous values during training.keep 4 decimal places of metric,
            checkpoint name is: {file_prefix}_key_metric=0.XXXX.pth.
        key_metric_name: the name of key_metric in ignite metrics dictionary.
            If None, use `engine.state.key_metric` instead.
        key_metric_n_saved: save top N checkpoints or sessions, sorted by the value of key
            metric in descending order.
        key_metric_filename: set a fixed filename to set the best metric model, if not None,
            `key_metric_n_saved` should be 1 and only keep the best metric model.
        epoch_level: save checkpoint during training for every N epochs or every N iterations.
            `True` is epoch level, `False` is iteration level.
        save_interval: save checkpoint every N epochs, default is 0 to save no checkpoint.
        n_saved: save latest N checkpoints of epoch level or iteration level, 'None' is to save all.

    Note:
        CheckpointHandler can be used during training, validation or evaluation.
        example of saved files:

            - checkpoint_iteration=400.pt
            - checkpoint_iteration=800.pt
            - checkpoint_epoch=1.pt
            - checkpoint_final_iteration=1000.pt
            - checkpoint_key_metric=0.9387.pt

    """

    def __init__(
        self,
        save_dir: str,
        save_dict: Dict,
        name: Optional[str] = None,
        file_prefix: str = "",
        save_final: bool = False,
        final_filename: Optional[str] = None,
        save_key_metric: bool = False,
        key_metric_name: Optional[str] = None,
        key_metric_mode: str = 'max',
        key_metric_n_saved: int = 1,
        key_metric_filename: Optional[str] = None,
        key_metric_save_after_epoch: int = 0,
        epoch_level: bool = True,
        save_interval: int = 0,
        n_saved: Optional[int] = None,
    ) -> None:
        super(CheckpointSaverEx, self).__init__(
            save_dir=save_dir,
            save_dict=save_dict,
            name=name,
            file_prefix=file_prefix,
            save_final=save_final,
            final_filename=final_filename,
            save_key_metric=save_key_metric,
            key_metric_name=key_metric_name,
            key_metric_n_saved=key_metric_n_saved,
            key_metric_filename=key_metric_filename,
            epoch_level=epoch_level,
            save_interval=save_interval,
            n_saved=n_saved
        )
        self.key_metric_save_after_epoch = key_metric_save_after_epoch
        if self.key_metric_save_after_epoch > 0:
            self.save_dir = os.path.join(self.save_dir, f'Models_after_{key_metric_save_after_epoch}epoch')

        class _DiskSaver(DiskSaver):
            """
            Enhance the DiskSaver to support fixed filename.

            """

            def __init__(self, dirname: str, filename: Optional[str] = None):
                super().__init__(dirname=dirname, require_empty=False)
                self.filename = filename

            def __call__(self, checkpoint: Dict, filename: str, metadata: Optional[Dict] = None) -> None:
                if self.filename is not None:
                    filename = self.filename
                super().__call__(checkpoint=checkpoint, filename=filename, metadata=metadata)

            def remove(self, filename: str) -> None:
                if self.filename is not None:
                    filename = self.filename
                super().remove(filename=filename)

        if save_final:

            def _final_func(engine: Engine):
                return engine.state.iteration

            self._final_checkpoint = Checkpoint(
                to_save=self.save_dict,
                save_handler=_DiskSaver(dirname=self.save_dir, filename=final_filename),
                filename_prefix=file_prefix,
                score_function=_final_func,
                score_name="final_iteration",
            )

        if save_key_metric:

            def _score_func(engine: Engine, save_after_epoch=0):
                if isinstance(key_metric_name, str):
                    metric_name = key_metric_name
                elif hasattr(engine.state, "key_metric_name") and isinstance(engine.state.key_metric_name, str):
                    metric_name = engine.state.key_metric_name
                else:
                    raise ValueError(
                        f"Incompatible values: save_key_metric=True and key_metric_name={key_metric_name}."
                    )
                if key_metric_mode == 'max':
                    return round(engine.state.metrics[metric_name], 4)
                elif key_metric_mode == 'min':
                    return -round(engine.state.metrics[metric_name], 4)
                else:
                    raise ValueError("key_metric_mode must be 'max' or 'min'") 

            if key_metric_filename is not None and key_metric_n_saved > 1:
                raise ValueError("if using fixed filename to save the best metric model, we should only save 1 model.")

            self._key_metric_checkpoint = Checkpoint(
                to_save=self.save_dict,
                save_handler=_DiskSaver(dirname=self.save_dir, filename=key_metric_filename),
                filename_prefix=file_prefix,
                score_function=_score_func,
                score_name="key_metric",
                n_saved=key_metric_n_saved,
            )

        if save_interval > 0:

            def _interval_func(engine: Engine):
                return engine.state.epoch if self.epoch_level else engine.state.iteration

            self._interval_checkpoint = Checkpoint(
                to_save=self.save_dict,
                save_handler=_DiskSaver(dirname=self.save_dir),
                filename_prefix=file_prefix,
                score_function=_interval_func,
                score_name="epoch" if self.epoch_level else "iteration",
                n_saved=n_saved,
            )

    def metrics_completed(self, engine: Engine) -> None:
        """Callback to compare metrics and save models in train or validation when epoch completed.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        assert callable(self._key_metric_checkpoint), "Error: _key_metric_checkpoint function not specified."

        if engine.state.epoch > self.key_metric_save_after_epoch:
            self._key_metric_checkpoint(engine)
