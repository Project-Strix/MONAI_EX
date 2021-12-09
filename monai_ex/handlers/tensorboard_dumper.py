import os
from pathlib import Path
import logging
from typing import TYPE_CHECKING, Optional, Union
from monai.utils import exact_version, optional_import

if TYPE_CHECKING:
    from ignite.engine import Engine, Events
else:
    Engine, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Engine")
    Events, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Events")

from medlp.utilities.utils import dump_tensorboard, plot_summary


class TensorboardDumper:
    """
    Dumper Tensorboard content to local plot and images.
    """

    def __init__(
        self,
        log_dir: Union[Path, str],
        epoch_level: bool = True,
        interval: int = 1,
        save_image: bool = False,
        logger_name: Optional[str] = None,
    ):
        if save_image:
            raise NotImplementedError("Save image not supported yet.")

        if not os.path.isdir(log_dir):
            raise FileNotFoundError(f"{log_dir} not exists!")

        self.log_dir = log_dir
        self.epoch_level = epoch_level
        self.interval = interval
        self.logger = logging.getLogger(logger_name)
        self.db_file = None

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self.epoch_level:
            engine.add_event_handler(Events.EPOCH_COMPLETED(every=self.interval), self)
        else:
            engine.add_event_handler(
                Events.ITERATION_COMPLETED(every=self.interval), self
            )

    def __call__(self, engine: Engine) -> None:
        if self.db_file is None:
            files = os.listdir(self.log_dir)
            if len(files) == 0:
                self.logger.warn(
                    f"No tensorboard db is found in the dir({self.log_dir})"
                )
                return
            elif len(files) > 1:
                self.logger.warn(
                    "Multiple tensorboard db files are found! Skip dumping."
                )
                return
            else:
                self.db_file = os.path.join(self.log_dir, files[0])

        summary = dump_tensorboard(self.db_file, dump_keys=None, save_image=False)
        summary = dict(
            filter(
                lambda x: "loss" not in x[0] and "Learning_rate" not in x[0],
                summary.items(),
            )
        )
        plot_summary(
            summary, output_fpath=os.path.join(self.log_dir, "metric_summary.png")
        )
