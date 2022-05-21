import logging
from typing import TYPE_CHECKING, Optional

from monai.utils import exact_version, optional_import

if TYPE_CHECKING:
    from ignite.engine import Engine, Events
else:
    Engine, _ = optional_import("ignite.engine", "0.4.7", exact_version, "Engine")
    Events, _ = optional_import("ignite.engine", "0.4.7", exact_version, "Events")


class FreezeNetHandler:
    """Event handler to freeze network at the beginning of epoch.
    """
    def __init__(
        self, network, freeze_mode, freeze_params, logger_name: Optional[str] = None
    ) -> None:
        self.net = network
        self.freeze_mode = freeze_mode
        self.freeze_params = freeze_params
        self.logger_name = logger_name
        self.logger = logging.getLogger(logger_name)

    def attach(self, engine: Engine) -> None:
        if self.logger_name is None:
            self.logger = engine.logger
        if not engine.has_event_handler(self, Events.EPOCH_STARTED):
            engine.add_event_handler(Events.EPOCH_STARTED, self)
    
    def __call__(self, engine: Engine) -> None:
        if self.freeze_mode == "until":
            if engine.state.epoch == 1:
                self.net.freeze(True)
                self.logger.warning(f"{self.net._get_name()} is froze!")
            elif engine.state.epoch == int(self.freeze_params):
                self.net.freeze(False)
                self.logger.warning(f"{self.net._get_name()} is unfroze!")
        elif self.freeze_mode == "auto":
            raise NotImplementedError
        elif self.freeze_mode == "full" and engine.state.epoch == 1:
            self.net.freeze(True)
            self.logger.warning(f"{self.net._get_name()} is froze!")
        elif self.freeze_mode == "subtask":
            if engine.state.epoch == 1:
                self.net.freeze(True, subtask=self.freeze_params[0])
                self.logger.warning(f"{self.net._get_name()}'s subtask{self.freeze_params[0]} is froze!")
            elif engine.state.epoch == self.freeze_params[1]:
                self.net.freeze(False, subtask=self.freeze_params[0])
                self.logger.warning(f"{self.net._get_name()}'s subtask{self.freeze_params[0]} is unfroze!")
        else:
            raise ValueError(f"Got unexpected freeze mode {self.freeze_mode}")
