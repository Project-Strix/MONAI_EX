from typing import TYPE_CHECKING, Callable, Optional

from monai.utils import ensure_tuple, exact_version, optional_import
from monai.handlers import LrScheduleHandler

Events, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Engine")


class LrScheduleTensorboardHandler(LrScheduleHandler):
    """
    Ignite handler to update the Learning Rate based on PyTorch LR scheduler.
    """

    def __init__(
        self,
        lr_scheduler,
        summary_writer,
        print_lr: bool = True,
        name: Optional[str] = None,
        epoch_level: bool = True,
        step_transform: Callable = lambda engine: (),
    ):
        super().__init__(
            lr_scheduler = lr_scheduler,
            print_lr = print_lr,
            name = name,
            epoch_level = epoch_level,
            step_transform = step_transform
        )
        self.writer = summary_writer

    def __call__(self, engine):
        try:
            args = ensure_tuple(self.step_transform(engine))
        except KeyError as e:
            self.logger.warn('Cannot get specified key from the step_transform. Skip lr scheduler.')
        else:
            self.lr_scheduler.step(*args)

            if self.print_lr:
                self.logger.info(f"Current learning rate: {self.lr_scheduler._last_lr[0]}")
            if self.writer is not None:
                self.writer.add_scalar("Learning_rate", self.lr_scheduler._last_lr[0], engine.state.epoch)


