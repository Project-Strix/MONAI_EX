from typing import TYPE_CHECKING, Optional, Callable

from monai.utils import exact_version, optional_import

if TYPE_CHECKING:
    from ignite.engine import Engine, Events
else:
    Engine, _ = optional_import("ignite.engine", "0.4.7", exact_version, "Engine")
    Events, _ = optional_import("ignite.engine", "0.4.7", exact_version, "Events")


class LearningRateRecordHandler:
    def __init__(
        self,
        loss_transform: Callable,
        logger_name: Optional[str] = None,
    ) -> None:
        self.loss_transform = loss_transform
        self.logger_name = logger_name
        self.history = {"loss": [], "lr": []}
        self.best_loss = -float("inf")

    def attach(self, engine: Engine) -> None:
        """
        Register a set of Ignite Event-Handlers to a specified Ignite engine.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.

        """
        if self.logger_name is None:
            self.logger = engine.logger
        if not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)

    def iteration_completed(self, engine: Engine) -> None:
        loss = self.loss_transform(engine.state.output)
        if loss is not None:
            self.history["loss"].append(float(loss.detach().cpu().numpy()))
        # if all weights use same lr
        self.history["lr"].append(engine.optimizer.param_groups[0]['lr'])
