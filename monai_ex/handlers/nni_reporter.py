import logging
from typing import TYPE_CHECKING, Optional

from monai.utils import exact_version, optional_import

if TYPE_CHECKING:
    from ignite.engine import Engine, Events
    import nni as NNi
else:
    Engine, _ = optional_import("ignite.engine", "0.4.7", exact_version, "Engine")
    Events, _ = optional_import("ignite.engine", "0.4.7", exact_version, "Events")
    NNi, _ = optional_import("nni")


class NNIReporterHandler:
    """
    NNIReporter

    Args:

    """

    def __init__(
        self,
        metric_name: str,
        max_epochs: int,
        logger_name: Optional[str] = None,
    ) -> None:
        self.metric_name = metric_name
        self.logger_name = logger_name
        self.max_epochs = max_epochs
        self.logger = logging.getLogger(logger_name)

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self.logger_name is None:
            self.logger = engine.logger
        engine.add_event_handler(
            Events.EPOCH_COMPLETED, self.report_intermediate_result
        )
        engine.add_event_handler(Events.COMPLETED, self.report_final_result)
        engine.add_event_handler(Events.TERMINATE, self.report_final_result)

    def report_intermediate_result(self, engine):
        self.logger.info(f"{engine.state.epoch} report intermediate")
        NNi.report_intermediate_result(engine.state.metrics[self.metric_name])

    def report_final_result(self, engine):
        if engine.state.epoch == self.max_epochs:
            self.logger.info(f"{engine.state.epoch} report final")
            NNi.report_final_result(engine.state.metrics[self.metric_name])
