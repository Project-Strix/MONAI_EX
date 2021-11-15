from typing import TYPE_CHECKING, Optional, Callable
import logging
from monai_ex.utils import exact_version, optional_import

if TYPE_CHECKING:
    from ignite.engine import Engine, Events
else:
    Engine, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Engine")
    Events, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Events")

Torchviz, _ = optional_import("torchviz")


class TorchVisualizer:
    """
    TorchVisualizer for visualize network architecture using PyTorchViz.
    """

    def __init__(
        self,
        net,
        outfile_path: str,
        output_transform: Callable = lambda x: x,
        logger_name: Optional[str] = None,
    ) -> None:
        self.net = net
        assert net is not None, "Network model should be input"
        self.outfile_path = outfile_path
        self.logger_name = logger_name
        self.logger = logging.getLogger(logger_name)
        self.output_transform = output_transform

    def attach(self, engine: Engine) -> None:
        if self.logger_name is None:
            self.logger = engine.logger
        engine.add_event_handler(Events.STARTED, self)

    def __call__(self, engine: Engine) -> None:
        output = self.output_transform(engine.state.output)
        if output is not None:
            try:
                dot = Torchviz.make_dot(output, dict(self.net.named_parameters()))
                print(output)
                print()
            except:
                self.logger.error("Generate graph failded")
            else:
                try:
                    dot.render(self.outfile_path)
                except:
                    self.logger.error(
                        f"""Failded to save torchviz graph to {self.outfile_path},
                                    Please make sure you have installed graphviz properly!"""
                    )
