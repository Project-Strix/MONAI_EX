import logging
from typing import TYPE_CHECKING, Optional

from monai.utils import exact_version, optional_import
from monai_ex.networks.layers.snip import SNIP, apply_prune_mask

if TYPE_CHECKING:
    from ignite.engine import Engine, Events
    from ignite.metrics import Metric
    from ignite.handlers import Checkpoint
    from torch.utils.tensorboard import SummaryWriter
else:
    Engine, _ = optional_import("ignite.engine", "0.4.7", exact_version, "Engine")
    SummaryWriter, _ = optional_import("torch.utils.tensorboard", name="SummaryWriter")
    Events, _ = optional_import("ignite.engine", "0.4.7", exact_version, "Events")
    Metric, _ = optional_import("ignite.metrics", "0.4.7", exact_version, "Metric")
    Checkpoint, _ = optional_import(
        "ignite.handlers", "0.4.7", exact_version, "Checkpoint"
    )


class SNIP_prune_handler:
    def __init__(
        self,
        net,
        prepare_batch_fn,
        loss_fn,
        prune_percent,
        data_loader,
        device="cuda",
        snip_device="cpu",
        verbose=False,
        logger_name: Optional[str] = None,
    ) -> None:
        self.net = net
        self.prepare_batch_fn = prepare_batch_fn
        self.loss_fn = loss_fn
        self.prune_percent = prune_percent
        self.data_loader = data_loader
        self.device = device
        self.snip_device = snip_device
        self.verbose = verbose
        self.logger_name = logger_name
        self.logger = logging.getLogger(logger_name)

    def __call__(self, engine: Engine) -> None:
        self.logger.debug("-------------- In SNIP handler ---------------")
        keep_masks = SNIP(
            self.net,
            self.prepare_batch_fn,
            self.loss_fn,
            self.prune_percent,
            self.data_loader,
            self.snip_device,
            None,
        )
        net_ = apply_prune_mask(self.net, keep_masks, self.device, self.verbose)
        # self.net.load_state_dict(net_.state_dict())
