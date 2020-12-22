from typing import TYPE_CHECKING, Callable, Dict, Optional, Sequence

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from monai.engines.trainer import Trainer
from monai.engines.utils import CommonKeys as Keys
from monai_ex.engines.utils import default_prepare_batch_ex
from monai_ex.inferers import Inferer, UnifiedInferer
from monai.transforms import Transform
from monai.utils import exact_version, optional_import

if TYPE_CHECKING:
    from ignite.engine import Engine
    from ignite.metrics import Metric
else:
    Engine, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Engine")
    Metric, _ = optional_import("ignite.metrics", "0.4.2", exact_version, "Metric")


class RcnnTrainer(Trainer):
    def __init__(
        self,
        device: torch.device,
        max_epochs: int,
        train_data_loader: DataLoader,
        network: torch.nn.Module,
        optimizer: Optimizer,
        loss_functions: Sequence[Callable],
        prepare_batch: Callable = default_prepare_batch_ex,
        iteration_update: Optional[Callable] = None,
        inferer: Inferer = UnifiedInferer(),
        post_transform: Optional[Transform] = None,
        key_train_metric: Optional[Dict[str, Metric]] = None,
        additional_metrics: Optional[Dict[str, Metric]] = None,
        train_handlers: Optional[Sequence] = None,
        amp: bool = False,
    ) -> None:
        super().__init__(
            device=device,
            max_epochs=max_epochs,
            data_loader=train_data_loader,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            post_transform=post_transform,
            key_metric=key_train_metric,
            additional_metrics=additional_metrics,
            handlers=train_handlers,
            amp=amp,
        )

        self.network = network
        self.optimizer = optimizer
        self.loss_functions = loss_functions
        self.inferer = inferer

    def _iteration(self, engine: Engine, batchdata: Dict[str, torch.Tensor]):
        
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")

        images, targets = self.prepare_batch(batchdata)

        images = images.to(engine.state.device)
        targets = [target.to(engine.state.device) for target in targets]

        self.network.train()
        self.optimizer.zero_grad()
        if self.amp and self.scaler is not None:
            with torch.cuda.amp.autocast():
                predictions, loss_dict = self.inferer(images, targets, self.network)
                losses = sum(loss for loss in loss_dict.values())
            self.scaler.scale(losses).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            predictions, loss_dict = self.inferer(images, targets, self.network)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            self.optimizer.step()

        return {Keys.IMAGE: images, Keys.LABEL: targets, Keys.PRED: predictions, Keys.LOSS: losses.item()}