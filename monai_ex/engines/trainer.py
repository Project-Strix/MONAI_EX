from logging import raiseExceptions
from typing import TYPE_CHECKING, Callable, Tuple, Dict, Optional, Sequence
import warnings

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from monai.engines.trainer import Trainer, SupervisedTrainer
from monai.inferers import Inferer, SimpleInferer
# from monai.engines.utils import CommonKeys as Keys
from monai_ex.engines.utils import default_prepare_batch_ex
from monai_ex.engines.utils import CustomKeys as Keys
from monai_ex.inferers import Inferer, UnifiedInferer
from monai.transforms import Transform
from monai.utils import exact_version, optional_import

if TYPE_CHECKING:
    from ignite.engine import Engine
    from ignite.metrics import Metric
else:
    Engine, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Engine")
    Metric, _ = optional_import("ignite.metrics", "0.4.2", exact_version, "Metric")


class SiameseTrainer(SupervisedTrainer):
    """SiameseTrainer
    
    Supervised Trainer designed for SiameseNet.
    Input batchdata should contains two elements.
    """

    def __init__(
        self,
        device,
        max_epochs,
        train_data_loader,
        network,
        optimizer,
        loss_function,
        epoch_length=None,
        non_blocking=False,
        prepare_batch=default_prepare_batch_ex,
        iteration_update=None,
        inferer=None,
        post_transform=None,
        key_train_metric=None,
        additional_metrics=None,
        train_handlers=None,
        amp=False,
    ):
        super(SiameseTrainer, self).__init__(
            device,
            max_epochs,
            train_data_loader,
            network,
            optimizer,
            loss_function,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            inferer=inferer,
            post_transform=post_transform,
            key_train_metric=key_train_metric,
            additional_metrics=additional_metrics,
            train_handlers=train_handlers,
            amp=amp,
        )
        # self.network = network
        # self.optimizer = optimizer
        # self.loss_function = loss_function
        # self.inferer = SimpleInferer() if inferer is None else inferer

    def _iteration(self, engine: Engine, batchdata: Tuple[Dict[str, torch.Tensor]]):
        """
        Callback function for the Siamese Training processing logic of 1 iteration in Ignite Engine.
        Return below items in a dictionary:
            - IMAGE: two image Tensors for model input, already moved to device.
            - LABEL: siamese label Tensor corresponding to the image, already moved to device.
            - PRED: two prediction results of siamese model.
            - LOSS: loss value computed by loss function.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
            batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.

        Raises:
            ValueError: When ``batchdata`` is None.

        """
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")

        if len(batchdata) != 2:
            raise ValueError(f"len of batchdata should be 2, but got {len(batchdata)}")

        batch1 = self.prepare_batch(batchdata[0], engine.state.device, engine.non_blocking)
        batch2 = self.prepare_batch(batchdata[1], engine.state.device, engine.non_blocking)

        if len(batch1) == 2:
            inputs1, targets1 = batch1
            inputs2, targets2 = batch2
            args: Tuple = tuple()
            kwargs: Dict = dict()
        else:
            inputs1, targets1, args, kwargs = batch1
            inputs2, targets2, args, kwargs = batch2

        self.network.train()
        self.optimizer.zero_grad()
        if self.amp and self.scaler is not None:
            with torch.cuda.amp.autocast():
                output1 = self.inferer(inputs1, self.network, *args, **kwargs)
                output2 = self.inferer(inputs2, self.network, *args, **kwargs)

                if isinstance(output1, tuple) and len(output1)==2:  # 2 outputs
                    loss = self.loss_function(output1[0], output2[0], output1[1], output2[1], targets1, targets2)
                elif isinstance(output1, torch.Tensor):
                    loss = self.loss_function(output1, output2, targets1, targets2)
                else:
                    raise NotImplementedError(f'SiameseNet expected 1or2 outputs,'
                                              f'but got {type(output1)} with size of {len(output1)}')

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            output1 = self.inferer(inputs1, self.network, *args, **kwargs)
            output2 = self.inferer(inputs2, self.network, *args, **kwargs)
            if len(output1) == 1:
                loss = self.loss_function(output1, output2, targets1, targets2)
            elif len(output2) == 2:
                loss = self.loss_function(output1[0], output2[0], output1[1], output2[1], targets1, targets2)
            else:
                raise NotImplementedError(f'SiameseNet expected 1or2 outputs, but got {len(output1)}')

            loss.backward()
            self.optimizer.step()

        if isinstance(output1, tuple) and len(output1)==2:
            return {
                Keys.IMAGE: torch.cat((inputs1,inputs2), dim=0),
                Keys.LABEL: torch.cat((targets1,targets2), dim=0),
                Keys.LATENT: torch.cat((output1[0],output2[0]), dim=0),
                Keys.PRED: torch.cat((output1[1],output2[1]), dim=0),
                Keys.LOSS: loss.item()
            }
        else:
            return {
                Keys.IMAGE: torch.cat((inputs1, inputs2), dim=0),
                Keys.LABEL: torch.cat((targets1, targets2), dim=0),
                Keys.LATENT: torch.cat((output1, output2), dim=0),
                Keys.LOSS: loss.item()
            }


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

        return {
            Keys.IMAGE: images,
            Keys.LABEL: targets,
            Keys.PRED: predictions,
            Keys.LOSS: losses.item(),
        }
