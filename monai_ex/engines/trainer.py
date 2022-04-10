import warnings
from logging import raiseExceptions
from typing import TYPE_CHECKING, Callable, Tuple, Dict, Optional, Sequence, Union, Iterable, List

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from monai.engines.trainer import Trainer, SupervisedTrainer
from monai.engines.utils import IterationEvents
from monai.transforms import apply_transform
from monai_ex.engines.utils import (
    default_prepare_batch_ex,
    default_metric_cmp_fn,
    CustomKeys as Keys
)
from monai_ex.inferers import Inferer, UnifiedInferer
from monai.transforms import Transform
from monai.utils import exact_version, optional_import

if TYPE_CHECKING:
    from ignite.engine import Engine, Events, EventEnum
    from ignite.metrics import Metric
else:
    Events, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Events")
    Engine, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Engine")
    Metric, _ = optional_import("ignite.metrics", "0.4.4", exact_version, "Metric")
    EventEnum, _ = optional_import("ignite.engine", "0.4.4", exact_version, "EventEnum")


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
            post_transform=None,
            key_train_metric=key_train_metric,
            additional_metrics=additional_metrics,
            train_handlers=train_handlers,
            amp=amp,
        )
        if post_transform is not None:
            @self.on(Events.ITERATION_COMPLETED)
            def run_post_transform(engine: Engine) -> None:
                assert post_transform is not None
                engine.state.output = apply_transform(post_transform, engine.state.output)

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
                Keys.LABEL: torch.cat((targets1, targets2), dim=0),
                Keys.LATENT: torch.cat((output1[0], output2[0]), dim=0),
                Keys.PRED: torch.cat((output1[1], output2[1]), dim=0),
                Keys.LOSS: loss.item()
            }
        else:
            return {
                Keys.IMAGE: torch.cat((inputs1, inputs2), dim=0),
                Keys.LABEL: torch.cat((targets1, targets2), dim=0),
                Keys.LATENT: torch.cat((output1, output2), dim=0),
                Keys.LOSS: loss.item()
            }


class SupervisedTrainerEx(SupervisedTrainer):
    """Extension of MONAI's SupervisedTrainer.
    Extended: custom_keys.

    """
    def __init__(
        self,
        device: torch.device,
        max_epochs: int,
        train_data_loader: Union[Iterable, DataLoader],
        network: torch.nn.Module,
        optimizer: Optimizer,
        loss_function: Callable,
        epoch_length: Optional[int] = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch_ex,
        iteration_update: Optional[Callable] = None,
        inferer: Optional[Inferer] = None,
        post_transform: Optional[Transform] = None,
        key_train_metric: Optional[Dict[str, Metric]] = None,
        additional_metrics: Optional[Dict[str, Metric]] = None,
        train_handlers: Optional[Sequence] = None,
        amp: bool = False,
        event_names: Optional[List[Union[str, EventEnum]]] = None,
        event_to_attr: Optional[dict] = None,
        custom_keys: Optional[dict] = None,
    ) -> None:
        super().__init__(
            device,
            max_epochs,
            train_data_loader,
            network,
            optimizer,
            loss_function,
            epoch_length,
            non_blocking,
            prepare_batch,
            iteration_update,
            inferer,
            post_transform,
            key_train_metric,
            additional_metrics,
            train_handlers,
            amp,
            event_names,
            event_to_attr,
        )
        if custom_keys is None:
            self.keys = {"IMAGE": Keys.IMAGE, "LABEL": Keys.LABEL, "PRED": Keys.PRED, "LOSS": Keys.LOSS}
        else:
            self.keys = custom_keys

    def _iteration(self, engine: Engine, batchdata: Dict[str, torch.Tensor]):
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        batch = self.prepare_batch(batchdata, engine.state.device, engine.non_blocking)
        if len(batch) == 2:
            inputs, targets = batch
            args: Tuple = ()
            kwargs: Dict = {}
        else:
            inputs, targets, args, kwargs = batch
        # put iteration outputs into engine.state
        engine.state.output = {self.keys["IMAGE"]: inputs, self.keys["LABEL"]: targets}

        def _compute_pred_loss():
            engine.state.output[self.keys["PRED"]] = self.inferer(inputs, self.network, *args, **kwargs)
            engine.fire_event(IterationEvents.FORWARD_COMPLETED)
            if engine.state.output[self.keys["PRED"]].shape != targets.shape and \
               1 in engine.state.output[self.keys["PRED"]].shape:
                engine.state.output[self.keys["PRED"]].squeeze_()
            engine.state.output[self.keys["LOSS"]] = self.loss_function(engine.state.output[self.keys["PRED"]], targets).mean()
            engine.fire_event(IterationEvents.LOSS_COMPLETED)

        self.network.train()
        self.optimizer.zero_grad()
        if self.amp and self.scaler is not None:
            with torch.cuda.amp.autocast():
                _compute_pred_loss()
            self.scaler.scale(engine.state.output[self.keys["LOSS"]]).backward()
            engine.fire_event(IterationEvents.BACKWARD_COMPLETED)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            _compute_pred_loss()
            engine.state.output[self.keys["LOSS"]].backward()
            engine.fire_event(IterationEvents.BACKWARD_COMPLETED)
            self.optimizer.step()
        engine.fire_event(IterationEvents.MODEL_COMPLETED)

        return engine.state.output


class MultiTaskTrainer(Trainer):
    def __init__(
        self,
        device: torch.device,
        max_epochs: int,
        train_data_loader: Union[Iterable, DataLoader],
        network: torch.nn.Module,
        optimizer: Optimizer,
        loss_functions: List[Callable],
        epoch_length: Optional[int] = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch_ex,
        iteration_update: Optional[Callable] = None,
        inferer: Optional[Inferer] = None,
        postprocessing: Optional[Transform] = None,
        key_train_metric: Optional[Dict[str, Metric]] = None,
        additional_metrics: Optional[Dict[str, Metric]] = None,
        metric_cmp_fn: Callable = default_metric_cmp_fn,
        train_handlers: Optional[Sequence] = None,
        amp: bool = False,
        event_names: Optional[List[Union[str, EventEnum]]] = None,
        event_to_attr: Optional[dict] = None,
        decollate: bool = True,
        optim_set_to_none: bool = False,
        custom_keys: Optional[dict] = None,
    ) -> None:
        super().__init__(
            device=device,
            max_epochs=max_epochs,
            data_loader=train_data_loader,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            postprocessing=postprocessing,
            key_metric=key_train_metric,
            additional_metrics=additional_metrics,
            metric_cmp_fn=metric_cmp_fn,
            handlers=train_handlers,
            amp=amp,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
        )
        if custom_keys is None:
            self.keys = {
                "IMAGE": Keys.IMAGE,
                "LABEL": Keys.LABEL,
                "PRED": Keys.PRED,
                "LOSS": Keys.LOSS
            }
        else:
            self.keys = custom_keys

    def _iteration(self, engine: Engine, batchdata: Dict[str, torch.Tensor]):
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        batch = self.prepare_batch(batchdata, engine.state.device, engine.non_blocking)
        if len(batch) == 2:
            inputs, targets = batch
            args: Tuple = ()
            kwargs: Dict = {}
        else:
            inputs, targets, args, kwargs = batch
        # put iteration outputs into engine.state
        engine.state.output = {self.keys["IMAGE"]: inputs, self.keys["LABEL"]: targets}

        def _compute_pred_loss():
            preds = self.inferer(inputs, self.network, *args, **kwargs)
            engine.state.output[self.keys["PRED"]] = preds
            engine.fire_event(IterationEvents.FORWARD_COMPLETED)
            if not isinstance(preds, tuple):
                raise ValueError(
                    "Predictions must be tuple in multi-task framework",
                    f"but got {type(engine.state.output[self.keys['PRED']])}"
                )
            if not isinstance(targets, tuple):
                raise ValueError(
                    "Targets must be tuple in multi-task framework",
                    f"but got {type(targets)}"
                )
            if len(preds) != len(targets):
                raise ValueError(
                    "Predictions len must equal to targets len",
                    f"but got {len(preds)} != {len(targets)}"
                )

            if len(preds) != len(self.loss_functions):
                raise ValueError(
                    "Pred len must equal to loss functions len",
                    f"but got {len(preds)} != {len(self.loss_functions)}"
                )

            total_loss = []
            for pred, target, loss_fn in zip(preds, targets, self.loss_functions):
                if pred.dim > target.dim and 1 in pred.shape:
                    pred.squeeze_()
                loss = loss_fn(pred, target).mean()
                total_loss.append(loss)

            engine.state.output[self.keys["LOSS"]] = torch.mean(total_loss)
            engine.fire_event(IterationEvents.LOSS_COMPLETED)

        self.network.train()
        # `set_to_none` only work from PyTorch 1.7.0
        if not pytorch_after(1, 7):
            self.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad(set_to_none=self.optim_set_to_none)

        if self.amp and self.scaler is not None:
            with torch.cuda.amp.autocast():
                _compute_pred_loss()
            self.scaler.scale(engine.state.output[self.keys["LOSS"]]).backward()
            engine.fire_event(IterationEvents.BACKWARD_COMPLETED)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            _compute_pred_loss()
            engine.state.output[self.keys["LOSS"]].backward()
            engine.fire_event(IterationEvents.BACKWARD_COMPLETED)
            self.optimizer.step()
        engine.fire_event(IterationEvents.MODEL_COMPLETED)

        return engine.state.output


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
