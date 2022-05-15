from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from monai.config import IgniteInfo
from monai.engines.trainer import SupervisedTrainer, Trainer
from monai.engines.utils import IterationEvents, default_metric_cmp_fn
from monai.inferers import Inferer
from monai.transforms import Transform, apply_transform
from monai.utils import min_version, optional_import, pytorch_after
from monai_ex.engines.utils import CustomKeys as Keys
from monai_ex.engines.utils import default_prepare_batch_ex
from monai_ex.inferers import Inferer
from monai_ex.utils import GenericException as StrixException
from monai_ex.utils import ensure_same_dim, trycatch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from ignite.engine import Engine, EventEnum, Events
    from ignite.metrics import Metric
else:
    Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")
    Metric, _ = optional_import("ignite.metrics", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Metric")
    EventEnum, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "EventEnum")


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

                if isinstance(output1, tuple) and len(output1) == 2:  # 2 outputs
                    loss = self.loss_function(output1[0], output2[0], output1[1], output2[1], targets1, targets2)
                elif isinstance(output1, torch.Tensor):
                    loss = self.loss_function(output1, output2, targets1, targets2)
                else:
                    raise NotImplementedError(
                        f"SiameseNet expected 1or2 outputs," f"but got {type(output1)} with size of {len(output1)}"
                    )

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
                raise NotImplementedError(f"SiameseNet expected 1or2 outputs, but got {len(output1)}")

            loss.backward()
            self.optimizer.step()

        if isinstance(output1, tuple) and len(output1) == 2:
            return {
                Keys.IMAGE: torch.cat((inputs1, inputs2), dim=0),
                Keys.LABEL: torch.cat((targets1, targets2), dim=0),
                Keys.LATENT: torch.cat((output1[0], output2[0]), dim=0),
                Keys.PRED: torch.cat((output1[1], output2[1]), dim=0),
                Keys.LOSS: loss.item(),
            }
        else:
            return {
                Keys.IMAGE: torch.cat((inputs1, inputs2), dim=0),
                Keys.LABEL: torch.cat((targets1, targets2), dim=0),
                Keys.LATENT: torch.cat((output1, output2), dim=0),
                Keys.LOSS: loss.item(),
            }


class SupervisedTrainerEx(SupervisedTrainer):
    """Extension of MONAI's SupervisedTrainer.
    Extended: 
      `custom_keys`: Input custom_keys setting.
      `ensure_dims`: Add simple confirmation for tensor dims of `pred` and `label`
                     before feeding into loss fn.
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
        postprocessing: Optional[Transform] = None,
        key_train_metric: Optional[Dict[str, Metric]] = None,
        additional_metrics: Optional[Dict[str, Metric]] = None,
        metric_cmp_fn: Callable = default_metric_cmp_fn,
        train_handlers: Optional[Sequence] = None,
        amp: bool = False,
        event_names: Optional[List[Union[str, EventEnum]]] = None,
        event_to_attr: Optional[dict] = None,
        decollate: bool = True,
        custom_keys: Optional[dict] = None,
        ensure_dims: bool = False,
    ) -> None:
        super().__init__(
            device=device,
            max_epochs=max_epochs,
            train_data_loader=train_data_loader,
            network=network,
            optimizer=optimizer,
            loss_function=loss_function,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            inferer=inferer,
            postprocessing=postprocessing,
            key_train_metric=key_train_metric,
            additional_metrics=additional_metrics,
            metric_cmp_fn=metric_cmp_fn,
            train_handlers=train_handlers,
            amp=amp,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
        )
        if custom_keys is None:
            self.keys = {"IMAGE": Keys.IMAGE, "LABEL": Keys.LABEL, "PRED": Keys.PRED, "LOSS": Keys.LOSS}
        else:
            self.keys = custom_keys
        self.ensure_dims = ensure_dims

    @trycatch()
    def _iteration(self, engine: Engine, batchdata: Dict[str, torch.Tensor]):
        if batchdata is None:
            raise StrixException(
                "No data were fed into the Trainer engine. "
                "Consider the possibility that Transforms did not succeed or "
                "there is a problem with your dataset."
            )
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

            if self.ensure_dims:
                engine.state.output[self.keys["LOSS"]] = self.loss_function(
                    *ensure_same_dim(engine.state.output[self.keys["PRED"]], targets)
                ).mean()
            else:
                engine.state.output[self.keys["LOSS"]] = self.loss_function(
                    engine.state.output[self.keys["PRED"]], targets
                ).mean()

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
        loss_function: List[Callable],
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
        self.network = network
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.inferer = inferer
        self.optim_set_to_none = optim_set_to_none

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
            preds = self.inferer(inputs, self.network, *args, **kwargs)
            engine.state.output[self.keys["PRED"]] = preds
            engine.fire_event(IterationEvents.FORWARD_COMPLETED)
            if not isinstance(preds, tuple):
                raise ValueError(
                    "Predictions must be tuple in multi-task framework",
                    f"but got {type(engine.state.output[self.keys['PRED']])}",
                )
            if not isinstance(targets, tuple):
                raise ValueError(f"Targets must be tuple in multi-task framework, but got {type(targets)}")
            if len(preds) != len(targets):
                raise ValueError(f"Predictions len must equal to targets, but got {len(preds)} != {len(targets)}")

            loss = self.loss_function(preds, targets)

            engine.state.output[self.keys["LOSS"]] = loss
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
