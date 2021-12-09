from typing import (
    TYPE_CHECKING,
    Callable,
    Tuple,
    Dict,
    Optional,
    Sequence,
    Union,
    Iterable,
    List,
)

import torch
from torch.utils.data import DataLoader

from monai_ex.engines.utils import CustomKeys as Keys
from monai_ex.engines.utils import default_prepare_batch_ex
from monai.engines import Evaluator, SupervisedEvaluator
from monai.engines.utils import IterationEvents
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import Transform
from monai.utils import ForwardMode, ensure_tuple, exact_version, optional_import
from monai.visualize.class_activation_maps import ModelWithHooks

if TYPE_CHECKING:
    from ignite.engine import Engine, EventEnum
    from ignite.metrics import Metric
else:
    Engine, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Engine")
    Metric, _ = optional_import("ignite.metrics", "0.4.4", exact_version, "Metric")
    EventEnum, _ = optional_import("ignite.engine", "0.4.4", exact_version, "EventEnum")


class SiameseEvaluator(Evaluator):
    """
    Siamese evaluation method with image and label(optional), inherits from evaluator and Workflow.

    Args:
        device: an object representing the device on which to run.
        val_data_loader: Ignite engine use data_loader to run, must be torch.DataLoader.
        network: use the network to run model forward.
        epoch_length: number of iterations for one epoch, default to `len(val_data_loader)`.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function to parse image and label for current iteration.
        iteration_update: the callable function for every iteration, expect to accept `engine`
            and `batchdata` as input parameters. if not provided, use `self._iteration()` instead.
        inferer: inference method that execute model forward on input data, like: SlidingWindow, etc.
        post_transform: execute additional transformation for the model output data.
            Typically, several Tensor based transforms composed by `Compose`.
        key_val_metric: compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_val_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics: more Ignite metrics that also attach to Ignite Engine.
        val_handlers: every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, SegmentationSaver, etc.
        amp: whether to enable auto-mixed-precision evaluation, default is False.

    """

    def __init__(
        self,
        device: torch.device,
        val_data_loader: DataLoader,
        network: torch.nn.Module,
        loss_function: Callable,
        epoch_length: Optional[int] = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch_ex,
        iteration_update: Optional[Callable] = None,
        inferer: Optional[Inferer] = None,
        post_transform: Optional[Transform] = None,
        key_val_metric: Optional[Dict[str, Metric]] = None,
        additional_metrics: Optional[Dict[str, Metric]] = None,
        val_handlers: Optional[Sequence] = None,
        amp: bool = False,
    ) -> None:
        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            post_transform=post_transform,
            key_val_metric=key_val_metric,
            additional_metrics=additional_metrics,
            val_handlers=val_handlers,
            amp=amp,
        )

        self.network = network
        self.loss_function = loss_function
        self.inferer = SimpleInferer() if inferer is None else inferer

    def _iteration(
        self, engine: Engine, batchdata: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        callback function for the Supervised Evaluation processing logic of 1 iteration in Ignite Engine.
        Return below items in a dictionary:
            - IMAGE: image Tensor data for model input, already moved to device.
            - LABEL: label Tensor data corresponding to the image, already moved to device.
            - PRED: prediction result of model.

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

        batch1 = self.prepare_batch(
            batchdata[0], engine.state.device, engine.non_blocking
        )
        batch2 = self.prepare_batch(
            batchdata[1], engine.state.device, engine.non_blocking
        )

        if len(batch1) == 2:
            inputs1, targets1 = batch1
            inputs2, targets2 = batch2
            args: Tuple = tuple()
            kwargs: Dict = dict()
        else:
            inputs1, targets1, args, kwargs = batch1
            inputs2, targets2, args, kwargs = batch2

        # execute forward computation
        self.network.eval()
        with torch.no_grad():
            if self.amp:
                with torch.cuda.amp.autocast():
                    output1 = self.inferer(inputs1, self.network, *args, **kwargs)
                    output2 = self.inferer(inputs2, self.network, *args, **kwargs)
                    if len(output1) == 1:
                        if self.loss_function:
                            loss = self.loss_function(
                                output1, output2, targets1, targets2
                            ).item()
                        else:
                            loss = 0
                    elif len(output1) == 2:  # Contrastive+CE
                        if self.loss_function:
                            loss = self.loss_function(
                                output1[0],
                                output2[0],
                                output1[1],
                                output2[1],
                                targets1,
                                targets2,
                            ).item()
                        else:
                            loss = 0
                    else:
                        raise NotImplementedError(
                            f"SiameseNet expected 1or2 outputs, but got {len(output1)}"
                        )
            else:
                output1 = self.inferer(inputs1, self.network, *args, **kwargs)
                output2 = self.inferer(inputs2, self.network, *args, **kwargs)
                if len(output1) == 1:
                    if self.loss_function:
                        loss = self.loss_function(
                            output1, output2, targets1, targets2
                        ).item()
                    else:
                        loss = 0
                elif len(output1) == 2:  # Contrastive+CE
                    if self.loss_function:
                        loss = self.loss_function(
                            output1[0],
                            output2[0],
                            output1[1],
                            output2[1],
                            targets1,
                            targets2,
                        ).item()
                    else:
                        loss = 0
                else:
                    raise NotImplementedError(
                        f"SiameseNet expected 1or2 outputs, but got {len(output1)}"
                    )
        if len(output1) == 1:
            return {
                Keys.IMAGE: torch.cat((inputs1, inputs2), dim=0),
                Keys.LABEL: torch.cat((targets1, targets2), dim=0),
                Keys.LATENT: torch.cat((output1, output2), dim=0),
                Keys.LOSS: loss.item(),
            }
        elif len(output2) == 2:
            return {
                Keys.IMAGE: torch.cat((inputs1, inputs2), dim=0),
                Keys.LABEL: torch.cat((targets1, targets2), dim=0),
                Keys.LATENT: torch.cat((output1[0], output2[0]), dim=0),
                Keys.PRED: torch.cat((output1[1], output2[1]), dim=0),
                Keys.LOSS: loss.item(),
            }


class SupervisedEvaluatorEx(SupervisedEvaluator):
    """Extension of MONAI's SupervisedEvaluator.
    Extended: custom_keys: for support medlp custom input keys.
              output_latent_code: output registered forward&backward latent code.
              target_latent_layer: target layer name of the latent code.

    """

    def __init__(
        self,
        device: torch.device,
        val_data_loader: Union[Iterable, DataLoader],
        network: torch.nn.Module,
        epoch_length: Optional[int] = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch_ex,
        iteration_update: Optional[Callable] = None,
        inferer: Optional[Inferer] = None,
        post_transform: Optional[Transform] = None,
        key_val_metric: Optional[Dict[str, Metric]] = None,
        additional_metrics: Optional[Dict[str, Metric]] = None,
        val_handlers: Optional[Sequence] = None,
        amp: bool = False,
        mode: Union[ForwardMode, str] = ForwardMode.EVAL,
        event_names: Optional[List[Union[str, EventEnum]]] = None,
        event_to_attr: Optional[dict] = None,
        custom_keys: Optional[dict] = None,
        output_latent_code: bool = False,
        target_latent_layer: Optional[str] = None,
    ) -> None:
        super().__init__(
            device,
            val_data_loader,
            network,
            epoch_length,
            non_blocking,
            prepare_batch,
            iteration_update,
            inferer,
            post_transform,
            key_val_metric,
            additional_metrics,
            val_handlers,
            amp,
            mode,
            event_names,
            event_to_attr,
        )
        if custom_keys is None:
            self.keys = {
                "IMAGE": Keys.IMAGE,
                "LABEL": Keys.LABEL,
                "PRED": Keys.PRED,
                "LOSS": Keys.LOSS,
            }
        else:
            self.keys = custom_keys

        if output_latent_code and target_latent_layer is not None:
            self.network_latent = ModelWithHooks(
                self.network, target_latent_layer, register_forward=True
            )
        else:
            self.network_latent = None

    def _iteration(
        self, engine: Engine, batchdata: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
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
        # execute forward computation
        with self.mode(self.network):
            if self.amp:
                with torch.cuda.amp.autocast():
                    if self.network_latent:
                        (
                            engine.state.output[self.keys["PRED"]],
                            engine.state.output[self.keys["FORWARD"]],
                            engine.state.output[self.keys["BACKWARD"]],
                        ) = self.inferer(inputs, self.network_latent, *args, **kwargs)
                    else:
                        engine.state.output[self.keys["PRED"]] = self.inferer(
                            inputs, self.network, *args, **kwargs
                        )
            else:
                if self.network_latent:
                    (
                        engine.state.output[self.keys["PRED"]],
                        engine.state.output[self.keys["FORWARD"]],
                        engine.state.output[self.keys["BACKWARD"]],
                    ) = self.inferer(inputs, self.network_latent, *args, **kwargs)
                else:
                    engine.state.output[self.keys["PRED"]] = self.inferer(
                        inputs, self.network, *args, **kwargs
                    )
        engine.fire_event(IterationEvents.FORWARD_COMPLETED)
        engine.fire_event(IterationEvents.MODEL_COMPLETED)

        return engine.state.output
