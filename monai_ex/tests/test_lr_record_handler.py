import pytest

from ignite.engine import Engine, Events
import torch
from torch.optim import SGD
from monai.engines.workflow import Workflow
from monai_ex.handlers.lr_record_handler import LearningHistoryRecordHandler

class TestHandlerCheckpointSaver:

    data = [0] * 8
    net = torch.nn.PReLU()
    optimizer = SGD(net.parameters(), lr=0.1)

    def test_mismatch_loss(self):
        def _train_func(engine, batch):
            return {"loss": engine.state.iteration}

        engine = Engine(_train_func)
        engine.optimizer = self.optimizer
        history_handler1 = LearningHistoryRecordHandler(lambda x: x["loss1"], None)
        history_handler1.attach(engine)

        with pytest.raises(KeyError):
            engine.run(self.data)

        engine.remove_event_handler(history_handler1.iteration_completed, Events.ITERATION_COMPLETED)

    def test_loss(self):
        def _train_func(engine, batch):
            return {"loss": engine.state.iteration}

        engine = Engine(_train_func)
        engine.optimizer = self.optimizer
        history_handler2 = LearningHistoryRecordHandler(lambda x: x["loss"], None)
        history_handler2.attach(engine)

        engine.run(self.data)
        assert history_handler2.history["loss"] == list(range(1, 9))
