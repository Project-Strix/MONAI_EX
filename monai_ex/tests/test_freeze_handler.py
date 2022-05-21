import logging
import pytest

import torch
import torch.optim as optim
from ignite.engine import Engine

from monai_ex.handlers.freeze_handler import FreezeNetHandler


class TestFreezeHandler:

    class FreezableNet(torch.nn.Module):
        def freeze(self, on):
            pass
        def forward(self, x):
            return x

    def test_freezing_exception(self):
        data = torch.rand(10)

        # set up engine
        def _train_func(engine, batch):
            engine.state.metrics["val_loss"] = engine.state.iteration

        engine = Engine(_train_func)

        # set up testing handler
        net = torch.nn.Linear(10, 2)
        optimizer = optim.SGD(net.parameters(), lr=0.02)
        handler = FreezeNetHandler(net, "until", 4)
        handler.attach(engine)

        with pytest.raises(AttributeError):  #no freeze() func.
            assert engine.run(data, max_epochs=5)
    
    def test_freezing_exception(self, caplog):
        data = torch.rand(10)

        # set up engine
        def _train_func(engine, batch):
            engine.state.metrics["val_loss"] = engine.state.iteration

        engine = Engine(_train_func)
        net = self.FreezableNet()
        handler2 = FreezeNetHandler(net, "until", 4, logger_name="freezing")
        handler2.attach(engine)
        with caplog.at_level(logging.INFO, logger="freezing"):
            engine.run(data, max_epochs=5)
            logs = [rec.message for rec in caplog.records]
            assert len(logs) == 2
            assert 'is froze' in logs[0]
            assert 'is unfroze' in logs[1]
