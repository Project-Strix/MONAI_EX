# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Dict, Optional

import torch

from monai.utils import exact_version, optional_import
from monai.handlers import CheckpointLoader

Events, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Events")
Checkpoint, _ = optional_import("ignite.handlers", "0.4.2", exact_version, "Checkpoint")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Engine")


class CheckpointLoaderEx(CheckpointLoader):
    """
    Extension verion of MONAI's CheckpointLoader .
    Extended: strict, skip_mismatch

    CheckpointLoader acts as an Ignite handler to load checkpoint data from file.
    It can load variables for network, optimizer, lr_scheduler, etc.
    If saving checkpoint after `torch.nn.DataParallel`, need to save `model.module` instead
    as PyTorch recommended and then use this loader to load the model.

    Args:
        load_path: the file path of checkpoint, it should be a PyTorch `pth` file.
        load_dict: target objects that load checkpoint to. examples::

            {'network': net, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

        name: identifier of logging.logger to use, if None, defaulting to ``engine.logger``.
        map_location: when loading the module for distributed training/evaluation,
            need to provide an appropriate map_location argument to prevent a process
            to step into others’ devices. If map_location is missing, torch.load will
            first load the module to CPU and then copy each parameter to where it was
            saved, which would result in all processes on the same machine using the
            same set of devices.
        strict: whether to strictly enforce that the keys in :attr:`state_dict` with :attr:`prefix`
                match the names of parameters and buffers in this module
        skip_mismatch: whether to skip loading of layers where there is a mismatch in the 
                       number of weights, or a mismatch in the shape of the weight

    """

    def __init__(
        self,
        load_path: str,
        load_dict: Dict,
        name: Optional[str] = None,
        map_location: Optional[Dict] = None,
        strict=True,
        skip_mismatch=False,
    ) -> None:
        super(CheckpointLoaderEx, self).__init__(
            load_path=load_path,
            load_dict=load_dict,
            name=name,
            map_location=map_location
        )
        self.strict = strict
        self.skip_mismatch = skip_mismatch

    def __call__(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        checkpoint = torch.load(self.load_path, map_location=self.map_location)
        key = 'net'
        if len(self.load_dict) == 1:
            key = list(self.load_dict.keys())[0]
            if not (key in checkpoint):
                checkpoint = {key: checkpoint}
        
        if self.skip_mismatch:
            if key in self.load_dict.keys():
                model_dict = self.load_dict[key].state_dict().copy()
                filtered_dict = {k: v for k, v in checkpoint[key].items() if v.shape == model_dict[k].shape}
                model_dict.update(filtered_dict)
                checkpoint[key] = model_dict
            else:
                raise ValueError("Cannot find network key '{}' in model".format(key))

        Checkpoint.load_objects(to_load=self.load_dict, checkpoint=checkpoint, strict=self.strict)
        self.logger.info(f"Restored all variables from {self.load_path}")
