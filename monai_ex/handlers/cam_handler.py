import logging
from typing import TYPE_CHECKING, Optional

import torch
import numpy as np
import nibabel as nib

from monai_ex.visualize import GradCAM, LayerCAM
from monai_ex.utils import exact_version, optional_import
from medlp.utilities.utils import apply_colormap_on_image

from PIL import Image
from utils_cw import Normalize2

if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Engine")


class GradCamHandler:
    def __init__(
        self,
        net,
        target_layers,
        target_class,
        data_loader,
        prepare_batch_fn,
        method: str = "gradcam",
        fusion: bool = False,
        hierarchical: bool = False,
        save_dir: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
        logger_name: Optional[str] = None,
    ) -> None:
        self.net = net
        self.target_layers = target_layers
        self.target_class = target_class
        self.data_loader = data_loader
        self.prepare_batch_fn = prepare_batch_fn
        self.save_dir = save_dir
        self.device = device
        self.logger = logging.getLogger(logger_name)
        self.fusion = fusion
        self.hierarchical = hierarchical
        if method == "gradcam":
            self.cam = GradCAM(nn_module=self.net, target_layers=self.target_layers)
        elif method == "layercam":
            self.cam = LayerCAM(
                nn_module=self.net,
                target_layers=self.target_layers,
                hierarchical=hierarchical,
            )

        if fusion:
            self.suffix = "fusion"
        elif hierarchical:
            self.suffix = "hierarchical"
        else:
            self.suffix = ""

    def __call__(self, engine: Engine) -> None:
        for i, batchdata in enumerate(self.data_loader):
            batch = self.prepare_batch_fn(batchdata, self.device, False)
            if len(batch) == 2:
                inputs, targets = batch
            else:
                raise NotImplementedError

            if isinstance(inputs, (tuple, list)):
                self.logger.warn(
                    f"Got multiple inputs with size of {len(batch)},"
                    "select the first one as image data."
                )
                origin_img = inputs[0].cpu().detach().numpy().squeeze(1)
            else:
                origin_img = inputs.cpu().detach().numpy().squeeze(1)

            self.logger.debug(f"Input len: {len(inputs)}, shape: {origin_img.shape}")

            cam_result = self.cam(
                inputs,
                class_idx=self.target_class,
                img_spatial_size=origin_img.shape[1:],
            )

            self.logger.debug(
                f"Image batchdata shape: {origin_img.shape}, "
                f"CAM batchdata shape: {cam_result.shape}"
            )

            if len(origin_img.shape[1:]) == 3:
                for j, (img_slice, cam_slice) in enumerate(zip(origin_img, cam_result)):
                    file_name = (
                        f"batch{i}_{j}_cam_{self.suffix}_{self.target_layers}.nii.gz"
                    )
                    nib.save(
                        nib.Nifti1Image(img_slice.squeeze(), np.eye(4)),
                        self.save_dir / f"batch{i}_{j}_images.nii.gz",
                    )

                    if cam_slice.shape[0] > 1 and self.fusion:
                        output_cam = cam_slice.mean(axis=0).squeeze()
                    elif self.hierarchical:
                        output_cam = np.flip(
                            cam_slice.transpose(1, 2, 3, 0), 3
                        ).squeeze()
                    else:
                        output_cam = cam_slice.transpose(1, 2, 3, 0).squeeze()

                    nib.save(
                        nib.Nifti1Image(output_cam, np.eye(4)),
                        self.save_dir / file_name,
                    )

            elif len(origin_img.shape[1:]) == 2:
                cam_result = np.uint8(cam_result.squeeze(1) * 255)
                for j, (img_slice, cam_slice) in enumerate(zip(origin_img, cam_result)):
                    img_slice = np.uint8(Normalize2(img_slice) * 255)

                    img_slice = Image.fromarray(img_slice)
                    no_trans_heatmap, heatmap_on_image = apply_colormap_on_image(
                        img_slice, cam_slice, "hsv"
                    )

                    heatmap_on_image.save(
                        self.save_dir / f"batch{i}_{j}_heatmap_on_img.png"
                    )
            else:
                raise NotImplementedError(f"Cannot support ({origin_img.shape}) data.")

        engine.terminate()
