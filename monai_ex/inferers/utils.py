from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from monai.inferers.utils import _get_scan_interval
from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.utils import BlendMode, PytorchPadMode, fall_back_tuple


def sliding_window_classification(
    inputs: torch.Tensor,
    roi_size: Union[Sequence[int], int],
    sw_batch_size: int,
    predictor: Callable[[torch.Tensor], torch.Tensor],
    overlap: float = 0.25,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:

    num_spatial_dims = len(inputs.shape) - 2
    assert 0 <= overlap < 1, "overlap must be >= 0 and < 1."

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size_ = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    # TODO: Enable batch sizes > 1 in future
    if batch_size > 1:
        raise NotImplementedError("Currently only inputs with batch size = 1 are supported.")

    if device is None:
        device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=PytorchPadMode(padding_mode).value, value=cval)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)

    slice_batches = []
    for slice_index in range(0, len(slices), sw_batch_size):
        slice_index_range = range(slice_index, min(slice_index + sw_batch_size, len(slices)))
        input_slices = []
        for curr_index in slice_index_range:
            curr_slice = slices[curr_index]
            if len(curr_slice) == 3:
                input_slices.append(inputs[0, :, curr_slice[0], curr_slice[1], curr_slice[2]])
            else:
                input_slices.append(inputs[0, :, curr_slice[0], curr_slice[1]])
        slice_batches.append(torch.stack(input_slices))

    # Perform predictions
    output_rois = list()
    for data in slice_batches:
        cls_prob = predictor(data)  # batched patch classification
        output_rois.append(cls_prob.to(device))

    # stitching output image
    output_classes = output_rois[0].shape[1]
    output_shape = [batch_size, output_classes] + list(image_size)

    # Create importance map
    importance_map = compute_importance_map(get_valid_patch_size(image_size, roi_size), mode=mode, device=device)

    # allocate memory to store the full output and the count for overlapping parts
    output_image = torch.zeros(output_shape, dtype=torch.float32, device=device)
    count_map = torch.zeros(output_shape, dtype=torch.float32, device=device)

    for window_id, slice_index in enumerate(range(0, len(slices), sw_batch_size)):
        slice_index_range = range(slice_index, min(slice_index + sw_batch_size, len(slices)))

        # store the result in the proper location of the full output. Apply weights from importance map.
        for curr_index in slice_index_range:
            curr_slice = slices[curr_index]
            curr_roi = output_rois[window_id][curr_index-slice_index, :]
            if len(curr_slice) == 3:
                output_image[0, :, curr_slice[0], curr_slice[1], curr_slice[2]] += (
                    importance_map * curr_roi.view(-1,1,1,1)
                )
                count_map[0, :, curr_slice[0], curr_slice[1], curr_slice[2]] += importance_map
            else:
                output_image[0, :, curr_slice[0], curr_slice[1]] += (
                    importance_map * curr_roi.view(-1,1,1)
                )
                count_map[0, :, curr_slice[0], curr_slice[1]] += importance_map

    # account for any overlapping sections
    output_image = output_image / count_map

    if num_spatial_dims == 3:
        return output_image[
            ...,
            pad_size[4] : image_size_[0] + pad_size[4],
            pad_size[2] : image_size_[1] + pad_size[2],
            pad_size[0] : image_size_[2] + pad_size[0],
        ]
    return output_image[
        ..., pad_size[2] : image_size_[0] + pad_size[2], pad_size[0] : image_size_[1] + pad_size[0]
    ]  # 2D
