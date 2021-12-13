from typing import Dict, Hashable, Mapping, Optional, Sequence, Union, List

import torch
import numpy as np

from monai.config import KeysCollection
from monai.transforms.compose import MapTransform, Transform, Randomizable
from monai.transforms.utils import (
    map_binary_to_indices,
    generate_pos_neg_label_crop_centers,
)
from monai.transforms import RandCropByPosNegLabeld, SpatialCrop, ResizeWithPadOrCrop

from monai_ex.utils import ImageMetaKey as Key
from monai_ex.utils import (
    fall_back_tuple,
    ensure_list,
    ensure_list_rep,
    ensure_tuple_rep,
)
from monai_ex.transforms.croppad.array import (
    CenterMask2DSliceCrop,
    FullMask2DSliceCrop,
    GetMaxSlices3direcCrop,
    KSpaceResample,
)


class CenterMask2DSliceCropd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        mask_key: KeysCollection,
        roi_size: Union[Sequence[int], int],
        crop_mode: str,
        center_mode: str,
        z_axis: int,
        n_slices: Union[Sequence[int], int] = 3,
    ) -> None:
        super().__init__(keys)
        self.mask_key = mask_key
        n_slices = ensure_list_rep(n_slices, len(keys))

        self.cropper = [
            CenterMask2DSliceCrop(
                roi_size=roi_size,
                crop_mode=crop_mode,
                center_mode=center_mode,
                z_axis=z_axis,
                mask_data=None,
                n_slices=n_slice,
            )
            for n_slice in n_slices
        ]

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for i, key in enumerate(self.keys):
            d[key] = self.cropper[i](d[key], d[self.mask_key])
        return d


class FullMask2DSliceCropd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        mask_key: KeysCollection,
        roi_size: Union[Sequence[int], int],
        crop_mode: str,
        z_axis: int,
        n_slices: int = 3,
    ) -> None:
        super().__init__(keys)
        self.mask_key = mask_key
        self.cropper = FullMask2DSliceCrop(
            roi_size=roi_size,
            crop_mode=crop_mode,
            z_axis=z_axis,
            mask_data=None,
            n_slices=n_slices,
        )

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        mask = d[self.mask_key]
        centers = self.cropper.get_center_pos_(mask)
        results: List[Dict[Hashable, np.ndarray]] = [dict() for _ in centers]
        for key in data.keys():
            if key in self.keys:
                img = d[key]
                for i, crop in enumerate(self.cropper(img, msk=mask)):
                    results[i][key] = crop
            else:
                for i in range(len(centers)):
                    results[i][key] = data[key]

        return results


class GetMaxSlices3direcCropd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        mask_key: KeysCollection,
        roi_size: Union[Sequence[int], int],
        crop_mode: str,
        center_mode: str,
        n_slices: int,
    ) -> None:
        super().__init__(keys)
        self.mask_key = mask_key
        self.cropper = GetMaxSlices3direcCrop(
            roi_size=roi_size,
            crop_mode=crop_mode,
            center_mode=center_mode,
            mask_data=None,
            n_slices=n_slices,
        )

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.cropper(d[key], d[self.mask_key])
        return d


class RandCropByPosNegLabelExd(RandCropByPosNegLabeld):
    """Dictionary-based version :py:class:`monai_ex.transforms.RandCropByPosNegLabelEx`."""

    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        spatial_size: Union[Sequence[int], int],
        pos: float = 1.0,
        neg: float = 1.0,
        offset: float = 0.0,
        num_samples: int = 1,
        image_key: Optional[str] = None,
        image_threshold: float = 0.0,
        fg_indices_key: Optional[str] = None,
        bg_indices_key: Optional[str] = None,
        meta_key_postfix: str = "meta_dict",
        allow_missing_keys: bool = False,
        target_label: Optional[int] = None,
    ) -> None:
        super().__init__(
            keys=keys,
            label_key=label_key,
            spatial_size=spatial_size,
            pos=pos,
            neg=neg,
            num_samples=num_samples,
            image_key=image_key,
            image_threshold=image_threshold,
            fg_indices_key=fg_indices_key,
            bg_indices_key=bg_indices_key,
            meta_key_postfix=meta_key_postfix,
            allow_missing_keys=allow_missing_keys,
        )
        self.offset = offset
        self.target_label = target_label
        if self.offset < 0:
            raise ValueError(f"Offset value must greater than 0, but got {offset}")

    def randomize(
        self,
        label: np.ndarray,
        fg_indices: Optional[np.ndarray] = None,
        bg_indices: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
    ) -> None:
        self.spatial_size = fall_back_tuple(self.spatial_size, default=label.shape[1:])
        if np.greater(self.spatial_size, label.shape[1:]).any():
            self.centers = [
                None,
            ] * self.num_samples
            return

        if fg_indices is None or bg_indices is None:
            fg_indices_, bg_indices_ = map_binary_to_indices(
                label, image, self.image_threshold
            )
        else:
            fg_indices_ = fg_indices
            bg_indices_ = bg_indices
        self.centers = generate_pos_neg_label_crop_centers(
            self.spatial_size,
            self.num_samples,
            self.pos_ratio,
            label.shape[1:],
            fg_indices_,
            bg_indices_,
            self.R,
        )
        self.offset_centers = []
        for center in self.centers:
            if 0 < self.offset <= 1:
                offset = [
                    self.R.randint(self.offset * sz // 2) * self.R.choice([1, -1])
                    for sz in self.spatial_size
                ]
            elif self.offset > 1:
                offset = [
                    self.R.randint(self.offset) * self.R.choice([1, -1])
                    for sz in self.spatial_size
                ]
            else:
                offset = [
                    0,
                ] * len(self.spatial_size)
            # print('Offset: ', offset, "Center: ", center)
            self.offset_centers.append([int(c + b) for c, b in zip(center, offset)])
        self.centers = self.offset_centers

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> List[Dict[Hashable, np.ndarray]]:
        d = dict(data)
        label = d[self.label_key]
        image = d[self.image_key] if self.image_key else None
        fg_indices = (
            d.get(self.fg_indices_key) if self.fg_indices_key is not None else None
        )
        bg_indices = (
            d.get(self.bg_indices_key) if self.bg_indices_key is not None else None
        )

        if self.target_label is not None:
            label = (label == self.target_label).astype(np.uint8)

        self.randomize(label, fg_indices, bg_indices, image)
        if not isinstance(self.spatial_size, tuple):
            raise TypeError(
                f"Expect spatial_size to be tuple, but got {type(self.spatial_size)}"
            )
        if self.centers is None:
            raise AssertionError
        results: List[Dict[Hashable, np.ndarray]] = [
            {} for _ in range(self.num_samples)
        ]

        for i, center in enumerate(self.centers):
            for key in self.key_iterator(d):
                img = d[key]
                if np.greater(self.spatial_size, img.shape[1:]).any():
                    cropper = ResizeWithPadOrCrop(spatial_size=self.spatial_size)
                else:
                    cropper = SpatialCrop(roi_center=tuple(center), roi_size=self.spatial_size)  # type: ignore
                results[i][key] = cropper(img)
            # fill in the extra keys with unmodified data
            for key in set(data.keys()).difference(set(self.keys)):
                results[i][key] = data[key]
            # add `patch_index` to the meta data
            for key in self.key_iterator(d):
                meta_data_key = f"{key}_{self.meta_key_postfix}"
                if meta_data_key not in results[i]:
                    results[i][meta_data_key] = {}  # type: ignore
                results[i][meta_data_key][Key.PATCH_INDEX] = i

        return results


class RandCrop2dByPosNegLabeld(Randomizable, MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        spatial_size: Union[Sequence[int], int],
        crop_mode: str,
        z_axis: int,
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        image_key: Optional[str] = None,
        image_threshold: float = 0.0,
        fg_indices_key: Optional[str] = None,
        bg_indices_key: Optional[str] = None,
    ) -> None:
        super().__init__(keys)
        self.spatial_size = ensure_tuple_rep(spatial_size, 2)
        self.label_key = label_key
        self.num_samples = num_samples
        self.image_key = image_key
        self.image_threshold = image_threshold
        self.centers: Optional[List[List[np.ndarray]]] = None
        self.fg_indices_key = fg_indices_key
        self.bg_indices_key = bg_indices_key

        if pos < 0 or neg < 0:
            raise ValueError(
                f"pos and neg must be nonnegative, got pos={pos} neg={neg}."
            )
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.pos_ratio = pos / (pos + neg)
        if crop_mode not in ["single", "cross", "parallel"]:
            raise ValueError("Cropping mode must be one of 'single, cross, parallel'")
        self.crop_mode = crop_mode
        self.z_axis = z_axis

    def get_new_spatial_size(self):
        spatial_size_ = ensure_list(self.spatial_size)
        if self.crop_mode == "single":
            spatial_size_.insert(self.z_axis, 1)
        elif self.crop_mode == "parallel":
            spatial_size_.insert(self.z_axis, 3)
        else:
            spatial_size_ = [
                max(spatial_size_),
            ] * 3

        return spatial_size_

    def randomize(
        self,
        label: np.ndarray,
        fg_indices: Optional[np.ndarray] = None,
        bg_indices: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
    ) -> None:
        if fg_indices is None or bg_indices is None:
            fg_indices_, bg_indices_ = map_binary_to_indices(
                label, image, self.image_threshold
            )
        else:
            fg_indices_ = fg_indices
            bg_indices_ = bg_indices

        self.centers = generate_pos_neg_label_crop_centers(
            self.get_new_spatial_size(),
            self.num_samples,
            self.pos_ratio,
            label.shape[1:],
            fg_indices_,
            bg_indices_,
            self.R,
        )

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> List[Dict[Hashable, np.ndarray]]:
        d = dict(data)
        label = d[self.label_key]
        image = d[self.image_key] if self.image_key else None
        fg_indices = (
            d.get(self.fg_indices_key, None)
            if self.fg_indices_key is not None
            else None
        )
        bg_indices = (
            d.get(self.bg_indices_key, None)
            if self.bg_indices_key is not None
            else None
        )

        self.randomize(label, fg_indices, bg_indices, image)
        assert isinstance(self.spatial_size, tuple)
        assert self.centers is not None
        results: List[Dict[Hashable, np.ndarray]] = [
            dict() for _ in range(self.num_samples)
        ]
        for key in data.keys():
            if key in self.keys:
                img = d[key]
                for i, center in enumerate(self.centers):
                    if self.crop_mode in ["single", "parallel"]:
                        size_ = self.get_new_spatial_size()
                        slice_ = SpatialCrop(roi_center=tuple(center), roi_size=size_)(
                            img
                        )
                        results[i][key] = np.moveaxis(slice_.squeeze(0), self.z_axis, 0)
                    else:
                        cross_slices = np.zeros(shape=(3,) + self.spatial_size)
                        for k in range(3):
                            size_ = np.insert(self.spatial_size, k, 1)
                            slice_ = SpatialCrop(
                                roi_center=tuple(center), roi_size=size_
                            )(img)
                            cross_slices[k] = slice_.squeeze()
                        results[i][key] = cross_slices
            else:
                for i in range(self.num_samples):
                    results[i][key] = data[key]

        return results


class KSpaceResampled(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai_ex.transforms.KSpaceResample`.

    This transform assumes the ``data`` dictionary has a key for the input
    data's metadata and contains `affine` field.  The key is formed by ``key_{meta_key_postfix}``.

    After resampling the input array, this transform will write the new affine
    to the `affine` field of metadata which is formed by ``key_{meta_key_postfix}``.

    see also:
        :py:class:`monai_ex.transforms.KSpaceResample`
    """

    def __init__(
        self,
        keys: KeysCollection,
        pixdim: Union[Sequence[float], float],
        diagonal: bool = False,
        device: Optional[torch.device] = None,
        tolerance: float = 0.0001,
        meta_key_postfix: str = "meta_dict",
    ) -> None:
        super().__init__(keys)
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.resizer = KSpaceResample(
            pixdim=pixdim,
            diagonal=diagonal,
            device=device,
            tolerance=tolerance,
        )

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key, metakey_postfix in zip(self.keys, self.meta_key_postfix):
            meta_key = f"{key}_{metakey_postfix}"
            # create metadata if necessary
            if meta_key not in d:
                d[meta_key] = {"affine": None}
            meta_data = d[meta_key]

            d[key], old_affine, new_affine = self.resizer(
                d[key],
                affine=meta_data["affine"],
            )
            meta_data["affine"] = new_affine
        return d


CenterMask2DSliceCropD = CenterMask2DSliceCropDict = CenterMask2DSliceCropd
FullMask2DSliceCropD = FullMask2DSliceCropDict = FullMask2DSliceCropd
GetMaxSlices3direcCropD = GetMaxSlices3direcCropDict = GetMaxSlices3direcCropd
RandCropByPosNegLabelExD = RandCropByPosNegLabelExDict = RandCropByPosNegLabelExd
RandCrop2dByPosNegLabelD = RandCrop2dByPosNegLabelDict = RandCrop2dByPosNegLabeld
KSpaceResampleD = KSpaceResampleDict = KSpaceResampled
