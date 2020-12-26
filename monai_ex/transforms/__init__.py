from monai.transforms import *

from .compose import *
from .intensity.array import *
from .intensity.dictionary import *
from .io.array import *
from .io.dictionary import *
from .spatial.array import *
from .spatial.dictionary import *
from .register import LOADER, CHANNELER, ORIENTER, RESCALER, RESIZER, CROPADER, AUGMENTOR, UTILS

LOADER.register('LoadImage', LoadImageD)
LOADER.register('LoadNifti', LoadNiftiD)
LOADER.register('LoadPNG', LoadPNGExD)
LOADER.register('LoadNpy', LoadNumpyD)

CHANNELER.register('AddChannelFirst', AddChannelD)
CHANNELER.register('AsChannelFirst', AsChannelFirstD)
CHANNELER.register('AsChannelLast', AsChannelLastD)
CHANNELER.register('RepeatChannel', RepeatChannelD)
CHANNELER.register('SplitChannel', SplitChannelD)

ORIENTER.register('Orientation', OrientationD)
ORIENTER.register('Rotate90', Rotate90D)
ORIENTER.register('Flip', FlipD)

RESCALER.register('NormalizeIntensity', NormalizeIntensityD)
RESCALER.register('ScaleIntensity', ScaleIntensityD)
RESCALER.register('ScaleIntensityRange', ScaleIntensityRangeD)
RESCALER.register('ScaleIntensityRangePercentiles', ScaleIntensityRangePercentilesD)
RESCALER.register('ScaleIntensityByDicomInfo', ScaleIntensityByDicomInfoD)

RESIZER.register('Spacing', SpacingD)
RESIZER.register('Resize', ResizeD)
RESIZER.register('Zoom', ZoomD)
RESIZER.register('FixedResize', FixedResizeD)
RESIZER.register('ResizeWithPadOrCrop', ResizeWithPadOrCropD)

CROPADER.register('CenterSpatialCrop', CenterSpatialCropD)
CROPADER.register('SpatialCrop', SpatialCropD)
CROPADER.register('CropForeground', CropForegroundD)
CROPADER.register('DivisiblePad', DivisiblePadD)
CROPADER.register('SpatialPad', SpatialPadD)
CROPADER.register('BorderPad', BorderPadD)

AUGMENTOR.register('RandSpatialCrop', RandSpatialCropD)
AUGMENTOR.register('RandCropByPosNegLabel', RandCropByPosNegLabelD)
AUGMENTOR.register('RandGaussianNoise', RandGaussianNoiseD)
AUGMENTOR.register('RandShiftIntensity', RandShiftIntensityD)
AUGMENTOR.register('RandScaleIntensity', RandScaleIntensityD)
AUGMENTOR.register('RandAdjustContrast', RandAdjustContrastD)
AUGMENTOR.register('RandGaussianSmooth', RandGaussianSmoothD)
AUGMENTOR.register('RandGaussianSharpen', RandGaussianSharpenD)
AUGMENTOR.register('RandHistogramShift', RandHistogramShiftD)
AUGMENTOR.register('RandRotate', RandRotateD)
AUGMENTOR.register('RandFlip', RandFlipD)
AUGMENTOR.register('RandZoom', RandZoomD)
AUGMENTOR.register('RandAffine', RandAffineD)
AUGMENTOR.register('Rand2DElastic', Rand2DElasticD)
AUGMENTOR.register('Rand3DElastic', Rand3DElasticD)
AUGMENTOR.register('RandRotate90', RandRotate90D)
AUGMENTOR.register('RandSpatialCropSamples', RandSpatialCropSamplesD)
AUGMENTOR.register('RandWeightedCrop', RandWeightedCropD)
# AUGMENTOR.register('RandLocalPixelShuffle', RandLocalPixelShuffleD)
# AUGMENTOR.register('RandImageInpainting', RandImageInpaintingD)
# AUGMENTOR.register('RandImageOutpainting', RandImageOutpaintingD)
# AUGMENTOR.register('RandNonlinear', RandNonlinearD)


UTILS.register('Rotate', RotateD)
UTILS.register('CastToType', CastToTypeD)
UTILS.register('ToTensor', ToTensorD)
UTILS.register('SqueezeDim', SqueezeDimD)
UTILS.register('LabelToMask', LabelToMaskD)
UTILS.register('LabelMorphology', LabelMorphologyD)
UTILS.register('BoundingRect', BoundingRectD)
UTILS.register('ShiftIntensity', ShiftIntensityD)
UTILS.register('ThresholdIntensity', ThresholdIntensityD)
UTILS.register('MaskIntensity', MaskIntensityD)
UTILS.register('MaskIntensityEx', MaskIntensityExD)
UTILS.register('GaussianSmooth', GaussianSmoothD)
UTILS.register('GaussianSharpen', GaussianSharpenD)
