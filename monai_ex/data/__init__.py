from monai.data import *
from .nifti_saver import NiftiSaverEx
from .register import DATASETYPE

DATASETYPE.register('Dataset', Dataset)
DATASETYPE.register('IterableDataset', IterableDataset)
DATASETYPE.register('PersistentDataset', PersistentDataset)
DATASETYPE.register('CacheNTransDataset', CacheNTransDataset)
DATASETYPE.register('CacheDataset', CacheDataset)
DATASETYPE.register('SmartCacheDataset', SmartCacheDataset)
DATASETYPE.register('ArrayDataset', ArrayDataset)
DATASETYPE.register('ZipDataset', ZipDataset)
DATASETYPE.register('PatchDataset', PatchDataset)
DATASETYPE.register('GridPatchDataset', GridPatchDataset)