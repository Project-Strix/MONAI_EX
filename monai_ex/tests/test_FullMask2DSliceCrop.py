#%% 
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch
from monai_ex.transforms.croppad.array import FullMask2DSliceCrop
from monai_ex.transforms.croppad.dictionary import FullMask2DSliceCropD
from torchvision.utils import make_grid


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')


# image_path = "/homes/yliu/Data/clwang_data/esophageal_cancer_nii/esophageal_cancer_nii/sg_vibe_1mm/668597/092_radial_selfgated_vibe_1mm_high_phase.nii.gz"
# mask_path = "/homes/yliu/Data/clwang_data/esophageal_cancer_nii/esophageal_cancer_nii/sg_vibe_1mm/668597/092_radial_selfgated_vibe_1mm_high phase_ROI.nii.gz"

# image_nii = nib.load(image_path)
# mask_nii = nib.load(mask_path)


# cropper = FullMask2DSliceCrop(roi_size=(32,32), crop_mode='parallel', z_axis=2, mask_data=mask_nii.get_fdata()[np.newaxis, ...], n_slices=3)
# crops = cropper(image_nii.get_fdata()[np.newaxis, ...])

# print(f'{len(crops)} crops')

# show(make_grid([torch.Tensor(crop) for crop in crops], nrow=7, normalize=True))
# %%

image_path = "/homes/yliu/Data/clwang_data/esophageal_cancer_nii/esophageal_cancer_nii/sg_vibe_1mm/668597/092_radial_selfgated_vibe_1mm_high_phase.nii.gz"
mask_path = "/homes/yliu/Data/clwang_data/esophageal_cancer_nii/esophageal_cancer_nii/sg_vibe_1mm/668597/092_radial_selfgated_vibe_1mm_high phase_ROI.nii.gz"

image_nii = nib.load(image_path)
mask_nii = nib.load(mask_path)

data_list = []
data_list.append({'image': image_nii.get_fdata()[np.newaxis,...], 'mask': mask_nii.get_fdata()[np.newaxis,...]})

cropper = FullMask2DSliceCropD(keys='image', mask_key='mask', roi_size=(32,32), crop_mode='single', z_axis=2)
crops = cropper(data_list[0])
print(crops)