import pytest
from monai.data import Dataset
import nibabel as nib
from monai_ex.transforms import RandDropD, Compose, ExtractCenterlineD, LoadImageD, EnsureChannelFirstD
from monai_ex.transforms import RandDrop, ExtractCenterline

@pytest.mark.parametrize(
    argnames="roi_number,roi_size", argvalues=[(10, 10)]
)
def test_randdrop(roi_number, roi_size):
    # image = nib.load("/homes/clwang/Data/Vessels/BrainMRA1/Normal025-MRA.nii.gz")
    # label = nib.load("/MRIData/ITKTubeTK/Database/Normal-025/AuxillaryData/VascularNetwork.nii.gz")
    # centerline = ExtractCenterline()(label.get_fdata())
    # output = RandDrop(roi_number=5, roi_size=10)(image.get_fdata(), centerline)
    # nib.save(nib.Nifti1Image(output, affine=image.affine), filename='/homes/dxli/Code/Strix/MONAI_EX/monai_ex/tests/output.nii.gz')
    source_dataset = Dataset(
        [{
            "image": "/homes/clwang/Data/Vessels/BrainMRA1/Normal025-MRA.nii.gz",
            "label": "/MRIData/ITKTubeTK/Database/Normal-025/AuxillaryData/VascularNetwork.nii.gz"
        }],
        transform=Compose([
            LoadImageD(['image', 'label']),
            EnsureChannelFirstD(['image', 'label']),
            ExtractCenterlineD(
                keys=['label'],
                output_key='centerline'
            ),
            RandDropD(
                keys=['image'],
                cline_key='centerline',
                roi_number=roi_number,
                roi_size=roi_size,
            )
        ])
    )
    for dataset in source_dataset:
        print(type(dataset))
