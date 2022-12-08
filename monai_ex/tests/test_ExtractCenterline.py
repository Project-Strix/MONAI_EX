import pytest
from monai.data import Dataset
from monai_ex.transforms import Compose, ExtractCenterlineD, LoadImageD, EnsureChannelFirstD


def test_extractcenterline():
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
            )
        ])
    )
    for dataset in source_dataset:
        print(dataset)
