import pytest
from monai.data import Dataset
import nibabel as nib
from monai_ex.transforms import RandDropD, Compose, ExtractCenterlineD, LoadImageD, EnsureChannelFirstD, GenerateSyntheticDataD


@pytest.mark.parametrize(
    argnames="roi_number,roi_size", argvalues=[(30, 5), (30, 7), (30, 10)]
)
def test_randdrop(roi_number, roi_size):
    # dim = 3
    # spatial_size = (100,) * dim

    # generator = GenerateSyntheticDataD(
    #     ["image", "label"],
    #     *spatial_size,
    #     num_objs=1,
    #     rad_max=5,
    #     rad_min=4,
    #     noise_max=0,
    #     num_seg_classes=1,
    #     channel_dim=0,
    # )

    # outputs = generator({"image": "1", "label": "1"})
    # cropper = ExtractCenterlineD(keys=["label"], output_key='centerline')

    # centerline = cropper(outputs)
    # converter = RandDropD(
    #     keys=['image'],
    #     cline_key='centerline',
    #     roi_number=roi_number,
    #     roi_size=roi_size
    # )
    # output = converter(centerline)
    original_file = nib.load("/homes/clwang/Data/Vessels/BrainMRA1/Normal025-MRA.nii.gz")
    original_image = original_file.get_fdata()
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
                roi_key='centerline',
                roi_number=roi_number,
                roi_size=roi_size,
            )
        ])
    )

    for dataset in source_dataset:
        assert dataset['image'].squeeze().all() == original_image.all()
        nib.save(nib.Nifti1Image(dataset['image'].squeeze(), affine=original_file.affine), 'test_img.nii.gz')
    # output_item = source_dataset[0]
    # assert label.get_fdata().all() == output.all()
