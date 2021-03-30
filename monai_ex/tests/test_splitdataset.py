from monai.transforms.utility.array import DataStats
from monai_ex.data import SplitDataset, DataLoader
from numpy.lib.npyio import load

datasets = list(range(11))

def test_splitdataset(shuffle):
    data = SplitDataset(data=datasets, shuffle=shuffle)
    for d in data:
        print(d)
    print()

def test_splitdataloader(shuffle):
    data = SplitDataset(data=datasets, shuffle=shuffle)
    loader = DataLoader(data, batch_size=5, num_workers=2)
    for d in loader:
        print(d)
    print()

if __name__ == "__main__":
    test_splitdataset(shuffle=False)
    # test_splitdataset(shuffle=True)

    test_splitdataloader(shuffle=False)
