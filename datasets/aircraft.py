from torchvision import datasets
from torch.utils.data import ConcatDataset
from pathlib import Path
# path = './fgvc-aircraft-2013b/data/'


def Aircraft(path, transform):
    train_dataset=datasets.FGVCAircraft(path, split='trainval', transform=transform, download=True)
    test_dataset=datasets.FGVCAircraft(path, split='test', transform=transform)
    dataset=ConcatDataset([train_dataset, test_dataset])
    return dataset

