# pytorch dataloader class for loading the CelebA dataset
import torch
from torch.utils.data import Dataset
from PIL import Image

class CelebADataset(Dataset):
    """Dataset class for loading the CelebA dataset."""
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    # __getitem__ for one idx
    def __get_one_item__(self, idx):
        img_name = self.data[idx]
        image = Image.open(img_name)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    # __getitem__ for slices and wrap around
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self.__get_one_item__(i) for i in range(start, stop, step)]
        elif isinstance(idx, int):
            if idx < 0:
                idx = len(self) + idx
            if idx >= len(self):
                idx = idx % len(self)
            return self.__get_one_item__(idx)
        else:
            raise TypeError('Invalid argument type:', type(idx))