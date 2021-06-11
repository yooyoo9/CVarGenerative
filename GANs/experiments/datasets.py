import numpy as np

from torch.utils.data import Dataset
from torchvision import datasets, transforms

class Datasets(Dataset):
    def __init__(self, img_size):
        self.transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
            ]
        )
        self.dataset = None

    def __getitem__(self, index):
        data, label = self.dataset[index]
        return data, index, label

    def __len__(self):
        return len(self.dataset)


class MNIST(Datasets):
    def __init__(self, root, train, img_size):
        super().__init__(img_size)
        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=self.transform,
        )

class CelebA(Datasets):
    def __init__(self, root, train, img_size):
        super().__init__(img_size)
        split = "train" if train else "valid"
        self.dataset = datasets.CelebA(
            root=root,
            split=split,
            download=True,
            transform=self.transform,
        )

class CIFAR10(Datasets):
    def __init__(self, root, train, img_size):
        super().__init__(img_size)
        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=True,
            transform=self.transform,
        )
