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
        data, _ = self.dataset[index]
        return data, index

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

class ImbalancedMNIST(Datasets):
    def __init__(self, root, train, img_size):
        super().__init__(img_size)
        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=self.transform,
        )
        self.train = train
        ratio = [7, 6, 5, 9, 3, 8, 2, 4, 1, 0]
        self.class_distr = np.array([0.7 ** ratio[i] for i in range(10)])
        self.class_distr /= np.sum(self.class_distr)
        self.idx = self.resample()

    def resample(self):
        targets = self.dataset.train_labels if self.train else self.dataset.test_labels
        _, class_counts = np.unique(targets, return_counts=True)
        # Get class indices for resampling
        class_indices = [np.where(targets == i)[0] for i in range(10)]
        # Get class indices for reduced class count
        idx = []
        for i in range(10):
            cur_count = int(class_counts[i] * self.class_distr[i])
            idx.append(class_indices[i][:cur_count])
        idx = np.hstack(idx)
        np.random.shuffle(idx)
        return idx

    def __getitem__(self, index):
        img, _ = self.dataset[self.idx[index]]
        return img, index

    def __len__(self):
        return len(self.idx)


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
