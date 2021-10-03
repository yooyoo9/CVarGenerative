import numpy as np

import torch
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms


class Datasets(Dataset):
    def __init__(self, img_size, mean, std):
        self.transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.dataset = None

    def __getitem__(self, index):
        data, _ = self.dataset[index]
        return data, index

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def get_split(dataset, train):
        if train != 2:
            n_train = int(len(dataset) * 0.7)
            n_val = len(dataset) - n_train
            train_ds, val_ds = random_split(dataset, [n_train, n_val])
            return train_ds if train == 0 else val_ds
        else:
            return dataset


class GaussianDataSet(Dataset):
    def __init__(self, path, idx, train):
        data = np.load(path)[idx]
        input_data = data[:-1]
        self.n_clusters = data[-1, 0]

        # Normalize the data
        input_data = input_data - input_data.mean(axis=0)
        input_data = input_data / input_data.std(axis=0)
        np.random.shuffle(input_data)

        n_train = int(len(input_data) * 0.7)
        n_val = n_train + int(len(input_data) * 0.2)
        if train == 0:
            # Training data
            self.data = input_data[:n_train]
        elif train == 1:
            # Validation data
            self.data = input_data[n_train:n_val]
        else:
            # Test data
            self.data = input_data[n_val:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cur = torch.tensor(self.data[idx]).type("torch.FloatTensor")
        return cur, idx


class MNIST(Datasets):
    def __init__(self, root, train, img_size):
        mean, std = (0.1307,), (0.3081,)
        super().__init__(img_size, mean, std)
        dataset = datasets.MNIST(
            root=root,
            train=False if train == 2 else True,
            download=True,
            transform=self.transform,
        )
        self.dataset = self.get_split(dataset, train)


class ImbalancedMNIST(Datasets):
    def __init__(self, root, train, img_size):
        mean, std = (0.1307,), (0.3081,)
        super().__init__(img_size, mean, std)
        dataset = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=self.transform,
        )
        self.dataset = self.get_split(dataset, train)
        self.train = False if train == 2 else True
        ratio = [7, 6, 5, 9, 3, 8, 2, 4, 1, 0]
        self.class_distr = np.array([0.7 ** ratio[i] for i in range(10)])
        self.class_distr /= np.sum(self.class_distr)
        self.idx = self.resample()

    def resample(self):
        if self.train:
            targets = self.dataset.dataset.train_labels
        else:
            targets = self.dataset.test_labels
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
        if self.train:
            img, _ = self.dataset.dataset[self.idx[index]]
        else:
            img, _ = self.dataset[self.idx[index]]
        return img, index

    def __len__(self):
        return len(self.idx)


class CelebA(Datasets):
    def __init__(self, root, train, img_size):
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        super().__init__(img_size, mean, std)

        if train == 0:
            split = "train"
        elif train == 1:
            split = "valid"
        else:
            split = "test"

        self.dataset = datasets.CelebA(
            root=root,
            split=split,
            download=True,
            transform=self.transform,
        )


class CIFAR10(Datasets):
    def __init__(self, root, train, img_size):
        mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
        super().__init__(img_size, mean, std)
        dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=True,
            transform=self.transform,
        )
        self.dataset = self.get_split(dataset, train)


class ImbalancedCIFAR10(Datasets):
    def __init__(self, root, train, img_size):
        mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
        super().__init__(img_size, mean, std)
        dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=True,
            transform=self.transform,
        )
        self.dataset = self.get_split(dataset, train)
        self.train = False if train == 2 else True
        ratio = [7, 6, 5, 9, 3, 8, 2, 4, 1, 0]
        self.class_distr = np.array([0.7 ** ratio[i] for i in range(10)])
        self.class_distr /= np.sum(self.class_distr)
        self.idx = self.resample()

    def resample(self):
        if self.train:
            targets = self.dataset.dataset.targets
        else:
            targets = self.dataset.targets
        targets = np.array(targets)
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
        if self.train:
            img, _ = self.dataset.dataset[self.idx[index]]
        else:
            img, _ = self.dataset[self.idx[index]]
        return img, index

    def __len__(self):
        return len(self.idx)
