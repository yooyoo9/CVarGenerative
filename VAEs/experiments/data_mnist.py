import numpy as np

from torch.utils.data import Dataset
from torchvision import datasets


class ImbalancedMNIST(Dataset):
    def __init__(self, root, train, download, transform):
        self.dataset = datasets.MNIST(root=root, download=download, transform=transform)
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
        img, target = self.dataset[self.idx[index]]
        return img, target

    def __len__(self):
        return len(self.idx)
