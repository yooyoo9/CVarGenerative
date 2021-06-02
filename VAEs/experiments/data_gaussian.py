import numpy as np
import argparse
from sklearn import datasets

import torch
from torch.utils.data import Dataset

class GaussianDataSet(Dataset):
    def __init__(self, path, idx, train):
        data = np.load(path)[idx]
        input_data = data[:-1]
        self.n_clusters = data[-1, 0]

        # Normalize the data
        input_data = input_data - input_data.mean(axis=0)
        input_data = input_data / input_data.std(axis=0)

        n_train = int(len(input_data) * 0.8)
        if train:
            # Training data
            self.data = input_data[:n_train]
        else:
            # Validation data
            self.data = input_data[n_train:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cur = torch.tensor(self.data[idx]).type("torch.FloatTensor")
        return cur, idx


def generate_data(n, path):
    sample_distr = [
        [0.01, 0.99],
        [0.01, 0.19, 0.8],
        [0.45, 0.45, 0.05, 0.05],
        [0.92, 0.02, 0.02, 0.02, 0.02],
        [0.6, 0.1, 0.1, 0.1, 0.1],
    ]

    std = [
        [0.01, 0.7],
        [0.02, 0.2, 0.2],
        [1, 1, 0.01, 0.01],
        [1, 0.1, 0.2, 0.1, 0.2],
        [0.1, 0.1, 0.1, 0.1, 0.1],
    ]

    centers = [
        [[0, -5.5], [0, -2]],
        [[-1.2, 5.2], [-1, 4.2], [-0.2, 5]],
        [[-12, 0], [12, 0], [0, -6], [0, 6]],
        [[0, 0], [8.2, 2.6], [4.2, 4.9], [8.2, -2.6], [4.2, -4.9]],
        [[0, 0], [-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]],
    ]

    m = len(sample_distr)
    x = np.empty((m, n + 1, 2))
    for i in range(m):
        cur_distr = np.round(np.array(sample_distr[i]) * n).astype(int)
        cur_std = np.array(std[i])
        cur_centers = np.array(centers[i])

        x_cur, _ = datasets.make_blobs(
            n_samples=cur_distr, centers=cur_centers, cluster_std=cur_std
        )

        x[i, :n] = x_cur
        x[i, -1] = np.array([len(centers[i]), 0])
    np.save(path, x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--number_generated_samples", type=int, default=1000)
    parser.add_argument("--path", type=str, default="../input/gaussian/data.npy")
    args = parser.parse_args()

    generate_data(args.number_generated_samples, args.path)
