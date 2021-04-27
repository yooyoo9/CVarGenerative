import numpy as np
from sklearn import cluster, datasets

import matplotlib.pyplot as plt


def generate_data(n, path):
    X = np.empty((3, n, 2))

    sample_distr = np.round(np.array([0.9, 0.1]) * n).astype(int)
    std = np.array([0.2, 0.1])
    centers = np.array([[0, 0], [1, 0]])
    X[0], _ = datasets.make_blobs(
        n_samples=sample_distr, centers=centers, cluster_std=std
    )

    sample_distr = np.round(np.array([0.6, 0.1, 0.1, 0.1, 0.1]) * n).astype(int)
    std = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    centers = np.array([[0, 0], [-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]])
    X[1], _ = datasets.make_blobs(
        n_samples=sample_distr, centers=centers, cluster_std=std
    )

    sample_distr = np.round(np.array([0.1, 0.3, 0.6]) * n).astype(int)
    std = np.array([0.1, 0.2, 0.3])
    centers = np.array([[-1, 5], [-1, 2.5], [0.3, 5]])
    X[2], _ = datasets.make_blobs(
        n_samples=sample_distr, centers=centers, cluster_std=std
    )
    np.save(path, X)
