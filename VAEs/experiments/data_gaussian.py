import numpy as np
from sklearn import datasets


def generate_data(n, path):
    x = np.empty((3, n, 2))

    sample_distr = np.round(np.array([0.9, 0.1]) * n).astype(int)
    std = np.array([0.2, 0.1])
    centers = np.array([[0, 0], [1, 0]])
    x[0], _ = datasets.make_blobs(
        n_samples=sample_distr, centers=centers, cluster_std=std
    )

    sample_distr = np.round(np.array([0.6, 0.1, 0.1, 0.1, 0.1]) * n).astype(int)
    std = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    centers = np.array([[0, 0], [-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]])
    x[1], _ = datasets.make_blobs(
        n_samples=sample_distr, centers=centers, cluster_std=std
    )

    sample_distr = np.round(np.array([0.1, 0.3, 0.6]) * n).astype(int)
    std = np.array([0.1, 0.2, 0.3])
    centers = np.array([[-1, 5], [-1, 2.5], [0.3, 5]])
    x[2], _ = datasets.make_blobs(
        n_samples=sample_distr, centers=centers, cluster_std=std
    )
    np.save(path, x)
