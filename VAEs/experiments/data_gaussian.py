import numpy as np
from sklearn import datasets


def generate_data(n, path):
    sample_distr = [
        [0.05, 0.95],
        [0.02, 0.28, 0.7],
        [0.35, 0.35, 0.1, 0.1, 0.1],
        [0.92, 0.02, 0.02, 0.02, 0.02],
        [0.6, 0.1, 0.1, 0.1, 0.1],
    ]

    std = [
        [0.01, 0.7],
        [0.02, 0.2, 0.2],
        [1, 1, 0.01, 0.01, 0.01],
        [1, 0.1, 0.2, 0.1, 0.2],
        [0.1, 0.1, 0.1, 0.1, 0.1],
    ]

    centers = [
        [[0, -5.5], [0, -2]],
        [[-1, 5.1], [-1, 4.2], [-0.2, 5]],
        [[-12, 0], [12, 0], [0, -6], [0, 6], [0, 0]],
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

        # Normalize the data
        x_cur -= x_cur.mean()
        x_cur /= x_cur.std()

        x[i, :n] = x_cur
        x[i, -1] = np.array([len(centers[i]), 0])
    np.save(path, x)
