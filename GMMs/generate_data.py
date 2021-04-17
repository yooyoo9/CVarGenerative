import numpy as np
from sklearn import datasets


def generate_data(n, path_x, path_y):
    sample_distr = [
        np.round(np.array([0.9, 0.1]) * n).astype(int),
        np.round(np.array([0.7, 0.05, 0.05, 0.1, 0.1]) * n).astype(int),
        np.round(np.array([0.02, 0.28, 0.7]) * n).astype(int),
    ]
    std = [
        np.array([0.15, 0.05]),
        np.array([0.175, 0.1, 0.1, 0.1, 0.1]),
        np.array([0.1, 0.2, 0.3]),
    ]
    centers = [
        np.array([[0, 0], [0.5, 0]]),
        np.array([[0, 0], [-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]]),
        np.array([[-1, 5.1], [-1, 4.2], [0, 5]]),
    ]

    m = len(sample_distr)
    data = np.empty((m, n + 1, 2))
    data_y = np.empty((m, n))
    for i in range(m):
        x_cur, y_cur = datasets.make_blobs(
            n_samples=sample_distr[i], centers=centers[i], cluster_std=std[i]
        )

        # Normalize the data
        x_cur -= x_cur.mean()
        x_cur /= x_cur.std()

        data[i, :-1] = x_cur
        data[i, -1] = [len(centers[i]), 0]
        data_y[i] = y_cur
    np.save(path_x, data)
    np.save(path_y, data_y)
