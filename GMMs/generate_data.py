import numpy as np
from sklearn import datasets


def generate_data(n, path_x, path_y):
    sample_distr = [
        np.round(np.array([0.01, 0.99]) * n).astype(int),
        np.round(np.array([0.78, 0.05, 0.05, 0.05, 0.05, 0.01, 0.01]) * n).astype(int),
        np.round(np.array([0.02, 0.28, 0.7]) * n).astype(int),
        np.round(np.array([0.49, 0.49, 0.005, 0.01, 0.005]) * n).astype(int),
        np.round(np.array([0.92, 0.02, 0.02, 0.02, 0.02]) * n).astype(int),
    ]
    std = [
        np.array([0.01, 0.7]),
        np.array([0.08, 0.02, 0.02, 0.02, 0.02, 0.002, 0.002]),
        np.array([0.02, 0.2, 0.2]),
        np.array([1, 1, 0.01, 0.01, 0.01]),
        np.array([1, 0.1, 0.2, 0.1, 0.2]),
    ]
    centers = [
        np.array([[0, -5.5], [0, -2]]),
        np.array(
            [
                [0, 0],
                [-0.3, -0.4],
                [-0.3, 0.4],
                [0.3, 0.4],
                [0.3, -0.4],
                [0, 0.7],
                [0, -0.7],
            ]
        ),
        np.array([[-1, 5.1], [-1, 4.2], [-0.2, 5]]),
        np.array([[-12, 0], [12, 0], [0, -6], [0, 6], [0, 0]]),
        np.array([[0, 0], [8.2, 2.6], [4.2, 4.9], [8.2, -2.6], [4.2, -4.9]]),
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
