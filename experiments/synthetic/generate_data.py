import os
import numpy as np
import argparse
from sklearn import datasets

sample_distr = [
    [0.01, 0.99],
    [0.78, 0.05, 0.05, 0.05, 0.05, 0.01, 0.01],
    [0.02, 0.28, 0.7],
    [0.49, 0.49, 0.005, 0.01, 0.005],
    [0.92, 0.02, 0.02, 0.02, 0.02],
]
std = [
    [0.01, 0.7],
    [0.08, 0.02, 0.02, 0.02, 0.02, 0.002, 0.002],
    [0.02, 0.2, 0.2],
    [1, 1, 0.01, 0.01, 0.01],
    [1, 0.1, 0.2, 0.1, 0.2],
]
centers = [
    [[0, -5], [0, -2]],
    [
        [0, 0],
        [-0.3, -0.4],
        [-0.3, 0.4],
        [0.3, 0.4],
        [0.3, -0.4],
        [0, 0.7],
        [0, -0.7],
    ],
    [[-1, 5.1], [-1, 4.2], [-0.2, 5]],
    [[-12, 0], [12, 0], [0, -6], [0, 6], [0, 0]],
    [[0, 0], [8.2, 2.6], [4.2, 4.9], [8.2, -2.6], [4.2, -4.9]],
]


def generate_data(n):
    path_x = os.path.join("input", "X" + str(n) + ".npy")
    path_y = os.path.join("input", "Y" + str(n) + ".npy")

    m = len(sample_distr)
    data = np.empty((2 * m, n + 1, 2))
    data_y = np.empty((2 * m, n))
    for i in range(m):
        cur_distr = [np.round(np.array(sample_distr[i]) * n).astype(int), n]
        cur_std = np.array(std[i])
        cur_centers = np.array(centers[i])

        for j in range(2):
            x_cur, y_cur = datasets.make_blobs(
                n_samples=cur_distr[j], centers=cur_centers, cluster_std=cur_std
            )

            x_cur -= x_cur.mean()
            x_cur /= x_cur.std()

            data[2 * i + j, :-1] = x_cur
            data[2 * i + j, -1] = [len(centers[i]), 0]
            data_y[2 * i + j] = y_cur
    np.save(path_x, data)
    np.save(path_y, data_y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n", type=int, help="number of samples per dataset", default=10000
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    generate_data(args.n)
