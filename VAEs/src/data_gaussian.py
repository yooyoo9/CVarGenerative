import numpy as np
from sklearn import cluster, datasets
np.random.seed(31415)

n = 5
n_train, n_val = 5000, 500
def generate(n, n_train, n_val):
    X_train = np.empty((n, n_train, 2))
    X_val = np.empty((n, n_val, 2))
    for i in range(n):
        # k = np.random.randint(2, 7)
        k = 4
        std = 0.2 * np.random.rand(k)
    
        cur_train, _ = datasets.make_blobs(n_samples = n_train, centers = k,
                                           cluster_std = std)
        cur_val, _ = datasets.make_blobs(n_samples = n_val, centers = k,
                                         cluster_std = std)
        cur_train = (cur_train / np.max(np.abs(cur_train)) + 1.0) * 0.5
        cur_val = (cur_val / np.max(np.abs(cur_val)) + 1.0) * 0.5

        X_train[i], X_val[i] = cur_train, cur_val
    return X_train, X_val

X_train, X_val = generate(n, n_train, n_val)
np.save('../data_train.npy', X_train)
np.save('../data_val.npy', X_val)

