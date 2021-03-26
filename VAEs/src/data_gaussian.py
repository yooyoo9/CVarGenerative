import numpy as np
from sklearn import cluster, datasets
np.random.seed(31415)

n_train, n_val = 1000, 100
n_samples = 500

def generate(n, n_samples):
    data = np.empty((n, n_samples, 2))
    for i in range(n):
        if i%100 == 0:
            print(i)
        k = np.random.randint(1, 4)
        
        std = 2 * np.random.rand(k)
        X, Y = datasets.make_blobs(n_samples = n_samples, centers=k, cluster_std=std)
        X = (X / np.max(np.abs(X)) + 1.0) * 0.5
        # X = X[np.argsort( np.round(X[:, 0]*10) * 10 + np.round(X[:, 1]*10) )]
        data[i] = X
    return data

train = generate(n_train, n_samples)
val = generate(n_val, n_samples)
np.save("../data_train.npy", train)
np.save("../data_val.npy", val)

