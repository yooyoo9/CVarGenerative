import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.mixture import GaussianMixture
from itertools import cycle, islice

from cvar_em import CVaR_EM

np.random.seed(31415)

param = {
    "alpha": 0.3,
    "epochs": 10,
    "n_samples": 200,
    "n": 10,
    "path_X": "../data_X.npy",
    "path_y": "../data_y.npy",
    "path_out": "../output/",
}


def generate(n, n_samples, pathX, pathy):
    """Generates the dataset and saves it as a numpy array.

    Parameters
    ----------
    n: int
        Number of datasets to generate
    n_samples: int
        Number of datapoints per dataset
    pathX: string
        Path to save the dataset
    pathy: string
        Path to save the clusters of the correponding datapoints
    """

    data = np.empty((n, n_samples + 1, 2))
    Y = np.empty((n, n_samples))
    for i in range(n):
        k = np.random.randint(2, 5)
        std = 2 * np.random.rand(k)
        X, y = datasets.make_blobs(n_samples=n_samples, centers=k, cluster_std=std)
        # Normalize the data
        X -= X.mean()
        X /= X.std()

        data[i, :-1] = X
        data[i, -1] = [k, 0]
        Y[i] = y
    np.save(pathX, data)
    np.save(pathy, Y)


# Check if data already present, if not generate
if not os.path.isfile(param["path_X"]):
    generate(param["n"], param["n_samples"], param["path_X"], param["path_y"])
X = np.load(param["path_X"])
y = (np.load(param["path_y"])).astype(int)

fig = plt.figure(figsize=[10, 6])
for i in range(param["n"]):
    curX = X[i, :-1]
    cury = y[i]
    n_clusters = int(X[i, -1, 0])

    cvar = CVaR_EM(
        n_components=n_clusters,
        epochs=param["epochs"],
        num_actions=param["n_samples"],
        size=int(np.ceil(param["alpha"] * param["n_samples"])),
        eta=np.sqrt(1 / param["alpha"] * np.log(1 / param["alpha"])),
    )

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        tol=1e-3,
        max_iter=100,
        n_init=10,
        init_params="kmeans",
    )

    cvar_y = cvar.fit_predict(curX)
    gmm_y = gmm.fit_predict(curX)

    colors = np.array(
        list(
            islice(
                cycle(
                    [
                        "#377eb8",
                        "#ff7f00",
                        "#4daf4a",
                        "#f781bf",
                        "#a65628",
                        "#984ea3",
                        "#999999",
                        "#e41a1c",
                        "#dede00",
                    ]
                ),
                int(max(max(cvar_y), max(gmm_y)) + 1),
            )
        )
    )
    # add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])

    # Plot the result
    ax = plt.subplot(1, 3, 1)
    ax.set_title("True distribution")
    ax.scatter(curX[:, 0], curX[:, 1], s=10, color=colors[cury])
    ax1 = plt.subplot(1, 3, 2)
    ax1.set_title("GMM")
    ax1.scatter(curX[:, 0], curX[:, 1], s=10, color=colors[gmm_y])
    ax2 = plt.subplot(1, 3, 3)
    ax2.set_title("CVaR_EM")
    ax2.scatter(curX[:, 0], curX[:, 1], s=10, color=colors[cvar_y])
    plt.savefig(param["path_out"] + str(i) + ".png")
    plt.clf()
