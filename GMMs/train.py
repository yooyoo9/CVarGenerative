import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture

from cvar_em import CVarEM
from generate_data import generate_data

np.random.seed(31415)

param = {
    "alpha": 0.3,
    "lr_hedge": 0.1,
    "n_samples": 400,
    "n_init": 50,
    "n_init_cvar": 50,
    "dir": ["data", "output"],
    "path_X": "data/data_X.npy",
    "path_y": "data/data_y.npy",
    "path_out": "output/",
}

# Create directories for the output if they do not exist
for cur_dir in param["dir"]:
    if not os.path.exists(cur_dir):
        os.makedirs(cur_dir)

# Check if data already present, if not generate
if not os.path.isfile(param["path_X"]):
    generate_data(param["n_samples"], param["path_X"], param["path_y"])
X = np.load(param["path_X"])
y = (np.load(param["path_y"])).astype(int)

fig = plt.figure(figsize=[10, 6])
for i in range(len(X)):
    print(i)
    curX = X[i, :-1]
    cur_y = y[i]
    n_clusters = int(X[i, -1, 0])

    cvar = CVarEM(
        n_components=n_clusters,
        n_init=param["n_init_cvar"],
        num_actions=param["n_samples"],
        size=int(np.ceil(param["alpha"] * param["n_samples"])),
        lr=param["lr_hedge"],
    )

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        tol=1e-3,
        max_iter=100,
        n_init=param["n_init"],
        init_params="kmeans",
    )

    gmm_y = gmm.fit_predict(curX)
    cvar_y, cvar_loss, cvar_weight = cvar.fit_predict(curX)

    # Plot the result
    colors = np.array(
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
    )
    ax = plt.subplot(1, 3, 1)
    ax.set_title("True distribution")
    ax.scatter(curX[:, 0], curX[:, 1], s=10, color=colors[cur_y])
    ax1 = plt.subplot(1, 3, 2)
    ax1.set_title("GMM")
    ax1.scatter(curX[:, 0], curX[:, 1], s=10, color=colors[gmm_y])
    ax2 = plt.subplot(1, 3, 3)
    ax2.set_title("CVaR_EM")
    ax2.scatter(curX[:, 0], curX[:, 1], s=10, color=colors[cvar_y])
    plt.savefig(param["path_out"] + "data" + str(i) + "_img.png")
    plt.clf()

    plt.plot(cvar_loss)
    plt.savefig(param["path_out"] + "data" + str(i) + "_loss.png")
    plt.title("Cvar loss")
    plt.clf()
    plt.plot(cvar_weight)
    plt.savefig(param["path_out"] + "data" + str(i) + "_weight.png")
    plt.title("Probabilities of datapoints")
    plt.clf()
