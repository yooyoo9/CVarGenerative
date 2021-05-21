import numpy as np
import matplotlib.pyplot as plt

X = np.load("data/data_X.npy")
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

fontsize = 16
for i in range(5):
    cur = np.load("output/data" + str(i) + "_predictions.npy")
    true_y = cur[0]
    gmm_y = cur[1]
    cvar_y = cur[2]
    data = X[i, :-1]

    fig = plt.figure(figsize=[6, 3])
    ax = plt.subplot(1, 3, 1)
    ax.set_title("True distribution", fontsize=fontsize)
    ax.axis("equal")
    ax.scatter(data[:, 0], data[:, 1], s=10, color=colors[true_y])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax1 = plt.subplot(1, 3, 2)
    ax1.set_title("Usual EM", fontsize=fontsize)
    ax1.axis("equal")
    ax1.scatter(data[:, 0], data[:, 1], s=10, color=colors[gmm_y])
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax2 = plt.subplot(1, 3, 3)
    ax2.set_title("CVaR-EM", fontsize=fontsize)
    ax2.axis("equal")
    ax2.scatter(data[:, 0], data[:, 1], s=10, color=colors[cvar_y])
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    fig.tight_layout()
    plt.savefig("output/data" + str(i) + "_img.png")
    plt.clf()
