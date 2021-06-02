import matplotlib.pyplot as plt
import numpy as np

from figure import *

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

for i in range(5):
    cur = np.load("output/data" + str(i) + "_predictions.npy")
    true_y = cur[0]
    gmm_y = cur[1]
    cvar_y = cur[2]
    data = X[i, :-1]

    set_figure_params(serif=False, fontsize=9)
    fig, (ax, ax1, ax2) = plt.subplots(ncols=3, nrows=1)
    fig.set_size_inches(6.75 / 2, 2.0)
    ax.set_title("True distribution")
    ax.axis("equal")
    ax.scatter(data[:, 0], data[:, 1], s=10, color=colors[true_y])
    hide_all_ticks(ax)
    ax1.set_title("Usual EM")
    ax1.axis("equal")
    ax1.scatter(data[:, 0], data[:, 1], s=10, color=colors[gmm_y])
    hide_all_ticks(ax1)
    ax2.set_title("CVaR-EM")
    ax2.axis("equal")
    ax2.scatter(data[:, 0], data[:, 1], s=10, color=colors[cvar_y])
    hide_all_ticks(ax2)
    adapt_figure_size_from_axes((ax, ax1, ax2))
    fig.tight_layout()
    plt.savefig("output/data" + str(i) + "_img.png")
    plt.clf()
