from figure import *

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

for i in range(10):
    cur = np.load("output/data" + str(i) + "_predictions.npy")
    data, true_y, gmm_y, cvar_y = cur[:, :-3], cur[:, -3], cur[:, -2], cur[:, -1]
    true_y, gmm_y, cvar_y = true_y.astype(int), gmm_y.astype(int), cvar_y.astype(int)

    set_figure_params(serif=False, fontsize=9)
    fig, (ax, ax1, ax2) = plt.subplots(ncols=3, nrows=1)
    fig.set_size_inches(6.75 / 2, 2.0)
    ax.set_title("True distribution")
    ax.axis("equal")
    ax.scatter(data[:, 0], data[:, 1], s=1, color=colors[true_y])
    hide_all_ticks(ax)
    ax1.set_title("Usual EM")
    ax1.axis("equal")
    ax1.scatter(data[:, 0], data[:, 1], s=1, color=colors[gmm_y])
    hide_all_ticks(ax1)
    ax2.set_title("CVaR-EM")
    ax2.axis("equal")
    ax2.scatter(data[:, 0], data[:, 1], s=1, color=colors[cvar_y])
    hide_all_ticks(ax2)
    adapt_figure_size_from_axes((ax, ax1, ax2))
    fig.tight_layout()
    plt.savefig("output/data" + str(i) + "_img.png")
    plt.clf()
