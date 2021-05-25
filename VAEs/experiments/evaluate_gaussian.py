import numpy as np
import os
import torch
import matplotlib
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture

seed = 31415
np.random.seed(seed)
torch.manual_seed(seed)


def evaluate_output(idx, vae, rocka, ada, device, alpha, dataset):
    names = {
        "usual": "VAE",
        "rocka": "Rockafellar VAE",
        "ada": "CVar-VAE",
    }
    path_out = "../output/out_gaussian/"

    true_data = dataset.data
    n_clusters = int(dataset.n_clusters)
    k_cvar = int(np.round(alpha * len(true_data)))
    output_file = open(path_out + "output" + str(idx) + ".txt", "w")

    vae.eval()
    rocka.eval()
    ada.eval()
    with torch.no_grad():
        recons = vae.sample(true_data.shape[0], device)
        recons_rocka = rocka.sample(true_data.shape[0], device)
        recons_ada = ada.sample(true_data.shape[0], device)

    gmm = GaussianMixture(
        n_components=n_clusters,
        n_init=20,
        random_state=seed,
    )
    data = [true_data, recons, recons_rocka, recons_ada]
    np.save(path_out + "data" + str(idx) + "_predictions.npy", np.array(data))
    name = ["True data", "VAE", "Rockafellar VAE", "CVaR-VAE"]
    ll_true = np.empty(4)
    ll_cur = np.empty(4)
    for i in range(4):
        gmm.fit(data[i])
        cur = -gmm.score_samples(true_data)
        ll_true[i] = np.mean(cur)
        cur = -gmm.score_samples(data[i])
        ll_cur[i] = np.mean(cur)
    output_file.write("Mean NLL of the true-data given reconstruction\n")
    for i in range(4):
        output_file.write(f"{name[i]}: {ll_true[i]:.2f}\n")
    output_file.write("\n")
    output_file.write("NLL of data set\n")
    for i in range(4):
        output_file.write(f"{name[i]}: {ll_cur[i]:.2f}\n")
    output_file.close()

    fig = plt.figure(figsize=[8, 3])
    ax1 = plt.subplot(141)
    ax2 = plt.subplot(142, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(143, sharex=ax1, sharey=ax1)
    ax4 = plt.subplot(144, sharex=ax1, sharey=ax1)
    recons = recons.cpu()
    recons_rocka = recons_rocka.cpu()
    recons_ada = recons_ada.cpu()
    fontsize = 18
    ax1.scatter(true_data[:, 0], true_data[:, 1], s=10, color="black")
    ax1.set_title("Validation data", {"fontsize": fontsize})
    # ax1.axis("equal")
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax2.scatter(recons[:, 0], recons[:, 1], s=10, color="black")
    ax2.set_title(names["usual"], {"fontsize": fontsize})
    # ax2.axis("equal")
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax3.scatter(recons_rocka[:, 0], recons_rocka[:, 1], s=10, color="black")
    ax3.set_title(names["rocka"], {"fontsize": fontsize})
    # ax3.axis("equal")
    ax3.xaxis.set_visible(False)
    ax3.yaxis.set_visible(False)
    ax4.scatter(recons_ada[:, 0], recons_ada[:, 1], s=10, color="black")
    ax4.set_title(names["ada"], {"fontsize": fontsize})
    # ax4.axis("equal")
    ax4.xaxis.set_visible(False)
    ax4.yaxis.set_visible(False)
    fig.tight_layout()
    plt.savefig(path_out + "output" + str(idx) + ".png")
    plt.clf()
