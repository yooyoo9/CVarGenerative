import numpy as np
import os
import torch
import argparse
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture

from data_gaussian import GaussianDataSet
from util.figure import *

seed = 31415
np.random.seed(seed)
torch.manual_seed(seed)

models = ["true", "vae", "rocka", "ada"]
name = ["True data", "VAE", "Rockafellar", "CVaR"]

def evaluate(dataset, idx, alpha, path_model, path_out):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_file = open(path_out + "output" + idx + ".txt", "w")
    true_data = dataset.data
    n_clusters = int(dataset.n_clusters)
    k_cvar = int(np.round(alpha * len(true_data)))

    gmm = GaussianMixture(
        n_components=n_clusters,
        n_init=20,
        random_state=seed,
    )
    recons = [true_data]
    for i in range(1, 4):
        cur = models[i]
        model = torch.load(
            path_model + cur, map_location=torch.device("cpu")
        ).to(device)
        model.eval()
        with torch.no_grad():
            data = model.sample(true_data.shape[0], device)
        recons.append(data)
    np.save(path_out + "recons" + idx + ".npy", np.array(recons))

    ll_true = np.empty(4)
    ll_cur = np.empty(4)
    for i in range(4):
        gmm.fit(recons[i])
        ll_true[i] = np.mean(-gmm.score_samples(true_data))
        ll_cur[i] = np.mean(-gmm.score_samples(recons[i]))

    output_file.write("Mean NLL of the true-data given reconstruction\n")
    for i in range(4):
        output_file.write(f"{name[i]}: {ll_true[i]:.2f}\n")
    output_file.write("\n")
    output_file.write("NLL of data set\n")
    for i in range(4):
        output_file.write(f"{name[i]}: {ll_cur[i]:.2f}\n")
    output_file.close()

    set_figure_params(serif=False, fontsize=9)
    fig, axes = plt.subplots(ncols=4, nrows=1, sharex=True, sharey=True)
    fig.set_size_inches(6.75/2, 2.0)
    for i in range(4):
        cur_rec, ax, title = recons[i], axes[i], name[i]
        if i != 0:
            cur_rec = cur_rec.cpu()
        ax.scatter(cur_rec[:, 0], cur_rec[:, 1], s=1, color="black")
        ax.set_title(title)
        hide_all_ticks(ax)
    adapt_figure_size_from_axes(axes)
    fig.tight_layout()
    plt.savefig(path_out + "output" + idx + ".png")
    plt.show()
    plt.close()

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=int, default=0, choices=set((0,1,2,3,4)))
parser.add_argument("--alpha", type=float, default=0.3)
parser.add_argument("--path_data", type=str, default="../input/gaussian/data.npy")
parser.add_argument("--path_model", type=str, default="../models/gaussian/")
parser.add_argument("--path_out", type=str, default="../output/gaussian/")
args = parser.parse_args()

args.path_model += str(args.dataset)
dataset = GaussianDataSet(args.path_data, args.dataset, train=False)

evaluate(
    dataset,
    str(args.dataset),
    args.alpha,
    args.path_model,
    args.path_out,
)
    
