import numpy as np
import argparse
import matplotlib.pyplot as plt

from util.figure import *
    
titles = ["True samples", "Usual VAE", "Rockafellar VAE", "CVaR-VAE"]
models = ["true", "vae", "rocka", "ada"]

def plot_manifolds(path):
    """Plots n x n digit images decoded from the latent space."""
    set_figure_params(serif=False, fontsize=9)
    fig, axes = plt.subplots(ncols=3, nrows=1)
    fig.set_size_inches(6.75, 3.0)
    for k in range(1, 4):
        cur, ax, title = models[k], axes[k-1], titles[k]
        path_out = path + "manifold_" + cur + ".npy"
        img = np.load(path_out)
        ax.set_title(title)
        ax.imshow(img, cmap="Greys_r")
        hide_all_ticks(ax)
    adapt_figure_size_from_axes(axes)
    fig.tight_layout()
    plt.savefig(path + "manifold.png")
    plt.close()
    
    
def generate_samples(path):
    set_figure_params(serif=False, fontsize=9)
    fig, axes = plt.subplots(ncols=2, nrows=2)
    axes = axes.flatten()
    fig.set_size_inches(6.75, 6.0)
    for k in range(4):
        cur, ax, title = models[k], axes[k], titles[k] 
        path_out = path + "recons_" + cur + ".npy" 
        img = np.load(path_out)
        ax.set_title(title)
        ax.imshow(img, cmap="Greys") 
        hide_all_ticks(ax) 
    adapt_figure_size_from_axes(axes)
    fig.tight_layout()
    plt.savefig(path + "reconstructions.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist", choices=set(("mnist", "mnist_imb", "cifar10", "celeba")))
    parser.add_argument("--path", type=str, default="../output/")
    args = parser.parse_args()
    args.path += args.dataset + "/"

    # Generate manifolds
    if args.dataset == "mnist" or args.dataset == "mnist_imb":
        plot_manifolds(args.path)

    # Generate reconstructions
    generate_samples(args.path)
