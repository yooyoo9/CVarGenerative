import numpy as np
import argparse
import matplotlib.pyplot as plt

from util.figure import *
    
titles = ["True samples", "Usual GAN", "Rockafellar GAN", "CVaR-GAN"]
models = ["true", "vae", "rocka", "ada"]
    
def generate_samples(path, reverse):
    set_figure_params(serif=False, fontsize=9)
    fig, axes = plt.subplots(ncols=2, nrows=2)
    axes = axes.flatten()
    fig.set_size_inches(6.75, 6.0)
    for k in range(4):
        cur, ax, title = models[k], axes[k], titles[k] 
        path_out = path + "recons_" + cur + ".npy" 
        img = np.load(path_out)
        ax.set_title(title)
        if reverse:
            img = 1 - img
        ax.imshow(img)
        hide_all_ticks(ax) 
    adapt_figure_size_from_axes(axes)
    fig.tight_layout()
    plt.savefig(path + "reconstructions.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist", choices=set(("mnist", "cifar10", "celeba")))
    parser.add_argument("--path", type=str, default="../output/")
    args = parser.parse_args()
    args.path += args.dataset + "/"
    reverse = True if args.dataset == "mnist" else False
    generate_samples(args.path, reverse)
