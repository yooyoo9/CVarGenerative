import os
import argparse

from util.figure import *

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


def get_model_title(model):
    if model == 'vae':
        titles = ["True samples", "VAE", "Trunc-VAE", "AdaCVaR-VAE"]
        models = ["true", "vae", "trunc", "ada"]
    else:
        titles = ["True samples", "GAN", "Trunc-GAN", "AdaCVaR-GAN"]
        models = ["true", "gan", "trunc", "ada"]
    return models, titles


def synthetic_gmm(dataset):
    cur = np.load("experiments/synthetic/gmm/data" + str(dataset) + "_predictions.npy")
    data, true_y, gmm_y, cvar_y = cur[:, :-3], cur[:, -3], cur[:, -2], cur[:, -1]
    true_y, gmm_y, cvar_y = true_y.astype(int), gmm_y.astype(int), cvar_y.astype(int)

    set_figure_params(serif=False, fontsize=9)
    fig, (ax, ax1, ax2) = plt.subplots(ncols=3, nrows=1)
    fig.set_size_inches(6.75, 2.0)

    ax.set_title("True distribution")
    ax.axis("equal")
    ax.scatter(data[:, 0], data[:, 1], s=1, color=colors[true_y])
    hide_all_ticks(ax)
    ax1.set_title("EM")
    ax1.axis("equal")
    ax1.scatter(data[:, 0], data[:, 1], s=1, color=colors[gmm_y])
    hide_all_ticks(ax1)
    ax2.set_title("CVaR-EM")
    ax2.axis("equal")
    ax2.scatter(data[:, 0], data[:, 1], s=1, color=colors[cvar_y])
    hide_all_ticks(ax2)
    adapt_figure_size_from_axes((ax, ax1, ax2))
    fig.tight_layout()
    plt.savefig('experiments/synthetic/gmm/output' + str(dataset) + '_gmm.png')
    plt.close()


def synthetic(model, dataset):
    models, titles = get_model_title(model)
    path = os.path.join('VAEs/output/gaussian/' + str(dataset))
    set_figure_params(serif=False, fontsize=9)
    fig, axes = plt.subplots(ncols=4, nrows=1, sharex=True, sharey=True)
    fig.set_size_inches(6.75, 2.0)
    for i in range(4):
        if i == 0:
            data = np.load('experiments/synthetic/input/X10000.npy')
            input_data = data[:-1]
            np.random.shuffle(input_data)
            n_val = int(len(input_data) * 0.7) + int(len(input_data) * 0.2)
            recons = input_data[n_val:]
        else:
            recons = np.load(os.path.join(path, models[i] + '.npy')).cpu()
        ax, title = axes[i], titles[i]
        ax.scatter(recons[:, 0], recons[:, 1], s=1, color="black")
        ax.set_title(title)
        hide_all_ticks(ax)
    adapt_figure_size_from_axes(axes)
    fig.tight_layout()
    plt.savefig('experiments/synthetic/output/output' + str(dataset) + '_' + model + '.png')
    plt.close()


def plot_manifolds(model, dataset):
    """Plots n x n digit images decoded from the latent space."""
    models, titles = get_model_title(model)
    set_figure_params(serif=False, fontsize=9)
    fig, axes = plt.subplots(ncols=3, nrows=1)
    fig.set_size_inches(6.75, 3.0)
    for k in range(1, 4):
        cur, ax, title = models[k], axes[k - 1], titles[k]
        path_out = os.path.join('experiments', dataset, model, 'manifold_' + cur + '.npy')
        img = np.load(path_out)
        ax.set_title(title)
        ax.imshow(img, cmap="Greys")
        hide_all_ticks(ax)
    adapt_figure_size_from_axes(axes)
    fig.tight_layout()
    plt.savefig(os.path.join('experiments', dataset, model, 'manifold.png'))
    plt.close()


def generate_samples(model, dataset):
    models, titles = get_model_title(model)
    set_figure_params(serif=False, fontsize=9)
    fig, axes = plt.subplots(ncols=2, nrows=2)
    axes = axes.flatten()
    fig.set_size_inches(6.75, 6.0)
    for k in range(4):
        cur, ax, title = models[k], axes[k], titles[k]
        path_out = os.join.path('experiments', dataset, model, 'recons_' + cur + '.npy')
        img = np.load(path_out)
        ax.set_title(title)
        if dataset == 'mnist' or dataset == 'mnist_imb':
            img = 1 - img
        ax.imshow(img)
        hide_all_ticks(ax)
    adapt_figure_size_from_axes(axes)
    fig.tight_layout()
    plt.savefig(os.join.path('experiments', dataset, model, 'reconstructions.png'))
    plt.close()


if __name__ == "__main__":
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gaussian", choices=set(("gaussian", "mnist", "mnist_imb", "cifar10", "celeba")))
    args = parser.parse_args()
    args.path += os.path.join(args.path, args.dataset)
    reverse = False

    # Generate manifolds
    if args.dataset == "gaussian":
        for i in range(10):
            synthetic_gmm(i)
            synthetic('vae', i)
            synthetic('gan', i)
    else:
        if args.dataset == 'mnist' or args.dataset == 'mnist_imb':
            plot_manifolds('vae', args.dataset)
            plot_manifolds('gan', args.dataset)
        generate_samples('vae', args.dataset)
        generate_samples('gan', args.dataset)

