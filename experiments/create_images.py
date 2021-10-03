import os
import argparse
from scipy import linalg
import matplotlib as mpl

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
    ]
)

std = np.array([0.08, 0.02, 0.02, 0.02, 0.02, 0.002, 0.002])
centers = np.array([
        [0, 0],
        [-0.3, -0.4],
        [-0.3, 0.4],
        [0.3, 0.4],
        [0.3, -0.4],
        [0, 0.7],
        [0, -0.7],
    ])


def plot_ellipse(mean, covar, color='red', ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    v, w = linalg.eigh(covar)
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    u = w[0] / linalg.norm(w[0])

    # Plot an ellipse to show the Gaussian component
    angle = np.arctan(u[1] / u[0])
    angle = 180. * angle / np.pi  # convert to degrees

    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(0.2)
    ax.add_artist(ell)


def get_model_title(model):
    if model == 'vae':
        titles = ["True data", "VAE", "TruncRAVAE", "AdaRAVAE"]
        models = ["true", "orig", "trunc", "ada"]
    else:
        titles = ["True data", "GAN", "TruncRAGAN", "AdaRAGAN"]
        models = ["true", "orig", "trunc", "ada"]
    return models, titles


def synthetic_gmm(dataset):
    cur = np.load("experiments/synthetic/gmm/data" + str(dataset) + "_predictions.npy")
    means = np.load("experiments/synthetic/gmm/data" + str(dataset) + "_means.npy")
    covs = np.load("experiments/synthetic/gmm/data" + str(dataset) + "_covariances.npy")
    data, true_y, gmm_y, cvar_y = cur[:, :-3], cur[:, -3], cur[:, -2], cur[:, -1]
    true_y, gmm_y, cvar_y = true_y.astype(int), gmm_y.astype(int), cvar_y.astype(int)

    set_figure_params(serif=True, fontsize=8)
    fig, (ax, ax1, ax2) = plt.subplots(ncols=3, nrows=1, gridspec_kw = {'wspace':0, 'hspace':0})
    fig.set_size_inches(5.5, 1)

    ax.set_title("True data")
    ax.axis("equal")
    for i in range(7):
        ax.scatter(data[:, 1][true_y==i], data[:, 0][true_y==i], s=0.1, c=colors[i])
        cur_cov = np.eye(2) * std[i]
        plot_ellipse(np.flip(means[1][i]), cur_cov, colors[i], ax)
    hide_all_ticks(ax)
    ax1.set_title("EM")
    ax1.axis("equal")
    for i in range(7):
        print(means[1][i])
        ax1.scatter(data[:, 1][gmm_y == i], data[:, 0][gmm_y == i], s=0.1, c=colors[i])
        plot_ellipse(np.flip(means[0][i]), covs[0][i], colors[i], ax1)
    hide_all_ticks(ax1)
    ax2.set_title("HAdaCVaR-EM")
    ax2.axis("equal")
    for i in range(7):
        ax2.scatter(data[:, 1][cvar_y == i], data[:, 0][cvar_y == i], s=0.1
                    , c=colors[i])
        plot_ellipse(np.flip(means[1][i]), covs[1][i], colors[i], ax2)
    hide_all_ticks(ax2)
    adapt_figure_size_from_axes((ax, ax1, ax2))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.8, bottom=0.05)
    plt.savefig('experiments/synthetic/gmm/output' + str(dataset) + '_gmm.png', dpi=300)
    plt.close()


def synthetic(model, dataset):
    models, titles = get_model_title(model)
    path = os.path.join('experiments/synthetic', model, 'output', str(dataset))
    set_figure_params(serif=True, fontsize=8)
    fig, axes = plt.subplots(ncols=3, nrows=1, gridspec_kw={'wspace': 0, 'hspace': 0})
    fig.set_size_inches(5.5/2.0, 1)
    for i in range(1, 4):
        if i == 0:
            data = np.load('experiments/synthetic/input/X10000.npy')
            input_data = data[dataset][:-1]
            np.random.shuffle(input_data)
            n_val = int(len(input_data) * 0.7) + int(len(input_data) * 0.2)
            recons = input_data[n_val:]
        else:
            recons = np.load(os.path.join(path, models[i] + '.npy'))
        ax, title = axes[i-1], titles[i]
        ax.scatter(recons[:, 1], recons[:, 0], s=0.1, color="black")
        ax.axis("equal")
        ax.set_title(title)
        hide_all_ticks(ax)
    adapt_figure_size_from_axes(axes)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.8, bottom=0.05)
    save_path = os.path.join('experiments/synthetic', model, 'output', 'synthetic' + str(dataset-1) + '.png')
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_manifolds(model, dataset):
    """Plots n x n digit images decoded from the latent space."""
    models, titles = get_model_title(model)
    set_figure_params(serif=True, fontsize=9)
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
    set_figure_params(serif=True, fontsize=8)
    for k in range(4):
        ax = plt.figure()
        ax.set_size_inches(5.5 / 2.0, 5.5 / 2.0)
        cur, title = models[k], titles[k]
        path_out = os.path.join('experiments', dataset, model, 'recons_' + cur + '.npy')
        img = np.load(path_out)
        plt.imshow(img)
        hide_all_ticks(plt)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        plt.savefig(os.path.join('experiments', dataset, model, dataset + '_' + cur + '.png'), dpi=300)
        plt.close()


if __name__ == "__main__":
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gaussian")
    args = parser.parse_args()

    # Generate manifolds
    if args.dataset == "gaussian":
        for i in [2,3]:
            synthetic_gmm(i)
            synthetic('vae', i)
            synthetic('gan', i)
    else:
        if args.dataset == 'mnist' or args.dataset == 'mnist_imb':
            plot_manifolds('vae', args.dataset)
        generate_samples('vae', args.dataset)
        generate_samples('gan', args.dataset)

