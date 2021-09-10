import numpy as np
import argparse
import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from scipy.stats import norm

from util.mnist_classifier import MnistClassifier
from util.inception_score import inception_score
from util.datasets import MNIST, ImbalancedMNIST, CelebA, CIFAR10


def generate_loss_graph(loss_D, loss_G, path):
    fig = plt.figure()
    x = np.arange(1, len(loss_D)+1)
    plt.plot(x, loss_D, label="loss D")
    plt.plot(x, loss_G, label="loss G")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path)

    
def generate_reconstructions(path, true_data, nb, device, n_channel, img_size):
    path_true = path["path_out"] + "reconstructions/true_data/"
    if not os.path.exists(path_true):
        os.makedirs(path_true)
    for k in range(2):
        cur_path = path_true + str(k) + "/"
        nb = min(nb, len(true_data))
        if not os.path.exists(cur_path):
            os.makedirs(cur_path)
        idxs = np.random.choice(len(true_data), nb, False)
        for i in idxs:
            save_image(true_data[i][0], cur_path + str(i) + ".png")
            
    for cur in ['orig', 'trunc', 'ada']:
        path_out_data = path["path_out"] + "reconstructions/" + cur + "_data/"
        if not os.path.exists(path_out_data):
            os.makedirs(path_out_data)
        model = torch.load(
            path["path_model"] + cur, map_location=torch.device("cpu")
        ).to(device)
        for k in range(2):
            cur_path = path_out_data + str(k) + "/"
            if not os.path.exists(cur_path):
                os.makedirs(cur_path)
            with torch.no_grad():
                for i in range(nb//128):
                    rec_data = model.sample(128, device).view(-1, n_channel, img_size, img_size)
                    rec_data = transforms.Normalize(0.5, 0.5)(rec_data)
                    for j in range(128):
                        save_image(rec_data[j], cur_path + str(128*i+j) + ".png", normalize=True)

def plot_manifolds(path, device):
    """Plots n x n digit images decoded from the latent space."""
    n = 20
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    image_width = image_height = 28 * n
    for cur in ["vae", "rocka", "ada"]:
        model = torch.load(
            path["path_model"]+cur, map_location=torch.device("cpu")
        ).to(device)
        path_out = path["path_out"] + "manifold_" + cur + ".npy"
        image = np.zeros((image_height, image_width))
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z = (
                    torch.from_numpy(np.array([[xi, yi]]))
                    .type(torch.FloatTensor)
                    .to(device)
                )
                x_decoded = model.decode(z)
                digit = torch.reshape(x_decoded[0], (28, 28))
                image[28 * i : 28 * (i + 1), 28 * j : 28 * (j + 1)] = (
                    digit.detach().cpu().numpy()
                )
        np.save(path_out, image)


def calculate_class_ratios(path, true_data, device):
    classifier = MnistClassifier()
    ratios = dict()
    for cur in ["true", "orig", "trunc", "ada"]:
        path_out = path["path_out"] + "recons_" + cur + ".npy"
        cur_ratio = np.empty((10, 10))
        for k in range(10):
            if cur == "true":
                data, _ = next(iter(DataLoader(true_data, 1000, shuffle=True)))
            else:
                model = torch.load(
                    path["path_model"]+cur, map_location=torch.device("cpu")
                ).to(device)

                with torch.no_grad():
                    data = model.sample(1000, device)
            pred_labels = classifier.predict(data)
            for j in range(10):
                idx = np.argwhere(pred_labels == j)[:, 0]
                cur_ratio[j, k] = len(idx)
        cur_ratio /= 10.0
        cur_mean = np.mean(cur_ratio, axis=1)
        cur_std = np.std(cur_ratio, axis=1)
        ratios[cur] = [cur_mean, cur_std]
    return ratios


def generate_samples_mnist(path, true_data, device):
    classifier = MnistClassifier()
    recons = np.empty((100, 1, 28, 28))
    for cur in ["true", "orig", "rocka", "ada"]:
        path_out = path["path_out"] + "recons_" + cur + ".npy"

        for j in range(10):
            cur_data = np.array([])
            while len(cur_data) != 10:
                if cur == "true":
                    data, _ = next(iter(DataLoader(true_data, 1000, shuffle=True)))
                else:
                    model = torch.load(
                        path["path_model"]+cur, map_location=torch.device("cpu")
                    ).to(device)

                    with torch.no_grad():
                        data = model.sample(1000, device)
                pred_labels = classifier.predict(data)
                data = data.cpu().numpy().reshape(-1, 1, 28, 28)
                idx = np.argwhere(pred_labels == j)[:, 0]
                if len(cur_data) == 0:
                    cur_data = data[idx[:10]]
                else:
                    cur_data = np.concatenate((cur_data, data[idx[:10-len(cur_data)]]))
            recons[10 * j: 10 * (j + 1)] = cur_data

        img = torch.from_numpy(recons).view(100, 1, 28, 28)
        img = np.transpose(make_grid(img.to(device), nrow=10, padding=5, normalize=True).cpu(), (1,2,0))
        np.save(path_out, img)


def generate_samples(path, true_data, device):
    for cur in ["true", "orig", "rocka", "ada"]:
        path_out = path["path_out"] + "recons_" + cur + ".npy"
        if cur == "true":
            data = next(iter(DataLoader(true_data, 64)))[0]
        else:
            model = torch.load(
                path["path_model"] + cur, map_location=torch.device("cpu")
            ).to(device)
            with torch.no_grad():
                data = model.sample(64, device)
        
        img = np.transpose(make_grid(data, padding=5, normalize=True).cpu(),(1,2,0))
        np.save(path_out, img)


def calculate_inception_score(path, nb, device, n_channel, img_size):
    scores = dict()
    for cur in ["orig", "rocka", "ada"]:
        model = torch.load(
            path["path_model"] + cur, map_location=torch.device("cpu")
        ).to(device)
        cur_score = np.empty(10)
        for i in range(10):
            rec_data = torch.tensor([])
            with torch.no_grad():
                for _ in range(nb//128):
                    cur_data = model.sample(128, device)
                    rec_data = torch.cat((rec_data, cur_data.cpu()), dim=0)
            rec_data = rec_data.numpy().reshape(-1, n_channel, img_size, img_size)
            if n_channel != 3:
                rec_data = np.repeat(rec_data, 3, 1)
            cur_score[i], _ = inception_score(
                torch.from_numpy(rec_data).type(torch.FloatTensor),
                batch_size=32,
                resize=True,
                splits=10,
            )
        scores[cur] = [np.mean(cur_score), np.std(cur_score)]
    return scores


def evaluate(model, dataset, name, path, nb_recons, n_inception, n_channel, img_size, loss_graph):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_file = open(path["path_out"] + "output.txt", "w")

    if model == 'gan' and loss_graph:
        for cur in ["gan", "rocka", "ada"]:
            loss_D = np.load(path["path_out"]+"lossD_" + cur + ".npy")
            loss_G = np.load(path["path_out"]+"lossG_" + cur + ".npy")
            path_loss = path["path_out"] + "/losses_" + cur
            generate_loss_graph(loss_D, loss_G, path_loss)

    # Generate reconstructions
    generate_reconstructions(
        path,
        dataset,
        nb_recons,
        device,
        n_channel,
        img_size,
    )

    # Generate manifolds
    if name == "mnist" or name == "mnist_imb":
        plot_manifolds(path, device)

    # Generate samples
    if name == "mnist" or name == "mnist_imb":
        generate_samples_mnist(path, dataset, device)

        output_file.write("Class ratios: \n")
        output_file.write("Mean: \n")
        ratios = calculate_class_ratios(path, dataset, device)
        for key, val in ratios.items():
            output_file.write(key + ": \t")
            np.savetxt(output_file, val[0], newline="\t", fmt="%.2f")
            output_file.write("\n")
        output_file.write("Std: \n")
        for key, val in ratios.items():
            output_file.write(key + ": \t")
            np.savetxt(output_file, val[1], newline=" ", fmt="%.2f")
            output_file.write("\n")
        output_file.write("\n")
    else:
        generate_samples(path, dataset, device)

    # Calculate inception score
    scores = calculate_inception_score(path, n_inception, device, n_channel, img_size)
    output_file.write("Inception scores\n")
    for key, val in scores.items():
        output_file.write("{}: {:.2f} +- {:.2f}\n".format(key, val[0], val[1]))
    output_file.write("\n")

    output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vae", choices=set(('vae', "gan")))
    parser.add_argument("--dataset", default="mnist", choices=set(('gaussian', "mnist", "mnist_imb", "cifar", "celeba")))
    parser.add_argument("--number_generated_imgs", type=int, default=5000)
    parser.add_argument("--nb_inception", type=int, default=1000)
    parser.add_argument("--loss_graph", action='store_true', default=False)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    path = {
        "path_data": os.path.join('experiments', args.dataset, 'input'),
        "path_model": os.path.join('experiments', args.dataset, args.model, 'model'),
        "path_out": os.path.join('experiments', args.dataset, args.model, 'output')
    }

    if args.dataset == "mnist" or args.dataset == "mnist_imb":
        img_size, n_channel = 28, 1
    else:
        img_size, n_channel = 64, 3

    if args.dataset == "cifar":
        dataset = CIFAR10
    elif args.dataset == "celeba":
        dataset = CelebA
    elif args.dataset == "mnist":
        dataset = MNIST
    else:
        dataset = ImbalancedMNIST
    true_data = dataset(
        root=args.path_data,
        train=False,
        img_size=img_size
    )

    evaluate(
        args.model,
        true_data,
        args.dataset,
        path,
        args.number_generated_imgs,
        args.nb_inception,
        n_channel,
        img_size,
        args.loss_graph
    )
