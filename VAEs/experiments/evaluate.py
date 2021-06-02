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
from datasets import MNIST, ImbalancedMNIST, CelebA, CIFAR10

seed = 31415
np.random.seed(seed)
torch.manual_seed(seed)
    
def generate_reconstructions(path, true_data, nb, device, n_channel, img_size):
    path_true = path["path_out"] + "true_data/"
    nb = min(nb, len(true_data))
    if not os.path.exists(path_true):
        os.makedirs(path_true)
    idxs = np.random.choice(len(true_data), nb, False)
    for i in idxs:
        save_image(true_data[i][0], path_true + str(i) + ".png")
    for cur in ["vae", "rocka", "ada"]:
        path_out_data = path["path_out"] + cur + "_data/"
        model_G = torch.load(
            path["path_model"] + cur, map_location=torch.device("cpu")
        ).to(device)
        if not os.path.exists(path_out_data):
            os.makedirs(path_out_data)
        with torch.no_grad():
            for i in range(nb//128):
                rec_data = model_G.sample(128, device).view(-1, n_channel, img_size, img_size)
                rec_data = transforms.Normalize(0.5, 0.5)(rec_data)
                for j in range(128):
                    save_image(rec_data[j], path_out_data + str(128*i+j) + ".png", normalize=True)

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
    
def generate_samples_mnist(path, true_data, device):
    classifier = MnistClassifier()
    ratios = dict()
    for cur in ["true", "vae", "rocka", "ada"]:
        path_out = path["path_out"] + "recons_" + cur + ".npy"
        if cur == "true":
            data, _ = next(iter(DataLoader(true_data, 500)))
        else:
            model = torch.load(
                path["path_model"]+cur, map_location=torch.device("cpu")
            ).to(device)

            with torch.no_grad():
                data = model.sample(1000, device)
    
        pred_labels = classifier.predict(data)
        data = data.cpu().numpy().reshape(-1, 1, 28, 28)
        recons = np.empty((100, 1, 28, 28))
        cur_ratio = np.empty(10)
        for j in range(10):
            idx = np.argwhere(pred_labels == j)[:, 0]
            cur_ratio[j] = len(idx)
            cur_data = data[idx[:10]]
            if len(cur_data) < 10:
                cur_data = np.concatenate((cur_data, np.zeros((10 - len(cur_data), 1, 28, 28))))
            recons[10 * j: 10 * (j + 1)] = cur_data
        ratios[cur] = cur_ratio

        img = torch.from_numpy(recons).view(100, 1, 28, 28)
        img = np.transpose(make_grid(img.to(device), nrow=10, padding=5, normalize=True).cpu(), (1,2,0))
        np.save(path_out, img)
    return ratios

def generate_samples(path, true_data, device):
    for cur in ["true", "vae", "rocka", "ada"]:
        path_out = path["path_out"] + "recons_" + cur + ".npy"
        if cur == "true":
            dataloader = DataLoader(true_data, 64)
            data = next(iter(dataloader))[0]
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
    for cur in ["vae", "rocka", "ada"]:
        model_G = torch.load(
            path["path_model"] + cur, map_location=torch.device("cpu")
        ).to(device)
        rec_data = torch.tensor([])
        with torch.no_grad():
            for _ in range(nb//128):
                cur_data = model_G.sample(128, device)
                rec_data = torch.cat((rec_data, cur_data.cpu()), dim=0)
        rec_data = rec_data.numpy().reshape(-1, n_channel, img_size, img_size)
        if n_channel != 3:
            rec_data = np.repeat(rec_data, 3, 1)
        scores[cur], _ = inception_score(
            torch.from_numpy(rec_data).type(torch.FloatTensor),
            batch_size=32,
            resize=True,
            splits=10,
        )
    return scores

def evaluate(dataset, name, path, nb_recons, n_inception, n_channel, img_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_file = open(path["path_out"] + "output.txt", "w")

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
        output_file.write("Class ratios: \n")
        ratios = generate_samples_mnist(path, dataset, device)
        for key, val in ratios.items():
            output_file.write(key + ": ")
            np.savetxt(output_file, val, newline=" ", fmt="%4d")
            output_file.write("\n")
        output_file.write("\n")
    else:
        generate_samples(path, dataset, device)


    # Calculate inception score
    scores = calculate_inception_score(path, n_inception, device, n_channel, img_size)
    output_file.write("Inception scores\n")
    for key, val in scores.items():
        output_file.write("{}: {:.2f} \n".format(key, val))
    output_file.write("\n")

    output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist", choices=set(("mnist", "mnist_imb", "cifar10", "celeba")))
    parser.add_argument("--number_generated_imgs", type=int, default=5000)
    parser.add_argument("--nb_inception", type=int, default=1000)
    parser.add_argument("--path_data", type=str, default="../input/")
    parser.add_argument("--path_model", type=str, default="../models/")
    parser.add_argument("--path_out", type=str, default="../output/")
    args = parser.parse_args()

    args.path_data += args.dataset + "/"
    args.path_model += args.dataset + "/"
    args.path_out += args.dataset + "/"
    path = {
        "path_data": args.path_data,
        "path_model": args.path_model,
        "path_out": args.path_out,
    }

    if args.dataset == "mnist" or args.dataset == "mnist_imb":
        img_size, n_channel = 28, 1
    else:
        img_size, n_channel = 64, 3

    if args.dataset == "cifar10":
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
        true_data,
        args.dataset,
        path,
        args.number_generated_imgs,
        args.nb_inception,
        n_channel,
        img_size,
    )
