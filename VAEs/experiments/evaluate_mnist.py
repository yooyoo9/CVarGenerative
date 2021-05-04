import numpy as np
import os
import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torchvision.utils import save_image
from scipy.stats import norm

from util.mnist_classifier import MnistClassifier
from data_mnist import ImbalancedMNIST
from inception_score import inception_score

seed = 31415
np.random.seed(seed)
torch.manual_seed(seed)


def generate_classes(model, device, path_out):
    model.eval()
    with torch.no_grad():
        data = model.sample(1000, device)
    classifier = MnistClassifier()
    pred_labels = classifier.predict(data)
    data = data.cpu().numpy()
    recons = np.empty((100, 1, 28, 28))
    classes = np.empty(10)
    for j in range(10):
        idx = np.argwhere(pred_labels == j)[:, 0]
        classes[j] = len(idx)
        cur = data[idx[:10]]
        if len(cur) < 10:
            cur = np.concatenate((cur, np.zeros((10 - len(cur), 1, 28, 28))))
        recons[10 * j: 10 * (j + 1)] = cur
    img = torch.from_numpy(recons).view(100, 1, 28, 28)
    save_image(img, path_out + ".png", nrow=10)
    return classes


def plot_manifold(model, device, path_out):
    """Plots n x n digit images decoded from the latent space."""
    n = 20
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    image_width = image_height = 28 * n
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
            image[28 * i: 28 * (i + 1), 28 * j: 28 * (j + 1)] = (
                digit.detach().cpu().numpy()
            )
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap="Greys_r")
    plt.axis("Off")
    plt.savefig(path_out + "_manifold.png")
    plt.close()


def evaluate(imbalanced):
    name_out = "mod" if imbalanced else "usual"
    dataset = ImbalancedMNIST if imbalanced else datasets.MNIST

    param = {
        "path_data": "../input/mnist/",
        "path_model": "../models/mnist_" + name_out + "/",
        "path_out": "../output/out_mnist_" + name_out + "/",
        "path_true": "../output/out_mnist_" + name_out + "/true_data/",
    }

    names = {
        "vae": "Usual VAE",
        "rockar": "Rockarfellar VAE",
        "ada": "AdaCVaR VAE",
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = 1000
    output_file = open(param["path_out"] + "output.txt", "w")

    if not os.path.exists(param["path_true"]):
        # Generate true images
        os.makedirs(param["path_true"])
        valid_set = dataset(
            root=param["path_data"],
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        true_ratio = np.zeros(10)
        idxs = np.random.choice(len(valid_set), n, False)
        for i in idxs:
            save_image(valid_set[i][0], param["path_true"] + str(i) + ".png")
            true_ratio[valid_set[i][1]] += 1
        output_file.write("True class ratio: ")
        np.savetxt(output_file, true_ratio, newline=" ", fmt="%4d")
        output_file.write("\n\n")

    for cur in ["vae", "rockar", "ada"]:
        output_file.write(names[cur] + "\n")
        model = torch.load(
            param["path_model"] + cur, map_location=torch.device("cpu")
        ).to(device)
        path_out_data = param["path_out"] + cur + "_data/"
        path_out = param["path_out"] + cur

        model.eval()
        class_ratio = generate_classes(model, device, path_out)
        output_file.write("Predicted class ratio: ")
        np.savetxt(output_file, class_ratio, newline=" ", fmt="%4d")
        output_file.write("\n")

        plot_manifold(model, device, path_out)

        if not os.path.exists(path_out_data):
            # Generate reconstructed images
            os.makedirs(path_out_data)
            with torch.no_grad():
                rec_data = model.sample(n, device)
            for i in range(n):
                save_image(rec_data[i], path_out_data + str(i) + ".png")

        with torch.no_grad():
            rec_data = model.sample(n, device).cpu().numpy()
        rec_data = np.repeat(rec_data, 3, 1)
        score, _ = inception_score(
            torch.from_numpy(rec_data).type(torch.FloatTensor),
            batch_size=32,
            resize=True,
            splits=10,
        )
        output_file.write("Inception score: {:.2f} \n".format(score))
        output_file.write("\n")
    output_file.close()


if __name__ == "__main__":
    evaluate(imbalanced=False)
    evaluate(imbalanced=True)
