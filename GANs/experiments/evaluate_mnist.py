import numpy as np
import os
import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torchvision.utils import save_image
from scipy.stats import norm

from mnist_classifier import MnistClassifier
from inception_score import inception_score

seed = 31415
np.random.seed(seed)
torch.manual_seed(seed)


def generate_classes(model, device, path_out):
    model.eval()
    with torch.no_grad():
        data = model.sample(5000, device)
    classifier = MnistClassifier()
    pred_labels = classifier.predict(data)
    data = data.cpu().numpy().reshape(-1, 1, 28, 28)
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

def evaluate():
    # name_out = "mod" if imbalanced else "usual"
    # dataset = ImbalancedMNIST if imbalanced else datasets.MNIST
    name_out = "usual"
    dataset = datasets.MNIST

    param = {
        "path_data": "../input/mnist/",
        "path_model": "../models/mnist_" + name_out + "/",
        "path_out": "../output/out_mnist_" + name_out + "/",
        "path_true": "../output/out_mnist_" + name_out + "/true_data/",
    }

    names = {
        "gan": "Usual GAN",
        "ada": "AdaCVaR GAN",
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_generated_images = 5000
    n_inception = 1000
    output_file = open(param["path_out"] + "output.txt", "w")

    # Generate true images
    if not os.path.exists(param["path_true"]):
        os.makedirs(param["path_true"])
    valid_set = dataset(
        root=param["path_data"],
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    true_ratio = np.zeros(10)
    idxs = np.random.choice(len(valid_set), n_generated_images, False)
    for i in idxs:
        save_image(valid_set[i][0], param["path_true"] + str(i) + ".png")
        true_ratio[valid_set[i][1]] += 1.0
    true_ratio = (np.round(true_ratio)).astype(int)
    output_file.write("True class ratio: ")
    np.savetxt(output_file, true_ratio, newline=" ", fmt="%4d")
    output_file.write("\n\n")

    for cur in ["gan", "ada"]:
        output_file.write(names[cur] + "\n")
        model_G = torch.load(
            param["path_model"] + cur + "G", map_location=torch.device("cpu")
        ).to(device)
        path_out_data = param["path_out"] + cur + "_data/"
        path_out = param["path_out"] + cur

        loss_D = np.load(param["path_out"]+"/loss_" + cur + "_D.npy")
        loss_G = np.load(param["path_out"]+"/loss_" + cur + "_G.npy")
        fig = plt.figure()
        x = np.arange(1, len(loss_D)+1)
        plt.plot(x, loss_D, label="loss D")
        plt.plot(x, loss_G, label="loss G")
        plt.xlabel("Number of epochs")
        plt.ylabel("Loss")
        plt.savefig(param["path_out"] + "/losses_" + cur)

        model_G.eval()
        class_ratio = generate_classes(model_G, device, path_out)
        output_file.write("Predicted class ratio: ")
        np.savetxt(output_file, class_ratio, newline=" ", fmt="%4d")
        output_file.write("\n")

        # Generate reconstructed images
        if not os.path.exists(path_out_data):
            os.makedirs(path_out_data)
        with torch.no_grad():
            for i in range(n_generated_images//1000):
                rec_data = model_G.sample(1000, device).view(-1, 1, 28, 28)
                for j in range(1000):
                    save_image(rec_data[j], path_out_data + str(1000*i+j) + ".png")

        with torch.no_grad():
            rec_data = model_G.sample(n_inception, device).cpu().numpy().reshape(-1, 1, 28, 28)
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
    evaluate()
