import numpy as np
import argparse
import os
import torch
import subprocess

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from util.mnist_classifier import MnistClassifier
from util.inception_score import inception_score
from datasets import MNIST, ImbalancedMNIST, CelebA, CIFAR10, ImbalancedCIFAR10


def generate_reconstructions(path, true_data, nb, device, n_channel, img_size):
    path_true = os.path.join(path["path_out"], "reconstructions", "true_data")
    if not os.path.exists(path_true):
        os.makedirs(path_true)
    for k in range(2):
        cur_path = os.path.join(path_true, str(k))
        nb = min(nb, len(true_data))
        if not os.path.exists(cur_path):
            os.makedirs(cur_path)
        idxs = np.random.choice(len(true_data), nb, False)
        for i in idxs:
            save_image(true_data[i][0], os.path.join(cur_path, str(i) + ".png"))

    for cur in ["orig", "trunc", "ada"]:
        path_out_data = os.path.join(path["path_out"], "reconstructions", cur + "_data")
        if not os.path.exists(path_out_data):
            os.makedirs(path_out_data)
        model = torch.load(
            os.path.join(path["path_model"], cur), map_location=torch.device("cpu")
        ).to(device)
        for k in range(2):
            cur_path = os.path.join(path_out_data, str(k))
            if not os.path.exists(cur_path):
                os.makedirs(cur_path)
            with torch.no_grad():
                for i in range(nb // 128):
                    rec_data = model.sample(128, device).view(
                        -1, n_channel, img_size, img_size
                    )
                    rec_data = transforms.Normalize(0.5, 0.5)(rec_data)
                    for j in range(128):
                        save_image(
                            rec_data[j],
                            os.path.join(cur_path, str(128 * i + j) + ".png"),
                            normalize=True,
                        )


def calculate_class_ratios(path, true_data, device):
    classifier = MnistClassifier()
    ratios = dict()
    for cur in ["true", "orig", "trunc", "ada"]:
        path_out = os.path.join(path["path_out"], "recons_" + cur + ".npy")
        cur_ratio = np.empty((10, 10))
        for k in range(10):
            if cur == "true":
                data, _ = next(iter(DataLoader(true_data, 1000, shuffle=True)))
            else:
                model = torch.load(
                    os.path.join(path["path_model"], cur),
                    map_location=torch.device("cpu"),
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
    for cur in ["true", "orig", "trunc", "ada"]:
        path_out = os.path.join(path["path_out"], "recons_" + cur + ".npy")

        for j in range(10):
            cur_data = np.array([])
            while len(cur_data) != 10:
                if cur == "true":
                    data, _ = next(iter(DataLoader(true_data, 1000, shuffle=True)))
                else:
                    model = torch.load(
                        os.path.join(path["path_model"], cur),
                        map_location=torch.device("cpu"),
                    ).to(device)

                    with torch.no_grad():
                        data = model.sample(1000, device)
                pred_labels = classifier.predict(data)
                data = data.cpu().numpy().reshape(-1, 1, 28, 28)
                idx = np.argwhere(pred_labels == j)[:, 0]
                if len(cur_data) == 0:
                    cur_data = data[idx[:10]]
                else:
                    cur_data = np.concatenate(
                        (cur_data, data[idx[: 10 - len(cur_data)]])
                    )
            recons[10 * j : 10 * (j + 1)] = cur_data

        img = torch.from_numpy(recons).view(100, 1, 28, 28)
        img = np.transpose(
            make_grid(img.to(device), nrow=10, padding=5, normalize=True).cpu(),
            (1, 2, 0),
        )
        np.save(path_out, img)


def generate_samples(path, true_data, device):
    for cur in ["true", "orig", "trunc", "ada"]:
        path_out = os.path.join(path["path_out"], "recons_" + cur + ".npy")
        if cur == "true":
            data = next(iter(DataLoader(true_data, 64)))[0]
        else:
            model = torch.load(
                os.path.join(path["path_model"], cur), map_location=torch.device("cpu")
            ).to(device)
            with torch.no_grad():
                data = model.sample(64, device)

        img = np.transpose(make_grid(data, padding=5, normalize=True).cpu(), (1, 2, 0))
        np.save(path_out, img)


def calculate_inception_score(path, nb, device, n_channel, img_size):
    scores = dict()
    for cur in ["orig", "trunc", "ada"]:
        model = torch.load(
            os.path.join(path["path_model"], cur), map_location=torch.device("cpu")
        ).to(device)
        cur_score = np.empty(10)
        for i in range(10):
            rec_data = torch.tensor([])
            with torch.no_grad():
                for _ in range(nb // 128):
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


def calculate_fid_score(path):
    path_true = os.path.join(path["path_out"], "reconstructions", "true_data")
    scores = dict()
    for cur in ["orig", "trunc", "ada"]:
        path_out_data = os.path.join(path["path_out"], "reconstructions", cur + "_data")
        cur_scores = []
        for j in range(2):
            cur_path1 = os.path.join(path_out_data, str(j))
            for k in range(2):
                cur_path = os.path.join(path_true, str(k))
                res = subprocess.run(
                    ["python3", "-m", "pytorch_fid", cur_path, cur_path1],
                    stdout=subprocess.PIPE,
                )
                res = res.stdout.decode("utf-8")
                cur_ans = float(res[6:-1])
                cur_scores.append(cur_ans)
        cur_scores = np.array(cur_scores)
        scores[cur] = [np.mean(cur_scores), np.std(cur_scores)]
    return scores


def evaluate(dataset, name, path, nb_recons, n_inception, n_channel, img_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_file = open(os.path.join(path["path_out"], "output.txt"), "w")

    # Generate reconstructions
    generate_reconstructions(
        path,
        dataset,
        nb_recons,
        device,
        n_channel,
        img_size,
    )

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

    scores = calculate_fid_score(path)
    output_file.write("FID scores\n")
    for key, val in scores.items():
        output_file.write("{}: {:.2f} +- {:.2f}\n".format(key, val[0], val[1]))
    output_file.write("\n")
    output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--number_generated_imgs", type=int, default=5000)
    parser.add_argument("--nb_inception", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    path = {
        "path_data": os.path.join("experiments", args.dataset, "input"),
        "path_model": os.path.join("experiments", args.dataset, "gan", "model"),
        "path_out": os.path.join("experiments", args.dataset, "gan", "output"),
    }

    if args.dataset == "mnist" or args.dataset == "mnist_imb":
        img_size, n_channel = 28, 1
    else:
        img_size, n_channel = 64, 3

    if args.dataset == "cifar":
        dataset = CIFAR10
    elif args.dataset == "cifar_imb":
        dataset = ImbalancedCIFAR10
    elif args.dataset == "celeba":
        dataset = CelebA
    elif args.dataset == "mnist":
        dataset = MNIST
    else:
        dataset = ImbalancedMNIST
    true_data = dataset(root=path["path_data"], train=False, img_size=img_size)

    evaluate(
        true_data,
        args.dataset,
        path,
        args.number_generated_imgs,
        args.nb_inception,
        n_channel,
        img_size,
    )
