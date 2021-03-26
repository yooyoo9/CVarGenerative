import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from sklearn import cluster, datasets

from train import VAEalg, CVaRalg

seed = 31415
np.random.seed(seed)
torch.manual_seed(seed)


class GaussianDataSet(Dataset):
    def __init__(self, path, idx, train):
        data = np.load(path)[idx]

        # Normalize the data
        data = data - data.mean(axis=0)
        data = data / data.std(axis=0)

        n_train = int(len(data) * 0.8)
        if train:
            # Training data
            self.data = data[:n_train]
        else:
            # Validation data
            self.data = data[n_train:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cur = torch.tensor(self.data[idx]).type("torch.FloatTensor")
        return cur, idx


# learning params
model_param = {"x_dim": 2, "hidden_dims": [100], "z_dim": 16, "beta": 0.0}

param = {
    "epochs": 50,
    "batch_size": 200,
    "lr": 1e-4,
    "alpha": 0.3,
    "dir": ["../models/", "../output/out_gaussian/"],
    "path_data": "../data.npy",
    "path_vae": "../models/vae_gaussian",
    "path_cvar": "../models/vae_gaussian_cvar",
    "path_out": "../output/out_gaussian/",
    "save_model": False,
    "nb": 10,  # number of datasets
    "data_size": 5000,
}

criterion = torch.nn.MSELoss(reduction="none")

# Create directories for the output if they do not exist
for cur_dir in param["dir"]:
    if not os.path.exists(cur_dir):
        os.makedirs(cur_dir)

# Generate data
if not os.path.isfile(param["path_data"]):
    X = np.empty((param["nb"], param["data_size"], 2))
    for i in range(param["nb"]):
        k = np.random.randint(2, 7)
        std = np.random.rand(k)
        X[i], _ = datasets.make_blobs(
            n_samples=param["data_size"], centers=k, cluster_std=std
        )
    np.save(param["path_data"], X)

for i in range(param["nb"]):
    train_set = GaussianDataSet(param["path_data"], i, train=True)
    valid_set = GaussianDataSet(param["path_data"], i, train=False)

    vae = VAEalg(
        param["path_vae"] + str(i),
        model_param,
        train_set,
        valid_set,
        param["batch_size"],
        param["lr"],
        criterion,
    )
    vae.train(param["epochs"], param["save_model"])

    cvar = CVaRalg(
        param["path_cvar"] + str(i),
        model_param,
        train_set,
        valid_set,
        param["batch_size"],
        param["lr"],
        criterion,
        param["alpha"],
    )
    cvar.train(param["epochs"], param["save_model"])

    # Compare usual VAE with CVaR VAE
    fig = plt.figure()
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)
    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")

    vae.model.eval()
    cvar.model.eval()
    with torch.no_grad():
        for (data, idx) in vae.val_loader:
            data = data.to(vae.device)
            data = data.view(data.size(0), -1)
            recons, _, _ = vae.model(data)
            recons_cvar, _, _ = cvar.model(data)
            ax1.scatter(data[:, 0], data[:, 1], s=10, color="black")
            ax2.scatter(recons[:, 0], recons[:, 1], s=10, color="black")
            ax3.scatter(recons_cvar[:, 0], recons_cvar[:, 1], s=10, color="black")
    plt.savefig(param["path_out"] + "output" + str(i) + ".png")
    plt.clf()
