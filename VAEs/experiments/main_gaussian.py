import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from sklearn import cluster, datasets

from train import VAEalg, CVaRalg

seed = 764003779
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
model_param = {
    "x_dim": 2,
    "hidden_dims": [100],
    "z_dim": 16,
    "beta": 0.0,
    "constrained_output": False,
}

param = {
    "epochs": 100,
    "batch_size": 200,
    "lr": 1e-4,
    "alpha": 0.3,
    "print": False,
    "save_model": True,
    "nb": 100,  # number of datasets
    "data_size": 1000,
    "dir": ["../models/", "../output/out_gaussian/", "input/gaussian/"],
    "path_data": "../input/gaussian/data.npy",
    "path_vae": "../models/gaussian/vae",
    "path_cvar": "../models/gaussian/cvar",
    "path_out": "../output/out_gaussian/",
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
        k = np.random.randint(4, 10)
        sample_distr = np.zeros(k)
        while sum(sample_distr) != param["data_size"]:
            sample_distr = np.random.dirichlet(np.random.rand(k), 1)[0]
            sample_distr = np.round(param["data_size"] * sample_distr).astype(int)
        std = 10 * np.random.rand(k)
        centers = 10 * np.random.rand(k, 2)
        X[i], _ = datasets.make_blobs(
            n_samples=sample_distr, centers=centers, cluster_std=std
        )
    np.save(param["path_data"], X)


fig = plt.figure(figsize=[15, 6])
# for i in range(param["nb"]):
for i in [0, 2, 13, 31, 33, 69, 72, 82]:
    print(f"Dataset {i} of {param['nb']}")
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
    vae.train(param["epochs"], param["save_model"], param["print"])

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
    cvar.train(param["epochs"], param["save_model"], param["print"])

    # Compare usual VAE with CVaR VAE
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)
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
