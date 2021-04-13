import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from sklearn import cluster, datasets

from cvarVAE.train import VAEalg, RockarfellarAlg, CVaRalg
from data import generate_data

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
    "hidden_dims": [100, 100],
    "z_dim": 16,
    "constrained_output": False,
}

param = {
    "epochs": 1000,
    "batch_size": 200,
    "lr": 1e-4,
    "alpha": 0.3,
    "beta_usual": 1,
    "beta_roc": 1,
    "print": True,
    "model_name": "VAE",
    "save_model": True,
    "nb": 1,  # number of datasets
    "data_size": 1000,
    "dir": ["../models/gaussian/", "../output/out_gaussian/", "../input/gaussian/"],
    "path_data": "../input/gaussian/one_gaussian.npy",
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
    generate_data(param["data_size"], param["path_data"])

fig = plt.figure(figsize=[15, 6])
for i in range(param["nb"]):
    print(f"Dataset {i} of {param['nb']}")
    train_set = GaussianDataSet(param["path_data"], i, train=True)
    valid_set = GaussianDataSet(param["path_data"], i, train=False)

    vae = VAEalg(
        param["model_name"],
        param["path_vae"] + str(i),
        model_param,
        train_set,
        valid_set,
        param["batch_size"],
        param["lr"],
        criterion,
        param["beta_usual"],
    )
    vae.train(param["epochs"], param["save_model"], param["print"])

    cvar = RockarfellarAlg(
        param["model_name"],
        param["path_cvar"] + str(i),
        model_param,
        train_set,
        valid_set,
        param["batch_size"],
        param["lr"],
        criterion,
        param["alpha"],
        param["beta_roc"],
    )
    cvar.train(param["epochs"], param["save_model"], param["print"])

    # Compare usual VAE with CVaR VAE
    vae.model.eval()
    cvar.model.eval()
    with torch.no_grad():
        data = valid_set.data
        recons = vae.model.sample(data.shape[0], vae.device)
        recons_cvar = cvar.model.sample(data.shape[0], cvar.device)
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(133, sharex=ax1, sharey=ax1)
    recons = recons.cpu()
    recons_cvar = recons_cvar.cpu()
    ax1.scatter(data[:, 0], data[:, 1], s=10, color="black")
    ax1.title.set_text("Validation data")
    ax2.scatter(recons[:, 0], recons[:, 1], s=10, color="black")
    ax2.title.set_text("Usual VAE output")
    ax3.scatter(recons_cvar[:, 0], recons_cvar[:, 1], s=10, color="black")
    ax3.title.set_text("Rockarfellar VAE output")
    plt.savefig(param["path_out"] + "output" + str(i) + ".png")
    plt.show()
    plt.clf()
