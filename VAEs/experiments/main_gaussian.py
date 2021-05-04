import numpy as np
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

from util.train import VaeAlg, Rockarfellar, AdaCVar
from data_gaussian import generate_data

seed = 764003779
np.random.seed(seed)
torch.manual_seed(seed)


class GaussianDataSet(Dataset):
    def __init__(self, path, idx, train):
        input_data = np.load(path)[idx]

        # Normalize the data
        input_data = input_data - input_data.mean(axis=0)
        input_data = input_data / input_data.std(axis=0)

        n_train = int(len(input_data) * 0.8)
        if train:
            # Training data
            self.data = input_data[:n_train]
        else:
            # Validation data
            self.data = input_data[n_train:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cur = torch.tensor(self.data[idx]).type("torch.FloatTensor")
        return cur, idx


# learning params
model_param = {
    "x_dim": 2,
    "hidden_dims": [100, 100],
    "z_dim": 2,
    "constrained_output": False,
}

exp3_param = {"gamma": 0.9, "beta": 0.0, "eps": 0.0, "iid_batch": False}

param = {
    "epochs": 1000,
    "batch_size": 200,
    "lr": 1e-4,
    "alpha": 0.3,
    "beta_usual": 0.08,
    "beta_rockar": 0.2,
    "beta_ada": 0.08,
    "early_stop": 50,
    "print": True,
    "model_name": "VAE",  # or VaeImg
    "model_name_usual": "VAE usual",
    "model_name_rockar": "Rockarfellar alg",
    "model_name_ada": "AdaCVar alg",
    "save_model": True,
    "nb": 1,  # number of datasets
    "data_size": 1000,
    "dir": [
        "../../models/gaussian/",
        "../../output/out_gaussian/",
        "../../input/gaussian/",
    ],
    "path_data": "../../input/gaussian/one_gaussian.npy",
    "path_vae": "../../models/gaussian/vae",
    "path_rockar": "../../models/gaussian/rockar",
    "path_ada": "../../models/gaussian/ada",
    "path_out": "../../output/out_gaussian/",
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

    vae = VaeAlg(
        param["model_name"],
        param["path_vae"] + str(i),
        model_param,
        train_set,
        valid_set,
        param["batch_size"],
        param["lr"],
        criterion,
        param["beta_usual"],
        param["early_stop"],
    )
    vae.train(param["epochs"], param["save_model"], param["print"])

    rockar = Rockarfellar(
        param["model_name"],
        param["path_rockar"] + str(i),
        model_param,
        train_set,
        valid_set,
        param["batch_size"],
        param["lr"],
        criterion,
        param["alpha"],
        param["beta_rockar"],
        param["early_stop"],
    )
    rockar.train(param["epochs"], param["save_model"], param["print"])

    ada = AdaCVar(
        param["model_name"],
        param["path_ada"] + str(i),
        model_param,
        exp3_param,
        train_set,
        valid_set,
        param["batch_size"],
        param["lr"],
        criterion,
        param["alpha"],
        param["beta_ada"],
        param["early_stop"],
    )
    ada.train(param["epochs"], param["save_model"], param["print"])

    # Compare usual VAE with CVaR VAE
    vae.model.eval()
    rockar.model.eval()
    ada.model.eval()
    with torch.no_grad():
        data = valid_set.data
        recons = vae.model.sample(data.shape[0], vae.device)
        recons_rockar = rockar.model.sample(data.shape[0], rockar.device)
        recons_ada = ada.model.sample(data.shape[0], ada.device)
    ax1 = plt.subplot(141)
    ax2 = plt.subplot(142, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(143, sharex=ax1, sharey=ax1)
    ax4 = plt.subplot(144, sharex=ax1, sharey=ax1)
    recons = recons.cpu()
    recons_rockar = recons_rockar.cpu()
    recons_ada = recons_ada.cpu()
    ax1.scatter(data[:, 0], data[:, 1], s=10, color="black")
    ax1.title.set_text("Validation data")
    ax2.scatter(recons[:, 0], recons[:, 1], s=10, color="black")
    ax2.title.set_text(param["model_name_usual"] + " output")
    ax3.scatter(recons_rockar[:, 0], recons_rockar[:, 1], s=10, color="black")
    ax3.title.set_text(param["model_name_rockar"] + " output")
    ax4.scatter(recons_ada[:, 0], recons_ada[:, 1], s=10, color="black")
    ax4.title.set_text(param["model_name_ada"] + " output")
    plt.savefig(param["path_out"] + "output" + str(i) + ".png")
    plt.show()
    plt.clf()
