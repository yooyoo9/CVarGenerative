import numpy as np
import os

import torch
from torch.utils.data import Dataset

from util.train import VaeAlg, Rockafellar, AdaCVar
from data_gaussian import generate_data
from evaluate_gaussian import evaluate_output

seed = 764003779
np.random.seed(seed)
torch.manual_seed(seed)


class GaussianDataSet(Dataset):
    def __init__(self, path, idx, train):
        data = np.load(path)[idx]
        input_data = data[:-1]
        self.n_clusters = data[-1, 0]

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
    "hidden_dims": [128, 128],
    "z_dim": 2,
    "constrained_output": False,
}

exp3_param = {"gamma": 0.8, "beta": 0.0, "eps": 0.0, "iid_batch": False}

# 0.5, (0.4, 0.5, 0.4), 0.5, 0.5, 0.5, 0.5
param = {
    "epochs": 100,
    "batch_size": 256,
    "lr": 1e-4,
    "alpha": 0.3,
    "beta_usual": 0.5,
    "beta_rocka": 0.5,
    "beta_ada": 0.5,
    "early_stop": 1000,
    "print": True,
    "model_name": "VAE",  # or VaeImg
    "save_model": True,
    "nb": 5,  # number of datasets
    "data_size": 1000,
    "dir": [
        "../models/gaussian/",
        "../output/out_gaussian/",
        "../input/gaussian/",
    ],
    "path_data": "../input/gaussian/gaussians.npy",
    "path_vae": "../models/gaussian/vae",
    "path_rocka": "../models/gaussian/rocka",
    "path_ada": "../models/gaussian/ada",
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

    rocka = Rockafellar(
        param["model_name"],
        param["path_rocka"] + str(i),
        model_param,
        train_set,
        valid_set,
        param["batch_size"],
        param["lr"],
        criterion,
        param["alpha"],
        param["beta_rocka"],
        param["early_stop"],
    )

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

    vae.train(param["epochs"], param["save_model"], param["print"])
    rocka.train(param["epochs"], param["save_model"], param["print"])
    ada.train(param["epochs"], param["save_model"], param["print"])

    evaluate_output(
        i, vae.model, rocka.model, ada.model, vae.device, param["alpha"], valid_set
    )
