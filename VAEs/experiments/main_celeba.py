import numpy as np
import os
import torch

from torchvision import datasets, transforms
from torch.utils.data import Dataset, random_split
from PIL import Image

from util.train import VaeAlg, Rockafellar, AdaCVar

seed = 31415
np.random.seed(seed)
torch.manual_seed(seed)


class CelebA(Dataset):
    def __init__(self, path_data, split="train", image_size=64):
        self.path_data = path_data

        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.dataset = datasets.CelebA(
            root=path_data,
            split=split,
            download=True,
            transform=transform,
        )

    def __getitem__(self, index):
        data, _ = self.dataset[index]
        return data, index

    def __len__(self):
        return len(self.dataset)


# learning params
model_param = {
    "x_dim": 3,
    "hidden_dims": [512, 256, 128],
    "z_dim": 128,
    "constrained_output": True,
}

exp3_param = {"gamma": 0.9, "beta": 0.0, "eps": 0.0, "iid_batch": False}

param = {
    "epochs": 1000,
    "batch_size": 128,
    "lr": 1e-4,
    "alpha": 0.3,
    "beta": 1.0,
    "early_stop": 300,
    "print": True,
    "image_size": 64,
    "model_name": "VaeCeleba",
    "model_name_usual": "VAE usual",
    "model_name_rocka": "Rockafellar alg",
    "model_name_ada": "AdaCVar alg",
    "save_model": True,
}

out_param = {
    "dir": [
        "../models/celeba/",
        "../output/out_celeba/",
        "../input/",
    ],
    "path_data": "../input/celeba/",
    "path_vae": "../models/celeba/vae",
    "path_rocka": "../models/celeba/rocka",
    "path_ada": "../models/celeba/ada",
    "path_out": "../output/out_celeba/",
}

criterion = torch.nn.MSELoss(reduction="none")

# Create directories for the output if they do not exist
for cur_dir in out_param["dir"]:
    if not os.path.exists(cur_dir):
        os.makedirs(cur_dir)

# Generate dat
# Create the dataset
train_set = CelebA(out_param["path_data"], "train", param["image_size"])
valid_set = CelebA(out_param["path_data"], "valid", param["image_size"])

vae = VaeAlg(
    param["model_name"],
    out_param["path_vae"],
    model_param,
    train_set,
    valid_set,
    param["batch_size"],
    param["lr"],
    criterion,
    beta=param["beta"],
    early_stop=param["early_stop"],
)

rocka = Rockafellar(
    param["model_name"],
    out_param["path_rocka"],
    model_param,
    train_set,
    valid_set,
    param["batch_size"],
    param["lr"],
    criterion,
    param["alpha"],
    beta=param["beta"],
    early_stop=param["early_stop"],
)

ada = AdaCVar(
    param["model_name"],
    out_param["path_ada"],
    model_param,
    exp3_param,
    train_set,
    valid_set,
    param["batch_size"],
    param["lr"],
    criterion,
    param["alpha"],
    param["beta"],
    early_stop=param["early_stop"],
)

stop_vae = True
stop_rocka = True
stop_ada = False
while True:
    if not stop_vae:
        stop_vae = vae.train(200, param["save_model"], param["print"])
    if not stop_rocka:
        stop_rocka = rocka.train(200, param["save_model"], param["print"])
    if not stop_ada:
        stop_ada = ada.train(200, param["save_model"], param["print"])

    if stop_vae and stop_rocka and stop_ada:
        break
