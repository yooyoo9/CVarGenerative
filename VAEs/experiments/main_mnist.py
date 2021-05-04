import numpy as np
import os
import torch

from torchvision import datasets, transforms

from util.train import VaeAlg, Rockarfellar, AdaCVar
from data_mnist import ImbalancedMNIST
from evaluate_mnist import generate_classes, plot_manifold

seed = 31415
np.random.seed(seed)
torch.manual_seed(seed)

# learning params
model_param = {
    "x_dim": 1,
    "hidden_dims": [512, 512],  # TODO
    "z_dim": 2,
    "constrained_output": True,
}

exp3_param = {"gamma": 0.9, "beta": 0.0, "eps": 0.0, "iid_batch": False}

param = {
    "imbalanced": False,
    "epochs": 1000,
    "batch_size": 256,
    "lr": 1e-4,
    "alpha": 0.3,
    "beta": 1.0,
    "early_stop": 50,
    "print": True,
    "model_name": "VaeImg",
    "model_name_usual": "VAE usual",
    "model_name_rockar": "Rockarfellar alg",
    "model_name_ada": "AdaCVar alg",
    "save_model": True,
}

name_out = "mod" if param["imbalanced"] else "usual"
dataset = ImbalancedMNIST if param["imbalanced"] else datasets.MNIST

out_param = {
    "dir": [
        "../models/mnist_" + name_out + "/",
        "../output/out_mnist_" + name_out + "/",
        "../input/mnist/",
    ],
    "path_data": "../input/mnist/",
    "path_vae": "../models/mnist_" + name_out + "/vae",
    "path_rockar": "../models/mnist_" + name_out + "/rockar",
    "path_ada": "../models/mnist_" + name_out + "/ada",
    "path_out": "../output/out_mnist_" + name_out + "/",
}

criterion = torch.nn.MSELoss(reduction="none")

# Create directories for the output if they do not exist
for cur_dir in out_param["dir"]:
    if not os.path.exists(cur_dir):
        os.makedirs(cur_dir)

# Generate data
train_set = dataset(
    root=out_param["path_data"],
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)
valid_set = dataset(
    root=out_param["path_data"],
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)

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

rockar = Rockarfellar(
    param["model_name"],
    out_param["path_rockar"],
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

stop_vae = False
stop_rockar = False
stop_ada = False
for _ in range(param["epochs"] // 50):
    if not stop_vae:
        stop_vae = vae.train(50, param["save_model"], param["print"])
        generate_classes(vae.model, vae.device, out_param["path_out"] + "vae")
        plot_manifold(vae.model, vae.device, out_param["path_out"] + "vae")
    if not stop_rockar:
        stop_rockar = rockar.train(50, param["save_model"], param["print"])
        generate_classes(rockar.model, rockar.device, out_param["path_out"] + "rockar")
        plot_manifold(rockar.model, rockar.device, out_param["path_out"] + "rockar")
    if not stop_ada:
        stop_ada = ada.train(50, param["save_model"], param["print"])
        generate_classes(ada.model, ada.device, out_param["path_out"] + "ada")
        plot_manifold(ada.model, ada.device, out_param["path_out"] + "ada")

    if stop_vae and stop_rockar and stop_ada:
        break
