import numpy as np
import os
import torch

from torchvision import datasets, transforms

from train import GanAlg, AdaCVar

seed = 31415
np.random.seed(seed)
torch.manual_seed(seed)

# learning params
model_param = {
    "x_dim": 28*28,
    "hidden_dims_D": [1024, 512, 256],  # TODO
    "hidden_dims_G": [256, 512, 1024],  # TODO
    "z_dim": 128,
}

exp3_param = {"gamma": 0.9, "beta": 0.0, "eps": 0.0, "iid_batch": False}

param = {
    "epochs_gan": 350,
    "epochs_ada": 400,
    "batch_size": 128,
    "lr": 0.0002,
    "alpha": 0.3,
    "print": True,
    "model_name_usual": "GAN usual",
    "model_name_ada": "AdaCVar alg",
    "save_model": True,
}
# name_out = "mod" if param["imbalanced"] else "usual"
# dataset = ImbalancedMNIST if param["imbalanced"] else datasets.MNIST
name_out = "usual"
dataset = datasets.MNIST

out_param = {
    "dir": [
        "../models/mnist_" + name_out + "/",
        "../output/out_mnist_" + name_out + "/",
        "../input/mnist/",
    ],
    "path_data": "../input/mnist/",
    "path_gan_D": "../models/mnist_" + name_out + "/ganD",
    "path_gan_G": "../models/mnist_" + name_out + "/ganG",
    "path_ada_D": "../models/mnist_" + name_out + "/adaD",
    "path_ada_G": "../models/mnist_" + name_out + "/adaG",
    "path_out": "../output/out_mnist_" + name_out + "/",
}

criterion = torch.nn.BCELoss(reduction="none")

# Create directories for the output if they do not exist
for cur_dir in out_param["dir"]:
    if not os.path.exists(cur_dir):
        os.makedirs(cur_dir)

# Generate data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_set = dataset(
    root=out_param["path_data"],
    train=True,
    download=True,
    transform=transform
)

gan = GanAlg(
    out_param["path_gan_D"],
    out_param["path_gan_G"],
    model_param,
    train_set,
    param["batch_size"],
    param["lr"],
    criterion,
)

ada = AdaCVar(
    out_param["path_ada_D"],
    out_param["path_ada_G"],
    model_param,
    exp3_param,
    train_set,
    param["batch_size"],
    param["lr"],
    criterion,
    param["alpha"],
)

gan.train(param["epochs_gan"], param["save_model"], param["print"])
ada.train(param["epochs_ada"], param["save_model"], param["print"])
