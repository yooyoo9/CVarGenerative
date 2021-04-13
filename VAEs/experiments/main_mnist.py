import numpy as np
import os
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from cvarVAE.train import VAEalg, RockarfellarAlg, CVaRalg

seed = 31415
np.random.seed(seed)
torch.manual_seed(seed)

# learning params
model_param = {
    "x_dim": 1,
    "hidden_dims": [512],
    "z_dim": 16,
    "beta": 0.0,
    "constrained_output": True,
}

param = {
    "epochs": 5,
    "batch_size": 64,
    "lr": 0.0001,
    "alpha": 0.3,
    "beta": 1.0,
    "print": True,
    "model_name": "VAEimg",
    "save_model": True,
    "dir": ["../models/mnist/", "../output/out_mnist/", "../input/mnist/"],
    "path_data": "../input/mnist/",
    "path_vae": "../models/mnist/vae",
    "path_cvar": "../models/mnist/cvar",
    "path_out": "../output/out_mnist/",
}
criterion = torch.nn.BCELoss(reduction="none")

# Create directories for the output if they do not exist
for cur_dir in param["dir"]:
    if not os.path.exists(cur_dir):
        os.makedirs(cur_dir)

# Generate data
train_set = datasets.MNIST(
    root=param["path_data"], train=True, download=True, transform=transforms.ToTensor()
)
valid_set = datasets.MNIST(
    root=param["path_data"], train=False, download=True, transform=transforms.ToTensor()
)

train_loader_vae = DataLoader(train_set, batch_size=param["batch_size"], shuffle=True)
val_loader = DataLoader(valid_set, batch_size=param["batch_size"], shuffle=True)

vae = VAEalg(
    param["model_name"],
    param["path_vae"],
    model_param,
    train_set,
    valid_set,
    param["batch_size"],
    param["lr"],
    criterion,
    beta=param["beta"],
)
vae.train(param["epochs"], param["save_model"], param["print"])

cvar = RockarfellarAlg(
    param["model_name"],
    param["path_cvar"],
    model_param,
    train_set,
    valid_set,
    param["batch_size"],
    param["lr"],
    criterion,
    param["alpha"],
    beta=param["beta"],
)
cvar.train(param["epochs"], param["save_model"], param["print"])

vae.model.eval()
cvar.model.eval()
with torch.no_grad():
    for idx, (data, _) in enumerate(vae.val_loader):
        data = data.to(vae.device)
        data = data.view(data.size(0), -1)
        recons, _, _ = vae.model(data)
        recons_cvar, _, _ = cvar.model(data)

        if idx % 5 == 0:
            orig = data.view(data.shape[0], 1, 28, 28)
            vae_img = recons.view(data.shape[0], 1, 28, 28)
            cvar_img = recons_cvar.view(data.shape[0], 1, 28, 28)
            all_img = torch.cat((orig[:8], vae_img[:8], cvar_img[:8]))
            save_image(all_img.cpu(), f"{param['path_out']}output{idx}.png", nrow=8)
