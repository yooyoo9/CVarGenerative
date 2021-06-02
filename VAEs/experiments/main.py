import numpy as np
import os
import torch
import argparse

from torchvision import datasets, transforms

from util.train import VaeAlg, Rockafellar, AdaCVar
from datasets import MNIST, ImbalancedMNIST, CelebA, CIFAR10

seed = 31415
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs_vae", type=int, default=200)
parser.add_argument("--epochs_rocka", type=int, default=200)
parser.add_argument("--epochs_ada", type=int, default=200)
parser.add_argument("--learning_rate", type=float, default=0.0005)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--dataset", default="celeba", choices=set(("mnist", "mnist_imb", "cifar10", "celeba")))
parser.add_argument("--alpha", type=float, default=0.3)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--print_loss", action='store_true', default=False)
parser.add_argument("--save_model", action='store_true', default=False)
parser.add_argument("--path_data", type=str, default="../input/")
parser.add_argument("--path_model", type=str, default="../models/")
args = parser.parse_args()

args.path_data += args.dataset + "/"
args.path_model += args.dataset + "/"

exp3_param = {"gamma": 0.9, "beta": 0.0, "eps": 0.0, "iid_batch": False}

criterion = torch.nn.MSELoss(reduction="none")

# Create directories for the output if they do not exist
for cur_dir in [args.path_data, args.path_model]:
    if not os.path.exists(cur_dir):
        os.makedirs(cur_dir)

# Generate data
model_name = "VaeImg"
if args.dataset == "mnist" or args.dataset == "mnist_imb":
    dataset = MNIST if args.dataset == "mnist" else ImbalancedMNIST
    img_size = 28
    model_param = {
        "n_channel": 1,
        "hidden_dims": [512, 512],
        "z_dim": 2,
        "img_size": img_size,
    }
else:
    dataset = CIFAR10 if args.dataset == "cifar10" else CelebA
    img_size = 64
    model_param = {
        "n_channel": 3,
        "hidden_dims": [512, 256, 128],
        "z_dim": 128,
        "img_size": img_size,
    }

train_set = dataset(args.path_data, True, img_size)
valid_set = dataset(args.path_data, False, img_size)

vae = VaeAlg(
    model_name,
    args.path_model,
    model_param,
    train_set,
    valid_set,
    args.batch_size,
    args.learning_rate,
    criterion,
    args.beta
)

rocka = Rockafellar(
    model_name,
    args.path_model,
    model_param,
    train_set,
    valid_set,
    args.batch_size,
    args.learning_rate,
    criterion,
    args.alpha,
    args.beta
)

ada = AdaCVar(
    model_name,
    args.path_model,
    model_param,
    exp3_param,
    train_set,
    valid_set,
    args.batch_size,
    args.learning_rate,
    criterion,
    args.alpha,
    args.beta
)

# vae.train(args.epochs_vae, args.save_model, args.print_loss)
rocka.train(args.epochs_rocka, args.save_model, args.print_loss)
# ada.train(args.epochs_ada, args.save_model, args.print_loss)
