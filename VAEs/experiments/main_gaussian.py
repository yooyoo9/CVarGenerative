import numpy as np
import os
import argparse

import torch

from util.train import VaeAlg, Rockafellar, AdaCVar
from data_gaussian import GaussianDataSet

seed = 764003779
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs_vae", type=int, default=200)
parser.add_argument("--epochs_rocka", type=int, default=200)
parser.add_argument("--epochs_ada", type=int, default=200)
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--dataset", type=int, default=3, choices=set((0,1,2,3,4)))
parser.add_argument("--alpha", type=float, default=0.3)
parser.add_argument("--beta_vae", type=float, default=0.5)
parser.add_argument("--beta_rocka", type=float, default=0.5)
parser.add_argument("--beta_ada", type=float, default=0.5)
#(0.6, 0.7, 0.6), (0.4, 0.5, 0.4), (0.5, 0.5, 0.4), 0.5, 0.5
parser.add_argument("--print_loss", action='store_true', default=False)
parser.add_argument("--save_model", action='store_true', default=False)
parser.add_argument("--path_data", type=str, default="../input/gaussian/data.npy")
parser.add_argument("--path_model", type=str, default="../models/gaussian/")
args = parser.parse_args()

exp3_param = {"gamma": 0.8, "beta": 0.0, "eps": 0.0, "iid_batch": False}
criterion = torch.nn.MSELoss(reduction="none")

# learning params
model_param = {
    "x_dim": 2,
    "hidden_dims": [128, 128],
    "z_dim": 2,
}

# Create directories for the output if they do not exist
if not os.path.exists(args.path_model):
    os.makedirs(args.path_model)
args.path_model += str(args.dataset)

# Generate data
train_set = GaussianDataSet(args.path_data, args.dataset, train=True)
valid_set = GaussianDataSet(args.path_data, args.dataset, train=False)

vae = VaeAlg(
    "VAE",
    args.path_model,
    model_param,
    train_set,
    valid_set,
    args.batch_size,
    args.learning_rate,
    criterion,
    args.beta_vae,
)

rocka = Rockafellar(
    "VAE",
    args.path_model,
    model_param,
    train_set,
    valid_set,
    args.batch_size,
    args.learning_rate,
    criterion,
    args.alpha,
    args.beta_rocka,
)

ada = AdaCVar(
    "VAE",
    args.path_model,
    model_param,
    exp3_param,
    train_set,
    valid_set,
    args.batch_size,
    args.learning_rate,
    criterion,
    args.alpha,
    args.beta_ada,
)

vae.train(args.epochs_vae, args.save_model, args.print_loss)
rocka.train(args.epochs_rocka, args.save_model, args.print_loss)
ada.train(args.epochs_ada, args.save_model, args.print_loss)
