import numpy as np
import os
import argparse
import torch

from torchvision import datasets, transforms

from util.train import GanAlg, Rockafellar, AdaCVar
from datasets import MNIST, CelebA, CIFAR10

seed = 31415
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs_gan", type=int, default=200)
parser.add_argument("--epochs_rocka", type=int, default=50)
parser.add_argument("--epochs_ada", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=0.0002)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--dataset", default="cifar10", choices=set(("mnist", "cifar10", "celeba")))
parser.add_argument("--z_dim", type=int, default=100)
parser.add_argument("--alpha", type=float, default=0.3)
parser.add_argument("--print_loss", action='store_true', default=False)
parser.add_argument("--save_model", action='store_true', default=False)
parser.add_argument("--path_data", type=str, default="../input/")
parser.add_argument("--path_model", type=str, default="../models/")
parser.add_argument("--path_out", type=str, default="../output/")
args = parser.parse_args()

args.path_data += args.dataset + "/"
args.path_model += args.dataset + "/"
args.path_out += args.dataset + "/"

if args.dataset == "mnist":
    args.img_size, args.hidden_dim = 28, 32
else:
    args.img_size, args.hidden_dim = 64, 64

model_param = {
    "img_size": args.img_size,
    "hidden_dim": args.hidden_dim,
    "z_dim": args.z_dim,
}
exp3_param = {"gamma": 0.1, "beta": 0.0, "eps": 0.0, "iid_batch": False}
criterion = torch.nn.BCELoss(reduction="none")

# Create directories for the output if they do not exist
for cur_dir in [args.path_data, args.path_model, args.path_out]:
    if not os.path.exists(cur_dir):
        os.makedirs(cur_dir)

# Generate data
if args.dataset == "mnist":
    dataset = MNIST
elif args.dataset == "cifar10":
    dataset = CIFAR10
else:
    dataset = CelebA
train_set = dataset(
    root=args.path_data,
    train=True,
    img_size=args.img_size
)

gan = GanAlg(
    args.path_model,
    args.path_out,
    model_param,
    train_set,
    args.batch_size,
    args.learning_rate,
    criterion,
)

rocka = Rockafellar(
    args.path_model,
    args.path_out,
    model_param,
    train_set,
    args.batch_size,
    args.learning_rate,
    criterion,
    args.alpha,
)

ada = AdaCVar(
    args.path_model,
    args.path_out,
    model_param,
    exp3_param,
    train_set,
    args.batch_size,
    args.learning_rate,
    criterion,
    args.alpha,
)

# gan.train(args.epochs_gan, args.save_model, args.print_loss)
# rocka.train(args.epochs_rocka, args.save_model, args.print_loss)
ada.train(args.epochs_ada, args.save_model, args.print_loss)
