import numpy as np
import os
import torch
import argparse
import wandb

from train import GanAlg, TruncCVar, AdaCVar
from datasets import GaussianDataSet, MNIST, CelebA, CIFAR10

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_nb", type=int, default=0)
parser.add_argument("--dataset", default="celeba", choices=set(("gaussian", "mnist", "mnist_imb", "cifar10", "celeba")))

parser.add_argument("--model", type=str, default='ada', choices=['vae', 'ada', 'trunc'])
parser.add_argument("--alpha", type=float, default=0.3)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--early_stop", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--optim", type=str, default='adam', choices=['adam', 'sgd'])
parser.add_argument("--learning_rate", type=float, default=0.0002)
parser.add_argument("--momentum", type=float, default=0.0)
parser.add_argument("--exp3_gamma", type=float, default=0.2)
parser.add_argument("--z_dim", type=int, default=100)
parser.add_argument("--save_model", action='store_true', default=False)
args = parser.parse_args()
wandb.init(project='cvar-generative', entity='yooyoo9', config=args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Create directories for the output if they do not exist
if args.dataset == 'gaussian':
    path_data = 'experiments/synthetic/input/X10000.npy'
    path_model = os.path.join('experiments/synthetic/gan/model/', str(args.dataset_nb))
    path_out = os.path.join('experiments/synthetic/gan/output', str(args.dataset_nb))
    for cur in [path_model, path_out]:
        if not os.path.exists(cur):
            os.makedirs(cur)
    path_out = os.path.join('experiments/synthetic/gan/output', str(args.dataset_nb), args.model + '.npy')
else:
    path_data = os.path.join('experiments', args.dataset, 'input')
    path_model = os.path.join('experiments', args.dataset, 'gan', 'model')
    for cur_dir in [path_data, path_model]:
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)

optim_param = {
    'name': args.optim,
    'lr': args.learning_rate,
    'momentum': args.momentum,
}
criterion = torch.nn.BCELoss(reduction="none")

# Generate data
if args.dataset == 'gaussian':
    gan_model = 'gan'
    train_set = GaussianDataSet(path_data, args.dataset, train=0)
    valid_set = GaussianDataSet(path_data, args.dataset, train=1)
    test_set = GaussianDataSet(path_data, args.dataset, train=2)
else:
    if args.dataset == "mnist":
        dataset, gan_model, img_size = MNIST, 'img28', 28
    elif args.dataset == "cifar":
        dataset, gan_model, img_size = CIFAR10, 'img64', 64
    else:
        dataset, gan_model, img_size = CelebA, 'img64', 64
    train_set = dataset(path_data, 0, img_size)
    valid_set = dataset(path_data, 1, img_size)
    test_set = dataset(path_data, 2, img_size)

if args.model == 'gan':
    model = GanAlg(
        args.path_model,
        args.path_out,
        args.z_dim,
        train_set,
        args.batch_size,
        args.learning_rate,
        criterion,
    )
if args.model == 'trunc':
    model = TruncCVar(
        args.path_model,
        args.path_out,
        args.z_dim,
        train_set,
        args.batch_size,
        args.learning_rate,
        criterion,
        args.alpha,
    )
if args.model == 'ada':
    exp3_param = {"gamma": args.exp3_gamma, "beta": 0.0, "eps": 0.0, "iid_batch": False}
    model = AdaCVar(
        args.path_model,
        args.path_out,
        args.z_dim,
        exp3_param,
        train_set,
        args.batch_size,
        args.learning_rate,
        criterion,
        args.alpha,
    )

model.train(args.epochs_ada, args.early_stop, args.save_model)
