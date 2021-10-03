import numpy as np
import os
import torch
import argparse
import wandb

from train import VaeAlg, TruncCVar, AdaCVar
from datasets import GaussianDataSet, MNIST, ImbalancedMNIST, CelebA, CIFAR10, ImbalancedCIFAR10


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_nb", type=int, default=0)
parser.add_argument("--dataset", default="celeba")
parser.add_argument("--model", type=str, default='ada', choices=['vae', 'ada', 'trunc'])
parser.add_argument("--alpha", type=float, default=0.3)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--early_stop", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--optim", type=str, default='adam', choices=['adam', 'sgd'])
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--momentum", type=float, default=0.0)
parser.add_argument("--exp3_gamma", type=float, default=0.2)
parser.add_argument("--save_model", action='store_true', default=False)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
wandb.init(project='cvar-generative', entity='yooyoo9', config=args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Create directories for the output if they do not exist
if args.dataset == 'gaussian':
    path_data = 'experiments/synthetic/input/X10000.npy'
    path_model = os.path.join('experiments/synthetic/vae/model/', str(args.dataset_nb))
    path_out = os.path.join('experiments/synthetic/vae/output', str(args.dataset_nb))
    for cur in [path_model, path_out]:
        if not os.path.exists(cur):
            os.makedirs(cur)
    path_out = os.path.join('experiments/synthetic/vae/output', str(args.dataset_nb), args.model + '.npy')
else:
    path_data = os.path.join('experiments', args.dataset, 'input')
    path_model = os.path.join('experiments', args.dataset, 'vae', 'model', str(args.seed))
    for cur_dir in [path_data, path_model]:
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)

optim_param = {
    'name': args.optim,
    'lr': args.learning_rate,
    'momentum': args.momentum,
}
criterion = torch.nn.MSELoss(reduction="none")

# Generate data
if args.dataset == 'gaussian':
    vae_model = 'VAE'
    train_set = GaussianDataSet(path_data, args.dataset_nb, train=0)
    valid_set = GaussianDataSet(path_data, args.dataset_nb, train=1)
    test_set = GaussianDataSet(path_data, args.dataset_nb, train=2)
    model_param = {
        "x_dim": 2,
        "hidden_dims": [64, 64],
        "z_dim": 2,
    }
else:
    vae_model = 'VaeImg'
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
        if args.dataset == 'cifar':
            dataset = CIFAR10
        elif args.dataset == 'cifar_imb':
            dataset = ImbalancedCIFAR10
        else:
            dataset = CelebA
        img_size = 64
        model_param = {
            "n_channel": 3,
            "hidden_dims": [512, 256, 128],
            "z_dim": 128,
            "img_size": img_size,
        }

    train_set = dataset(path_data, 0, img_size)
    valid_set = dataset(path_data, 1, img_size)
    test_set = dataset(path_data, 2, img_size)

if args.model == 'vae':
    model = VaeAlg(
        vae_model,
        path_model,
        model_param,
        train_set,
        valid_set,
        test_set,
        args.batch_size,
        optim_param,
        criterion,
        args.alpha,
        args.beta,
    )
elif args.model == 'trunc':
    model = TruncCVar(
        vae_model,
        path_model,
        model_param,
        train_set,
        valid_set,
        test_set,
        args.batch_size,
        optim_param,
        criterion,
        args.alpha,
        args.beta
    )
elif args.model == 'ada':
    exp3_param = {"gamma": args.exp3_gamma, "beta": 0.0, "eps": 0.0, "iid_batch": False}
    model = AdaCVar(
        vae_model,
        path_model,
        model_param,
        exp3_param,
        train_set,
        valid_set,
        test_set,
        args.batch_size,
        optim_param,
        criterion,
        args.alpha,
        args.beta
    )

model.train(args.epochs, args.early_stop, args.save_model)

if args.dataset == 'gaussian':
    test_loss, test_mean, test_worst, test_cvar = model.evaluate(val=False, output_path=path_out)
    wandb.log({
        'test_loss': test_loss,
        'test_mean': test_mean,
        'test_worst': test_worst,
        'test_cvar': test_cvar
    })
