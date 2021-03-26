import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from vae import VAE

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from PIL import Image, ImageFilter

from adacvar.util.cvar import CVaR
from adacvar.util.adaptive_algorithm import Exp3Sampler

from train_cvar import cvar_train
from train_vae import train

seed = 31415
np.random.seed(seed)
torch.manual_seed(seed)

class GaussianDataSet(Dataset):
    def __init__(self, path, transform):
        self.transform = transform
        self.data = np.load(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cur = self.transform(self.data[idx]).type('torch.FloatTensor')
        return cur, idx


# learning param
param = {
    'epochs' : 100,
    'batch_size' : 64,
    'lr' : 1e-3,
    'decay': 1e-3,
    'transform' : transforms.ToTensor(),
    'x_dim': 1000,
    'hidden_dims' : [500, 200],
    'z_dim' : 4,
    'alpha' : 0.3,
    'path_vae': '../models/vae_gaussian',
    'path_cvar': '../models/vae_gaussian_cvar',
    'path_out': '../output/out_gaussian/'
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_set = GaussianDataSet('../data_train.npy', param['transform'])
valid_set = GaussianDataSet('../data_val.npy', param['transform'])
criterion = nn.MSELoss(reduction='none')
# criterion = nn.BCELoss(reduction='none')

train_loader_vae = DataLoader(
    train_set, batch_size = param['batch_size'])
val_loader = DataLoader(
    valid_set, batch_size = param['batch_size'])

if os.path.exists(param['path_vae']):
    model_vae = torch.load(param['path_vae'])
else:
    model_vae = VAE(
        x_dim=param['x_dim'],
        hidden_dims=param['hidden_dims'],
        z_dim = param['z_dim'])
model_vae.to(device)
optimizer_vae = optim.Adam(model_vae.parameters(), lr=param['lr'],
                           weight_decay=param['decay'])

train(model_vae, optimizer_vae, criterion, train_loader_vae, val_loader,
      param['epochs'], device, param['path_vae'])

if os.path.exists(param['path_cvar']):
    model_cvar = torch.load(param['path_cvar'])
else:
    model_cvar = VAE(
        x_dim=param['x_dim'],
        hidden_dims=param['hidden_dims'],
        z_dim = param['z_dim'])
model_cvar.to(device)
optimizer_cvar = optim.Adam(model_cvar.parameters(), lr=param['lr'],
                            weight_decay=param['decay'])

exp3 = Exp3Sampler(
    param['batch_size'],
    num_actions = len(train_set),
    size = int(np.ceil(param['alpha'] * len(train_set))),
    eta = np.sqrt(1 / param['alpha'] * np.log(1 / param['alpha'])),
    gamma=0,
    beta=0,
    eps=0,
    iid_batch=False)
train_loader_cvar = DataLoader(train_set, batch_sampler=exp3)
cvar = CVaR(alpha=1, learning_rate=0).to(device)

'''
cvar_train(model_cvar, optimizer_cvar, criterion, cvar, train_loader_cvar,
           exp3, val_loader, param['epochs'], param['alpha'],
           device, param['path_cvar'])
'''

def output(model, model_cvar, criterion, data_loader, device, path, size):
    fig = plt.figure()
    model.eval()
    model_cvar.eval()
    with torch.no_grad():
        for (data, idx) in data_loader:
            data = data.to(device)
            data = data.view(data.size(0), -1)
            recons, _, _ = model(data)
            recons_cvar, _, _ = model_cvar(data)

            orig = data.view(-1, size, 2).numpy()
            vae = recons.view(-1, size, 2).numpy()
            cvar = recons_cvar.view(-1, size, 2).numpy()
            for i in range(data.shape[0]):
                plt.subplot(1, 3, 1)
                plt.axis('off')
                plt.scatter(orig[i, :, 0], orig[i, :, 1], s=10, color='black')
                plt.subplot(1, 3, 2)
                plt.axis('off')
                plt.scatter(vae[i, :, 0], vae[i, :, 1], s=10, color='black')
                plt.subplot(1, 3, 3)
                plt.axis('off')
                plt.scatter(cvar[i,:,0], cvar[i,:,1], s=10, color='black')
                
                plt.savefig(path+str(idx[i].item())+'.png')
                plt.clf()


output(model_vae, model_vae, criterion, val_loader, device, param['path_out'],
       param['x_dim']//2)
