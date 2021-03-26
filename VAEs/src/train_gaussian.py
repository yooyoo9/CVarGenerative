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
    def __init__(self, path, transform, nb):
        self.transform = transform
        self.data = np.load(path)[nb]
        print(self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cur = torch.tensor(self.data[idx]).type('torch.FloatTensor')
        return cur, idx

# learning param
param = {
    'epochs' : 100,
    'batch_size' : 64,
    'lr' : 1e-4,
    'decay': 0,
    'transform' : transforms.ToTensor(),
    'x_dim': 2,
    'hidden_dims' : [100],
    'z_dim' : 16,
    'alpha' : 0.3,
    'dir': ['../models/', '../output/out_gaussian/'],
    'path_vae': '../models/vae_gaussian',
    'path_cvar': '../models/vae_gaussian_cvar',
    'path_out': '../output/out_gaussian/',
    'save_model': True,
    'nb': 1
}

for cur_dir in param['dir']:
    if not os.path.exists(cur_dir):
        os.makedirs(cur_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.MSELoss(reduction='none')
# criterion = nn.BCELoss(reduction='none')

def output(model, model_cvar, criterion, data_loader, device, path, size, nb):
    fig = plt.figure()
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    
    model.eval()
    model_cvar.eval()
    with torch.no_grad():
        for (data, idx) in data_loader:
            data = data.to(device)
            data = data.view(data.size(0), -1)
            recons, _, _ = model(data)
            recons_cvar, _, _ = model_cvar(data)
            ax1.scatter(data[:, 0], data[:, 1], s=10, color='black')
            ax2.scatter(recons[:, 0], recons[:, 1], s=10, color='black')
            ax3.scatter(recons_cvar[:,0], recons_cvar[:,1], s=10, color='black')
        plt.savefig(path+'output'+str(nb)+'.png')
        plt.clf()


for i in range(param['nb']):
    train_set = GaussianDataSet('../data_train.npy', param['transform'], i)
    valid_set = GaussianDataSet('../data_val.npy', param['transform'], i)

    train_loader_vae = DataLoader(
        train_set, batch_size = param['batch_size'])
    val_loader = DataLoader(
        valid_set, batch_size = param['batch_size'])

    cur_path_vae = param['path_vae']+str(i)
    cur_path_cvar = param['path_cvar']+str(i)
    if os.path.exists(cur_path_vae):
        model_vae = torch.load(cur_path_vae)
    else:
        model_vae = VAE(
            x_dim=param['x_dim'],
            hidden_dims=param['hidden_dims'],
            z_dim = param['z_dim'])
    model_vae.to(device)
    optimizer_vae = optim.Adam(model_vae.parameters(), lr=param['lr'],
                               weight_decay=param['decay'])

    train(model_vae, optimizer_vae, criterion, train_loader_vae, val_loader,
          param['epochs'], device, cur_path_vae, param['save_model'])

    if os.path.exists(cur_path_cvar):
        model_cvar = torch.load(cur_path_cvar)
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
               device, cur_path_cvar, param['save_model'])
    '''
    
    output(model_vae, model_vae, criterion, val_loader, device,
           param['path_out'], param['x_dim']//2, i)

