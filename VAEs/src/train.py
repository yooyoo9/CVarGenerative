import numpy as np
import os
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

class ImgDataSet(Dataset):
    def __init__(self, main_dir, transform, size):
        self.main_dir = main_dir
        self.transform = transform
        self.total_imgs = os.listdir(main_dir)
        self.size = size

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert('L').resize((self.size, self.size))
        tensor_image = self.transform(image)
        return tensor_image, idx

# learning param
param = {
    'epochs' : 50,
    'batch_size' : 64,
    'lr' : 0.0001,
    'img_size' : 100,
    'transform' : transforms.ToTensor(),
    'hidden_dims' : [512],
    'z_dim' : 8,
    'alpha' : 0.3,
    'path_vae': '../models/vae_100',
    'path_cvar': '../models/vae_cvar_100',
    'path_out': '../output/out100/'
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_set = ImgDataSet('../input/train/', param['transform'], param['img_size'])
valid_set = ImgDataSet('../input/val/', param['transform'], param['img_size'])
criterion = nn.BCELoss(reduction='none')

train_loader_vae = torch.utils.data.DataLoader(
    train_set, batch_size = param['batch_size'], shuffle=True)
val_loader = torch.utils.data.DataLoader(
    valid_set, batch_size = param['batch_size'], shuffle=True)

if os.path.exists(param['path_vae']):
    model_vae = torch.load(param['path_vae'])
else:
    model_vae = VAE(
        x_dim=param['img_size']**2,
        hidden_dims=param['hidden_dims'],
        z_dim = param['z_dim'])
model_vae.to(device)
optimizer_vae = optim.Adam(model_vae.parameters(), lr=param['lr'])

train(model_vae, optimizer_vae, criterion, train_loader_vae, val_loader,
      param['epochs'], device, param['path_vae'])


if os.path.exists(param['path_cvar']):
    model_cvar = torch.load(param['path_cvar'])
else:
    model_cvar = VAE(
        x_dim=param['img_size']**2,
        hidden_dims=param['hidden_dims'],
        z_dim = param['z_dim'])
model_cvar.to(device)
optimizer_cvar = optim.Adam(model_cvar.parameters(), lr=param['lr'])

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

cvar_train(model_cvar, optimizer_cvar, criterion, cvar, train_loader_cvar,
           exp3, val_loader, param['epochs'], param['alpha'],
           device, param['path_cvar'])

def output(model, model_cvar, criterion, data_loader, device, img_size, path):
    model.eval()
    model_cvar.eval()
    with torch.no_grad():
        for (data, idx) in data_loader:
            data = data.to(device)
            data = data.view(data.size(0), -1)
            recons, _, _ = model(data)
            recons_cvar, _, _ = model_cvar(data)

            orig = data.view(data.shape[0], 1, img_size, img_size)
            vae = recons.view(data.shape[0], 1, img_size, img_size)
            cvar = recons_cvar.view(data.shape[0], 1, img_size, img_size)
            for i in range(data.shape[0]):
                together = torch.cat((orig[i], vae[i], cvar[i]), 2)
                save_image(together.cpu(), f"{path}output{idx[i]}.png", nrow=1)

output(model_vae, model_cvar, criterion, val_loader, device,
       param['img_size'], param['path_out'])
