import torch
from torch import nn
import numpy as np

import networkx as nx

class VAE(nn.Module):
    def __init__(self, x_dim, hidden_dims, z_dim):
        super().__init__()
        self.z_dim = z_dim
        
        # Encoder
        modules = []
        cur = x_dim
        print(cur)
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(nn.Linear(cur, h_dim), nn.ReLU())
            )
            cur = h_dim
        modules.append(nn.Linear(cur, 2*z_dim))
        self.encoder = nn.Sequential(*modules)

        # Decoder
        modules = []
        cur = z_dim
        hidden_dims.reverse()
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(nn.Linear(cur, h_dim), nn.ReLU())
            )
            cur = h_dim
        modules.append( nn.Sequential(nn.Linear(cur, x_dim), nn.Sigmoid() ))
        self.decoder = nn.Sequential(*modules)

        
    def encode(self, x):
        """Encodes the input using the encoder network.

        Parameters
        ----------
        x: Tensor
            Input tensor to encoder

        Returns
        -------
        mu, logvar: Tensor, Tensor
            Latent variables
        """
        result = self.encoder(x)
        result = result.view(-1, 2, self.z_dim)
        
        mu = result[:, 0]
        logvar = result[:, 1]
        return mu, logvar

    
    def decode(self, z):
        """Decodes the latent codes onto the input space.

        Parameters
        ----------
        z: Tensor
            Input tensor to decoder, latent variables.

        Returns
        -------
        result: Tensor
        """
        result = self.decoder(z)
        return result

    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from a Gaussian N(mu, var)
        from N(0, 1)

        Parameters
        ----------
        mu: Tensor
            Mean of the latent Gaussian
        logvar: Tensor
            Log of variance of the latent Gaussian

        Returns
        -------
        samples: Tensor
            Samples from the Gaussian
        """
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    
    def loss(self, x, recons, mu, logvar, criterion):
        """Computes the loss function."""

        # Reconstruction loss
        '''
        Trying some sorting
        recons = recons.view(-1, 500, 2)
        d1, d2, _ = recons.size()
        ind = np.empty(recons.shape)
        for i in range(d1):
            ind[i] = find_indices(x[i].to_numpy(), recons[i].to_numpy())
        ind = torch.tensor(
        recons = recons[
            torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(),
            ind.flatten()
        ].view(d1, 1000)
        '''
        rec_loss = torch.sum(criterion(recons, x), dim=1)

        # Kl-Divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1)

        loss = rec_loss + kl_loss
        return loss
        
