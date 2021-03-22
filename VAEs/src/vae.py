import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self, x_dim, hidden_dims, z_dim):
        super().__init__()
        # Encoder
        modules = []
        cur = x_dim
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(cur, h_dim),
                    nn.ReLU()
                )
            )
            cur = h_dim
        self.encoder = nn.Sequential(*modules)

        # Output of the encoder: mean and variance
        self.mu = nn.Linear(hidden_dims[-1], z_dim)
        self.var = nn.Linear(hidden_dims[-1], z_dim)

        # Decoder
        modules = []
        cur = z_dim
        hidden_dims.reverse()
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(cur, h_dim),
                    nn.ReLU()
                )
            )
            cur = h_dim
        modules.append( nn.Sequential(nn.Linear(cur, x_dim), nn.Sigmoid() ))
        self.decoder = nn.Sequential(*modules)

        
    def encode(self, x):
        result = self.encoder(x)
        mu = self.mu(result)
        log_var = self.var(result)
        return mu, log_var

    
    def decode(self, z):
        result = self.decoder(z)
        return result

    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    
    def loss(self, x, recons, mu, log_var, criterion):
        recons_loss = torch.sum(criterion(recons, x), axis=1)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1)
        loss = recons_loss + kld_loss
        return loss
