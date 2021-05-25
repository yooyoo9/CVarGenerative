import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, x_dim, hidden_dims):
        super().__init__()
        modules = []
        cur = x_dim
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(cur, h_dim),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Dropout(p=0.3)
                )
            )
            cur = h_dim
        modules.append(
            nn.Sequential(
                nn.Linear(cur, 1),
                nn.Sigmoid()
            )
        )
        self.model = nn.Sequential(*modules)

    def forward(self, z):
        z = z.view(-1, 28*28)
        result = self.model(z)
        return result


class Generator(nn.Module):
    def __init__(self, z_dim, hidden_dims, x_dim):
        super().__init__()
        self.z_dim = z_dim
        modules = []
        cur = z_dim
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(cur, h_dim),
                    nn.LeakyReLU(negative_slope=0.2)
                )
            )
            cur = h_dim
        modules.append(
            nn.Sequential(
                nn.Linear(cur, x_dim),
                nn.Tanh()
            )
        )
        self.model = nn.Sequential(*modules)

    def forward(self, z):
        result = self.model(z)
        result = result.view(-1, 28*28)
        return result
                
    def sample(self, num_samples, device):
        """Samples from the latent space and return the corresponding objects in the input space.

        Parameters
        ----------
        num_samples: int
            Number of samples
        device: int
            Device to run the model

        Returns
        -------
        samples: torch.tensor

        """
        latent_var = torch.randn(num_samples, self.z_dim).to(device)
        samples = self.forward(latent_var)
        return samples
