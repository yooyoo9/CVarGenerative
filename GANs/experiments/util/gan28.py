import torch
from torch import nn

class Discriminator28(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, z):
        result = self.model(z)
        return result.view(-1)


class Generator28(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.z_dim = z_dim
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, hidden_dim * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d( hidden_dim * 2, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),

            nn.ConvTranspose2d( hidden_dim, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        result = self.model(z)
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
        latent_var = torch.randn(num_samples, self.z_dim, 1, 1).to(device)
        samples = self.forward(latent_var)
        return samples
