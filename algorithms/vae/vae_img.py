import torch
from torch import nn


class VaeImg(nn.Module):
    def __init__(
        self,
        n_channel,
        hidden_dims,
        z_dim,
        img_size,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.last_dim = hidden_dims[-1]
        self.img_size = img_size
        self.nb = img_size // (2**len(hidden_dims))

        # Encoder
        modules = []
        cur = n_channel
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(cur, h_dim, 3, 2, 1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(),
                )
            )
            cur = h_dim
        self.encoder = nn.Sequential(*modules)
        self.mu = nn.Linear(hidden_dims[-1] * self.nb * self.nb, z_dim)
        self.logvar = nn.Linear(hidden_dims[-1] * self.nb * self.nb, z_dim)

        # Decoder
        modules = []
        self.decoder_input = nn.Linear(z_dim, hidden_dims[-1] * self.nb * self.nb)
        hidden_dims.reverse()
        cur = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(cur, h_dim, 3, 2, 1, 1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(),
                )
            )
            cur = h_dim
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(cur, cur, 3, 2, 1, 1),
            nn.BatchNorm2d(cur),
            nn.ReLU(),
            nn.Conv2d(cur, n_channel, 3, padding=1),
            nn.Tanh(),
        )

    def encode(self, x):
        """Encodes the input using the encoder network.

        Parameters
        ----------
        x: torch.tensor
            Input tensor to encoder

        Returns
        -------
        mu, logvar: torch.tensor, torch.tensor
            Latent variables
        """
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        mu = self.mu(result)
        logvar = self.logvar(result)
        return mu, logvar

    def decode(self, z):
        """Decodes the latent codes onto the input space.

        Parameters
        ----------
        z: torch.tensor
            Input tensor to decoder, latent variables.

        Returns
        -------
        result: torch.tensor
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.last_dim, self.nb, self.nb)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterization trick to sample from a Gaussian N(mu, var)
        from N(0, 1)

        Parameters
        ----------
        mu: torch.tensor
            Mean of the latent Gaussian
        logvar: torch.tensor
            Log of variance of the latent Gaussian

        Returns
        -------
        samples: torch.tensor
            Samples from the Gaussian
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

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
        samples = self.decode(latent_var)
        return samples
