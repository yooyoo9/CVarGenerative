import torch
from torch import nn


class VAEimg(nn.Module):
    def __init__(
        self,
        x_dim,
        hidden_dims,
        z_dim,
        kernel_size=3,
        stride=2,
        padding=1,
        output_padding=1,
        constrained_output=False,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.last_dim = hidden_dims[-1]

        # Encoder
        modules = []
        cur = x_dim
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        cur,
                        h_dim,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(),
                )
            )
            cur = h_dim
        self.encoder = nn.Sequential(*modules)
        self.encoder_output = nn.Linear(hidden_dims[-1] * 4, 2 * z_dim)

        # Decoder
        modules = []
        self.decoder_input = nn.Linear(z_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()
        cur = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        cur,
                        h_dim,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(),
                )
            )
            cur = h_dim

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    cur,
                    cur,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                ),
                nn.BatchNorm2d(cur),
                nn.ReLU(),
                nn.Conv2d(
                    cur, out_channels=3, kernel_size=kernel_size, padding=padding
                ),
            )
        )

        if constrained_output:
            modules.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*modules)

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
        result = result.view(-1, 2, self.z_dim)

        mu = result[:, 0]
        logvar = result[:, 1]
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
        result = result.view(-1, self.last_dim, 2, 2)
        result = self.decoder(result)
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
