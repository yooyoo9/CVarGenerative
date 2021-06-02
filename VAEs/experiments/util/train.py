import numpy as np
import os
import torch
from torch import optim
from torch.utils.data import DataLoader

from adacvar.util.cvar import CVaR
from adacvar.util.adaptive_algorithm import Exp3Sampler

from .vae import VAE
from .vae_img import VaeImg

class VaeAlg:
    def __init__(
        self,
        model_name,
        model_path,
        model_param,
        train_set,
        valid_set,
        batch_size,
        lr,
        criterion,
        beta,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not hasattr(self, "name"):
            self.name = "vae"
        self.model_path = model_path + self.name

        if os.path.exists(self.model_path):
            print('load')
            self.model = torch.load(self.model_path, map_location=torch.device("cpu"))
        else:
            if model_name == "VAE":
                self.model = VAE(
                    x_dim=model_param["x_dim"],
                    hidden_dims=model_param["hidden_dims"],
                    z_dim=model_param["z_dim"],
                )
            else:
                self.model = VaeImg(
                    n_channel=model_param["n_channel"],
                    hidden_dims=model_param["hidden_dims"],
                    z_dim=model_param["z_dim"],
                    img_size=model_param["img_size"],
                )
        self.model.to(self.device)

        self.train_loader = DataLoader(train_set, batch_size)
        self.val_loader = DataLoader(valid_set, batch_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = criterion
        self.beta = beta

    def loss(self, x, recons, mu, logvar):
        """Computes the loss function."""

        # Reconstruction loss
        rec_loss = torch.sum(
            torch.flatten(self.criterion(recons, x), start_dim=1), dim=1
        )

        # Kl-Divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1)
        loss = rec_loss + self.beta * kl_loss
        return torch.sum(loss)

    def train(self, epochs, save_model, out):
        """Trains the VAE using the training set

        Parameters
        ----------
        epochs: int
            Number of epochs to train
        save_model: bool
            If set to true, saves the model after training
        out: bool
            If set to true, outputs loss after each epoch
        """
        self.model.train()
        for epoch_idx in range(epochs):
            if out:
                print(f"Epoch {epoch_idx+1} of {epochs}")
            running_loss = 0.0
            for batch_idx, (data, *_) in enumerate(self.train_loader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                recons, mu, logvar = self.model(data)
                loss = self.loss(data, recons, mu, logvar)
                running_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            train_loss = running_loss / len(self.train_loader.dataset)
            val_loss = self.evaluate()

            if out:
                print(f"Train Loss: {train_loss:.4f}   Val Loss: {val_loss:.4f}")

            if save_model and epoch_idx % 50 == 0:
                torch.save(self.model, self.model_path)
        if save_model:
            torch.save(self.model, self.model_path)

    def evaluate(self):
        """Evaluates the VAE using the validation data.

        Returns
        -------
        val_loss: float
            The validation loss
        """
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for (data, *_) in self.val_loader:
                data = data.to(self.device)
                recons, mu, logvar = self.model(data)
                loss = self.loss(data, recons, mu, logvar)
                running_loss += loss.item()
            val_loss = running_loss / len(self.val_loader.dataset)
        return val_loss


class Rockafellar(VaeAlg):
    def __init__(
        self,
        model_name,
        model_path,
        model_param,
        train_set,
        valid_set,
        batch_size,
        lr,
        criterion,
        alpha,
        beta,
    ):
        self.name = "rocka"
        super().__init__(
            model_name,
            model_path,
            model_param,
            train_set,
            valid_set,
            batch_size,
            lr,
            criterion,
            beta,
        )

        self.alpha = alpha
        self.roc_loss = torch.ones(1, requires_grad=True, device=self.device)
        self.optimizer = optim.Adam(
            [{"params": self.model.parameters()}, {"params": self.roc_loss}], lr=lr
        )

    def loss(self, x, recons, mu, logvar):
        """Computes the loss function."""

        # Reconstruction loss
        rec_loss = torch.sum(
            torch.flatten(self.criterion(recons, x), start_dim=1), dim=1
        )

        # Kl-Divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1)
        loss = rec_loss + self.beta * kl_loss

        loss = torch.nn.ReLU()(loss - self.roc_loss)
        loss = self.roc_loss + torch.sum(loss) / (self.alpha * len(loss))
        return loss


class AdaCVar(VaeAlg):
    def __init__(
        self,
        model_name,
        model_path,
        model_param,
        exp3_param,
        train_set,
        valid_set,
        batch_size,
        lr,
        criterion,
        alpha,
        beta,
    ):
        self.name = "ada"
        super().__init__(
            model_name,
            model_path,
            model_param,
            train_set,
            valid_set,
            batch_size,
            lr,
            criterion,
            beta,
        )

        self.exp3 = Exp3Sampler(
            batch_size,
            num_actions=len(train_set),
            size=int(np.ceil(alpha * len(train_set))),
            eta=np.sqrt(1 / alpha * np.log(1 / alpha)),
            gamma=exp3_param["gamma"],
            beta=exp3_param["beta"],
            eps=exp3_param["eps"],
            iid_batch=exp3_param["iid_batch"],
        )
        self.train_loader = DataLoader(train_set, batch_sampler=self.exp3)
        self.cvar = CVaR(alpha=1, learning_rate=0).to(self.device)
        self.k = int(np.ceil(alpha * len(self.val_loader.dataset)))

    def loss(self, x, recons, mu, logvar):
        """Computes the loss function."""

        # Reconstruction loss
        rec_loss = torch.sum(
            torch.flatten(self.criterion(recons, x), start_dim=1), dim=1
        )

        # Kl-Divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1)
        loss = rec_loss + self.beta * kl_loss
        return loss

    def train(self, epochs, save_model, out):
        """Trains the CVaR VAE using the training set

        Parameters
        ----------
        epochs: int
            Number of epochs to train
        save_model: bool
            If set to true, saves the model after training
        out: bool
            If set to true, outputs loss after each epoch
        """
        for epoch_idx in range(epochs):
            if out:
                print(f"Epoch {epoch_idx+1} of {epochs}")
            self.model.train()
            running_loss = 0.0
            for batch_idx, (data, idx) in enumerate(self.train_loader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                self.cvar.zero_grad()

                recons, mu, logvar = self.model(data)
                loss = self.loss(data, recons, mu, logvar)

                weights = 1.0
                prob = self.exp3.probabilities
                self.exp3.update(
                    1 - np.clip(loss.cpu().detach().numpy(), 0, 1), idx, prob
                )

                cvar_loss = (
                    torch.tensor(weights).float().to(self.device) * self.cvar(loss)
                ).mean()
                running_loss += loss.sum().item()
                cvar_loss.backward()

                self.optimizer.step()
                self.cvar.step()

                self.exp3.normalize()

            train_loss = running_loss / len(self.train_loader.dataset)
            val_loss = self.evaluate()

            if out:
                print(f"Train Loss: {train_loss:.4f}   Val Loss: {val_loss:.4f}")
            if save_model and epoch_idx % 50 == 0:
                torch.save(self.model, self.model_path)
        if save_model:
            torch.save(self.model, self.model_path)

    def evaluate(self):
        """Evaluates the CVaR VAE using the validation data.

        Returns
        -------
        val_loss: float
            The validation loss
        """
        self.model.eval()
        top_k = None
        count = 0
        running_loss = 0.0
        with torch.no_grad():
            for data, _ in self.val_loader:
                count += data.shape[0]
                data = data.to(self.device)
                recons, mu, logvar = self.model(data)
                losses = self.loss(data, recons, mu, logvar).sort(descending=True)[0]
                if top_k is None:
                    top_k = losses[: self.k]
                else:
                    top_k = (torch.cat((top_k, losses))).sort(descending=True)[0]
                    top_k = top_k[: self.k]
                running_loss += losses.sum().item()
            val_loss = running_loss / len(self.val_loader.dataset)
        return val_loss
