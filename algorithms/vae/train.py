import numpy as np
import os
import torch
import wandb
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from sklearn.mixture import GaussianMixture

from adacvar.util.cvar import CVaR
from adacvar.util.adaptive_algorithm import Exp3Sampler

from vae import VAE
from vae_img import VaeImg


class VaeAlg:
    def __init__(
        self,
        model_name,
        model_path,
        model_param,
        train_set,
        valid_set,
        test_set,
        batch_size,
        optim_param,
        criterion,
        alpha,
        beta,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not hasattr(self, "name"):
            self.name = "orig"
        self.model_path = os.path.join(model_path, self.name)

        if os.path.exists(self.model_path):
            self.model = torch.load(self.model_path, map_location=torch.device("cpu"))
        else:
            if model_name == "VAE":
                self.gaussian = True
                self.model = VAE(
                    x_dim=model_param["x_dim"],
                    hidden_dims=model_param["hidden_dims"],
                    z_dim=model_param["z_dim"],
                )
            else:
                self.gaussian = False
                self.img_size = model_param["img_size"]
                self.model = VaeImg(
                    n_channel=model_param["n_channel"],
                    hidden_dims=model_param["hidden_dims"],
                    z_dim=model_param["z_dim"],
                    img_size=model_param["img_size"],
                )
        self.model.to(self.device)

        self.train_loader = DataLoader(train_set, batch_size, shuffle=True)
        self.val_loader = DataLoader(valid_set, batch_size, shuffle=False)
        self.test_loader = DataLoader(test_set, batch_size, shuffle=False)
        if optim_param['name'] == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=optim_param['lr'])
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=optim_param['lr'], momentum=optim_param['momentum'])
        self.criterion = criterion
        self.alpha = alpha
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
        return loss

    def train(self, epochs, early_stop, save_model=False):
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
        epochs_no_improve = 0
        min_val_loss = 1e8
        for epoch_idx in range(epochs):
            running_loss = 0.0
            for batch_idx, (data, *_) in enumerate(self.train_loader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                recons, mu, logvar = self.model(data)
                loss = torch.sum(self.loss(data, recons, mu, logvar))
                running_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            train_loss = running_loss / len(self.train_loader.dataset)
            val_loss, val_ll_true, val_ll_cur, val_cvar = self.evaluate(val=True)
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss
            })
            if self.gaussian:
                wandb.log({
                    'val_nll_true': val_ll_true,
                    'val_nll_cur': val_ll_cur,
                    'val_cvar': val_cvar
                })
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve == early_stop:
                return
        if save_model:
            torch.save(self.model, self.model_path)

    def eval_gaussian(self, dataset, output_path):
        true_data = dataset.data
        n_clusters = int(dataset.n_clusters)
        gmm = GaussianMixture(
            n_components=n_clusters,
            n_init=20,
        )
        recons = self.model.sample(true_data.shape[0], self.device).detach().cpu().numpy()
        if output_path is not None:
            np.save(output_path, recons)
        fig = plt.figure()
        plt.scatter(recons[:, 0], recons[:, 1], s=1, color="black")
        wandb.log({'recons': wandb.Image(fig)})
        plt.close()
        gmm.fit(recons)
        nll_true = np.mean(-gmm.score_samples(true_data))
        nll_cur = np.mean(-gmm.score_samples(recons))
        k = int(np.round(self.alpha * len(dataset)))
        cvar_cur = np.mean((-gmm.score_samples(recons))[-k:])
        return nll_true, nll_cur, cvar_cur

    def create_recons(self):
        with torch.no_grad():
            data = self.model.sample(64, self.device)
        img = np.transpose(make_grid(data, padding=5, normalize=True).cpu(), (1, 2, 0))
        fig = plt.figure()
        plt.imshow(img)
        wandb.log({'recons': wandb.Image(fig)})
        plt.close()

    def evaluate(self, val, output_path=None):
        """Evaluates the VAE using the validation data.

        Returns
        -------
        val_loss: float
            The validation loss
        """
        self.model.eval()
        dataloader = self.val_loader if val else self.test_loader
        running_loss = 0.0
        nll_true, nll_cur, cvar_cur = None, None, None
        with torch.no_grad():
            for (data, *_) in dataloader:
                data = data.to(self.device)
                recons, mu, logvar = self.model(data)
                loss = torch.sum(self.loss(data, recons, mu, logvar))
                running_loss += loss.item()
            val_loss = running_loss / len(dataloader.dataset)

            if self.gaussian:
                nll_true, nll_cur, cvar_cur = self.eval_gaussian(dataloader.dataset, output_path)
            elif np.random.randint(10) == 0:
                self.create_recons()
        return val_loss, nll_true, nll_cur, cvar_cur


class TruncCVar(VaeAlg):
    def __init__(
        self,
        model_name,
        model_path,
        model_param,
        train_set,
        valid_set,
        test_set,
        batch_size,
        optim_param,
        criterion,
        alpha,
        beta,
    ):
        if not hasattr(self, "name"):
            self.name = "trunc"
        super().__init__(
            model_name,
            model_path,
            model_param,
            train_set,
            valid_set,
            test_set,
            batch_size,
            optim_param,
            criterion,
            alpha,
            beta,
        )
        self.exp3 = None
        self.cvar = CVaR(alpha=alpha, learning_rate=optim_param['lr']).to(self.device)

    def train(self, epochs, early_stop, save_model):
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
        self.model.train()
        epochs_no_improve = 0
        min_val_loss = 1e8
        for epoch_idx in range(epochs):
            running_loss = 0.0
            for batch_idx, (data, idx) in enumerate(self.train_loader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                self.cvar.zero_grad()

                recons, mu, logvar = self.model(data)
                loss = self.loss(data, recons, mu, logvar)

                weights = 1.0
                if self.exp3 is not None:
                    prob = self.exp3.probabilities
                    self.exp3.update(
                        1 - np.clip(loss.cpu().detach().numpy(), 0, 1), idx, prob
                    )
                    self.exp3.normalize()

                cvar_loss = (
                    torch.tensor(weights).float().to(self.device) * self.cvar(loss)
                ).mean()
                cvar_loss.backward()

                self.optimizer.step()
                self.cvar.step()

                running_loss += loss.sum().item()
            train_loss = running_loss / len(self.train_loader.dataset)
            val_loss, val_ll_true, val_ll_cur, val_cvar = self.evaluate(val=True)
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss
            })
            if self.gaussian:
                wandb.log({
                    'val_nll_true': val_ll_true,
                    'val_nll_cur': val_ll_cur,
                    'val_cvar': val_cvar
                })
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve == early_stop:
                return
        if save_model:
            torch.save(self.model, self.model_path)

    def evaluate(self, val, output_path=None):
        """Evaluates the CVaR VAE using the validation data.

        Returns
        -------
        val_loss: float
            The validation loss
        """
        dataloader = self.val_loader if val else self.test_loader
        self.model.eval()
        nll_true, nll_cur, cvar_cur = None, None, None
        running_loss = 0.0
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(self.device)
                recons, mu, logvar = self.model(data)
                loss = self.loss(data, recons, mu, logvar).sum()
                running_loss += loss.item()
        val_loss = running_loss / len(dataloader.dataset)

        if self.gaussian:
            nll_true, nll_cur, cvar_cur = self.eval_gaussian(dataloader.dataset, output_path)
        elif np.random.randint(10) == 0:
            self.create_recons()
        return val_loss, nll_true, nll_cur, cvar_cur


class AdaCVar(TruncCVar):
    def __init__(
        self,
        model_name,
        model_path,
        model_param,
        exp3_param,
        train_set,
        valid_set,
        test_set,
        batch_size,
        optim_param,
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
            test_set,
            batch_size,
            optim_param,
            criterion,
            alpha,
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
