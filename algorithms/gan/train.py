import numpy as np
import os
import wandb
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture

from adacvar.util.cvar import CVaR
from adacvar.util.adaptive_algorithm import Exp3Sampler

from gan import Discriminator, Generator
from gan28 import Discriminator28, Generator28
from gan64 import Discriminator64, Generator64


class GanAlg:
    def __init__(
        self,
        gan_model,
        path_model,
        z_dim,
        train_set,
        valid_set,
        test_set,
        batch_size,
        lr,
        criterion,
        alpha,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not hasattr(self, "name"):
            self.name = "orig"
        self.path_D = os.path.join(path_model, self.name + "D")
        self.path_G = os.path.join(path_model, self.name + "G")
        self.gaussian = False
        self.alpha = alpha

        if os.path.exists(self.path_D):
            self.model_D = torch.load(self.path_D, map_location=torch.device("cpu"))
            self.model_G = torch.load(self.path_G, map_location=torch.device("cpu"))
        else:
            if gan_model == "gan":
                self.gaussian = True
                self.model_D = Discriminator()
                self.model_G = Generator(z_dim)
            elif gan_model == "img28":
                self.model_D = Discriminator28()
                self.model_G = Generator28(z_dim)
            else:
                self.model_D = Discriminator64()
                self.model_G = Generator64(z_dim)
        self.model_D.to(self.device)
        self.model_G.to(self.device)

        self.z_dim = z_dim
        self.train_loader = DataLoader(train_set, batch_size)
        self.val_loader = DataLoader(valid_set, batch_size, shuffle=False)
        self.test_loader = DataLoader(test_set, batch_size, shuffle=False)
        self.num_batches = len(self.train_loader)
        self.optimizer_D = optim.Adam(self.model_D.parameters(), lr=lr)
        self.optimizer_G = optim.Adam(self.model_G.parameters(), lr=lr)
        self.criterion = criterion

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def train(self, epochs, save_model):
        for epoch_idx in range(epochs):
            running_loss_D = running_loss_G = 0.0
            for batch_idx, (real_images, *_) in enumerate(self.train_loader):
                batch_size = real_images.shape[0]

                # Train Discriminator
                self.optimizer_D.zero_grad()
                real_images = real_images.to(self.device)
                output_real = self.model_D(real_images)
                label = torch.full(
                    (batch_size,), 1.0, dtype=torch.float, device=self.device
                )
                loss_D_real = torch.mean(self.criterion(output_real, label))
                loss_D_real.backward()

                if self.gaussian:
                    z = torch.randn(batch_size, self.z_dim, device=self.device)
                else:
                    z = torch.randn(batch_size, self.z_dim, 1, 1, device=self.device)
                fake_images = self.model_G(z)
                output_fake = self.model_D(fake_images.detach())
                label.fill_(0.0)
                loss_D_fake = torch.mean(self.criterion(output_fake, label))
                loss_D_fake.backward()
                loss_D = (loss_D_real + loss_D_fake).item()
                self.optimizer_D.step()

                # Train Generator
                self.optimizer_G.zero_grad()
                label.fill_(1.0)
                output_D = self.model_D(fake_images)
                loss_G = torch.mean(self.criterion(output_D, label))
                loss_G.backward()
                loss_G = loss_G.item()
                self.optimizer_G.step()
                running_loss_D += loss_D
                running_loss_G += loss_G
            running_loss_D /= self.num_batches
            running_loss_G /= self.num_batches
            mean, cvar, worst = self.evaluate(val=True)
            wandb.log({"loss_D": running_loss_D, "loss_G": running_loss_G})
            if self.gaussian:
                wandb.log({"val_mean": mean, "val_worst": worst, "val_cvar": cvar})
            if save_model and epoch_idx % 50 == 0:
                torch.save(self.model_D, self.path_D)
                torch.save(self.model_G, self.path_G)
        if save_model:
            torch.save(self.model_D, self.path_D)
            torch.save(self.model_G, self.path_G)

    def eval_gaussian(self, dataset, output_path):
        true_data = dataset.data
        n_clusters = int(dataset.n_clusters)
        gmm = GaussianMixture(
            n_components=n_clusters,
            n_init=20,
        )
        recons = (
            self.model_G.sample(true_data.shape[0], self.device).detach().cpu().numpy()
        )
        if output_path is not None:
            np.save(output_path, recons)
        fig = plt.figure()
        plt.scatter(recons[:, 0], recons[:, 1], s=1, color="black")
        wandb.log({"recons": wandb.Image(fig)})
        plt.close()
        gmm.fit(recons)
        scores = -gmm.score_samples(true_data)
        k = int(np.round(self.alpha * len(dataset)))
        cvar = np.mean(np.sort(scores)[-k:])
        worst = np.max(scores)
        mean = np.mean(scores)
        return mean, cvar, worst

    def create_recons(self):
        data = self.model_G.sample(64, self.device)
        img = np.transpose(make_grid(data, padding=5, normalize=True).cpu(), (1, 2, 0))
        fig = plt.figure()
        plt.imshow(img)
        wandb.log({"recons": wandb.Image(fig)})
        plt.close()

    def evaluate(self, val, output_path=None):
        """Evaluates the VAE using the validation data.

        Returns
        -------
        val_loss: float
            The validation loss
        """
        self.model_G.eval()
        dataloader = self.val_loader if val else self.test_loader
        mean, cvar, worst = None, None, None
        with torch.no_grad():
            if self.gaussian:
                mean, cvar, worst = self.eval_gaussian(dataloader.dataset, output_path)
            else:
                self.create_recons()
        self.model_G.train()
        return mean, cvar, worst


class TruncCVar(GanAlg):
    def __init__(
        self,
        gan_model,
        path_model,
        model_param,
        train_set,
        valid_set,
        test_set,
        batch_size,
        lr,
        criterion,
        alpha,
    ):
        if not hasattr(self, "name"):
            self.name = "trunc"
        super().__init__(
            gan_model,
            path_model,
            model_param,
            train_set,
            valid_set,
            test_set,
            batch_size,
            lr,
            criterion,
            alpha,
        )
        self.exp3 = None
        self.cvar = CVaR(alpha=alpha, learning_rate=lr).to(self.device)

    def train(self, epochs, save_model):
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
            running_loss_D = running_loss_G = 0.0
            nb = 0
            for batch_idx, (real_images, idx) in enumerate(self.train_loader):
                nb += 1
                batch_size = real_images.shape[0]

                # Train Discriminator
                self.optimizer_D.zero_grad()
                self.cvar.zero_grad()
                real_images = real_images.to(self.device)
                output_real = self.model_D(real_images)
                label = torch.full(
                    (batch_size,), 1.0, dtype=torch.float, device=self.device
                )
                loss_D_real = self.criterion(output_real, label)

                # Update AdaCVaR based on output of discriminator
                weights = 1.0
                if self.exp3 is not None:
                    prob = self.exp3.probabilities
                    self.exp3.update(
                        1
                        - np.clip(
                            loss_D_real.cpu().detach().numpy().reshape(batch_size), 0, 1
                        ),
                        idx,
                        prob,
                    )
                    self.exp3.normalize()

                loss_D = (
                    torch.tensor(weights).float().to(self.device)
                    * self.cvar(loss_D_real)
                ).sum()
                loss_D.backward()

                if self.gaussian:
                    z = torch.randn(batch_size, self.z_dim, device=self.device)
                else:
                    z = torch.randn(batch_size, self.z_dim, 1, 1, device=self.device)
                fake_images = self.model_G(z)
                output_fake = self.model_D(fake_images.detach())
                label.fill_(0.0)
                loss_D_fake = torch.mean(self.criterion(output_fake, label))
                loss_D_fake.backward()
                loss_D = (loss_D + loss_D_fake).item()
                self.optimizer_D.step()
                self.cvar.step()

                # Train Generator
                self.optimizer_G.zero_grad()
                label.fill_(1.0)
                output_D = self.model_D(fake_images)
                loss_G = torch.mean(self.criterion(output_D, label))
                loss_G.backward()
                loss_G = loss_G.item()
                self.optimizer_G.step()

                running_loss_D += loss_D
                running_loss_G += loss_G
            running_loss_D /= nb
            running_loss_G /= nb
            mean, cvar, worst = self.evaluate(val=True)
            wandb.log({"loss_D": running_loss_D, "loss_G": running_loss_G})
            if self.gaussian:
                wandb.log({"val_mean": mean, "val_worst": worst, "val_cvar": cvar})
            if save_model and epoch_idx % 50 == 0:
                torch.save(self.model_D, self.path_D)
                torch.save(self.model_G, self.path_G)
        if save_model:
            torch.save(self.model_D, self.path_D)
            torch.save(self.model_G, self.path_G)


class AdaCVar(TruncCVar):
    def __init__(
        self,
        gan_model,
        path_model,
        model_param,
        exp3_param,
        train_set,
        valid_set,
        test_set,
        batch_size,
        lr,
        criterion,
        alpha,
    ):
        self.name = "ada"
        super().__init__(
            gan_model,
            path_model,
            model_param,
            train_set,
            valid_set,
            test_set,
            batch_size,
            lr,
            criterion,
            alpha,
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
