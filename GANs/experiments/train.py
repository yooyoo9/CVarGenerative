import numpy as np
import os
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from adacvar.util.cvar import CVaR
from adacvar.util.adaptive_algorithm import Exp3Sampler

from gan import Discriminator, Generator

class GanAlg:
    def __init__(
        self,
        path_D,
        path_G,
        model_param,
        train_set,
        batch_size,
        lr,
        criterion,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path_D = path_D
        self.path_G = path_G

        if os.path.exists(path_D):
            self.model_D = torch.load(path_D, map_location=torch.device("cpu"))
            self.model_G = torch.load(path_G, map_location=torch.device("cpu"))
        else:
            self.model_D = Discriminator(
                    x_dim=model_param["x_dim"],
                    hidden_dims=model_param["hidden_dims_D"],
                )

            self.model_G = Generator(
                    z_dim=model_param["z_dim"],
                    hidden_dims=model_param["hidden_dims_G"],
                    x_dim=model_param["x_dim"],
                )
        self.model_D.to(self.device)
        self.model_G.to(self.device)

        self.z_dim = model_param["z_dim"]
        self.train_loader = DataLoader(train_set, batch_size)
        self.optimizer_D = optim.Adam(self.model_D.parameters(), lr=lr)
        self.optimizer_G = optim.Adam(self.model_G.parameters(), lr=lr)
        self.criterion = criterion

    def loss(self, output, real, size):
        if real == 1:
            real_labels = torch.ones(size, 1)
        else:
            real_labels = torch.zeros(size, 1)
        real_labels = real_labels.to(self.device)
        return self.criterion(output, real_labels)
    
    def train(self, epochs, save_model, out):
        losses_D = []
        losses_G = []
        for epoch_idx in range(epochs):
            if out:
                print(f"Epoch {epoch_idx+1} of {epochs}")
            running_loss_D = 0.0
            running_loss_G = 0.0
            for batch_idx, (data, *_) in enumerate(self.train_loader):
                data = data.to(self.device)
                batch_size = data.shape[0]

                # Train Discriminator
                self.optimizer_D.zero_grad()
                output_real = self.model_D(data)
                loss_D = torch.sum(self.loss(output_real, 1, batch_size))

                z = Variable(torch.randn(batch_size, self.z_dim).to(self.device))
                output_fake = self.model_D(self.model_G(z))
                loss_D += torch.sum(self.loss(output_fake, 0, batch_size))
                running_loss_D += loss_D.item()
                loss_D.backward()
                self.optimizer_D.step()
                

                # Train Generator
                self.optimizer_G.zero_grad()
                z = Variable(torch.randn(batch_size, self.z_dim).to(self.device))
                output_G = self.model_G(z)
                output_D = self.model_D(output_G)
                loss_G = torch.sum(self.loss(output_D, 1, batch_size))
                running_loss_G += loss_G.item()
                loss_G.backward()
                self.optimizer_G.step()
            train_loss_D = running_loss_D / len(self.train_loader.dataset)
            train_loss_G = running_loss_G / len(self.train_loader.dataset)
            losses_D.append(train_loss_D)
            losses_G.append(train_loss_G)

            if out:
                print(f"Loss D: {train_loss_D:.4f}   Loss G: {train_loss_G:.4f}")
            if save_model and epoch_idx % 50 == 0:
                torch.save(self.model_D, self.path_D)
                torch.save(self.model_G, self.path_G)
        if save_model:
            torch.save(self.model_D, self.path_D)
            torch.save(self.model_G, self.path_G)
            np.save("loss_D.npy", np.array(losses_D))
            np.save("loss_G.npy", np.array(losses_G))


class AdaCVar(GanAlg):
    def __init__(
        self,
        path_D,
        path_G,
        model_param,
        exp3_param,
        train_set,
        batch_size,
        lr,
        criterion,
        alpha,
    ):
        super().__init__(
            path_D,
            path_G,
            model_param,
            train_set,
            batch_size,
            lr,
            criterion,
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
        self.k = int(np.ceil(alpha * len(self.train_loader.dataset)))

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
        losses_D = []
        losses_G = []
        for epoch_idx in range(epochs):
            if out:
                print(f"Epoch {epoch_idx+1} of {epochs}")
            running_loss_D = 0.0
            running_loss_G = 0.0
            for batch_idx, (data, idx) in enumerate(self.train_loader):
                data = data.to(self.device)
                batch_size = data.shape[0]

                # Train Discriminator
                self.optimizer_D.zero_grad()
                self.cvar.zero_grad()
                output_real = self.model_D(data)
                loss_D = self.loss(output_real, 1, batch_size)

                # Update AdaCVaR based on output of discriminator
                weights = 1.0
                prob = self.exp3.probabilities
                self.exp3.update(
                    1 - np.clip(loss_D.cpu().detach().numpy().reshape(batch_size), 0, 1), idx, prob
                )

                cvar_loss = (
                    torch.tensor(weights).float().to(self.device) * self.cvar(loss_D)
                ).mean()
                cvar_loss.backward(retain_graph=True)
                self.cvar.step()
                self.exp3.normalize()

                loss_D = torch.sum(loss_D)

                z = Variable(torch.randn(batch_size, self.z_dim).to(self.device))
                output_fake = self.model_D(self.model_G(z))
                loss_D += torch.sum(self.loss(output_fake, 0, batch_size))
                running_loss_D += loss_D.item()
                loss_D.backward()
                self.optimizer_D.step()

                # Train Generator
                self.optimizer_G.zero_grad()
                z = Variable(torch.randn(batch_size, self.z_dim).to(self.device))
                output_G = self.model_G(z)
                output_D = self.model_D(output_G)
                loss_G = torch.sum(self.loss(output_D, 1, batch_size))
                running_loss_G += loss_G.item()
                loss_G.backward()
                self.optimizer_G.step()

            train_loss_D = running_loss_D / len(self.train_loader.dataset)
            train_loss_G = running_loss_G / len(self.train_loader.dataset)
            losses_D.append(train_loss_D)
            losses_G.append(train_loss_G)

            if out:
                print(f"Loss D: {train_loss_D:.4f}   Loss G: {train_loss_G:.4f}")
            if save_model and epoch_idx % 50 == 0:
                torch.save(self.model_D, self.path_D)
                torch.save(self.model_G, self.path_G)
        if save_model:
            torch.save(self.model_D, self.path_D)
            torch.save(self.model_G, self.path_G)
            np.save("loss_ada_D.npy", np.array(losses_D))
            np.save("loss_ada_G.npy", np.array(losses_G))
            losses_D = []
            losses_G = []
