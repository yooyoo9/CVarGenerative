import numpy as np
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

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
        path_output,
        z_dim,
        train_set,
        batch_size,
        lr,
        criterion,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not hasattr(self, 'name'):
            self.name = "orig"
        self.path_D = path_model + self.name + "D"
        self.path_G = path_model + self.name + "G"
        self.path_output = path_output

        if os.path.exists(self.path_D):
            self.model_D = torch.load(self.path_D, map_location=torch.device("cpu"))
            self.model_G = torch.load(self.path_G, map_location=torch.device("cpu"))
        else:
            if gan_model == 'gan':
                self.model_D = Discriminator()
                self.model_G = Generator(z_dim)
            elif gan_model == 'img28':
                self.model_D = Discriminator28()
                self.model_G = Generator28(z_dim)
            else:
                self.model_D = Discriminator64()
                self.model_G = Generator64(z_dim)
            self.model_D.apply(self.weights_init)
            self.model_G.apply(self.weights_init)
        self.model_D.to(self.device)
        self.model_G.to(self.device)

        self.z_dim = z_dim
        self.train_loader = DataLoader(train_set, batch_size)
        self.optimizer_D = optim.Adam(self.model_D.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_G = optim.Adam(self.model_G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.criterion = criterion

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def train(self, epochs, save_model, out):
        losses_D = []
        losses_G = []
        for epoch_idx in range(epochs):
            if out:
                print(f"Epoch {epoch_idx+1} of {epochs}")
            running_loss_D = running_loss_G = 0.0
            nb = 0
            for batch_idx, (real_images, *_) in enumerate(self.train_loader):
                nb += 1
                batch_size = real_images.shape[0]

                # Train Discriminator
                self.optimizer_D.zero_grad()
                real_images = real_images.to(self.device)
                output_real = self.model_D(real_images)
                label = torch.full((batch_size,), 1., dtype=torch.float, device=self.device)
                loss_D_real = torch.mean(self.criterion(output_real, label))
                loss_D_real.backward()

                z = torch.randn(batch_size, self.z_dim, 1, 1, device=self.device)
                fake_images = self.model_G(z)
                output_fake = self.model_D(fake_images.detach())
                label.fill_(0.)
                loss_D_fake = torch.mean(self.criterion(output_fake, label))
                loss_D_fake.backward()
                loss_D = (loss_D_real + loss_D_fake).item()
                self.optimizer_D.step()
                

                # Train Generator
                self.optimizer_G.zero_grad()
                label.fill_(1.)
                output_D = self.model_D(fake_images)
                loss_G = torch.mean(self.criterion(output_D, label))
                loss_G.backward()
                loss_G = loss_G.item()
                self.optimizer_G.step()
                running_loss_D += loss_D
                running_loss_G += loss_G
            losses_D.append(running_loss_D / nb)
            losses_G.append(running_loss_G / nb)

            if out:
                print(f"Loss D: {loss_D:.4f}   Loss G: {loss_G:.4f}")
            if save_model and epoch_idx % 50 == 0:
                torch.save(self.model_D, self.path_D)
                torch.save(self.model_G, self.path_G)
        if save_model:
            torch.save(self.model_D, self.path_D)
            torch.save(self.model_G, self.path_G)
            np.save(self.path_output + "lossD_"+ self.name+".npy", np.array(losses_D))
            np.save(self.path_output + "lossG_"+ self.name+".npy", np.array(losses_G))


class TruncCVar(GanAlg):
    def __init__(
        self,
        path_model,
        path_output,
        model_param,
        train_set,
        batch_size,
        lr,
        criterion,
        alpha,
    ):
        if not hasattr(self, "name"):
            self.name = "rocka"
        super().__init__(
            path_model,
            path_output,
            model_param,
            train_set,
            batch_size,
            lr,
            criterion,
        )
        self.exp3 = None
        self.cvar = CVaR(alpha=alpha, learning_rate=lr).to(self.device)

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
            running_loss_D = running_loss_G = 0.0
            nb = 0
            for batch_idx, (real_images, idx, _) in enumerate(self.train_loader):
                nb += 1
                batch_size = real_images.shape[0]

                # Train Discriminator
                self.optimizer_D.zero_grad()
                self.cvar.zero_grad()
                real_images = real_images.to(self.device)
                output_real = self.model_D(real_images)
                label = torch.full((batch_size,), 1., dtype=torch.float, device=self.device)
                loss_D_real = self.criterion(output_real, label)

                # Update AdaCVaR based on output of discriminator
                weights = 1.0
                if self.exp3 is not None:
                    prob = self.exp3.probabilities
                    self.exp3.update(
                        1 - np.clip(loss_D_real.cpu().detach().numpy().reshape(batch_size), 0, 1), idx, prob
                    )
                    self.exp3.normalize()

                loss_D = (
                    torch.tensor(weights).float().to(self.device) * self.cvar(loss_D_real)
                ).mean()
                loss_D.backward()

                z = torch.randn(batch_size, self.z_dim, 1, 1, device=self.device)
                fake_images = self.model_G(z)
                output_fake = self.model_D(fake_images.detach())
                label.fill_(0.)
                loss_D_fake = torch.mean(self.criterion(output_fake, label))
                loss_D_fake.backward()
                loss_D = (loss_D + loss_D_fake).item()
                self.optimizer_D.step()
                self.cvar.step()

                # Train Generator
                self.optimizer_G.zero_grad()
                label.fill_(1.)
                output_D = self.model_D(fake_images)
                loss_G = torch.mean(self.criterion(output_D, label))
                loss_G.backward()
                loss_G = loss_G.item()
                self.optimizer_G.step()

                running_loss_D += loss_D
                running_loss_G += loss_G
            losses_D.append(running_loss_D / nb)
            losses_G.append(running_loss_G / nb)

            if out:
                print(f"Loss D: {loss_D:.4f}   Loss G: {loss_G:.4f}")
            if save_model and epoch_idx % 50 == 0:
                torch.save(self.model_D, self.path_D)
                torch.save(self.model_G, self.path_G)
        if save_model:
            torch.save(self.model_D, self.path_D)
            torch.save(self.model_G, self.path_G)
            np.save(self.path_output + "lossD_"+ self.name+".npy", np.array(losses_D))
            np.save(self.path_output + "lossG_"+ self.name+".npy", np.array(losses_G))


class AdaCVar(TruncCVar):
    def __init__(
        self,
        path_model,
        path_output,
        model_param,
        exp3_param,
        train_set,
        batch_size,
        lr,
        criterion,
        alpha,
    ):
        self.name = "ada"
        super().__init__(
            path_model,
            path_output,
            model_param,
            train_set,
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
