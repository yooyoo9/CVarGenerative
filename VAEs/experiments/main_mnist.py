import numpy as np
import os
import torch

from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.utils import save_image
from sklearn.cluster import KMeans

from cvarVAE.train import VaeAlg, Rockarfellar, AdaCVar

seed = 31415
np.random.seed(seed)
torch.manual_seed(seed)


class ImbalancedMNIST(Dataset):
    def __init__(self, root, train, download, transform):
        self.dataset = datasets.MNIST(root=root, download=download, transform=transform)
        self.train = train
        self.class_distr = np.array([0.8 ** i for i in range(10)])
        self.class_distr /= np.sum(self.class_distr)
        self.idx = self.resample()

    def resample(self):
        targets = self.dataset.train_labels if self.train else self.dataset.test_labels
        _, class_counts = np.unique(targets, return_counts=True)
        # Get class indices for resampling
        class_indices = [np.where(targets == i)[0] for i in range(10)]
        # Get class indices for reduced class count
        idx = []
        for i in range(10):
            cur_count = int(class_counts[i] * self.class_distr[i])
            idx.append(class_indices[i][:cur_count])
        idx = np.hstack(idx)
        np.random.shuffle(idx)
        return idx

    def __getitem__(self, index):
        img, target = self.dataset[self.idx[index]]
        return img, target

    def __len__(self):
        return len(self.idx)


# learning params
model_param = {
    "x_dim": 1,
    "hidden_dims": [512, 512],  # TODO
    "z_dim": 16,
    "constrained_output": True,
}

exp3_param = {"gamma": 0.9, "beta": 0.0, "eps": 0.0, "iid_batch": False}

param = {
    "imbalanced": False,
    "epochs": 10000,
    "batch_size": 256,
    "lr": 1e-4,
    "alpha": 0.3,
    "beta": 1.0,
    "print": True,
    "model_name": "VaeImg",
    "model_name_usual": "VAE usual",
    "model_name_rockar": "Rockarfellar alg",
    "model_name_ada": "AdaCVar alg",
    "save_model": True,
    "dir": ["../models/mnist_usual/", "../output/out_mnist_usual/", "../input/mnist/"],
    "path_data": "../input/mnist/",
    "path_vae": "../models/mnist_usual/vae",
    "path_rockar": "../models/mnist_usual/rockar",
    "path_ada": "../models/mnist_usual/ada",
    "path_out": "../output/out_mnist_usual/",
}
criterion = torch.nn.MSELoss(reduction="none")

# Create directories for the output if they do not exist
for cur_dir in param["dir"]:
    if not os.path.exists(cur_dir):
        os.makedirs(cur_dir)

# Generate data
if param["imbalanced"]:
    train_set = ImbalancedMNIST(
        root=param["path_data"],
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    valid_set = ImbalancedMNIST(
        root=param["path_data"],
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
else:
    train_set = datasets.MNIST(
        root=param["path_data"],
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    valid_set = datasets.MNIST(
        root=param["path_data"],
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

vae = VaeAlg(
    param["model_name"],
    param["path_vae"],
    model_param,
    train_set,
    valid_set,
    param["batch_size"],
    param["lr"],
    criterion,
    beta=param["beta"],
)

rockar = Rockarfellar(
    param["model_name"],
    param["path_rockar"],
    model_param,
    train_set,
    valid_set,
    param["batch_size"],
    param["lr"],
    criterion,
    param["alpha"],
    beta=param["beta"],
)

ada = AdaCVar(
    param["model_name"],
    param["path_ada"],
    model_param,
    exp3_param,
    train_set,
    valid_set,
    param["batch_size"],
    param["lr"],
    criterion,
    param["alpha"],
    param["beta"],
)


def classify(data):
    pred = KMeans(n_clusters=10).fit_predict(data)
    res = np.empty(data.shape)
    cur = 0
    for i in range(10):
        idx = np.argwhere(pred == i)[:, 0]
        res[cur : cur + len(idx)] = data[idx]
        cur += len(idx)
    return res


for i in range(param["epochs"] // 50):
    vae.train(50, param["save_model"], param["print"])
    rockar.train(50, param["save_model"], param["print"])
    ada.train(50, param["save_model"], param["print"])

    vae.model.eval()
    rockar.model.eval()
    ada.model.eval()
    with torch.no_grad():
        recons = vae.model.sample(64, vae.device).cpu().numpy()
        recons_rockar = rockar.model.sample(64, rockar.device).cpu().numpy()
        recons_ada = rockar.model.sample(64, rockar.device).cpu().numpy()

        recons = classify(recons.reshape((64, -1)))
        recons_rockar = classify(recons_rockar.reshape((64, -1)))
        recons_ada = classify(recons_ada.reshape((64, -1)))

        vae_img = torch.from_numpy(recons).view(64, 1, 28, 28)
        rockar_img = torch.from_numpy(recons_rockar).view(64, 1, 28, 28)
        ada_img = torch.from_numpy(recons_ada).view(64, 1, 28, 28)

        save_image(vae_img, f"{param['path_out']}vae.png", nrow=8)
        save_image(rockar_img, f"{param['path_out']}rockarfellar.png", nrow=8)
        save_image(ada_img, f"{param['path_out']}adacvar.png", nrow=8)
