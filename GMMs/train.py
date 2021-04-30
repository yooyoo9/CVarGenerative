import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture

from cvar_em import CVarEM
from generate_data import generate_data

np.random.seed(31415)

param = {
    "alpha": 0.3,
    "lr_hedge": 0.1,
    "n_samples": 400,
    "n_init": 100,
    "n_init_cvar": 50,
    "dir": ["data", "output"],
    "path_X": "data/data_X.npy",
    "path_y": "data/data_y.npy",
    "path_out": "output/",
}

# Create directories for the output if they do not exist
for cur_dir in param["dir"]:
    if not os.path.exists(cur_dir):
        os.makedirs(cur_dir)

# Check if data already present, if not generate
if not os.path.isfile(param["path_X"]):
    generate_data(param["n_samples"], param["path_X"], param["path_y"])
X = np.load(param["path_X"])
y = (np.load(param["path_y"])).astype(int)

plt.figure(figsize=[10, 6])
# for i in range(len(X)):
for i in [0]:
    print(i)
    curX = X[i, :-1]
    cur_y = y[i]
    n_clusters = int(X[i, -1, 0])

    cvar = CVarEM(
        n_components=n_clusters,
        n_init=param["n_init_cvar"],
        num_actions=param["n_samples"],
        size=int(np.ceil(param["alpha"] * param["n_samples"])),
        lr=param["lr_hedge"],
        path=param["path_out"],
    )

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        tol=1e-3,
        max_iter=100,
        n_init=param["n_init"],
        init_params="kmeans",
    )

    gmm_y = gmm.fit_predict(curX)
    cvar.fit_predict(curX, cur_y, gmm_y, str(i))
