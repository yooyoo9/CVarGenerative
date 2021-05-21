import numpy as np
import os

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
    "eps": [0.017, 0.031, 0.024, 0.028, 0.025],
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

for i in range(len(X)):
    print(i)
    name = str(i)
    curX = X[i, :-1]
    cur_y = y[i]
    n_clusters = int(X[i, -1, 0])

    k = int(np.ceil(param["alpha"] * param["n_samples"]))
    cvar = CVarEM(
        n_components=n_clusters,
        n_init=param["n_init_cvar"],
        num_actions=param["n_samples"],
        size=k,
        lr=param["lr_hedge"],
        eps=param["eps"][i],
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
    nll = -gmm.score_samples(curX)
    best, cvar_loss = cvar.fit_predict(curX)

    np.save(param["path_out"] + "cvar_loss" + name + ".npy", cvar_loss)
    np.save(param["path_out"] + "prob" + name + ".npy", best["prob"])

    pred = np.array([cur_y, gmm_y, best["pred"]])
    np.save(param["path_out"] + "data" + name + "_predictions.npy", pred)

    output_file = open(param["path_out"] + "output" + name + ".txt", "w")
    output_file.write("EM-algorithm\n")
    output_file.write("Current loss: {:.3f}\n".format(np.mean(nll)))
    output_file.write("Worst loss: {:.3f}\n".format(np.max(nll)))
    output_file.write("Average k-losses: {:.3f}\n\n".format(np.mean(np.sort(nll)[-k:])))
    output_file.write("CVaR-EM\n")
    output_file.write("Current loss: {:.3f}\n".format(best["loss"]))
    output_file.write("Worst loss: {:.3f}\n".format(best["worst_loss"]))
    output_file.write("Average k-losses: {:.3f}\n".format(best["cvar_loss"]))
    output_file.close()
