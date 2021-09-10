import numpy as np
import os
import argparse
import wandb

from sklearn.mixture import GaussianMixture

from cvar_em import CVarEM

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=int, default=0)
parser.add_argument('--alpha', type=float, default=0.3)
parser.add_argument('--lr_hedge', type=float, default=0.1)
parser.add_argument("--n_init", type=int, default=100)
parser.add_argument("--n_init_cvar", type=int, default=50)
parser.add_argument('--stopping_threshold', type=int, default=1)
parser.add_argument("--reproduce", action='store_true', default=False)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

wandb.init(project="cvar-generative", entity="yooyoo9", config=args)
np.random.seed(args.seed)

if args.reproduce:
    thres = [1000, 1000, 3000, 1000, 1000, 1000, 1000, 1000, 2000, 1000]
    lr = [0.5, 0.5, 0.01, 0.01, 0.5, 0.5, 0.5, 0.5, 0.01, 0.01]
    alphas = [0.1, 0.1, 0.3, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    datasets = np.arange(10)
else:
    datasets = [args.dataset]

output_path = 'experiments/synthetic/gmm'
if not os.path.exists(output_path):
    os.makedirs(output_path)

data_source = 'experiments/synthetic/input'
path_X = os.path.join(data_source, "X1200.npy")
path_y = os.path.join(data_source, "Y1200.npy")
X = np.load(path_X)
y = (np.load(path_y)).astype(int)

for i in datasets:
    if args.reproduce:
        args.stopping_threshold = thres[i]
        args.lr_hedge = lr[i]
        args.alpha = alphas[i]
    name = str(i)
    curX = X[i, :-1]
    cur_y = y[i]
    idx_array = np.random.permutation(len(curX))
    n = len(curX) // 3
    curX, cur_y = curX[idx_array], cur_y[idx_array]
    X_train, X_val = curX[:n], curX[n : 2 * n]
    X_test, y_test = curX[2 * n:], cur_y[2 * n:]

    n_clusters = int(X[i, -1, 0])
    cvar = CVarEM(
        n_components=n_clusters,
        n_init=args.n_init_cvar,
        num_actions=n,
        size=int(np.ceil(args.alpha * n)),
        val_size=int(np.ceil(args.alpha * n)),
        test_size=int(np.ceil(args.alpha * n)),
        lr=args.lr_hedge,
        threshold=args.stopping_threshold,
    )

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        tol=1e-3,
        max_iter=100,
        n_init=args.n_init,
        init_params="kmeans",
    )

    gmm.fit(X_train)
    gmm_y = gmm.predict(X_test)
    nll = -gmm.score_samples(X_test)
    k_test = int(np.ceil(args.alpha * n))
    wandb.log(
        {
            "EM current loss": np.mean(nll),
            "EM worst loss": np.max(nll),
            "EM cvar loss": np.mean(np.sort(nll)[-k_test:]),
        }
    )

    y_pred = cvar.fit_predict(X_train, X_val, X_test)

    results = np.empty((len(X_test), len(X_test[0]) + 3))
    results[:, :-3] = X_test
    results[:, -3], results[:, -2], results[:, -1] = y_test, gmm_y, y_pred
    np.save(os.path.join(output_path, "data" + name + "_predictions.npy"), results)
