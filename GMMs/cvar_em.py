import numpy as np

from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

from adacvar.util.adaptive_algorithm import Hedge

import matplotlib.pyplot as plt
from itertools import cycle, islice


class CVarEM:
    """Implementation of the CVaR_EM algorithm.

    Parameters
    ----------
    n_components: int
        Number of clusters of the mixture.
    n_init: int
        Number of initializations for the Gaussian Mixture algorithm.
    num_actions: int
        Number of data points in dataset.
    size: int
        Number of data points to select at each iteration.
    lr: float
        Learning rate of the Hedge algorithm.

    Methods
    -------
    sample: Adaptive data sampler
    fit_predict: Fit the data using the Hedge algorithm and predicts output.
    """

    def __init__(self, n_components, n_init, num_actions, size, lr=0.2):
        self.num_actions = num_actions
        self.size = size
        self.gm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            tol=1e-3,
            max_iter=100,
            n_init=n_init,
            init_params="kmeans",  # kmeans or random
        )
        self.hedge = Hedge(
            num_actions=num_actions,
            size=size,
            eta=lr
        )

    def negative_log_likelihood(self, data):
        weights = self.gm.weights_
        means = self.gm.means_
        cov = self.gm.covariances_
        m = len(weights)
        n = len(data)
        
        ll = np.empty(n)
        for i in range(n):
            ll[i] = sum([weights[j] * multivariate_normal.pdf(data[i], means[j], cov[j]) for j in range(m)])
        ll = np.log(ll)
        return -ll

    def fit_predict(self, data):
        self.gm.fit(data)
        nll = self.negative_log_likelihood(data)
        clip_max = 5*np.max(nll)

        idx = np.zeros(self.size)
        last_idx = np.ones(self.size)
        loss = []
        cvar_loss = []
        cnt = 0
        while not np.array_equal(idx, last_idx):
            cnt += 1
            last_idx = idx
            prob = self.hedge.probabilities
            prob /= np.sum(prob)
            idx = np.random.choice(
                np.arange(self.num_actions),
                self.size,
                p=prob,
                replace=False
            )
            idx = np.sort(idx)
            self.gm.fit(data[idx])
            # weighted log probabilities for each sample in X
            nll = self.negative_log_likelihood(data)
            cvar_loss += [ np.sum(np.sort(nll)[-self.size:]) ]
            # print("CVar Loss: " + str(cvar_loss[-1]))
            cost = nll / clip_max
            cost = np.clip(cost, 0, 1) 
            loss += [-self.gm.score(data)]
            # loss += [np.sum(nll)]
            # print(abs(np.sum(nll) + self.gm.score(X)))
            self.hedge.update(cost)
            self.hedge.normalize()
            if cnt % 20 == 0:
                print(str(cnt) + " " + str(loss[-1]) + " " + str(cvar_loss[-1]))
                ss = np.array([1 if cur <= 1e-4 else 0 for cur in self.hedge.probabilities])
                print(np.sum(ss))
                color_ar = [
                    "#377eb8",
                    "#ff7f00",
                    "#4daf4a",
                    "#f781bf",
                    "#a65628",
                    "#984ea3",
                    "#999999",
                    "#e41a1c",
                    "#dede00",
                ]
                cycle_nb = 3
                colors = np.array(list(islice(cycle(color_ar), cycle_nb)))
                # add black color for outliers (if any)
                colors = np.append(colors, ["#000000"])
                cvar_y = self.gm.predict(data)
                plt.scatter(data[:, 0], data[:, 1], s=10, color=colors[cvar_y])
                plt.savefig("output/" + str(cnt) + ".png")
                plt.clf()

                plt.plot(loss)
                plt.savefig("loss/" + str(cnt) + ".png")
                plt.clf()
                plt.plot(cvar_loss)
                plt.savefig("cvar_loss/" + str(cnt) + ".png")
                plt.clf()
                plt.plot(self.hedge.probabilities)
                plt.savefig("weights/" + str(cnt) + ".png")
                plt.clf()

        print(cnt)
        return self.gm.predict(data), loss, self.hedge.probabilities
