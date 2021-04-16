import numpy as np

from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

from adacvar.util.adaptive_algorithm import Hedge
from adacvar.util.learning_rate_decay import RobbinsMonro, AdaGrad

import matplotlib.pyplot as plt
from itertools import cycle, islice

import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

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

    def __init__(self, n_components, n_init, num_actions, size, lr=0.1, conv_rate=10.0, thres = 0.9):
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
            eta = 0.01
        )
        self.conv_rate = conv_rate
        self.threshold = np.round(thres * self.size)

    def fit_predict(self, data):
        self.gm.fit(data)
        score = self.gm.score_samples(data)
        min_clip = np.min(score)
        max_clip = np.max(score)

        idx = np.zeros(self.size)
        last_idx = np.ones(self.size)
        loss = []
        cvar_loss = []
        cnt = 0
        # while len(np.intersect1d(idx, last_idx)) < self.threshold:
        self.hedge.normalize()
        while not np.array_equal(idx, last_idx):
            cnt += 1
            last_idx = idx
            prob = self.hedge.probabilities
            prob /= np.sum(prob)
            # print(f"{min(prob)} {max(prob)}")
            idx = np.random.choice(
                np.arange(self.num_actions),
                self.size,
                p=prob,
                replace=False
            )
            idx = np.sort(idx)
            self.gm.fit(data[idx])
            # weighted log probabilities for each sample in X
            ll = self.gm.score_samples(data)
            reward = (ll-min_clip).clip(0)
            cvar_loss += [ np.sum(np.sort(ll)[:self.size]) ]
            loss += [-self.gm.score(data)]
            self.hedge.update(reward)
            self.hedge.normalize()
            if cnt % 20 == 0:
                print(str(cnt) + " " + str(loss[-1]) + " " + str(cvar_loss[-1]))
                ss = np.array([1 if cur <= 1e-4 else 0 for cur in self.hedge.probabilities])
                cur = np.count_nonzero(np.equal(idx, last_idx))
                print(str(np.sum(ss)) + " " + str(cur))
                print(str(min(self.hedge._kdpp.values)) + " " + str(max(self.hedge._kdpp.values)))
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
                cycle_nb = 5
                colors = np.array(list(islice(cycle(color_ar), cycle_nb)))
                # add black color for outliers (if any)
                # colors = np.append(colors, ["#000000"])
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
        return self.gm.predict(data), loss, cvar_loss, self.hedge.probabilities
