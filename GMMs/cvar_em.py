import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture

from adacvar.util.adaptive_algorithm import Hedge


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
    path: string
        Directory to save the output.

    Methods
    -------
    sample: Adaptive data sampler
    fit_predict: Fit the data using the Hedge algorithm and predicts output.
    generate_output: Compares the predictions with other algorithms and the true labels.
    """

    def __init__(self, n_components, n_init, num_actions, size, lr, path):
        self.num_actions = num_actions
        self.size = size
        self.gm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            tol=1e-3,
            max_iter=100,
            n_init=n_init,
            init_params="kmeans",
        )
        self.hedge = Hedge(num_actions=num_actions, size=size, eta=lr)
        self.path = path
        self.colors = np.array(
            [
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
        )

    def fit_predict(self, data, true_y, gmm_y, name):
        self.gm.fit(data)
        score = self.gm.score_samples(data)
        min_clip = np.min(score)
        max_clip = np.max(score)
        range_score = max_clip - min_clip

        loss = []
        cvar_loss = []
        prob = np.ones(len(data))
        best_idx, best_score = np.zeros(self.size), -1e8
        cnt = 0
        while np.linalg.norm(np.sort(prob)[-self.size :] - 1.0 / self.size) > 0.015:
            cnt += 1
            prob = self.hedge.probabilities
            prob /= np.sum(prob)
            idx = np.random.choice(
                np.arange(self.num_actions), self.size, p=prob, replace=False
            )
            idx = np.sort(idx)
            self.gm.fit(data[idx])
            # weighted log probabilities for each sample in X
            ll = self.gm.score_samples(data)
            reward = ((ll - min_clip) / range_score).clip(0, 1)
            cvar_loss += [np.mean(np.sort(ll)[: self.size])]
            loss += [-ll.mean()]
            self.hedge.update(reward)
            self.hedge.normalize()
            if cvar_loss[-1] > best_score:
                best_score = cvar_loss[-1]
                best_idx = idx
                cur = np.linalg.norm(np.sort(prob)[-self.size :] - 1.0 / self.size)
                print(str(cnt) + " " + str(cur))
                cvar_y = self.gm.predict(data)
                self.generate_output(data, true_y, gmm_y, cvar_y, loss, name)
        self.gm.fit(data[best_idx])

    def generate_output(self, data, true_y, gmm_y, cvar_y, loss, name):
        pred = np.array([true_y, gmm_y, cvar_y])
        np.save(self.path + "data" + str(0) + "_predictions.npy", pred)

        plt.figure(figsize=[10, 6])
        ax = plt.subplot(1, 3, 1)
        ax.set_title("True distribution")
        ax.axis("equal")
        ax.scatter(data[:, 0], data[:, 1], s=10, color=self.colors[true_y])
        ax1 = plt.subplot(1, 3, 2)
        ax1.set_title("GMM")
        ax1.axis("equal")
        ax1.scatter(data[:, 0], data[:, 1], s=10, color=self.colors[gmm_y])
        ax2 = plt.subplot(1, 3, 3)
        ax2.set_title("CVaR_EM")
        ax2.axis("equal")
        ax2.scatter(data[:, 0], data[:, 1], s=10, color=self.colors[cvar_y])
        plt.savefig(self.path + "data" + name + "_img.png")
        plt.clf()

        plt.plot(loss)
        plt.title("Cvar loss")
        plt.savefig(self.path + "data" + name + "_loss.png")
        plt.clf()

        plt.plot(self.hedge.probabilities)
        plt.title("Probabilities of datapoints")
        plt.savefig(self.path + "data" + name + "_weight.png")
        plt.clf()
