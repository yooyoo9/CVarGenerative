import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture

from adacvar.util.adaptive_algorithm import Hedge


class CVarEM:
    """Implementation of the CVaR-EM algorithm.

    Parameters
    ----------
    n_components: int
        Number of clusters of the mixture.
    n_init: int
        Number of initializations for the Gaussian Mixture algorithm.
    num_actions: int
        Number of data points in dataset.
    size: int
        Number of data points to selected by sampler.
    lr: float
        Learning rate of the Hedge algorithm.
    eps: float
        Stopping criterion for CVaR-EM.

    Methods
    -------
    fit_predict: Fit the data using CVaR-EM and outputs its prediction.
    """

    def __init__(self, n_components, n_init, num_actions, size, lr, eps):
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
        self.eps = eps

    def fit_predict(self, data):
        """Implementation of CVaR-EM.

        Parameters
        ----------
        data: np.array
            Array of datapoints to be fitted.

        Returns
        -------
        best: dict
            Contains predictions of datapoints, mean NLL, worst NLL and CVaR of NLL
        loss: np.array
            NLL of the dataset at each round
        cvar_loss: np.array
            CVaR of the NLL of the data at each round
        """
        self.gm.fit(data)
        score = self.gm.score_samples(data)
        min_clip = np.min(score)
        max_clip = np.max(score)
        range_score = max_clip - min_clip

        loss = []
        cvar_loss = []
        prob = np.ones(len(data))
        best = dict()
        best["cvar_loss"] = -1e8
        while np.linalg.norm(np.sort(prob)[-self.size :] - 1.0 / self.size) > self.eps:
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
            if cvar_loss[-1] > best["cvar_loss"]:
                best["loss"] = loss[-1]
                best["worst_loss"] = np.max(-ll)
                best["cvar_loss"] = cvar_loss[-1]
                best["pred"] = self.gm.predict(data)
                best["prob"] = self.hedge.probabilities
        return best, loss
