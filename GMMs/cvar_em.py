import numpy as np
import os

from sklearn.mixture import GaussianMixture

from adacvar.util.adaptive_algorithm import Exp3


class CVaR_EM:
    """Implementation of the CVaR_EM algorithm.

    Parameters
    ----------
    n_components: int
        Number of clusters of the mixture.
    n_init: int
        Number of initializations for the Gaussian Mixture algorithm.
    num_actions: int
        Number of datapoints in dataset.
    size: int
        Number of datapoints to select at each iteration.
    lr: float
        Learning rate of the Hedge algorithm.

    Methods
    -------
    sample: Adaptive data sampler
    fit_predict: Fit the data using the Hedge algorithm and predicts output.
    """

    def __init__(self, n_components, n_init, num_actions, size, lr=0.01):
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
        self.lr = lr

    def fit_predict(self, X):
        idx = np.zeros(self.size)
        last_idx = np.ones(self.size)
        weights = np.ones(self.num_actions)
        loss = []
        while not np.array_equal(idx, last_idx):
            last_idx = idx
            probabilities = weights / np.sum(weights)
            idx = np.random.choice(
                np.arange(self.num_actions),
                self.size,
                False,  # with or without replacement
                probabilities,
            )
            idx = np.sort(idx)
            curX = X[idx]

            self.gm.fit(curX)
            # weighted log probabilities for each sample in X
            cost = -self.gm.score_samples(X)
            loss += [np.sum(cost)]
            weights *= (1 - self.lr) ** cost
        return self.gm.predict(X), loss, weights
