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
    epochs: int
        Number of epochs to train.
    num_actions: int
        Number of datapoints in dataset.
    size: int
        Number of datapoints to select at each iteration.
    eta: callable or float
        Learning rate scheduler
    gamma: float, optional
        Mixing of uniform distribution.

    Methods
    -------
    sample: Adaptive data sampler
    fit_predict: Fit the data using the CVaR_EM algorithm and predicts otuput.
    """

    def __init__(self, n_components, epochs, n_init, num_actions, size, eta, gamma=0.0):
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
        self.epochs = epochs
        self._exp3 = Exp3(num_actions, size, eta, gamma)

    def sample(self):
        weights = self._exp3.probabilities
        idx = np.random.choice(
            self.num_actions,
            self.size,
            p=weights / np.sum(weights),
            replace=False,
        )
        return idx

    def fit_predict(self, X):
        idx = np.zeros(self.size)
        for i in range(self.epochs):
            idx = np.sort(self.sample())
            curX = X[idx]
            self.gm.fit(curX)
            # negative log likelihood of the Gaussian mixture given X
            loss = np.clip((-self.gm.score_samples(curX) / 20 + 1) / 2, 0, 1)
            prob = self._exp3.probabilities
            self._exp3.update(loss, idx, prob)
            self._exp3.normalize()
        return self.gm.predict(X)
