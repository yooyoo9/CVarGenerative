import numpy as np

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

    Methods
    -------
    sample: Adaptive data sampler
    fit_predict: Fit the data using the Hedge algorithm and predicts output.
    """

    def __init__(self, n_components, n_init, num_actions, size, lr=0.1):
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
        self.hedge = Hedge(num_actions=num_actions, size=size, eta=lr)

    def fit_predict(self, data):
        self.gm.fit(data)
        score = self.gm.score_samples(data)
        min_clip = np.min(score)
        max_clip = np.max(score)
        range_score = max_clip - min_clip

        loss = []
        cvar_loss = []
        prob = np.ones(len(data))
        best_idx, best_score = np.zeros(self.size), -1e8
        while np.linalg.norm(np.sort(prob)[-self.size :] - 1.0 / self.size) > 0.04:
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
            if cvar_loss[-1] > best_score:
                best_score = cvar_loss[-1]
                best_idx = idx
            self.hedge.update(reward)
            self.hedge.normalize()
        self.gm.fit(data[best_idx])
        return self.gm.predict(data), loss, self.hedge.probabilities
