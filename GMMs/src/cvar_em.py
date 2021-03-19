import numpy as np
from sklearn.mixture import GaussianMixture

np.random.seed(176371)

class CVaR_EM(GaussianMixture):
    def __init__(self, n_components=1, *, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10, k=30, epsilon=1e-2):
        super().__init__(
            n_components=n_components, covariance_type=covariance_type,
            tol=tol, reg_covar=reg_covar, max_iter=max_iter, n_init=n_init,
            init_params=init_params, weights_init=weights_init,
            means_init=means_init, precisions_init=precisions_init,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)

        self.k = k
        self.epsilon = epsilon
    
    def _m_step(self, X, log_resp):
        # weights = pi_c
        # log_resp: n_samples * n_components matrix, log P(c | xi)
        # estimate_log_prob: returns log P(x | theta)
        # n_samples, n_components = log_prob.shape
        old = new = 0
        while True:
            old = new
            log_prob = self._estimate_log_prob(X)
            weights = np.broadcast_to(self.weights_, log_prob.shape)
            log_marginal_prob = log_prob + weights
            ind = np.argsort(np.sum(log_resp @ log_marginal_prob.T, axis=1))
            new = np.sum(log_marginal_prob[ind])
            if np.abs(new - old) < self.epsilon:
                break
            super()._m_step(X[ind], log_resp[ind])

