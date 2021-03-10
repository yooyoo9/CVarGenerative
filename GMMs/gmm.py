import warnings
import random

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

np.random.seed(176371)

class CVaR_EM(GaussianMixture):
    def __init__(self, n_components=1, *, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10, k=30):
        super().__init__(
            n_components=n_components, covariance_type=covariance_type,
            tol=tol, reg_covar=reg_covar, max_iter=max_iter, n_init=n_init,
            init_params=init_params, weights_init=weights_init,
            means_init=means_init, precisions_init=precisions_init,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)

        self.k = k
    
    def _m_step(self, X, log_resp):
        n_samples, _ = X.shape
        ranking = np.exp(self._estimate_log_prob(X)) @ self.weights_
        ind = np.argsort(ranking)[:self.k]
        X = X[ind]
        log_resp = log_resp[ind]
        super()._m_step(X, log_resp)


n_samples = 1500
noisy_circles = datasets.make_circles(
    n_samples=n_samples, factor=.5, noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs3 = datasets.make_blobs(n_samples=n_samples, centers=3)
blobs4 = datasets.make_blobs(n_samples=n_samples, centers=4)
blobs5 = datasets.make_blobs(n_samples=n_samples, centers=5)
no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
def aniso(k):
    X, y = datasets.make_blobs(n_samples=n_samples, centers=k)
    transformation = 2.0 * (np.random.rand(2, 2) - 0.5)
    X_aniso = np.dot(X, transformation)
    return (X_aniso, y)
aniso3 = aniso(3)
aniso4 = aniso(4)
aniso5 = aniso(5)

# blobs with varied variances
def varied(k):
    std = 4 * np.random.rand(k)
    varied = datasets.make_blobs(n_samples=n_samples, centers=k,
                             cluster_std=std)
    return varied
varied3 = varied(3)
varied4 = varied(4)
varied5 = varied(5)

plt.figure()
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1

default_base = {'quantile': .3,
                'eps': .3,
                'damping': .9,
                'preference': -200,
                'n_neighbors': 10,
                'n_clusters': 4,
                'min_samples': 20,
                'xi': 0.05,
                'min_cluster_size': 0.1}

datasets = [
    # (noisy_circles, {'damping': .77, 'preference': -240,
    # 'quantile': .2, 'n_clusters': 2,
    # 'min_samples': 20, 'xi': 0.25}),
    # (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
    (varied3, {'eps': .18, 'n_neighbors': 2, 'n_clusters': 3,
              'min_samples': 5, 'xi': 0.035, 'min_cluster_size': .1}),
    (varied4, {'eps': .18, 'n_neighbors': 2, 'n_clusters': 4,
              'min_samples': 5, 'xi': 0.035, 'min_cluster_size': .1}),
    (varied5, {'eps': .18, 'n_neighbors': 2, 'n_clusters': 5,
              'min_samples': 5, 'xi': 0.035, 'min_cluster_size': .1}),
    (aniso3, {'eps': .15, 'n_neighbors': 2, 'n_clusters': 3,
             'min_samples': 20, 'xi': 0.1, 'min_cluster_size': .2}),
    (aniso4, {'eps': .15, 'n_neighbors': 2, 'n_clusters': 4,
             'min_samples': 20, 'xi': 0.1, 'min_cluster_size': .2}),
    (aniso5, {'eps': .15, 'n_neighbors': 2, 'n_clusters': 5,
             'min_samples': 20, 'xi': 0.1, 'min_cluster_size': .2}),
    (blobs3, {'n_clusters': 3}),
    (blobs4, {'n_clusters': 4}),
    (blobs5, {'n_clusters': 5}),
    #(no_structure, {})]
    ]

for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    X, y = dataset
    X = StandardScaler().fit_transform(X)

    CVaR_EM = CVaR_EM(
        max_iter=1000, n_components=params['n_clusters'], covariance_type='full')
    gmm = GaussianMixture(
        n_components=params['n_clusters'], covariance_type='full')

    clustering_algorithms = (
        ('GMM', gmm),
        ('CVaR_EM', CVaR_EM),
    )

    for name, algorithm in clustering_algorithms:
        # catch warnings related to kneighbors_graph
        best_seed = -1
        best_score = -1e7
        with warnings.catch_warnings():
            # warnings.filterwarnings("ignore", category=UserWarning)
            for i in range(10):
                seed = random.randint(1, 1e5)
                algorithm.set_params(random_state =seed)
                algorithm.fit(X)
                score = np.sum(algorithm.score(X))
                if best_score < score:
                    best_score = score
                    best_seed = seed
            algorithm.set_params(random_state = best_seed)
            algorithm.fit(X)

        y_pred = algorithm.predict(X)

        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                     int(max(y_pred) + 1))))
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plot_num += 1

plt.show()
