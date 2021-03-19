import numpy as np
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt
from itertools import cycle, islice
from cvar_em import CVaR_EM

np.random.seed(31415)

param = {'n_samples': 1500,
         'n': 100,
         'threshold': 10,
         }

def label(y, n_clusters):
    ind=0
    labels = dict()
    for i in range(len(y)):
        if y[i] in labels.keys():
            y[i] = labels[y[i]]
        else:
            labels[y[i]] = y[i] = ind
            ind += 1
    return y

def predict(algorithm, X):
    best_seed = -1
    best_score = -1e7
    for i in range(10):
        seed = np.random.randint(1, 1e5)
        algorithm.set_params(random_state =seed)
        algorithm.fit(X)
        score = np.sum(algorithm.score(X))
        if best_score < score:
            best_score = score
            best_seed = seed
    algorithm.set_params(random_state = best_seed)
    algorithm.fit(X)
    return algorithm.predict(X)

fig = plt.figure(figsize=[10, 6])
for i in range(param['n']):
    if i%10 == 0:
        print(i)
    n_clusters = np.random.randint(1, 6)

    std = 4.0 * np.random.rand(n_clusters)
    sample_distr = np.random.rand(n_clusters)
    sample_distr *= param['n_samples'] / np.sum(sample_distr)
    sample_distr = (np.round(sample_distr)).astype(int)
    X, y = datasets.make_blobs(n_samples = sample_distr, cluster_std=std)
    X = StandardScaler().fit_transform(X)

    cvar = CVaR_EM(
        max_iter=1000, n_components=n_clusters, covariance_type='full')
    gmm = GaussianMixture(
        n_components=n_clusters, covariance_type='full')
    cvar_y = label(predict(cvar, X), n_clusters)
    gmm_y = label(predict(gmm, X), n_clusters)

    if np.linalg.norm(cvar_y-gmm_y, 0) > param['threshold']:
        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                     int(max(max(cvar_y), max(gmm_y)) + 1))))
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        ax = plt.subplot(1, 3, 1)
        ax.set_title("True distribution")
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y])
        
        ax1 = plt.subplot(1, 3, 2)
        ax1.set_title("GMM")
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[gmm_y])

        ax2 = plt.subplot(1, 3, 3)
        ax2.set_title("CVaR_EM")
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[cvar_y])

        plt.savefig('../output/'+str(i)+'.png')
        plt.clf()
