import numpy as np
import wandb
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture

from adacvar.util.adaptive_algorithm import Hedge


colors = np.array(
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

    def __init__(self, n_components, n_init, num_actions, size, val_size, test_size, lr, threshold):
        self.num_actions = num_actions
        self.size = size
        self.val_size = val_size
        self.test_size = test_size
        self.gm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            tol=1e-3,
            max_iter=100,
            n_init=n_init,
            init_params="kmeans",
        )
        self.hedge = Hedge(num_actions=num_actions, size=size, eta=lr)
        self.threshold = threshold

    def fit_predict(self, data, val_data, test_data):
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

        prob = np.ones(len(data))
        best_loss = 1e8
        nb_iters = 0
        while nb_iters < self.threshold:
            # np.linalg.norm(np.sort(prob)[-self.size:] - 1.0 / self.size) > self.eps:
            nb_iters += 1
            prob = self.hedge.probabilities
            prob /= np.sum(prob)
            idx = np.random.choice(
                np.arange(self.num_actions), self.size, p=prob, replace=False
            )
            idx = np.sort(idx)
            self.gm.fit(data[idx])
            # weighted log probabilities for each sample in X
            nll = -self.gm.score_samples(data)
            reward = ((-nll - min_clip) / range_score).clip(0, 1)

            val_nll = -self.gm.score_samples(val_data)
            val_cvar = np.mean(np.sort(val_nll)[-self.val_size:])
            wandb.log({
                'CVaR-EM cvar loss': val_cvar,
                'CVaR-EM worst loss': np.max(val_nll),
                'CVaR-EM mean loss': np.mean(val_nll)
            })

            if val_cvar < best_loss:
                test_nll = -self.gm.score_samples(test_data)
                test_cvar = np.mean(np.sort(test_nll)[-self.test_size:])
                best_loss = test_cvar
                best_worst_loss = np.max(test_nll)
                best_mean_loss = np.mean(test_nll)
                prediction = self.gm.predict(test_data)

                fig = plt.figure()
                plt.axis("equal")
                plt.scatter(test_data[:, 0], test_data[:, 1], s=1, color=colors[prediction])
                wandb.log({'prediction': wandb.Image(fig)})
                plt.close()
            self.hedge.update(reward)
            self.hedge.normalize()
        wandb.log({
            'CVaR-EM final cvar loss': best_loss,
            'CVaR-EM final worst loss': best_worst_loss,
            'CVaR-EM final mean loss': best_mean_loss
        })
        return prediction
