import numpy as np
from scipy.stats import norm, wasserstein_distance, entropy


def generate_data(n_samples=1000, n_dim=1, means=None, covs=None, **kwargs):
    assert means is not None
    assert covs is not None
    num_gaussians = len(means)
    assert len(covs) == num_gaussians

    data = np.zeros((n_samples, n_dim))
    for i in range(n_samples):
        gaussian = np.random.randint(num_gaussians)
        data[i] = np.random.multivariate_normal(means[gaussian], covs[gaussian])

    return data


def true_density_mix(means=None, covs=None):
    # mixture of gaussians
    assert means is not None
    assert covs is not None
    num_gaussians = len(means)
    assert len(covs) == num_gaussians

    def density(x):
        # x has shape (n_eval, n_dim)
        assert len(covs) == num_gaussians

        d = np.zeros((x.shape[0], 1))
        for i in range(num_gaussians):
            d += norm.pdf(x, loc=means[i], scale=np.sqrt(covs[i])).reshape(-1, 1)

        return d

    return density


def evaluate_estimator(
    estimator,
    true_density,
    n_eval=1000,
    **kwargs,
):
    X_grid = np.linspace(-5, 5, n_eval)
    true_probs = true_density(X_grid).reshape(-1)
    kde_probs = estimator(X_grid)

    epsilon = 1e-10
    kl = entropy(true_probs + epsilon, kde_probs + epsilon)

    print(f"KL Divergence: {kl}")

    return kl
