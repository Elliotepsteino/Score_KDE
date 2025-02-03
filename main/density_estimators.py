import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import norm
from sklearn.neighbors import KernelDensity


def silverman_estimator(data, dim=1):
    """Classic Silverman's rule KDE"""
    n = len(data)
    h = 1.06 * np.std(data) * n ** (-1 / (4 + dim))
    kde = KernelDensity(bandwidth=h, kernel="gaussian")
    kde.fit(data.reshape(-1, 1))
    return lambda x: np.exp(kde.score_samples(x.reshape(-1, 1)))


def variable_bandwidth_estimator(data, n_iter=3, dim=1):
    """Adaptive bandwidth KDE with iterative refinement"""
    n = len(data)
    h = 1.06 * np.std(data) * n ** (-1 / (4 + dim))
    h_i = np.full(n, h)

    for _ in range(n_iter):
        # Compute pilot density estimate
        pilot_kde = silverman_estimator(data, dim)
        f = pilot_kde(data)

        # Update bandwidths
        h_i = h * (f / np.max(f)) ** (-0.5)

    def estimator(x):
        return np.mean(
            [norm.pdf(x, loc=xi, scale=hi) for xi, hi in zip(data, h_i)], axis=0
        )

    return estimator


def score_informed_estimator(data, score_fn=None, dim=1, eps=1e-3):
    """KDE using score function information"""
    n = len(data)
    h = 1.06 * np.std(data) * n ** (-1 / (4 + dim))

    # Default score estimator using finite differences
    if score_fn is None:
        pilot_kde = silverman_estimator(data, dim)
        score_fn = lambda x: (pilot_kde(x + eps) - pilot_kde(x - eps)) / (2 * eps)

    def estimator(x):
        # Implement your score-based bandwidth adjustment here
        adjusted_h = h * np.exp(-0.5 * score_fn(x) ** 2)
        return np.mean([norm.pdf(x, loc=xi, scale=adjusted_h) for xi in data], axis=0)

    return estimator


# %%
DENSITY_ESTIMATORS = {
    "kde": lambda data: gaussian_kde(data),
    "silverman": silverman_estimator,
    "variable_bandwidth": variable_bandwidth_estimator,
    "score_informed": score_informed_estimator,
}
