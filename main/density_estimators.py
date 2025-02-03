import numpy as np
import torch
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


def score_informed_kde(X, score_fn=None, n_iter=2, device="cpu"):
    """
    Adaptive KDE using either:
    1. Provided score function (with auto-diff derivatives)
    2. Finite difference estimates (fallback)

    Args:
        X: Input data (numpy array)
        score_fn: PyTorch model taking (N,1) tensor, returning scores
        n_iter: Number of bandwidth refinement iterations
        device: Compute device ('cpu' or 'cuda')

    Returns:
        Optimal bandwidths (numpy array)
    """
    X_np = X.copy()
    n = len(X)
    h_i = np.full(n, 1.06 * np.std(X) * n ** (-1 / 5))

    # Convert to PyTorch tensor if using score_fn
    if score_fn is not None:
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device).view(-1, 1)
        X_tensor.requires_grad_(True)

    for _ in range(n_iter):
        s = np.zeros(n)
        s_prime = np.zeros(n)

        if score_fn is not None:
            # Batch compute scores and derivatives
            with torch.set_grad_enabled(True):
                scores = score_fn(X_tensor).squeeze()
                grad_outputs = torch.ones_like(scores)
                gradients = torch.autograd.grad(
                    scores,
                    X_tensor,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True,
                )[0].squeeze()

            s = scores.detach().cpu().numpy()
            s_prime = gradients.detach().cpu().numpy()
        else:
            raise NotImplementedError(
                "Finite difference score estimation not implemented"
            )

        # Bandwidth update logic
        f_X = np.array(
            [
                np.mean(
                    np.exp(-0.5 * ((X - x) / h_i) ** 2) / (h_i * np.sqrt(2 * np.pi))
                )
                for x in X
            ]
        )
        f_X = np.clip(f_X, 1e-10, None)

        h_i_new = (1 / (2 * np.sqrt(np.pi) * n)) ** (1 / 5) * (
            f_X / ((f_X * (s_prime + s**2)) ** 2 + 1e-10)
        ) ** (1 / 5)
        h_i = 0.5 * h_i_new + 0.5 * h_i  # Dampened update

    # return a function that computes the KDE
    def kde(x):
        return np.mean(
            [norm.pdf(x, loc=xi, scale=hi) for xi, hi in zip(X_np, h_i)], axis=0
        )

    return kde


# %%
DENSITY_ESTIMATORS = {
    "kde": lambda data: gaussian_kde(data),
    "silverman": silverman_estimator,
    "variable_bandwidth": variable_bandwidth_estimator,
    "score_informed": score_informed_estimator,
    "score_informed_torch": score_informed_kde,
}
