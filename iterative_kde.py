import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.stats import norm, wasserstein_distance, entropy
from tqdm import tqdm


# =====================
# 1. Define True Densities
# =====================
def true_density_case1(x):
    return 0.5 * norm.pdf(x, loc=-3, scale=1) + 0.5 * norm.pdf(x, loc=3, scale=2)


def true_density_case2(x):
    return norm.pdf(x, loc=0, scale=2)


def true_density_case3(x):
    return 0.7 * norm.pdf(x, loc=-2, scale=1) + 0.3 * norm.pdf(x, loc=3, scale=1.5)


def true_density_case4(x):
    return 0.5 * norm.pdf(x, loc=-1, scale=0.8) + 0.5 * norm.pdf(x, loc=1.5, scale=0.8)


cases = {
    "Case 1: Bimodal": true_density_case1,
    "Case 2: Gaussian": true_density_case2,
    "Case 3: Asymmetric": true_density_case3,
    "Case 4: Close Bimodal": true_density_case4,
}


# =====================
# 2. Core Functions
# =====================
def generate_data(true_density, n=500):
    if true_density == true_density_case1:
        X1 = np.random.normal(-3, 1, n // 2)
        X2 = np.random.normal(3, 2, n // 2)
        return np.concatenate([X1, X2])
    elif true_density == true_density_case2:
        return np.random.normal(0, 2, n)
    elif true_density == true_density_case3:
        X1 = np.random.normal(-2, 1, int(0.7 * n))
        X2 = np.random.normal(3, 1.5, n - int(0.7 * n))
        return np.concatenate([X1, X2])
    elif true_density == true_density_case4:
        X1 = np.random.normal(-1, 0.8, n // 2)
        X2 = np.random.normal(1.5, 0.8, n // 2)
        return np.concatenate([X1, X2])


def compute_metrics(true_density, kde, X_kde, X_grid, h_i=None, n_samples=5000):
    true_probs = true_density(X_grid[:, 0])
    kde_probs = np.exp(kde.score_samples(X_grid))

    epsilon = 1e-10
    kl = entropy(true_probs + epsilon, kde_probs + epsilon)

    if h_i is None:
        samples_kde = kde.sample(n_samples).flatten()
    else:
        idx = np.random.choice(len(X_kde), size=n_samples)
        samples_kde = np.array(
            [np.random.normal(X_kde[i], h_i[i]) for i in idx]
        ).flatten()

    samples_true = generate_data(true_density, n_samples)
    wd = wasserstein_distance(samples_true, samples_kde)

    return kl, wd


# =====================
# 3. KDE Implementations
# =====================
def silverman_kde(X):
    h = 1.06 * np.std(X) * len(X) ** (-1 / 5)
    kde = KernelDensity(bandwidth=h, kernel="gaussian")
    kde.fit(X.reshape(-1, 1))
    return kde, h


def score_informed_kde(X, n_iter=2):
    X_2d = X.reshape(-1, 1)
    n = len(X)
    h_i = np.full(n, 1.06 * np.std(X) * n ** (-1 / 5))

    for _ in range(n_iter):
        s = np.zeros(n)
        s_prime = np.zeros(n)
        eps = 1e-3

        for i in range(n):
            x = X[i]
            u = (X - x) / h_i[i]
            k = np.exp(-0.5 * u**2) / (h_i[i] * np.sqrt(2 * np.pi))
            f = np.mean(k)
            df = np.mean(-(X - x) * k / (h_i[i] ** 2))
            s[i] = df / f if f > 1e-10 else 0

            x_plus = x + eps
            u_plus = (X - x_plus) / h_i[i]
            k_plus = np.exp(-0.5 * u_plus**2) / (h_i[i] * np.sqrt(2 * np.pi))
            f_plus = np.mean(k_plus)
            df_plus = np.mean(-(X - x_plus) * k_plus / (h_i[i] ** 2))
            s_plus = df_plus / f_plus if f_plus > 1e-10 else 0

            x_minus = x - eps
            u_minus = (X - x_minus) / h_i[i]
            k_minus = np.exp(-0.5 * u_minus**2) / (h_i[i] * np.sqrt(2 * np.pi))
            f_minus = np.mean(k_minus)
            df_minus = np.mean(-(X - x_minus) * k_minus / (h_i[i] ** 2))
            s_minus = df_minus / f_minus if f_minus > 1e-10 else 0

            s_prime[i] = (s_plus - s_minus) / (2 * eps)

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
        h_i = 0.5 * h_i_new + 0.5 * h_i

    return h_i


class VariableKDE:
    def __init__(self, X, h_i):
        self.X = X.reshape(-1, 1)
        self.h_i = h_i

    def score_samples(self, X_eval):
        X_eval = X_eval.reshape(-1, 1)
        scores = []
        for x in X_eval[:, 0]:
            ks = np.exp(-0.5 * ((self.X[:, 0] - x) / self.h_i) ** 2) / (
                self.h_i * np.sqrt(2 * np.pi)
            )
            scores.append(np.log(np.mean(ks)))
        return np.array(scores)


# =====================
# 4. Simulation and Plotting
# =====================
np.random.seed(42)
n = 500
X_grid = np.linspace(-8, 8, 1000).reshape(-1, 1)

plt.figure(figsize=(15, 10))
metrics = []

for idx, (name, density) in enumerate(cases.items(), 1):
    X = generate_data(density, n)

    # Compute KDEs
    kde_silver, h_silver = silverman_kde(X)
    h_i = score_informed_kde(X)
    kde_score = VariableKDE(X, h_i)

    # Calculate metrics
    kl_silver, wd_silver = compute_metrics(density, kde_silver, X, X_grid)
    kl_score, wd_score = compute_metrics(density, kde_score, X, X_grid, h_i=h_i)

    metrics.append(
        {
            "Case": name,
            "KL Silverman": kl_silver,
            "KL Score": kl_score,
            "WD Silverman": wd_silver,
            "WD Score": wd_score,
        }
    )

    # Plot
    plt.subplot(2, 2, idx)
    plt.hist(X, bins=30, density=True, alpha=0.2, color="grey")
    plt.plot(X_grid[:, 0], density(X_grid[:, 0]), "k--", label="Truth")
    plt.plot(
        X_grid[:, 0],
        np.exp(kde_silver.score_samples(X_grid)),
        "r-",
        label=f"Silverman (KL={kl_silver:.2f})",
    )
    plt.plot(
        X_grid[:, 0],
        np.exp(kde_score.score_samples(X_grid)),
        "b-",
        label=f"Score (KL={kl_score:.2f})",
    )
    plt.title(name)
    plt.legend()

plt.tight_layout()
plt.show()

# Print metrics table
print("\nMetric Comparison:")
print(
    "{:<20} | {:<10} | {:<10} | {:<10} | {:<10}".format(
        "Case", "KL Silver", "KL Score", "WD Silver", "WD Score"
    )
)
for m in metrics:
    print(
        "{:<20} | {:<10.2f} | {:<10.2f} | {:<10.2f} | {:<10.2f}".format(
            m["Case"],
            m["KL Silverman"],
            m["KL Score"],
            m["WD Silverman"],
            m["WD Score"],
        )
    )
