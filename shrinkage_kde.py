import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm

###############################################################################
# Matplotlib settings for high-quality PDF
###############################################################################
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42

###############################################################################
# 1) Define three different mixtures
###############################################################################
mixture_params_list = [
    {'pi': 0.4, 'mu1': -2, 'sigma1': 0.5, 'mu2':  2, 'sigma2': 1.0},
    {'pi': 0.3, 'mu1': -3, 'sigma1': 0.5, 'mu2':  2, 'sigma2': 0.7},
    {'pi': 0.5, 'mu1': -1, 'sigma1': 1.5, 'mu2':  3, 'sigma2': 0.8},
]

###############################################################################
# 2) Mixture sampling & pdf
###############################################################################
def mixture_pdf(x, params):
    """Mixture PDF: pi * N(mu1, sigma1^2) + (1 - pi) * N(mu2, sigma2^2)."""
    pi_     = params['pi']
    mu1     = params['mu1']
    sigma1  = params['sigma1']
    mu2     = params['mu2']
    sigma2  = params['sigma2']
    return pi_ * norm.pdf(x, mu1, sigma1) + (1 - pi_) * norm.pdf(x, mu2, sigma2)

def sample_from_mixture(n, params):
    """Sample n points from the specified mixture distribution."""
    pi_     = params['pi']
    mu1     = params['mu1']
    sigma1  = params['sigma1']
    mu2     = params['mu2']
    sigma2  = params['sigma2']
    z = np.random.rand(n) < pi_
    x_samps = np.zeros(n)
    x_samps[z]  = np.random.normal(mu1, sigma1, size=z.sum())
    x_samps[~z] = np.random.normal(mu2, sigma2, size=(~z).sum())
    return x_samps

###############################################################################
# 3) Score function & updates
###############################################################################
def score_function(x, params):
    """
    Score function = derivative of log p(x) for a 2-component Gaussian mixture
    = (dp/dx) / p(x).
    """
    p_x = mixture_pdf(x, params)
    pi_     = params['pi']
    mu1     = params['mu1']
    sigma1  = params['sigma1']
    mu2     = params['mu2']
    sigma2  = params['sigma2']
    
    d_comp1 = pi_ * norm.pdf(x, mu1, sigma1) * ((mu1 - x) / (sigma1**2))
    d_comp2 = (1 - pi_) * norm.pdf(x, mu2, sigma2) * ((mu2 - x) / (sigma2**2))
    dp_dx   = d_comp1 + d_comp2
    return dp_dx / (p_x + 1e-15)

def two_step_update(x, params, alpha=0.1, steps=2):
    """2-step fixed: x <- x + alpha*s(x), repeated 'steps' times."""
    x_new = x.copy()
    for _ in range(steps):
        grad_log_p = score_function(x_new, params)
        x_new += alpha * grad_log_p
    return x_new

###############################################################################
# 4) Bandwidth selection & the Score-Debiased method
###############################################################################
def silverman_bandwidth(data):
    """
    Standard Silverman's rule-of-thumb (1D):
      h = 0.9 * min(std, IQR/1.34) * n^{-1/5}.
    """
    n = len(data)
    std_dev = np.std(data)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    sigma = min(std_dev, iqr / 1.34)
    return 0.9 * sigma * n**(-1/5)

def one_step_debiased_data(x, params):
    """
    Score-Debiased KDE:
      1) h = 0.4 * sigma * n^{-1/9}
      2) x <- x + (h^2/2)*s(x).
    """
    n = len(x)
    std_dev = np.std(x)
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    sigma = min(std_dev, iqr / 1.34)

    h = 0.4 * sigma * n**(-1/9)

    delta = (h**2) / 2.0
    x_new = x + delta * score_function(x, params)
    return x_new, h

###############################################################################
# 5) Vectorized KDE evaluation
###############################################################################
def kde_pdf_eval(x_points, data, bandwidth):
    """
    Evaluate the KDE at points x_points (shape (M,)) given data (shape (N,))
    and a known bandwidth (scalar). Vectorized approach: O(M*N).
    """
    M = x_points.size
    N = data.size
    z = (x_points.reshape(M,1) - data.reshape(1,N)) / bandwidth
    pdf_matrix = (1.0 / np.sqrt(2.0*np.pi)) * np.exp(-0.5 * z**2)
    kde_vals = pdf_matrix.mean(axis=1) / bandwidth
    return kde_vals

###############################################################################
# 6) KL divergence (MC) & MISE (grid)
###############################################################################
def kl_divergence(data, bandwidth, params, nsamples=10_000):
    """
    Monte Carlo estimate of KL(p || q). 
    data & bandwidth define q as a KDE, while p is the mixture.
    """
    x_samps = sample_from_mixture(nsamples, params)
    log_p   = np.log(mixture_pdf(x_samps, params) + 1e-15)
    q_vals  = kde_pdf_eval(x_samps, data, bandwidth) + 1e-15
    log_q   = np.log(q_vals)
    return np.mean(log_p - log_q)

def approximate_mise(data, bandwidth, params, x_min=-8.0, x_max=8.0, step=0.05):
    """
    Approx MISE = ∫ (p(x) - q(x))^2 dx by Riemann sum on [x_min, x_max].
    p is mixture, q is KDE with data & bandwidth.
    """
    x_grid = np.arange(x_min, x_max+step, step)
    p_vals = mixture_pdf(x_grid, params)
    q_vals = kde_pdf_eval(x_grid, data, bandwidth)
    diff_sq = (p_vals - q_vals)**2
    return np.sum(diff_sq) * step

###############################################################################
# 7) Example with n=200: Show difference histograms & one “demo” seed
###############################################################################
N_SEEDS   = 100
N_DATA    = 200
ALPHA     = 0.1  
STEPS     = 2

kl_silverman   = []
kl_2step       = []
kl_onestep     = []

for i, params in enumerate(mixture_params_list):
    kl_silver_i = []
    kl_2step_i  = []
    kl_1step_i  = []

    for seed in range(N_SEEDS):
        np.random.seed(seed)
        x_data = sample_from_mixture(N_DATA, params)

        # Silverman KDE
        h_silv = silverman_bandwidth(x_data)
        kl_silv = kl_divergence(x_data, h_silv, params)

        # 2-step fixed
        x_2s = two_step_update(x_data, params, alpha=ALPHA, steps=STEPS)
        h_2s = silverman_bandwidth(x_2s)
        kl_2s_val = kl_divergence(x_2s, h_2s, params)

        # Score-Debiased KDE
        x_1st, h_1st = one_step_debiased_data(x_data, params)
        kl_1st_val   = kl_divergence(x_1st, h_1st, params)

        kl_silver_i.append(kl_silv)
        kl_2step_i.append(kl_2s_val)
        kl_1step_i.append(kl_1st_val)

    kl_silverman.append(np.array(kl_silver_i))
    kl_2step.append(np.array(kl_2step_i))
    kl_onestep.append(np.array(kl_1step_i))

# --- Demo seed plot ---
demo_seeds = [0, 0, 0]
x_plot_grid = np.linspace(-6, 6, 400)

fig, axes = plt.subplots(1, 3, figsize=(15,5))
for i, params in enumerate(mixture_params_list):
    seed_demo = demo_seeds[i]
    np.random.seed(seed_demo)
    x_data_demo = sample_from_mixture(N_DATA, params)

    # Silverman
    h_silv_demo = silverman_bandwidth(x_data_demo)
    q_silv_demo = kde_pdf_eval(x_plot_grid, x_data_demo, h_silv_demo)
    kl_silv_demo= kl_divergence(x_data_demo, h_silv_demo, params)

    # 2-step fixed
    x_2s_demo = two_step_update(x_data_demo, params)
    h_2s_demo = silverman_bandwidth(x_2s_demo)
    q_2s_demo = kde_pdf_eval(x_plot_grid, x_2s_demo, h_2s_demo)
    kl_2s_demo= kl_divergence(x_2s_demo, h_2s_demo, params)

    # Score-Debiased
    x_1st_demo, h_1st_demo = one_step_debiased_data(x_data_demo, params)
    q_1st_demo = kde_pdf_eval(x_plot_grid, x_1st_demo, h_1st_demo)
    kl_1st_demo= kl_divergence(x_1st_demo, h_1st_demo, params)

    # True PDF
    p_vals = mixture_pdf(x_plot_grid, params)

    ax = axes[i]
    ax.plot(x_plot_grid, p_vals, 'k--', label="True PDF")
    ax.plot(x_plot_grid, q_silv_demo, 'b', 
            label=f"Silverman KDE (KL={kl_silv_demo:.3f})")
    ax.plot(x_plot_grid, q_2s_demo, 'g',
            label=f"2-step fixed (KL={kl_2s_demo:.3f})")
    ax.plot(x_plot_grid, q_1st_demo, 'r',
            label=f"Score-Debiased KDE (KL={kl_1st_demo:.3f})")

    ax.set_title(f"Mixture {i+1}, seed={seed_demo}")
    ax.set_xlabel("x")
    if i == 0:
        ax.set_ylabel("Density")
    if i == 2:
        ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig("figures/example_comparison.pdf", dpi=300, bbox_inches="tight")
plt.show()

# --- Histograms comparing differences in KL at n=200 ---
fig, axes = plt.subplots(1, 3, figsize=(15,5))
for i in range(len(mixture_params_list)):
    diffs = kl_silverman[i] - kl_2step[i]
    mean_diff = np.mean(diffs)
    std_diff  = np.std(diffs)

    ax = axes[i]
    ax.hist(diffs, bins=15, alpha=0.7, color='C0', edgecolor='k')
    ax.axvline(mean_diff, color='red', linestyle='--', label=f"mean={mean_diff:.3f}")
    ax.set_title(f"Mixture {i+1}\nSilverman KDE - 2-step fixed\nstd={std_diff:.3f}")
    ax.set_xlabel("KL(Silverman) - KL(2-step)")
    if i == 0:
        ax.set_ylabel("Count")
    ax.legend()

plt.tight_layout()
plt.savefig("figures/silverman_vs_2step.pdf", dpi=300, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15,5))
for i in range(len(mixture_params_list)):
    diffs = kl_onestep[i] - kl_2step[i]
    mean_diff = np.mean(diffs)
    std_diff  = np.std(diffs)

    ax = axes[i]
    ax.hist(diffs, bins=15, alpha=0.7, color='C2', edgecolor='k')
    ax.axvline(mean_diff, color='red', linestyle='--', label=f"mean={mean_diff:.3f}")
    ax.set_title(f"Mixture {i+1}\nScore-Debiased - 2-step fixed\nstd={std_diff:.3f}")
    ax.set_xlabel("KL(Score-Debiased) - KL(2-step)")
    if i == 0:
        ax.set_ylabel("Count")
    ax.legend()

plt.tight_layout()
plt.savefig("figures/onestep_vs_2step.pdf", dpi=300, bbox_inches="tight")
plt.show()

###############################################################################
# 9) Scaling experiment for KL & MISE: n up to 10000
###############################################################################
n_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 4000, 10000]
N_SEEDS_SCALING = 50  # reduce if too slow at large n

avg_kl_silver = np.zeros((len(mixture_params_list), len(n_list)))
avg_kl_2step  = np.zeros((len(mixture_params_list), len(n_list)))
avg_kl_1step  = np.zeros((len(mixture_params_list), len(n_list)))

avg_mise_silver = np.zeros((len(mixture_params_list), len(n_list)))
avg_mise_2step  = np.zeros((len(mixture_params_list), len(n_list)))
avg_mise_1step  = np.zeros((len(mixture_params_list), len(n_list)))

for i, params in enumerate(mixture_params_list):
    for j, n_data in enumerate(n_list):
        kl_vals_silv = []
        kl_vals_2st  = []
        kl_vals_1st  = []
        mise_silv    = []
        mise_2st     = []
        mise_1st     = []

        for seed in range(N_SEEDS_SCALING):
            np.random.seed(seed)
            x_data = sample_from_mixture(n_data, params)

            # Silverman KDE
            h_silv = silverman_bandwidth(x_data)
            kl_s   = kl_divergence(x_data, h_silv, params, nsamples=5000)
            mise_s = approximate_mise(x_data, h_silv, params, step=0.05)

            # 2-step fixed
            x_2s   = two_step_update(x_data, params)
            h_2s   = silverman_bandwidth(x_2s)
            kl_2   = kl_divergence(x_2s, h_2s, params, nsamples=5000)
            mise_2 = approximate_mise(x_2s, h_2s, params, step=0.05)

            # Score-Debiased
            x_1s, h_1s = one_step_debiased_data(x_data, params)
            kl_1   = kl_divergence(x_1s, h_1s, params, nsamples=5000)
            mise_1 = approximate_mise(x_1s, h_1s, params, step=0.05)

            kl_vals_silv.append(kl_s)
            kl_vals_2st.append(kl_2)
            kl_vals_1st.append(kl_1)

            mise_silv.append(mise_s)
            mise_2st.append(mise_2)
            mise_1st.append(mise_1)

        avg_kl_silver[i, j]   = np.mean(kl_vals_silv)
        avg_kl_2step[i, j]    = np.mean(kl_vals_2st)
        avg_kl_1step[i, j]    = np.mean(kl_vals_1st)

        avg_mise_silver[i, j] = np.mean(mise_silv)
        avg_mise_2step[i, j]  = np.mean(mise_2st)
        avg_mise_1step[i, j]  = np.mean(mise_1st)

# --- Plot KL scaling on log–log with single legend at the bottom ---
fig, axes = plt.subplots(1, 3, figsize=(15,5), sharey=False)

method_labels = {
    'Silverman': "Silverman KDE",
    'TwoStep':   "2-step fixed",
    'OneStep':   "Score-Debiased KDE",
}

# We'll gather lines for a single legend
all_lines = []
all_labels= []

for i in range(len(mixture_params_list)):
    ax = axes[i]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (log scale)")
    if i == 0:
        ax.set_ylabel("Mean KL (log scale)")
    ax.set_title(f"Mixture {i+1}")

    # Plot each method
    line_silv, = ax.plot(n_list, avg_kl_silver[i], 'o-b', alpha=0.8, 
                         label=method_labels['Silverman'])
    line_2step,= ax.plot(n_list, avg_kl_2step[i],  's-g', alpha=0.8, 
                         label=method_labels['TwoStep'])
    line_1step,= ax.plot(n_list, avg_kl_1step[i],  'd-r', alpha=0.8, 
                         label=method_labels['OneStep'])

    if i == 0:
        all_lines.extend([line_silv, line_2step, line_1step])
        all_labels.extend([
            method_labels['Silverman'], 
            method_labels['TwoStep'], 
            method_labels['OneStep']
        ])

    # Regression slopes
    log_n = np.log(n_list)
    slope_silv, _ = np.polyfit(log_n, np.log(avg_kl_silver[i] + 1e-15), 1)
    slope_2step,_ = np.polyfit(log_n, np.log(avg_kl_2step[i]   + 1e-15), 1)
    slope_1step,_ = np.polyfit(log_n, np.log(avg_kl_1step[i]   + 1e-15), 1)

    slope_text = (f"Silver slope: {slope_silv:.2f}\n"
                  f"2-step slope: {slope_2step:.2f}\n"
                  f"Debiased slope: {slope_1step:.2f}")
    ax.text(0.05, 0.95, slope_text,
            transform=ax.transAxes, ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # Reference line n^{-4/9}
    ref_slope = -4.0/9.0
    if 100 in n_list:
        idx_100 = n_list.index(100)
        anchor_val = avg_kl_1step[i, idx_100] + 1e-15
        c = anchor_val / (100.0**(ref_slope))
        n_arr = np.array(n_list)
        ref_vals = c * (n_arr**(ref_slope))
        ax.plot(n_arr, ref_vals, 'k--', label=r"Ref $n^{-4/9}$")

# Create a single legend below the subplots
fig.legend(all_lines, all_labels, loc='lower center', ncol=3, fontsize=10)
plt.tight_layout(rect=[0,0.05,1,1])  # leave space at bottom
plt.savefig("figures/scaling_experiment_kl.pdf", dpi=300)
plt.show()

# --- Plot MISE scaling similarly ---
fig, axes = plt.subplots(1, 3, figsize=(15,5), sharey=False)

all_lines = []
all_labels= []

for i in range(len(mixture_params_list)):
    ax = axes[i]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (log scale)")
    if i == 0:
        ax.set_ylabel("Mean MISE (log scale)")
    ax.set_title(f"Mixture {i+1}")

    # Plot each method
    line_silv, = ax.plot(n_list, avg_mise_silver[i], 'o-b', alpha=0.8, 
                         label=method_labels['Silverman'])
    line_2step,= ax.plot(n_list, avg_mise_2step[i],  's-g', alpha=0.8, 
                         label=method_labels['TwoStep'])
    line_1step,= ax.plot(n_list, avg_mise_1step[i],  'd-r', alpha=0.8, 
                         label=method_labels['OneStep'])

    if i == 0:
        all_lines.extend([line_silv, line_2step, line_1step])
        all_labels.extend([
            method_labels['Silverman'], 
            method_labels['TwoStep'], 
            method_labels['OneStep']
        ])

    # Regression slopes
    log_n = np.log(n_list)
    slope_silv, _ = np.polyfit(log_n, np.log(avg_mise_silver[i] + 1e-15), 1)
    slope_2step,_ = np.polyfit(log_n, np.log(avg_mise_2step[i]   + 1e-15), 1)
    slope_1step,_ = np.polyfit(log_n, np.log(avg_mise_1step[i]   + 1e-15), 1)

    slope_text = (f"Silver slope: {slope_silv:.2f}\n"
                  f"2-step slope: {slope_2step:.2f}\n"
                  f"Debiased slope: {slope_1step:.2f}")
    ax.text(0.05, 0.95, slope_text,
            transform=ax.transAxes, ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # Reference line n^{-4/9}
    ref_slope = -4.0/9.0
    if 100 in n_list:
        idx_100 = n_list.index(100)
        anchor_val = avg_mise_1step[i, idx_100] + 1e-15
        c = anchor_val / (100.0**(ref_slope))
        n_arr = np.array(n_list)
        ref_vals = c * (n_arr**(ref_slope))
        ax.plot(n_arr, ref_vals, 'k--', label=r"Ref $n^{-4/9}$")

fig.legend(all_lines, all_labels, loc='lower center', ncol=3, fontsize=10)
plt.tight_layout(rect=[0,0.05,1,1])  # leaves space at bottom
plt.savefig("figures/scaling_experiment_mise.pdf", dpi=300)
plt.show()









