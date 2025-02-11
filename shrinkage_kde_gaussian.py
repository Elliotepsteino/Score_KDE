import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm

###############################################################################
# Matplotlib settings for publication-quality plots
###############################################################################
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13

###############################################################################
# 1) Define three different mixtures (Updated)
###############################################################################
#  - Mixture1: same as original
#  - Mixture2: pi=0.3, mu1=-2, sigma1=0.4, mu2=4, sigma2=1.5
#  - Mixture3: pi=0.5, mu1=0, sigma1=0.4, mu2=0, sigma2=1.5
mixture_params_list = [
    # Mixture 1
    {'pi': 0.4, 'mu1': -2, 'sigma1': 0.5, 'mu2':  2, 'sigma2': 1.0},
    # Mixture 2
    {'pi': 0.3, 'mu1': -2, 'sigma1': 0.4, 'mu2':  4, 'sigma2': 1.5},
    # Mixture 3
    {'pi': 0.5, 'mu1':  0, 'sigma1': 0.4, 'mu2':  0, 'sigma2': 1.5},
]

###############################################################################
# 2) Mixture sampling & pdf
###############################################################################
def mixture_pdf(x, params):
    """Mixture PDF: pi * N(mu1, sigma1^2) + (1 - pi)*N(mu2, sigma2^2)."""
    pi_     = params['pi']
    mu1     = params['mu1']
    sigma1  = params['sigma1']
    mu2     = params['mu2']
    sigma2  = params['sigma2']
    return pi_ * norm.pdf(x, mu1, sigma1) + (1 - pi_) * norm.pdf(x, mu2, sigma2)

def sample_from_mixture(n, params):
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
# 3) Score function (with optional noise)
###############################################################################
def score_function(x, params, noise_std=0.0):
    p_x = mixture_pdf(x, params)
    pi_     = params['pi']
    mu1     = params['mu1']
    sigma1  = params['sigma1']
    mu2     = params['mu2']
    sigma2  = params['sigma2']

    d_comp1 = pi_ * norm.pdf(x, mu1, sigma1) * ((mu1 - x) / (sigma1**2))
    d_comp2 = (1 - pi_) * norm.pdf(x, mu2, sigma2) * ((mu2 - x) / (sigma2**2))
    dp_dx   = d_comp1 + d_comp2
    true_score = dp_dx / (p_x + 1e-15)

    if noise_std > 0:
        noise = np.random.normal(0, noise_std, size=x.shape)
        return true_score + noise
    else:
        return true_score

###############################################################################
# 4) Silverman bandwidth & Score-Debiased
###############################################################################
def silverman_bandwidth(data):
    n = len(data)
    std_dev = np.std(data)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    sigma = min(std_dev, iqr / 1.34)
    return 0.9 * sigma * n**(-1/5)

def one_step_debiased_data(x, params, noise_std=0.0):
    n = len(x)
    std_dev = np.std(x)
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    sigma = min(std_dev, iqr / 1.34)

    h = 0.4 * sigma * n**(-1/9)
    delta = (h**2) / 2.0

    s_x = score_function(x, params, noise_std=noise_std)
    x_new = x + delta * s_x
    return x_new, h

###############################################################################
# 5) Vectorized KDE
###############################################################################
def kde_pdf_eval(x_points, data, bandwidth):
    M = x_points.size
    N = data.size
    z = (x_points.reshape(M,1) - data.reshape(1,N)) / bandwidth
    pdf_matrix = (1.0 / np.sqrt(2.0*np.pi)) * np.exp(-0.5 * z**2)
    return pdf_matrix.mean(axis=1) / bandwidth

###############################################################################
# 6) KL divergence (MC) & MISE (grid)
###############################################################################
def kl_divergence(data, bandwidth, params, nsamples=10_000):
    x_samps = sample_from_mixture(nsamples, params)
    log_p   = np.log(mixture_pdf(x_samps, params) + 1e-15)
    q_vals  = kde_pdf_eval(x_samps, data, bandwidth) + 1e-15
    log_q   = np.log(q_vals)
    return np.mean(log_p - log_q)

def approximate_mise(data, bandwidth, params, x_min=-8.0, x_max=8.0, step=0.05):
    x_grid = np.arange(x_min, x_max+step, step)
    p_vals = mixture_pdf(x_grid, params)
    q_vals = kde_pdf_eval(x_grid, data, bandwidth)
    return np.sum((p_vals - q_vals)**2) * step

###############################################################################
# 7) EXAMPLE PLOT at n=200, seed=0 with noise=0,2,4,8 (plus Silverman)
###############################################################################
noise_levels = [0.0, 2.0, 4.0, 8.0]
n_example = 200
seed_example = 0

fig, axes = plt.subplots(1, 3, figsize=(18,6))
all_lines = []
all_labels= []

for i, params in enumerate(mixture_params_list):
    np.random.seed(seed_example)
    x_data_demo = sample_from_mixture(n_example, params)

    # True PDF
    x_plot_grid = np.linspace(-6,6,400)
    p_vals = mixture_pdf(x_plot_grid, params)
    axes[i].plot(x_plot_grid, p_vals, 'k--', label="True PDF")

    # Silverman
    h_silv_demo = silverman_bandwidth(x_data_demo)
    pdf_silv_vals = kde_pdf_eval(x_plot_grid, x_data_demo, h_silv_demo)
    mise_silv_demo = approximate_mise(x_data_demo, h_silv_demo, params)
    line_silv, = axes[i].plot(x_plot_grid, pdf_silv_vals, 'b-',
        label=f"Silverman (MISE={mise_silv_demo:.3f})")

    if i==0:
        all_lines.append(line_silv)
        all_labels.append(f"Silverman (MISE={mise_silv_demo:.3f})")

    # Score-Debiased
    colors = ['r','m','g','c']
    for idx, nl in enumerate(noise_levels):
        x_deb, h_deb = one_step_debiased_data(x_data_demo, params, noise_std=nl)
        pdf_deb_vals = kde_pdf_eval(x_plot_grid, x_deb, h_deb)
        mise_deb_demo= approximate_mise(x_deb, h_deb, params)

        label_str = f"Score-Deb(std={nl:.0f}) (MISE={mise_deb_demo:.3f})"
        line_deb, = axes[i].plot(x_plot_grid, pdf_deb_vals, colors[idx]+'-',
                        label=label_str)
        if i==0:
            all_lines.append(line_deb)
            all_labels.append(label_str)

    axes[i].set_title(f"Mixture {i+1}, seed={seed_example}")
    axes[i].set_xlabel("x")
    if i == 0:
        axes[i].set_ylabel("Density")

# Single legend, move it down to avoid covering x-axis
fig.legend(all_lines, all_labels, loc='lower center', ncol=2,
           bbox_to_anchor=(0.5, -0.06), borderaxespad=1.2)
plt.tight_layout(rect=[0,0.1,1,1])  # more bottom space
plt.savefig("figures/example_comparison_kdes.pdf", bbox_inches="tight")
plt.show()

###############################################################################
# 8) kl_diff_histograms.pdf (but actually MISE difference)
#    MISE(Silverman) - MISE(ScoreDeb(noise=0)), at n=200, over 100 seeds
###############################################################################
N_SEEDS = 100
N_DATA = 200

mise_diff_arrays = []  # store MISE(Silverman) - MISE(ScoreDeb(0)) for each mixture

for i, params in enumerate(mixture_params_list):
    diffs = []
    for seed in range(N_SEEDS):
        np.random.seed(seed)
        x_data = sample_from_mixture(N_DATA, params)

        # Silverman
        h_silv = silverman_bandwidth(x_data)
        mise_silv = approximate_mise(x_data, h_silv, params)

        # Score-Debiased(noise=0)
        x_deb, h_deb = one_step_debiased_data(x_data, params, noise_std=0)
        mise_deb = approximate_mise(x_deb, h_deb, params)

        diffs.append(mise_silv - mise_deb)
    mise_diff_arrays.append(np.array(diffs))

fig, axes = plt.subplots(1, 3, figsize=(18,6))
for i, params in enumerate(mixture_params_list):
    diffs = mise_diff_arrays[i]
    mean_diff = np.mean(diffs)
    std_diff  = np.std(diffs)

    ax = axes[i]
    ax.hist(diffs, bins=15, alpha=0.7, color='C0', edgecolor='k')
    ax.axvline(mean_diff, color='red', linestyle='--',
               label=f"mean={mean_diff:.3f}")
    ax.set_title(f"Mixture {i+1}\nMISE(Silv)-MISE(Deb(0)), std={std_diff:.3f}")
    ax.set_xlabel("MISE(Silverman) - MISE(Deb(0))")
    if i == 0:
        ax.set_ylabel("Count")
    ax.legend(loc='best')

plt.tight_layout()
plt.savefig("figures/kl_diff_histograms.pdf", bbox_inches="tight")
plt.show()

###############################################################################
# 9) SCALING EXPERIMENT: methods = [Silverman, ScoreDeb(noise=0,2,4,8)]
#    (A) scaling_experiment_kl.pdf
#    (B) scaling_experiment_mise.pdf
###############################################################################
#n_list = [10,20,50,100,200,500,1000,2000,4000,10000]
n_list = [10,20,50,100,200]
N_SEEDS_SCALING = 50

# We'll store arrays for each mixture & method
avg_kl_silver   = np.zeros((len(mixture_params_list), len(n_list)))
avg_mise_silver = np.zeros((len(mixture_params_list), len(n_list)))

avg_kl_deb_0    = np.zeros((len(mixture_params_list), len(n_list)))
avg_mise_deb_0  = np.zeros((len(mixture_params_list), len(n_list)))

avg_kl_deb_2    = np.zeros((len(mixture_params_list), len(n_list)))
avg_mise_deb_2  = np.zeros((len(mixture_params_list), len(n_list)))

avg_kl_deb_4    = np.zeros((len(mixture_params_list), len(n_list)))
avg_mise_deb_4  = np.zeros((len(mixture_params_list), len(n_list)))

avg_kl_deb_8    = np.zeros((len(mixture_params_list), len(n_list)))
avg_mise_deb_8  = np.zeros((len(mixture_params_list), len(n_list)))

for i, params in enumerate(mixture_params_list):
    for j, n_data in enumerate(n_list):
        kl_silv_vals = []
        mise_silv_vals= []

        kl_0_vals = []
        mise_0_vals= []

        kl_2_vals = []
        mise_2_vals= []

        kl_4_vals = []
        mise_4_vals= []

        kl_8_vals = []
        mise_8_vals= []

        for seed in range(N_SEEDS_SCALING):
            np.random.seed(seed)
            x_data = sample_from_mixture(n_data, params)

            # Silverman
            h_silv = silverman_bandwidth(x_data)
            kl_s   = kl_divergence(x_data, h_silv, params, nsamples=5000)
            mise_s = approximate_mise(x_data, h_silv, params, step=0.05)
            kl_silv_vals.append(kl_s)
            mise_silv_vals.append(mise_s)

            # Score-Deb(noise=0)
            x0, h0 = one_step_debiased_data(x_data, params, noise_std=0)
            kl0    = kl_divergence(x0, h0, params, nsamples=5000)
            mise0  = approximate_mise(x0, h0, params, step=0.05)
            kl_0_vals.append(kl0)
            mise_0_vals.append(mise0)

            # Score-Deb(noise=2)
            x2, h2 = one_step_debiased_data(x_data, params, noise_std=2)
            kl2    = kl_divergence(x2, h2, params, nsamples=5000)
            mise2  = approximate_mise(x2, h2, params, step=0.05)
            kl_2_vals.append(kl2)
            mise_2_vals.append(mise2)

            # Score-Deb(noise=4)
            x4, h4 = one_step_debiased_data(x_data, params, noise_std=4)
            kl4    = kl_divergence(x4, h4, params, nsamples=5000)
            mise4  = approximate_mise(x4, h4, params, step=0.05)
            kl_4_vals.append(kl4)
            mise_4_vals.append(mise4)

            # Score-Deb(noise=8)
            x8, h8 = one_step_debiased_data(x_data, params, noise_std=8)
            kl8    = kl_divergence(x8, h8, params, nsamples=5000)
            mise8  = approximate_mise(x8, h8, params, step=0.05)
            kl_8_vals.append(kl8)
            mise_8_vals.append(mise8)

        avg_kl_silver[i, j]   = np.mean(kl_silv_vals)
        avg_mise_silver[i, j] = np.mean(mise_silv_vals)

        avg_kl_deb_0[i, j]    = np.mean(kl_0_vals)
        avg_mise_deb_0[i, j]  = np.mean(mise_0_vals)

        avg_kl_deb_2[i, j]    = np.mean(kl_2_vals)
        avg_mise_deb_2[i, j]  = np.mean(mise_2_vals)

        avg_kl_deb_4[i, j]    = np.mean(kl_4_vals)
        avg_mise_deb_4[i, j]  = np.mean(mise_4_vals)

        avg_kl_deb_8[i, j]    = np.mean(kl_8_vals)
        avg_mise_deb_8[i, j]  = np.mean(mise_8_vals)

###############################################################################
# 9A) scaling_experiment_kl.pdf
###############################################################################
fig, axes = plt.subplots(1, 3, figsize=(18,6), sharey=False)

methods_kl = [
    ("Silverman KDE",         avg_kl_silver, 'o-b'),
    ("Score-Deb(std=0)",      avg_kl_deb_0,  's-r'),
    ("Score-Deb(std=2)",      avg_kl_deb_2,  'D-m'),
    ("Score-Deb(std=4)",      avg_kl_deb_4,  '^-g'),
    ("Score-Deb(std=8)",      avg_kl_deb_8,  'v-c'),
]

for i_m, params in enumerate(mixture_params_list):
    ax = axes[i_m]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (log scale)")
    if i_m == 0:
        ax.set_ylabel("Mean KL (log scale)")
    ax.set_title(f"Mixture {i_m+1}")

    for label_str, arr, style in methods_kl:
        ax.plot(n_list, arr[i_m], style, alpha=0.8, label=label_str)

    # Slopes in lower-left corner
    log_n = np.log(n_list)
    slope_texts = []
    for (lbl, arr, _) in methods_kl:
        y_vals = np.log(arr[i_m] + 1e-15)
        slope, _ = np.polyfit(log_n, y_vals, 1)
        slope_texts.append(f"{lbl}: {slope:.2f}")
    slope_str = "\n".join(slope_texts)

    ax.text(0.03, 0.05, slope_str, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.03))
plt.tight_layout(rect=[0,0.06,1,1])
plt.savefig("figures/scaling_experiment_kl.pdf", bbox_inches="tight")
plt.show()

###############################################################################
# 9B) scaling_experiment_mise.pdf
###############################################################################
fig, axes = plt.subplots(1, 3, figsize=(18,6), sharey=False)

methods_mise = [
    ("Silverman KDE",         avg_mise_silver, 'o-b'),
    ("Score-Deb(std=0)",      avg_mise_deb_0,  's-r'),
    ("Score-Deb(std=2)",      avg_mise_deb_2,  'D-m'),
    ("Score-Deb(std=4)",      avg_mise_deb_4,  '^-g'),
    ("Score-Deb(std=8)",      avg_mise_deb_8,  'v-c'),
]

for i_m, params in enumerate(mixture_params_list):
    ax = axes[i_m]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (log scale)")
    if i_m == 0:
        ax.set_ylabel("Mean MISE (log scale)")
    ax.set_title(f"Mixture {i_m+1}")

    for label_str, arr, style in methods_mise:
        ax.plot(n_list, arr[i_m], style, alpha=0.8, label=label_str)

    # Slopes in lower-left corner
    log_n = np.log(n_list)
    slope_texts = []
    for (lbl, arr, _) in methods_mise:
        y_vals = np.log(arr[i_m] + 1e-15)
        slope, _ = np.polyfit(log_n, y_vals, 1)
        slope_texts.append(f"{lbl}: {slope:.2f}")
    slope_str = "\n".join(slope_texts)

    ax.text(0.03, 0.05, slope_str, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.03))
plt.tight_layout(rect=[0,0.06,1,1])
plt.savefig("figures/scaling_experiment_mise.pdf", bbox_inches="tight")
plt.show()


