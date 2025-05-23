import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
###############################################################################
# Matplotlib settings for publication-quality plots
###############################################################################
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 19
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 17.7
plt.rcParams['xtick.labelsize'] = 17
plt.rcParams['ytick.labelsize'] = 17

# ### UPDATED ### Use Computer Modern (LaTeX font) for all plots
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']

LINEWIDTH = 2.5

###############################################################################
# 1) Define three different mixtures (Updated)
###############################################################################
mixture_params_list = [
    {'pi': 0.4, 'mu1': -2, 'sigma1': 0.5, 'mu2':  2, 'sigma2': 1.0},  # M1
    {'pi': 0.3, 'mu1': -2, 'sigma1': 0.4, 'mu2':  4, 'sigma2': 1.5},  # M2
    {'pi': 0.5, 'mu1':  0, 'sigma1': 0.4, 'mu2':  1.5, 'sigma2': 1.5},  # M3
]

###############################################################################
# 2) Mixture sampling & pdf
###############################################################################
from scipy.stats import norm

if not os.path.exists("figures"):
    os.makedirs("figures")

def kde_score_eval(x_points, data, bandwidth):
    """
    Returns the KDE-based empirical score, i.e. d/dx log(\hat{p}(x)),
    evaluated at each point in x_points, using 'data' and the given bandwidth.
    
    - \hat{p}(x) = 1/(n*h) sum_i phi((x - X_i)/h)
    - derivative wrt x => 1/(n*h^2) sum_i [ -z_i phi(z_i) ], with z_i=(x - X_i)/h
    - \hat{s}(x) = [d/dx \hat{p}(x)] / \hat{p}(x)
    """
    x_points = np.asarray(x_points)
    data = np.asarray(data)
    M = x_points.size
    N = data.size

    z = (x_points.reshape(M,1) - data.reshape(1,N)) / bandwidth
    pdf_mat = (1.0 / np.sqrt(2.0*np.pi)) * np.exp(-0.5 * z**2)

    # \hat{p}(x) = mean of phi(...) / bandwidth
    p_hat_x = pdf_mat.mean(axis=1) / bandwidth

    # derivative of \hat{p}(x)
    derivative_mat = -(z * pdf_mat)  # elementwise => -z_i * phi(z_i)
    d_hat_p_x = derivative_mat.mean(axis=1) / (bandwidth**2)

    # to avoid division by zero
    score_hat_x = d_hat_p_x / (p_hat_x + 1e-15)
    return score_hat_x

def one_step_debiased_data_emp_kde(x):
    """
    Performs one-step score-debiasing using an EMPIRICAL score (via Silverman-KDE)
    with no noise.
    
    x_{new} = x + (h^2/2)* \hat{s}(x),
    where \hat{s}(x) is d/dx log(\hat{p}(x)) from KDE.
    """
    n = len(x)
    std_dev = np.std(x)
    iqr     = np.percentile(x, 75) - np.percentile(x, 25)
    sigma   = min(std_dev, iqr / 1.34)

    # same step-size heuristic
    h = 0.4 * sigma * n**(-1/9)
    delta = (h**2) / 2.0

    # get the empirical score using the data itself
    #h_estimate_score = 0.9 * sigma * n**(-1/7)
    h_estimate_score = h
    s_hat = kde_score_eval(x, x, h_estimate_score)
    #s_hat = kde_score_eval(x, x, h)
    x_deb = x + delta * s_hat
    return x_deb, h


def mixture_pdf(x, params):
    pi_ = params['pi']
    return pi_*norm.pdf(x, params['mu1'], params['sigma1']) \
         + (1-pi_)*norm.pdf(x, params['mu2'], params['sigma2'])

def sample_from_mixture(n, params):
    pi_ = params['pi']
    z = np.random.rand(n) < pi_
    x_samps = np.zeros(n)
    x_samps[z]  = np.random.normal(params['mu1'], params['sigma1'], size=z.sum())
    x_samps[~z] = np.random.normal(params['mu2'], params['sigma2'], size=(~z).sum())
    return x_samps

###############################################################################
# 3) Score function (with optional noise)
###############################################################################
def score_function(x, params, noise_std=0.0):
    p_x = mixture_pdf(x, params)
    pi_ = params['pi']
    mu1, s1 = params['mu1'], params['sigma1']
    mu2, s2 = params['mu2'], params['sigma2']

    d_comp1 = pi_*norm.pdf(x, mu1, s1)*((mu1 - x)/(s1**2))
    d_comp2 = (1-pi_)*norm.pdf(x, mu2, s2)*((mu2 - x)/(s2**2))
    dp_dx   = d_comp1 + d_comp2
    base_score = dp_dx / (p_x + 1e-15)
    if noise_std>0:
        return base_score + np.random.normal(0, noise_std, size=x.shape)
    else:
        return base_score

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
    iqr     = np.percentile(x, 75) - np.percentile(x, 25)
    sigma   = min(std_dev, iqr / 1.34)

    h = 0.4*sigma*n**(-1/9)
    delta = (h**2)/2.0

    s_x = score_function(x, params, noise_std=noise_std)
    return x + delta*s_x, h

###############################################################################
# 5) Vectorized KDE
###############################################################################
def kde_pdf_eval(x_points, data, bandwidth):
    M = x_points.size
    N = data.size
    z = (x_points.reshape(M,1)-data.reshape(1,N))/bandwidth
    pdf_mat = (1.0/np.sqrt(2.0*np.pi))*np.exp(-0.5*z**2)
    return pdf_mat.mean(axis=1)/bandwidth

###############################################################################
# 6) KL divergence & MISE
###############################################################################
def kl_divergence(data, bandwidth, params, nsamples=10_000):
    x_samps = sample_from_mixture(nsamples, params)
    log_p   = np.log(mixture_pdf(x_samps, params)+1e-15)
    q_vals  = kde_pdf_eval(x_samps, data, bandwidth)+1e-15
    log_q   = np.log(q_vals)
    return np.mean(log_p-log_q)

def approximate_mise(data, bandwidth, params, x_min=-8, x_max=8, step=0.05):
    x_grid = np.arange(x_min, x_max+step, step)
    p_vals = mixture_pdf(x_grid, params)
    q_vals = kde_pdf_eval(x_grid, data, bandwidth)
    return np.sum((p_vals - q_vals)**2)*step

###############################################################################
# 7) example_comparison_kdes.pdf
###############################################################################
noise_levels = [0.0, 2.0, 4.0, 8.0]
n_example = 200
seed_example = 0

fig, axes = plt.subplots(1, 3, figsize=(24,6))
all_lines = []
all_labels= []

for i, params in enumerate(mixture_params_list):
    np.random.seed(seed_example)
    x_data_demo = sample_from_mixture(n_example, params)

    x_grid = np.linspace(-6,6,400)
    p_vals = mixture_pdf(x_grid, params)

    # Light gray grid
    axes[i].grid(True, linestyle='--', alpha=0.4, color='gray')

    # Subplot title => "Gaussian Mixture i"
    axes[i].set_title(f"Gaussian Mixture {i+1}")

    axes[i].plot(x_grid, p_vals, 'k--', label="True PDF")

    # Silverman
    h_silv = silverman_bandwidth(x_data_demo)
    pdf_silv = kde_pdf_eval(x_grid, x_data_demo, h_silv)
    mise_silv = approximate_mise(x_data_demo, h_silv, params)

    line_silv, = axes[i].plot(x_grid, pdf_silv, 'b-',
        label=f"Silverman (MISE={mise_silv:.3f})")
    if i==0:
        all_lines.append(line_silv)
        all_labels.append(f"Silverman KDE")

    # Score-Debiased
    colors = ['r','m','g','c']
    for idx, nl in enumerate(noise_levels):
        x_deb, h_deb = one_step_debiased_data(x_data_demo, params, noise_std=nl)
        pdf_deb = kde_pdf_eval(x_grid, x_deb, h_deb)
        mise_deb = approximate_mise(x_deb, h_deb, params)
        label_str = f"SD-KDE, std={nl:.0f}"
        line_deb, = axes[i].plot(x_grid, pdf_deb, colors[idx]+'-',
                                 label=label_str)
        if i==0:
            all_lines.append(line_deb)
            all_labels.append(label_str)

    axes[i].set_xlabel("x")
    if i==0:
        axes[i].set_ylabel("Density")

fig.legend(all_lines, all_labels, loc='lower center', ncol=2,
           bbox_to_anchor=(0.5, -0.06), borderaxespad=1.2)
plt.tight_layout(rect=[0,0.1,1,1])
plt.savefig("figures/example_comparison_kdes.pdf", bbox_inches="tight")

###############################################################################
# 8) mise_diff_histograms.pdf => MISE difference hist, rotate x ticks
###############################################################################
N_SEEDS = 100
N_DATA  = 200
mise_diff_arrays = []

for i, params in enumerate(mixture_params_list):
    diffs = []
    for seed in range(N_SEEDS):
        np.random.seed(seed)
        x_data = sample_from_mixture(N_DATA, params)
        h_silv = silverman_bandwidth(x_data)
        mise_silv = approximate_mise(x_data, h_silv, params)

        x_deb, h_deb = one_step_debiased_data(x_data, params, noise_std=0)
        mise_deb = approximate_mise(x_deb, h_deb, params)

        diffs.append(mise_silv - mise_deb)
    mise_diff_arrays.append(np.array(diffs))

fig, axes = plt.subplots(1,3, figsize=(24,6))
for i, params in enumerate(mixture_params_list):
    diffs = mise_diff_arrays[i]
    mean_diff = np.mean(diffs)
    std_diff  = np.std(diffs)

    ax = axes[i]
    # Light gray grid
    ax.grid(True, linestyle='--', alpha=0.4, color='gray')

    # Title => "Gaussian Mixture i"
    ax.set_title(f"Gaussian Mixture {i+1}")

    ax.hist(diffs, bins=15, alpha=0.7, color='C0', edgecolor='k')
    label_txt = f"mean={mean_diff:.3f}, std={std_diff:.3f}"
    ax.axvline(mean_diff, color='red', linestyle='--', label=label_txt)
    ax.set_xlabel("MISE(Silverman) - MISE(SD-KDE)")
    if i == 0:
        ax.set_ylabel("Count")
    ax.legend(loc='best')

    # ### UPDATED ### rotate x-tick labels 30 degrees
    ax.tick_params(axis='x', labelrotation=30)

plt.tight_layout()
plt.savefig("figures/mise_diff_histograms.pdf", bbox_inches="tight")

###############################################################################
# 9) SCALING EXPERIMENT => scaling_experiment_kl.pdf & scaling_experiment_mise.pdf
###############################################################################
#n_list = [10,20,50,100,200,500,1000,2000,5000,10000, 20000, 50000]
#n_list = [10,20,50,100,200]
n_list = [10,20,50,100,200,500,1000,2000,5000,10000,20000,50000] 
N_SEEDS_SCALING = 50

avg_kl_silver   = np.zeros((3, len(n_list)))
avg_mise_silver = np.zeros((3, len(n_list)))
avg_kl_deb_0    = np.zeros((3, len(n_list)))
avg_mise_deb_0  = np.zeros((3, len(n_list)))
avg_kl_deb_2    = np.zeros((3, len(n_list)))
avg_mise_deb_2  = np.zeros((3, len(n_list)))
avg_kl_deb_4    = np.zeros((3, len(n_list)))
avg_mise_deb_4  = np.zeros((3, len(n_list)))
avg_kl_deb_8    = np.zeros((3, len(n_list)))
avg_mise_deb_8  = np.zeros((3, len(n_list)))

for i, params in enumerate(mixture_params_list):
    for j, n_data in enumerate(n_list):
        kl_silv_vals, mise_silv_vals = [], []
        kl_0_vals,   mise_0_vals    = [], []
        kl_2_vals,   mise_2_vals    = [], []
        kl_4_vals,   mise_4_vals    = [], []
        kl_8_vals,   mise_8_vals    = [], []

        for seed in range(N_SEEDS_SCALING):
            np.random.seed(seed)
            x_data = sample_from_mixture(n_data, params)

            h_silv = silverman_bandwidth(x_data)
            kl_s   = kl_divergence(x_data, h_silv, params, nsamples=5000)
            mise_s = approximate_mise(x_data, h_silv, params, step=0.05)
            kl_silv_vals.append(kl_s)
            mise_silv_vals.append(mise_s)

            x0, h0 = one_step_debiased_data(x_data, params, noise_std=0)
            kl0    = kl_divergence(x0, h0, params, nsamples=5000)
            mise0  = approximate_mise(x0, h0, params, step=0.05)
            kl_0_vals.append(kl0)
            mise_0_vals.append(mise0)

            x2, h2 = one_step_debiased_data(x_data, params, noise_std=2)
            kl2    = kl_divergence(x2, h2, params, nsamples=5000)
            mise2  = approximate_mise(x2, h2, params, step=0.05)
            kl_2_vals.append(kl2)
            mise_2_vals.append(mise2)

            x4, h4 = one_step_debiased_data(x_data, params, noise_std=4)
            kl4    = kl_divergence(x4, h4, params, nsamples=5000)
            mise4  = approximate_mise(x4, h4, params, step=0.05)
            kl_4_vals.append(kl4)
            mise_4_vals.append(mise4)

            x8, h8 = one_step_debiased_data(x_data, params, noise_std=8)
            kl8    = kl_divergence(x8, h8, params, nsamples=5000)
            mise8  = approximate_mise(x8, h8, params, step=0.05)
            kl_8_vals.append(kl8)
            mise_8_vals.append(mise8)

        avg_kl_silver[i,j]   = np.mean(kl_silv_vals)
        avg_mise_silver[i,j] = np.mean(mise_silv_vals)

        avg_kl_deb_0[i,j] = np.mean(kl_0_vals)
        avg_mise_deb_0[i,j] = np.mean(mise_0_vals)

        avg_kl_deb_2[i,j] = np.mean(kl_2_vals)
        avg_mise_deb_2[i,j] = np.mean(mise_2_vals)

        avg_kl_deb_4[i,j] = np.mean(kl_4_vals)
        avg_mise_deb_4[i,j] = np.mean(mise_4_vals)

        avg_kl_deb_8[i,j] = np.mean(kl_8_vals)
        avg_mise_deb_8[i,j] = np.mean(mise_8_vals)




###############################################################################
# scaling_experiment_kl.pdf
###############################################################################
fig, axes = plt.subplots(1, 3, figsize=(24,7), sharey=False)

methods_kl = [
    ("Silverman KDE",         avg_kl_silver, 'o-b'),
    ("SD-KDE, std = 0",         avg_kl_deb_0,  's-r'),
    ("SD-KDE, std = 2",         avg_kl_deb_2,  'D-m'),
    ("SD-KDE, std = 4",         avg_kl_deb_4,  '^-g'),
    ("SD-KDE, std = 8",         avg_kl_deb_8,  'v-c'),
]

for i_m in range(3):
    ax = axes[i_m]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Gaussian Mixture {i_m+1}")
    ax.set_xlabel(r"$n$ (log scale)")

    # Light gray grid
    ax.grid(True, linestyle='--', alpha=0.4, color='gray')

    if i_m == 0:
        ax.set_ylabel(r"Mean KL (log scale)")

    # Plot each method
    for label_str, arr, style in methods_kl:
        ax.plot(n_list, arr[i_m], style, alpha=0.8, label=label_str)

    # dotted black line with slope -8/9
    ref_idx = 2  # e.g. n=50
    anchor_val = np.maximum(1e-15, methods_kl[0][1][i_m][ref_idx])
    anchor_n   = n_list[ref_idx]
    c = anchor_val * (anchor_n**(8/9.0))
    theory_vals = [c*(it**(-8/9.0)) for it in n_list]
    theory_line, = ax.plot(n_list, theory_vals, 'k:', alpha=0.8)
    
    if i_m == 0:
        theory_line.set_label(r"Theoretical Convergence Rate $n^{-8/9}$")
    else:
        theory_line.set_label(None)

    # slope text
    log_n = np.log(n_list)
    slope_texts = []
    for (lbl, arr, _) in methods_kl:
        y_vals = np.log(arr[i_m] + 1e-15)
        slope, _ = np.polyfit(log_n, y_vals, 1)
        slope_texts.append(f"{lbl}: {slope:.2f}")
    slope_str = "\n".join(slope_texts)

    ax.text(0.03, 0.05, slope_str,
            transform=ax.transAxes, ha='left', va='bottom', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

handles, labels = axes[0].get_legend_handles_labels()
# ### UPDATED ### Move legend down slightly
fig.legend(handles, labels, loc='lower center', ncol=3,
           bbox_to_anchor=(0.5, -0.04))  # shift from -0.03 to -0.08
plt.tight_layout(rect=[0,0.07,1,1])     # allow more bottom space
plt.savefig("figures/scaling_experiment_kl.pdf", bbox_inches="tight")


###############################################################################
# scaling_experiment_mise.pdf
###############################################################################
fig, axes = plt.subplots(1, 3, figsize=(24,6.5), sharey=False)

methods_mise = [
    ("Silverman KDE",         avg_mise_silver, {'marker': 'o', 'linestyle': '-', 'color': 'red'}),
    ("SD-KDE, std = 0",         avg_mise_deb_0,  {'marker': 's', 'linestyle': '-', 'color': 'navy'}), 
    ("SD-KDE, std = 2",         avg_mise_deb_2,  {'marker': 'D', 'linestyle': '-', 'color': 'royalblue'}),
    ("SD-KDE, std = 4",         avg_mise_deb_4,  {'marker': '^', 'linestyle': '-', 'color': 'cornflowerblue'}),
    #("SD-KDE, std=8",         avg_mise_deb_8,  {'marker': 'v', 'linestyle': '-', 'color': 'lightskyblue'})
]
# Save just the labels and data arrays, not the style dictionaries
methods_mise_save = [(label, arr) for label, arr, _ in methods_mise]
methods_dict = {label: arr for label, arr in methods_mise_save}
np.save('methods_mise.npy', methods_dict)

for i_m in range(3):
    ax = axes[i_m]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Gaussian Mixture {i_m+1}")
    ax.set_xlabel(r"$n$ (log scale)")

    # Light gray grid
    ax.grid(True, linestyle='--', alpha=0.4, color='gray')

    if i_m == 0:
        ax.set_ylabel(r"Mean MISE (log scale)")

    for label_str, arr, style in methods_mise:
        if isinstance(style, str):  # String format like 'r:'
            ax.plot(n_list, arr[i_m], style, alpha=0.8, label=label_str, linewidth=LINEWIDTH)
        else:  # Dictionary format with explicit style parameters
            ax.plot(n_list, arr[i_m], alpha=0.8, label=label_str, linewidth=LINEWIDTH, **style)

    # dotted black line slope -8/9
    ref_idx = 0
    anchor_val = (n_list[ref_idx]**(4/5))*np.maximum(1e-15, methods_mise[3][1][i_m][ref_idx])  # Use SD-KDE std=4 values
    #breakpoint()
    theory_vals = [anchor_val*(it**(-8/9.0)) for it in n_list]
    theory_vals_silverman = [anchor_val*(it**(-4/5.0)) for it in n_list]
    # Dashed line
    theory_line, = ax.plot(n_list, theory_vals, alpha=0.8, color='navy', linestyle=":", linewidth=LINEWIDTH)
    theory_line_silverman, = ax.plot(n_list, theory_vals_silverman, alpha=0.8, color='r', linestyle=":", linewidth=LINEWIDTH)
    
    
    if i_m == 0:
        theory_line.set_label(r"Theoretical Convergence Rate SD-KDE $n^{-8/9}$")
        theory_line_silverman.set_label(r"Theoretical Convergence Rate Silverman $n^{-4/5}$")
    else:
        theory_line.set_label(None)

    # slope text
    log_n = np.log(n_list)
    slope_texts = []
    for (lbl, arr, _) in methods_mise:
        y_vals = np.log(arr[i_m] + 1e-15)
        slope, _ = np.polyfit(log_n, y_vals, 1)
        slope_texts.append(f"{lbl}: {slope:.2f}")

    slope_str = "\n".join(slope_texts)
    ax.text(0.03, 0.05, slope_str,
            transform=ax.transAxes, ha='left', va='bottom', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

handles, labels = axes[0].get_legend_handles_labels()
# ### UPDATED ### Move legend down slightly
fig.legend(handles, labels, loc='lower center', ncol=3,
           bbox_to_anchor=(0.5, -0.04))
plt.tight_layout(rect=[0,0.07,1,1])
plt.savefig("figures/scaling_experiment_mise.pdf", bbox_inches="tight")

plot_empirical = True
if plot_empirical:
    ###############################################################################
    # 10) Empirical SD-KDE vs Silverman KDE vs Oracle SD-KDE => empirical_sd_kde_mise.pdf
    ###############################################################################
    plot_theory_lines = False  # set True if you want dotted reference slopes again

    # Reuse the same n_list and N_SEEDS_SCALING from above
    # n_list = [10,20,50,100,200]
    # N_SEEDS_SCALING = 10

    # Create arrays to store average MISE over multiple seeds
    emp_avg_mise_silver = np.zeros((3, len(n_list)))
    emp_avg_mise_emp    = np.zeros((3, len(n_list)))
    emp_avg_mise_oracle = np.zeros((3, len(n_list)))  # SD-KDE w/ oracle score (noise_std=0)

    for i, params in enumerate(mixture_params_list):
        for j, n_data in enumerate(n_list):
            mise_silverman_vals = []
            mise_emp_vals       = []
            mise_oracle_vals    = []

            for seed in range(N_SEEDS_SCALING):
                np.random.seed(seed)
                x_data = sample_from_mixture(n_data, params)

                # 1) Silverman bandwidth + KDE
                h_silv = silverman_bandwidth(x_data)
                mise_silver = approximate_mise(x_data, h_silv, params, step=0.05)
                mise_silverman_vals.append(mise_silver)

                # 2) Oracle SD-KDE (score_function w/ noise_std=0)
                x_oracle, h_oracle = one_step_debiased_data(x_data, params, noise_std=0.0)
                mise_oracle = approximate_mise(x_oracle, h_oracle, params, step=0.05)
                mise_oracle_vals.append(mise_oracle)

                # 3) Empirical SD-KDE
                x_emp, h_emp = one_step_debiased_data_emp_kde(x_data)
                mise_emp_val = approximate_mise(x_emp, h_emp, params, step=0.05)
                mise_emp_vals.append(mise_emp_val)

            emp_avg_mise_silver[i,j] = np.mean(mise_silverman_vals)
            emp_avg_mise_emp[i,j]    = np.mean(mise_emp_vals)
            emp_avg_mise_oracle[i,j] = np.mean(mise_oracle_vals)

    # Now plot the results in log-log space, similar to 'scaling_experiment_mise.pdf'
    fig, axes = plt.subplots(1, 3, figsize=(24,6.5), sharey=False)

    # We have three methods now:
    methods_mise_all = [
        ("Silverman KDE", emp_avg_mise_silver,
         {'marker': 'o', 'linestyle': '-', 'color': 'red'}),
        
        ("Emp-SD-KDE", emp_avg_mise_emp,
         {'marker': 'D', 'linestyle': '-', 'color': 'purple'}),
         ("SD-KDE", emp_avg_mise_oracle,
         {'marker': 's', 'linestyle': '-', 'color': 'navy'}),
    ]

    for i_m in range(3):
        ax = axes[i_m]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"Gaussian Mixture {i_m+1}")
        ax.set_xlabel(r"$n$ (log scale)")

        # Light gray grid
        ax.grid(True, linestyle='--', alpha=0.4, color='gray')

        if i_m == 0:
            ax.set_ylabel(r"Mean MISE (log scale)")

        # Plot each method
        for label_str, arr, style_dict in methods_mise_all:
            ax.plot(n_list, arr[i_m],
                    alpha=0.8, label=label_str, linewidth=LINEWIDTH, **style_dict)

        # Optionally add reference lines for theoretical convergence rates
        if plot_theory_lines:
            log_n = np.log(n_list)
            anchor_idx = 0

            # For Silverman ~ n^{-4/5}
            silver_anchor_val = emp_avg_mise_silver[i_m][anchor_idx] + 1e-15
            anchor_n = n_list[anchor_idx]
            c_silver = silver_anchor_val * (anchor_n ** (4/5.0))
            silver_theory_vals = [c_silver * (it ** (-4/5.0)) for it in n_list]
            silver_theory_line, = ax.plot(n_list, silver_theory_vals, 
                                          alpha=0.8, color='r', linestyle=":",
                                          linewidth=LINEWIDTH)

            # Emp/Oracle SD-KDE ~ n^{-8/9}
            emp_anchor_val = emp_avg_mise_emp[i_m][anchor_idx] + 1e-15
            c_emp = emp_anchor_val * (anchor_n ** (8/9.0))
            emp_theory_vals = [c_emp * (it ** (-8/9.0)) for it in n_list]
            emp_theory_line, = ax.plot(n_list, emp_theory_vals,
                                       alpha=0.8, color='purple', linestyle=":",
                                       linewidth=LINEWIDTH)

            if i_m == 0:
                silver_theory_line.set_label(r"Silverman $\sim n^{-4/5}$")
                emp_theory_line.set_label(r"SD-KDE $\sim n^{-8/9}$")

        # Add slope text
        log_n = np.log(n_list)
        slope_texts = []
        for (lbl, arr, _) in methods_mise_all:
            y_vals = np.log(arr[i_m] + 1e-15)
            slope, _ = np.polyfit(log_n, y_vals, 1)
            slope_texts.append(f"{lbl}: slope={slope:.2f}")

        slope_str = "\n".join(slope_texts)
        ax.text(0.03, 0.05, slope_str,
                transform=ax.transAxes, ha='left', va='bottom', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout(rect=[0,0.07,1,1])
    plt.savefig("figures/empirical_sd_kde_mise.pdf", bbox_inches="tight")


