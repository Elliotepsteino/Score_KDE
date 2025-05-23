import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
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

# Use LaTeX + Computer Modern
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']

###############################################################################
# 1) Define three Laplace mixtures
###############################################################################
mixture_params_list = [
    # Mixture 1
    {'pi': 0.4, 'mu1': -2, 'sigma1': 0.5, 'mu2':  2, 'sigma2': 1.0},
    # Mixture 2
    {'pi': 0.3, 'mu1': -2, 'sigma1': 0.4, 'mu2':  4, 'sigma2': 1.5},
    # Mixture 3
    {'pi': 0.5, 'mu1':  0, 'sigma1': 0.4, 'mu2':  1.5, 'sigma2': 1.5},
]

if not os.path.exists("figures"):
    os.makedirs("figures")

###############################################################################
# 2) Laplace PDF and Mixture
###############################################################################
def laplace_pdf(x, loc, scale):
    return (1.0/(2.0*scale)) * np.exp(-np.abs(x - loc)/scale)

def mixture_laplace_pdf(x, params):
    pi_ = params['pi']
    p1 = laplace_pdf(x, params['mu1'], params['sigma1'])
    p2 = laplace_pdf(x, params['mu2'], params['sigma2'])
    return pi_*p1 + (1 - pi_)*p2

def sample_laplace_mixture(n, params):
    pi_ = params['pi']
    z = np.random.rand(n) < pi_
    x = np.zeros(n)
    x[z] = np.random.laplace(params['mu1'], params['sigma1'], size=z.sum())
    x[~z]= np.random.laplace(params['mu2'], params['sigma2'], size=(~z).sum())
    return x

###############################################################################
# 3) Laplace mixture score function
###############################################################################
def laplace_score_function(x, params, noise_std=0.0):
    pi_ = params['pi']
    p_x = mixture_laplace_pdf(x, params)

    p1 = laplace_pdf(x, params['mu1'], params['sigma1'])
    sign1 = np.sign(x - params['mu1'])
    deriv1 = pi_ * p1 * ( - sign1/(params['sigma1']+1e-15) )

    p2 = laplace_pdf(x, params['mu2'], params['sigma2'])
    sign2 = np.sign(x - params['mu2'])
    deriv2 = (1 - pi_) * p2 * ( - sign2/(params['sigma2']+1e-15) )

    dp_dx = deriv1 + deriv2
    base_score = dp_dx / (p_x + 1e-15)

    if noise_std > 0:
        noise = np.random.normal(0, noise_std, size=x.shape)
        return base_score + noise
    else:
        return base_score

###############################################################################
# 4) Silverman bandwidth & one-step update
###############################################################################
def silverman_bandwidth(data):
    n = len(data)
    std_dev = np.std(data)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    sigma = min(std_dev, iqr/1.34)
    return 0.9 * sigma * n**(-1/5)

def one_step_debiased_data(x, params, noise_std=0.0):
    n = len(x)
    std_dev = np.std(x)
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    sigma_est = min(std_dev, iqr/1.34)

    h = 0.4 * sigma_est * n**(-1/9)
    delta = (h**2)/2.0

    s_vals = laplace_score_function(x, params, noise_std=noise_std)
    x_new = x + delta*s_vals
    return x_new, h

###############################################################################
# 5) Evaluate KDE
###############################################################################
def kde_pdf_eval(x_grid, data, bandwidth):
    M = x_grid.size
    kde_vals = np.zeros(M)
    for xi in data:
        z = (x_grid - xi)/bandwidth
        pdf_part = (1.0/np.sqrt(2.0*np.pi))*np.exp(-0.5*z**2)
        kde_vals += pdf_part/bandwidth
    kde_vals /= len(data)
    return kde_vals

###############################################################################
# 6) MISE & KL for Laplace mixture
###############################################################################
def approximate_mise_laplace(x_data, params, bandwidth, 
                             x_min=-8, x_max=8, step=0.05):
    grid = np.arange(x_min, x_max+step, step)
    p_vals = mixture_laplace_pdf(grid, params)
    kde_vals= kde_pdf_eval(grid, x_data, bandwidth)
    return np.sum((p_vals - kde_vals)**2)*step

def kl_divergence_laplace(x_data, params, bandwidth, nsamples=10000):
    z_samps = sample_laplace_mixture(nsamples, params)
    log_p   = np.log(mixture_laplace_pdf(z_samps, params)+1e-15)

    q_vals = np.zeros(nsamples)
    n_data = len(x_data)
    for i in range(nsamples):
        xi = z_samps[i]
        z = (xi - x_data)/bandwidth
        pdf_part = (1.0/np.sqrt(2.0*np.pi))*np.exp(-0.5*z**2)
        q_vals[i]= pdf_part.sum()/(n_data*bandwidth)
    log_q = np.log(q_vals+1e-15)

    return np.mean(log_p - log_q)

###############################################################################
# 7) Run Laplace experiments
###############################################################################
def run_experiments(n_list, n_seeds):
    """
    Run all experiments with given parameters
    
    Args:
        n_list: List of n values for scaling experiments
        n_seeds: Number of seeds for averaging results
    """
    import numpy as np
    
    # Example comparison KDEs
    noise_levels = [0.0, 2.0, 4.0, 8.0]
    n_example = 200
    seed_example = 0

    fig, axes = plt.subplots(1, 3, figsize=(18,6))
    all_lines = []
    all_labels= []

    for i, params in enumerate(mixture_params_list):
        np.random.seed(seed_example)
        x_data_demo = sample_laplace_mixture(n_example, params)

        x_grid = np.linspace(-6,6,400)
        p_vals = mixture_laplace_pdf(x_grid, params)

        # Light gray grid
        axes[i].grid(True, linestyle='--', alpha=0.4, color='gray')

        # Subplot title => "Laplace Mixture i"
        axes[i].set_title(f"Laplace Mixture {i+1}")

        # Plot true PDF
        axes[i].plot(x_grid, p_vals, 'k--', label="True PDF")

        # Silverman
        h_silv = silverman_bandwidth(x_data_demo)
        pdf_silv = kde_pdf_eval(x_grid, x_data_demo, h_silv)
        # remove (MISE=xxx) from the legend
        line_silv, = axes[i].plot(x_grid, pdf_silv, 'b-',
                                  label="Silverman")

        if i==0:
            all_lines.append(line_silv)
            all_labels.append("Silverman KDE")

        # SD-KDE
        colors = ['r','m','g','c']
        for idx, nl in enumerate(noise_levels):
            x_deb, h_deb = one_step_debiased_data(x_data_demo, params, noise_std=nl)
            pdf_deb = kde_pdf_eval(x_grid, x_deb, h_deb)

            # remove (MISE=xxx), rename Score-Deb => "SD-KDE, std=..."
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
    plt.savefig("figures/example_comparison_kdes_laplace.pdf", bbox_inches="tight")
    plt.show()

    # KL difference histograms
    N_DATA = 200
    mise_diff_arrays = []

    for i, params in enumerate(mixture_params_list):
        diffs = []
        for seed in range(n_seeds):
            np.random.seed(seed)
            x_data = sample_laplace_mixture(N_DATA, params)

            h_silv = silverman_bandwidth(x_data)
            mise_silv = approximate_mise_laplace(x_data, params, h_silv)

            x_deb, h_deb = one_step_debiased_data(x_data, params, noise_std=0)
            mise_deb = approximate_mise_laplace(x_deb, params, h_deb)

            diffs.append(mise_silv - mise_deb)
        mise_diff_arrays.append(np.array(diffs))

    fig, axes = plt.subplots(1,3, figsize=(18,6))
    for i, params in enumerate(mixture_params_list):
        diffs = mise_diff_arrays[i]
        mean_diff = np.mean(diffs)
        std_diff  = np.std(diffs)

        ax = axes[i]
        ax.grid(True, linestyle='--', alpha=0.4, color='gray')  # add grid
        ax.set_title(f"Laplace Mixture {i+1}")
        ax.hist(diffs, bins=15, alpha=0.7, color='C0', edgecolor='k')

        # legend label => mean=..., std=...
        label_txt = f"mean={mean_diff:.3f}, std={std_diff:.3f}"
        ax.axvline(mean_diff, color='red', linestyle='--', label=label_txt)
        ax.set_xlabel("MISE(Silverman) - MISE(SD-KDE)")
        if i==0:
            ax.set_ylabel("Count")
        ax.legend(loc='best')

        # rotate x ticks 30 deg
        ax.tick_params(axis='x', labelrotation=30)

    plt.tight_layout()
    plt.savefig("figures/kl_diff_histograms_laplace.pdf", bbox_inches="tight")
    plt.show()

    # Initialize arrays for scaling experiments
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

    # Run scaling experiments
    for i, params in enumerate(mixture_params_list):
        for j, n_data in enumerate(n_list):
            kl_silv_vals   = []
            mise_silv_vals = []
            kl_0_vals, mise_0_vals  = [], []
            kl_2_vals, mise_2_vals  = [], []
            kl_4_vals, mise_4_vals  = [], []
            kl_8_vals, mise_8_vals  = [], []

            for seed in range(n_seeds):
                np.random.seed(seed)
                x_data = sample_laplace_mixture(n_data, params)

                # Silverman
                h_silv = silverman_bandwidth(x_data)
                kl_s   = kl_divergence_laplace(x_data, params, h_silv, nsamples=5000)
                mise_s = approximate_mise_laplace(x_data, params, h_silv)
                kl_silv_vals.append(kl_s)
                mise_silv_vals.append(mise_s)

                # SD-KDE(noise=0)
                x0, h0 = one_step_debiased_data(x_data, params, noise_std=0)
                kl0    = kl_divergence_laplace(x0, params, h0, nsamples=5000)
                mise0  = approximate_mise_laplace(x0, params, h0)
                kl_0_vals.append(kl0)
                mise_0_vals.append(mise0)

                x2, h2 = one_step_debiased_data(x_data, params, noise_std=2)
                kl2    = kl_divergence_laplace(x2, params, h2, nsamples=5000)
                mise2  = approximate_mise_laplace(x2, params, h2)
                kl_2_vals.append(kl2)
                mise_2_vals.append(mise2)

                x4, h4 = one_step_debiased_data(x_data, params, noise_std=4)
                kl4    = kl_divergence_laplace(x4, params, h4, nsamples=5000)
                mise4  = approximate_mise_laplace(x4, params, h4)
                kl_4_vals.append(kl4)
                mise_4_vals.append(mise4)

                x8, h8 = one_step_debiased_data(x_data, params, noise_std=8)
                kl8    = kl_divergence_laplace(x8, params, h8, nsamples=5000)
                mise8  = approximate_mise_laplace(x8, params, h8)
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

    # Plot KL results
    fig, axes = plt.subplots(1,3, figsize=(18,6), sharey=False)

    # ### rename "Score-Deb(std=)" => "SD-KDE, std=.."
    methods_kl = [
        ("Silverman KDE",         avg_kl_silver, 'o-b'),
        ("SD-KDE, std=0",         avg_kl_deb_0,  's-r'),
        ("SD-KDE, std=2",         avg_kl_deb_2,  'D-m'),
        ("SD-KDE, std=4",         avg_kl_deb_4,  '^-g'),
        ("SD-KDE, std=8",         avg_kl_deb_8,  'v-c'),
    ]

    for i_m, params in enumerate(mixture_params_list):
        ax = axes[i_m]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$n$ (log scale)")
        if i_m==0:
            ax.set_ylabel("Mean KL (log scale)")
        ax.set_title(f"Laplace Mixture {i_m+1}")

        ax.grid(True, linestyle='--', alpha=0.4, color='gray')

        for label_str, arr, style in methods_kl:
            ax.plot(n_list, arr[i_m], style, alpha=0.8, label=label_str)

        import numpy as np
        log_n = np.log(n_list)
        slope_texts = []
        for (lbl, arr, _) in methods_kl:
            y_vals = np.log(arr[i_m]+1e-15)
            slope, _ = np.polyfit(log_n, y_vals, 1)
            slope_texts.append(f"{lbl}: {slope:.2f}")
        slope_str = "\n".join(slope_texts)
        ax.text(0.03, 0.05, slope_str, transform=ax.transAxes,
                ha='left', va='bottom', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.03))
    plt.tight_layout(rect=[0,0.06,1,1])
    plt.savefig("figures/scaling_experiment_kl_laplace.pdf", bbox_inches="tight")
    plt.show()

    # Plot MISE results
    fig, axes = plt.subplots(1,3, figsize=(18,6), sharey=False)
    methods_mise = [
        ("Silverman KDE",         avg_mise_silver, 'o-b'),
        ("SD-KDE, std=0",         avg_mise_deb_0,  's-r'),
        ("SD-KDE, std=2",         avg_mise_deb_2,  'D-m'),
        ("SD-KDE, std=4",         avg_mise_deb_4,  '^-g'),
        ("SD-KDE, std=8",         avg_mise_deb_8,  'v-c'),
    ]

    for i_m, params in enumerate(mixture_params_list):
        ax = axes[i_m]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$n$ (log scale)")
        if i_m==0:
            ax.set_ylabel("Mean MISE (log scale)")
        ax.set_title(f"Laplace Mixture {i_m+1}")

        ax.grid(True, linestyle='--', alpha=0.4, color='gray')

        for label_str, arr, style in methods_mise:
            ax.plot(n_list, arr[i_m], style, alpha=0.8, label=label_str)

        slope_texts = []
        log_n = np.log(n_list)
        for (lbl, arr, _) in methods_mise:
            y_vals = np.log(arr[i_m]+1e-15)
            slope, _ = np.polyfit(log_n, y_vals, 1)
            slope_texts.append(f"{lbl}: {slope:.2f}")
        slope_str = "\n".join(slope_texts)

        ax.text(0.03, 0.05, slope_str, transform=ax.transAxes,
                ha='left', va='bottom', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.03))
    plt.tight_layout(rect=[0,0.06,1,1])
    plt.savefig("figures/scaling_experiment_mise_laplace.pdf", bbox_inches="tight")
    plt.show()

def main():
    """Main function with argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Laplace KDE experiments')
    parser.add_argument('--n_seeds', type=int, default=50,
                      help='Number of seeds for averaging results (default: 50)')
    parser.add_argument('--n_list', type=str, 
                      default='10,20,50,100,200,500,1000,2000,5000,10000',
                      help='Comma-separated list of n values (default: 10,20,50,...,10000)')
    
    args = parser.parse_args()
    # python shrinkage_kde_laplace.py --n_seeds 50 --n_list "10,20,50,100"
    # Parse n_list from string
    n_list = [int(n) for n in args.n_list.split(',')]
    
    # Create figures directory if it doesn't exist
    if not os.path.exists("figures"):
        os.makedirs("figures")
    
    # Run all experiments
    run_experiments(n_list, args.n_seeds)

if __name__ == "__main__":
    main()
