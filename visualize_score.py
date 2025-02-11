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
# Common mixture parameters (numbers are interpreted differently for Laplace)
###############################################################################
mixture_params_list = [
    # Mixture 1
    {'pi': 0.4, 'mu1': -2, 'sigma1': 0.5, 'mu2':  2, 'sigma2': 1.0},
    # Mixture 2
    {'pi': 0.3, 'mu1': -2, 'sigma1': 0.4, 'mu2':  4, 'sigma2': 1.5},
    # Mixture 3
    {'pi': 0.5, 'mu1':  0, 'sigma1': 0.4, 'mu2':  0, 'sigma2': 1.5},
]

###############################################################################
# (A) Gaussian Mixture code
###############################################################################
def gaussian_pdf(x, mu, sigma):
    return norm.pdf(x, loc=mu, scale=sigma)

def mixture_gaussian_pdf(x, params):
    pi_ = params['pi']
    p1  = gaussian_pdf(x, params['mu1'], params['sigma1'])
    p2  = gaussian_pdf(x, params['mu2'], params['sigma2'])
    return pi_*p1 + (1 - pi_)*p2

def gaussian_score_function(x, params):
    """
    s(x) = [d/dx( p(x) )] / p(x), for a 2-component Gaussian mixture
    """
    pi_ = params['pi']
    p_x = mixture_gaussian_pdf(x, params)

    comp1_deriv = pi_ * gaussian_pdf(x, params['mu1'], params['sigma1']) \
                       * ((params['mu1'] - x)/(params['sigma1']**2))
    comp2_deriv = (1-pi_) * gaussian_pdf(x, params['mu2'], params['sigma2']) \
                          * ((params['mu2'] - x)/(params['sigma2']**2))
    dp_dx = comp1_deriv + comp2_deriv
    return dp_dx / (p_x + 1e-15)

def plot_gaussian_score_functions_pub(mixture_params_list, x_min=-6, x_max=6):
    """
    Produces score_function_gaussian.pdf with 3 subplots (one per mixture):
      - In each subplot:
         Left axis => Density (blue)
         Right axis => Score (red)
         Ticks on both sides in black
      - ONLY subplot #1 has 'Density' label (left)
      - ONLY subplot #3 has 'Score' label (right)
      - Subplot #2 => no y-axis labels
    A single legend: 'Density' (blue), 'Score' (red).
    Subplot titles are "Mixture 1 Gaussian", "Mixture 2 Gaussian", etc.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18,6))
    x_grid = np.linspace(x_min, x_max, 400)

    for i, params in enumerate(mixture_params_list):
        ax_left = axes[i]                 # left axis => density
        ax_right = ax_left.twinx()        # right axis => score

        # Make spines/ticks black on both axes
        for spine in ax_left.spines.values():
            spine.set_color('black')
        ax_left.tick_params(axis='x', colors='black')
        ax_left.tick_params(axis='y', colors='black')

        for spine in ax_right.spines.values():
            spine.set_color('black')
        ax_right.tick_params(axis='y', colors='black')

        # Evaluate data
        density_vals = mixture_gaussian_pdf(x_grid, params)
        score_vals   = gaussian_score_function(x_grid, params)

        # Plot
        ax_left.plot(x_grid, density_vals, color='blue')
        ax_right.plot(x_grid, score_vals,  color='red')

        # Title, x-label
        ax_left.set_title(f"Mixture {i+1} Gaussian", color='black')
        ax_left.set_xlabel("x", color='black')

        # Subplot #1 => label "Density" on left
        # Subplot #3 => label "Score" on right
        if i == 0:
            ax_left.set_ylabel("Density", color='black')
        elif i == 2:
            ax_right.set_ylabel("Score", color='black')

    # Single legend at bottom
    import matplotlib.lines as mlines
    density_line = mlines.Line2D([], [], color='blue', label='Density')
    score_line   = mlines.Line2D([], [], color='red',  label='Score')
    fig.legend([density_line, score_line], ['Density', 'Score'],
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0,0.06,1,1])
    plt.savefig("figures/score_function_gaussian.pdf", bbox_inches="tight")
    plt.show()

###############################################################################
# (B) Laplace Mixture code
###############################################################################
def laplace_pdf(x, loc, scale):
    # Laplace pdf = 1/(2*scale) * exp(-|x-loc|/scale)
    return (1.0/(2.0*scale)) * np.exp(-np.abs(x - loc)/scale)

def mixture_laplace_pdf(x, params):
    pi_ = params['pi']
    p1  = laplace_pdf(x, params['mu1'], params['sigma1'])
    p2  = laplace_pdf(x, params['mu2'], params['sigma2'])
    return pi_*p1 + (1 - pi_)*p2

def laplace_score_function(x, params):
    """
    s(x) = derivative wrt x of log( p(x) ) for a 2-component Laplace mixture
           => sum of each component's derivative / mixture pdf
    For single Laplace(mu,b): derivative = - sign(x-mu)/b
    """
    pi_ = params['pi']
    p_x = mixture_laplace_pdf(x, params)

    p1 = laplace_pdf(x, params['mu1'], params['sigma1'])
    sign1 = np.sign(x - params['mu1'])
    d1 = pi_ * p1 * ( - sign1/(params['sigma1']+1e-15) )

    p2 = laplace_pdf(x, params['mu2'], params['sigma2'])
    sign2 = np.sign(x - params['mu2'])
    d2 = (1-pi_)* p2 * ( - sign2/(params['sigma2']+1e-15) )

    dp_dx = d1 + d2
    return dp_dx / (p_x + 1e-15)

def plot_laplace_score_functions_pub(mixture_params_list, x_min=-6, x_max=6):
    """
    Produces score_function_laplace.pdf with 3 subplots (one per mixture):
      - In each subplot:
         Left axis => Density (blue)
         Right axis => Score (red)
         Ticks on both sides in black
      - ONLY subplot #1 has 'Density' label (left)
      - ONLY subplot #3 has 'Score' label (right)
      - Subplot #2 => no y-axis labels
    A single legend: 'Density' (blue), 'Score' (red).
    Subplot titles are "Mixture 1 Laplace", "Mixture 2 Laplace", etc.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18,6))
    x_grid = np.linspace(x_min, x_max, 400)

    for i, params in enumerate(mixture_params_list):
        ax_left  = axes[i]                # left axis => density
        ax_right = ax_left.twinx()        # right axis => score

        # Make spines/ticks black on both axes
        for spine in ax_left.spines.values():
            spine.set_color('black')
        ax_left.tick_params(axis='x', colors='black')
        ax_left.tick_params(axis='y', colors='black')

        for spine in ax_right.spines.values():
            spine.set_color('black')
        ax_right.tick_params(axis='y', colors='black')

        # Evaluate data
        density_vals = mixture_laplace_pdf(x_grid, params)
        score_vals   = laplace_score_function(x_grid, params)

        # Plot
        ax_left.plot(x_grid, density_vals, color='blue')
        ax_right.plot(x_grid, score_vals,  color='red')

        # Title, x-label
        ax_left.set_title(f"Mixture {i+1} Laplace", color='black')
        ax_left.set_xlabel("x", color='black')

        # Subplot #1 => label "Density" on left
        # Subplot #3 => label "Score" on right
        if i == 0:
            ax_left.set_ylabel("Density", color='black')
        elif i == 2:
            ax_right.set_ylabel("Score", color='black')

    # Single legend at bottom
    import matplotlib.lines as mlines
    density_line = mlines.Line2D([], [], color='blue', label='Density')
    score_line   = mlines.Line2D([], [], color='red',  label='Score')
    fig.legend([density_line, score_line], ['Density', 'Score'],
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0,0.06,1,1])
    plt.savefig("figures/score_function_laplace.pdf", bbox_inches="tight")
    plt.show()

###############################################################################
# Example usage
###############################################################################
if __name__ == "__main__":
    # Gaussian version
    plot_gaussian_score_functions_pub(mixture_params_list)
    print("Saved 'score_function_gaussian.pdf' in figures/ folder.")

    # Laplace version
    plot_laplace_score_functions_pub(mixture_params_list)
    print("Saved 'score_function_laplace.pdf' in figures/ folder.")




