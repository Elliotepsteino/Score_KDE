import torch
import math
import matplotlib.pyplot as plt

def run_score_informed_kde_experiment(
    mix_weights,
    means,
    covs,
    N=2000,
    N_eval=2000,
    score_noise_std=0.05,
    base_h=0.4,
    fixed_bandwidths=None,
    device='cpu',
    do_plot=False,
    use_silverman=False,
    compare_silverman_fixed=False,
    lambda_mix=0.9
):
    """
    Runs an experiment comparing:
      (1) A set of user-provided fixed bandwidths,
      (2) [OPTIONAL] A Silverman-based fixed bandwidth (compare_silverman_fixed=True),
      (3) A *convex combination* of Silverman + score-based local bandwidth,
          controlled by lambda_mix.

    on a 2D mixture-of-Gaussians distribution.

    Arguments:
      mix_weights (Tensor): shape (K,) for K mixture components
      means (Tensor): shape (K,2)
      covs (Tensor): shape (K,2,2)
      N (int): number of samples for KDE "training"
      N_eval (int): number of samples for Monte Carlo KL evaluation
      score_noise_std (float): std dev of noise added to score queries
      base_h (float): user-provided base bandwidth factor (used only if use_silverman=False
                      in the local method)
      fixed_bandwidths (list): set of fixed bandwidths to compare
      device (str): 'cpu' or 'cuda'
      do_plot (bool): if True, produce a 2D contour plot
      use_silverman (bool): if True, compute base_h from data using Silverman's rule (2D)
                            for the local method's "score-based" portion
      compare_silverman_fixed (bool): if True, also compute a single global Silverman
                                      bandwidth for a standard KDE
      lambda_mix (float): convex combination factor [0,1].
          h_i = lambda_mix*h_silverman + (1-lambda_mix)*(C / ||score_i||).

    Returns:
      A dict with:
        "KL_fixed": {bw: KL_value, ...} for each user fixed bandwidth
        "KL_score_informed": the KL for the "convex combination" local method
        "KL_silverman_fixed": if compare_silverman_fixed=True, the KL for the single
                              Silverman-based global bandwidth
        "C": local bandwidth scaling factor used in the score-based method
        "base_h": the final base_h used for that method
        "silverman_h": the Silverman bandwidth computed from the data
        "score_informed_bw_stats": a dict with "mean", "min", "max" of the local bandwidths
    """

    if fixed_bandwidths is None:
        fixed_bandwidths = [0.1, 0.2, 0.5, 1.0]

    # 1) Define log p(x) and the sampling / scoring functions
    def log_gaussian(x, mean, cov):
        d = x.shape[-1]
        cov_inv = torch.inverse(cov)
        cov_det = torch.det(cov)
        diff = x - mean
        exponent = -0.5 * (diff.unsqueeze(0) @ cov_inv @ diff.unsqueeze(1)).squeeze()
        normalizer = -0.5 * (d * math.log(2.0 * math.pi) + torch.log(cov_det))
        return normalizer + exponent

    def log_mixture_pdf(x):
        log_comps = []
        for k in range(len(mix_weights)):
            log_comps.append(log_gaussian(x, means[k], covs[k]))
        log_comps = torch.stack(log_comps)
        log_ws = torch.log(mix_weights)
        return torch.logsumexp(log_ws + log_comps, dim=0)

    def score_function(x):
        x_var = x.clone().detach().requires_grad_(True)
        lp = log_mixture_pdf(x_var)
        lp.backward()
        return x_var.grad

    def sample_mixture(n_samples):
        with torch.no_grad():
            comp_indices = torch.multinomial(mix_weights, n_samples, replacement=True)
            chosen_means = means[comp_indices]
            chosen_covs = covs[comp_indices]
            chol = torch.linalg.cholesky(chosen_covs)
            eps = torch.randn(n_samples, 2, device=device)
            samples = chosen_means + torch.einsum('nij,ni->nj', chol, eps)
        return samples

    # 2) KDE definitions
    def gaussian_kernel(x, center, h):
        h = max(h, 1e-8)
        diff = x - center
        norm_sq = diff.dot(diff)
        return torch.exp(-0.5 * norm_sq / (h**2)) / (2.0 * math.pi * (h**2))

    def kde_estimate_density(x, data, bandwidth):
        vals = [gaussian_kernel(x, data[i], bandwidth) for i in range(data.shape[0])]
        return torch.mean(torch.stack(vals))

    def log_kde_density(x, data, bw):
        val = kde_estimate_density(x, data, bw)
        return torch.log(torch.clamp(val, min=1e-12))

    # 3) Generate "training" data & query scores
    data = sample_mixture(N).to(device)

    all_scores = []
    for i in range(data.shape[0]):
        s_i = score_function(data[i])
        s_i_noisy = s_i + score_noise_std * torch.randn_like(s_i)
        all_scores.append(s_i_noisy.detach())
    all_scores = torch.stack(all_scores, dim=0).to(device)

    # Compute data std for Silverman
    data_std = torch.std(data, dim=0).mean()  # average across x & y
    silverman_h = data_std * (N ** (-1.0 / 6.0))

    # 4) If use_silverman == True, override base_h for local method
    if use_silverman:
        base_h = silverman_h

    # Now define local bandwidth scale: C = median(||score_i||) * base_h
    score_norms = all_scores.norm(p=2, dim=1)
    med_score_norm = torch.median(score_norms)
    C = med_score_norm * base_h

    # 5) Score-informed KDE with a convex combination of silverman_h + local
    #    h_i = lambda_mix * silverman_h + (1-lambda_mix)*(C/(||score_i|| + eps))
    def score_informed_kde(x, data, scores, C, silverman_h, eps=1e-4, alpha=0.9):
        vals = []
        for i in range(data.shape[0]):
            norm_si = scores[i].norm(p=2)
            h_score = C / (norm_si + eps)
            # convex combination:
            h_i = alpha * silverman_h + (1.0 - alpha) * h_score
            vals.append(gaussian_kernel(x, data[i], h_i))
        return torch.mean(torch.stack(vals))

    def log_score_informed_density(x, data, scores, C):
        val = score_informed_kde(
            x, data, scores, C, silverman_h=silverman_h, eps=1e-4, alpha=lambda_mix
        )
        return torch.log(torch.clamp(val, min=1e-12))

    # We'll also compute the distribution of bandwidths h_i across the training data
    all_bandwidths = []
    for i in range(data.shape[0]):
        s_i = all_scores[i]
        norm_si = s_i.norm(p=2)
        h_score = C / (norm_si + 1e-4)
        h_i = lambda_mix * silverman_h + (1.0 - lambda_mix) * h_score
        all_bandwidths.append(h_i.item())
    all_bandwidths = torch.tensor(all_bandwidths, dtype=torch.float32)
    bw_mean = all_bandwidths.mean().item()
    bw_min = all_bandwidths.min().item()
    bw_max = all_bandwidths.max().item()

    # 6) Evaluate KL: KL(p||p_hat)
    eval_points = sample_mixture(N_eval).to(device)

    def log_true_p(x):
        return log_mixture_pdf(x)

    def estimate_KL_fixed_bw(bw):
        lphat_vals = []
        lp_vals = []
        for x_ in eval_points:
            lphat_vals.append(log_kde_density(x_, data, bw))
            lp_vals.append(log_true_p(x_))
        return torch.mean(torch.stack(lp_vals) - torch.stack(lphat_vals))

    def estimate_KL_score_informed():
        lphat_vals = []
        lp_vals = []
        for x_ in eval_points:
            lphat_vals.append(log_score_informed_density(x_, data, all_scores, C))
            lp_vals.append(log_true_p(x_))
        return torch.mean(torch.stack(lp_vals) - torch.stack(lphat_vals))

    KL_fixed = {}
    for bw in fixed_bandwidths:
        KL_fixed[bw] = estimate_KL_fixed_bw(bw).item()

    KL_silverman_fixed = None
    if compare_silverman_fixed:
        KL_silverman_fixed = estimate_KL_fixed_bw(silverman_h).item()

    KL_score_informed = estimate_KL_score_informed().item()

    
    if do_plot:
        num_grid = 80
        grid_x = torch.linspace(-3, 7, num_grid)
        xx, yy = torch.meshgrid(grid_x, grid_x, indexing='xy')
        points_2d = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

        # True density
        lp_grid = []
        for pt in points_2d:
            lp_grid.append(log_true_p(pt))
        lp_grid = torch.stack(lp_grid).reshape(num_grid, num_grid)
        p_grid = torch.exp(lp_grid)

        # Score-informed
        lphat_grid = []
        for pt in points_2d:
            lphat_grid.append(log_score_informed_density(pt, data, all_scores, C))
        lphat_grid = torch.stack(lphat_grid).reshape(num_grid, num_grid)
        phat_grid = torch.exp(lphat_grid)

        silverman_vals = []
        for pt in points_2d:
            val = kde_estimate_density(pt, data, silverman_h)
            silverman_vals.append(torch.log(torch.clamp(val, min=1e-12)))
        silverman_vals = torch.stack(silverman_vals).reshape(num_grid, num_grid)
        silverman_grid = torch.exp(silverman_vals)

        # Plot all 3
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        cs1 = axs[0].contourf(xx.numpy(), yy.numpy(), p_grid.numpy(), levels=30)
        axs[0].scatter(data[:, 0].cpu().numpy(), data[:, 1].cpu().numpy(),
                       s=5, c='black', alpha=0.4)
        axs[0].set_title("True Density p(x)")
        plt.colorbar(cs1, ax=axs[0])

        cs2 = axs[1].contourf(xx.numpy(), yy.numpy(), silverman_grid.detach().numpy(), levels=30)
        axs[1].scatter(data[:, 0].cpu().numpy(), data[:, 1].cpu().numpy(),
                       s=5, c='black', alpha=0.4)
        axs[1].set_title(f"Silverman KDE (h={silverman_h:.3f}, KL={KL_silverman_fixed:.4f})")
        plt.colorbar(cs2, ax=axs[1])

        cs3 = axs[2].contourf(xx.numpy(), yy.numpy(), phat_grid.detach().numpy(), levels=30)
        axs[2].scatter(data[:, 0].cpu().numpy(), data[:, 1].cpu().numpy(),
                       s=5, c='black', alpha=0.4)
        axs[2].set_title(f"Score-Informed KDE (lambda={lambda_mix}, KL={KL_score_informed:.4f})")
        plt.colorbar(cs3, ax=axs[2])

        plt.tight_layout()
        plt.savefig("score_informed_kde.png")
        plt.show()

    # 8) Return results
    out = {
        "KL_fixed": KL_fixed,
        "KL_score_informed": KL_score_informed,
        "KL_silverman_fixed": KL_silverman_fixed,
        "C": C,
        "base_h": base_h,
        "silverman_h": silverman_h,
        "score_informed_bw_stats": {
            "mean": bw_mean,
            "min": bw_min,
            "max": bw_max
        }
    }
    return out


def main():
    # -- Initialize mixture parameters and experiment settings here --
    mix_weights = torch.tensor([0.4, 0.6], dtype=torch.float32)
    means = torch.tensor([
        [0.0, 0.0],
        [3.0, 3.0]
    ], dtype=torch.float32)
    covs = torch.stack([
        torch.tensor([[1.0, 0.0],
                      [0.0, 1.0]], dtype=torch.float32),
        torch.tensor([[1.5, 0.3],
                      [0.3, 1.0]], dtype=torch.float32),
    ], dim=0)

    device = "cpu"

    # Customize experiment hyperparameters
    N = 100
    N_eval = 1000
    score_noise_std = 0.05
    user_fixed_bws = [0.1, 0.2, 0.5, 1.0]

    results = run_score_informed_kde_experiment(
        mix_weights,
        means,
        covs,
        N=N,
        N_eval=N_eval,
        score_noise_std=score_noise_std,
        base_h=0.4,                  
        fixed_bandwidths=user_fixed_bws,
        device=device,
        do_plot=True,               # <--- set to True so we see the plots
        use_silverman=True,        
        compare_silverman_fixed=True,
        lambda_mix=0.9
    )

    # Print results
    print("KL divergences (KL(p||p_hat)) for user fixed bandwidths:")
    for bw, val in results["KL_fixed"].items():
        print(f"  Fixed h={bw}: {val:.4f}")

    if results["KL_silverman_fixed"] is not None:
        print(f"  Silverman-based fixed h={results['silverman_h']:.4f}: {results['KL_silverman_fixed']:.4f}")

    print(f"Score-Informed KDE (lambda={0.9}): KL = {results['KL_score_informed']:.4f}")
    print("  base_h used (local method) =", results["base_h"])
    print("  C =", results["C"])
    print("  silverman_h =", results["silverman_h"])
    bw_stats = results["score_informed_bw_stats"]
    print(f"  Local bandwidth stats: mean={bw_stats['mean']:.4f},"
          f" min={bw_stats['min']:.4f}, max={bw_stats['max']:.4f}")

if __name__ == "__main__":
    main()
