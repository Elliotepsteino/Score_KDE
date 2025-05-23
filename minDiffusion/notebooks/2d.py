# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## First, define 2D synthetic datasets

import numpy as np
import torch
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


def plot_dataset(dataset, n_samples=1000, plot_score=True):
    """Visualize dataset samples and score field."""
    samples = dataset.sample(n_samples)
    
    # Create grid for score visualization
    x = np.linspace(-4, 4, 20)
    y = np.linspace(-4, 4, 20)
    X, Y = np.meshgrid(x, y)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Compute scores on grid
    if plot_score:
        scores = dataset.score(grid_points)
        U = scores[:, 0].reshape(X.shape)
        V = scores[:, 1].reshape(X.shape)
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    # Plot samples
    plt.subplot(121)
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=10)
    plt.title('Samples')
    plt.axis('equal')
    
    if plot_score:
        # Plot score field
        plt.subplot(122)
        plt.quiver(X, Y, U, V, alpha=0.5)
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.2, s=10)
        plt.title('Score Field')
        plt.axis('equal')
    
    plt.tight_layout()
    plt.show()


# +
class MixtureOfGaussians:
    def __init__(self, n_components=3, seed=42):
        np.random.seed(seed)
        
        # Define mixture parameters
        self.n_components = n_components
        self.weights = np.ones(n_components) / n_components
        
        # Generate random means in a circle
        angles = np.linspace(0, 2*np.pi, n_components, endpoint=False)
        radius = 2.0
        self.means = radius * np.column_stack((np.cos(angles), np.sin(angles)))
        
        # Generate random covariance matrices
        self.covs = []
        for _ in range(n_components):
            A = np.random.randn(2, 2)
            self.covs.append(0.3 * A.T @ A)
        
        # Create scipy multivariate normal objects
        self.components = [
            multivariate_normal(mean, cov) 
            for mean, cov in zip(self.means, self.covs)
        ]

    def sample(self, n_samples):
        """Generate samples from the mixture."""
        samples = []
        for _ in range(n_samples):
            # Choose component
            k = np.random.choice(self.n_components, p=self.weights)
            # Generate sample
            sample = self.components[k].rvs()
            samples.append(sample)
        return np.array(samples)

    def log_density(self, x):
        """Compute log density at points x."""
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
            
        densities = np.zeros((x.shape[0], self.n_components))
        for k in range(self.n_components):
            densities[:, k] = self.weights[k] * self.components[k].pdf(x)
        
        return np.log(densities.sum(axis=1))

    def score(self, x):
        """Compute score (gradient of log density) at points x."""
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
            
        scores = np.zeros_like(x)
        densities = np.zeros((x.shape[0], self.n_components))
        
        # First compute component densities for normalization
        for k in range(self.n_components):
            densities[:, k] = self.weights[k] * self.components[k].pdf(x)
        total_density = densities.sum(axis=1)
        
        # Compute score for each component and combine
        for k in range(self.n_components):
            diff = x - self.means[k]
            component_score = -np.linalg.solve(self.covs[k], diff.T).T
            scores += (densities[:, k] / total_density)[:, np.newaxis] * component_score
            
        return scores
    
# Mixture of Gaussians
print("Generating Mixture of Gaussians...")
mog = MixtureOfGaussians(n_components=3)
plot_dataset(mog)

# +
import numpy as np
from scipy.stats import multivariate_normal

class TwoSpirals:
    def __init__(self, noise_scale=0.1, seed=42):
        self.noise_scale = noise_scale
        self.sigma2 = noise_scale**2
        np.random.seed(seed)

    def _spiral_xy(self, t):
        """Generate points along a spiral."""
        r = 2 * t
        angle = 3 * np.pi * t
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        return x, y

    def _get_spiral_points(self, n_points=1000):
        """Get points along both spirals."""
        t = np.linspace(0.2, 1, n_points)
        x1, y1 = self._spiral_xy(t)
        spiral1 = np.column_stack([x1, y1])
        spiral2 = -spiral1  # Second spiral is reflection of first
        return spiral1, spiral2

    def _get_closest_spiral_points(self, x_eval, n_points=1000):
        """Find closest points on both spirals for given evaluation points."""
        spiral1, spiral2 = self._get_spiral_points(n_points)
        
        if len(x_eval.shape) == 1:
            x_eval = x_eval.reshape(1, -1)
            
        closest_points = []
        min_distances = []
        
        for x in x_eval:
            # Compute distances to both spirals
            dist1 = np.sum((spiral1 - x)**2, axis=1)
            dist2 = np.sum((spiral2 - x)**2, axis=1)
            
            # Find closest points
            idx1 = np.argmin(dist1)
            idx2 = np.argmin(dist2)
            
            closest_points.append([spiral1[idx1], spiral2[idx2]])
            min_distances.append([dist1[idx1], dist2[idx2]])
            
        return np.array(closest_points), np.array(min_distances)

    def log_density(self, x):
        """Compute log density at points x using mixture of Gaussians along spirals."""
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
            
        closest_points, distances = self._get_closest_spiral_points(x)
        
        # Compute log density using minimum distance to either spiral
        min_distances = np.min(distances, axis=1)
        log_density = -0.5 * min_distances / self.sigma2 - np.log(2 * np.pi * self.sigma2)
        
        return log_density

    def score(self, x):
        """Compute score function at points x."""
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
            
        closest_points, distances = self._get_closest_spiral_points(x)
        
        weights = np.exp(-0.5 * distances / self.sigma2)
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        score = np.zeros_like(x)
        for i in range(len(x)):
            direction1 = (closest_points[i, 0] - x[i]) / self.sigma2
            direction2 = (closest_points[i, 1] - x[i]) / self.sigma2
            score[i] = weights[i, 0] * direction1 + weights[i, 1] * direction2
            
        return score

    def sample(self, n_samples):
        """Generate samples from the two spiral distribution."""
        n_per_spiral = n_samples // 2
        
        # Generate parameters along the spirals
        t = np.random.uniform(0.2, 1, size=n_per_spiral)
        
        # First spiral
        x1, y1 = self._spiral_xy(t)
        # Second spiral - rotate by pi
        x2, y2 = self._spiral_xy(t)
        x2 = -x2
        y2 = -y2
        
        # Add noise
        noise = np.random.normal(0, self.noise_scale, size=(2, n_per_spiral))
        x1 += noise[0]
        y1 += noise[1]
        x2 += noise[0]
        y2 += noise[1]
        
        # Combine spirals
        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])
        
        return np.column_stack([x, y])

# Create and plot the spirals
spirals = TwoSpirals(noise_scale=0.05)
plot_dataset(spirals)

# +
from sklearn.datasets import make_moons
from sklearn.neighbors import KernelDensity

class TwoMoons:
    def __init__(self, noise_scale=0.1, seed=42, n_ref=10000):
        self.noise_scale = noise_scale
        self.sigma2 = noise_scale**2
        self.seed = seed
        np.random.seed(seed)
        
        # Generate a large reference sample for KDE
        self.ref_samples = self.sample(n_ref)
        
        # Fit KDE on reference samples using Silverman's rule
        bandwidth = silverman_bandwidth_2d(self.ref_samples)
        self.kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        self.kde.fit(self.ref_samples)

    def log_density(self, x):
        """Compute log density at points x using sklearn's KDE."""
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        return self.kde.score_samples(x)

    def sample(self, n_samples):
        """Generate samples using sklearn's make_moons with normalization."""
        X, _ = make_moons(n_samples=n_samples, noise=self.noise_scale, random_state=self.seed)
        
        return X

    def _moon_points(self, t):
        """Generate points along the moon curves."""
        # First moon (upper)
        x1 = np.cos(t * np.pi)
        y1 = np.sin(t * np.pi) / 2

        # Second moon (lower)
        x2 = 1 - np.cos(t * np.pi)
        y2 = -np.sin(t * np.pi) / 2 - 0.5

        return np.column_stack([x1, y1]), np.column_stack([x2, y2])

    def _get_closest_moon_points(self, x_eval, n_points=1000):
        """Find closest points on both moons for given evaluation points."""
        t = np.linspace(0, 1, n_points)
        moon1_points, moon2_points = self._moon_points(t)
        
        closest_points = []
        min_distances = []
        
        for x in x_eval:
            # Compute distances to both moons
            dist1 = np.sum((moon1_points - x)**2, axis=1)
            dist2 = np.sum((moon2_points - x)**2, axis=1)
            
            # Find closest points on each moon
            idx1 = np.argmin(dist1)
            idx2 = np.argmin(dist2)
            
            closest_points.append([moon1_points[idx1], moon2_points[idx2]])
            min_distances.append([dist1[idx1], dist2[idx2]])
            
        return np.array(closest_points), np.array(min_distances)

    def score(self, x_eval):
        """
        Compute score (gradient of log density) at given points.
        The score is approximated based on distance to nearest moon points.
        """
        if len(x_eval.shape) == 1:
            x_eval = x_eval.reshape(1, -1)
            
        closest_points, distances = self._get_closest_moon_points(x_eval)
        
        # Compute weights for each moon based on distances
        weights = np.exp(-0.5 * distances / self.sigma2)
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        # Compute score as weighted sum of directions to closest points
        score = np.zeros_like(x_eval)
        for i in range(len(x_eval)):
            direction1 = (closest_points[i, 0] - x_eval[i]) / self.sigma2
            direction2 = (closest_points[i, 1] - x_eval[i]) / self.sigma2
            score[i] = weights[i, 0] * direction1 + weights[i, 1] * direction2
            
        return score

    def sample(self, n_samples):
        """Generate samples using sklearn's make_moons."""
        X, _ = make_moons(n_samples=n_samples, noise=self.noise_scale, random_state=42)
        return X
    
# Create and plot the spirals
spirals = TwoMoons(noise_scale=0.1)
plot_dataset(spirals)
# -

# ## Now, we train a diffusion model on each dataset.

# +
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128
epochs = 1500
lr = 1e-3
T = 1000  # Total timesteps
hidden_dim = 512
noise_scale = 1

class SyntheticDiffusion:
    def __init__(self, dataset_name, n_samples=10000, noise=0.05):
        self.dataset_name = dataset_name
        
        # Load the specified dataset
        if dataset_name == 'moons':
            dataset = TwoMoons(noise_scale=noise)
        elif dataset_name == 'gaussian':
            dataset = MixtureOfGaussians(n_components=3)
        elif dataset_name == 'spirals':
            dataset = TwoSpirals(noise_scale=noise)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        # Generate samples and store the dataset object for score computation
        self.X = dataset.sample(n_samples)
        self.dataset = dataset
        
        # Normalize the data
        self.data_mean = self.X.mean(axis=0)
        self.data_std = self.X.std(axis=0)
        self.X = (self.X - self.data_mean) / self.data_std
        
        # Convert to tensor and create dataloader
        self.X = torch.tensor(self.X, dtype=torch.float32).to(device)
        self.dataloader = DataLoader(TensorDataset(self.X), batch_size=batch_size, shuffle=True)
        
        # Setup diffusion parameters
        self.setup_diffusion()
        
        # Initialize model
        self.model = MLPDiffusion().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
    def setup_diffusion(self):
        self.betas = torch.linspace(1e-4, 0.02, T).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)
    
    def train(self):
        best_loss = float('inf')
        best_model = None
        
        tqdm_epochs = tqdm(range(epochs))
        for epoch in tqdm_epochs:
            for batch in self.dataloader:
                x = batch[0]
                batch_size = x.size(0)
                
                # Random timesteps
                t = torch.randint(0, T, (batch_size,), device=device).long()
                
                # Generate noise and noisy samples
                noise = noise_scale * torch.randn_like(x)
                sqrt_alpha_bar = self.sqrt_alpha_bars[t].unsqueeze(-1)
                sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t].unsqueeze(-1)
                x_noisy = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise
                
                # Predict and compute loss
                pred_noise = self.model(x_noisy, t)
                loss = nn.functional.mse_loss(pred_noise, noise)
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            tqdm_epochs.set_description(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")
            
            # Plot trajectories every 50 epochs
            if (epoch + 1) % 50 == 0:
                self.plot_trajectories(epoch)
            
            # Save best model
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model = self.model.state_dict()
        
        torch.save(best_model, f"best_model_{self.dataset_name}.pth")
    
    def plot_trajectories(self, epoch):
        n_traj = 5
        x_traj = torch.randn(n_traj, 2).to(device)
        trajectories = [x_traj.cpu().detach().numpy()]
        
        for t_step in reversed(range(T)):
            t_tensor = torch.full((n_traj,), t_step, device=device, dtype=torch.long)
            pred_noise = self.model(x_traj, t_tensor)
            alpha_t = self.alphas[t_step]
            alpha_bar_t = self.alpha_bars[t_step]
            beta_t = self.betas[t_step]
            
            if t_step > 0:
                z = torch.randn_like(x_traj)
            else:
                z = torch.zeros_like(x_traj)
            
            x_traj = (x_traj - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise) / torch.sqrt(alpha_t)
            x_traj = x_traj + torch.sqrt(beta_t) * z
            
            if t_step % 20 == 0:
                trajectories.append(x_traj.cpu().detach().numpy())
        
        plt.figure(figsize=(15, 5))
        
        # Plot original data
        plt.subplot(131)
        plt.scatter(self.X.cpu()[:, 0], self.X.cpu()[:, 1], alpha=0.5, s=1)
        plt.title("Original Data")
        
        # Plot trajectories
        plt.subplot(132)
        for i in range(n_traj):
            traj_points = np.vstack([state[i, :] for state in trajectories])
            gradient = np.linspace(1, 0, len(traj_points))
            plt.scatter(traj_points[:, 0], traj_points[:, 1],
                       c=gradient, cmap="gray", s=40)
            plt.plot(traj_points[:, 0], traj_points[:, 1],
                    color="gray", linewidth=1, alpha=0.5)
        plt.title(f"Diffusion Trajectories\nEpoch {epoch+1}")
        
        # Plot score field
        plt.subplot(133)
        x = np.linspace(-3, 3, 20)
        y = np.linspace(-3, 3, 20)
        X, Y = np.meshgrid(x, y)
        points = np.column_stack([X.ravel(), Y.ravel()])
        points = points * self.data_std + self.data_mean
        scores = self.dataset.score(points)
        U = scores[:, 0].reshape(X.shape)
        V = scores[:, 1].reshape(X.shape)
        plt.quiver(X, Y, U, V, alpha=0.5)
        plt.scatter(self.X.cpu()[:, 0], self.X.cpu()[:, 1], alpha=0.2, s=1)
        plt.title("Score Field")
        
        plt.tight_layout()
        plt.show()

    def sample(self, n_samples=1000):
        self.model.eval()
        with torch.no_grad():
            x = torch.randn(n_samples, 2).to(device)
            
            for t_step in reversed(range(T)):
                t = torch.full((n_samples,), t_step, device=device, dtype=torch.long)
                pred_noise = self.model(x, t)
                
                alpha_t = self.alphas[t_step]
                alpha_bar_t = self.alpha_bars[t_step]
                beta_t = self.betas[t_step]
                
                if t_step > 0:
                    z = torch.randn_like(x)
                else:
                    z = torch.zeros_like(x)
                    
                x = (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise) / torch.sqrt(alpha_t)
                x += torch.sqrt(beta_t) * z
            
            # Denormalize samples
            samples = x.cpu().numpy()
            samples = samples * self.data_std + self.data_mean
            return samples

class MLPDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = nn.Embedding(T, hidden_dim)
        self.input = nn.Linear(2, hidden_dim)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)
        self.activation = nn.SiLU()  # Changed to SiLU activation

    def forward(self, x, t):
        t_embed = self.time_embed(t)
        h = self.input(x)
        h += t_embed
        h = self.activation(h)
        h = self.hidden1(h)
        h = self.activation(h)
        h = self.hidden2(h)
        h = self.activation(h)
        return self.output(h)

# Training and evaluation
def train_all_datasets():
    datasets = ['moons', 'gaussian', 'spirals']
    for dataset_name in datasets:
        print(f"\nTraining on {dataset_name} dataset:")
        diffusion = SyntheticDiffusion(dataset_name)
        diffusion.train()
        
        # Generate and plot final samples
        samples = diffusion.sample(1000)
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.scatter(diffusion.X.cpu()[:, 0], diffusion.X.cpu()[:, 1],
                   alpha=0.5, label='Original Data')
        plt.legend()
        plt.title(f"{dataset_name} - Original Data")
        
        plt.subplot(122)
        plt.scatter(samples[:, 0], samples[:, 1],
                   alpha=0.5, label='Generated Samples')
        plt.legend()
        plt.title(f"{dataset_name} - Generated Samples")
        plt.show()

if __name__ == "__main__":
    train_all_datasets()
# -

# ## Now, let's look at our KDE method.

# +
import numpy as np
from scipy.stats import multivariate_normal

def silverman_bandwidth_2d(data):
    """
    Compute Silverman's rule of thumb bandwidth for 2D data.
    Returns a scalar bandwidth (not a matrix).
    """
    n, d = data.shape
    assert d == 2, "Data must be 2-dimensional"
    
    # Compute standard deviation for each dimension
    stds = np.std(data, axis=0)
    
    # Compute IQR for each dimension
    iqr_estimates = np.array([
        np.percentile(data[:, i], 75) - np.percentile(data[:, i], 25)
        for i in range(d)
    ]) / 1.34

    # print(f"stds: {stds}, iqr_estimates: {iqr_estimates}")
    
    # Take minimum of std and IQR-based estimate for each dimension
    sigmas = np.minimum(stds, iqr_estimates)
    
    # Use average sigma
    sigma = np.mean(sigmas)
    
    # Silverman's rule in 2D
    return sigma * (n ** (-1/6))

def one_step_debiased_data_2d(X, score_fn, k=0.65):
    """
    Perform one-step debiasing for 2D data.
    Returns debiased data and bandwidth.
    """
    n, d = X.shape
    assert d == 2, "Data must be 2-dimensional"
    
    # Compute standard deviation for each dimension
    stds = np.std(X, axis=0)
    
    # Compute IQR for each dimension
    iqr_estimates = np.array([
        np.percentile(X[:, i], 75) - np.percentile(X[:, i], 25)
        for i in range(d)
    ]) / 1.34
    
    # Take minimum of std and IQR-based estimate
    sigmas = np.minimum(stds, iqr_estimates)
    sigma = np.mean(sigmas)
    
    # Bandwidth for debiasing (analogous to 0.4 * sigma * n^(-1/9) in 1D)
    h = k * sigma * n**(-1/(d+7))
    
    # Compute delta (scalar)
    delta = (h**2) / 2.0
    
    # Compute score
    score = score_fn(X)
    
    # Update points (similar to 1D case)
    X_new = X + delta * score
    
    return X_new, h

def kde_pdf_eval_2d(x_points, data, bandwidth):
    """
    Evaluate 2D KDE at given points.
    x_points: array of shape (M, 2) where M is number of evaluation points
    data: array of shape (N, 2) where N is number of data points
    bandwidth: scalar bandwidth
    """
    M = len(x_points)
    N = len(data)
    d = 2  # dimension
    
    # Compute pairwise squared distances
    diff = x_points.reshape(M, 1, d) - data.reshape(1, N, d)
    squared_distances = np.sum(diff**2, axis=2)
    
    # Compute kernel values
    kernel = np.exp(-squared_distances / (2 * bandwidth**2))
    
    # Normalize and sum
    return np.sum(kernel, axis=1) / (N * (2 * np.pi * bandwidth**2)**(d/2))

def approximate_mise_2d(data, bandwidth, true_density_fn, x_range=(-4, 4), y_range=(-4, 4), n_grid=50):
    """
    Approximate MISE for 2D KDE
    """
    x = np.linspace(x_range[0], x_range[1], n_grid)
    y = np.linspace(y_range[0], y_range[1], n_grid)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Compute true and estimated densities
    true_density = true_density_fn(points)
    estimated_density = kde_pdf_eval_2d(points, data, bandwidth)
    
    # Compute MISE
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    return np.sum((true_density - estimated_density)**2) * dx * dy

def demo_kde_2d():
    # Generate synthetic 2D data (mixture of Gaussians)
    n_samples = 1000
    mix_weights = [0.5, 0.5]
    means = [np.array([-1.0, -1.0]), np.array([1.0, 1.0])]
    covs = [np.array([[0.5, 0.1], [0.1, 0.5]]), np.array([[0.5, -0.1], [-0.1, 0.5]])]
    
    # Generate data
    data = np.zeros((n_samples, 2))
    for i in range(n_samples):
        k = np.random.choice(2, p=mix_weights)
        data[i] = np.random.multivariate_normal(means[k], covs[k])
    
    # Define true density function
    def true_density(x):
        density = np.zeros(len(x))
        for w, mu, cov in zip(mix_weights, means, covs):
            density += w * multivariate_normal.pdf(x, mu, cov)
        return density
    
    # Define score function
    def score_fn(x):
        score = np.zeros_like(x)
        for w, mu, cov in zip(mix_weights, means, covs):
            pdf = multivariate_normal.pdf(x, mu, cov)
            diff = x - mu
            score_k = -np.linalg.solve(cov, diff.T).T
            score += w * pdf[:, None] * score_k
        normalizer = np.sum([w * multivariate_normal.pdf(x, mu, cov) 
                           for w, mu, cov in zip(mix_weights, means, covs)], axis=0)
        return score / normalizer[:, None]
    
    # Compute bandwidths and perform debiasing
    h_silverman = silverman_bandwidth_2d(data)
    X_debiased, h_debiased = one_step_debiased_data_2d(data, score_fn)
    
    # Plot results
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'savefig.dpi': 300,
        'font.family': 'serif',
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13
    })
    
    # Compute all densities first
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()])
    
    Z_true = true_density(points).reshape(100, 100)
    Z_silverman = kde_pdf_eval_2d(points, data, h_silverman).reshape(100, 100)
    Z_debiased = kde_pdf_eval_2d(points, X_debiased, h_debiased).reshape(100, 100)
    
    # Compute MISE
    mise_silverman = approximate_mise_2d(data, h_silverman, true_density)
    mise_debiased = approximate_mise_2d(X_debiased, h_debiased, true_density)
    
    # Create figure with more width between subplots
    fig = plt.figure(figsize=(15, 5))
    plt.subplots_adjust(wspace=0.3)
    
    # Plot original data density
    ax1 = plt.subplot(131)
    im1 = ax1.pcolormesh(X, Y, Z_true, shading='auto', cmap='viridis')
    # plt.colorbar(im1, ax=ax1, label='Density')
    ax1.set_title(f"True Density\n$n$={n_samples}", pad=15)
    ax1.set_aspect('equal')
    
    # Plot Silverman KDE
    ax2 = plt.subplot(132)
    im2 = ax2.pcolormesh(X, Y, Z_silverman, shading='auto', cmap='viridis')
    # plt.colorbar(im2, ax=ax2, label='Density')
    ax2.set_title(f"Silverman KDE\n$n$={n_samples}, $h$={h_silverman:.3f}\nMISE={mise_silverman:.6f}", pad=15)
    ax2.set_aspect('equal')
    
    # Plot debiased KDE
    ax3 = plt.subplot(133)
    im3 = ax3.pcolormesh(X, Y, Z_debiased, shading='auto', cmap='viridis')
    # plt.colorbar(im3, ax=ax3, label='Density')
    ax3.set_title(f"Score-Debiased KDE\n$n$={n_samples}, $h$={h_debiased:.3f}\nMISE={mise_debiased:.6f}", pad=15)
    ax3.set_aspect('equal')
    
    plt.show()

if __name__ == "__main__":
    for i in range(1):
        demo_kde_2d()
# -

# ## Now, use the diffusion model to debias the data for our KDE method.

# ### Compare true vs. estimated score fields.

# +
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters (same as training)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T = 1000
hidden_dim = 512

class MLPDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = nn.Embedding(T, hidden_dim)
        self.input = nn.Linear(2, hidden_dim)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)
        self.activation = nn.SiLU()

    def forward(self, x, t):
        t_embed = self.time_embed(t)
        h = self.input(x)
        h += t_embed
        h = self.activation(h)
        h = self.hidden1(h)
        h = self.activation(h)
        h = self.hidden2(h)
        h = self.activation(h)
        return self.output(h)

class DiffusionScoreEstimator:
    def __init__(self, model_path, data_mean=None, data_std=None):
        # Load model
        self.model = MLPDiffusion().to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Setup diffusion parameters
        self.setup_diffusion()
        
        # Store normalization parameters
        self.data_mean = data_mean if data_mean is not None else np.array([0., 0.])
        self.data_std = data_std if data_std is not None else np.array([1., 1.])

    def setup_diffusion(self):
        self.betas = torch.linspace(1e-4, 0.02, T).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)

    def normalize(self, x):
        """Normalize input data"""
        return (x - self.data_mean) / self.data_std

    def denormalize(self, x):
        """Denormalize data"""
        return x * self.data_std + self.data_mean

    def score_function(self, x):
        """
        Compute score function for given points x
        x: numpy array of shape (n_samples, 2)
        """
        self.model.eval()
        with torch.no_grad():
            # Normalize input
            x_norm = self.normalize(x)
            x_tensor = torch.tensor(x_norm, dtype=torch.float32).to(device)
            
            # For score estimation, we use small t (close to original data)
            t = torch.zeros(x_tensor.shape[0], dtype=torch.long).to(device)
            
            # Get predicted noise
            pred_noise = self.model(x_tensor, t)
            
            # Convert noise prediction to score
            alpha_t = self.alphas[0]
            score = -pred_noise / np.sqrt(1 - alpha_t.cpu().numpy())
            
            # Move to CPU and adjust score for normalization
            score = score.cpu().numpy() / self.data_std
            
            return score

def plot_score_comparison(points, true_score_fn, diff_score_estimator, title="Score Field Comparison"):
    """Plot true vs estimated score fields"""
    scores_true = true_score_fn(points)
    scores_diff = diff_score_estimator.score_function(points)
    
    plt.figure(figsize=(15, 5))
    
    # True score field
    plt.subplot(131)
    x = points[:, 0].reshape(20, 20)
    y = points[:, 1].reshape(20, 20)
    u = scores_true[:, 0].reshape(20, 20)
    v = scores_true[:, 1].reshape(20, 20)
    plt.quiver(x, y, u, v, alpha=0.5)
    plt.title("True Score Field")
    
    # Estimated score field
    plt.subplot(132)
    u_est = scores_diff[:, 0].reshape(20, 20)
    v_est = scores_diff[:, 1].reshape(20, 20)
    plt.quiver(x, y, u_est, v_est, alpha=0.5)
    plt.title("Diffusion Score Field")
    
    # Error magnitude
    plt.subplot(133)
    error_mag = np.sqrt(np.sum((scores_true - scores_diff)**2, axis=1)).reshape(20, 20)
    plt.pcolormesh(x, y, error_mag, shading='auto')
    plt.colorbar(label='Error Magnitude')
    plt.title("Score Error")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Load the models
model_paths = {
    'gaussian': '/scratch/Score_KDE/minDiffusion/notebooks/best_model_gaussian.pth',
    'moons': '/scratch/Score_KDE/minDiffusion/notebooks/best_model_moons.pth',
    'spirals': '/scratch/Score_KDE/minDiffusion/notebooks/best_model_spirals.pth'
}

# Create grid of points for evaluation
x = np.linspace(-3, 3, 20)
y = np.linspace(-3, 3, 20)
X, Y = np.meshgrid(x, y)
points = np.column_stack([X.ravel(), Y.ravel()])

# Load and evaluate each model
for dataset_name, model_path in model_paths.items():
    try:
        print(f"\nEvaluating {dataset_name} model...")
        
        # Create score estimator
        estimator = DiffusionScoreEstimator(model_path)
        
        # Get true score function (you'll need to import the appropriate synthetic dataset class)
        if dataset_name == 'gaussian':
            true_dataset = MixtureOfGaussians(n_components=3)
        elif dataset_name == 'moons':
            true_dataset = TwoMoons(noise_scale=0.1)
        elif dataset_name == 'spirals':
            true_dataset = TwoSpirals(noise_scale=0.1)
        
        # Plot comparison
        plot_score_comparison(points, true_dataset.score, estimator, 
                            f"Score Field Comparison - {dataset_name}")
        
    except FileNotFoundError:
        print(f"Model file not found for {dataset_name}")
    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")


# -

# ### KDE with true vs. estimated score fields.

# +
def approximate_mise_2d(data, bandwidth, log_density_fn, debiased_data=None, x_range=(-4, 4), y_range=(-4, 4), n_grid=50, debug=False):
    """
    Approximate MISE for 2D KDE with optional debugging.
    If debiased_data is provided, use it instead of original data for density estimation.
    """
    x = np.linspace(x_range[0], x_range[1], n_grid)
    y = np.linspace(y_range[0], y_range[1], n_grid)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()])
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # Get true density
    true_density = np.exp(log_density_fn(points))
    
    # Use debiased data if provided, otherwise use original data
    data_to_use = debiased_data if debiased_data is not None else data
    estimated_density = kde_pdf_eval_2d(points, data_to_use, bandwidth)
    
    if debug:
        print(f"\nDebugging MISE calculation:")
        print(f"Grid dx: {dx:.6f}, dy: {dy:.6f}")
        print(f"Using {'debiased' if debiased_data is not None else 'original'} data")
        print(f"\nTrue density stats:")
        print(f"  Min: {true_density.min():.6f}")
        print(f"  Max: {true_density.max():.6f}")
        print(f"  Mean: {true_density.mean():.6f}")
        print(f"\nEstimated density stats:")
        print(f"  Min: {estimated_density.min():.6f}")
        print(f"  Max: {estimated_density.max():.6f}")
        print(f"  Mean: {estimated_density.mean():.6f}")
    
    # Normalize densities to integrate to 1
    true_integral = np.sum(true_density) * dx * dy
    est_integral = np.sum(estimated_density) * dx * dy
    
    if debug:
        print(f"\nIntegrals before normalization:")
        print(f"  True density integral: {true_integral:.6f}")
        print(f"  Estimated density integral: {est_integral:.6f}")
    
    true_density = true_density / true_integral
    estimated_density = estimated_density / est_integral
    
    # Compute squared differences
    squared_diff = (true_density - estimated_density)**2
    
    if debug:
        print(f"\nSquared differences:")
        print(f"  Min: {squared_diff.min():.6f}")
        print(f"  Max: {squared_diff.max():.6f}")
        print(f"  Mean: {squared_diff.mean():.6f}")
    
    # Compute MISE
    mise = np.mean(squared_diff)
    
    if debug:
        print(f"\nFinal MISE: {mise:.6f}")
    
    return mise

class KDEComparison:
    def __init__(self, true_dataset, diffusion_model_path, n_samples=1000):
        self.true_dataset = true_dataset
        self.n_samples = n_samples
        
        # Load diffusion model
        self.diff_estimator = DiffusionScoreEstimator(diffusion_model_path)
        
        # Generate samples from true distribution
        self.samples = true_dataset.sample(n_samples)

        # Divide bandwidth by a constant factor
        self.k_factor = 2 # for spiral
    
    def perform_kde(self, x_eval, use_true_score=True):
        """
        Perform one-step debiased KDE using either true or diffusion score
        """
        # Get score function
        if use_true_score:
            score_fn = self.true_dataset.score
        else:
            score_fn = self.diff_estimator.score_function

        
        # Get Silverman's bandwidth for original data
        h_silverman = silverman_bandwidth_2d(self.samples) / self.k_factor
        
        if use_true_score:
            # Perform one-step debiasing with true score
            X_debiased, h_debiased = one_step_debiased_data_2d(self.samples, score_fn, k=0.55/self.k_factor)
        else:
            # Perform one-step debiasing with diffusion score
            X_debiased, h_debiased = one_step_debiased_data_2d(self.samples, score_fn, k=0.55/self.k_factor)
        
        # Evaluate KDE with original samples (Silverman)
        density_silverman = kde_pdf_eval_2d(x_eval, self.samples, h_silverman)
        
        # Evaluate KDE with debiased samples
        density_debiased = kde_pdf_eval_2d(x_eval, X_debiased, h_debiased)
        
        return density_silverman, density_debiased, h_silverman, h_debiased
    
    def evaluate_densities(self, grid_points=50):
        """
        Evaluate and compare densities on a grid
        """
        # Create evaluation grid
        x = np.linspace(-3, 3, grid_points)
        y = np.linspace(-3, 3, grid_points)
        X, Y = np.meshgrid(x, y)
        xy_eval = np.column_stack([X.ravel(), Y.ravel()])
        
        # Compute true density
        true_density = np.exp([self.true_dataset.log_density(p) for p in xy_eval])
        
        # Compute densities with true score
        silverman_true, debiased_true, h_silv_true, h_deb_true = self.perform_kde(xy_eval, use_true_score=True)
        # Get debiased points for true score
        X_debiased_true, _ = one_step_debiased_data_2d(self.samples, self.true_dataset.score, k=0.55/self.k_factor)
        
        mise_silv_true = approximate_mise_2d(self.samples, h_silv_true, self.true_dataset.log_density)
        mise_deb_true = approximate_mise_2d(self.samples, h_deb_true, self.true_dataset.log_density, debiased_data=X_debiased_true)
        
        # Compute densities with diffusion score
        silverman_diff, debiased_diff, h_silv_diff, h_deb_diff = self.perform_kde(xy_eval, use_true_score=False)
        # Get debiased points for diffusion score
        X_debiased_diff, _ = one_step_debiased_data_2d(self.samples, self.diff_estimator.score_function, k=0.55/self.k_factor)
        
        mise_deb_diff = approximate_mise_2d(self.samples, h_deb_diff, self.true_dataset.log_density, debiased_data=X_debiased_diff)
        
        # Plot results with improved visualization
        plt.rcParams.update({
            'savefig.dpi': 300,
            'font.family': 'serif',
            'font.size': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 16,
            'xtick.labelsize': 13,
            'ytick.labelsize': 13
        })
        
        fig = plt.figure(figsize=(20, 5))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        
        # True density
        ax1 = plt.subplot(1, 4, 1)
        im1 = ax1.pcolormesh(X, Y, true_density.reshape(grid_points, grid_points), 
                           shading='auto', cmap='viridis')
        # plt.colorbar(im1, ax=ax1, label='Density')
        ax1.set_title(f"True Density\n$n$={self.n_samples}", pad=15)
        ax1.set_aspect('equal')
        
        # Silverman
        ax2 = plt.subplot(1, 4, 2)
        im2 = ax2.pcolormesh(X, Y, silverman_true.reshape(grid_points, grid_points), 
                           shading='auto', cmap='viridis')
        # plt.colorbar(im2, ax=ax2, label='Density')
        ax2.set_title(f"Silverman\n$n$={self.n_samples}, $h$={h_silv_true:.3f}\nMISE={mise_silv_true:.6f}", pad=15)
        ax2.set_aspect('equal')
        
        # Score-Debiased (True)
        ax3 = plt.subplot(1, 4, 3)
        im3 = ax3.pcolormesh(X, Y, debiased_true.reshape(grid_points, grid_points), 
                           shading='auto', cmap='viridis')
        # plt.colorbar(im3, ax=ax3, label='Density')
        ax3.set_title(f"SD-KDE (True)\n$n$={self.n_samples}, $h$={h_deb_true:.3f}\nMISE={mise_deb_true:.6f}", pad=15)
        ax3.set_aspect('equal')
        
        # Score-Debiased (Diffusion)
        ax4 = plt.subplot(1, 4, 4)
        im4 = ax4.pcolormesh(X, Y, debiased_diff.reshape(grid_points, grid_points), 
                           shading='auto', cmap='viridis')
        # plt.colorbar(im4, ax=ax4, label='Density')
        ax4.set_title(f"SD-KDE (Diffusion)\n$n$={self.n_samples}, $h$={h_deb_diff:.3f}\nMISE={mise_deb_diff:.6f}", pad=15)
        ax4.set_aspect('equal')
        
        plt.show()

# Example usage
if __name__ == "__main__":
    # Paths to models
    model_paths = {
        'gaussian': '/scratch/Score_KDE/minDiffusion/notebooks/best_model_gaussian.pth',
        'moons': '/scratch/Score_KDE/minDiffusion/notebooks/best_model_moons.pth',
        'spirals': '/scratch/Score_KDE/minDiffusion/notebooks/best_model_spirals.pth'
    }

    # Create and evaluate for each dataset
    datasets = {
        # 'gaussian': lambda seed : MixtureOfGaussians(n_components=3, seed=0),
        # 'moons': lambda seed : TwoMoons(noise_scale=0.05, seed=seed),
        'spirals': lambda seed : TwoSpirals(noise_scale=0.1, seed=seed)
    }

    for name, dataset_fn in datasets.items():
        print(f"\nEvaluating {name} dataset:")
        for i in range(1):
            try:
                dataset = dataset_fn(i)
                kde_comp = KDEComparison(dataset, model_paths[name], n_samples=1000)
                kde_comp.evaluate_densities()
            except Exception as e:
                print(f"Error processing {name}: {str(e)}")
# -

# ## OLD CODE
#

# +
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128
epochs = 500
lr = 1e-3
T = 1000  # Total timesteps
hidden_dim = 512
noise_scale = 1

# Generate moons dataset
X, _ = make_moons(n_samples=10000, noise=0.05)
# Normalize the data
X = (X - X.mean(axis=0)) / X.std(axis=0)
X = torch.tensor(X, dtype=torch.float32).to(device)
dataset = TensorDataset(X)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Noise schedule
betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
sqrt_alpha_bars = torch.sqrt(alpha_bars)
sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)

# Diffusion model architecture
class MLPDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = nn.Embedding(T, hidden_dim)
        self.input = nn.Linear(2, hidden_dim)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)
        self.activation = nn.ReLU()

    def forward(self, x, t):
        t_embed = self.time_embed(t)
        h = self.input(x)
        h += t_embed
        h = self.activation(h)
        h = self.hidden1(h)
        h = self.activation(h)
        h = self.hidden2(h)
        h = self.activation(h)
        return self.output(h)

model = MLPDiffusion().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Save best model
best_loss = float('inf')
best_model = None

# Training loop
tqdm_epochs = tqdm(range(epochs))
for epoch in tqdm_epochs:
    for batch in dataloader:
        x = batch[0]
        batch_size = x.size(0)
        
        # Random timesteps for each sample
        t = torch.randint(0, T, (batch_size,), device=device).long()
        
        # Generate noise and noisy samples
        noise = noise_scale*torch.randn_like(x)
        sqrt_alpha_bar = sqrt_alpha_bars[t].unsqueeze(-1)
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bars[t].unsqueeze(-1)
        x_noisy = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise
        
        # Predict and compute loss
        pred_noise = model(x_noisy, t)
        loss = nn.functional.mse_loss(pred_noise, noise)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Update the tqdm progress bar
    tqdm_epochs.set_description(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")
    
    # --- New Code for Plotting Diffusion Trajectories ---
    # Every 50 epochs, sample a few diffusion trajectories using the current model state.
    if (epoch + 1) % 50 == 0:
        # Import and clear output for updated display in notebook
        from IPython.display import clear_output
        clear_output(wait=True)
        tqdm_epochs.set_description(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")

        # Plot the diffusion trajectories
        n_traj = 5  # Number of trajectories to track
        x_traj = torch.randn(n_traj, 2).to(device)  # Start with a random noise sample for each trajectory
        trajectories = [x_traj.cpu().detach().numpy()]  # store the initial state
        
        # Run the reverse diffusion process and record states every 200 timesteps
        for t_step in reversed(range(T)):
            t_tensor = torch.full((n_traj,), t_step, device=device, dtype=torch.long)
            pred_noise = model(x_traj, t_tensor)
            alpha_t = alphas[t_step]
            alpha_bar_t = alpha_bars[t_step]
            beta_t = betas[t_step]
            
            if t_step > 0:
                z = torch.randn_like(x_traj)
            else:
                z = torch.zeros_like(x_traj)
            
            x_traj = (x_traj - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise) / torch.sqrt(alpha_t)
            x_traj = x_traj + torch.sqrt(beta_t) * z
            
            # Store the state every 20 reverse steps (adjust as needed)
            if t_step % 20 == 0:
                trajectories.append(x_traj.cpu().detach().numpy())
        
        # # Plot each trajectory
        # plt.figure(figsize=(8, 6))
        # for i in range(n_traj):
        #     # Stack the saved states to form a continuous trajectory for sample i
        #     traj_points = np.vstack([state[i, :] for state in trajectories])
        #     plt.plot(traj_points[:, 0], traj_points[:, 1], marker='o', label=f"Trajectory {i+1}")
        # plt.title(f"Diffusion Trajectories at Epoch {epoch+1}")
        # plt.legend()
        # plt.show()
        # Plot each trajectory with gradient shading
        plt.figure(figsize=(8, 6))
        for i in range(n_traj):
            # Stack the saved states to form a continuous trajectory for sample i
            traj_points = np.vstack([state[i, :] for state in trajectories])
            # Create a gradient that goes from light (1, pure noise) to dark (0, final sample)
            gradient = np.linspace(1, 0, len(traj_points))
            # Plot the points with changing colors using a grayscale colormap
            plt.scatter(traj_points[:, 0],
                        traj_points[:, 1],
                        c=gradient, cmap="gray",
                        s=40, label=f"Trajectory {i+1}")
            # Optionally, connect the points with a line for clarity
            plt.plot(traj_points[:, 0], traj_points[:, 1], color="gray", linewidth=1, alpha=0.5)
        plt.title(f"Diffusion Trajectories at Epoch {epoch+1}")
        # plt.legend()
        plt.show()

    # Save the best model
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_model = model.state_dict()

# Save the best model
torch.save(best_model, f"best_model.pth")

# Sampling function
def sample(model, n_samples=1000):
    model.eval()
    with torch.no_grad():
        x = torch.randn(n_samples, 2).to(device)
        for t_step in reversed(range(T)):
            t = torch.full((n_samples,), t_step, device=device, dtype=torch.long)
            pred_noise = model(x, t)
            
            alpha_t = alphas[t_step]
            alpha_bar_t = alpha_bars[t_step]
            beta_t = betas[t_step]
            
            if t_step > 0:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)
                
            x = (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise) / torch.sqrt(alpha_t)
            x += torch.sqrt(beta_t) * z
            
    return x.cpu().numpy()

# Generate and plot samples
samples = sample(model, 1000)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), alpha=0.5, label='Original Data')
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, label='Generated Samples')
plt.legend()
plt.title("Diffusion Model Results")
plt.show()

# +
# Generate and plot samples with a heatmap for the samples using seaborn
samples = sample(model, 10000)

import seaborn as sns

plt.figure(figsize=(8, 6))

# Plot the generated samples as a density heatmap using seaborn
# Adjust bandwidth and kernel parameters for better density estimation
sns.kdeplot(
    x=samples[:, 0], 
    y=samples[:, 1], 
    fill=True, 
    cmap="viridis",
    alpha=0.7,
    levels=20,  # Reduced number of levels
    bw_adjust=0.3,  # Reduce bandwidth for sharper distribution
    thresh=0,  # Remove threshold to show full distribution
)

# Plot the true data distribution on top for reference
# plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), alpha=0.5, color='red', s=10, label='Original Data')

plt.title("Diffusion Model Results: Samples Density Heatmap")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')  # Make the plot aspect ratio 1:1
plt.show()

# Plot a heatmap for the true data distribution
plt.figure(figsize=(8, 6))

sns.kdeplot(
    x=X[:, 0].cpu(), 
    y=X[:, 1].cpu(), 
    fill=True, 
    cmap="viridis",
    alpha=0.7,
    levels=20,  # Reduced number of levels
    bw_adjust=0.3,  # Reduce bandwidth for sharper distribution
    thresh=0,  # Remove threshold to show full distribution
)

plt.title("Diffusion Model Results: True Data Density Heatmap")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')  # Make the plot aspect ratio 1:1
plt.show()


# +
def get_score(model, x, t=0):
    """
    Convert noise prediction to score.
    Args:
        model: The trained diffusion model
        x: Input data points where we want to evaluate score
        t: Timestep
    Returns:
        score: The score (∇x log p(x)) at points x
    """
    # Get the predicted noise
    eps_theta = model(x, t)
    
    # Convert noise prediction to score using the formula:
    # score = -eps_theta / sqrt_one_minus_alpha_bars[t]
    score = -eps_theta / sqrt_one_minus_alpha_bars[t].view(-1, 1)
    
    return score

# Create a grid of points
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
xx, yy = np.meshgrid(x, y)
grid_points = torch.tensor(np.column_stack([xx.ravel(), yy.ravel()]), dtype=torch.float32).to(device)

# Choose a timestep
for t in range(0, T, 999):
    t_tensor = torch.ones(grid_points.shape[0], dtype=torch.long).to(device) * t

    # Get scores
    scores = get_score(model, grid_points, t_tensor)
    scores = scores.cpu().detach().numpy()

    # Plot the score field
    plt.figure(figsize=(10, 10))
    plt.quiver(xx, yy, scores[:, 0].reshape(50, 50), scores[:, 1].reshape(50, 50))
    plt.title(f'Score Field (∇x log p(x)) at t={t}')
    plt.show()
# -


