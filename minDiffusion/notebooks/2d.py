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

class TwoSpirals:
    def __init__(self, noise_scale=0.1, seed=42):
        np.random.seed(seed)
        self.noise_scale = noise_scale

    def _spiral_xy(self, t, spiral_type=1):
        """Generate points along a spiral."""
        r = t
        angle = spiral_type * (4 * t + np.pi)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        return x, y

    def sample(self, n_samples):
        """Generate samples from the spiral distribution."""
        n_per_spiral = n_samples // 2
        
        # Generate parameters along the spirals
        t = np.random.uniform(0, 1, size=n_per_spiral)
        
        # Generate points for both spirals
        x1, y1 = self._spiral_xy(t, 1)
        x2, y2 = self._spiral_xy(t, -1)
        
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

    def log_density(self, x, grid_size=100):
        """
        Approximate log density using KDE.
        Note: This is an approximation as the true density is not analytically tractable.
        """
        from scipy.stats import gaussian_kde
        # Generate large number of samples for KDE
        samples = self.sample(10000)
        kde = gaussian_kde(samples.T)
        return np.log(kde(x.T))

    def score(self, x, epsilon=1e-6):
        """
        Approximate score function using finite differences on log density.
        Note: This is a numerical approximation.
        """
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
            
        scores = np.zeros_like(x)
        for i in range(x.shape[1]):
            x_plus = x.copy()
            x_plus[:, i] += epsilon
            x_minus = x.copy()
            x_minus[:, i] -= epsilon
            
            scores[:, i] = (self.log_density(x_plus) - self.log_density(x_minus)) / (2 * epsilon)
        
        return scores



# +
class TwoSpirals:
    def __init__(self, noise_scale=0.1, seed=42):
        np.random.seed(seed)
        self.noise_scale = noise_scale
        self.sigma2 = noise_scale**2  # Variance for density estimation

    def _spiral_xy(self, t):
        """Generate points along a spiral."""
        r = 2 * t
        angle = 3 * np.pi * t
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        return x, y

    def _get_closest_spiral_points(self, x_eval, n_points=1000):
        """Find closest points on both spirals for given evaluation points."""
        t = np.linspace(0.2, 1, n_points)
        
        # Get points on both spirals
        x1, y1 = self._spiral_xy(t)
        spiral1_points = np.column_stack([x1, y1])
        spiral2_points = -spiral1_points  # Second spiral is reflection of first
        
        # For each evaluation point, find closest point on each spiral
        closest_points = []
        min_distances = []
        
        for x in x_eval:
            # Compute distances to both spirals
            dist1 = np.sum((spiral1_points - x)**2, axis=1)
            dist2 = np.sum((spiral2_points - x)**2, axis=1)
            
            # Find closest points on each spiral
            idx1 = np.argmin(dist1)
            idx2 = np.argmin(dist2)
            
            closest_points.append([spiral1_points[idx1], spiral2_points[idx2]])
            min_distances.append([dist1[idx1], dist2[idx2]])
            
        return np.array(closest_points), np.array(min_distances)

    def score(self, x_eval):
        """
        Compute score (gradient of log density) at given points.
        The score is approximated based on distance to nearest spiral points.
        """
        if len(x_eval.shape) == 1:
            x_eval = x_eval.reshape(1, -1)
            
        closest_points, distances = self._get_closest_spiral_points(x_eval)
        
        # Compute weights for each spiral based on distances
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

class TwoMoons:
    def __init__(self, noise_scale=0.1, seed=42):
        self.noise_scale = noise_scale
        self.sigma2 = noise_scale**2
        np.random.seed(seed)

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
epochs = 500
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


