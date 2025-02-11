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


