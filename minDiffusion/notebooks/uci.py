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

# Load UCI datasets

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
noise_scale = 1

class UCIDiffusion:
    def __init__(self, dataset_name, hidden_dim=512):
        self.dataset_name = dataset_name
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Load and preprocess data
        self.load_data()
        
        # Setup diffusion parameters
        self.setup_diffusion()
        
        # Initialize model with wider layers for higher dimensional data
        self.model = MLPDiffusion(self.input_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
    def load_data(self):
        print(f"Loading {self.dataset_name} dataset...")
        # Load the specified UCI dataset
        try:
            data = np.load(
                f"/scratch/Score_KDE/minDiffusion/uci/datasets/{self.dataset_name}/data.npy",
                allow_pickle=True,
            )
            
            # Remove any NaN or infinite values
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            print(data.shape)
            print(data[0])
            
            # Normalize the data
            self.data_mean = data.mean(axis=0)
            self.data_std = data.std(axis=0)
            self.data_std[self.data_std == 0] = 1  # Prevent division by zero
            data_normalized = (data - self.data_mean) / self.data_std
            
            # Convert to tensor
            self.X = torch.tensor(data_normalized, dtype=torch.float32).to(device)
            self.input_dim = self.X.shape[1]
            
            print(f"Dataset shape: {self.X.shape}")
            
            # Create dataset and dataloader
            dataset = TensorDataset(self.X)
            self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
        
    def setup_diffusion(self):
        # Noise schedule - using a slightly different beta schedule for high-dimensional data
        self.betas = torch.linspace(1e-4, 0.02, T).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)
    
    def train(self):
        best_loss = float('inf')
        best_model = None
        losses = []  # Track losses for plotting
        
        print(f"\nTraining {self.dataset_name} diffusion model...")
        # Training loop
        tqdm_epochs = tqdm(range(epochs))
        for epoch in tqdm_epochs:
            epoch_losses = []
            for batch in self.dataloader:
                x = batch[0]
                batch_size = x.size(0)
                
                # Random timesteps for each sample
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
            
            # Calculate average epoch loss
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_loss)
            
            # Update progress bar
            tqdm_epochs.set_description(f"Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model = self.model.state_dict()
            
            # Plot loss every 50 epochs
            if (epoch + 1) % 50 == 0:
                plt.figure(figsize=(10, 4))
                plt.plot(losses)
                plt.title(f'{self.dataset_name} Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.yscale('log')
                plt.show()
        
        # Save the best model
        torch.save(best_model, f"best_model_{self.dataset_name}.pth")
    
    def sample(self, n_samples=1000):
        self.model.eval()
        with torch.no_grad():
            # Initialize with Gaussian noise
            x = torch.randn(n_samples, self.input_dim).to(device)
            
            # Reverse diffusion process
            for t_step in tqdm(reversed(range(T)), desc="Sampling"):
                t = torch.full((n_samples,), t_step, device=device, dtype=torch.long)
                pred_noise = self.model(x, t)
                
                alpha_t = self.alphas[t_step]
                alpha_bar_t = self.alpha_bars[t_step]
                beta_t = self.betas[t_step]
                
                # Only add noise if t > 0
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
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.time_embed = nn.Embedding(T, hidden_dim)
        
        # Wider architecture for higher dimensional data
        self.input = nn.Linear(input_dim, hidden_dim)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.hidden2 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.hidden3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output = nn.Linear(hidden_dim, input_dim)
        
        self.activation = nn.SiLU()  # Using SiLU (Swish) activation
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, t):
        t_embed = self.time_embed(t)
        h = self.input(x)
        h += t_embed
        
        h = self.activation(h)
        h = self.hidden1(h)
        h = self.activation(h)
        h = self.dropout(h)
        
        h = self.hidden2(h)
        h = self.activation(h)
        h = self.dropout(h)
        
        h = self.hidden3(h)
        h = self.activation(h)
        
        return self.output(h)

def train_and_evaluate(dataset_name):
    print(f"\nTraining on {dataset_name} dataset:")
    diffusion = UCIDiffusion(dataset_name)
    diffusion.train()
    
    # Generate samples
    samples = diffusion.sample(1000)
    
    # Plot original vs generated data distributions
    plot_distributions(diffusion.X.cpu().numpy() * diffusion.data_std + diffusion.data_mean, 
                      samples,
                      dataset_name)

def plot_distributions(original, generated, dataset_name):
    n_features = original.shape[1]
    n_cols = min(4, n_features)  # Maximum 4 columns
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Try to load feature names
    try:
        with open(f"/scratch/Score_KDE/minDiffusion/uci/datasets/{dataset_name}/feature_names.txt", 'r') as f:
            feature_names = f.read().splitlines()
    except:
        feature_names = [f'Feature {i+1}' for i in range(n_features)]
    
    for i in range(n_features):
        axes[i].hist(original[:, i], bins=50, alpha=0.5, label='Original', density=True)
        axes[i].hist(generated[:, i], bins=50, alpha=0.5, label='Generated', density=True)
        axes[i].set_title(feature_names[i])
        axes[i].legend()
    
    # Hide empty subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'{dataset_name} - Original vs Generated Distributions')
    plt.tight_layout()
    plt.show()

# Train on each dataset
datasets = ['AReM', 'CASP']
for dataset in datasets:
    train_and_evaluate(dataset)
# -


