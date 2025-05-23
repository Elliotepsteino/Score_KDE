from typing import Optional
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from mindiffusion.ddpm import DDPM
from data.datagen_2d import TwoDDataset

BASE_DIR = "/pscratch/sd/j/jwl50/Score_KDE/minDiffusion"
WANDB_PATH = f"{BASE_DIR}/wandb"
PLOT_PATH = f"{BASE_DIR}/contents"

class EpsMLP(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=128, out_dim=2):
        super(EpsMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
    """
    Forward pass for the EpsMLP model.
    Input:
        x: (B, 2, 1, 1)
        t: (B)
    """
    def forward(self, x, t):
        x = x.view(x.size(0), -1) # (B, 2)
        if len(t.shape) > 1:
            t = t.reshape(-1)
        print(f"x shape: {x.shape}, t shape: {t.shape}")
        x_time = torch.cat((x, t.unsqueeze(-1)), dim=1) # (B, 3)
        print(f"x_time shape: {x_time.shape}")
        return self.net(x_time).reshape(x.size(0), 2, 1, 1) # (B, 2, 1, 1)


def train_2d(n_epoch: int = 10000, device: str = "cuda:0", load_pth: Optional[str] = None) -> None:
    # Initialize wandb
    wandb.init(
        dir=WANDB_PATH,
        project="minDiffusion",
        name="2d_diffusion",
        config={
            "n_epoch": n_epoch,
            "device": device,
            "batch_size": 512,
            "learning_rate": 1e-5,
            "data_mode": "swissroll"
        },
    )

    # Instantiate the diffusion model with our small MLP network.
    eps_model = EpsMLP(in_dim=3, hidden_dim=128, out_dim=2)
    ddpm = DDPM(eps_model=eps_model, betas=(1e-4, 0.02), n_T=1000)

    if load_pth is not None:
        ddpm.load_state_dict(torch.load(f"{BASE_DIR}/ddpm_2d.pth"))

    ddpm.to(device)
    wandb.watch(ddpm, log="all")

    # Prepare the dataset and data loader for 2D points.
    dataset = TwoDDataset(mode="swissroll", batch_size=512)
    dataloader = DataLoader(dataset, batch_size=512, num_workers=0)

    optim = torch.optim.Adam(ddpm.parameters(), lr=1e-5)

    for i in range(n_epoch):
        print(f"Epoch {i} : ")
        ddpm.train()
        pbar = tqdm(dataloader)
        loss_ema = None
        for x in pbar:
            # x: (B, 2, 1, 1)
            ddpm.train()
            x = x.to(device)
            optim.zero_grad()
            loss = ddpm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

            if i % 1000 == 0:
                ddpm.eval()
                with torch.no_grad():
                    # Sample 8 points; note shape argument is (2,) for 2D.
                    samples = ddpm.sample(8, (2,), device).unsqueeze(-1).unsqueeze(-1)
                    samples_np = samples.cpu().numpy()

                    # Plot the generated samples as a scatter.
                    plt.figure()
                    plt.scatter(samples_np[:, 0], samples_np[:, 1], c='blue', marker='o')
                    plt.title(f"Epoch {i} Samples")
                    plot_path = f"{PLOT_PATH}/ddpm_sample_2d_epoch{i}.png"
                    plt.savefig(plot_path)
                    plt.close()

                    wandb.log({
                        "epoch": i,
                        "loss": loss_ema,
                        "sample_scatter": wandb.Image(plot_path)
                    })

                    torch.save(ddpm.state_dict(), f"{BASE_DIR}/ddpm_2d.pth")


if __name__ == "__main__":
    train_2d()
    wandb.finish()
