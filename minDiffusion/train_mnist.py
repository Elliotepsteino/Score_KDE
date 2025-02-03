from typing import Dict, Optional, Tuple
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torchvision.transforms.functional import to_pil_image
from sympy import Ci

from mindiffusion.unet import NaiveUnet
from mindiffusion.ddpm import DDPM

from datasets import load_dataset

BASE_DIR = "/pscratch/sd/j/jwl50/Score_KDE/minDiffusion"
WANDB_PATH = f"{BASE_DIR}/wandb"

# Define a PyTorch Dataset wrapper for the Hugging Face MNIST dataset
class HFDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item["image"]
        label = item["label"]
        if self.transform:
            image = self.transform(image)
        return image, label
    

def train_mnist(
    n_epoch: int = 10000, device: str = "cuda:0", load_pth: Optional[str] = None
) -> None:
    # Initialize wandb tracking
    wandb.init(
        dir=WANDB_PATH,
        project="minDiffusion",
        name="mnist",
        config={
            "n_epoch": n_epoch,
            "device": device,
            "batch_size": 512,
            "learning_rate": 1e-5,
        },
    )

    ddpm = DDPM(eps_model=NaiveUnet(1, 1, n_feat=128, num_downsamples=1), betas=(1e-4, 0.02), n_T=1000)

    if load_pth is not None:
        ddpm.load_state_dict(torch.load(f"{BASE_DIR}/ddpm_mnist.pth"))

    ddpm.to(device)
    wandb.watch(ddpm, log="all")

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    hf_dataset = load_dataset("ylecun/mnist", split="train")
    dataset = HFDataset(hf_dataset, transform=tf)

    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=16)
    optim = torch.optim.Adam(ddpm.parameters(), lr=1e-5)

    for i in range(n_epoch):
        print(f"Epoch {i} : ")
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        ddpm.eval()
        with torch.no_grad():
            xh = ddpm.sample(8, (1, 28, 28), device)
            xset = torch.cat([xh, x[:8]], dim=0)
            grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
            save_image(grid, f"./contents/ddpm_sample_mnist{i}.png")
            
            # Log metrics and sample image grid to wandb
            wandb.log({
                "epoch": i,
                "loss": loss_ema,
                "sample_grid": wandb.Image(to_pil_image(grid), caption=f"Epoch {i} sample")
            })

            # save model
            torch.save(ddpm.state_dict(), f"{BASE_DIR}/ddpm_mnist.pth")


if __name__ == "__main__":
    train_mnist()
    wandb.finish()

