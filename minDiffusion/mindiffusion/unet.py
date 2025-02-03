"""
Simple Unet Structure.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper module: a LayerNorm for 2D convolutional features.
class LayerNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-6, affine=True):
        super().__init__()
        # nn.LayerNorm expects the normalized shape to be the last dimensions.
        self.layer_norm = nn.LayerNorm(num_features, eps=eps, elementwise_affine=affine)
    
    def forward(self, x):
        # x has shape (N, C, H, W) – rearrange to (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        # Return to original shape: (N, C, H, W)
        return x.permute(0, 3, 1, 2)


class Conv3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_res: bool = False) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            LayerNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            LayerNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            LayerNorm2d(out_channels),
            nn.ReLU(),
        )
        self.is_res = is_res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.main(x)
        if self.is_res:
            x = x + self.conv(x)
            return x / 1.414
        else:
            return self.conv(x)


# UnetDown is kept simple: it applies a Conv3 then a MaxPool2d.
class UnetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UnetDown, self).__init__()
        # Note: we wrap the conv and pooling in a Sequential.
        # For the single downsampling branch we will extract the convolutional output
        # (which retains the input's spatial dimensions) from the first layer.
        layers = [Conv3(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# The original UnetUp used in the multi-downsampling branch.
# (We leave it unchanged.)
class UnetUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UnetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            Conv3(out_channels, out_channels),
            Conv3(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Concatenate directly, assuming input spatial sizes already match.
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class TimeSiren(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super(TimeSiren, self).__init__()
        self.lin1 = nn.Linear(1, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1)
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x


class NaiveUnet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_feat: int = 256, num_downsamples: int = 3) -> None:
        super(NaiveUnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_feat = n_feat
        self.num_downsamples = num_downsamples

        self.init_conv = Conv3(in_channels, n_feat, is_res=True)

        if self.num_downsamples == 1:
            # For smaller inputs (e.g., 28x28 MNIST), use a single downsampling branch.
            # We cannot use the entire UnetDown since it includes pooling.
            # Instead, we extract the "skip" connection from the conv layer inside UnetDown,
            # and then use the pooled output for upsampling.
            self.down1 = UnetDown(n_feat, n_feat)
            # Define an upsampling layer to bring the pooled feature from 14x14 back to 28x28.
            self.upsample = nn.ConvTranspose2d(n_feat, n_feat, kernel_size=2, stride=2)
            # After concatenation (channels: n_feat from skip + n_feat from upsampled output),
            # reduce the channels to out_channels via a couple of conv layers.
            self.single_up = nn.Sequential(
                Conv3(2 * n_feat, n_feat),
                nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=1),
            )
        else:
            # Original architecture with three downsamplings
            self.down1 = UnetDown(n_feat, n_feat)
            self.down2 = UnetDown(n_feat, 2 * n_feat)
            self.down3 = UnetDown(2 * n_feat, 2 * n_feat)
            self.to_vec = nn.Sequential(nn.AvgPool2d(4), nn.ReLU())
            self.timeembed = TimeSiren(2 * n_feat)
            self.up0 = nn.Sequential(
                nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 4, 4),
                LayerNorm2d(2 * n_feat),
                nn.ReLU(),
            )
            self.up1 = UnetUp(4 * n_feat, 2 * n_feat)
            self.up2 = UnetUp(4 * n_feat, n_feat)
            self.up3 = UnetUp(2 * n_feat, n_feat)
            self.out = nn.Conv2d(2 * n_feat, self.out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)  # For MNIST, x: (N, n_feat, 28, 28)
        if self.num_downsamples == 1:
            # For single downsampling branch:
            # Instead of using x directly as the skip—which is (N, n_feat, 28, 28)—
            # we extract the pre-pooling activation from the UnetDown module.
            # Here, self.down1.model[0] is the Conv3 block.
            skip = self.down1.model[0](x)  # Expected shape: (N, n_feat, 28, 28)
            d1 = self.down1(x)             # After pooling: (N, n_feat, 14, 14)
            d1_up = self.upsample(d1)        # Upsample d1: (N, n_feat, 28, 28)
            combined = torch.cat((skip, d1_up), dim=1)  # (N, 2*n_feat, 28, 28)
            out = self.single_up(combined)  # Reduce channels to out_channels
            return out
        else:
            # Original forward pass with three downsampling stages.
            down1 = self.down1(x)       # (N, n_feat, H/2, W/2)
            down2 = self.down2(down1)    # (N, 2*n_feat, H/4, W/4)
            down3 = self.down3(down2)    # (N, 2*n_feat, H/8, W/8)

            thro = self.to_vec(down3)
            temb = self.timeembed(t).view(-1, self.n_feat * 2, 1, 1)

            thro = self.up0(thro + temb)
            up1 = self.up1(thro, down3) + temb
            up2 = self.up2(up1, down2)
            up3 = self.up3(up2, down1)
            out = self.out(torch.cat((up3, x), 1))
            return out
