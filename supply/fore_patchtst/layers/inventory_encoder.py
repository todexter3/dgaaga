import torch
from torch import nn

class InventoryEncoder(nn.Module):
    """
    Inventory state encoder (daily level)
    Input:  x_inv [B, d_inv]
    Output: z_inv [B, d_latent]
    """
    def __init__(self, d_inv, d_latent=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_inv, 32),
            nn.GELU(),
            nn.Linear(32, d_latent)
        )

    def forward(self, x_inv):
        return self.encoder(x_inv)


class InventoryAutoEncoder(nn.Module):
    """
    For pretraining inventory encoder
    """
    def __init__(self, d_inv, d_latent=8):
        super().__init__()
        self.encoder = InventoryEncoder(d_inv, d_latent)
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, 32),
            nn.GELU(),
            nn.Linear(32, d_inv)
        )

    def forward(self, x_inv):
        z = self.encoder(x_inv)
        x_hat = self.decoder(z)
        return z, x_hat