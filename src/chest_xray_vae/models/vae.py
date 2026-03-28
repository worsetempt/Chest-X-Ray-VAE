from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 128,
        hidden_dims: Sequence[int] = (32, 64, 128, 256),
    ):
        super().__init__()
        if len(hidden_dims) != 4:
            raise ValueError("This implementation expects exactly 4 hidden dims for 64x64 inputs.")

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dims = tuple(hidden_dims)
        self.final_channels = self.hidden_dims[-1]
        self.feature_hw = 4

        modules = []
        prev_c = in_channels
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(prev_c, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                )
            )
            prev_c = h_dim
        self.enc = nn.Sequential(*modules)

        flattened_dim = self.final_channels * self.feature_hw * self.feature_hw
        self.fc_mu = nn.Linear(flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(flattened_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flattened_dim)

        rev = list(self.hidden_dims[::-1])
        dec_modules = []
        for i in range(len(rev) - 1):
            dec_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(rev[i], rev[i + 1], kernel_size=4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                )
            )

        self.dec = nn.Sequential(
            *dec_modules,
            nn.ConvTranspose2d(rev[-1], in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor):
        h = self.enc(x).flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        h = self.fc_dec(z).view(-1, self.final_channels, self.feature_hw, self.feature_hw)
        return self.dec(h)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar


def vae_loss(x: torch.Tensor, xhat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float):
    recon = F.binary_cross_entropy(xhat, x, reduction="sum") / x.size(0)
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()
    total = recon + beta * kl
    return total, recon, kl
