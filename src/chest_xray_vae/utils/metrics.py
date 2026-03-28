from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def beta_schedule(epoch_idx: int, beta: float, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return beta
    t = min(1.0, (epoch_idx + 1) / warmup_epochs)
    return beta * t


def to_3ch(x_1ch: torch.Tensor) -> torch.Tensor:
    return x_1ch.repeat(1, 3, 1, 1)


def to_uint8_0_255(x_float_0_1: torch.Tensor) -> torch.Tensor:
    return (x_float_0_1.clamp(0, 1) * 255.0).round().to(torch.uint8)


def resize_for_inception(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] != (299, 299):
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    return x


def build_fid_and_is(device: torch.device):
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    inc = InceptionScore(normalize=True).to(device)
    return fid, inc


@torch.no_grad()
def save_image_grid(images: torch.Tensor, path: str | Path, nrow: int = 8, title: str | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    grid = vutils.make_grid(images.cpu(), nrow=nrow, padding=2, normalize=True)
    vutils.save_image(grid, path)

    plt.figure(figsize=(8, 8))
    if title:
        plt.title(title)
    plt.axis("off")
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)).squeeze(), cmap="gray")
    plt.tight_layout()
    plt.savefig(path.with_suffix(".preview.png"), dpi=150)
    plt.close()


def save_history_plots(history: dict, out_dir: str | Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        ("total", "Loss Curve (Total)", "Loss"),
        ("recon", "Loss Curve (Recon)", "Recon"),
        ("kl", "Loss Curve (KL)", "KL"),
    ]

    for key, title, ylabel in plot_specs:
        plt.figure()
        plt.plot(history[f"train_{key}"], label=f"train_{key}")
        plt.plot(history[f"val_{key}"], label=f"val_{key}")
        plt.legend()
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(out_dir / f"{key}.png", dpi=150)
        plt.close()

    plt.figure()
    plt.plot(history["beta"], label="beta")
    plt.legend()
    plt.title("Beta Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("beta")
    plt.tight_layout()
    plt.savefig(out_dir / "beta_schedule.png", dpi=150)
    plt.close()
