from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from chest_xray_vae.config import DataConfig


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def build_dataset(root: str | Path, split: str, image_size: int):
    root = Path(root)
    split_root = root / split
    if not split_root.exists():
        raise FileNotFoundError(f"Dataset split not found: {split_root}")
    return ImageFolder(split_root.as_posix(), transform=build_transform(image_size))


def build_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    device_type: str,
):
    use_workers = max(0, int(num_workers))
    persistent = use_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=use_workers,
        pin_memory=(pin_memory and device_type == "cuda"),
        drop_last=shuffle,
        persistent_workers=persistent,
    )


def create_train_val_loaders(cfg: DataConfig, device_type: str):
    train_ds = build_dataset(cfg.root, "train", cfg.image_size)
    val_ds = build_dataset(cfg.root, "val", cfg.image_size)

    train_loader = build_loader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        device_type=device_type,
    )
    val_loader = build_loader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        device_type=device_type,
    )
    return train_ds, val_ds, train_loader, val_loader


def create_eval_loader(cfg: DataConfig, device_type: str):
    ds = build_dataset(cfg.root, cfg.split, cfg.image_size)
    if cfg.max_eval_images is not None:
        ds = Subset(ds, list(range(min(cfg.max_eval_images, len(ds)))))
    loader = build_loader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        device_type=device_type,
    )
    return ds, loader
