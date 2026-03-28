from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from chest_xray_vae.config import config_to_dict, load_train_config
from chest_xray_vae.data import create_train_val_loaders
from chest_xray_vae.models.vae import VAE, vae_loss
from chest_xray_vae.utils.checkpoint import save_checkpoint
from chest_xray_vae.utils.io import ensure_dir, save_json
from chest_xray_vae.utils.metrics import beta_schedule, save_history_plots, save_image_grid, select_device
from chest_xray_vae.utils.seed import set_seed


def run_training(config_path: str):
    cfg = load_train_config(config_path)
    set_seed(cfg.seed)

    device = select_device(cfg.train.device)
    torch.backends.cudnn.benchmark = True

    train_ds, val_ds, train_loader, val_loader = create_train_val_loaders(cfg.data, device.type)

    model = VAE(
        in_channels=cfg.model.in_channels,
        latent_dim=cfg.model.latent_dim,
        hidden_dims=cfg.model.hidden_dims,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    exp_dir = ensure_dir(cfg.output.exp_dir)
    ckpt_dir = ensure_dir(exp_dir / "checkpoints")
    plot_dir = ensure_dir(exp_dir / "plots")
    sample_dir = ensure_dir(exp_dir / "samples")

    save_json(exp_dir / "config_snapshot.json", config_to_dict(cfg))

    history = {
        "train_total": [],
        "train_recon": [],
        "train_kl": [],
        "val_total": [],
        "val_recon": [],
        "val_kl": [],
        "beta": [],
    }
    best_val = float("inf")

    for epoch in range(cfg.train.epochs):
        beta_t = beta_schedule(epoch, cfg.train.beta, cfg.train.beta_warmup_epochs)
        history["beta"].append(beta_t)

        model.train()
        total_sum = recon_sum = kl_sum = 0.0

        for x, _ in tqdm(train_loader, desc=f"Train {epoch + 1}/{cfg.train.epochs}"):
            x = x.to(device, non_blocking=True)

            xhat, mu, logvar = model(x)
            loss, recon, kl = vae_loss(x, xhat, mu, logvar, beta=beta_t)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_sum += loss.item()
            recon_sum += recon.item()
            kl_sum += kl.item()

        n_train = len(train_loader)
        train_total = total_sum / n_train
        train_recon = recon_sum / n_train
        train_kl = kl_sum / n_train

        model.eval()
        v_total_sum = v_recon_sum = v_kl_sum = 0.0
        last_batch = None

        with torch.no_grad():
            for x, _ in tqdm(val_loader, desc=f"Val {epoch + 1}/{cfg.train.epochs}"):
                x = x.to(device, non_blocking=True)
                xhat, mu, logvar = model(x)
                loss, recon, kl = vae_loss(x, xhat, mu, logvar, beta=beta_t)

                v_total_sum += loss.item()
                v_recon_sum += recon.item()
                v_kl_sum += kl.item()
                last_batch = (x.detach().cpu(), xhat.detach().cpu())

        n_val = len(val_loader)
        val_total = v_total_sum / n_val
        val_recon = v_recon_sum / n_val
        val_kl = v_kl_sum / n_val

        history["train_total"].append(train_total)
        history["train_recon"].append(train_recon)
        history["train_kl"].append(train_kl)
        history["val_total"].append(val_total)
        history["val_recon"].append(val_recon)
        history["val_kl"].append(val_kl)

        print(
            f"Epoch {epoch + 1:03d}/{cfg.train.epochs} | beta={beta_t:.4f} | "
            f"train total={train_total:.4f} recon={train_recon:.4f} kl={train_kl:.4f} | "
            f"val total={val_total:.4f} recon={val_recon:.4f} kl={val_kl:.4f}"
        )

        save_checkpoint(
            ckpt_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            best_val=best_val,
            config=config_to_dict(cfg),
        )

        if (epoch + 1) % cfg.output.save_every == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch + 1:03d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                best_val=best_val,
                config=config_to_dict(cfg),
            )

        if val_total < best_val:
            best_val = val_total
            save_checkpoint(
                ckpt_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                best_val=best_val,
                config=config_to_dict(cfg),
            )

        if last_batch is not None:
            x_real, x_recon = last_batch
            preview = torch.cat([x_real[:8], x_recon[:8]], dim=0)
            save_image_grid(
                preview,
                sample_dir / f"val_recon_epoch_{epoch + 1:03d}.png",
                nrow=8,
                title=f"Validation reconstructions epoch {epoch + 1}",
            )

    save_json(exp_dir / "metrics_history.json", history)
    save_history_plots(history, plot_dir)

    model.eval()
    with torch.no_grad():
        z = torch.randn(64, cfg.model.latent_dim, device=device)
        samples = model.decode(z).cpu()
    save_image_grid(samples, sample_dir / "generated_samples.png", nrow=8, title="Generated samples")

    print(f"Training complete. Best val loss: {best_val:.4f}")
    print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)}")
    print(f"Artifacts saved to: {exp_dir}")
