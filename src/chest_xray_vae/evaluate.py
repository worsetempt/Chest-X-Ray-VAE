from __future__ import annotations

import math

import torch
from tqdm import tqdm

from chest_xray_vae.config import config_to_dict, load_eval_config
from chest_xray_vae.data import create_eval_loader
from chest_xray_vae.models.vae import VAE, vae_loss
from chest_xray_vae.utils.checkpoint import load_model_checkpoint
from chest_xray_vae.utils.io import ensure_dir, save_json
from chest_xray_vae.utils.metrics import (
    build_fid_and_is,
    resize_for_inception,
    save_image_grid,
    select_device,
    to_3ch,
    to_uint8_0_255,
)
from chest_xray_vae.utils.seed import set_seed


@torch.no_grad()
def sample_fake(model: VAE, batch_size: int, latent_dim: int, device: torch.device):
    z = torch.randn(batch_size, latent_dim, device=device)
    x = model.decode(z).clamp(0, 1)
    return resize_for_inception(x)


def run_evaluation(config_path: str):
    cfg = load_eval_config(config_path)
    set_seed(cfg.seed)

    device = select_device(cfg.eval.device)
    out_dir = ensure_dir(cfg.eval.out_dir)
    save_json(out_dir / "config_snapshot.json", config_to_dict(cfg))

    ds, loader = create_eval_loader(cfg.data, device.type)

    model = VAE(
        in_channels=cfg.model.in_channels,
        latent_dim=cfg.model.latent_dim,
        hidden_dims=cfg.model.hidden_dims,
    ).to(device)
    checkpoint = load_model_checkpoint(model, cfg.eval.checkpoint_path, device)
    model.eval()

    fid, inc = build_fid_and_is(device)

    total_sum = recon_sum = kl_sum = 0.0
    num_batches = 0

    for x, _ in tqdm(loader, desc="Eval real"):
        x = x.to(device, non_blocking=True)
        xhat, mu, logvar = model(x)
        loss, recon, kl = vae_loss(x, xhat, mu, logvar, beta=1.0)

        total_sum += loss.item()
        recon_sum += recon.item()
        kl_sum += kl.item()
        num_batches += 1

        real_for_metrics = resize_for_inception(x)
        fid.update(to_uint8_0_255(to_3ch(real_for_metrics)), real=True)

    num_real = len(ds)
    num_fake_target = int(cfg.eval.num_fake_mult * num_real)

    fake_done = 0
    while fake_done < num_fake_target:
        bs = min(cfg.data.batch_size, num_fake_target - fake_done)
        gen = sample_fake(model, bs, cfg.model.latent_dim, device)
        gen_u8 = to_uint8_0_255(to_3ch(gen))
        fid.update(gen_u8, real=False)
        inc.update(gen_u8)
        fake_done += bs

    fid_value = float(fid.compute().item())
    is_mean, is_std = inc.compute()

    metrics = {
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "num_real_images": num_real,
        "num_fake_images": num_fake_target,
        "eval_total_loss": total_sum / max(1, num_batches),
        "eval_recon_loss": recon_sum / max(1, num_batches),
        "eval_kl_loss": kl_sum / max(1, num_batches),
        "fid_2048": fid_value,
        "inception_score_mean": float(is_mean.item()),
        "inception_score_std": float(is_std.item()),
    }
    save_json(out_dir / "metrics.json", metrics)

    with torch.no_grad():
        examples = sample_fake(model, 4, cfg.model.latent_dim, device).cpu()
        save_image_grid(examples, out_dir / "generated_examples_4.png", nrow=4, title="Generated examples")

        n = 64
        side = int(math.sqrt(n))
        grid_samples = sample_fake(model, side * side, cfg.model.latent_dim, device).cpu()
        save_image_grid(grid_samples, out_dir / "generated_grid.png", nrow=side, title="Generated grid")

    print("Evaluation complete.")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"Artifacts saved to: {out_dir}")
