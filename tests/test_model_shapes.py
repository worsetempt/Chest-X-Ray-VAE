from chest_xray_vae.models.vae import VAE
import torch


def test_vae_output_shape():
    model = VAE(in_channels=1, latent_dim=128, hidden_dims=(32, 64, 128, 256))
    x = torch.randn(2, 1, 64, 64)
    xhat, mu, logvar = model(x)
    assert xhat.shape == x.shape
    assert mu.shape == (2, 128)
    assert logvar.shape == (2, 128)
