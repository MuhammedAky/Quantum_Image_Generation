import torch
import torch.nn.functional as F

def vae_loss(
    reconstruction,
    original,
    mu,
    log_var,
    beta=1.0
):


# Orijinal görüntü ile yeniden oluşturulan
# görüntü arasındaki ortalama karesel hata.
    reconstruction_loss = F.mse_loss(
        reconstruction,
        original,
        reduction="mean"
    )

# VAE'nin KL divergence kaybı.
# Bu terim latent dağılımın standart normal
# dağılıma yaklaşmasını sağlar.
    kl_loss = -0.5 * torch.mean(
        1
        + log_var
        - mu.pow(2)
        - log_var.exp()
    )

# Toplam VAE kaybı.
    total_loss = (
        reconstruction_loss
        + beta * kl_loss
    )

    return {
        "total_loss": total_loss,
        "reconstruction_loss": reconstruction_loss,
        "kl_loss": kl_loss
    }

