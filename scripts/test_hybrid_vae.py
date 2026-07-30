import sys
from pathlib import Path

import torch


PROJECT_ROOT = (

    Path(__file__)

    .resolve()

    .parents[1]
)


sys.path.insert(

    0,

    str(PROJECT_ROOT)
)


from models.hybrid_vae import (

    HybridVAE
)


# Model oluştur

model = HybridVAE(

    input_dim=784,

    hidden_dim=256,

    latent_dim=8
)


# Sahte MNIST batch'i

images = torch.rand(

    4,

    784
)


# Modeli çalıştır

(

    reconstruction,

    mu,

    log_var,

    z,

    quantum_latent

) = model(

    images
)


print(

    "Input:",

    images.shape
)


print(

    "Reconstruction:",

    reconstruction.shape
)


print(

    "Mu:",

    mu.shape
)


print(

    "Log variance:",

    log_var.shape
)


print(

    "Latent z:",

    z.shape
)


print(

    "Quantum latent:",

    quantum_latent.shape
)