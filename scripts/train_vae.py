import sys
import time

from pathlib import Path


# -------------------------
# Project path
# -------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# -------------------------
# Libraries
# -------------------------

import torch
import torch.nn.functional as F

from torchvision import (
    datasets,
    transforms
)

from torch.utils.data import (
    DataLoader
)

import matplotlib.pyplot as plt


# -------------------------
# Project imports
# -------------------------

from models.vae import VAE

from evaluation.metrics import (
    calculate_psnr,
    calculate_ssim
)


# -------------------------
# Configuration
# -------------------------

LATENT_DIM = 32

HIDDEN_DIM = 128

BATCH_SIZE = 128

LEARNING_RATE = 0.001

NUM_EPOCHS = 10


# -------------------------
# Device
# -------------------------

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"
)


print(
    f"Using device: {device}"
)


# -------------------------
# Dataset
# -------------------------

transform = transforms.ToTensor()


train_dataset = datasets.FashionMNIST(

    root=(
        PROJECT_ROOT /
        "data"
    ),

    train=True,

    download=True,

    transform=transform
)


test_dataset = datasets.FashionMNIST(

    root=(
        PROJECT_ROOT /
        "data"
    ),

    train=False,

    download=True,

    transform=transform
)


# -------------------------
# Data loaders
# -------------------------

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True
)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False
)


# -------------------------
# Model
# -------------------------

model = VAE(

    input_dim=784,

    hidden_dim=HIDDEN_DIM,

    latent_dim=LATENT_DIM

).to(device)


# -------------------------
# Optimizer
# -------------------------

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE
)


# -------------------------
# VAE loss
# -------------------------

def vae_loss(

    reconstructed,

    original,

    mu,

    logvar

):

    # Reconstruction loss

    reconstruction_loss = (

        F.mse_loss(

            reconstructed,

            original,

            reduction="sum"

        )
    )


    # KL divergence

    kl_divergence = (

        -0.5

        * torch.sum(

            1

            + logvar

            - mu.pow(2)

            - logvar.exp()

        )
    )


    total_loss = (

        reconstruction_loss

        + kl_divergence
    )


    return (

        total_loss,

        reconstruction_loss,

        kl_divergence
    )


# -------------------------
# Training
# -------------------------

start_time = time.time()


for epoch in range(

    NUM_EPOCHS

):

    model.train()


    total_loss = 0.0

    total_reconstruction_loss = 0.0

    total_kl_loss = 0.0


    for images, _ in train_loader:


        images = (

            images

            .view(

                images.size(0),

                -1

            )

            .to(device)
        )


        optimizer.zero_grad()


        reconstructed, mu, logvar = (

            model(images)
        )


        (

            loss,

            reconstruction_loss,

            kl_loss

        ) = vae_loss(

            reconstructed,

            images,

            mu,

            logvar
        )


        loss.backward()


        optimizer.step()


        total_loss += (

            loss.item()
        )


        total_reconstruction_loss += (

            reconstruction_loss.item()
        )


        total_kl_loss += (

            kl_loss.item()
        )


    average_total_loss = (

        total_loss

        / len(train_dataset)
    )


    average_reconstruction_loss = (

        total_reconstruction_loss

        / len(train_dataset)
    )


    average_kl_loss = (

        total_kl_loss

        / len(train_dataset)
    )


    print(

        f"Epoch "

        f"[{epoch + 1}"

        f"/{NUM_EPOCHS}] "

        f"Total Loss: "

        f"{average_total_loss:.6f} | "

        f"Reconstruction: "

        f"{average_reconstruction_loss:.6f} | "

        f"KL: "

        f"{average_kl_loss:.6f}"

    )


training_time = (

    time.time()

    - start_time
)


# -------------------------
# Evaluation
# -------------------------

model.eval()


total_squared_error = 0.0

total_number_of_pixels = 0

total_ssim = 0.0

number_of_images = 0


with torch.no_grad():

    for images, _ in test_loader:


        original_images = (

            images.to(device)
        )


        flattened_images = (

            original_images

            .view(

                original_images.size(0),

                -1

            )
        )


        reconstructed, mu, logvar = (

            model(
                flattened_images
            )
        )


        squared_error = (

            (

                flattened_images

                - reconstructed

            )

            ** 2
        )


        total_squared_error += (

            squared_error

            .sum()

            .item()
        )


        total_number_of_pixels += (

            flattened_images

            .numel()
        )


        reconstructed_images = (

            reconstructed

            .view(

                -1,

                1,

                28,

                28
            )
        )


        batch_ssim = (

            calculate_ssim(

                original_images,

                reconstructed_images
            )
        )


        batch_size = (

            original_images

            .size(0)
        )


        total_ssim += (

            batch_ssim

            * batch_size
        )


        number_of_images += (

            batch_size
        )


test_mse = (

    total_squared_error

    / total_number_of_pixels
)


test_psnr = (

    calculate_psnr(

        test_mse
    )
)


test_ssim = (

    total_ssim

    / number_of_images
)


# -------------------------
# Results
# -------------------------

print(
    "\nTest Results"
)


print(

    f"Latent dimension: "

    f"{LATENT_DIM}"
)


print(

    f"Test MSE: "

    f"{test_mse:.6f}"
)


print(

    f"Test PSNR: "

    f"{test_psnr:.2f} dB"
)


print(

    f"Test SSIM: "

    f"{test_ssim:.4f}"
)


print(

    f"Training time: "

    f"{training_time:.2f} seconds"
)