import sys
from pathlib import Path

# Projenin ana klasörünü bul
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ana klasörü Python'ın modül arama yoluna ekle
sys.path.insert(0, str(PROJECT_ROOT))


import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


# Proje içindeki dosyaları import et
from models.autoencoder import Autoencoder

from evaluation.metrics import (
    calculate_mse,
    calculate_psnr,
    calculate_ssim
)


# -------------------------
# Configuration
# -------------------------

BATCH_SIZE = 128
LATENT_DIM = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10


# -------------------------
# Device
# -------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# -------------------------
# Data transformation
# -------------------------

transform = transforms.ToTensor()


# -------------------------
# Load datasets
# -------------------------

train_dataset = datasets.MNIST(
    root=PROJECT_ROOT / "data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root=PROJECT_ROOT / "data",
    train=False,
    download=True,
    transform=transform
)


# -------------------------
# DataLoaders
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
# Create model
# -------------------------

model = Autoencoder(
    latent_dim=LATENT_DIM
).to(device)


# -------------------------
# Loss function
# -------------------------

criterion = nn.MSELoss()


# -------------------------
# Optimizer
# -------------------------

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -------------------------
# Training
# -------------------------

for epoch in range(NUM_EPOCHS):

    model.train()

    total_loss = 0.0

    for images, _ in train_loader:

        images = images.to(device)

        optimizer.zero_grad()

        reconstructed = model(images)

        loss = criterion(
            reconstructed,
            images
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()


    average_loss = (
        total_loss /
        len(train_loader)
    )


    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Loss: {average_loss:.4f}"
    )

# -------------------------
# Evaluation
# -------------------------

model.eval()

total_mse = 0.0
total_ssim = 0.0

number_of_batches = 0


with torch.no_grad():

    for images, _ in test_loader:

        images = images.to(device)

        reconstructed = model(images)

        batch_mse = calculate_mse(
            images,
            reconstructed
        )

        batch_ssim = calculate_ssim(
            images,
            reconstructed
        )

        total_mse += batch_mse

        total_ssim += batch_ssim

        number_of_batches += 1


average_mse = (
    total_mse /
    number_of_batches
)


average_ssim = (
    total_ssim /
    number_of_batches
)


average_psnr = calculate_psnr(
    average_mse
)


print("\nTest Results")

print(
    f"Test MSE: "
    f"{average_mse:.6f}"
)

print(
    f"Test PSNR: "
    f"{average_psnr:.2f} dB"
)

print(
    f"Test SSIM: "
    f"{average_ssim:.4f}"
)

# Move tensors to CPU
images = images.cpu()
reconstructed = reconstructed.cpu()


# -------------------------
# Visualization
# -------------------------

plt.figure(figsize=(8, 4))


plt.subplot(1, 2, 1)

plt.imshow(
    images[0].squeeze(),
    cmap="gray"
)

plt.title("Original")

plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(
    reconstructed[0].squeeze(),
    cmap="gray"
)

plt.title("Reconstructed")

plt.axis("off")


plt.tight_layout()

plt.show()