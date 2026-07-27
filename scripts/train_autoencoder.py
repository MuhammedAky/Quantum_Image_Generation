import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.autoencoder import Autoencoder


# -------------------------
# Configuration
# -------------------------

BATCH_SIZE = 128
LATENT_DIM = 8
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


with torch.no_grad():

    images, _ = next(iter(test_loader))

    images = images.to(device)

    reconstructed = model(images)


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