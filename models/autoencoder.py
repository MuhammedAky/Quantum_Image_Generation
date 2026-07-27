import torch.nn as nn


class Autoencoder(nn.Module):
    """
    A simple fully connected Autoencoder for MNIST images.

    Input:
        28 x 28 grayscale image

    Encoder:
        784 -> 128 -> latent_dim

    Decoder:
        latent_dim -> 128 -> 784

    Output:
        Reconstructed 28 x 28 grayscale image
    """

    def __init__(self, latent_dim=8):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the Autoencoder.

        Args:
            x: Input image tensor with shape
               [batch_size, 1, 28, 28]

        Returns:
            Reconstructed image tensor with shape
            [batch_size, 1, 28, 28]
        """

        z = self.encoder(x)

        reconstruction = self.decoder(z)

        reconstruction = reconstruction.view(
            -1,
            1,
            28,
            28
        )

        return reconstruction