import torch
import torch.nn as nn


class VAE(nn.Module):

    def __init__(
        self,
        input_dim=784,
        hidden_dim=128,
        latent_dim=8
    ):

        super().__init__()

        # -------------------------
        # Encoder
        # -------------------------

        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU()
        )


        # Latent distribution mean

        self.fc_mu = nn.Linear(
            hidden_dim,
            latent_dim
        )


        # Latent distribution log variance

        self.fc_logvar = nn.Linear(
            hidden_dim,
            latent_dim
        )


        # -------------------------
        # Decoder
        # -------------------------

        self.decoder = nn.Sequential(

            nn.Linear(
                latent_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                input_dim
            ),

            nn.Sigmoid()
        )


    def encode(
        self,
        x
    ):

        hidden = self.encoder(x)

        mu = self.fc_mu(hidden)

        logvar = self.fc_logvar(hidden)

        return mu, logvar


    def reparameterize(
        self,
        mu,
        logvar
    ):

        std = torch.exp(
            0.5 * logvar
        )

        epsilon = torch.randn_like(
            std
        )

        z = (
            mu +
            epsilon * std
        )

        return z


    def decode(
        self,
        z
    ):

        reconstructed = self.decoder(z)

        return reconstructed


    def forward(
        self,
        x
    ):

        mu, logvar = self.encode(x)

        z = self.reparameterize(
            mu,
            logvar
        )

        reconstructed = self.decode(z)

        return (
            reconstructed,
            mu,
            logvar
        )