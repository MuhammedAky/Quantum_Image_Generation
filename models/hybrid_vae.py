import torch
import torch.nn as nn

from models.quantum_layer import (
    SingleQubitLayer
)


class HybridVAE(nn.Module):

    def __init__(
        self,
        input_dim=784,
        hidden_dim=256,
        latent_dim=8
    ):

        super().__init__()

        self.input_dim = input_dim

        self.hidden_dim = hidden_dim

        self.latent_dim = latent_dim


        # =================================
        # KLASİK VAE ENCODER
        # =================================

        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU()
        )


        # VAE ortalama vektörü

        self.mu_layer = nn.Linear(

            hidden_dim,

            latent_dim
        )


        # VAE log varyans vektörü

        self.log_var_layer = nn.Linear(

            hidden_dim,

            latent_dim
        )


        # =================================
        # KLASİK → KUANTUM
        # =================================

        # Latent vektörü tek açıya indir

        self.to_quantum = nn.Sequential(

            nn.Linear(

                latent_dim,

                1
            ),

            nn.Tanh()
        )


        # =================================
        # TEK-QUBIT KUANTUM KATMANI
        # =================================

        self.quantum_layer = (

            SingleQubitLayer()
        )


        # =================================
        # KUANTUM → KLASİK
        # =================================

        # Tek ölçüm değerini tekrar
        # latent boyuta genişlet

        self.from_quantum = nn.Sequential(

            nn.Linear(

                1,

                latent_dim
            ),

            nn.ReLU()
        )


        # =================================
        # KLASİK DECODER
        # =================================

        self.decoder = nn.Sequential(

            nn.Linear(

                latent_dim,

                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(

                hidden_dim,

                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(

                hidden_dim,

                input_dim
            ),

            nn.Sigmoid()
        )


    # =====================================
    # ENCODER
    # =====================================

    def encode(
        self,
        x
    ):

        hidden = self.encoder(
            x
        )

        mu = self.mu_layer(
            hidden
        )

        log_var = self.log_var_layer(
            hidden
        )

        return (

            mu,

            log_var
        )


    # =====================================
    # REPARAMETERIZATION
    # =====================================

    def reparameterize(
        self,
        mu,
        log_var
    ):

        std = torch.exp(

            0.5 * log_var
        )

        epsilon = torch.randn_like(

            std
        )

        z = (

            mu

            +

            epsilon * std
        )

        return z


    # =====================================
    # QUANTUM BOTTLENECK
    # =====================================

    def quantum_bottleneck(
        self,
        z
    ):

        # [batch, latent_dim]
        # →
        # [batch, 1]

        angle = self.to_quantum(
            z
        )


        # Tanh çıktısı yaklaşık
        # [-1, 1] aralığındadır.
        #
        # Bunu açı aralığına ölçekliyoruz.

        angle = (

            angle

            *

            torch.pi
        )


        # Kuantum devresini çalıştır

        quantum_output = (

            self.quantum_layer(

                angle
            )
        )


        # [batch, 1]
        # →
        # [batch, latent_dim]

        quantum_latent = (

            self.from_quantum(

                quantum_output
            )
        )


        return (

            quantum_latent
        )


    # =====================================
    # DECODER
    # =====================================

    def decode(
        self,
        quantum_latent
    ):

        reconstruction = (

            self.decoder(

                quantum_latent
            )
        )

        return (

            reconstruction
        )


    # =====================================
    # FORWARD
    # =====================================

    def forward(
        self,
        x
    ):

        # 1. Klasik encoder

        mu, log_var = (

            self.encode(

                x
            )
        )


        # 2. VAE örnekleme

        z = (

            self.reparameterize(

                mu,

                log_var
            )
        )


        # 3. Kuantum bottleneck

        quantum_latent = (

            self.quantum_bottleneck(

                z
            )
        )


        # 4. Klasik decoder

        reconstruction = (

            self.decode(

                quantum_latent
            )
        )


        return (

            reconstruction,

            mu,

            log_var,

            z,

            quantum_latent
        )