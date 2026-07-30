import sys
import random
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import (

    DataLoader,

    Subset
)

from torchvision import (

    datasets,

    transforms
)

from tqdm import tqdm


# =====================================
# PROJE YOLU
# =====================================

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


from evaluation.metrics import (

    vae_loss
)


# =====================================
# DENEY AYARLARI
# =====================================

SEED = 42

BATCH_SIZE = 4

EPOCHS = 1

LEARNING_RATE = 1e-3

LATENT_DIM = 32

HIDDEN_DIM = 256

BETA = 1.0


# İlk test için küçük veri alt kümesi

TRAIN_SAMPLES = 1000

TEST_SAMPLES = 200


# =====================================
# RASTGELELİK
# =====================================

torch.manual_seed(

    SEED
)


np.random.seed(

    SEED
)


random.seed(

    SEED
)


# =====================================
# VERİ SETİ
# =====================================

transform = (

    transforms.ToTensor()
)


train_dataset = (

    datasets.MNIST(

        root=(

            PROJECT_ROOT

            /

            "data"
        ),

        train=True,

        download=True,

        transform=transform
    )
)


test_dataset = (

    datasets.MNIST(

        root=(

            PROJECT_ROOT

            /

            "data"
        ),

        train=False,

        download=True,

        transform=transform
    )
)


# İlk deney için küçük alt kümeler

train_dataset = (

    Subset(

        train_dataset,

        range(

            TRAIN_SAMPLES
        )
    )
)


test_dataset = (

    Subset(

        test_dataset,

        range(

            TEST_SAMPLES
        )
    )
)


train_loader = (

    DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True
    )
)


test_loader = (

    DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False
    )
)


# =====================================
# MODEL
# =====================================

model = (

    HybridVAE(

        input_dim=784,

        hidden_dim=HIDDEN_DIM,

        latent_dim=LATENT_DIM
    )
)


# =====================================
# OPTIMIZER
# =====================================

optimizer = (

    torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE
    )
)


# =====================================
# EĞİTİM
# =====================================

for epoch in range(

    1,

    EPOCHS + 1
):

    model.train()


    epoch_total_loss = 0.0

    epoch_reconstruction_loss = 0.0

    epoch_kl_loss = 0.0


    progress = tqdm(

        train_loader,

        desc=(

            f"Epoch "

            f"{epoch}/"

            f"{EPOCHS}"
        )
    )


    for images, _ in progress:


        # [batch, 1, 28, 28]
        #
        # →
        #
        # [batch, 784]

        images = (

            images.view(

                images.size(0),

                -1
            )
        )


        # Eski gradyanları temizle

        optimizer.zero_grad()


        # Modeli çalıştır

        (

            reconstruction,

            mu,

            log_var,

            _,

            _

        ) = model(

            images
        )


        # VAE loss

        losses = (

            vae_loss(

                reconstruction,

                images,

                mu,

                log_var,

                beta=BETA
            )
        )


        # Geri yayılım

        losses[

            "total_loss"

        ].backward()


        # Ağırlıkları güncelle

        optimizer.step()


        # Değerleri kaydet

        epoch_total_loss += (

            losses[

                "total_loss"

            ].item()
        )


        epoch_reconstruction_loss += (

            losses[

                "reconstruction_loss"

            ].item()
        )


        epoch_kl_loss += (

            losses[

                "kl_loss"

            ].item()
        )


        progress.set_postfix(

            total=(

                f"{losses['total_loss'].item():.5f}"
            ),

            reconstruction=(

                f"{losses['reconstruction_loss'].item():.5f}"
            ),

            kl=(

                f"{losses['kl_loss'].item():.5f}"
            )
        )


    number_of_batches = (

        len(

            train_loader
        )
    )


    print()

    print(

        f"Epoch: {epoch}"
    )


    print(

        "Average total loss:",

        epoch_total_loss

        /

        number_of_batches
    )


    print(

        "Average reconstruction loss:",

        epoch_reconstruction_loss

        /

        number_of_batches
    )


    print(

        "Average KL loss:",

        epoch_kl_loss

        /

        number_of_batches
    )


# =====================================
# MODELİ KAYDET
# =====================================

checkpoint_directory = (

    PROJECT_ROOT

    /

    "results"

    /

    "checkpoints"
)


checkpoint_directory.mkdir(

    parents=True,

    exist_ok=True
)


checkpoint_path = (

    checkpoint_directory

    /

    "hybrid_vae"

    /

    f"hybrid_vae_latent_"

    f"{LATENT_DIM}.pt"
)


checkpoint_path.parent.mkdir(

    parents=True,

    exist_ok=True
)


torch.save(

    {

        "model_state_dict":

            model.state_dict(),


        "latent_dim":

            LATENT_DIM,


        "hidden_dim":

            HIDDEN_DIM,


        "beta":

            BETA,


        "seed":

            SEED

    },

    checkpoint_path
)


print()

print(

    "Model saved:"
)

print(

    checkpoint_path
)