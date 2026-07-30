import sys
from pathlib import Path

import torch


# Proje ana dizinini bul

PROJECT_ROOT = (

    Path(__file__)

    .resolve()

    .parents[1]
)


# Proje ana dizinini Python yoluna ekle

sys.path.insert(

    0,

    str(PROJECT_ROOT)
)


from models.quantum_layer import (

    SingleQubitLayer
)


# Kuantum katmanını oluştur

quantum_layer = (

    SingleQubitLayer()
)


# 4 örnekten oluşan test batch'i

angles = torch.tensor(

    [

        [0.0],

        [0.5],

        [1.0],

        [1.5]

    ],

    dtype=torch.float32
)


# Kuantum katmanını çalıştır

outputs = (

    quantum_layer(

        angles
    )
)


print(

    "Input shape:",

    angles.shape
)


print(

    "Output shape:",

    outputs.shape
)


print(

    "Quantum outputs:",

    outputs
)