import torch
import torch.nn as nn
import pennylane as qml

class SingleQubitLayer(nn.Module):


    def __init__(self):

        super().__init__()

        # Tek qubit kullanan ideal kuantum simülatörü
        self.device = qml.device(
            "default.qubit",
            wires=1
        )

        # Eğitim sırasında öğrenilecek kuantum parametresi
        self.theta = nn.Parameter(
            torch.tensor(
                0.1,
                dtype=torch.float32
            )
        )

        # PennyLane kuantum düğümü
        self.qnode = qml.QNode(
            self.quantum_circuit,
            self.device,
            interface="torch",
            diff_method="backprop"
        )

    def quantum_circuit(
        self,
        angle,
        theta
    ):

        # Klasik bilgiyi qubit'e açı olarak yükle
        qml.RY(
            angle,
            wires=0
        )

        # Eğitim sırasında öğrenilecek kuantum dönüşü
        qml.RY(
            theta,
            wires=0
        )

        # Faz dönüşü
        qml.RZ(
            theta,
            wires=0
        )

        # Pauli-Z beklenti değerini ölç
        return qml.expval(
            qml.PauliZ(0)
        )

    def forward(
        self,
        angles
    ):

        # Beklenen giriş şekli:
        # [batch_size, 1]

        angles = angles.squeeze(
            dim=-1
        )

        outputs = []

        # Batch içindeki her örneği
        # kuantum devresinden geçir
        for angle in angles:

            output = self.qnode(
                angle,
                self.theta
            )

            outputs.append(
                output
            )

        # Çıktıları tek bir tensörde birleştir
        outputs = torch.stack(
            outputs
        )

        # PennyLane bazı sürümlerde float64 döndürebilir.
        # PyTorch katmanlarımız float32 kullandığı için
        # veri tipini eşitliyoruz.
        outputs = outputs.to(
            dtype=torch.float32
        )

        # Çıkış şekli:
        # [batch_size, 1]

        return outputs.unsqueeze(
            dim=-1
        )