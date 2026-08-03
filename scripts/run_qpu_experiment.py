import math
import time

import pennylane as qml

from qiskit_ibm_runtime import (
    QiskitRuntimeService
)


def main():

    print(
        "IBM Quantum servisine bağlanılıyor..."
    )

    service = QiskitRuntimeService()

    print(
        "Gerçek QPU seçiliyor..."
    )

    backend = service.least_busy(
        operational=True,
        simulator=False,
        min_num_qubits=1,
    )

    print(
        f"Seçilen QPU: "
        f"{backend.name}"
    )

    print(
        f"QPU toplam qubit sayısı: "
        f"{backend.num_qubits}"
    )

    print(
        "PennyLane QPU cihazı oluşturuluyor..."
    )

    qpu_device = qml.device(
        "qiskit.remote",
        wires=backend.num_qubits,
        backend=backend,
        shots=1024,
    )

    @qml.qnode(qpu_device)
    def qpu_circuit(angle):

        qml.RY(
            angle,
            wires=0,
        )

        return qml.expval(
            qml.PauliZ(0)
        )

    angle = 0

    print()

    print(
        f"Giriş açısı: {angle}"
    )

    print(
        "Kullanılan shot sayısı: 1024"
    )

    print(
        "Devre QPU'ya gönderiliyor..."
    )

    start_time = time.time()

    result = qpu_circuit(
        angle
    )

    end_time = time.time()

    elapsed_time = (
        end_time
        -
        start_time
    )

    print()

    print(
        "QPU işlemi tamamlandı."
    )

    print(
        f"QPU ölçüm sonucu: "
        f"{result}"
    )

    print(
        f"Geçen süre: "
        f"{elapsed_time:.2f} saniye"
    )

    ideal_result = math.cos(
        angle
    )

    difference = abs(
        float(result)
        -
        ideal_result
    )

    print(
        f"İdeal sonuç: "
        f"{ideal_result}"
    )

    print(
        f"Mutlak fark: "
        f"{difference}"
    )


if __name__ == "__main__":
    main()