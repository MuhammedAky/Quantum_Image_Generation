from qiskit_ibm_runtime import QiskitRuntimeService


def main():

    service = QiskitRuntimeService()

    backend = service.least_busy(
        operational=True,
        simulator=False,
        min_num_qubits=1,
    )

    print(
        "Seçilen gerçek kuantum cihazı:"
    )

    print(
        backend.name
    )

    print(
        "Qubit sayısı:"
    )

    print(
        backend.num_qubits
    )


if __name__ == "__main__":
    main()