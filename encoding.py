import pennylane as qml
import numpy as np

# Define the quantum circuit for amplitude encoding
def amplitude_encoding(image, wires):
    n_qubits = len(wires)
    for i in range(n_qubits):
        if image[i] == 1:
            qml.PauliX(wires=wires[i])

dev = qml.device("default.qubit", wires=5)
# Define the quantum function that creates the quantum state
@qml.qnode(dev)
def quantum_state(image):
    amplitude_encoding(image, wires=[0, 1, 2, 3, 4])
    return qml.state()

image = np.array([0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
normalized_image = image / np.max(image)
quantum_state_vector = quantum_state(normalized_image)
print(quantum_state_vector)