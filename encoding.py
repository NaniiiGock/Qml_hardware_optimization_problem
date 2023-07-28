import pennylane as qml
import numpy as np

# quantum circuit for amplitude encoding
dev = qml.device("default.qubit", wires=5)

# function that encodes image into a state
@qml.qnode(dev)
def encode(image, wires=range(5)):
    qml.AmplitudeEmbedding(image, wires, pad_with=0, normalize=True)
    return qml.state()

image = np.array([0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
print(encode(image))