import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from generate_dataset import create_dataset
from qiskit.providers.fake_provider import FakeAthensV2

dev = qml.device('default.qubit', wires=4)

def amplitude_encoding(image, wires):
    for idx, pixel in enumerate(image):
        idx = idx%4
        qml.RY(2 * np.arcsin(np.sqrt(pixel)), wires=wires[idx])

def encode(image, wires=range(4)):
    qml.AmplitudeEmbedding(image, wires, pad_with=0, normalize=True)

@qml.qnode(dev)
def quantum_state(image):
    encode(image,wires = [0,1,2,3])
    return qml.state()



BAS, labels = create_dataset(4)

states = []

for image in BAS:
    states.append(tuple(quantum_state(image)))

print(len(set(states)))

backend = FakeAthensV2()


def block(weights, wires):
    qml.RY(weights[0], wires=wires[0])
    qml.RY(weights[1], wires=wires[1])
    qml.CNOT(wires=wires)

dev = qml.device('qiskit.aer', wires=4)


@qml.qnode(dev, interface="autograd")
def circuit(image, template_weights):
    encode(image, wires=range(4))
    qml.TTN(
        wires=range(4),
        n_block_wires=2,
        block=block,
        n_params_block=2,
        template_weights=template_weights,
    )
    return qml.expval(qml.PauliZ(wires=3))


weights = np.random.random(size=[3, 2])

def costfunc(params):
    cost = 0
    for i in range(int(len(BAS)/2)):
        if ((i%2) == 0):
            cost += circuit(BAS[i], params)
        else:
            cost -= circuit(BAS[i], params)
    return cost

params = np.random.random(size=[3, 2], requires_grad=True)
optimizer = qml.GradientDescentOptimizer(stepsize=0.1)

for k in range(100):
    if k % 20 == 0:
        print(f"Step {k}, cost: {costfunc(params)}")
    params = optimizer.step(costfunc, params)

for image in BAS[14:]:
    fig, ax = qml.draw_mpl(circuit, expansion_strategy="device")(image, params)
    plt.figure(figsize=[1.8, 1.8])
    plt.imshow(np.reshape(image, [4, 4]), cmap="gray")
    plt.title(
        f"Exp. Val. = {circuit(image,params):.0f};"
        + f" Label = {'Bars' if circuit(image,params)<0 else 'Stripes'}",
        fontsize=8,
    )
    plt.xticks([])
    plt.yticks([])
    plt.show()