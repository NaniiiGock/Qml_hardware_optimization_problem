import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from generate_dataset import create_dataset
from qiskit.providers.fake_provider import FakeAthensV2

import qiskit
import qiskit_aer.noise as noise

def encode(image, wires=range(4)):
    qml.AmplitudeEmbedding(image, wires, pad_with=0, normalize=True)


BAS, labels = create_dataset(4)

backend = FakeAthensV2()

def block(weights, wires):
    qml.RY(weights[0], wires=wires[0])
    qml.RY(weights[1], wires=wires[1])
    qml.CNOT(wires=wires)

p = 0.01
my_bitflip = noise.pauli_error([('X', p), ('I', 1 - p)])
my_noise_model = noise.NoiseModel()
my_noise_model.add_quantum_error(my_bitflip, ["h", "CX"], [0])
dev = qml.device('qiskit.aer', wires=4, noise_model = my_noise_model)

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
false_count = 0
for indx, image in enumerate(BAS[14:]):
    fig, ax = qml.draw_mpl(circuit, expansion_strategy="device")(image, params)
    plt.figure(figsize=[1.8, 1.8])
    plt.imshow(np.reshape(image, [4, 4]), cmap="gray")
    if circuit(image,params)<0:
        if indx%2==0:
            false_count+=1
    else:
        if indx%2==1:
            false_count+=1
        
    plt.title(
        f"Exp. Val. = {circuit(image,params):.0f};"
        + f" Label = {'Bars' if circuit(image,params)<0 else 'Stripes'}",

        fontsize=8,
    )
    plt.xticks([])
    plt.yticks([])
    # plt.show()

print(1 - (false_count/14))